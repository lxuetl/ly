"""Zero-training diagnosis of spatial/temporal geometry benefit quadrants."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from dmvfn_adapter import DMVFNVideoPredictor
from geometry_residual_model import GeometryFeatureProposalCorrector
from train_all_datasets import make_splits
from train_geometry_residual import latest_uv
from train_pointodyssey_dmvfn_injection import geometry_for_source
from trajectory_model import ImplicitConditionedJointForecaster, VelocityResidualImplicitForecaster
from video_forecast_dataset import VideoForecastDataset, video_pad_collate


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root', default='data/d4rt_forecasting_cache')
    p.add_argument('--legacy-subsets', default='adt_mini,ds_mini,po_mini,pstudio_mini')
    p.add_argument('--po-cache', required=True)
    p.add_argument('--po-split-manifest', required=True)
    p.add_argument('--trajectory-checkpoint', required=True)
    p.add_argument('--backbone-checkpoint', required=True)
    p.add_argument('--corrector-checkpoint', required=True)
    p.add_argument('--reference-summary', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--spatial-margin', type=float, default=0.002)
    p.add_argument('--temporal-margin', type=float, default=0.0)
    p.add_argument('--robust-temporal-margin', type=float, default=0.0005)
    p.add_argument('--min-motion', type=float, default=0.025)
    p.add_argument('--min-reliability', type=float, default=0.30)
    p.add_argument('--mask-power', type=float, default=1.5)
    p.add_argument('--dilation-kernel', type=int, default=1)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def test_paths(a):
    subsets = [x.strip() for x in a.legacy_subsets.split(',') if x.strip()]
    splits, _ = make_splits(a.cache_root, subsets, 42)
    spec = json.loads(Path(a.po_split_manifest).read_text())
    po_root = Path(a.po_cache) / 'po50'
    for name in spec['clips']['test']:
        path = po_root / name
        with np.load(path) as z:
            if np.asarray(z['future_valid']).any():
                splits['test'].append(path)
    return splits['test']


def domain_name(path):
    text = str(path).replace('\\', '/').lower()
    if '/po50/' in text:
        return 'pointodyssey50'
    for name in ('adt_mini', 'ds_mini', 'po_mini', 'pstudio_mini'):
        if name in text:
            return name
    return Path(path).parent.name or 'unknown'


class Stats:
    def __init__(self):
        self.data = defaultdict(lambda: defaultdict(float))

    def add(self, group, mask, spatial, temporal, robust_margin):
        mask = mask.bool()
        n = int(mask.sum())
        if not n:
            return
        s_pos = spatial > 0
        t_pos = temporal > 0
        values = self.data[group]
        values['count'] += n
        values['A_spatial_pos_temporal_pos'] += int((mask & s_pos & t_pos).sum())
        values['B_spatial_pos_temporal_neg'] += int((mask & s_pos & ~t_pos).sum())
        values['C_spatial_neg_temporal_pos'] += int((mask & ~s_pos & t_pos).sum())
        values['D_spatial_neg_temporal_neg'] += int((mask & ~s_pos & ~t_pos).sum())
        values['spatial_advantage_sum'] += float(spatial[mask].sum())
        values['temporal_advantage_sum'] += float(temporal[mask].sum())
        values['temporal_robust_positive'] += int((mask & (temporal > robust_margin)).sum())
        values['temporal_robust_negative'] += int((mask & (temporal < -robust_margin)).sum())
        values['temporal_neutral'] += int((mask & (temporal.abs() <= robust_margin)).sum())

    def finish(self):
        result = {}
        quadrant_keys = ('A_spatial_pos_temporal_pos', 'B_spatial_pos_temporal_neg',
                         'C_spatial_neg_temporal_pos', 'D_spatial_neg_temporal_neg')
        for group, raw in sorted(self.data.items()):
            n = max(raw['count'], 1)
            item = {'count': int(raw['count'])}
            for key in quadrant_keys:
                item[key] = {'count': int(raw[key]), 'fraction': raw[key] / n}
            item['spatial_positive_fraction'] = (raw[quadrant_keys[0]] + raw[quadrant_keys[1]]) / n
            item['temporal_positive_fraction'] = (raw[quadrant_keys[0]] + raw[quadrant_keys[2]]) / n
            item['mean_spatial_advantage'] = raw['spatial_advantage_sum'] / n
            item['mean_temporal_advantage'] = raw['temporal_advantage_sum'] / n
            for key in ('temporal_robust_positive', 'temporal_robust_negative', 'temporal_neutral'):
                item[key + '_fraction'] = raw[key] / n
            result[group] = item
        return result


def aligned_support_status(mask, flow):
    b, k, _, h, w = mask.shape
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=mask.device, dtype=mask.dtype),
                            torch.linspace(-1, 1, w, device=mask.device, dtype=mask.dtype), indexing='ij')
    base_grid = torch.stack((xx, yy), -1)[None].expand(b, -1, -1, -1)
    persistent = torch.zeros_like(mask, dtype=torch.bool)
    new_support = torch.zeros_like(mask, dtype=torch.bool)
    for step in range(1, k):
        relative = (flow[:, step] - flow[:, step - 1]).permute(0, 2, 3, 1)
        previous = F.grid_sample(mask[:, step - 1], base_grid + 2 * relative,
                                 mode='bilinear', padding_mode='zeros', align_corners=True)
        current = mask[:, step] > .01
        persistent[:, step] = current & (previous > .01)
        new_support[:, step] = current & (previous <= .01)
    return persistent, new_support


def main():
    a = parse_args()
    if not torch.cuda.is_available() and a.device.startswith('cuda'):
        raise RuntimeError('CUDA is required for full-test diagnosis; restore the GPU instance first.')
    device = torch.device(a.device)
    paths = test_paths(a)
    dataset = VideoForecastDataset(a.cache_root, 'test', 64, paths=paths, cache_items=False)
    loader = DataLoader(dataset, 1, collate_fn=video_pad_collate, num_workers=0)

    bck = torch.load(a.backbone_checkpoint, map_location='cpu')
    c, k = dataset[0]['context_rgb'].shape[0], dataset[0]['future_rgb'].shape[0]
    base = DMVFNVideoPredictor(c, k)
    base.load_state_dict(bck['model'])
    base.to(device).eval()

    jck = torch.load(a.trajectory_checkpoint, map_location='cpu')
    joint_cls = VelocityResidualImplicitForecaster if jck.get('model_variant') == 'velocity_residual' else ImplicitConditionedJointForecaster
    joint = joint_cls(jck['future_frames'], jck['hidden_dim'])
    joint.load_state_dict(jck['model'])
    joint.to(device).eval()

    meta = json.loads(Path(a.reference_summary).read_text())['meta']
    corrector = GeometryFeatureProposalCorrector(
        feature_channels=meta['feature_channels'],
        proposal_residual_scale=meta['proposal_residual_scale'],
        benefit_gate_bias=meta['benefit_gate_bias'],
        benefit_topk_fraction=meta['benefit_topk_fraction'],
        benefit_temporal_align_weight=meta['benefit_temporal_align_weight'],
        min_motion=a.min_motion,
        min_reliability=a.min_reliability,
        mask_power=a.mask_power,
        dilation_kernel=a.dilation_kernel,
        base_motion_threshold=meta['base_motion_threshold'],
        base_motion_ramp=meta['base_motion_ramp'],
        flow_coherence_scale=meta.get('flow_coherence_scale', 0.0),
        first_step_scale=meta['first_step_scale'],
        horizon_ramp_steps=meta['horizon_ramp_steps'])
    corrector.load_state_dict(torch.load(a.corrector_checkpoint, map_location='cpu')['model'])
    corrector.to(device).eval()

    stats = Stats()
    motion_edges = (0.01, 0.03, 0.06)
    with torch.no_grad():
        for index, batch in enumerate(loader):
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            base_pred = base(batch['context_rgb'])
            geometry, weight, future_uv = geometry_for_source('predicted', joint, batch, k, 'uv_only', True, True)
            pred, _, _ = corrector(base_pred, batch['context_rgb'][:, -1], geometry, future_uv,
                                    weight, latest_uv(batch))
            proposal = corrector.last_proposal
            support = corrector.last_support > .01
            selected = (corrector.last_gate_selection > .5) & support
            flow = corrector.last_flow
            target = batch['future_rgb']
            last = batch['context_rgb'][:, -1:]

            base_spatial = (base_pred - target).abs().mean(2, keepdim=True)
            proposal_spatial = (proposal - target).abs().mean(2, keepdim=True)
            applied_spatial = (pred - target).abs().mean(2, keepdim=True)
            spatial_proposal = base_spatial - proposal_spatial - a.spatial_margin
            spatial_applied = base_spatial - applied_spatial - a.spatial_margin

            target_prev = torch.cat((last, target[:, :-1]), 1)
            base_prev = torch.cat((last, base_pred[:, :-1]), 1)
            proposal_prev = torch.cat((last, proposal[:, :-1]), 1)
            applied_prev = torch.cat((last, pred[:, :-1]), 1)
            target_change = target - target_prev
            base_temporal = ((base_pred - base_prev) - target_change).abs().mean(2, keepdim=True)
            proposal_temporal = ((proposal - proposal_prev) - target_change).abs().mean(2, keepdim=True)
            applied_temporal = ((pred - applied_prev) - target_change).abs().mean(2, keepdim=True)
            temporal_proposal = base_temporal - proposal_temporal - a.temporal_margin
            temporal_applied = base_temporal - applied_temporal - a.temporal_margin

            persistent, new_support = aligned_support_status(corrector.last_support, flow)
            domain = domain_name(paths[index])
            magnitude = torch.linalg.vector_norm(flow, dim=2, keepdim=True)
            motion_masks = {
                'motion_0_0.01': magnitude < motion_edges[0],
                'motion_0.01_0.03': (magnitude >= motion_edges[0]) & (magnitude < motion_edges[1]),
                'motion_0.03_0.06': (magnitude >= motion_edges[1]) & (magnitude < motion_edges[2]),
                'motion_0.06_plus': magnitude >= motion_edges[2]}

            for scope, mask in (('support', support), ('selected', selected)):
                stats.add(f'{scope}/all', mask, spatial_proposal, temporal_proposal, a.robust_temporal_margin)
                stats.add(f'{scope}/domain/{domain}', mask, spatial_proposal, temporal_proposal, a.robust_temporal_margin)
                for step in range(k):
                    stats.add(f'{scope}/horizon/t+{step+1}', mask[:, step:step+1],
                              spatial_proposal[:, step:step+1], temporal_proposal[:, step:step+1], a.robust_temporal_margin)
                for name, motion_mask in motion_masks.items():
                    stats.add(f'{scope}/motion/{name}', mask & motion_mask,
                              spatial_proposal, temporal_proposal, a.robust_temporal_margin)
                stats.add(f'{scope}/visibility/persistent', mask & persistent,
                          spatial_proposal, temporal_proposal, a.robust_temporal_margin)
                stats.add(f'{scope}/visibility/new_support', mask & new_support,
                          spatial_proposal, temporal_proposal, a.robust_temporal_margin)
            stats.add('applied_selected/all', selected, spatial_applied, temporal_applied, a.robust_temporal_margin)
            stats.add(f'applied_selected/domain/{domain}', selected, spatial_applied, temporal_applied, a.robust_temporal_margin)
            if (index + 1) % 20 == 0:
                print(f'processed {index+1}/{len(loader)}', flush=True)

    result = {
        'meta': {'num_sequences': len(loader), 'spatial_margin': a.spatial_margin,
                 'temporal_margin': a.temporal_margin,
                 'robust_temporal_margin': a.robust_temporal_margin,
                 'motion_magnitude_units': 'normalized UV displacement',
                 'primary_definition': 'proposal advantage over frozen backbone; applied_selected uses final corrected prediction'},
        'groups': stats.finish()}
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
