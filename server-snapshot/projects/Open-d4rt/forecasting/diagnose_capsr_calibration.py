from types import SimpleNamespace
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from capsr_calibrated_model import StructuredCaPSR
from train_capsr import frozen_prediction, load_frozen_models
from train_pointodyssey_dmvfn_injection import combined_splits
from train_simvp_ablation import move
from video_forecast_dataset import VideoForecastDataset, video_pad_collate


def main():
    device = torch.device("cuda")
    args = SimpleNamespace(
        cache_root="data/d4rt_forecasting_cache",
        po_cache="/autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po50",
        po_split_manifest="/autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey50_v2/po50/split_manifest.json",
        trajectory_checkpoint="/autodl-fs/data/d4rt_training/pointodyssey50_uv_velocity_residual_60/best.pt",
        backbone_checkpoint="outputs/simvp_dynamic_ablation/no_geometry/best.pt",
        legacy_subsets="adt_mini,ds_mini,po_mini,pstudio_mini",
        seed=42,
    )
    splits, _ = combined_splits(args)
    dataset = VideoForecastDataset(
        args.cache_root, "test", 64, paths=splits["test"], cache_items=False
    )
    loader = DataLoader(dataset, 1, collate_fn=video_pad_collate, num_workers=0)
    base, trajectory, _, _ = load_frozen_models(args, device)
    model = StructuredCaPSR().to(device)
    checkpoint = torch.load(
        "/autodl-fs/data/d4rt_training/capsr_calibrated_v5_20260827/best.pt",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    sums = {name: None for name in (
        "point", "point_lt_01", "point_lt_05", "point_gt_09",
        "camera_scale", "object_scale", "object_support",
    )}
    denoms = {name: None for name in sums}
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            base_prediction, xyz, uv = frozen_prediction(base, trajectory, batch)
            _, state = model(
                base_prediction, batch["context_rgb"][:, -1], xyz, uv,
                batch["past_xyz"].float(), batch["past_uv"].float(),
                batch["past_valid"], batch["past_confidence"].float(),
                mode="structured",
            )
            valid = batch["past_valid"].any(2)[:, None].expand_as(state["point_selection"]).float()
            camera_support = state["camera_valid"]
            object_support = state["object_support"]
            values = {
                "point": (state["point_selection"] * valid, valid),
                "point_lt_01": ((state["point_selection"] < .1).float() * valid, valid),
                "point_lt_05": ((state["point_selection"] < .5).float() * valid, valid),
                "point_gt_09": ((state["point_selection"] > .9).float() * valid, valid),
                "camera_scale": (state["camera_permission_scale"] * camera_support, camera_support),
                "object_scale": (state["object_permission_scale"] * object_support, object_support),
                "object_support": (object_support, torch.ones_like(object_support)),
            }
            for name, (numerator, denominator) in values.items():
                numerator = numerator.sum(tuple(i for i in range(numerator.ndim) if i != 1)).cpu()
                denominator = denominator.sum(tuple(i for i in range(denominator.ndim) if i != 1)).cpu()
                sums[name] = numerator if sums[name] is None else sums[name] + numerator
                denoms[name] = denominator if denoms[name] is None else denoms[name] + denominator
    result = {name: (sums[name] / denoms[name].clamp_min(1)).tolist() for name in sums}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
