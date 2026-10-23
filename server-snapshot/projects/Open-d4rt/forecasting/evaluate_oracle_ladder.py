"""Evaluate a frozen video predictor/corrector under an Oracle geometry ladder.

No model parameters are updated.  Every mode uses the same backbone, correction
operator, test split, and metric implementation; only the geometry evidence is
changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from geometry_residual_model import GeometryFeatureProposalCorrector
from simvp_geometry_model import build_video_predictor_from_checkpoint
from train_geometry_residual import evaluate, latest_uv
from train_pointodyssey_dmvfn_injection import combined_splits, geometry_for_source
from train_simvp_ablation import move, seed_all
from trajectory_model import (
    GeometryConditionEncoder,
    ImplicitConditionedJointForecaster,
    VelocityResidualImplicitForecaster,
)
from video_forecast_dataset import VideoForecastDataset, video_pad_collate


MODES = (
    "no_geometry",
    "history_uv",
    "predicted_uv",
    "oracle_uv",
    "oracle_uv_visibility",
    "oracle_xyz_uv_visibility",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--trajectory-checkpoint", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--corrector-checkpoint", required=True)
    p.add_argument("--reference-summary", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def make_corrector(meta, checkpoint, device):
    model = GeometryFeatureProposalCorrector(
        feature_channels=int(meta.get("feature_channels", 16)),
        proposal_residual_scale=float(meta.get("proposal_residual_scale", 0.08)),
        benefit_gate_bias=float(meta.get("benefit_gate_bias", -0.5)),
        benefit_topk_fraction=float(meta.get("benefit_topk_fraction", 0.0)),
        benefit_temporal_align_weight=float(meta.get("benefit_temporal_align_weight", 0.0)),
        benefit_temporal_ema_alpha=float(meta.get("benefit_temporal_ema_alpha", 1.0)),
        benefit_rank_margin=float(meta.get("benefit_rank_margin", 0.0)),
        dual_benefit_head=bool(meta.get("dual_benefit_head", False)),
        benefit_min_joint_probability=float(meta.get("benefit_min_joint_probability", 0.0)),
        temporal_risk_max_probability=float(meta.get("temporal_risk_max_probability", 1.0)),
        sigma=0.03,
        dilation_kernel=int(meta.get("dilation_kernel", 1)),
        min_motion=float(meta.get("min_motion", 0.025)),
        min_reliability=float(meta.get("min_reliability", 0.30)),
        mask_power=float(meta.get("mask_power", 1.5)),
        base_motion_threshold=float(meta.get("base_motion_threshold", 0.005)),
        base_motion_ramp=float(meta.get("base_motion_ramp", 0.04)),
        flow_coherence_scale=float(meta.get("flow_coherence_scale", 0.0)),
        first_step_scale=float(meta.get("first_step_scale", 0.5)),
        horizon_ramp_steps=int(meta.get("horizon_ramp_steps", 4)),
    ).to(device)
    state = torch.load(checkpoint, map_location=device)["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def predicted_state(joint, batch):
    valid = batch["past_valid"].bool()
    with torch.no_grad():
        xyz, uv = joint(
            batch["past_xyz"].float(),
            batch["past_uv"].float(),
            valid,
            batch["past_confidence"].float(),
        )
    return xyz, uv


def oracle_geometry(mode, joint, batch, k):
    """Isolate coordinate accuracy from future visibility information."""
    valid = batch["past_valid"].bool()
    point_ok = valid.any(2)
    anchor = latest_uv(batch)
    teacher_uv = torch.nan_to_num(batch["future_uv_teacher"].float()).clamp(0, 1)
    teacher_xyz = torch.nan_to_num(batch["future_xyz_teacher"].float())
    past_conf = torch.nan_to_num(batch["past_confidence"].float()).amax(2)

    if mode == "oracle_uv":
        xyz = torch.zeros_like(teacher_xyz)
        uv = teacher_uv
        confidence = past_conf[:, :, None].expand(-1, -1, k).clamp(0, 1)
        availability = point_ok[:, :, None].float().expand(-1, -1, k)
    elif mode in ("oracle_uv_visibility", "oracle_xyz_uv_visibility"):
        xyz = teacher_xyz if mode == "oracle_xyz_uv_visibility" else torch.zeros_like(teacher_xyz)
        uv = teacher_uv
        confidence = torch.nan_to_num(batch["future_confidence"].float()).clamp(0, 1)
        finite = torch.isfinite(batch["future_uv_teacher"]).all(-1)
        availability = (batch["future_valid"].bool() & finite).float()
    else:
        raise ValueError(mode)

    displacement = torch.linalg.vector_norm(uv - anchor[:, :, None], dim=-1)
    dynamic = ((displacement - 0.015) / 0.035).clamp(0, 1)
    reliability = availability * confidence * dynamic
    raw = GeometryConditionEncoder.raw_features(xyz, uv, confidence)
    if mode != "oracle_xyz_uv_visibility":
        raw = raw.clone()
        raw[..., :3] = 0
    return raw, reliability, uv


def run_mode(base, corrector, joint, batch, k, mode):
    with torch.no_grad():
        base_prediction = base(batch["context_rgb"])
        if mode == "no_geometry":
            zeros = torch.zeros_like(base_prediction)
            mask = base_prediction.new_zeros(
                base_prediction.shape[0], k, 1, base_prediction.shape[-2], base_prediction.shape[-1]
            )
            return base_prediction, zeros, mask
        if mode == "history_uv":
            geometry, reliability, uv = geometry_for_source(
                "history", joint, batch, k, "uv_only", True, True
            )
        elif mode == "predicted_uv":
            geometry, reliability, uv = geometry_for_source(
                "predicted", joint, batch, k, "uv_only", True, True
            )
        else:
            geometry, reliability, uv = oracle_geometry(mode, joint, batch, k)
        return corrector(
            base_prediction,
            batch["context_rgb"][:, -1],
            geometry,
            uv,
            reliability,
            latest_uv(batch),
        )


def main():
    a = parse_args()
    seed_all(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    splits, split_meta = combined_splits(a)
    test_ds = VideoForecastDataset(
        a.cache_root, "test", 64, paths=splits["test"], cache_items=False
    )
    if a.max_samples > 0:
        test_ds = Subset(test_ds, range(min(a.max_samples, len(test_ds))))
    test = DataLoader(
        test_ds, batch_size=1, collate_fn=video_pad_collate, num_workers=0
    )

    backbone_ckpt = torch.load(a.backbone_checkpoint, map_location="cpu")
    base = build_video_predictor_from_checkpoint(backbone_ckpt)
    incompatible = base.load_state_dict(backbone_ckpt["model"], strict=False)
    allowed_missing = {
        "latent_gate_logit", "skip_gate_logit", "latent_film.weight", "latent_film.bias",
        "skip_film.weight", "skip_film.bias",
    }
    bad_missing = set(incompatible.missing_keys) - allowed_missing
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"incompatible backbone: missing={sorted(bad_missing)}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    base.to(device).eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)

    trajectory_ckpt = torch.load(a.trajectory_checkpoint, map_location="cpu")
    trajectory_class = (
        VelocityResidualImplicitForecaster
        if trajectory_ckpt.get("model_variant") == "velocity_residual"
        else ImplicitConditionedJointForecaster
    )
    joint = trajectory_class(
        trajectory_ckpt["future_frames"], trajectory_ckpt["hidden_dim"]
    )
    joint.load_state_dict(trajectory_ckpt["model"])
    joint.to(device).eval()
    for parameter in joint.parameters():
        parameter.requires_grad_(False)

    reference = json.loads(Path(a.reference_summary).read_text())
    meta = reference["meta"]
    corrector = make_corrector(meta, a.corrector_checkpoint, device)
    k = int(trajectory_ckpt["future_frames"])

    import train_geometry_residual as evaluator

    original_forward = evaluator.forward
    evaluator.forward = lambda b, c, j, batch, steps: run_mode(
        b, c, j, batch, steps, a.mode
    )
    try:
        metrics = evaluate(base, corrector, joint, test, device, k, output)
    finally:
        evaluator.forward = original_forward

    result = {
        "mode": a.mode,
        "num_test_clips": len(test_ds),
        "metrics": metrics,
        "protocol": {
            "training": False,
            "backbone": "simvp",
            "backbone_checkpoint": str(Path(a.backbone_checkpoint)),
            "corrector_checkpoint": str(Path(a.corrector_checkpoint)),
            "trajectory_checkpoint": str(Path(a.trajectory_checkpoint)),
            "split": split_meta,
            "oracle_uv_without_visibility": mode_definition(a.mode),
        },
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def mode_definition(mode):
    if mode == "oracle_uv":
        return "teacher UV with history-derived availability/confidence"
    if mode == "oracle_uv_visibility":
        return "teacher UV with teacher future_valid/future_confidence"
    if mode == "oracle_xyz_uv_visibility":
        return "teacher XYZ+UV with teacher future_valid/future_confidence"
    return "not applicable"


if __name__ == "__main__":
    main()
