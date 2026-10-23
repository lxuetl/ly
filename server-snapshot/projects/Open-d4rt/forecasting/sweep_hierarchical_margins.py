"""Validation-only specialist-margin sweep for a trained hierarchical router."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from capsr_calibrated_model import StructuredCaPSR
from train_capsr import load_frozen_models
from train_pointodyssey_dmvfn_injection import combined_splits
from train_simvp_ablation import move, seed_all
from video_forecast_dataset import VideoForecastDataset, video_pad_collate
from train_unified_action_risk import (
    ACTION_NAMES, HierarchicalActionRiskCalibrator, Metrics,
    build_candidates, evaluate,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--trajectory-checkpoint", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--correction-checkpoint", required=True)
    p.add_argument("--router-checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-train", type=int, default=192)
    p.add_argument("--max-val", type=int, default=48)
    p.add_argument("--max-test", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--margins", default="0.05,0.075,0.10,0.15,0.20,0.30")
    return p.parse_args()


@torch.no_grad()
def sweep_validation(
    selector, margins, base, trajectory, camera_model, raft, raft_transform,
    loader, device,
):
    selector.eval()
    accumulators = {
        margin: {
            "metrics": Metrics(),
            "choice": torch.zeros(len(ACTION_NAMES), device=device),
            "count": 0.0,
        }
        for margin in margins
    }
    for batch in loader:
        batch = move(batch, device)
        baseline, actions, replay, cycle, disagreement, magnitude = build_candidates(
            base, trajectory, camera_model, raft, raft_transform, batch,
        )
        target = batch["future_rgb"]
        last = batch["context_rgb"][:, -1]
        for margin in margins:
            selector.specialist_margin = margin
            state = selector(baseline, actions, replay, cycle, disagreement, magnitude)
            item = accumulators[margin]
            item["metrics"].update(state["prediction"], target, last)
            item["choice"] += state["hard_choice"].sum((0, 1, 3, 4, 5))
            item["count"] += state["hard_choice"].sum().item()

    result = {}
    for margin, item in accumulators.items():
        result[str(margin)] = {
            "selected": item["metrics"].result(),
            "action_choice_fraction": {
                name: float(item["choice"][index] / max(item["count"], 1))
                for index, name in enumerate(ACTION_NAMES)
            },
        }
    return result


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    margins = [float(value) for value in args.margins.split(",")]
    splits, split_meta = combined_splits(args)

    def loader(split, maximum):
        paths = list(splits[split])[:maximum] if maximum > 0 else list(splits[split])
        dataset = VideoForecastDataset(args.cache_root, split, 64, paths=paths, cache_items=False)
        return DataLoader(
            dataset, args.batch_size, collate_fn=video_pad_collate, num_workers=0,
        )

    val_loader = loader("val", args.max_val)
    test_loader = loader("test", args.max_test)
    base, trajectory, _, _ = load_frozen_models(args, device)
    correction = torch.load(args.correction_checkpoint, map_location=device)
    correction_config = correction.get("config", {})
    camera_model = StructuredCaPSR(
        correction_config.get("sigma", 0.035), correction_config.get("hidden", 48),
        num_motion_clusters=correction_config.get("num_motion_clusters", 4),
    ).to(device)
    camera_model.load_state_dict(correction["model"])

    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
    raft_weights = Raft_Small_Weights.DEFAULT
    raft = raft_small(weights=raft_weights, progress=True).to(device)
    raft_transform = raft_weights.transforms()
    for frozen in (base, trajectory, camera_model, raft):
        frozen.eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)

    checkpoint = torch.load(args.router_checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    selector = HierarchicalActionRiskCalibrator(
        temperature=config.get("selection_temperature", 0.20),
        abstain_margin=config.get("abstain_margin", 0.0),
        specialist_margin=config.get("specialist_margin", 0.05),
    ).to(device)
    selector.load_state_dict(checkpoint["model"])

    validation = sweep_validation(
        selector, margins, base, trajectory, camera_model, raft, raft_transform,
        val_loader, device,
    )
    best_margin = min(margins, key=lambda value: validation[str(value)]["selected"]["MSE"])
    selector.specialist_margin = best_margin
    test = evaluate(
        selector, base, trajectory, camera_model, raft, raft_transform, test_loader, device,
    )
    result = {
        "validation": validation,
        "best_margin": best_margin,
        "test": test,
        "meta": {"split": split_meta, "router_epoch": checkpoint.get("epoch")},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
