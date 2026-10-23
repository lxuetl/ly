#!/usr/bin/env python3
"""Content-conditioned semantic alignment of VGGT-Omega tokens with TAU latents."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import train_simvp_feature_warp as core
import train_vggt_omega_latent_v2 as v2


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d4rt-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--sample-manifest", default="")
    p.add_argument("--omega-cache", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--alignment-epochs", type=int, default=4)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--alignment-lr", type=float, default=3e-4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--semantic-dim", type=int, default=256)
    p.add_argument("--feature-scale", type=float, default=0.12)
    p.add_argument("--temperature", type=float, default=0.08)
    p.add_argument("--alignment-weight", type=float, default=0.02)
    p.add_argument("--dynamic-weight", type=float, default=2.0)
    p.add_argument("--lpips-weight", type=float, default=0.03)
    p.add_argument("--temporal-weight", type=float, default=0.10)
    p.add_argument("--static-weight", type=float, default=0.10)
    p.add_argument("--correction-weight", type=float, default=0.01)
    p.add_argument("--rank-weight", type=float, default=0.50)
    p.add_argument("--wrong-reject-weight", type=float, default=0.05)
    p.add_argument("--rank-margin", type=float, default=2e-4)
    p.add_argument("--dynamic-threshold", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


class SemanticOmegaAdapterV3(nn.Module):
    """Use TAU content as queries over unpooled Omega memory slots."""

    def __init__(self, channels, future_frames, semantic_dim=256, feature_scale=0.12):
        super().__init__()
        self.channels = int(channels)
        self.future_frames = int(future_frames)
        self.semantic_dim = int(semantic_dim)
        self.feature_scale = float(feature_scale)

        self.tau_projection = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, semantic_dim), nn.GELU(),
            nn.Linear(semantic_dim, semantic_dim),
        )
        self.omega_projection = nn.Sequential(
            nn.LayerNorm(2048), nn.Linear(2048, semantic_dim), nn.GELU(),
            nn.Linear(semantic_dim, semantic_dim),
        )
        self.pose_projection = nn.Sequential(
            nn.LayerNorm(18), nn.Linear(18, semantic_dim), nn.GELU(),
            nn.Linear(semantic_dim, semantic_dim),
        )
        self.horizon_projection = nn.Sequential(
            nn.Linear(6, semantic_dim), nn.GELU(), nn.Linear(semantic_dim, semantic_dim)
        )
        self.context_time = nn.Parameter(torch.zeros(1, 32, 1, semantic_dim))
        self.slot_type = nn.Parameter(torch.zeros(1, 1, 17, semantic_dim))
        self.camera_attention = nn.MultiheadAttention(
            semantic_dim, 4, dropout=0.05, batch_first=True
        )
        self.register_attention = nn.MultiheadAttention(
            semantic_dim, 4, dropout=0.05, batch_first=True
        )
        self.semantic_norm = nn.LayerNorm(semantic_dim)
        self.match_head = nn.Sequential(
            nn.Linear(semantic_dim * 4, semantic_dim), nn.GELU(),
            nn.Linear(semantic_dim, 1),
        )
        self.film = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim), nn.GELU(),
            nn.Linear(semantic_dim, channels * 2),
        )
        self.channel_gate = nn.Sequential(nn.Linear(semantic_dim, channels), nn.Sigmoid())
        depth_hidden = max(16, channels // 2)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(5, depth_hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(depth_hidden, channels, 3, padding=1),
        )
        self.depth_horizon = nn.Sequential(nn.Linear(semantic_dim, channels), nn.Tanh())
        self.global_gate_logit = nn.Parameter(torch.tensor(-1.5))

        nn.init.normal_(self.context_time, std=0.01)
        nn.init.normal_(self.slot_type, std=0.01)
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        nn.init.zeros_(self.depth_encoder[-1].weight)
        nn.init.zeros_(self.depth_encoder[-1].bias)
        self.last = {}

    def _queries(self, hidden):
        batch, future_steps = hidden.shape[:2]
        content = hidden.mean((-2, -1))
        query = self.tau_projection(content)
        horizon = torch.linspace(
            0, 1, future_steps, device=hidden.device, dtype=hidden.dtype
        )
        basis = torch.stack((
            horizon, horizon.square(), horizon.pow(3),
            torch.sin(math.pi * horizon), torch.cos(math.pi * horizon),
            torch.ones_like(horizon),
        ), -1)
        return query + self.horizon_projection(basis)[None].expand(batch, -1, -1)

    def _memory(self, tokens, pose, use_tokens=True, use_pose=True):
        memory = self.omega_projection(tokens.to(dtype=self.context_time.dtype))
        if not use_tokens:
            memory = torch.zeros_like(memory)
        time = self.context_time[:, :memory.shape[1]].to(memory.dtype)
        slots = self.slot_type[:, :, :memory.shape[2]].to(memory.dtype)
        memory = memory + time + slots
        pose_feature = self.pose_projection(v2.OmegaLatentAdapterV2.pose_motion(
            pose.to(memory.dtype)
        ))
        if not use_pose:
            pose_feature = torch.zeros_like(pose_feature)
        memory = memory.clone()
        memory[:, :, 0] = memory[:, :, 0] + pose_feature
        return memory

    def forward(self, hidden, tokens, depth, confidence, pose,
                use_tokens=True, use_pose=True, use_depth=True):
        batch, future_steps, channels, height, width = hidden.shape
        if channels != self.channels or future_steps != self.future_frames:
            raise ValueError(f"unexpected hidden shape {tuple(hidden.shape)}")
        query = self._queries(hidden)
        memory = self._memory(tokens, pose, use_tokens, use_pose).to(hidden.dtype)
        camera_memory = memory[:, :, 0]
        register_memory = memory[:, :, 1:].reshape(batch, -1, self.semantic_dim)
        camera_context, camera_attention = self.camera_attention(
            query, camera_memory, camera_memory, need_weights=True
        )
        register_context, register_attention = self.register_attention(
            query, register_memory, register_memory, need_weights=True
        )
        aligned = self.semantic_norm(query + camera_context + register_context)

        match_features = torch.cat((
            query, aligned, (query - aligned).abs(), query * aligned
        ), -1)
        match_logits = self.match_head(match_features).squeeze(-1)
        tau_descriptor = F.normalize(query.mean(1), dim=-1)
        omega_descriptor = F.normalize(memory.mean((1, 2)), dim=-1)

        gamma, beta = self.film(aligned).chunk(2, -1)
        film_delta = hidden * torch.tanh(gamma[..., None, None]) + beta[..., None, None]
        if use_depth:
            statistics, confidence_map = v2.OmegaLatentAdapterV2.depth_statistics(
                depth, confidence
            )
            statistics = F.interpolate(
                statistics, (height, width), mode="bilinear", align_corners=False
            )
            confidence_map = F.interpolate(
                confidence_map, (height, width), mode="bilinear", align_corners=False
            )
            depth_feature = self.depth_encoder(statistics)[:, None]
            depth_delta = depth_feature * self.depth_horizon(aligned)[..., None, None]
        else:
            confidence_map = hidden.new_ones((batch, 1, height, width))
            depth_delta = torch.zeros_like(hidden)

        match_gate = torch.sigmoid(match_logits)[..., None, None, None]
        channel_gate = self.channel_gate(aligned)[..., None, None]
        spatial_gate = 0.25 + 0.75 * confidence_map[:, None]
        gate = torch.sigmoid(self.global_gate_logit) * match_gate * channel_gate * spatial_gate
        correction = self.feature_scale * torch.tanh(film_delta + depth_delta) * gate
        self.last = {
            "correction": correction,
            "gate": gate.detach(),
            "match_logits": match_logits,
            "match_probability": torch.sigmoid(match_logits).detach(),
            "tau_descriptor": tau_descriptor,
            "omega_descriptor": omega_descriptor,
            "camera_attention": camera_attention.detach(),
            "register_attention": register_attention.detach(),
        }
        return hidden + correction


def alignment_loss(correct_state, shuffled_state, temperature):
    correct_logits = correct_state["match_logits"]
    shuffled_logits = shuffled_state["match_logits"]
    binary = F.binary_cross_entropy_with_logits(
        correct_logits, torch.ones_like(correct_logits)
    ) + F.binary_cross_entropy_with_logits(
        shuffled_logits, torch.zeros_like(shuffled_logits)
    )
    tau = correct_state["tau_descriptor"]
    omega = correct_state["omega_descriptor"]
    if tau.shape[0] > 1:
        similarity = tau @ omega.t() / temperature
        labels = torch.arange(tau.shape[0], device=tau.device)
        contrastive = 0.5 * (
            F.cross_entropy(similarity, labels) +
            F.cross_entropy(similarity.t(), labels)
        )
    else:
        contrastive = binary.new_zeros(())
    margin = F.relu(
        0.5 - torch.sigmoid(correct_logits) + torch.sigmoid(shuffled_logits)
    ).mean()
    return binary + contrastive + margin, {
        "alignment_binary": binary,
        "alignment_contrastive": contrastive,
        "alignment_margin": margin,
        "correct_match_probability": torch.sigmoid(correct_logits).mean(),
        "shuffled_match_probability": torch.sigmoid(shuffled_logits).mean(),
    }


def run_alignment_stage(model, loader, backbone, device, a, output):
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.alignment_lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, a.alignment_epochs + 1):
        model.train()
        totals = defaultdict(float)
        batches = 0
        started = time.time()
        for raw in loader:
            batch = v2.move(raw, device)
            with torch.no_grad():
                _, hidden, _ = core.extract_frozen_features(backbone, batch["context_rgb"])
            model(hidden, *v2.omega_inputs(batch))
            correct_state = {key: value for key, value in model.last.items()
                             if not key.endswith("attention") and key not in ("gate", "correction")}
            model(hidden, *v2.omega_inputs(batch, shuffled=True))
            shuffled_state = {key: value for key, value in model.last.items()
                              if not key.endswith("attention") and key not in ("gate", "correction")}
            loss, pieces = alignment_loss(correct_state, shuffled_state, a.temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += loss.item()
            for key, value in pieces.items():
                totals[key] += float(value.detach())
            batches += 1
        row = {
            "stage": "semantic_alignment", "epoch": epoch,
            "epoch_seconds": time.time() - started,
            **{key: value / max(batches, 1) for key, value in totals.items()},
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    torch.save({"model": model.state_dict(), "history": history}, output / "alignment.pt")
    return history


@torch.no_grad()
def validate(model, loader, backbone, device, a):
    model.eval()
    totals = defaultdict(float)
    batches = 0
    for raw in loader:
        batch = v2.move(raw, device)
        _, hidden, skip = core.extract_frozen_features(backbone, batch["context_rgb"])
        baseline = core.decode_frozen(backbone, hidden, skip)
        correct_hidden = model(hidden, *v2.omega_inputs(batch))
        correct_probability = model.last["match_probability"].mean().item()
        correct = core.decode_frozen(backbone, correct_hidden, skip)
        shuffled_hidden = model(hidden, *v2.omega_inputs(batch, shuffled=True))
        shuffled_probability = model.last["match_probability"].mean().item()
        shuffled = core.decode_frozen(backbone, shuffled_hidden, skip)
        target = batch["future_rgb"]
        totals["MSE"] += (correct - target).square().mean().item()
        totals["MAE"] += (correct - target).abs().mean().item()
        totals["baseline_MSE"] += (baseline - target).square().mean().item()
        totals["shuffled_MSE"] += (shuffled - target).square().mean().item()
        totals["correct_vs_baseline_l1"] += (correct - baseline).abs().mean().item()
        totals["correct_vs_shuffled_l1"] += (correct - shuffled).abs().mean().item()
        totals["correct_match_probability"] += correct_probability
        totals["shuffled_match_probability"] += shuffled_probability
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def run_forecast_stage(model, loaders, backbone, lpips_model, device, a, output):
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, a.epochs)
    history = []
    best = float("inf")
    for epoch in range(1, a.epochs + 1):
        model.train()
        totals = defaultdict(float)
        batches = 0
        started = time.time()
        for raw in loaders["train"]:
            batch = v2.move(raw, device)
            with torch.no_grad():
                _, hidden, skip = core.extract_frozen_features(backbone, batch["context_rgb"])
                baseline = core.decode_frozen(backbone, hidden, skip)
            correct_hidden = model(hidden, *v2.omega_inputs(batch))
            correct_state = {key: value for key, value in model.last.items()
                             if not key.endswith("attention") and key != "gate"}
            correction = model.last["correction"]
            correct = core.decode_frozen(backbone, correct_hidden, skip)
            shuffled_hidden = model(hidden, *v2.omega_inputs(batch, shuffled=True))
            shuffled_state = {key: value for key, value in model.last.items()
                              if not key.endswith("attention") and key != "gate"}
            shuffled = core.decode_frozen(backbone, shuffled_hidden, skip)
            video_loss, pieces = v2.objective(
                correct, shuffled, baseline, batch["future_rgb"],
                batch["context_rgb"][:, -1], correction, lpips_model, a,
            )
            semantic_loss, semantic_pieces = alignment_loss(
                correct_state, shuffled_state, a.temperature
            )
            loss = video_loss + a.alignment_weight * semantic_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += loss.item()
            totals["video_loss"] += video_loss.item()
            totals["semantic_loss"] += semantic_loss.item()
            for key, value in {**pieces, **semantic_pieces}.items():
                totals[key] += float(value.detach())
            batches += 1
        scheduler.step()
        validation = validate(model, loaders["val"], backbone, device, a)
        mismatch = max(
            validation["correct_match_probability"] -
            validation["shuffled_match_probability"], 0.0
        )
        selection = validation["MSE"] - 1e-4 * mismatch
        row = {
            "stage": "semantic_forecast", "epoch": epoch,
            "epoch_seconds": time.time() - started,
            "lr": scheduler.get_last_lr()[0], "selection": selection,
            **{f"train_{key}": value / max(batches, 1) for key, value in totals.items()},
            **{f"val_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        state = {"model": model.state_dict(), "epoch": epoch,
                 "validation": validation, "config": vars(a)}
        torch.save(state, output / "last.pt")
        if selection < best:
            best = selection
            torch.save(state, output / "best.pt")
        print(json.dumps(row), flush=True)
    selected = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    return history, int(selected["epoch"])


def main():
    a = arguments()
    v2.seed_all(a.seed)
    root = Path(a.d4rt_root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "forecasting"))
    from simvp_geometry_model import build_video_predictor_from_checkpoint
    import lpips

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loaders, datasets, split_metadata = v2.make_loaders(a)
    checkpoint = torch.load(a.backbone_checkpoint, map_location="cpu", weights_only=False)
    if str(checkpoint.get("backbone", "simvp")).lower() != "tau":
        raise ValueError("semantic Omega alignment requires a TAU checkpoint")
    backbone = build_video_predictor_from_checkpoint(checkpoint).to(device)
    backbone.load_state_dict(checkpoint["model"])
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    future_frames = int(checkpoint["future_frames"])
    channels = int(backbone.backbone_config["hid_s"])
    model = SemanticOmegaAdapterV3(
        channels, future_frames, a.semantic_dim, a.feature_scale
    ).to(device)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad_(False)

    alignment_history = run_alignment_stage(
        model, loaders["train"], backbone, device, a, output
    )
    forecast_history, best_epoch = run_forecast_stage(
        model, loaders, backbone, lpips_model, device, a, output
    )
    metrics, records, bootstrap, utilization = v2.evaluate(
        model, loaders["test"], backbone, device, future_frames, a.seed
    )
    semantic_test = validate(model, loaders["test"], backbone, device, a)
    summary = {
        "metrics": metrics,
        "paired_bootstrap": bootstrap,
        "geometry_utilization": utilization,
        "semantic_alignment": semantic_test,
        "meta": {
            "config": vars(a), "split": split_metadata,
            "used": {name: len(dataset) for name, dataset in datasets.items()},
            "best_epoch": best_epoch,
            "alignment_epochs": len(alignment_history),
            "trainable_params": sum(p.numel() for p in model.parameters()),
            "architecture": (
                "TAU content-conditioned queries over unpooled VGGT camera/register memory; "
                "relative pose on camera slots; contrastive and matched-vs-shuffled calibration; "
                "confidence-gated latent FiLM and depth residual"
            ),
            "leakage_policy": "VGGT-Omega sees observed context frames only",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "per_sequence.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
