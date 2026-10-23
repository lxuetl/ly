"""Recover selected PointOdyssey train sequences from the official split archive.

The official train archive is one gzip stream split into four large files.  This
script downloads one part at a time, exposes the parts as one non-seekable byte
stream to ``tarfile``, deletes each consumed temporary part, and converts only
the requested sequences into the compact WorldTrack clip format.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import numpy as np

from convert_pointodyssey_vlbm import convert_sequence


DEFAULT_PARTS = (
    "train.tar.gz.partaa",
    "train.tar.gz.partab",
    "train.tar.gz.partac",
    "train.tar.gz.partad",
)

DEFAULT_TARGETS = (
    "character",
    "character0",
    "character0_",
    "character0_f",
    "character0_f2",
    "character1",
    "character1_f",
    "character2",
    "character2_",
    "character2_f",
    "character3",
    "character3_f",
    "character4",
    "character4_",
    "character4_f",
    "character5",
    "character5_",
    "character5_f",
    "character6",
    "character6_f",
    "gso_in_big",
    "gso_out_big",
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}", flush=True)


class DownloadingConcatStream(io.RawIOBase):
    """Concatenate resumably downloaded split files without storing them all."""

    def __init__(
        self,
        *,
        repo_id: str,
        parts: tuple[str, ...],
        spool_dir: Path,
        hf_cli: Path,
        retries: int,
    ) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.parts = parts
        self.spool_dir = spool_dir
        self.hf_cli = hf_cli
        self.retries = retries
        self.part_index = 0
        self.current_path: Path | None = None
        self.current_file: io.BufferedReader | None = None
        self.spool_dir.mkdir(parents=True, exist_ok=True)

    def readable(self) -> bool:
        return True

    def _download(self, name: str) -> Path:
        destination = self.spool_dir / name
        if destination.is_file() and destination.stat().st_size > 0:
            log(f"REUSE_COMPLETE_PART name={name} bytes={destination.stat().st_size}")
            return destination

        env = os.environ.copy()
        env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        env.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
        env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
        env.setdefault("HF_HUB_DISABLE_XET", "1")

        for attempt in range(1, self.retries + 1):
            log(f"DOWNLOAD_START name={name} attempt={attempt}")
            result = subprocess.run(
                [
                    str(self.hf_cli),
                    "download",
                    self.repo_id,
                    name,
                    "--repo-type",
                    "dataset",
                    "--local-dir",
                    str(self.spool_dir),
                ],
                env=env,
                check=False,
            )
            if result.returncode == 0 and destination.is_file() and destination.stat().st_size > 0:
                log(f"DOWNLOAD_DONE name={name} bytes={destination.stat().st_size}")
                return destination
            log(f"DOWNLOAD_RETRY name={name} rc={result.returncode}")
            time.sleep(min(30 * attempt, 300))
        raise RuntimeError(f"Failed to download {name} after {self.retries} attempts")

    def _finish_current(self) -> None:
        if self.current_file is not None:
            self.current_file.close()
            self.current_file = None
        if self.current_path is not None:
            size = self.current_path.stat().st_size if self.current_path.exists() else 0
            self.current_path.unlink(missing_ok=True)
            log(f"PART_CONSUMED_AND_REMOVED name={self.current_path.name} bytes={size}")
            self.current_path = None

    def _open_next(self) -> bool:
        self._finish_current()
        if self.part_index >= len(self.parts):
            return False
        name = self.parts[self.part_index]
        self.part_index += 1
        self.current_path = self._download(name)
        self.current_file = self.current_path.open("rb")
        log(f"PART_STREAMING name={name}")
        return True

    def readinto(self, buffer: bytearray | memoryview) -> int:
        view = memoryview(buffer)
        while True:
            if self.current_file is None and not self._open_next():
                return 0
            assert self.current_file is not None
            count = self.current_file.readinto(view)
            if count:
                return int(count)
            self._finish_current()

    def close(self) -> None:
        self._finish_current()
        super().close()


def member_sequence(name: str) -> tuple[str | None, tuple[str, ...]]:
    parts = PurePosixPath(name.lstrip("./")).parts
    if not parts:
        return None, ()
    if "train" in parts:
        index = parts.index("train")
        if len(parts) <= index + 1:
            return None, ()
        return parts[index + 1], tuple(parts[index + 2 :])
    if len(parts) >= 2:
        return parts[0], tuple(parts[1:])
    return None, ()


def wanted(relative: tuple[str, ...]) -> bool:
    if relative == ("anno.npz",):
        return True
    return (
        len(relative) == 2
        and relative[0] == "rgbs"
        and relative[1].lower().endswith((".jpg", ".jpeg", ".png"))
    )


def prepare_vlbm_fields(sequence_dir: Path) -> None:
    annotation_path = sequence_dir / "anno.npz"
    annotation = np.load(annotation_path, allow_pickle=True)
    trajs_2d = np.asarray(annotation["trajs_2d"])
    trajs_3d = np.asarray(annotation["trajs_3d"])
    intrinsics = np.asarray(annotation["intrinsics"])
    extrinsics = np.asarray(annotation["extrinsics"])
    visibility_key = "visibs" if "visibs" in annotation else "visibilities"
    visibilities = np.asarray(annotation[visibility_key])

    frame_count = int(trajs_2d.shape[0])
    if intrinsics.ndim == 2:
        intrinsics = np.repeat(intrinsics[None], frame_count, axis=0)
    if extrinsics.ndim == 2:
        extrinsics = np.repeat(extrinsics[None], frame_count, axis=0)

    np.save(sequence_dir / "trajs_2d.npy", trajs_2d.astype(np.float16))
    np.save(sequence_dir / "trajs_3d.npy", trajs_3d.astype(np.float16))
    np.save(sequence_dir / "intrinsics.npy", intrinsics.astype(np.float16))
    np.save(sequence_dir / "extrinsics.npy", extrinsics.astype(np.float16))
    np.save(sequence_dir / "visibilities.npy", visibilities.astype(np.float16))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--spool-dir", required=True, type=Path)
    parser.add_argument("--repo-id", default="aharley/pointodyssey")
    parser.add_argument("--hf-cli", type=Path, default=Path("/root/miniconda3/bin/hf"))
    parser.add_argument("--download-retries", type=int, default=30)
    parser.add_argument("--clip-frames", type=int, default=24)
    parser.add_argument("--max-clips-per-sequence", type=int, default=10)
    parser.add_argument("--max-tracks", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS))
    args = parser.parse_args()

    targets = set(args.targets)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.spool_dir.mkdir(parents=True, exist_ok=True)

    conversion_args = SimpleNamespace(
        clip_frames=args.clip_frames,
        max_clips_per_sequence=args.max_clips_per_sequence,
        max_tracks=args.max_tracks,
        seed=args.seed,
        compressed=False,
        overwrite=False,
    )

    stream = DownloadingConcatStream(
        repo_id=args.repo_id,
        parts=DEFAULT_PARTS,
        spool_dir=args.spool_dir,
        hf_cli=args.hf_cli,
        retries=args.download_retries,
    )
    recovered: set[str] = set()
    seen: set[str] = set()
    rows: list[dict] = []
    failures: dict[str, str] = {}

    def finish(sequence: str | None) -> None:
        if not sequence or sequence not in targets or sequence in recovered:
            return
        sequence_dir = args.work_dir / sequence
        annotation = sequence_dir / "anno.npz"
        rgb_dir = sequence_dir / "rgbs"
        rgb_count = len(list(rgb_dir.glob("*.jpg"))) + len(list(rgb_dir.glob("*.png")))
        if not annotation.is_file() or rgb_count < args.clip_frames:
            return
        try:
            prepare_vlbm_fields(sequence_dir)
            made = convert_sequence(sequence_dir, args.output_root, conversion_args)
            rows.extend(made)
            usable = sum(row.get("status") in {"ok", "exists"} for row in made)
            if usable <= 0:
                raise RuntimeError("conversion produced no usable clips")
            recovered.add(sequence)
            failures.pop(sequence, None)
            log(f"SEQUENCE_RECOVERED name={sequence} rgbs={rgb_count} clips={usable}")
            shutil.rmtree(sequence_dir, ignore_errors=True)
        except Exception as exc:  # retain files for diagnosis
            failures[sequence] = repr(exc)
            log(f"SEQUENCE_ERROR name={sequence} error={exc!r}")

    current_sequence: str | None = None
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                sequence, relative = member_sequence(member.name)
                if sequence != current_sequence:
                    finish(current_sequence)
                    current_sequence = sequence
                if sequence not in targets or not wanted(relative):
                    continue
                if ".." in relative or member.issym() or member.islnk():
                    continue
                seen.add(sequence)
                member.name = str(PurePosixPath(sequence, *relative))
                archive.extract(member, path=args.work_dir, set_attrs=False)
            finish(current_sequence)
    finally:
        stream.close()

    for sequence in sorted(targets - recovered):
        finish(sequence)

    missing = sorted(targets - recovered)
    for sequence in missing:
        failures.setdefault(sequence, "target files were not complete in the official archive")

    manifest_path = args.output_root / "official_recovery_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "targets": len(targets),
        "targets_seen": sorted(seen),
        "recovered": len(recovered),
        "recovered_names": sorted(recovered),
        "missing": missing,
        "failures": failures,
        "clips_ready": sum(row.get("status") in {"ok", "exists"} for row in rows),
    }
    (args.output_root / "official_recovery_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"RECOVERY_SUMMARY {json.dumps(summary, ensure_ascii=False)}")
    if not missing:
        shutil.rmtree(args.work_dir, ignore_errors=True)
        log("ALL_TARGETS_RECOVERED_AND_WORKDIR_CLEANED")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
