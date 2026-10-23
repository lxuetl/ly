"""Stream a PointOdyssey tar archive one sequence at a time into WorldTrack clips."""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

from convert_pointodyssey_vlbm import REQUIRED, convert_sequence


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--clip-frames", type=int, default=24)
    p.add_argument("--max-clips-per-sequence", type=int, default=10)
    p.add_argument("--max-tracks", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-sequences", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def wanted(parts: tuple[str, ...]) -> bool:
    if len(parts) == 2 and parts[1] in (*REQUIRED, "scene_info.json"):
        return True
    return len(parts) == 3 and parts[1] == "rgbs" and parts[2].lower().endswith((".jpg", ".jpeg"))


def safe_member(member: tarfile.TarInfo, sequence: str) -> bool:
    parts = PurePosixPath(member.name).parts
    return bool(parts and parts[0] == sequence and ".." not in parts and wanted(parts))


def main():
    a = args()
    # Compatibility attributes consumed by convert_sequence().
    a.compressed = False
    archive, work, out = Path(a.archive), Path(a.work_dir), Path(a.output_root)
    work.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    rows, current, count = [], None, 0

    def finish(name: str | None):
        nonlocal count
        if not name:
            return
        seq = work / name
        complete = all((seq / x).is_file() for x in REQUIRED) and (seq / "rgbs").is_dir()
        try:
            if complete:
                made = convert_sequence(seq, out, a)
                rows.extend(made); count += 1
                print(f"[stream {count}] {name}: {sum(x['status'] in ('ok','exists') for x in made)} clips", flush=True)
            else:
                rows.append({"status": "error", "sequence": name, "error": "incomplete archive members"})
                print(f"[stream] {name}: incomplete", flush=True)
        except Exception as exc:
            rows.append({"status": "error", "sequence": name, "error": repr(exc)})
            print(f"[stream] {name}: ERROR {exc}", flush=True)
        finally:
            shutil.rmtree(seq, ignore_errors=True)

    # r|gz keeps memory bounded and permits immediate processing/cleanup.
    with tarfile.open(archive, mode="r|gz") as tf:
        for member in tf:
            parts = PurePosixPath(member.name).parts
            if not parts:
                continue
            sequence = parts[0]
            if current is None:
                current = sequence
            elif sequence != current:
                finish(current)
                if a.limit_sequences and count >= a.limit_sequences:
                    current = None
                    break
                current = sequence
            if safe_member(member, sequence):
                tf.extract(member, path=work, set_attrs=False)
        if current is not None and (not a.limit_sequences or count < a.limit_sequences):
            finish(current)

    manifest = out / "stream_conversion_manifest.jsonl"
    manifest.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    summary = {"sequences_processed": count,
               "clips_ready": sum(x["status"] in ("ok", "exists") for x in rows),
               "errors": sum(x["status"] == "error" for x in rows)}
    (out / "stream_conversion_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
