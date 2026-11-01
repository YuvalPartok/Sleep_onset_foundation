#!/usr/bin/env python3
"""Train submission/submission.py with Benchopt, re-validate it, and ZIP it.

Steps (mirrors competition/tools/debug_submission.py):
1. Training run (``Sleep-onset[training=True]``): Solver.fit trains the model
   and the export writes submission.py + weights.pt into build/<label>/.
2. Inference-only run on the exported folder, exactly as Codabench runs it,
   to catch architecture/checkpoint mismatches.
3. ZIP the folder's files at the archive root into dist/<label>.zip.

Usage:
    python scripts/build_submission.py                 # synthetic, warm-up shapes
    python scripts/build_submission.py --data muse     # public Muse warm-up data
    python scripts/build_submission.py --data muse --solver submission/my_cnn.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "competition" / "tracks" / "sleep_onset"
DEFAULT_SOLVER = ROOT / "submission" / "submission.py"

DATASETS = {
    # Synthetic data with the real warm-up tensor contract (4 Muse channels,
    # 5 s at 128 Hz). Validates the pipeline only; the score is meaningless.
    "simulated": "Simulated[n_chans=4,n_times=640,n_train=300,n_test=150,sfreq=128]",
    # Public Muse warm-up data; run `benchopt prepare` on it first.
    "muse": "Interaxon2026Muse",
}


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=DATASETS, default="simulated")
    parser.add_argument("--solver", type=Path, default=DEFAULT_SOLVER,
                        help="solver file to train (default: submission/submission.py)")
    parser.add_argument("--label",
                        help="output name (default: <solver file stem>_<data>)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing build/dist output")
    args = parser.parse_args()

    benchopt = shutil.which("benchopt") or str(Path(sys.executable).with_name("benchopt"))
    if not Path(benchopt).exists():
        raise SystemExit("benchopt not found; activate the sleep_foundation env")

    solver = args.solver.resolve()
    if not solver.is_file():
        raise SystemExit(f"solver file not found: {solver}")
    stem = "mlp" if solver == DEFAULT_SOLVER else solver.stem
    label = args.label or f"{stem}_{args.data}"
    build_dir = ROOT / "build" / label
    zip_path = ROOT / "dist" / f"{label}.zip"
    if build_dir.exists() or zip_path.exists():
        if not args.force:
            raise SystemExit(f"{build_dir} or {zip_path} exists; use --force")
        shutil.rmtree(build_dir, ignore_errors=True)
        zip_path.unlink(missing_ok=True)
    build_dir.mkdir(parents=True)
    zip_path.parent.mkdir(exist_ok=True)

    env = {**os.environ, "COMPET_SUBMISSION_DIR": str(build_dir)}
    common = [benchopt, "run", str(TRACK), "-d", DATASETS[args.data],
              "-r", "1", "--no-cache", "--no-plot", "--no-html",
              "--no-display", "--local"]

    # 1. Train and export.
    run(common + ["-o", "Sleep-onset[training=True]", "-s", str(solver),
                  "--output", str(ROOT / "build" / f"{label}_train.parquet")], env)
    for name in ("submission.py", "weights.pt"):
        if not (build_dir / name).is_file():
            raise SystemExit(f"training run did not export {name}")

    # 2. Reload the exported folder inference-only, as Codabench does.
    run(common + ["-o", "Sleep-onset[training=False]",
                  "-s", str(build_dir / "submission.py"),
                  "--output", str(ROOT / "build" / f"{label}_eval.parquet")], env)

    # 3. ZIP with every file at the archive root.
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(build_dir.iterdir()):
            if path.is_file():
                zf.write(path, path.name)
    print(f"\nReady to upload: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            print(f"  {info.filename:20s} {info.file_size:>12,d} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
