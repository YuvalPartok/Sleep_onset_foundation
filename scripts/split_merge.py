#!/usr/bin/env python3
"""Split a merged ensemble back into its CV builds.

merge_builds.py concatenates the selected builds' members in candidate order
(5 fold members each, member j of a build belonging to fold j % 5). Given the
merged build and the labels of the builds it selected, in that order, this
writes build/<label>/weights.pt for each (plain averaging, the merged
quantiles), so they can be candidates again (scripts/trim_oof.py, then
merge_builds.py --oof-suffix _oofT). Existing builds are left untouched.

Usage:
    python scripts/split_merge.py r12_merge_r11 r8_ext1_s0 r9_all05_s0 ...
"""

import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    merged, labels = sys.argv[1], sys.argv[2:]
    src = ROOT / "build" / merged
    state = torch.load(src / "weights.pt", map_location="cpu", weights_only=True)
    n = len({k.split(".")[1] for k in state if k.startswith("members.")})
    if n % len(labels) or n // len(labels) != 5:
        raise SystemExit(f"{merged}: {n} members do not split into "
                         f"{len(labels)} builds of 5")
    for b, label in enumerate(labels):
        dst = ROOT / "build" / label
        if (dst / "weights.pt").exists():
            print(f"{label}: already present")
            continue
        out = {}
        for i in range(5):
            pre = f"members.{5 * b + i}."
            for k, v in state.items():
                if k.startswith(pre):
                    out[f"members.{i}." + k[len(pre):]] = v
        out["member_w"] = torch.ones(5)
        out["q_time"] = state["q_time"].clone()
        out["quantile"] = state["quantile"].clone()
        dst.mkdir(parents=True, exist_ok=True)
        torch.save(out, dst / "weights.pt")
        if (src / "submission.py").exists():
            shutil.copy(src / "submission.py", dst / "submission.py")
        print(f"{label}: members {5 * b}-{5 * b + 4} of {merged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
