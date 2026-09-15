"""Smoke-run the textbook lab scripts under docs/learn-zh/labs/ (CPU-only ones).

A lab declares hard requirements with a dedicated token line near its top:
    # LAB_REQUIRES: gpu        -> needs cuda:0 (skipped here)
    # LAB_REQUIRES: run_dir    -> needs a prior ch14 smoke-training run dir with both
                                  model_*.pt and *.onnx (skipped when none exists)
Everything else must exit 0 and print the trailing "✓ 全部通过" line, which is how
the labs self-check the numbers quoted in the textbook chapters.  Figures are
redirected to tmp_path via LEARNZH_FIG_DIR so the test never rewrites tracked PNGs.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
LABS_DIR = REPO / "docs" / "learn-zh" / "labs"
LAB_FILES = sorted(p for p in LABS_DIR.glob("ch*.py"))
_REQ = re.compile(r"^# LAB_REQUIRES:\s*(\w+)", re.M)


def _requirements(path: Path) -> set[str]:
    return set(_REQ.findall(path.read_text(encoding="utf-8")[:3000]))


def _usable_run_dir_exists() -> bool:
    for d in (REPO / "logs" / "rsl_rl" / "velocity").glob("*learnzh-*"):
        if list(d.glob("*.onnx")) and list(d.glob("model_*.pt")):
            return True
    return False


@pytest.mark.parametrize("lab", LAB_FILES, ids=[p.name for p in LAB_FILES])
def test_lab_runs(lab: Path, tmp_path: Path) -> None:
    req = _requirements(lab)
    if "gpu" in req:
        pytest.skip(f"{lab.name} needs a GPU")
    if "run_dir" in req and not _usable_run_dir_exists():
        pytest.skip(f"{lab.name} needs a learnzh-* smoke-training run dir with .onnx + model_*.pt (run ch14 first)")
    proc = subprocess.run(
        [sys.executable, str(lab)],
        cwd=str(LABS_DIR),
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "LEARNZH_FIG_DIR": str(tmp_path / "figures")},
    )
    assert proc.returncode == 0, f"{lab.name} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    assert "✓ 全部通过" in proc.stdout, f"{lab.name} did not print the pass marker"
