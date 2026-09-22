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
LAB_FILES = sorted([*LABS_DIR.glob("ch*.py"), *LABS_DIR.glob("appendix_*.py")])
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


_TWEAK = re.compile(r"^(?P<lhs>\s*[A-Za-z_][\w.\[\]\"']*\s*=\s*)(?P<old>.+?)\s*#\s*TWEAK-(?P<k>\d+):\s*(?P<new>.+?)\s*$")


def _tweaks() -> list[tuple[Path, str, int]]:
    out = []
    for lab in LAB_FILES:
        for i, line in enumerate(lab.read_text(encoding="utf-8").splitlines()):
            if m := _TWEAK.match(line):
                out.append((lab, m.group("k"), i))
    return out


@pytest.mark.parametrize("lab,k,lineno", _tweaks() or [pytest.param(None, "", 0, marks=pytest.mark.skip(reason="no TWEAK markers yet"))],
                         ids=lambda v: v.name if isinstance(v, Path) else str(v))
def test_lab_tweaks_do_not_crash(lab: Path, k: str, lineno: int, tmp_path: Path) -> None:
    """Every "改一改" exercise is marked in its lab as `NAME = value  # TWEAK-k: new value`.

    Apply the change to a copy and run it.  Checks that lock the chapter's numbers are
    EXPECTED to turn into ✗ (exit code 1); what must not happen is a traceback, i.e. the
    script has to reach done().  A diverging computation crashing the plotting code is the
    classic way this breaks.
    """
    req = _requirements(lab)
    if "gpu" in req:
        pytest.skip(f"{lab.name} needs a GPU")
    if "run_dir" in req and not _usable_run_dir_exists():
        pytest.skip(f"{lab.name} needs a learnzh-* run dir")
    lines = lab.read_text(encoding="utf-8").splitlines()
    m = _TWEAK.match(lines[lineno])
    lines[lineno] = f"{m.group('lhs')}{m.group('new')}"
    copy = tmp_path / lab.name
    copy.write_text("\n".join(lines) + "\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(copy)],
        cwd=str(LABS_DIR),
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "LEARNZH_FIG_DIR": str(tmp_path / "figures"), "PYTHONPATH": str(LABS_DIR)},
    )
    reached_done = "✓ 全部通过" in proc.stdout or "项没对上" in proc.stdout
    assert reached_done and "Traceback" not in proc.stderr, (
        f"{lab.name} TWEAK-{k} crashed instead of finishing:\n{proc.stdout[-1500:]}\n{proc.stderr[-2500:]}"
    )


def test_worked_examples() -> None:
    """The stdlib-only recomputation of every hand-worked number quoted in the chapters.

    It is frozen while the chapters are rewritten: a pass here plus
    test_learn_docs.py::test_worked_examples_are_still_quoted means the hand computations
    were moved, not changed.
    """
    proc = subprocess.run(
        [sys.executable, str(LABS_DIR / "worked_examples.py")],
        cwd=str(LABS_DIR),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"worked_examples.py failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    assert "全部手算复核通过" in proc.stdout
