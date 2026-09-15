"""教材实验脚本的公共小工具。

每个 chNN_*.py 只依赖这里的三个函数：
  banner(title)              打印一个分节标题
  table(headers, rows)       打印一张对齐的纯文本表格
  check(name, cond)          断言 + 打印 ✓ / ✗（失败时抛异常，脚本退出码非 0）
  savefig(fig, name)         把图保存到 docs/learn-zh/figures/<name>.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

FIG_DIR = Path(os.environ.get("LEARNZH_FIG_DIR") or Path(__file__).resolve().parent.parent / "figures")


def banner(title: str) -> None:
    line = "=" * max(60, len(title) + 4)
    print(f"\n{line}\n  {title}\n{line}")


def _dw(s: str) -> int:
    """显示宽度：中文/全角字符算 2 格，其余算 1 格。"""
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _rjust(s: str, w: int) -> str:
    return " " * (w - _dw(s)) + s


def table(headers: list[str], rows: list[list]) -> None:
    """打印对齐表格。数字自动保留 6 位有效数字。"""

    def cell(v) -> str:
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    srows = [[cell(v) for v in r] for r in rows]
    widths = [max(_dw(h), *(_dw(r[i]) for r in srows)) for i, h in enumerate(headers)]
    print("  ".join(_rjust(h, w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for r in srows:
        print("  ".join(_rjust(c, w) for c, w in zip(r, widths)))


_failures = 0


def check(name: str, cond: bool) -> None:
    global _failures
    mark = "✓" if cond else "✗"
    print(f"  {mark} {name}")
    if not cond:
        _failures += 1
        raise AssertionError(name)


def savefig(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    try:
        shown = path.relative_to(Path(__file__).resolve().parents[3])
    except ValueError:
        shown = path
    print(f"  图已保存: {shown}")
    return path


def use_headless_matplotlib():
    """无窗口环境下画图（服务器 / SSH）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    preferred = ["Noto Sans CJK SC", "Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Micro Hei", "PingFang SC", "SimHei"]
    plt.rcParams["font.family"] = [f for f in preferred if f in installed] + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def done() -> None:
    print("\n✓ 全部通过" if _failures == 0 else f"\n✗ {_failures} 项失败")
    sys.exit(0 if _failures == 0 else 1)
