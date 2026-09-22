"""教材实验脚本的公共小工具。

每个 chNN_*.py 只依赖这里的几个函数：
  banner(title)              打印一个分节标题
  table(headers, rows)       打印一张对齐的纯文本表格（floatfmt 定精度；num(v) 把单个数转成正文用的写法）
  check(name, cond)          核对一个数，打印 ✓ / ✗；✗ 不中断脚本，由 done() 汇总成非 0 退出码
  lines_in_order(text, ls)   核对"映射到项目"里引用的源码行还原样存在、顺序一致
  savefig(fig, name)         把图保存到 docs/learn-zh/figures/<name>.png
  done()                     全部 ✓ 才打印"✓ 全部通过"
课堂式讲解图的画法在 _draw.py。
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


def table(headers: list[str], rows: list[list], floatfmt: str = ".6g") -> None:
    """打印对齐表格。浮点数默认保留 6 位有效数字；要贴进正文的表，用 floatfmt 按正文引用的精度打印
    （如 ".4g" = 4 位有效数字，".4f" = 4 位小数）。个别格子想单独定格式，就先自己转成字符串。"""

    def cell(v) -> str:
        if isinstance(v, float):
            return format(v, floatfmt)
        return str(v)

    srows = [[cell(v) for v in r] for r in rows]
    widths = [max(_dw(h), *(_dw(r[i]) for r in srows)) for i, h in enumerate(headers)]
    print("  ".join(_rjust(h, w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for r in srows:
        print("  ".join(_rjust(c, w) for c, w in zip(r, widths)))


def num(v, places: int = 4) -> str:
    """要贴进正文的数：最多 places 位小数，去掉尾零（1.4364 → "1.4364"，2.5600 → "2.56"，−0.0 → "0"）。
    同一张表里大小悬殊的数（11.5636 和 301 并列）用它先转成字符串，比单个 floatfmt 好看。"""
    s = f"{v:.{places}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


_failures = 0


def check(name: str, cond: bool) -> None:
    """核对一个数：对了打印 ✓，没对上打印 ✗ 并记一笔，脚本继续往下跑。

    不中途抛异常：读者按"改一改"改了参数，锁正文数字的检查本来就该变 ✗，
    这时他要能看到后面的表和图。done() 会汇总，有 ✗ 就以非 0 退出码结束。
    """
    global _failures
    ok = bool(cond)
    print(f"  {'✓' if ok else '✗'} {name}")
    if not ok:
        _failures += 1


def lines_in_order(source: str, wanted: list[str]) -> bool:
    """wanted 里的每一行都原样出现在 source 里，并且先后顺序一致（用来核对正文引用的源码）。"""
    at = 0
    for line in wanted:
        at = source.find(line, at)
        if at < 0:
            return False
        at += len(line)
    return True


_has_cjk_font: bool | None = None  # use_headless_matplotlib() 查过之后才知道


def savefig(fig, name: str) -> Path:
    path = FIG_DIR / f"{name}.png"
    redirected = bool(os.environ.get("LEARNZH_FIG_DIR"))
    if _has_cjk_font is False and not redirected:
        # 没有中文字体时图里的中文全是方框；宁可不存，也不覆盖仓库里已经画好的图。
        print(f"  ⚠ 未找到中文字体，跳过保存 {path.name}（装好 Noto Sans CJK 后重跑即可；数字部分不受影响）")
        return path
    if _failures > 0 and not redirected:
        # 前面已经有检查没对上（多半是读者按"改一改"改了参数）：图照画，但不覆盖仓库里和正文配套的那张。
        import tempfile

        path = Path(tempfile.gettempdir()) / "learn-zh-figures" / f"{name}.png"
        print(f"  ⚠ 前面有检查没对上，这张图不覆盖教材里的图，另存到 {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    except (MemoryError, OverflowError, ValueError) as err:
        # 计算发散之后坐标会大到画不出来（inf / nan / 1e150）。这不是电脑坏了，跳过这张图继续跑。
        print(f"  ⚠ {name}.png 画不出来（{type(err).__name__}）：数字太大或不是有限值，多半是计算发散了。跳过这张图。")
        return path
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

    global _has_cjk_font
    installed = {f.name for f in font_manager.fontManager.ttflist}
    preferred = ["Noto Sans CJK SC", "Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Micro Hei", "PingFang SC", "SimHei"]
    found = [f for f in preferred if f in installed]
    if _has_cjk_font is None and not found:
        print("  ⚠ 未找到中文字体（Noto Sans CJK / 文泉驿 / 苹方 / 黑体），图里的中文会显示成方框")
    _has_cjk_font = bool(found)
    plt.rcParams["font.family"] = found + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def done() -> None:
    if _failures == 0:
        print("\n✓ 全部通过")
    else:
        print(f"\n✗ {_failures} 项没对上。如果你刚按“改一改”改过参数，这是预期的："
              "这些检查锁的是正文里的数，改回去就会全部通过。")
    sys.exit(0 if _failures == 0 else 1)
