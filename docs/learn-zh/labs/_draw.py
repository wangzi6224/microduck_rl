"""课堂式讲解图的公共画法：统一配色、字号和分步版式。

第 2 章后五张图（ch02_number_to_tensor 等）就是这个长相：标题写结论、①②③ 分步、
手算直接写进图里、每步下面一句灰色大白话。各章实验这样用：

    from _draw import plt, INK, BLUE, lesson_figure, lesson_panel, lesson_cells, hand, note

    fig, axes = lesson_figure(3, "矩阵乘向量：每行是一份配方")
    lesson_panel(axes[0], "① 输入 x：3 个数，顺序固定")
    lesson_cells(axes[0], [[1, 2, 3]], left=1.6, bottom=1.6, width=1.7, height=0.9)
    hand(axes[0], 0.4, 0.6, "1 × 1 + 0 × 2 + 2 × 3 = 7")
    note(axes[0], 0.4, 0.2, "同一位置配对相乘，再把三个乘积加起来。")
    savefig(fig, "chNN_xxx")

数据图（曲线、等高线、散点）也要长得像课堂式讲解图：

    fig = plt.figure(figsize=(9, 11))
    ax = fig.add_axes([0.12, 0.55, 0.8, 0.33])
    data_axes(ax, "旋钮 x", "旋钮 y")            # 中文坐标轴、浅网格、去掉上右边框
    ...画线、画点、用 arrow() 画箭头、用 bbox=WHITE_BOX 给压在线上的字垫白底...
    panel_title(fig, [ax], "① α = 0.01：从 (1, 2) 出发连走 300 步")   # 面板里的东西都画完之后再调用
    panel_note(fig, [ax], "每一步都垂直穿过等高线往里走。")

这里只放三个以上实验都会用到的东西；某一章独有的画法写在那一章的实验里。
图里的数字要和 check() 用同一组变量算出来，不要手抄。
"""

from __future__ import annotations

from _common import use_headless_matplotlib

plt = use_headless_matplotlib()

# 配色与第 2 章一致
INK = "#243442"     # 标题、格子里的数字
MUTED = "#546574"   # 每步下面的大白话
BLUE = "#2065a8"    # 第一组手算 / 标注
GREEN = "#28745a"   # 第二组手算 / 标注
ORANGE = "#b75b25"  # 高亮："看这里"
CELL = "#eaf1f8"    # 格子底色
CELL_EDGE = "#8ea5b9"
CELL_HOT = "#fff0d9"  # 高亮格子底色
GRID = "#dae2eb"      # 数据图的网格线
FAINT = "#9fb0c0"     # 数据图里不强调的线（背景等高线、参考曲线）
WHITE_BOX = dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2.5)  # 压在线上的字：bbox=WHITE_BOX

FS_TITLE = 23  # 整张图的标题（写结论）
FS_STEP = 20   # ①②③ 小标题、格子里的数字、手算
FS_NOTE = 18   # 灰色说明句
FS_SMALL = 17  # 格子旁的小标签、数据图的坐标轴标题
FS_TICK = 16   # 数据图的刻度数字（图内贴在线上的小数字不小于 13）
PANEL_LEFT = 0.055  # 数据面板的小标题和灰字统一从这条左边线起笔（figure 坐标）


def lesson_figure(n_panels: int, title: str, panel_height: float = 3.45, width: float = 9.0):
    """竖着排 n 个分步面板的整张图；title 写这张图的结论。返回 (fig, [ax, ...])。"""
    fig, axes = plt.subplots(n_panels, 1, figsize=(width, panel_height * n_panels))
    axes = [axes] if n_panels == 1 else list(axes)
    fig.subplots_adjust(top=1 - 0.24 / n_panels, hspace=0.14)
    fig.suptitle(title, fontsize=FS_TITLE, fontweight="bold", color=INK)
    return fig, axes


def lesson_panel(ax, title: str = "", xmax: float = 10.0, ymax: float = 4.0) -> None:
    """一个分步面板：坐标范围固定为 0..xmax × 0..ymax，去掉坐标轴，左上角写 "① …" 小标题（可不写）。"""
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    if title:
        ax.text(0, ymax - 0.3, title, fontsize=FS_STEP, fontweight="bold", color=INK, va="top")


def lesson_cells(ax, values, left: float, bottom: float, width: float = 1.0, height: float = 0.7,
                 facecolor: str = CELL, highlights=(), fmt: str = "{:g}", fontsize: int = FS_STEP) -> None:
    """按真实的行列画一张数字表；highlights 里的 (行, 列) 用橙色突出。values 是二维列表。"""
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            hot = (i, j) in highlights
            y = bottom + (len(values) - i - 1) * height
            ax.add_patch(plt.Rectangle((left + j * width, y), width, height,
                                       facecolor=CELL_HOT if hot else facecolor,
                                       edgecolor=ORANGE if hot else CELL_EDGE, lw=2 if hot else 1.2))
            text = value if isinstance(value, str) else fmt.format(value).replace("-", "−")
            ax.text(left + (j + 0.5) * width, y + height / 2, text, fontsize=fontsize, ha="center", va="center",
                    color=ORANGE if hot else INK, fontweight="bold" if hot else "normal")


def cell(ax, x: float, y: float, text: str, width: float = 1.0, height: float = 0.7,
         facecolor: str = CELL, edgecolor: str = CELL_EDGE, fontsize: int = FS_STEP, color: str = INK) -> None:
    """单独一个带字的格子（放标签、单位、运算符结果等）。"""
    ax.add_patch(plt.Rectangle((x, y), width, height, facecolor=facecolor, edgecolor=edgecolor, lw=1.2))
    ax.text(x + width / 2, y + height / 2, text, fontsize=fontsize, ha="center", va="center", color=color)


def hand(ax, x: float, y: float, text: str, color: str = BLUE, fontsize: int = FS_STEP, **kw) -> None:
    """写进图里的手算式或标注（蓝 / 绿）。"""
    ax.text(x, y, text, fontsize=fontsize, color=color, va="center", **kw)


def note(ax, x: float, y: float, text: str, fontsize: int = FS_NOTE, **kw) -> None:
    """每一步下面那句灰色的大白话。"""
    ax.text(x, y, text, fontsize=fontsize, color=MUTED, va="center", **kw)


# --- 数据图 -------------------------------------------------------------------------------
def data_axes(ax, xlabel: str, ylabel: str, grid: bool = True) -> None:
    """数据图的统一外观：中文坐标轴（带单位）、浅色网格、去掉上右边框。"""
    ax.set_xlabel(xlabel, fontsize=FS_SMALL, color=INK)
    ax.set_ylabel(ylabel, fontsize=FS_SMALL, color=INK)
    ax.tick_params(labelsize=FS_TICK, colors=MUTED, length=0, pad=6)
    if grid:
        ax.grid(color=GRID, linewidth=1.1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(CELL_EDGE)


def _extent(fig, axes_list):
    """几块坐标轴（含刻度、轴标题、图内文字）实际占到的上下边界，figure 坐标。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [a.get_tightbbox(renderer).transformed(fig.transFigure.inverted()) for a in axes_list]
    return min(b.y0 for b in boxes), max(b.y1 for b in boxes)


def panel_title(fig, axes_list, text: str, pad: float = 0.007) -> None:
    """在一块数据面板的上方写「① ……」小标题。要在面板里的东西都画完之后再调用。"""
    _, top = _extent(fig, axes_list)
    fig.text(PANEL_LEFT, top + pad, text, fontsize=FS_STEP, fontweight="bold", color=INK, va="bottom")


def panel_note(fig, axes_list, text: str, pad: float = 0.009) -> None:
    """在一块数据面板的下方写那句灰色的大白话。"""
    bottom, _ = _extent(fig, axes_list)
    fig.text(PANEL_LEFT, bottom - pad, text, fontsize=FS_NOTE, color=MUTED, va="top", linespacing=1.35)


def arrow(ax, start, end, color: str, lw: float = 3.0, style: str = "-|>", zorder: int = 5, **kw) -> None:
    """从 start 到 end 画一根实心箭头（两端不留空隙，箭头尖正好落在 end）。"""
    ax.annotate("", xy=end, xytext=start, zorder=zorder,
                arrowprops=dict(arrowstyle=style, lw=lw, color=color, shrinkA=0, shrinkB=0, **kw))
