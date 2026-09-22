"""第 1 章实验：函数、变化率、导数（缩 h）、求导规则、链式法则、指数函数，以及项目的钟形打分。

运行：uv run python docs/learn-zh/labs/ch01_derivative.py
纯 CPU，numpy + matplotlib。第 15 节只读两份源码的文字，不加载机器人、不训练。
小节编号与正文一一对应：实验第 K 节 = 正文 1.K 节（第 15 节对应「映射到项目」）。
正文“改一改”要改的三行都带 `# TWEAK-k:` 标记（第 4、7、9 节各一处）。
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

from _common import banner, check, done, lines_in_order, num, savefig, table
from _draw import (BLUE, CELL, CELL_EDGE, CELL_HOT, FAINT, FS_NOTE, FS_SMALL, FS_STEP, FS_TICK, FS_TITLE, GREEN, INK, MUTED,
                   ORANGE, WHITE_BOX, arrow, cell, data_axes, hand, lesson_cells, lesson_figure, lesson_panel, note,
                   panel_note, panel_title, plt)


# --- 本章独有的小工具 ---------------------------------------------------------------------
def machine(ax, x, y, text, width=1.9, height=0.95, color=INK):
    """一台“机器”：圆角方框里写它做的事（总览图、链式法则图、钟形打分图共用）。"""
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.16",
                                facecolor="white", edgecolor=color, lw=2.2))
    ax.text(x + width / 2, y + height / 2, text, fontsize=FS_SMALL, ha="center", va="center", color=color)


def pipeline(ax, y, items, x=0.3, value_w=1.25, machine_w=1.9, gap=0.42, height=0.95, hot_last=True):
    """一条流水线：数 → 机器 → 数 → 机器 → ……。items 交替给出“数的文字”和“机器的文字”。返回每个数的中心 x。"""
    centers = []
    for k, text in enumerate(items):
        is_value = k % 2 == 0
        w = value_w if is_value else machine_w
        if is_value:
            last = hot_last and k == len(items) - 1
            cell(ax, x, y, text, width=w, height=height, facecolor=CELL_HOT if last else CELL,
                 edgecolor=ORANGE if last else CELL_EDGE, color=ORANGE if last else INK)
            centers.append(x + w / 2)
        else:
            machine(ax, x, y, text, width=w, height=height)
        x += w
        if k < len(items) - 1:
            arrow(ax, (x + 0.06, y + height / 2), (x + gap - 0.06, y + height / 2), MUTED, lw=2.2)
            x += gap
    return centers


def shrink_h(func, at, hs):
    """缩 h：从 at 出发走 h，平均变化率 = (func(at + h) − func(at)) / h。"""
    return [[h, func(at + h), (func(at + h) - func(at)) / h] for h in hs]


# ---------------------------------------------------------------------------
banner("1. 函数：一台机器，喂进去一个数，吐出来一个数")


def f(x):
    return x**2          # 正文的 f(x) = x²；前端写法是 const square = (x) => x * x


for value in (3, -2, 0.5, 0):
    print(f"  喂 {value:g}，吐 f({value:g}) = {value:g} × {value:g} = {f(value):g}")
check("f(3) = 9，f(−2) = 4，f(0.5) = 0.25，f(0) = 0", f(3) == 9 and f(-2) == 4 and f(0.5) == 0.25 and f(0) == 0)
check("同一个输入，永远同一个输出：连喂三次 3，都吐 9", [f(3) for _ in range(3)] == [9, 9, 9])
check("自测：f(4) = f(−4) = 16", f(4) == 16 and f(-4) == 16)

# ---------------------------------------------------------------------------
banner("2. 函数的图像：把“喂了什么、吐了什么”记成表，每一行是一个点")
GRAPH_X = [-2, -1, 0, 1, 2, 3]
GRAPH_Y = [f(x) for x in GRAPH_X]
table(["喂 x", *[str(x) for x in GRAPH_X]], [["吐 f(x)", *GRAPH_Y]])
check("表里六个点：4、1、0、1、4、9", GRAPH_Y == [4, 1, 0, 1, 4, 9])
check("最低的点在 x = 0；x = −2 和 x = 2 一样高（图像左右对称）", min(GRAPH_Y) == f(0) == 0 and f(-2) == f(2) == 4)
check("自测：输出 4 对应两个输入 −2 和 2；输出 0 只对应一个输入 0（最低点）",
      [x for x in GRAPH_X if f(x) == 4] == [-2, 2] and [x for x in GRAPH_X if f(x) == 0] == [0])

# ---------------------------------------------------------------------------
banner("2b. 画图：figures/ch01_function_graph.png（表 → 点 → 曲线）")
fig = plt.figure(figsize=(9.4, 18.2))
gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 5.0, 5.0], hspace=0.50, left=0.12, right=0.95, top=0.925, bottom=0.075)
fig.suptitle("函数的图像：横着找输入，竖着读输出", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)

ax = fig.add_subplot(gs[0])
lesson_panel(ax, xmax=10, ymax=2.6)
lesson_cells(ax, [GRAPH_X, GRAPH_Y], 2.3, 0.35, 1.1, 0.8, highlights=((0, 5), (1, 5)))
ax.text(2.15, 1.55, "喂 x", ha="right", va="center", fontsize=FS_SMALL, color=MUTED)
ax.text(2.15, 0.75, "吐 f(x)", ha="right", va="center", fontsize=FS_SMALL, color=MUTED)
panel_title(fig, [ax], "① 先记一张表：喂了什么，吐了什么", pad=-0.006)
panel_note(fig, [ax], "f(x) = x²。橙色那一列：喂 3，吐 9。表里每一列，马上变成图上的一个点。", pad=0.0)


def graph_axes(ax):
    data_axes(ax, "输入 x（往右是正，往左是负）", "输出 f(x)（往上是大）")
    ax.set_xlim(-3.4, 3.9)
    ax.set_ylim(-1.2, 10.6)
    ax.set_xticks(range(-3, 4))
    ax.set_yticks(range(0, 11))          # 竖尺每个整数都标出来：要读的 9 必须在尺子上找得到
    ax.axhline(0, color=CELL_EDGE, lw=1.6)
    ax.axvline(0, color=CELL_EDGE, lw=1.6)


ax = fig.add_subplot(gs[1])
graph_axes(ax)
for x, y in zip(GRAPH_X, GRAPH_Y):
    ax.plot(x, y, "o", ms=11, color=ORANGE if x == 3 else BLUE, zorder=5)
arrow(ax, (3, 0), (3, 8.72), ORANGE, lw=2.4, zorder=4)
arrow(ax, (3, 9), (-3.36, 9), ORANGE, lw=2.4, zorder=4)      # 一直画到左边的竖尺，箭头尖正对着刻度 9
nine = ax.get_yticklabels()[9]
nine.set_color(ORANGE)
nine.set_fontweight("bold")
ax.text(2.88, 2.6, "先在横尺上\n找到 3，往上走", color=ORANGE, fontsize=FS_SMALL, ha="right", va="center", linespacing=1.3, bbox=WHITE_BOX)
ax.text(0.2, 9.35, "碰到点以后一路往左，看竖尺：读出 9", color=ORANGE, fontsize=FS_SMALL, ha="center", va="bottom")
ax.text(3.14, 9.0, "(3, 9)", color=ORANGE, fontsize=FS_SMALL, ha="left", va="center", fontweight="bold")
ax.text(-2.12, 4.35, "(−2, 4)", color=BLUE, fontsize=FS_SMALL, ha="right", va="bottom")
ax.text(0.18, -0.25, "(0, 0)", color=BLUE, fontsize=FS_SMALL, ha="left", va="top")
panel_title(fig, [ax], "② 一列一个点：横着找输入，竖着读输出")
panel_note(fig, [ax], "点越高，输出越大。喂 −2 那一点：横尺上标着 −2，高度 4。\n读数永远看尺子上标的数：这张图横着一格和竖着一格并不一样长。")

ax = fig.add_subplot(gs[2])
graph_axes(ax)
curve_x = np.linspace(-3.25, 3.25, 300)
ax.plot(curve_x, curve_x**2, color=BLUE, lw=3.5, zorder=3)
for x, y in zip(GRAPH_X, GRAPH_Y):
    ax.plot(x, y, "o", ms=9, color=BLUE, zorder=5)
ax.plot(0, 0, "o", ms=12, color=ORANGE, zorder=6)
ax.text(0.0, 1.1, "最低点：f(0) = 0", color=ORANGE, fontsize=FS_SMALL, ha="center", va="bottom", bbox=WHITE_BOX)
ax.plot([-2, 2], [4, 4], ls="--", lw=2, color=GREEN, zorder=2)
ax.text(0.0, 4.25, "f(−2) = f(2) = 4：左右一样高", color=GREEN, fontsize=FS_SMALL, ha="center", va="bottom", bbox=WHITE_BOX)
panel_title(fig, [ax], "③ 没算过的输入也各有一个点：全连起来，是一条 U 形的线")
panel_note(fig, [ax], "这条线叫抛物线。越往两边越高，而且越来越陡。")
savefig(fig, "ch01_function_graph")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("3. 变化率：变了多少 ÷ 走了多少；直线的变化率处处一样，叫斜率")
print(f"  开车 100 公里用 2 小时：平均速度 = 100 ÷ 2 = {100 / 2:g} 公里/小时")
check("100 ÷ 2 = 50", 100 / 2 == 50)


def line(x):
    return 2 * x + 1


LINE_X = [0, 1, 2, 3, 4]
table(["x", *[str(x) for x in LINE_X]], [["y = 2x + 1", *[line(x) for x in LINE_X]]])
slopes = {(a, b): (line(b) - line(a)) / (b - a) for a in LINE_X for b in LINE_X if a < b}
print("  任挑两列：从 x = 1 到 x = 4，y 从 3 到 9，(9 − 3) ÷ (4 − 1) =", slopes[(1, 4)])
check("y = 2x + 1：x 每加 1，y 加 2；任意两列算出的斜率都是 2", [line(x) for x in LINE_X] == [1, 3, 5, 7, 9]
      and all(s == 2 for s in slopes.values()))
downhill = [10 - 3 * x for x in (0, 1, 2)]
print("  下坡的直线 y = 10 − 3x：", downhill, "→ x 每加 1，y 减 3，斜率 −3")
check("y = 10 − 3x 的斜率是 −3（负号 = 往右走是下坡）", downhill == [10, 7, 4] and (downhill[2] - downhill[0]) / 2 == -3)


def down_line(x):
    return 10 - 3 * x


# ---------------------------------------------------------------------------
banner("3b. 画图：figures/ch01_slope.png（斜率 = 每往右 1 格，往上或往下几格）")
fig = plt.figure(figsize=(9.4, 14.6))
gs = fig.add_gridspec(2, 1, hspace=0.42, left=0.12, right=0.95, top=0.915, bottom=0.09)
fig.suptitle("斜率：每往右走 1，线往上（或往下）走多少", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
# 两幅用同一套刻度：哪条线更陡，才能用眼睛比
SLOPE_PANELS = [(line, [0, 1, 2, 3, 4], BLUE, "① y = 2x + 1：每往右走 1，往上走 2，斜率 2",
                 "横着走 1、竖着涨 2。换一段长的也一样：从 x = 1 到 x = 4，\n横着走 3、竖着涨 6，6 ÷ 3 = 2。直线上处处是同一个斜率。"),
                (down_line, [0, 1, 2, 3], GREEN, "② y = 10 − 3x：每往右走 1，往下走 3，斜率 −3",
                 "负号 = 往右走是下坡。两幅图的刻度相同：往右每走 1，\n这条线落 3、上面那条只涨 2，所以这条更陡——\n比陡不陡，看的是斜率的大小（不看正负号）。")]
for k, (func, xs_pts, color, title, caption) in enumerate(SLOPE_PANELS):
    ax = fig.add_subplot(gs[k])
    data_axes(ax, "输入 x", "输出 y")
    ax.set_xlim(-0.4, 4.6)
    ax.set_ylim(-0.6, 11.4)
    ax.set_xticks(range(0, 5))
    ax.set_yticks(range(0, 12))
    span = np.array([-0.3, 4.4]) if k == 0 else np.array([-0.3, 3.45])
    ax.plot(span, func(span), color=color, lw=3.5, zorder=3)
    for x in xs_pts:
        ax.plot(x, func(x), "o", ms=10, color=color, zorder=5)
    rise = func(2) - func(1)
    ax.plot([1, 2], [func(1), func(1)], color=ORANGE, lw=3.5, zorder=4)
    ax.plot([2, 2], [func(1), func(2)], color=ORANGE, lw=3.5, zorder=4)
    ax.text(1.5, func(1) + (-0.25 if k == 0 else 0.25), "往右 1", color=ORANGE, fontsize=FS_SMALL, ha="center", va="top" if k == 0 else "bottom")
    ax.text(2.1, (func(1) + func(2)) / 2, f"往{'上' if rise > 0 else '下'} {abs(rise):g}", color=ORANGE, fontsize=FS_SMALL, ha="left", va="center")
    if k == 0:
        ax.plot([1, 4], [func(1), func(1)], color=MUTED, lw=1.8, ls="--", zorder=2)
        ax.plot([4, 4], [func(1), func(4)], color=MUTED, lw=1.8, ls="--", zorder=2)
        ax.text(3.2, func(1) - 0.25, "横着走 3", color=MUTED, fontsize=FS_SMALL, ha="center", va="top")
        ax.text(4.08, 5.4, "竖着\n涨 6", color=MUTED, fontsize=FS_SMALL, ha="left", va="center", linespacing=1.3)
        ax.text(-0.2, 10.4, f"斜率 = {rise:g} ÷ 1 = {rise:g}", color=color, fontsize=FS_STEP, va="center")
    else:
        ax.text(1.55, 10.4, f"斜率 = −{abs(rise):g} ÷ 1 = −{abs(rise):g}", color=color, fontsize=FS_STEP, va="center")
    panel_title(fig, [ax], title)
    panel_note(fig, [ax], caption)
savefig(fig, "ch01_slope")
plt.close(fig)
check("图里标的数：y = 2x + 1 往右 1 涨 2、从 1 到 4 涨 6；y = 10 − 3x 往右 1 落 3；图上的点 (0,10)、(1,7)、(2,4)、(3,1)",
      line(2) - line(1) == 2 and line(4) - line(1) == 6 and down_line(2) - down_line(1) == -3
      and [down_line(x) for x in (0, 1, 2, 3)] == [10, 7, 4, 1])

# ---------------------------------------------------------------------------
banner("4. 导数：从一个位置出发走一小步 h，看平均变化率；把 h 越缩越小")
# “改一改”第 1 条改这一行：换一个位置量坡度。
X0 = 3.0  # TWEAK-1: 5.0
HS = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
print(f"  手算 h = 1：  f({X0 + 1:g}) = {f(X0 + 1):g}，涨了 {f(X0 + 1) - f(X0):g}，除以 1，平均变化率 {(f(X0 + 1) - f(X0)) / 1:g}")
print(f"  手算 h = 0.1：f({X0 + 0.1:g}) = {f(X0 + 0.1):.2f}，涨了 {f(X0 + 0.1) - f(X0):.2f}，除以 0.1，平均变化率 {(f(X0 + 0.1) - f(X0)) / 0.1:.1f}")
rows4 = shrink_h(f, X0, HS)
rule_at_x0 = 2 * X0                     # 1.6 节的规则：(x²)′ = 2x
print()
table(["h", f"f({X0:g}+h)", f"(f({X0:g}+h) − f({X0:g})) / h", f"与 {rule_at_x0:g} 的差"],
      [[num(h, 5), num(v, 10), num(rate, 5), num(rate - rule_at_x0, 5)] for h, v, rate in rows4])
check("手算 h = 1：f(4) = 16，涨 7，平均变化率 7", rows4[0][1] == 16 and rows4[0][2] == 7)
check("手算 h = 0.1：f(3.1) = 9.61，涨 0.61，平均变化率 6.1", round(rows4[1][1], 2) == 9.61 and round(rows4[1][2], 6) == 6.1)
check("平均变化率依次是 7、6.1、6.01、6.001、6.0001、6.00001：与 6 的差恰好等于 h",
      [round(r[2], 5) for r in rows4] == [7, 6.1, 6.01, 6.001, 6.0001, 6.00001]
      and all(math.isclose(r[2] - 6, r[0], rel_tol=1e-4) for r in rows4))
left = shrink_h(f, X0, [-1, -0.1, -0.01])
print("  从左边缩（h 取负数，往回走）：" + "、".join(num(r[2], 4) for r in left) + " → 也靠近同一个数")
check("从左边缩：5、5.9、5.99，同样靠近 6", [round(r[2], 4) for r in left] == [5, 5.9, 5.99])
print("  导数本身也是一台机器：f′(x) = 2x →", "，".join(f"f′({x:g}) = {2 * x:g}" for x in (3, 1, 0)))
check("f′(3) = 6，f′(1) = 2，f′(0) = 0（缩 h 逐个核对）",
      all(abs(shrink_h(f, x, [1e-6])[0][2] - 2 * x) < 1e-4 for x in (3, 1, 0)) and (2 * 3, 2 * 1, 2 * 0) == (6, 2, 0))
dy_rows = [[num(h, 3), num(f(3 + h) - f(3), 6), num((f(3 + h) - f(3)) / h, 3)] for h in (0.1, 0.001)]
print("  dy/dx 的数字落脚点（Δy ÷ Δx）：" + "；".join(f"Δx = {r[0]} 时 Δy = {r[1]}，比值 {r[2]}" for r in dy_rows))
check("Δx = 0.1 时 Δy = 0.61、比值 6.1；Δx = 0.001 时 Δy = 0.006001、比值 6.001",
      dy_rows == [["0.1", "0.61", "6.1"], ["0.001", "0.006001", "6.001"]])
check("进阶折叠块：(3 + h)² − 9 = 6h + h²，除以 h 恰好是 6 + h（所以表的最后一列就是 h）",
      all(math.isclose((3 + h) ** 2 - 9, 6 * h + h * h) and math.isclose(((3 + h) ** 2 - 9) / h, 6 + h) for h in HS))
secant_slopes = [(f(3 + h) - f(3)) / h for h in (2.0, 1.0, 0.5)]
print("  图里的三条割线：h = 2、1、0.5 时斜率", "、".join(f"{s:g}" for s in secant_slopes),
      f"；放大图：x = 3.1 处曲线高 {f(3.1):.2f}，切线高 9 + 6 × 0.1 = {9 + 6 * 0.1:.2f}")
check("图 ① 的灰字：x = 3.5 处连线高 9 + 7 × 0.5 = 12.5，曲线高 3.5² = 12.25", 9 + 7 * 0.5 == 12.5 and f(3.5) == 12.25)
check("图 ② 的割线斜率 8、7、6.5；图 ③ 里 3.1 处曲线高 9.61、切线高 9.60", secant_slopes == [8, 7, 6.5]
      and f"{f(3.1):.2f}" == "9.61" and f"{9 + 6 * 0.1:.2f}" == "9.60")
s_rate = shrink_h(lambda t: t**2, 3.0, [1e-6])[0][2]
check("自测：s(t) = t²，第 3 秒的瞬时速度 = 2 × 3 = 6 米/秒（缩 h 量到的也是 6）", abs(s_rate - 6) < 1e-4 and 2 * 3 == 6)

# ---------------------------------------------------------------------------
banner("4b. 画图：figures/ch01_secant_to_tangent.png（两点连线 → 缩 h → 切线）")
if X0 != 3.0:
    print("  位置改过了：这张讲解图是照着 x = 3 画的，跳过。看上面那张表就行。")
else:
    fig = plt.figure(figsize=(9.4, 18.6))
    gs = fig.add_gridspec(3, 1, hspace=0.58, left=0.12, right=0.95, top=0.925, bottom=0.075)
    fig.suptitle("导数：把 h 越缩越小，两点连线变成切线，斜率靠近 6", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
    xs = np.linspace(1.4, 5.3, 300)

    ax = fig.add_subplot(gs[0])
    data_axes(ax, "输入 x", "输出 f(x) = x²")
    ax.plot(xs, xs**2, color=BLUE, lw=3.5, zorder=3)
    ax.plot(xs, 9 + 7 * (xs - 3), color=MUTED, lw=2.2, ls="--", zorder=2)
    ax.plot([3, 4], [9, 9], color=GREEN, lw=3.5, zorder=4)
    ax.plot([4, 4], [9, 16], color=ORANGE, lw=3.5, zorder=4)
    ax.plot([3, 4], [9, 16], "o", color=INK, ms=10, zorder=5)
    ax.text(3.5, 8.2, "横着走 h = 1", color=GREEN, fontsize=FS_SMALL, ha="center", va="top")
    ax.text(4.08, 12.0, "竖着涨 16 − 9 = 7", color=ORANGE, fontsize=FS_SMALL, ha="left", va="center")
    ax.text(2.92, 9.8, "(3, 9)", color=INK, fontsize=FS_SMALL, ha="right", va="bottom")
    ax.text(3.92, 16.8, "(4, 16)", color=INK, fontsize=FS_SMALL, ha="right", va="bottom")
    ax.text(1.5, 24.0, "这条连线的斜率 = 7 ÷ 1 = 7", color=INK, fontsize=FS_STEP, va="center")
    ax.set_xlim(1.4, 5.3)
    ax.set_ylim(0, 27)
    ax.set_xticks([2, 3, 4, 5])
    panel_title(fig, [ax], "① 两点连线的斜率 = 竖着涨了多少 ÷ 横着走了多少")
    panel_note(fig, [ax], "蓝线是 f(x) = x²，虚线是过 (3, 9) 和 (4, 16) 的连线。从 3 走到 4，平均变化率是 7。\n3 到 4 之间，连线略高于曲线：x = 3.5 处连线高 12.5、曲线高 12.25。")

    ax = fig.add_subplot(gs[1])
    data_axes(ax, "输入 x", "输出 f(x) = x²")
    ax.plot(xs, xs**2, color=BLUE, lw=3.5, zorder=3)
    SECANTS = [(2.0, MUTED), (1.0, GREEN), (0.5, INK)]      # 只用 _draw.py 的配色；三条割线按斜率 8、7、6.5 从上到下排
    for k, (h, color) in enumerate(SECANTS):
        slope = (f(3 + h) - 9) / h
        ax.plot(xs, 9 + slope * (xs - 3), color=color, lw=2.2, ls="--", zorder=2)
        ax.plot(3 + h, f(3 + h), "o", color=color, ms=9, zorder=5)
        ax.text(1.5, 25.0 - 2.6 * k, f"h = {h:g}：连线的斜率 {slope:g}", color=color, fontsize=FS_SMALL, va="center")
    ax.plot(xs, 9 + 6 * (xs - 3), color=ORANGE, lw=3.2, zorder=4)
    ax.text(1.5, 25.0 - 2.6 * 3, "h 缩没了：切线，斜率 6", color=ORANGE, fontsize=FS_SMALL, va="center", fontweight="bold")
    ax.plot(3, 9, "o", color=INK, ms=10, zorder=6)
    ax.text(3.12, 7.6, "(3, 9)", color=INK, fontsize=FS_SMALL, ha="left", va="top")
    ax.set_xlim(1.4, 5.3)
    ax.set_ylim(0, 27)
    ax.set_xticks([2, 3, 4, 5])
    panel_title(fig, [ax], "② 把 h 缩短：连线绕着 (3, 9) 转，越来越贴着曲线")
    panel_note(fig, [ax], "虚线叫割线（割开曲线的两点连线），圆点是它的另一头。\nh 越小，割线越贴近橙线——只在 (3, 9) 擦着曲线的切线。它的斜率 6 就是导数。")

    ax = fig.add_subplot(gs[2])
    data_axes(ax, "输入 x（放大：只看 2.9 到 3.1）", "输出 f(x)")
    zoom = np.linspace(2.88, 3.12, 100)
    ax.plot(zoom, zoom**2, color=BLUE, lw=3.5, zorder=3)
    ax.plot(zoom, 9 + 6 * (zoom - 3), color=ORANGE, lw=2.4, ls="--", zorder=4)
    ax.plot(3, 9, "o", color=INK, ms=10, zorder=6)
    ax.plot(3.1, f(3.1), "o", color=BLUE, ms=9, zorder=6)
    ax.plot(3.1, 9 + 6 * 0.1, "o", color=ORANGE, ms=9, zorder=6)
    ax.annotate(f"曲线上：f(3.1) = {f(3.1):.2f}", xy=(3.1, f(3.1)), xytext=(2.89, 9.62), fontsize=FS_SMALL, color=BLUE, va="center",
                arrowprops=dict(arrowstyle="-", lw=1.3, color=BLUE, shrinkB=6))
    ax.annotate("切线上：9 + 6 × 0.1 = 9.60", xy=(3.1, 9.6), xytext=(2.97, 8.55), fontsize=FS_SMALL, color=ORANGE, va="center",
                arrowprops=dict(arrowstyle="-", lw=1.3, color=ORANGE, shrinkB=6))
    ax.set_xlim(2.88, 3.12)
    ax.set_ylim(8.2, 9.85)
    ax.set_xticks([2.9, 2.95, 3.0, 3.05, 3.1])
    panel_title(fig, [ax], "③ 放大看：在 x = 3 附近，曲线和切线几乎分不开")
    panel_note(fig, [ax], "所以走一小步时，可以拿切线当曲线用；走远了（回看 ②），两者就分开了。")
    savefig(fig, "ch01_secant_to_tangent")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("5. 导数的正负号：往右走是上坡、下坡，还是平的")
print("  在 x = −2 处缩 h（往右走一小步）：")
rows5 = shrink_h(f, -2.0, [0.1, 0.01, 0.001])
table(["h", "f(−2+h)", "(f(−2+h) − 4) / h"], [[num(h, 3), num(v, 6), num(rate, 3)] for h, v, rate in rows5])
check("x = −2：f(−1.9) = 3.61，变化率 −3.9、−3.99、−3.999，靠近 −4 = 2 × (−2)",
      round(rows5[0][1], 2) == 3.61 and [round(r[2], 3) for r in rows5] == [-3.9, -3.99, -3.999] and 2 * -2 == -4)
rows5_zero = shrink_h(f, 0.0, [0.1, 0.01])
print("  在 x = 0 处缩 h：" + "、".join(num(r[2], 3) for r in rows5_zero) + " → 靠近 0（平的）")
check("x = 0：变化率 0.1、0.01，靠近 0", [round(r[2], 3) for r in rows5_zero] == [0.1, 0.01])
SIGN_POINTS = [(-2.0, "往右走是下坡"), (0.0, "平的"), (3.0, "往右走是上坡")]
table(["位置 x", "导数 f′(x) = 2x", "这里的坡"], [[num(x), num(2 * x), text] for x, text in SIGN_POINTS])
check("三个位置的导数：−4、0、6", [2 * x for x, _ in SIGN_POINTS] == [-4, 0, 6])

print("\n  导数的大小 = 敏感程度：同样多拧 0.01，")
for x in (3.0, 1.0):
    print(f"    在 x = {x:g}（导数 {2 * x:g}）：f 实际变了 {f(x + 0.01) - f(x):.4f}，约等于 {2 * x:g} × 0.01 = {2 * x * 0.01:g}")
check("x = 3 处多拧 0.01，f 变 0.0601 ≈ 0.06；x = 1 处变 0.0201 ≈ 0.02",
      round(f(3.01) - f(3), 4) == 0.0601 and round(f(1.01) - f(1), 4) == 0.0201)

print("\n  Δf ≈ f′(x) × Δx 只在小步时准——导数只是当前位置的坡度：")
rows = [[num(dx), num(6 * dx), num(f(3 + dx) - f(3), 4), num(f(3 + dx) - f(3) - 6 * dx, 4)] for dx in (0.01, 0.1, 1.0)]
table(["从 3 走 Δx", "预测 6 × Δx", "实际 f(3+Δx) − 9", "差多少"], rows)
check("走 0.01：预测 0.06、实际 0.0601；走 0.1：0.6 对 0.61；走 1：6 对 7",
      rows == [["0.01", "0.06", "0.0601", "0.0001"], ["0.1", "0.6", "0.61", "0.01"], ["1", "6", "7", "1"]])
check("走到 x = 4 时坡度已经从 6 变成 8", 2 * 4 == 8)
check("自测：x = −1 处导数 −2；把 x 调大到 −0.9，f 从 1 降到 0.81", abs(shrink_h(f, -1.0, [1e-6])[0][2] + 2) < 1e-4
      and f(-1) == 1 and round(f(-0.9), 2) == 0.81)

# ---------------------------------------------------------------------------
banner("5b. 画图：figures/ch01_derivative_sign.png（正负号 = 上坡还是下坡；想变小就反着调）")
fig = plt.figure(figsize=(9.4, 15.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.82], hspace=0.40, left=0.12, right=0.95, top=0.915, bottom=0.085)
fig.suptitle("导数的正负号：往右走是上坡、下坡，还是平的", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
sx = np.linspace(-3.6, 4.1, 300)

ax = fig.add_subplot(gs[0])
data_axes(ax, "输入 x", "输出 f(x) = x²")
ax.plot(sx, sx**2, color=BLUE, lw=3.5, zorder=3)
for (x, _), text, dx, dy, ha in zip(SIGN_POINTS, ["导数 −4\n往右走是下坡", "导数 0：平的", "导数 6\n往右走是上坡"],
                                    [0.35, 0.0, -0.35], [3.2, -1.6, 3.6], ["left", "center", "right"]):
    span = np.array([x - 0.9, x + 0.9])
    ax.plot(span, f(x) + 2 * x * (span - x), color=ORANGE, lw=3.2, zorder=4)
    ax.plot(x, f(x), "o", color=INK, ms=10, zorder=5)
    ax.text(x + dx, f(x) + dy, text, color=ORANGE, fontsize=FS_SMALL, ha=ha, va="center", linespacing=1.3, bbox=WHITE_BOX, zorder=6)
ax.set_xlim(-3.7, 4.2)
ax.set_ylim(-3.2, 17.5)
ax.set_xticks(range(-3, 5))
ax.set_yticks(range(0, 17, 4))
panel_title(fig, [ax], "① 同一条曲线，三个位置，三种坡")
panel_note(fig, [ax], "橙色短线是各处的切线。导数就是它的斜率：负 = 下坡，0 = 平，正 = 上坡；\n数字越大（不看正负号）坡越陡：x = 3 处的 6 比 x = −2 处的 4 陡。")

ax = fig.add_subplot(gs[1])
data_axes(ax, "输入 x", "输出 f(x) = x²")
ax.plot(sx, sx**2, color=FAINT, lw=3, zorder=2)
arrow(ax, (-2.0, 4.0), (-1.1, 4.0), GREEN, lw=3.5, zorder=5)
arrow(ax, (3.0, 9.0), (2.1, 9.0), GREEN, lw=3.5, zorder=5)
for x in (-2.0, 0.0, 3.0):
    ax.plot(x, f(x), "o", color=INK, ms=10, zorder=6)
ax.text(-1.75, 5.0, "导数 −4（负）\n→ 把 x 调大", color=GREEN, fontsize=FS_SMALL, ha="left", va="bottom", linespacing=1.3, bbox=WHITE_BOX)
ax.text(2.75, 10.0, "导数 6（正）\n→ 把 x 调小", color=GREEN, fontsize=FS_SMALL, ha="right", va="bottom", linespacing=1.3, bbox=WHITE_BOX)
ax.text(0.0, -0.45, "导数 0：到底了，不用动", color=INK, fontsize=FS_SMALL, ha="center", va="top")      # 写在曲线下面，不盖住曲线
ax.set_xlim(-3.7, 4.2)
ax.set_ylim(-2.4, 17.5)
ax.set_xticks(range(-3, 5))
ax.set_yticks(range(0, 17, 4))
panel_title(fig, [ax], "② 想让 f 变小：朝导数正负号的反方向调")
panel_note(fig, [ax], "绿箭头两边都指向最低点。想让 f 变大就反过来：顺着导数的正负号调。")
savefig(fig, "ch01_derivative_sign")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("6. 求导规则：每一条都用缩 h 抽查一遍")
RULE_HS = [0.1, 0.01, 0.001]


def rule_table(label, func, at):
    rows = shrink_h(func, at, RULE_HS)
    table(["h", f"{label} 在 x = {at:g}+h 处的值", "平均变化率"], [[num(h, 3), num(v, 9), num(rate, 6)] for h, v, rate in rows])
    return rows


print("规则一：常数的导数是 0。f(x) = 5，喂什么都吐 5：", [(5 - 5) / h for h in RULE_HS])
check("(5)′ = 0", all((5 - 5) / h == 0 for h in RULE_HS))

print("\n规则二（幂）：(xⁿ)′ = n·xⁿ⁻¹。x³ 在 x = 2 处，规则说 3 × 2² = 12：")
rows_cube = rule_table("x³", lambda x: x**3, 2.0)
check("x³ 在 x = 2：2.1³ = 9.261 → 12.61；2.01³ = 8.120601 → 12.0601；再缩 → 12.006，靠近 12",
      round(rows_cube[0][1], 3) == 9.261 and round(rows_cube[0][2], 2) == 12.61 and round(rows_cube[1][1], 6) == 8.120601
      and round(rows_cube[1][2], 4) == 12.0601 and round(rows_cube[2][2], 3) == 12.006 and 3 * 2**2 == 12)
check("x 自己（直线 y = x）斜率是 1，套幂规则也是 (x¹)′ = 1 × x⁰ = 1；x² 在 3、1、−2 处缩 h 量到的 6、2、−4 都是“2 × 位置”",
      (lambda x: x)(5) - (lambda x: x)(4) == 1 and 1 * 7.0**0 == 1 and [2 * x for x in (3, 1, -2)] == [6, 2, -4])

print("\n规则三（常数倍照搬）：(5x²)′ = 5 × 2x = 10x。在 x = 3 处，规则说 30：")
rows_5x2 = rule_table("5x²", lambda x: 5 * x**2, 3.0)
check("5x² 在 x = 3：5 × 9.61 = 48.05 → 30.5；再缩 → 30.05、30.005，靠近 30",
      round(rows_5x2[0][1], 2) == 48.05 and [round(r[2], 3) for r in rows_5x2] == [30.5, 30.05, 30.005] and 10 * 3 == 30)

print("\n规则四（加法可以分开求导）：(x² + x³)′ = 2x + 3x²。在 x = 2 处，规则说 4 + 12 = 16：")
rows_sum = rule_table("x² + x³", lambda x: x**2 + x**3, 2.0)
check("x² + x³ 在 x = 2：4.0401 + 8.120601 = 12.160701 → 16.0701；再缩 → 16.007，靠近 16",
      round(rows_sum[1][1], 6) == 12.160701 and round(rows_sum[1][2], 4) == 16.0701 and round(rows_sum[2][2], 3) == 16.007
      and 2 * 2 + 3 * 2**2 == 16)

print("\n用规则回头算 1.3 节的直线：(2x + 1)′ = 2 + 0 = 2，正是它的斜率。")
check("(2x + 1)′ = 2 = 直线的斜率", shrink_h(line, 1.0, [0.5])[0][2] == 2)
print("⚠ 乘法不能分开求：x² = x · x，两个 x 各自的导数都是 1，1 × 1 = 1，可 (x²)′ 在 x = 3 处是 6，不是 1。")
check("相乘不能把导数直接相乘：1 × 1 = 1 ≠ 6", 1 * 1 != 2 * 3)
check("自测：(3x² + 7)′ = 6x，在 x = 2 处是 12", abs(shrink_h(lambda x: 3 * x**2 + 7, 2.0, [1e-6])[0][2] - 12) < 1e-4 and 6 * 2 == 12)

# ---------------------------------------------------------------------------
banner("7. 链式法则：机器套机器，总倍数 = 各台倍数相乘")
print(f"  汇率：1 元 → 0.14 美元；1 美元 → 0.9 欧元；所以 1 元 → 0.14 × 0.9 = {0.14 * 0.9:g} 欧元")
check("0.14 × 0.9 = 0.126", math.isclose(0.14 * 0.9, 0.126))

# “改一改”第 2 条改这一行：第一台机器从 2x + 1 换成 5x + 1。
G_SLOPE = 2.0  # TWEAK-2: 5.0


def g(x):
    return G_SLOPE * x + 1          # 第一台（内层）


def composed(x):
    return f(g(x))                  # 串起来：先 g 再 f；前端写法是 const pipeline = (x) => f(g(x))


CHAIN_X = 1.5
u0 = g(CHAIN_X)
inner_rate, outer_rate = G_SLOPE, 2 * u0
chain_rate = outer_rate * inner_rate
print(f"  手算 x = {CHAIN_X:g}：① u = g({CHAIN_X:g}) = {u0:g}  ② 内层倍数 g′ = {inner_rate:g}  ③ 外层倍数 f′(u) = 2u = {outer_rate:g}"
      f"  ④ 总倍数 = {outer_rate:g} × {inner_rate:g} = {chain_rate:g}；输出 y = {composed(CHAIN_X):g}")
check("x = 1.5：u = 4，y = 16；内层倍数 2，外层倍数 8，总倍数 16",
      u0 == 4 and composed(CHAIN_X) == 16 and inner_rate == 2 and outer_rate == 8 and chain_rate == 16)
rows7 = [[h, g(CHAIN_X + h), composed(CHAIN_X + h), (composed(CHAIN_X + h) - composed(CHAIN_X)) / h] for h in (0.1, 0.01, 0.001)]
print("  不信规则，缩 h 量一遍：")
table(["h", "u = g(1.5+h)", "y = u²", f"(y − {composed(CHAIN_X):g}) / h"], [[num(h, 3), num(a, 4), num(b, 6), num(c, 3)] for h, a, b, c in rows7])
check("缩 h：u = 4.2 → y = 17.64 → 16.4；u = 4.02 → y = 16.1604 → 16.04；再缩 → 16.004，靠近 16",
      [round(r[1], 3) for r in rows7[:2]] == [4.2, 4.02] and [round(r[2], 4) for r in rows7[:2]] == [17.64, 16.1604]
      and [round(r[3], 3) for r in rows7] == [16.4, 16.04, 16.004])
numeric7 = (composed(CHAIN_X + 1e-6) - composed(CHAIN_X)) / 1e-6
check("链式法则 = 缩 h 的数值结果", abs(numeric7 - chain_rate) < 1e-3)
wrong = 2 * CHAIN_X * inner_rate
print(f"  ⚠ 常见的错：外层的倍数要在“它自己的输入 u = {u0:g}”处算（2 × {u0:g} = {outer_rate:g}），不是在 x = {CHAIN_X:g} 处算（2 × {CHAIN_X:g} = {2 * CHAIN_X:g}）。"
      f"错的算法得 {wrong:g}。")
check("外层倍数用错位置会得 6，不是 16", wrong == 6)
a = G_SLOPE                          # (ax + 1)² = a²x² + 2ax + 1，用 1.6 节的规则求导得 2a²x + 2a
expanded = 2 * a * a * CHAIN_X + 2 * a
print(f"  另一条路：先展开 ({a:g}x + 1)² = {a * a:g}x² + {2 * a:g}x + 1，再用 1.6 节的规则求导得 {2 * a * a:g}x + {2 * a:g}，"
      f"在 x = {CHAIN_X:g} 处是 {expanded:g}。两条路同一个数。")
check("展开再求导：(2x + 1)(2x + 1) = 4x² + 2x + 2x + 1 = 4x² + 4x + 1，导数 8x + 4，在 1.5 处 8 × 1.5 + 4 = 16",
      (a * a, 2 * a, 2 * a * a) == (4, 4, 8) and expanded == 16 and math.isclose(composed(0.7), 4 * 0.7**2 + 4 * 0.7 + 1))
check("展开的那条路和链式法则给的是同一个数", math.isclose(expanded, chain_rate))
check("自测：y = (3x − 2)²，在 x = 2 处 u = 4，外层 8，内层 3，导数 24",
      abs(shrink_h(lambda x: (3 * x - 2) ** 2, 2.0, [1e-6])[0][2] - 24) < 1e-3 and 2 * (3 * 2 - 2) * 3 == 24)

# ---------------------------------------------------------------------------
banner("7b. 画图：figures/ch01_chain_rule.png（汇率相乘 → 两台机器 → 倍数相乘）")
if G_SLOPE != 2.0:
    print("  第一台机器改过了：这张讲解图是照着 2x + 1 画的，跳过。看上面的手算和表就行。")
else:
    fig, axes = lesson_figure(3, "链式法则：机器串机器，总倍数 = 各台倍数相乘", panel_height=3.9, width=9.6)
    ax = axes[0]
    lesson_panel(ax, "① 先看汇率：换两次钱", ymax=4.5)
    pipeline(ax, 2.1, ["1 元", "× 0.14", "0.14 美元", "× 0.9", "0.126 欧元"], x=0.2, value_w=1.98, machine_w=1.2, gap=0.34)
    hand(ax, 0.3, 1.35, "1 元直接换欧元：0.14 × 0.9 = 0.126")
    note(ax, 0.3, 0.65, "两次兑换串起来，总汇率 = 两个汇率相乘。")
    ax = axes[1]
    lesson_panel(ax, "② 两台机器串起来：先过 g，输出直接喂给 f", ymax=4.5)
    centers = pipeline(ax, 1.8, ["1.5", "g：乘 2 加 1", "4", "f：平方", "16"], value_w=1.2, machine_w=2.2, gap=0.42)
    for cx, name in zip(centers, ["输入 x", "中间的数 u", "输出 y"]):
        ax.text(cx, 3.0, name, ha="center", fontsize=FS_SMALL, color=MUTED)
    hand(ax, 0.3, 1.1, "u = 2 × 1.5 + 1 = 4，  y = 4² = 16")
    note(ax, 0.3, 0.45, "套在一起的函数叫复合函数：g 在里面（内层），f 在外面（外层）。")
    ax = axes[2]
    lesson_panel(ax, "③ x 动一点，y 动多少？每台各算一个倍数，再相乘", ymax=4.5)
    hand(ax, 0.3, 3.15, "内层 g：x 动一点，u 动它的 2 倍")
    hand(ax, 7.0, 3.15, "→ 倍数 2")
    hand(ax, 0.3, 2.45, "外层 f：u 动一点，y 动它的 2u = 8 倍", color=GREEN)
    hand(ax, 7.0, 2.45, "→ 倍数 8", color=GREEN)
    hand(ax, 0.3, 1.65, f"总倍数 = 8 × 2 = {chain_rate:g}", color=ORANGE)
    hand(ax, 4.2, 1.65, f"（缩 h 核对：{rows7[1][3]:.2f} → {rows7[2][3]:.3f} → 16）", color=MUTED, fontsize=FS_NOTE)
    note(ax, 0.3, 0.65, "注意外层的倍数在“它自己的输入 u = 4”处算，不是在 x = 1.5 处算。")
    savefig(fig, "ch01_chain_rule")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("8. 指数函数：每走一步乘同一个数；e 是“导数恰好等于自己”的那个底")
STEPS = [1, 2, 3, 4, 5, 10]
table(["x", *[str(x) for x in STEPS]], [["x²（每步加得越来越多）", *[x**2 for x in STEPS]], ["2ˣ（每步翻一倍）", *[2**x for x in STEPS]]])
check("2ˣ：2、4、8、16、32，到 x = 10 是 1024；x² 到 x = 10 才 100", [2**x for x in STEPS] == [2, 4, 8, 16, 32, 1024] and 10**2 == 100)
print("  往回走一步就除以 2：2⁰ = 1，2⁻¹ =", 2.0**-1, "，2⁻² =", 2.0**-2, "；半步：2^0.5 =", round(2**0.5, 3), "（连乘两次正好是 2）")
check("2⁰ = 1，2⁻¹ = 0.5，2⁻² = 0.25，2^0.5 ≈ 1.414", 2**0 == 1 and 2.0**-1 == 0.5 and 2.0**-2 == 0.25 and round(2**0.5, 3) == 1.414)

print("\n  2ˣ 在几个位置的变化率（缩 h，h = 0.000001），和 2ˣ 自己比：")
h = 1e-6
rows8 = [[x, 2.0**x, (2.0 ** (x + h) - 2.0**x) / h, (2.0 ** (x + h) - 2.0**x) / h / 2.0**x] for x in (0, 1, 2, 3)]
table(["x", "2ˣ", "变化率", "变化率 ÷ 2ˣ"], rows8, floatfmt=".3f")
check("2ˣ 的变化率总是它自己的 0.693 倍：0.693、1.386、2.773、5.545",
      [round(r[2], 3) for r in rows8] == [0.693, 1.386, 2.773, 5.545] and all(round(r[3], 3) == 0.693 for r in rows8))
console_rows = [[hh, 2.0**hh, (2.0**hh - 1) / hh] for hh in (1, 0.5, 0.1, 0.01)]
print("\n  读者自己缩一遍（浏览器控制台里敲 (2 ** 0.1 - 1) / 0.1 这样的式子）：在 x = 0 处，2⁰ = 1，")
table(["h", "2ʰ", "(2ʰ − 1) / h"], [[num(a), f"{b:.4f}", f"{c:.3f}"] for a, b, c in console_rows])
check("x = 0 处缩 h：h = 1 → 1，h = 0.5 → 0.828，h = 0.1 → 0.718，h = 0.01 → 0.696，靠近 0.693",
      [f"{r[2]:.3f}" for r in console_rows] == ["1.000", "0.828", "0.718", "0.696"])
check("字面值重算：2^0.5 ≈ 1.4142 → (1.4142 − 1) ÷ 0.5 = 0.828；2^0.1 ≈ 1.0718 → (1.0718 − 1) ÷ 0.1 = 0.718",
      f"{2**0.5:.4f}" == "1.4142" and round((1.4142 - 1) / 0.5, 3) == 0.828 and f"{2**0.1:.4f}" == "1.0718" and round((1.0718 - 1) / 0.1, 3) == 0.718)
print("\n  换一个底，这个倍数就换一个数：")
BASES = [2.0, 2.5, 2.7, math.e, 2.72, 3.0]
factors = [(a**h - 1) / h for a in BASES]
table(["底 a", "aˣ 的变化率 ÷ aˣ"], [[num(a, 3), fac] for a, fac in zip(BASES, factors)], floatfmt=".3f")
check("底 2 → 0.693，2.5 → 0.916，2.7 → 0.993，2.718 → 1.000，2.72 → 1.001，3 → 1.099：倍数恰好是 1 的底夹在 2.7 和 2.72 之间",
      [f"{v:.3f}" for v in factors] == ["0.693", "0.916", "0.993", "1.000", "1.001", "1.099"])
check("e ≈ 2.718", round(math.e, 3) == 2.718)

print("\n  所以 eˣ 的导数就是 eˣ 自己。缩 h（h = 0.001）核对：")
rows8e = [[x, math.exp(x), (math.exp(x + 0.001) - math.exp(x)) / 0.001] for x in (-1.0, 0.0, 1.0, 2.0)]
table(["x", "eˣ", "缩 h 量到的变化率"], rows8e, floatfmt=".4f")
print("  四个位置“量到的变化率 ÷ eˣ”：" + "、".join(f"{r[2] / r[1]:.4f}" for r in rows8e))
check("(eˣ)′ ≈ eˣ：h = 0.001 时，四个位置量到的变化率 ÷ eˣ 都是 1.0005", all(f"{r[2] / r[1]:.4f}" == "1.0005" for r in rows8e))
e_101, e_001 = math.exp(1.01), math.exp(0.01)
print(f"  读者手算版（h = 0.01）：e^0.01 = {e_001:.5f} → ({e_001:.5f} − 1) ÷ 0.01 = {(round(e_001, 5) - 1) / 0.01:.3f} ≈ e⁰ = 1；")
print(f"                      e^1.01 = {e_101:.5f} → ({e_101:.5f} − {math.e:.5f}) ÷ 0.01 = {(round(e_101, 5) - round(math.e, 5)) / 0.01:.3f} ≈ e¹ = {math.e:.3f}")
check("手算核对：e^0.01 = 1.01005 → 1.005；e^1.01 = 2.74560、e = 2.71828 → 2.732",
      round(e_001, 5) == 1.01005 and round((round(e_001, 5) - 1) / 0.01, 3) == 1.005
      and f"{e_101:.5f}" == "2.74560" and round(math.e, 5) == 2.71828 and round((round(e_101, 5) - round(math.e, 5)) / 0.01, 3) == 2.732)

print("\n  项目里主要碰到的是衰减版 e⁻ˣ（1.9 节马上要查这张表）：")
DECAY_X = [0, 0.1, 0.5, 0.9, 1, 2, 5, 10]
decay = {x: math.exp(-x) for x in DECAY_X}
table(["x", "e⁻ˣ"], [[num(x), f"{decay[x]:.6f}"] for x in DECAY_X])      # 固定 6 位小数：0.406570 后面几节还要用，全章写法一致
check("e⁻ˣ：0 → 1；0.1 → 0.904837；0.5 → 0.606531；0.9 → 0.406570；1 → 0.367879；2 → 0.135335；5 → 0.006738；10 → 0.000045",
      [f"{decay[x]:.6f}" for x in DECAY_X] == ["1.000000", "0.904837", "0.606531", "0.406570", "0.367879", "0.135335", "0.006738", "0.000045"])
check("从 1 开始一路变小，永远大于 0", all(decay[a] > decay[b] > 0 for a, b in zip(DECAY_X, DECAY_X[1:])) and decay[0] == 1)
check("x 每加 1，输出乘 1/e ≈ 0.37：e⁻¹ = 0.37，e⁻² ÷ e⁻¹ 也是 0.37", round(1 / math.e, 2) == 0.37 and round(decay[2] / decay[1], 2) == 0.37)
print(f"  ln 是 exp 的逆运算：“e 的几次方等于 0.367879？” ln(0.367879) = {math.log(0.367879):.4f}")
check("ln(0.367879) ≈ −1", round(math.log(0.367879), 4) == -1.0)
print("  折叠块：连续复利 (1 + 1/n)ⁿ →", "、".join(f"n = {n}: {(1 + 1 / n) ** n:.4f}" for n in (1, 2, 12, 365)), f"→ e = {math.e:.5f}")
check("复利：2、2.25、2.6130、2.7146 → 2.71828", [round((1 + 1 / n) ** n, 4) for n in (1, 2, 12, 365)] == [2, 2.25, 2.613, 2.7146])
check("折叠块：2^(x+h) = 2ˣ × 2ʰ，所以变化率 = 2ˣ × (2ʰ − 1)/h；h = 0.001 时括号是 0.6934",
      math.isclose(2.0**3.001, 2.0**3 * 2.0**0.001) and round((2**0.001 - 1) / 0.001, 4) == 0.6934)
check("自测：e⁻ˣ 的导数 = −e⁻ˣ（外层 eᵘ，内层 u = −x 的倍数 −1）；在 x = 1 处是 −0.368",
      abs(shrink_h(lambda x: math.exp(-x), 1.0, [1e-6])[0][2] + math.exp(-1)) < 1e-4 and round(-math.exp(-1), 3) == -0.368)

# ---------------------------------------------------------------------------
banner("9. 钟形打分：三台小机器串起来——平方 → 除以 σ² 取负 → exp")
# 线速度这一项的 σ²（项目配置里写的是 std = math.sqrt(0.1)）。“改一改”第 3 条改这一行。
SIGMA2 = 0.1  # TWEAK-3: 1.0
SIGMA = math.sqrt(SIGMA2)


def score(dev, s2=None):
    """钟形打分：exp(−偏差² / σ²)。"""
    return math.exp(-(dev**2) / (SIGMA2 if s2 is None else s2))


HAND_DEVS = [0.0, 0.1, -0.1, 0.3, 1.0]
print(f"  σ² = {SIGMA2:g}。偏差 = 指令 − 实际（单位 m/s），可正可负：")
table(["偏差", "① 平方", f"② 除以 {SIGMA2:g}、取负", "③ exp → 分数"],
      [[num(d), num(d * d), num(-d * d / SIGMA2), f"{score(d):.6f}"] for d in HAND_DEVS])      # 固定 6 位小数：和 1.8 节那张表、1.12 节起的写法一致
check("偏差 0 → 1 分；0.1 → 0.905；0.3 → 0.407；1 → 0.000045（几乎 0）",
      score(0) == 1 and round(score(0.1), 3) == 0.905 and round(score(0.3), 3) == 0.407 and round(score(1.0), 6) == 0.000045)
check("中间两步的数：0.1² = 0.01 → −0.1；0.3² = 0.09 → −0.9；1² = 1 → −10",
      [round(-d * d / SIGMA2, 6) for d in (0.1, 0.3, 1.0)] == [-0.1, -0.9, -10])
check("偏左偏右一样：偏差 −0.1 和 0.1 同分", score(-0.1) == score(0.1))
check("正文 JS 注释里的 score(0.3, 0.1) → 0.4066", round(score(0.3, 0.1), 4) == 0.4066)
check("分数永远在 0 和 1 之间，不会是负的", all(0 < score(d) <= 1 for d in np.linspace(-3, 3, 61)))
tail_gain = score(0.95) - score(1.0)
tail_remark = "离目标很远时，分数几乎不动" if tail_gain < 0.001 else "钟这么宽的时候，偏差 1 还没走到平尾巴上"
print(f"  平尾巴：偏差 1 → {score(1.0):.6f}，偏差 0.95 → {score(0.95):.6f}，只差 {tail_gain:.6f}——{tail_remark}")
check("平尾巴：偏差 1 和 0.95 的分数是 0.000045 和 0.000120", round(score(1.0), 6) == 0.000045 and f"{score(0.95):.6f}" == "0.000120")
check("自测：偏差 0.2 → 0.04 ÷ 0.1 = 0.4 → e⁻⁰·⁴ = 0.670；偏差 −0.3 和 0.3 同为 0.407",
      round(math.exp(-0.4), 3) == 0.670 and round(score(0.2), 3) == 0.670 and round(score(-0.3), 3) == 0.407)
STAND_STILL_DEV = 0.4        # 前后速度指令最大 0.4 m/s（第 15 节对着配置核对）；一只站着不动的机器人，偏差就是 0.4
print(f"  “改一改”第 3 条要看的数：指令 {STAND_STILL_DEV:g} m/s、站着不动（偏差 {STAND_STILL_DEV:g}）→ {STAND_STILL_DEV**2 / SIGMA2:g} → 得 {score(STAND_STILL_DEV):.3f} 分")
check("σ² = 0.1 时，站着不动（偏差 0.4）只得 0.202 分", round(score(STAND_STILL_DEV), 3) == 0.202)

# ---------------------------------------------------------------------------
banner("9b. 画图：figures/ch01_bell_score.png（三台小机器 → 一口钟）与 figures/ch01_overview.png（1.0 节总览）")
if SIGMA2 != 0.1 or X0 != 3.0 or G_SLOPE != 2.0:
    print("  参数改过了：这两张讲解图是照着正文的数字画的，跳过。看上面的表就行。")
else:
    fig = plt.figure(figsize=(9.6, 13.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.7, 6.2], hspace=0.22, left=0.11, right=0.96, top=0.905, bottom=0.10)
    fig.suptitle("钟形打分：正中得 1 分，偏得越远分越低，永不为负", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.98)
    ax = fig.add_subplot(gs[0])
    lesson_panel(ax, xmax=10, ymax=3.4)
    centers = pipeline(ax, 1.55, ["0.3", "平方", "0.09", "÷ 0.1\n取负", "−0.9", "exp", f"{score(0.3):.3f}"],
                       x=0.1, value_w=1.12, machine_w=1.1, gap=0.3, height=1.0)
    for cx, name in zip(centers, ["偏差", "偏差²", "≤ 0 的数", "分数"]):
        ax.text(cx, 2.82, name, ha="center", fontsize=FS_SMALL, color=MUTED)
    note(ax, 0.1, 0.62, "平方：偏左偏右一样，越偏越大。除以 0.1 再取负：变成一个 ≤ 0 的数。\nexp：0 变成 1，越负越接近 0。", linespacing=1.4)
    panel_title(fig, [ax], "① 三台小机器串起来：偏差 0.3 m/s → 0.407 分", pad=-0.004)

    ax = fig.add_subplot(gs[1])
    data_axes(ax, "偏差 = 指令 − 实际（m/s）；负 = 偏向另一边", "分数")
    dev = np.linspace(-1.45, 1.45, 600)
    ax.plot(dev, np.exp(-(dev**2) / SIGMA2), color=BLUE, lw=3.5, zorder=3)
    for d in (0.0, 0.1, -0.1, 0.3, -0.3, 1.0, -1.0):
        ax.plot(d, score(d), "o", color=ORANGE, ms=10, zorder=5)
    ax.text(0.0, 1.06, "偏差 0：1 分", color=ORANGE, fontsize=FS_SMALL, ha="center", va="bottom")
    ax.annotate(f"偏差 ±0.1：{score(0.1):.3f}", xy=(0.1, score(0.1)), xytext=(0.42, 0.93), fontsize=FS_SMALL, color=ORANGE, va="center",
                arrowprops=dict(arrowstyle="-", lw=1.3, color=ORANGE, shrinkB=6))
    ax.annotate(f"偏差 ±0.3：{score(0.3):.3f}", xy=(0.3, score(0.3)), xytext=(0.55, 0.52), fontsize=FS_SMALL, color=ORANGE, va="center",
                arrowprops=dict(arrowstyle="-", lw=1.3, color=ORANGE, shrinkB=6))
    ax.plot([-0.3, 0.3], [score(0.3)] * 2, ls="--", lw=2, color=GREEN, zorder=2)
    ax.text(-0.42, score(0.3), "左右一样高", color=GREEN, fontsize=FS_SMALL, ha="right", va="center", bbox=WHITE_BOX)
    ax.text(0.98, 0.09, "偏差 ±1：几乎 0 分\n（平尾巴）", color=ORANGE, fontsize=FS_SMALL, ha="center", va="bottom", linespacing=1.3)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-0.06, 1.2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    panel_title(fig, [ax], "② 把手算的分数标到图上，连起来是一口钟")
    panel_note(fig, [ax], "橙点是正文手算的那几行。顶上平、中间陡，\n两边的尾巴又平下来：贴着 0，但永远不到 0。")
    savefig(fig, "ch01_bell_score")
    plt.close(fig)

    fig, axes = lesson_figure(4, "这一章的四样东西：机器、量坡度、串机器、打分", panel_height=3.3, width=9.6)
    ax = axes[0]
    lesson_panel(ax, "① 函数：一台机器，喂一个数，吐一个数（1.1 节）")
    pipeline(ax, 1.55, ["3", "f：平方", "9"], x=0.4, value_w=1.2, machine_w=2.0)
    hand(ax, 6.0, 2.02, "f(3) = 3 × 3 = 9")
    note(ax, 0.4, 0.6, "同样的输入，永远得到同样的输出。")
    ax = axes[1]
    lesson_panel(ax, "② 导数：输入拧一点点，输出动多少（1.4 节）")
    hand(ax, 0.4, 2.55, f"3 → 3.1：  输出 9 → {f(3.1):.2f}，  {f(3.1) - 9:.2f} ÷ 0.1 = {rows4[1][2]:.1f}")
    hand(ax, 0.4, 1.85, f"3 → 3.01：输出 9 → {f(3.01):.4f}，{f(3.01) - 9:.4f} ÷ 0.01 = {rows4[2][2]:.2f}", color=GREEN)
    note(ax, 0.4, 0.85, "步子越缩越小，这个比值靠近 6。这个 6 就是 f 在 x = 3 处的导数。")
    ax = axes[2]
    lesson_panel(ax, "③ 链式法则：机器串机器，倍数相乘（1.7 节）")
    pipeline(ax, 1.55, ["1.5", "乘 2 加 1", "4", "平方", "16"], x=0.4, value_w=1.0, machine_w=1.75, gap=0.36)
    hand(ax, 0.4, 0.95, f"第一台的倍数 2 × 第二台的倍数 8 = 总倍数 {chain_rate:g}")
    ax = axes[3]
    lesson_panel(ax, "④ 钟形打分：项目给“偏了多少”打分的机器（1.9 节）")
    pipeline(ax, 1.55, ["0.3", "平方", "0.09", "÷ 0.1\n取负", "−0.9", "exp", f"{score(0.3):.3f}"],
             x=0.1, value_w=1.12, machine_w=1.1, gap=0.3, height=1.0)
    note(ax, 0.1, 0.75, "偏差 0.3 m/s 得 0.407 分。偏差越大分越低，但永远不扣成负分。")
    savefig(fig, "ch01_overview")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("10. 宽容度 σ：同样的偏差，扣多少分由它定")
PROJECT_S2 = [("站直", 0.05), ("线速度", 0.1), ("角速度", 0.5)]      # 项目里用钟形打分的奖励不止这三项（还有姿态、头部跟踪等）；这里挑三项，σ² 由第 15 节对着源码核对
print("  手算同一个偏差 0.3，换三种 σ²：")
for name, s2 in PROJECT_S2:
    print(f"    {name}  σ² = {s2:g}：0.09 ÷ {s2:g} = {0.09 / s2:g} → exp(−{0.09 / s2:g}) = {score(0.3, s2):.3f}")
check("偏差 0.3：σ² = 0.05 → 0.165；0.1 → 0.407；0.5 → 0.835",
      [round(score(0.3, s2), 3) for _, s2 in PROJECT_S2] == [0.165, 0.407, 0.835]
      and [round(0.09 / s2, 6) for _, s2 in PROJECT_S2] == [1.8, 0.9, 0.18])
DEVS10 = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
print()
table(["偏差", "偏差²", *[f"σ² = {s2:g}（{name}）" for name, s2 in PROJECT_S2]],
      [[num(d), num(d * d), *[score(d, s2) for _, s2 in PROJECT_S2]] for d in DEVS10], floatfmt=".4f")
check("σ² 越大越宽容：每一行从左到右分数都不降", all(score(d, 0.05) <= score(d, 0.1) <= score(d, 0.5) for d in DEVS10))
check("最后一行的两个 0.0000 不是真的 0：真值是 0.000000002 和 0.000045", f"{score(1.0, 0.05):.9f}" == "0.000000002"
      and f"{score(1.0, 0.1):.6f}" == "0.000045")
check("偏差为 0 时三条都恰好是 1；偏差越大分数越小", all(score(0, s2) == 1 for _, s2 in PROJECT_S2)
      and all(score(a, s2) > score(b, s2) for _, s2 in PROJECT_S2 for a, b in zip(DEVS10, DEVS10[1:])))

print("\n  σ 和 σ² 是两个数。“偏差多大算大”要和 σ 比：偏差恰好等于 σ 时，分数 = exp(−1)")
rows10 = [[name, s2, math.sqrt(s2), score(math.sqrt(s2), s2)] for name, s2 in PROJECT_S2]
table(["哪一项", "配置里的 σ²", "σ = √σ²", "偏差 = σ 时的分数"], rows10, floatfmt=".3f")
check("σ = 0.224、0.316、0.707；偏差 = σ 时分数都是 0.368", [round(r[2], 3) for r in rows10] == [0.224, 0.316, 0.707]
      and all(round(r[3], 3) == 0.368 for r in rows10))
check("偏差 0.1 m/s 只有 σ = 0.316 的约三分之一，分数还有 0.905", round(0.1 / math.sqrt(0.1), 2) == 0.32 and round(score(0.1, 0.1), 3) == 0.905)
check("自测：σ² = 0.5、偏差 0.707 → 0.5 ÷ 0.5 = 1 → 0.368", round(score(0.707, 0.5), 3) == 0.368 and round(0.707**2, 2) == 0.5)

# ---------------------------------------------------------------------------
banner("10b. 画图：figures/ch01_gaussian_kernel.png（同一条公式，三个 σ）")
KERNEL_COLORS = [GREEN, BLUE, ORANGE]
fig = plt.figure(figsize=(9.4, 15.6))
gs = fig.add_gridspec(2, 1, hspace=0.42, left=0.11, right=0.96, top=0.915, bottom=0.095)
fig.suptitle("同一条公式，σ 决定钟有多宽：σ 越大越宽容", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
dev = np.linspace(-1.5, 1.5, 600)
ax = fig.add_subplot(gs[0])
data_axes(ax, "偏差（每一项用它自己的单位；负 = 偏向另一边）", "分数")
for (name, s2), color in zip(PROJECT_S2, KERNEL_COLORS):
    ax.plot(dev, np.exp(-(dev**2) / s2), color=color, lw=3.2, zorder=3)
    ax.plot(0.3, score(0.3, s2), "o", color=color, ms=10, zorder=5)
ax.axvline(0.3, color=MUTED, ls=":", lw=2, zorder=1)
ax.text(0.34, 1.10, "偏差 0.3", color=MUTED, fontsize=FS_SMALL, ha="left", va="center")
for k, ((name, s2), color) in enumerate(zip(reversed(PROJECT_S2), reversed(KERNEL_COLORS))):
    ax.text(-1.46, 1.10 - 0.13 * k, f"{name} σ² = {s2:g}：{score(0.3, s2):.3f} 分", color=color, fontsize=FS_SMALL, va="center", bbox=WHITE_BOX)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-0.05, 1.2)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
panel_title(fig, [ax], "① 同样偏了 0.3，三条曲线给的分数差很多")
panel_note(fig, [ax], "绿 = 站直（σ² = 0.05，钟最窄），蓝 = 线速度（0.1），橙 = 角速度（0.5，钟最宽）。\n三项的单位不同，这里只比形状：σ 越大，钟越宽。")

ax = fig.add_subplot(gs[1])
data_axes(ax, "偏差的大小（不分左右，只画右半口钟）", "分数")
half = np.linspace(0, 1.3, 500)
ax.axhline(math.exp(-1), color=MUTED, ls=":", lw=2, zorder=1)
for ((name, s2), color), (tx, ty, ha) in zip(zip(PROJECT_S2, KERNEL_COLORS), [(0.205, 0.17, "right"), (0.40, 0.31, "left"), (0.735, 0.46, "left")]):
    s = math.sqrt(s2)
    ax.plot(half, np.exp(-(half**2) / s2), color=color, lw=3.2, zorder=3)
    ax.plot([s, s], [-0.05, math.exp(-1)], color=color, ls="--", lw=1.8, zorder=2)
    ax.plot(s, math.exp(-1), "o", color=color, ms=10, zorder=5)
    ax.text(tx, ty, f"σ = {s:.3f}", color=color, fontsize=FS_SMALL, ha=ha, va="center", bbox=WHITE_BOX, zorder=6)
ax.text(1.28, math.exp(-1) + 0.035, f"分数 {math.exp(-1):.2f}", color=MUTED, fontsize=FS_SMALL, ha="right", va="bottom")
ax.set_xlim(0, 1.3)
ax.set_ylim(-0.05, 1.2)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
panel_title(fig, [ax], "② σ 是一把尺子：偏差恰好等于 σ 时，分数掉到 0.37")
panel_note(fig, [ax], "配置里写的 0.05、0.1、0.5 是 σ²；横尺上对应的是 σ = 0.224、0.316、0.707。\n判断一个偏差算大算小，拿它和 σ 比。")
savefig(fig, "ch01_gaussian_kernel")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("11. 敏感程度：分数高，不等于对改善敏感——偏差同样减少 0.05，多拿的分差很多")
IMPROVE = [("A", 0.05, 0.0), ("B", 0.25, 0.20), ("C", 1.0, 0.95)]
gains = {name: score(new) - score(old) for name, old, new in IMPROVE}
table(["情况", "改善前的分数", "改善后的分数", "多拿到的分"],
      [[f"{name}：{old:.2f} → {new:.2f}", f"{score(old):.6f}", f"{score(new):.6f}", f"{gains[name]:.6f}"] for name, old, new in IMPROVE])
check("A：0.975310 → 1，+0.024690；B：0.535261 → 0.670320，+0.135059；C：0.000045 → 0.000120，+0.000075",
      [f"{score(old):.6f}" for _, old, _ in IMPROVE] == ["0.975310", "0.535261", "0.000045"]
      and [f"{score(new):.6f}" for _, _, new in IMPROVE] == ["1.000000", "0.670320", "0.000120"]
      and [f"{gains[n]:.6f}" for n in "ABC"] == ["0.024690", "0.135059", "0.000075"])
print(f"  B 多拿的分是 A 的 {gains['B'] / gains['A']:.2f} 倍")
check("B 的加分约是 A 的 5.47 倍；三者排序 B > A > C", round(gains["B"] / gains["A"], 2) == 5.47 and gains["B"] > gains["A"] > gains["C"])
gain_d = score(0.45) - score(0.5)
print(f"  自测 D：0.50 → 0.45，{score(0.5):.6f} → {score(0.45):.6f}，多拿 {gain_d:.6f}")
check("自测 D：0.082085 → 0.131994，多拿 0.049909（比 A 多、比 B 少）",
      f"{score(0.5):.6f}" == "0.082085" and f"{score(0.45):.6f}" == "0.131994" and f"{gain_d:.6f}" == "0.049909"
      and gains["B"] > gain_d > gains["A"])

# ---------------------------------------------------------------------------
banner("11b. 画图：figures/ch01_reward_zones.png（高而平 / 下降快 / 低而平）")
if SIGMA2 != 0.1:
    print("  σ² 改过了：这张图的标注是照着 σ² = 0.1 摆的，跳过。看上面那张表就行。")
else:
    fig = plt.figure(figsize=(9.4, 14.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[6.0, 2.6], hspace=0.42, left=0.11, right=0.96, top=0.91, bottom=0.085)
    fig.suptitle("偏差同样减少 0.05：中间那一段多拿的分最多", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
    ax = fig.add_subplot(gs[0])
    data_axes(ax, "偏差的大小（m/s，不分左右，所以只有右半口钟）", "分数")
    mag = np.linspace(0, 1.15, 500)
    ax.plot(mag, np.exp(-(mag**2) / SIGMA2), color=BLUE, lw=3.5, zorder=3)
    ax.axvspan(0.1, 0.45, color="#fdf3e6", zorder=0)          # 中间的陡坡段，只是示意，边界不是精确的分界线
    ZONE_NAMES = {"A": "顶上：高而平", "B": "中间：下降快", "C": "尾巴：低而平"}
    for (name, old, new), color, (tx, ty) in zip(IMPROVE, [GREEN, ORANGE, GREEN], [(0.17, 1.06), (0.42, 0.66), (0.64, 0.22)]):
        ax.plot([old, new], [score(old), score(new)], "o", color=color, ms=10, zorder=5)
        ax.plot([old, old], [score(old), score(new)], color=color, lw=4, zorder=4, solid_capstyle="butt")
        ax.plot([old, new], [score(new), score(new)], color=color, lw=1.6, ls="--", zorder=4)
        ax.annotate(f"{name}（{ZONE_NAMES[name]}）\n{old:.2f} → {new:.2f}，多拿 {gains[name]:.6f}", xy=(old, (score(old) + score(new)) / 2), xytext=(tx, ty),
                    fontsize=FS_SMALL, color=color, va="center", linespacing=1.3,
                    arrowprops=dict(arrowstyle="-", lw=1.3, color=color, shrinkB=4))
    ax.set_xlim(-0.03, 1.15)
    ax.set_ylim(-0.05, 1.2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    panel_title(fig, [ax], "① 三次改善都是往左挪 0.05；竖着的粗线 = 多拿的分")
    panel_note(fig, [ax], "A 在顶上，C 在尾巴上：那里曲线几乎是平的，挪一点分数不怎么变。\nB 在中间的陡坡上：同样挪 0.05，分数涨了一大截。\n浅橙色那一段只是示意陡坡大致在哪，边界不是精确的分界线。")

    ax = fig.add_subplot(gs[1])
    lesson_panel(ax, xmax=10, ymax=3.0)
    BAR_SCALE = 7.5 / gains["B"]          # 最长的一条画 7.5 个单位，其余按比例
    for k, ((name, _, _), color) in enumerate(zip(IMPROVE, [GREEN, ORANGE, GREEN])):
        y = 2.2 - k * 0.85
        ax.text(0.55, y, name, fontsize=FS_STEP, color=INK, ha="right", va="center")
        ax.add_patch(plt.Rectangle((0.75, y - 0.25), max(gains[name] * BAR_SCALE, 0.02), 0.5, facecolor=color, edgecolor="none"))
        ax.text(0.75 + gains[name] * BAR_SCALE + 0.15, y, f"{gains[name]:.6f}", fontsize=FS_SMALL, color=color, va="center")
    panel_title(fig, [ax], "② 把三份“多拿的分”按同一个比例画成条", pad=-0.004)
    panel_note(fig, [ax], "B 是 A 的 5.47 倍；C 的那一条短到几乎看不见。", pad=0.0)
    savefig(fig, "ch01_reward_zones")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("12. 钟形打分的导数：先缩 h 量，再用链式法则算，两边对上")


def d_score(x, s2=None):
    """链式法则：外层 eᵘ 的导数 eᵘ = 分数本身；内层 u = −x²/σ² 的导数 −2x/σ²。"""
    s2 = SIGMA2 if s2 is None else s2
    return -2 * x / s2 * score(x, s2)


XQ = 0.1
print(f"  先量：在偏差 x = {XQ:g} 处缩 h（分数 R({XQ:g}) = {score(XQ):.6f}）")
rows12 = [[h, score(XQ + h), score(XQ + h) - score(XQ), (score(XQ + h) - score(XQ)) / h] for h in (0.01, 0.001, 0.0001)]
table(["h", "R(0.1+h)", "分数变了多少", "平均变化率"], [[num(h), f"{a:.6f}", f"{b:.6f}", f"{c:.4f}"] for h, a, b, c in rows12])
check("缩 h：R(0.11) = 0.886034 → −1.8803；R(0.101) = 0.903021 → −1.8169；再缩 → −1.8104，靠近 −1.81",
      f"{rows12[0][1]:.6f}" == "0.886034" and f"{rows12[1][1]:.6f}" == "0.903021"
      and [f"{r[3]:.4f}" for r in rows12] == ["-1.8803", "-1.8169", "-1.8104"])
check("表里另外两列：R(0.1001) = 0.904656；分数变了 −0.018803、−0.001817、−0.000181；h = 0.001 那一行的变化率约 −1.817",
      f"{rows12[2][1]:.6f}" == "0.904656" and [f"{r[2]:.6f}" for r in rows12] == ["-0.018803", "-0.001817", "-0.000181"]
      and f"{rows12[1][3]:.3f}" == "-1.817" and round(0.904837 - 0.903021, 6) == 0.001816)      # 最后一项：两个六位小数直接相减
inner, outer = -2 * XQ / SIGMA2, score(XQ)
print(f"  再算：内层 u = −x²/{SIGMA2:g} = −{1 / SIGMA2:g} × x²，倍数 −{1 / SIGMA2:g} × 2x = −{2 / SIGMA2:g}x；在 x = {XQ:g} 处 −{2 / SIGMA2:g} × {XQ:g} = {inner:g}")
print(f"        外层 eᵘ 的倍数 = eᵘ = 分数本身 {outer:.6f} ≈ {outer:.3f}；相乘 = {inner * outer:.6f}（正文用舍入后的数：{inner:g} × {outer:.3f} = {inner * round(outer, 3):.2f}）")
check("链式法则：内层倍数 −2，外层倍数 0.904837，R′(0.1) = −1.809675", round(inner, 9) == -2 and f"{outer:.6f}" == "0.904837"
      and f"{d_score(XQ):.6f}" == "-1.809675")
check("正文“再算”那段：σ² = 0.1，除以 0.1 就是乘 10；内层倍数 −10 × 2x = −20x，在 0.1 处 −20 × 0.1 = −2；外层 ≈ 0.905；−2 × 0.905 = −1.81",
      1 / SIGMA2 == 10 and 2 / SIGMA2 == 20 and round(-20 * 0.1, 9) == -2 and f"{outer:.3f}" == "0.905"
      and round(-2 * 0.905, 2) == -1.81 and f"{d_score(XQ):.2f}" == "-1.81")
check("量的和算的对上了：h = 0.000001 时缩 h 得 −1.8097", f"{(score(XQ + 1e-6) - score(XQ)) / 1e-6:.4f}" == "-1.8097")

print("\n  五个位置，公式与缩 h（左右各走 0.000001 取平均）并排：")
FIVE = sorted([-0.3, 0.0, 0.1, SIGMA / math.sqrt(2), 0.6])      # 排序：TWEAK-3 改了 σ² 以后，σ/√2 仍落在该在的行
rows12b = []
for x in FIVE:
    numeric = (score(x + 1e-6) - score(x - 1e-6)) / 2e-6
    rows12b.append([x, score(x), -2 * x / SIGMA2 + 0.0, d_score(x) + 0.0, numeric + 0.0])
table(["偏差 x", "分数 R(x)", "系数 −2x/σ²", "导数 R′(x)", "缩 h 量到的"],
      [[num(x, 6), f"{r:.6f}", num(c, 4), f"{d:.6f}", f"{n:.6f}"] for x, r, c, d, n in rows12b])
check("五个点上公式 = 缩 h（相差不到 0.000001）", all(abs(r[3] - r[4]) < 1e-6 for r in rows12b))
check("R′(−0.3) = +2.439418 ≈ +2.44（偏差为负时导数为正）；R′(0) = 0（钟顶是平的）；R′(0.6) = −0.327885",
      f"{rows12b[0][3]:.6f}" == "2.439418" and f"{rows12b[0][3]:.2f}" == "2.44" and rows12b[1][3] == 0 and f"{rows12b[4][3]:.6f}" == "-0.327885")

real_up, pred_up = score(0.101) - score(0.1), d_score(0.1) * 0.001
real_down, pred_down = score(0.099) - score(0.1), d_score(0.1) * -0.001
print(f"\n  拿导数当“汇率”做预测：0.100 → 0.101，预测 {pred_up:.6f}，实际 {real_up:.6f}；0.100 → 0.099，预测 {pred_down:+.6f}，实际 {real_down:+.6f}")
check("折叠块：0.100 → 0.101，预测 −1.81 × 0.001 = −0.001810，实际 −0.001817（worked_examples 冻结值 −0.001816897）",
      f"{pred_up:.6f}" == "-0.001810" and round(-1.81 * 0.001, 6) == -0.00181 and f"{real_up:.6f}" == "-0.001817" and round(real_up, 9) == -0.001816897)
check("0.100 → 0.099：预测 +0.001810，实际 +0.001802", f"{pred_down:.6f}" == "0.001810" and f"{real_down:.6f}" == "0.001802")
check("自测：系数 −20 × 0.3 = −6；R(0.3) ≈ 0.4066；−6 × 0.4066 = −2.4396 ≈ −2.44，和 R′(−0.3) = +2.439418 大小相同、符号相反",
      round(-2 * 0.3 / SIGMA2, 9) == -6 and round(-20 * 0.3, 9) == -6 and f"{score(0.3):.4f}" == "0.4066"
      and round(-6 * 0.4066, 2) == -2.44 and f"{d_score(0.3):.2f}" == "-2.44" and math.isclose(d_score(0.3), -d_score(-0.3)))

# ---------------------------------------------------------------------------
banner("13. 最敏感的位置：坡度的大小 S = 系数 × 分数，在 σ/√2 处最大")


def steepness(x, s2=None):
    return abs(d_score(x, s2))


PEAK = SIGMA / math.sqrt(2)
S_ROWS = sorted([0.0, 0.10, 0.20, PEAK, 0.30, 0.60, 1.00])
table(["偏差 x", "系数 2x/σ²", "分数 R(x)", "坡度的大小 S(x)"],
      [[num(x, 6), num(2 * x / SIGMA2, 4), f"{score(x):.6f}", f"{steepness(x):.6f}"] for x in S_ROWS])
check("S 那一列：0、1.809675、2.681280、2.712488、2.439418、0.327885、0.000908",
      [f"{steepness(x):.6f}" for x in S_ROWS] == ["0.000000", "1.809675", "2.681280", "2.712488", "2.439418", "0.327885", "0.000908"])
check("系数那一列：0、2、4、4.4721、6、12、20（一路变大）；分数那一列一路变小",
      [num(2 * x / SIGMA2, 4) for x in S_ROWS] == ["0", "2", "4", "4.4721", "6", "12", "20"]
      and all(score(a) > score(b) for a, b in zip(S_ROWS, S_ROWS[1:])))
check("正文手算：R′(0.1) 的大小 = 系数 20 × 0.1 = 2，再乘分数 0.905：2 × 0.905 = 1.81", 2 / SIGMA2 == 20 and round(20 * 0.1, 9) == 2
      and round(2 * 0.905, 2) == 1.81 and f"{steepness(0.1):.2f}" == "1.81")
scan = np.arange(0, 3.0005, 0.001)
scan_s = 2 * scan / SIGMA2 * np.exp(-(scan**2) / SIGMA2)
scan_peak = float(scan[scan_s.argmax()])
print(f"  每隔 0.001 扫一遍：S 最大的位置是 {scan_peak:.3f}；σ/√2 = √({SIGMA2:g}/2) = {PEAK:.6f}；那里的分数 = {score(PEAK):.6f}")
check("扫描找到的峰在 0.224，就是 σ/√2 = √0.05 = 0.223607", round(scan_peak, 3) == 0.224 and f"{PEAK:.6f}" == "0.223607"
      and math.isclose(PEAK, math.sqrt(0.05)))
check("正文的手算：√2 ≈ 1.4142，σ ≈ 0.3162，0.3162 ÷ 1.4142 ≈ 0.224（表里的 0.223607）；它乘自己是 0.05，和站直那一项的 σ = √0.05 同数",
      f"{math.sqrt(2):.4f}" == "1.4142" and f"{SIGMA:.4f}" == "0.3162" and round(0.3162 / 1.4142, 3) == 0.224
      and f"{PEAK:.3f}" == "0.224" and f"{PEAK:.6f}" == "0.223607" and round(0.223607**2, 4) == 0.05 and math.isclose(PEAK, math.sqrt(0.05)))
check("最敏感处的分数 = exp(−0.5) = 0.606531 ≈ 0.607，约 0.61，远不是满分", math.isclose(score(PEAK), math.exp(-0.5))
      and f"{score(PEAK):.6f}" == "0.606531" and f"{score(PEAK):.3f}" == "0.607" and f"{score(PEAK):.2f}" == "0.61")
print(f"  左右各挪 0.01：S = {steepness(PEAK - 0.01):.4f} 和 {steepness(PEAK + 0.01):.4f}，峰上 {steepness(PEAK):.4f}")
check("它比左右邻居（±0.01）都陡：2.7070 < 2.7125 > 2.7071",
      steepness(PEAK) > steepness(PEAK - 0.01) and steepness(PEAK) > steepness(PEAK + 0.01)
      and f"{steepness(PEAK - 0.01):.4f}" == "2.7070" and f"{steepness(PEAK + 0.01):.4f}" == "2.7071" and f"{steepness(PEAK):.4f}" == "2.7125")

print("\n  折叠块：乘法的求导规则 (fg)′ = f′g + fg′，先用认识的函数验：")
print("    x · x：1 × x + x × 1 = 2x ✓      x² · x³：2x × x³ + x² × 3x² = 5x⁴，正是 (x⁵)′ ✓")
check("乘法规则：x · x → 2x；x² · x³ → 5x⁴（在 x = 2 处 80，缩 h 也得 80）",
      abs(shrink_h(lambda x: x**5, 2.0, [1e-7])[0][2] - 80) < 1e-3 and 2 * 2 * 2**3 + 2**2 * 3 * 2**2 == 80 == 5 * 2**4)


def d_steepness(x):
    return 2 / SIGMA2 * score(x) * (1 - 2 * x**2 / SIGMA2)


rows13 = [[x, 1 - 2 * x**2 / SIGMA2 + 0.0, d_steepness(x) + 0.0, (steepness(x + 1e-6) - steepness(x - 1e-6)) / 2e-6 + 0.0] for x in (0.2, PEAK, 0.3)]
table(["偏差 x", "括号 1 − 2x²/σ²", "S′(x)（公式）", "S′(x)（缩 h）"], [[num(x, 6), num(a, 4), num(b, 4), num(c, 4)] for x, a, b, c in rows13])
check("S′ 在峰的左边为正（0.2 处 2.6813）、峰上为 0、右边为负（0.3 处 −6.5051）；公式与缩 h 一致",
      [num(r[2], 4) for r in rows13] == ["2.6813", "0", "-6.5051"] and all(abs(r[2] - r[3]) < 1e-5 for r in rows13)
      and [num(r[1], 4) for r in rows13] == ["0.2", "0", "-0.8"])

print("\n  σ 变了，最敏感的位置跟着搬（另挑的三个 σ；第一行的 0.1 是 σ 本身，σ² 是 0.01）：")
SIGMAS = [0.1, math.sqrt(0.1), 1.0]
rows13s = [[s, s / math.sqrt(2), steepness(s / math.sqrt(2), s * s)] for s in SIGMAS]
table(["σ", "σ²", "最敏感的偏差 σ/√2", "那里的坡度"], [[num(s, 6), num(s * s, 6), f"{p:.6f}", num(a, 3)] for s, p, a in rows13s])
check("σ = 0.1（σ² = 0.01）→ 0.070711；σ = 0.316228（σ² = 0.1）→ 0.223607；σ = 1（σ² = 1）→ 0.707107",
      [f"{r[1]:.6f}" for r in rows13s] == ["0.070711", "0.223607", "0.707107"] and num(SIGMAS[1], 6) == "0.316228"
      and [num(s * s, 6) for s in SIGMAS] == ["0.01", "0.1", "1"])
check("峰的高度 8.578、2.712、0.858：σ 越大，峰越靠右、越矮", [num(r[2], 3) for r in rows13s] == ["8.578", "2.712", "0.858"])
check("第三行的峰 0.707107 和 1.10 节角速度那一项的 σ = √0.5 同数（巧合，角色不同）", math.isclose(rows13s[2][1], math.sqrt(0.5)))
check("自测：σ 从 0.1 改成 1（σ² 从 0.01 改成 1），最敏感的位置从约 0.071 移到 0.707", round(rows13s[0][1], 3) == 0.071 and round(rows13s[2][1], 3) == 0.707)
s005_narrow, s005_wide = steepness(0.05, 0.01), steepness(0.05, 1.0)
print(f"  自测：偏差 0.05 处，σ = 0.1 时 S = 10 × e^(−0.25) = {s005_narrow:.2f}；σ = 1 时 S = 0.1 × e^(−0.0025) = {s005_wide:.2f}；前者是后者的 {s005_narrow / s005_wide:.0f} 倍")
check("自测：S(0.05) 在 σ = 0.1 时 = 10 × e^(−0.25) ≈ 10 × 0.779 = 7.79，在 σ = 1 时 = 0.1 × e^(−0.0025) ≈ 0.1 × 0.998 ≈ 0.10；陡了七十多倍",
      round(2 * 0.05 / 0.01, 9) == 10 and round(0.05**2 / 0.01, 9) == 0.25 and round(2 * 0.05 / 1, 9) == 0.1 and round(0.05**2 / 1, 9) == 0.0025
      and f"{math.exp(-0.25):.3f}" == "0.779" and f"{math.exp(-0.0025):.3f}" == "0.998" and round(10 * 0.779, 2) == 7.79
      and round(0.1 * 0.998, 2) == 0.10 and f"{s005_narrow:.2f}" == "7.79" and f"{s005_wide:.2f}" == "0.10" and 70 < s005_narrow / s005_wide < 80)

# ---------------------------------------------------------------------------
banner("13b. 画图：figures/ch01_reward_sensitivity.png（最陡的地方不在最高处）")
if SIGMA2 != 0.1:
    print("  σ² 改过了：这张图的标注是照着 σ² = 0.1 摆的，跳过。看上面几张表就行。")
else:
    fig = plt.figure(figsize=(9.4, 19.6))
    gs = fig.add_gridspec(3, 1, hspace=0.52, left=0.12, right=0.95, top=0.93, bottom=0.075)
    fig.suptitle("最敏感的位置在半山腰：分数只有 0.61，坡却最陡", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.988)
    mag = np.linspace(0, 1.0, 500)

    ax = fig.add_subplot(gs[0])
    data_axes(ax, "偏差的大小（m/s）", "分数")
    ax.plot(mag, np.exp(-(mag**2) / SIGMA2), color=BLUE, lw=3.5, zorder=3)
    span = np.array([PEAK - 0.13, PEAK + 0.13])
    ax.plot(span, score(PEAK) + d_score(PEAK) * (span - PEAK), color=ORANGE, lw=3.2, zorder=4)
    ax.plot(PEAK, score(PEAK), "o", color=ORANGE, ms=11, zorder=5)
    ax.plot(0, 1, "o", color=GREEN, ms=11, zorder=5)
    ax.text(0.03, 1.06, "分数最高：偏差 0，坡度 0（平的）", color=GREEN, fontsize=FS_SMALL, va="bottom")
    ax.annotate(f"最陡：偏差 {PEAK:.3f}，分数 {score(PEAK):.3f}\n橙色切线的斜率 −{abs(d_score(PEAK)):.3f}", xy=(PEAK, score(PEAK)), xytext=(0.40, 0.72),
                fontsize=FS_SMALL, color=ORANGE, va="center", linespacing=1.3, arrowprops=dict(arrowstyle="-", lw=1.3, color=ORANGE, shrinkB=7))
    ax.set_xlim(-0.025, 1.0)
    ax.set_ylim(-0.05, 1.2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    panel_title(fig, [ax], "① 分数曲线（σ² = 0.1）：最高的地方是平的，最陡的地方在半山腰")
    panel_note(fig, [ax], "“最敏感”说的是坡最陡，不是分数最高。")

    ax = fig.add_subplot(gs[1])
    data_axes(ax, "偏差的大小（m/s）", "坡度的大小（分 / (m/s)）")
    ax.plot(mag, 2 * mag / SIGMA2 * np.exp(-(mag**2) / SIGMA2), color=BLUE, lw=3.5, zorder=3)
    for x, (dx, dy, ha) in zip((0.10, 0.20, 0.30, 0.60), [(0.02, -0.08, "left"), (-0.02, 0.08, "right"), (0.02, 0.08, "left"), (0.015, 0.1, "left")]):
        ax.plot(x, steepness(x), "o", color=BLUE, ms=9, zorder=5)
        ax.text(x + dx, steepness(x) + dy, f"{steepness(x):.2f}", color=BLUE, fontsize=FS_TICK, ha=ha, va="center")
    ax.plot(PEAK, steepness(PEAK), "o", color=ORANGE, ms=11, zorder=6)
    ax.axvline(PEAK, color=ORANGE, ls=":", lw=2, zorder=1)
    ax.text(PEAK + 0.02, 3.02, f"峰：偏差 {PEAK:.3f}，坡度 {steepness(PEAK):.2f}", color=ORANGE, fontsize=FS_SMALL, va="center")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.1, 3.3)
    panel_title(fig, [ax], "② 把每个位置的坡度画出来：先升后降，峰在 0.224")
    panel_note(fig, [ax], "蓝点旁的数是正文那张表的 S 一列。偏差很小或很大时，坡度都接近 0。")

    ax = fig.add_subplot(gs[2])
    data_axes(ax, "偏差的大小（同一个物理量、同一个单位）", "坡度的大小")
    wide = np.linspace(0, 1.5, 800)
    for (s, p, a), color, (tx, ty) in zip(rows13s, [GREEN, BLUE, ORANGE], [(0.30, 8.3), (0.42, 3.9), (0.82, 2.0)]):
        ax.plot(wide, 2 * wide / s**2 * np.exp(-(wide / s) ** 2), color=color, lw=3.2, zorder=3)
        ax.plot(p, a, "o", color=color, ms=10, zorder=5)
        ax.annotate(f"σ = {s:.3g}（σ² = {s * s:.2g}）：峰在 {p:.3f}", xy=(p, a), xytext=(tx, ty), fontsize=FS_SMALL, color=color, va="center",
                    arrowprops=dict(arrowstyle="-", lw=1.3, color=color, shrinkB=6))
    ax.set_xlim(0, 1.5)
    ax.set_ylim(-0.3, 9.6)
    panel_title(fig, [ax], "③ 换一个 σ，最敏感的位置跟着搬家（σ 大一倍，位置远一倍）")
    panel_note(fig, [ax], "σ 小：峰又高又靠近 0；σ 大：峰又矮又远，处处都缓。\n只有蓝线和 1.10 节图里的蓝线是同一条（σ² = 0.1）；绿、橙是另挑的 σ。")
    savefig(fig, "ch01_reward_sensitivity")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("14. σ 怎么选：让现在还值得改善的偏差，落在分得出高低的区间")
print("  假设现在常见的偏差是 0.20 m/s，改善到 0.19 值得鼓励。三种 σ：")
rows14 = []
for s in (0.01, SIGMA, 10.0):
    a, b = math.exp(-((0.20 / s) ** 2)), math.exp(-((0.19 / s) ** 2))
    a32 = float(np.exp(np.float32(-((0.20 / s) ** 2))))
    b32 = float(np.exp(np.float32(-((0.19 / s) ** 2))))
    rows14.append([s, a, b, b - a, a32, b32])


def tiny(value, exponent):
    """小到写不下的分数，写成 exp(指数) 的样子；正常大小的保留 6 位小数。"""
    return f"exp({exponent:.0f})" if value < 1e-6 else f"{value:.6f}"


table(["σ", "R(0.20)", "R(0.19)", "多拿的分", "float32 里存成"],
      [[num(s, 6), tiny(a, -((0.20 / s) ** 2)), tiny(b, -((0.19 / s) ** 2)), "几乎是 0" if d < 1e-6 else f"{d:.6f}",
        f"{a32:.6g} 和 {b32:.6g}"] for s, a, b, d, a32, b32 in rows14])
check("σ = 0.01（σ² = 0.0001）：指数是 −400 和 −361（0.2 ÷ 0.01 = 20、平方 400，和 0.04 ÷ 0.0001 同数），两个分数在 float32 里都下溢成 0",
      round(-((0.20 / 0.01) ** 2)) == -400 and round(0.04 / 0.0001) == 400 and math.isclose(0.01**2, 0.0001) and round(-((0.19 / 0.01) ** 2)) == -361 and rows14[0][4] == 0 and rows14[0][5] == 0)
check("σ = √0.1：0.670320 → 0.696979，多拿 0.026659（worked_examples 冻结值 0.026658952）",
      f"{rows14[1][1]:.6f}" == "0.670320" and f"{rows14[1][2]:.6f}" == "0.696979" and round(rows14[1][3], 9) == 0.026658952)
check("σ = 10：0.999600 → 0.999639，只多 0.000039", f"{rows14[2][1]:.6f}" == "0.999600" and f"{rows14[2][2]:.6f}" == "0.999639"
      and f"{rows14[2][3]:.6f}" == "0.000039")
D_TARGET = 0.2
print(f"  想让最陡的位置落在偏差 d = {D_TARGET:g}：从 σ ≈ √2 × d = {math.sqrt(2) * D_TARGET:.3f} 起步（只是这条曲线的几何关系，不是自动调参公式）")
check("√2 × 0.2 = 0.283；反过来 σ = 0.283 时峰在 0.2", round(math.sqrt(2) * D_TARGET, 3) == 0.283
      and math.isclose(math.sqrt(2) * D_TARGET / math.sqrt(2), D_TARGET))
WEIGHT = 2.0          # 项目里 track_linear_velocity 这一项的权重（第 15 节对着源码核对）
weighted = WEIGHT * scan_s
print(f"  乘上权重 {WEIGHT:g}：整条曲线和坡度都放大 {WEIGHT:g} 倍，峰高 {steepness(PEAK):.2f} → {WEIGHT * steepness(PEAK):.2f}，峰的位置仍在 {float(scan[weighted.argmax()]):.3f}")
check("权重只放大、不搬家：峰高 2.71 → 5.42（拿舍入后的数重算：2 × 2.71 = 5.42），位置不变", float(scan[weighted.argmax()]) == scan_peak
      and f"{steepness(PEAK):.2f}" == "2.71" and f"{WEIGHT * steepness(PEAK):.2f}" == "5.42" and round(2 * 2.71, 2) == 5.42)
check("已经下溢成 0 的分数，乘 10 还是 0", rows14[0][4] * 10 == 0)

# ---------------------------------------------------------------------------
banner("15. 映射到项目：正文引用的源码行和常数还在不在")
REPO = Path(__file__).resolve().parents[3]
REWARD_LINES = ["xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)",
                "z_error = torch.square(actual[:, 2])",
                "lin_vel_error = xy_error + z_error",
                "return torch.exp(-lin_vel_error / std**2)"]
CFG_LINES = ['cfg.rewards["upright"].params["std"] = math.sqrt(0.05)',
             'cfg.rewards["track_linear_velocity"].weight = 2.0',
             'cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.1)',
             'cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)']

mjlab_spec = importlib.util.find_spec("mjlab")
rewards_path = Path(mjlab_spec.origin).parent / "tasks" / "velocity" / "mdp" / "rewards.py" if mjlab_spec and mjlab_spec.origin else None
if rewards_path and rewards_path.is_file():
    rewards_text = rewards_path.read_text(encoding="utf-8")
    at = rewards_text.find("def track_linear_velocity(")
    print("mjlab/tasks/velocity/mdp/rewards.py 的 track_linear_velocity：")
    for code in REWARD_LINES:
        print("   ", code)
    check("正文映射块的四行代码原样存在于 track_linear_velocity 里，顺序一致", at >= 0 and lines_in_order(rewards_text[at:], REWARD_LINES)
          and rewards_text.find(REWARD_LINES[-1], at) < rewards_text.find("def track_angular_velocity("))
else:
    print("  （当前 Python 环境里没有 mjlab，跳过源码核对；用 uv run 运行就会核对）")
cfg_path = REPO / "src" / "mjlab_microduck" / "tasks" / "microduck_velocity_env_cfg.py"
if cfg_path.is_file():
    cfg_text = cfg_path.read_text(encoding="utf-8")
    at = cfg_text.find("def make_microduck_velocity_env_cfg(")
    print("microduck_velocity_env_cfg.py 的 make_microduck_velocity_env_cfg：站直 sqrt(0.05)、线速度 sqrt(0.1)（权重 2.0）、角速度 sqrt(0.5)")
    check("三个宽容度和线速度的权重 2.0 原样存在，顺序一致", at >= 0 and lines_in_order(cfg_text[at:], CFG_LINES))
    check("实验里用的 σ² 和权重与配置一致", [s2 for _, s2 in PROJECT_S2] == [0.05, 0.1, 0.5] and WEIGHT == 2.0)
    check("指令的前后速度范围是 ±0.4 m/s（“改一改”第 3 条用到）", "command.ranges.lin_vel_x = (-0.4, 0.4)" in cfg_text)
else:
    print("  （没找到项目的 env cfg，跳过这一项）")

done()
