"""第 3 章实验：两个旋钮的碗、偏导数、梯度、梯度下降、找直线、学习率与梯度裁剪。

运行：uv run python docs/learn-zh/labs/ch03_gradient_descent.py
纯 CPU，numpy + matplotlib。第 8 节只读两份源码的文字，不加载机器人、不训练。
小节编号与正文一一对应：实验第 K 节 = 正文 3.K 节（第 8 节对应「映射到项目」）。
正文“改一改”要改的三行都带 `# TWEAK-k:` 标记（第 5 节两处、第 7 节一处）。
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

from _common import banner, check, done, lines_in_order, savefig, table
from _draw import (BLUE, FAINT, FS_NOTE, FS_SMALL, FS_STEP, FS_TICK, FS_TITLE, GREEN, INK, MUTED, ORANGE, WHITE_BOX,
                   arrow, cell, data_axes, hand, lesson_cells, lesson_figure, lesson_panel, note, panel_note,
                   panel_title, plt)

np.set_printoptions(precision=4, suppress=True)
np.seterr(over="ignore", invalid="ignore")   # 发散时数字会大到变成 inf / nan——这正是要给读者看的现象，不用 numpy 再警告一遍


# --- 本章独有的小工具 ---------------------------------------------------------------------
def num(v, places=4) -> str:
    """要贴进正文的表：最多 places 位小数，去掉尾零（1.4364 → "1.4364"，2.5600 → "2.56"，−0.0 → "0"）。"""
    s = f"{v:.{places}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def ring_xy(height, n=240):
    """x² + 3y² = height 这一圈上的点。"""
    t = np.linspace(0, 2 * np.pi, n)
    return math.sqrt(height) * np.cos(t), math.sqrt(height / 3) * np.sin(t)


def ring_point(height, deg):
    """height 那一圈上、参数角为 deg 的那个点（用来把字贴在圈上）。"""
    t = math.radians(deg)
    return math.sqrt(height) * math.cos(t), math.sqrt(height / 3) * math.sin(t)


# ---------------------------------------------------------------------------
banner("1. 两个旋钮决定一个数：f(x, y) = x² + 3y² 的高度表")


def f(x, y):
    return x**2 + 3 * y**2


x0, y0 = 1.0, 2.0                      # 全章都站在这个位置
f0 = f(x0, y0)
print("手算 f(1, 2) = 1² + 3 × 2² = 1 + 12 =", 1**2 + 3 * 2**2)
print("机器 f(1, 2) =", f0)
check("f(1, 2) = 13", f0 == 13)

grid_x = [-2, -1, 0, 1, 2]
grid_y = [2, 1, 0, -1, -2]             # 从上往下排：和地图一样，上面是 y 大的一侧
height_table = [[f(x, y) for x in grid_x] for y in grid_y]
print("\n把位置铺成 5 × 5 的格子，每个位置算一个高度：")
table(["", *[f"x={x}" for x in grid_x]], [[f"y={y}", *row] for y, row in zip(grid_y, height_table)])

same_height: dict[int, list[tuple[int, int]]] = {}
for y in grid_y:
    for x in grid_x:
        same_height.setdefault(f(x, y), []).append((x, y))
print()
for h in (4, 7, 13):
    print(f"高度同为 {h:>2} 的位置：", same_height[h])
check("高度 4 的位置有 6 个：(±2, 0) 和 (±1, ±1)",
      sorted(same_height[4]) == sorted([(2, 0), (-2, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]))
check("高度 13 的位置有 4 个：(±1, ±2)", sorted(same_height[13]) == sorted([(1, 2), (1, -2), (-1, 2), (-1, -2)]))
check("碗底在 (0, 0)，高度 0，是表里最低的", min(same_height) == 0 and same_height[0] == [(0, 0)])
check("同样离碗底 1 格：f(0, 1) = 3 是 f(1, 0) = 1 的 3 倍（y 方向更陡）", f(0, 1) == 3 and f(1, 0) == 1)
check("自测：f(2, −1) = 7，f(3, 1) = f(0, 2) = 12", f(2, -1) == 7 and f(3, 1) == f(0, 2) == 12)
EVEN_RINGS = [1, 4, 7, 10, 13, 16]      # 每隔 3 画一圈：相邻两圈的高度差相等，“圈密 = 坡陡”才成立
check("每隔 3 一圈时：从碗底往 y 走 2 格（到高度 12）要跨过 4 圈，往 x 走 2 格（到高度 4）只跨过 1 圈、刚踩上第 2 圈",
      sum(ring < f(0, 2) for ring in EVEN_RINGS) == 4 and sum(ring < f(2, 0) for ring in EVEN_RINGS) == 1 and f(2, 0) in EVEN_RINGS)

# ---------------------------------------------------------------------------
banner("1b. 画图：figures/ch03_bowl_contours.png（高度表 → 同高的点连成圈 → 从上往下看碗）")

RINGS = [(1, MUTED), (4, BLUE), (7, GREEN), (13, ORANGE)]   # ② 里要连起来的几圈：高度、颜色
ring_color = dict(RINGS)

fig = plt.figure(figsize=(9.4, 17.2))
gs = fig.add_gridspec(3, 2, height_ratios=[3.2, 5.4, 3.9], width_ratios=[1.0, 1.12], hspace=0.50, wspace=0.26,
                      left=0.10, right=0.96, top=0.925, bottom=0.07)
fig.suptitle("等高线：把一样高的点连成圈，就是从正上方看这只碗", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)

ax = fig.add_subplot(gs[0, :])
lesson_panel(ax, xmax=10, ymax=3.75)
cw, ch, left, bottom = 0.95, 0.56, 1.05, 0.10
lesson_cells(ax, height_table, left, bottom, cw, ch, highlights=((0, 3),))
for j, x in enumerate(grid_x):
    ax.text(left + (j + 0.5) * cw, bottom + 5 * ch + 0.22, f"x={x}".replace("-", "−"), ha="center", fontsize=FS_SMALL, color=MUTED)
for i, y in enumerate(grid_y):
    ax.text(left - 0.15, bottom + (4 - i + 0.5) * ch, f"y={y}".replace("-", "−"), ha="right", va="center", fontsize=FS_SMALL, color=MUTED)
hand(ax, 6.15, 2.62, "站在 (1, 2)：", color=ORANGE)
hand(ax, 6.15, 1.95, f"1² + 3 × 2² = {f0:g}", color=ORANGE)
hand(ax, 6.15, 0.95, "正中间是碗底：0", color=BLUE, fontsize=FS_NOTE)
panel_title(fig, [ax], "① 先量高度：每个位置算一个数", pad=-0.004)
panel_note(fig, [ax], "横着数是旋钮 x，竖着数是旋钮 y，格子里的数是那个位置的高度。")

ax = fig.add_subplot(gs[1, :])
data_axes(ax, "旋钮 x", "旋钮 y")
ax.set_xlim(-4.1, 4.1)                 # 留够宽度：高度 13 那一圈左右伸到 ±3.6，要画成完整的一圈
ax.set_ylim(-2.5, 2.5)
ax.set_aspect("equal")
ax.set_xticks([-4, -3, *grid_x, 3, 4])
ax.set_yticks(sorted(grid_y))
for height, color in RINGS:
    rx, ry = ring_xy(height)
    ax.plot(rx, ry, color=color, lw=3 if height in (4, 13) else 2, zorder=2)
for y in grid_y:
    for x in grid_x:
        height = f(x, y)
        ax.plot(x, y, "o", ms=10, color=ring_color.get(height, FAINT), zorder=4)
        # 数字写在“离圈远”的一侧：左右看 x 的正负；x = 0 那一列的圈紧贴在外侧，所以写在靠碗底的一侧
        side = -1 if x < 0 else 1
        up = (-1 if y > 0 else 1) if x == 0 else (-1 if y < 0 else 1)
        ax.text(x + 0.09 * side, y + (0.07 if up > 0 else -0.09), f"{height:g}", fontsize=FS_SMALL,
                color=INK if height in ring_color else MUTED, fontweight="bold" if height in (4, 13) else "normal",
                ha="left" if side > 0 else "right", va="bottom" if up > 0 else "top", zorder=5)
for height, deg, text in [(4, 210, "高度 4 的圈"), (7, -20, "高度 7 的圈"), (13, 20, "高度 13 的圈")]:
    ax.text(*ring_point(height, deg), text, color=ring_color[height], fontsize=FS_SMALL, ha="center", va="center",
            bbox=WHITE_BOX, zorder=6)     # 字直接压在自己那一圈上，不会认错
panel_title(fig, [ax], "② 把一样高的点连起来")
panel_note(fig, [ax], "高度同为 4 的六个点落在同一个圈上；同为 13 的四个点，在更外面的一圈上。")

ax3 = fig.add_subplot(gs[2, 0], projection="3d", computed_zorder=False)
radius, theta = np.meshgrid(np.linspace(0, 4, 40), np.linspace(0, 2 * np.pi, 120))
ax3.plot_surface(radius * np.cos(theta), radius * np.sin(theta) / math.sqrt(3), radius**2,
                 color="#dfe9f3", edgecolor="none", alpha=0.45, shade=False, zorder=1)
for height in EVEN_RINGS:
    rx, ry = ring_xy(height)
    ax3.plot(rx, ry, np.full_like(rx, float(height)), color=ring_color.get(height, FAINT), lw=2.8, zorder=3)
ax3.plot([x0], [y0], [f0], "o", color=ORANGE, ms=9, mec="white", zorder=4)
ax3.set_box_aspect((8, 4.7, 5.6))
ax3.view_init(elev=28, azim=-66)
ax3.set_xticks([-4, -2, 0, 2, 4])
ax3.set_yticks([-2, 0, 2])
ax3.set_zticks([1, 4, 7, 10, 13, 16])
ax3.tick_params(labelsize=13, colors=MUTED, pad=0)
ax3.set_xlabel("x", fontsize=FS_SMALL, color=INK, labelpad=8)
ax3.set_ylabel("y", fontsize=FS_SMALL, color=INK, labelpad=4)
ax3.text2D(0.80, 0.88, "高度", transform=ax3.transAxes, fontsize=FS_SMALL, color=INK)

ax = fig.add_subplot(gs[2, 1])
data_axes(ax, "旋钮 x", "旋钮 y", grid=False)
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-2.9, 2.9)
ax.set_aspect("equal")
ax.set_anchor("C")
ax.set_xticks([-4, -2, 0, 2, 4])
ax.set_yticks([-2, 0, 2])
for height in EVEN_RINGS:
    rx, ry = ring_xy(height)
    ax.plot(rx, ry, color=ring_color.get(height, FAINT), lw=2.6)
    label_x = -math.sqrt(height) if height in (1, 7, 13) else math.sqrt(height)   # 左右交替着标，数字才不打架
    ax.text(label_x, 0.0, f"{height}", fontsize=15, color=INK, ha="center", va="center", bbox=WHITE_BOX)
ax.plot(0, 0, "*", color=INK, ms=14)
ax.plot(x0, y0, "o", color=ORANGE, ms=9, zorder=5)
ax.text(x0 + 0.30, y0 + 0.34, "(1, 2)", fontsize=15, color=ORANGE, va="bottom", bbox=WHITE_BOX)
panel_title(fig, [ax3, ax], "③ 这些圈叫等高线：左边是碗，右边是从正上方往下看", pad=0.0)
panel_note(fig, [ax3, ax], "★ 是碗底（高度 0）。每隔 3 画一圈（1、4、7、…、16），越往外越高。\ny 方向的圈挤得密：坡陡；x 方向的圈隔得开：坡缓。")
savefig(fig, "ch03_bowl_contours")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("2. 偏导数：站在 (1, 2)，一次只动一个旋钮")
print("只动 x（y 按住在 2 不动）。步子 h 越缩越小，平均变化率靠近 2：")
rows_x = [[h, f(x0 + h, y0), (f(x0 + h, y0) - f0) / h] for h in [1, 0.1, 0.01, 0.001]]
table(["h", "f(1+h, 2)", "(f(1+h, 2) − 13) / h"], rows_x)
print("\n只动 y（x 按住在 1 不动）。平均变化率靠近 12：")
rows_y = [[h, f(x0, y0 + h), (f(x0, y0 + h) - f0) / h] for h in [1, 0.1, 0.01, 0.001]]
table(["h", "f(1, 2+h)", "(f(1, 2+h) − 13) / h"], rows_y)
check("手算 h=0.1：f(1.1, 2) = 1.21 + 12 = 13.21，变化率 2.1", math.isclose(rows_x[1][1], 13.21) and math.isclose(rows_x[1][2], 2.1))
check("手算 h=0.1：f(1, 2.1) = 1 + 3 × 4.41 = 14.23，变化率 12.3", math.isclose(rows_y[1][1], 14.23) and math.isclose(rows_y[1][2], 12.3))
check("手算 h=0.01：变化率 2.01 和 12.03", math.isclose(rows_x[2][2], 2.01, abs_tol=1e-9) and math.isclose(rows_y[2][2], 12.03, abs_tol=1e-9))
check("只动 x 的那一列依次是 3、2.1、2.01、2.001；只动 y 的依次是 15、12.3、12.03、12.003",
      np.allclose([r[2] for r in rows_x], [3, 2.1, 2.01, 2.001]) and np.allclose([r[2] for r in rows_y], [15, 12.3, 12.03, 12.003]))

h = 1e-6
df_dx_num = (f(x0 + h, y0) - f0) / h          # 只动 x，y 当常数
df_dy_num = (f(x0, y0 + h) - f0) / h          # 只动 y，x 当常数
df_dx, df_dy = 2 * x0, 6 * y0                 # 第 1 章的求导规则：∂f/∂x = 2x，∂f/∂y = 6y
print("\n规则（解析）与缩 h（数值，h = 0.000001）对比：")
table(["", "数值", "解析"], [["∂f/∂x", df_dx_num, df_dx], ["∂f/∂y", df_dy_num, df_dy]])
check("偏导数：数值 ≈ 解析，∂f/∂x = 2，∂f/∂y = 12", abs(df_dx_num - 2) < 1e-4 and abs(df_dy_num - 12) < 1e-4
      and df_dx == 2 and df_dy == 12)
check("在 (1, 2) 处 y 方向的坡度是 x 方向的 6 倍", df_dy / df_dx == 6)
print("偏导数跟着位置变：换到 (−1, 1)，∂f/∂x =", 2 * -1.0, "，∂f/∂y =", 6 * 1.0)
check("自测：(−1, 1) 处的两个偏导数是 −2 和 6", 2 * -1.0 == -2 and 6 * 1.0 == 6)

# ---------------------------------------------------------------------------
banner("2b. 画图：figures/ch03_partial_slice.png（把碗切一刀，切口是第 1 章的抛物线）")
fig = plt.figure(figsize=(9.4, 17.2))
gs = fig.add_gridspec(3, 1, height_ratios=[4.9, 3.9, 3.9], hspace=0.50, left=0.12, right=0.95, top=0.915, bottom=0.085)
fig.suptitle("偏导数：把碗切一刀，量切口曲线在这一点的斜率", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)

ax = fig.add_subplot(gs[0])
data_axes(ax, "旋钮 x", "旋钮 y", grid=False)
ax.set_xlim(-4.4, 4.4)
ax.set_ylim(-2.9, 3.1)
ax.set_aspect("equal")
for height in EVEN_RINGS:
    rx, ry = ring_xy(height)
    ax.plot(rx, ry, color=FAINT, lw=1.8)
ax.axhline(y0, color=BLUE, lw=3, ls="--")
ax.axvline(x0, color=GREEN, lw=3, ls="--")
ax.plot(x0, y0, "o", color=ORANGE, ms=12, zorder=5)
ax.text(-4.2, y0 + 0.16, "横切：y 固定在 2，只动 x", color=BLUE, fontsize=FS_SMALL, va="bottom", bbox=WHITE_BOX)
ax.text(x0 + 0.18, -2.6, "竖切：x 固定在 1，只动 y", color=GREEN, fontsize=FS_SMALL, va="bottom", bbox=WHITE_BOX)
ax.text(x0 + 0.2, y0 + 0.18, "(1, 2)：高度 13", color=ORANGE, fontsize=FS_SMALL, va="bottom", bbox=WHITE_BOX)
ax.plot(0, 0, "*", color=INK, ms=13)
ax.text(-0.15, -0.12, "碗底", fontsize=15, color=INK, ha="right", va="top")
panel_title(fig, [ax], "① 站在 (1, 2)：横切只动 x，竖切只动 y")
panel_note(fig, [ax], "灰圈是 3.1 节的等高线（每隔 3 一圈）。沿虚线一刀切下去，看切口长什么样。")

SLICE_YLIM = (0, 40)      # 两张切口图用同一套刻度，斜率才能用眼睛比
for k, (spec, color, label, var, curve, slope, const) in enumerate([
        (gs[1], BLUE, "② 横切的切口：高度 = x² + 12，在 x = 1 处斜率 2", "x", lambda s: s**2 + 3 * y0**2, df_dx, x0),
        (gs[2], GREEN, "③ 竖切的切口：高度 = 1 + 3y²，在 y = 2 处斜率 12", "y", lambda s: x0**2 + 3 * s**2, df_dy, y0)]):
    ax = fig.add_subplot(spec)
    data_axes(ax, f"旋钮 {var}（另一个旋钮按住不动）", "高度 f")
    ss = np.linspace(const - 1.5, const + 1.5, 200)
    ax.plot(ss, curve(ss), color=color, lw=3.5)
    tt = np.linspace(const - 1.0, const + 1.0, 2)
    ax.plot(tt, f0 + slope * (tt - const), color=ORANGE, lw=3)
    ax.plot(const, f0, "o", color=ORANGE, ms=11, zorder=5)
    ax.set_xlim(const - 1.5, const + 1.5)
    ax.set_ylim(*SLICE_YLIM)
    ax.text(const + 0.08, f0 - 2.2, f"{var} = {const:g}，高度 13", color=INK, fontsize=FS_SMALL, va="top")
    if k == 0:
        ax.text(const - 1.4, 30, f"橙色切线的斜率 = ∂f/∂x = {slope:g}", color=ORANGE, fontsize=FS_STEP)
        ax.text(const - 1.4, 24, "x 多 0.1，高度约多 0.2", color=MUTED, fontsize=FS_NOTE)
    else:
        ax.text(const - 1.4, 34, f"橙色切线的斜率 = ∂f/∂y = {slope:g}", color=ORANGE, fontsize=FS_STEP)
        ax.text(const - 1.4, 28, "y 多 0.1，高度约多 1.2", color=MUTED, fontsize=FS_NOTE)
    panel_title(fig, [ax], label)
    if k == 0:
        panel_note(fig, [ax], "切口就是第 1 章的抛物线 x²，只是整体抬高了 12。")
    else:
        panel_note(fig, [ax], "两张切口图刻度相同：同样往右走一小段，③ 的切线爬得快得多。")
savefig(fig, "ch03_partial_slice")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("3. 梯度：把偏导数排成清单 (2, 12)；它指向上坡最陡的方向")
grad = np.array([df_dx, df_dy])
grad_norm = float(np.linalg.norm(grad))
grad_deg = math.degrees(math.atan2(grad[1], grad[0]))
print("梯度 ∇f(1, 2) =", grad, "  —— 第 1 项永远是 x 的偏导，第 2 项永远是 y 的偏导")
print(f"把它当箭头：长度 ‖∇f‖ = sqrt(2² + 12²) = sqrt(148) = {grad_norm:.4f}，方向角 = {grad_deg:.1f}°（由“往右 2、往上 12”反查出来）")
check("梯度 = (2, 12)，长度 sqrt(148) ≈ 12.17", np.array_equal(grad, [2, 12]) and math.isclose(grad_norm, math.sqrt(148)))

print("\n梯度当“价格表”用：两个旋钮一起动，高度变化 ≈ 2 × Δx + 12 × Δy（就是 2.2 节的点积）")
demo_step = np.array([0.03, -0.01])
demo_pred = float(grad @ demo_step)
demo_real = f(x0 + demo_step[0], y0 + demo_step[1]) - f0
print(f"  例：Δx = 0.03，Δy = −0.01 → 预测 2 × 0.03 + 12 × (−0.01) = {demo_pred:.4f}，实际 {demo_real:.4f}")
check("价格表预测 −0.06，实际 −0.0588", math.isclose(demo_pred, -0.06, abs_tol=1e-12) and math.isclose(demo_real, -0.0588, abs_tol=1e-9))
check("字面值重算：1.0609 + 3 × 3.9601 = 12.9412，减 13 得 −0.0588",
      round(1.03 ** 2, 4) == 1.0609 and round(1.99 ** 2, 4) == 3.9601
      and round(1.0609 + 3 * 3.9601, 4) == 12.9412 and round(12.9412 - 13, 4) == -0.0588)

STEP = 0.01
print(f"\n从 (1, 2) 出发，朝各个方向都走同样长的一步 {STEP}：")
print("  手算 0° 方向（只往 x 走）：f(1.01, 2) = 1.0201 + 12 = 13.0201，涨了", round(f(x0 + STEP, y0) - f0, 6))
print("  手算 90° 方向（只往 y 走）：f(1, 2.01) = 1 + 3 × 4.0401 = 13.1203，涨了", round(f(x0, y0 + STEP) - f0, 6))
check("0° 方向涨 0.0201，90° 方向涨 0.1203", math.isclose(f(x0 + STEP, y0) - f0, 0.0201, abs_tol=1e-9)
      and math.isclose(f(x0, y0 + STEP) - f0, 0.1203, abs_tol=1e-9))

rows, rises, steps = [], {}, {}
for deg in range(0, 360, 30):
    th = math.radians(deg)
    step_vec = STEP * np.array([math.cos(th), math.sin(th)])
    rise = f(x0 + step_vec[0], y0 + step_vec[1]) - f0
    rises[deg], steps[deg] = rise, step_vec
    # Δx、Δy 留 5 位小数（读者要拿去代公式）；两列涨幅按正文引用的 4 位小数打印
    rows.append([deg, num(step_vec[0], 5), num(step_vec[1], 5), rise, float(grad @ step_vec)])
table(["方向角°", "这一步 Δx", "这一步 Δy", "f 的实际涨幅", "梯度·这一步"], rows, floatfmt=".4f")
best_deg = max(rises, key=rises.get)
worst_deg = min(rises, key=rises.get)
unit_grad = grad / grad_norm
grad_step = STEP * unit_grad
rise_along_grad = f(x0 + grad_step[0], y0 + grad_step[1]) - f0
print(f"12 个方向里涨得最多的是 {best_deg}°，跌得最多的是 {worst_deg}°；梯度方向 = {grad_deg:.1f}°")
print(f"正好沿梯度方向走 {STEP}：这一步是 ({grad_step[0]:.5f}, {grad_step[1]:.5f})，涨 {rise_along_grad:.4f}，比 12 个候选都多")
check("每一步都一样长（0.01）", all(math.isclose(float(np.linalg.norm(v)), STEP) for v in steps.values())
      and math.isclose(float(np.linalg.norm(grad_step)), STEP))
check("12 个候选里最陡的是 90°：离梯度方向 80.5° 最近的那个", best_deg == 90 and abs(best_deg - grad_deg) < 15)
check("跌得最多的是正对面的 270°", worst_deg == 270)
check("沿梯度方向走，比 12 个候选涨得都多", rise_along_grad > max(rises.values()))
check("正文引用的几个数：梯度方向 80.5°，那一步是 (0.00164, 0.00986)，30° 方向涨 0.0775，沿梯度涨 0.1219（价格表预测 0.01 × 12.17 = 0.1217）",
      round(grad_deg, 1) == 80.5 and round(float(grad_step[0]), 5) == 0.00164 and round(float(grad_step[1]), 5) == 0.00986
      and round(rises[30], 4) == 0.0775 and round(rise_along_grad, 4) == 0.1219 and round(STEP * grad_norm, 4) == 0.1217)
check("“梯度·这一步”和实际涨幅每一行都相差不超过 0.0003", all(abs(r[3] - r[4]) <= 3e-4 + 1e-12 for r in rows))

print("\n“最陡”比较的必须是同样长的一步：")
print(f"  往 x 走 1 格：f(2, 2) − 13 = {f(x0 + 1, y0) - f0:g}；往 y 走 0.01 格：只涨 {f(x0, y0 + STEP) - f0:.4f}。步子不一样长，不能比。")
check("f(2, 2) − f(1, 2) = 3", f(x0 + 1, y0) - f0 == 3)

print("\n箭头越长 = 那里坡越陡，不等于离碗底越远。图 ① 里的四个位置：")
ARROW_POINTS = [(1.0, 2.0), (-1.0, 1.0), (0.5, -1.5), (2.0, -1.0)]        # 都落在画出来的圈上（高度 13、4、7、7）
rows = [[f"({px:g}, {py:g})", f(px, py), math.hypot(px, py), f"({2 * px:g}, {6 * py:g})", math.hypot(2 * px, 6 * py)]
        for px, py in ARROW_POINTS]
table(["位置", "高度", "离碗底多远", "梯度", "梯度的长度"], rows, floatfmt=".2f")
check("(1, 2) 和 (2, −1) 离碗底一样远（2.24），梯度长度却是 12.17 对 7.21",
      math.isclose(rows[0][2], rows[3][2]) and round(rows[0][4], 2) == 12.17 and round(rows[3][4], 2) == 7.21)
check("(0.5, −1.5) 离碗底更近（1.58），梯度反而比 (2, −1) 的长（9.06）", round(rows[2][2], 2) == 1.58 and round(rows[2][4], 2) == 9.06)
check("四个位置都落在每隔 3 一圈的等高线上", all(r[1] in (4, 7, 13) for r in rows))

print("\n进阶：点积 = 长度 × 长度 × cos(夹角)。步子长 1 时，变化率 = ‖∇f‖ × cos(方向与梯度的夹角)：")
rows = []
for deg in (0, 30, 90):
    th = math.radians(deg)
    direction = np.array([math.cos(th), math.sin(th)])
    between = abs(deg - grad_deg)
    rows.append([deg, float(grad @ direction), between, math.cos(math.radians(between)), grad_norm * math.cos(math.radians(between))])
rows.append([round(grad_deg, 1), float(grad @ unit_grad), 0.0, 1.0, grad_norm])
table(["方向角°", "∇f·方向", "与梯度的夹角°", "cos(夹角)", "‖∇f‖ × cos"], rows, floatfmt=".4g")
check("∇f·方向 = ‖∇f‖ × cos(夹角)，每一行都成立", all(math.isclose(r[1], r[4], abs_tol=1e-9) for r in rows))
check("90° 方向：12.17 × cos(9.46°) = 12.17 × 0.986 = 12", math.isclose(rows[2][4], 12.0, abs_tol=1e-9)
      and round(rows[2][2], 2) == 9.46 and round(rows[2][3], 3) == 0.986)
along_ring = np.array([6.0, -1.0]) / math.hypot(6, 1)     # 与 (2, 12) 垂直：2 × 6 + 12 × (−1) = 0
rise_along_ring = f(x0 + STEP * along_ring[0], y0 + STEP * along_ring[1]) - f0
print(f"与梯度垂直的方向 (6, −1)：点积 2 × 6 + 12 × (−1) = {float(grad @ [6, -1]):g}；沿它走 0.01，高度只变 {rise_along_ring:.6f}（沿着等高线走）")
check("垂直于梯度的方向：点积为 0，高度几乎不变（只变 0.000105）", grad @ [6, -1] == 0 and round(rise_along_ring, 6) == 0.000105)

# ---------------------------------------------------------------------------
banner("3b. 画图：figures/ch03_gradient_arrows.png（梯度箭头 + 12 个方向的涨幅）")
fig = plt.figure(figsize=(9.4, 17.4))
gs = fig.add_gridspec(2, 1, height_ratios=[5.6, 7.6], hspace=0.30, left=0.10, right=0.96, top=0.915, bottom=0.075)
fig.suptitle("梯度是一根箭头：指向上坡最陡的方向，垂直穿过等高线", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)

ax = fig.add_subplot(gs[0])
data_axes(ax, "旋钮 x", "旋钮 y", grid=False)
ax.set_xlim(-4.75, 4.75)
ax.set_ylim(-2.85, 3.35)
ax.set_aspect("equal")
for height in (1, 4, 7, 10, 13, 16, 19):
    rx, ry = ring_xy(height)
    ax.plot(rx, ry, color=FAINT, lw=1.8)
    if height > 1:     # 高度交替标在左下、左上两条斜线上，避开四根箭头；最里面那圈让给“碗底”两个字
        ax.text(*ring_point(height, 222 if height in (4, 10, 16) else 160), f"{height}", fontsize=15, color=MUTED,
                ha="center", va="center", bbox=WHITE_BOX)
ARROW_SCALE = 0.08
LABEL_AT = {(1.0, 2.0): (0.18, -0.12, "left"), (-1.0, 1.0): (-0.14, 0.12, "right"),
            (0.5, -1.5): (0.20, -0.10, "left"), (2.0, -1.0): (0.22, -0.12, "left")}
for px, py in ARROW_POINTS:
    tx, ty, ha = LABEL_AT[(px, py)]
    g = np.array([2 * px, 6 * py])
    here = (px, py) == (x0, y0)
    arrow(ax, (px, py), (px + ARROW_SCALE * g[0], py + ARROW_SCALE * g[1]), ORANGE if here else BLUE, lw=4 if here else 3)
    ax.plot(px, py, "o", color=INK, ms=7, zorder=6)
    ax.text(px + ARROW_SCALE * g[0] + tx, py + ARROW_SCALE * g[1] + ty, f"梯度 ({g[0]:g}, {g[1]:g})".replace("-", "−"),
            fontsize=FS_SMALL, color=ORANGE if here else BLUE, ha=ha, va="center", bbox=WHITE_BOX)
ax.text(x0 + 0.14, y0 - 0.10, "站在 (1, 2)", fontsize=15, color=INK, va="top", bbox=WHITE_BOX)
ax.plot(0, 0, "*", color=INK, ms=14)
ax.text(0.0, -0.20, "碗底：梯度 (0, 0)", fontsize=15, color=INK, ha="center", va="top", bbox=WHITE_BOX)
panel_title(fig, [ax], "① 碗上几个位置的梯度箭头（按 0.08 倍缩小画出）")
panel_note(fig, [ax], "灰圈每隔 3 一圈。每根箭头都垂直穿过脚下那一圈，指向更高的一侧；\n坡越陡的地方，箭头越长。")

ax = fig.add_subplot(gs[1])
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-1.75, 2.15)
ax.set_aspect("equal")
ax.axis("off")
ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, ls="--", lw=1.6, color=FAINT))
for deg, rise in rises.items():
    th = math.radians(deg)
    tip = np.array([math.cos(th), math.sin(th)])
    color = ORANGE if rise > 0 else BLUE
    arrow(ax, (0, 0), tuple(tip), color, lw=2.2 + 22 * abs(rise))
    lx, ly = 1.36 * tip[0], 1.36 * tip[1]
    if deg == 90:                      # 往左让一点，给梯度方向的引线留出路
        lx = -0.14
    ax.text(lx, ly, f"{deg}°\n{rise:+.4f}".replace("-", "−"), fontsize=FS_TICK, color=color, ha="center", va="center",
            fontweight="bold" if deg in (best_deg, worst_deg) else "normal", linespacing=1.25)
arrow(ax, (0, 0), tuple(unit_grad), INK, lw=3.4, zorder=7)      # 和另外 12 根一样长：箭头尖也落在虚线圈上
ax.annotate(f"梯度方向 {grad_deg:.1f}°：{rise_along_grad:+.4f}", xy=tuple(1.03 * unit_grad), xytext=(0.42, 1.86),
            fontsize=FS_SMALL, color=INK, fontweight="bold", ha="left", va="center",
            arrowprops=dict(arrowstyle="-", lw=1.3, color=INK, shrinkA=2, shrinkB=0))
ax.plot(0, 0, "o", color=INK, ms=9, zorder=8)
panel_title(fig, [ax], "② 在 (1, 2) 朝 12 个方向各走 0.01：高度涨了多少")
panel_note(fig, [ax], "13 根箭头一样长，尖都落在虚线圈上：每一步都是 0.01。\n橙 = 上坡，蓝 = 下坡，越粗变化越大；黑色那根是梯度方向。")
savefig(fig, "ch03_gradient_arrows")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("4. 梯度下降：朝下坡方向走一小步，重复；步子太大反而上升")
print("先看一个旋钮：L(w) = (w − 3)²，碗底在 w = 3。学习率 α = 0.1，从 w = 1 出发：")


def L1(w):
    return (w - 3) ** 2


def dL1(w):
    return 2 * (w - 3)          # 链式法则：外层平方 → 2(w − 3)，内层 (w − 3) 对 w 的导数是 1


w, rows, moves = 1.0, [], []
for k in range(5):
    rows.append([k, w, dL1(w), L1(w)])
    moves.append(-0.1 * dL1(w))
    w = w - 0.1 * dL1(w)        # θ ← θ − α∇L 的一维版本
table(["第几步", "w", "导数 2(w − 3)", "损失 (w − 3)²"], [[r[0], *[num(v) for v in r[1:]]] for r in rows])
print("每一步实际挪了多远：" + " → ".join(num(m) for m in moves[:4]) + "（α 一直是 0.1，是坡变缓了）")
check("一步：导数 −4，w 从 1 到 1.4，损失从 4 降到 2.56", rows[0][2] == -4 and math.isclose(rows[1][1], 1.4)
      and rows[0][3] == 4 and math.isclose(rows[1][3], 2.56))
check("自测：再走一步，w = 1.72，损失 1.6384", math.isclose(rows[2][1], 1.72) and math.isclose(rows[2][3], 1.6384))
check("步子自己在变小：0.4 → 0.32 → 0.256", np.allclose(moves[:3], [0.4, 0.32, 0.256]))

print("\n再看两个旋钮：站在 (1, 2)，梯度 (2, 12)，α = 0.01")
alpha_small = 0.01
step_small = -alpha_small * grad
new_small = np.array([x0, y0]) + step_small
f_small = f(*new_small)
pred_small = float(grad @ step_small)                 # 价格表预测：∇f·Δθ = −α‖∇f‖²
step_small_len = float(np.linalg.norm(step_small))
print(f"  这一步 Δθ = −0.01 × (2, 12) = {step_small}；新位置 = {new_small}")
print(f"  这一步的长度 = 0.01 × {grad_norm:.2f} = {step_small_len:.4f}，是 3.3 节那一小步（0.01）的 {step_small_len / STEP:.1f} 倍")
print(f"  手算新高度：0.98² + 3 × 1.88² = 0.9604 + 10.6032 = {0.98**2 + 3 * 1.88**2:.4f}；机器：{f_small:.4f}")
print(f"  预测变化：2 × (−0.02) + 12 × (−0.12) = {pred_small:.2f}  （= −α × ‖∇f‖² = −0.01 × 148）")
print(f"  实际变化：{f_small:.4f} − 13 = {f_small - f0:.4f}")
new_grad = np.array([2 * new_small[0], 6 * new_small[1]])
print(f"  为什么差一点：到了新位置，梯度已经从 (2, 12) 变成 {new_grad}，坡变缓了")
check("新位置 (0.98, 1.88)，新高度 11.5636", np.allclose(new_small, [0.98, 1.88]) and math.isclose(f_small, 11.5636, abs_tol=1e-9))
check("α 是倍数不是步长：这一步长约 0.12，是 3.3 节那一小步的约 12 倍", round(step_small_len, 2) == 0.12 and round(step_small_len / STEP) == 12)
check("预测下降 1.48 = 0.01 × (2² + 12²)", math.isclose(-pred_small, 1.48, abs_tol=1e-12) and math.isclose(alpha_small * grad_norm**2, 1.48))
check("实际下降 1.4364：接近 1.48 但不相等", math.isclose(f0 - f_small, 1.4364, abs_tol=1e-9))
check("新位置的梯度 (1.96, 11.28) 比出发时小", np.allclose(new_grad, [1.96, 11.28]))

print("\n同一个方向（负梯度），只换 α：")
rows = []
for alpha in [0.001, 0.01, 0.1, 1.0]:
    pos = np.array([x0, y0]) - alpha * grad
    rows.append([alpha, f"({pos[0]:g}, {pos[1]:g})", -alpha * grad_norm**2, f(*pos) - f0, f(*pos)])
table(["学习率 α", "新位置", "预测变化 −α‖∇f‖²", "实际变化", "新高度"], [[f"{r[0]:g}", r[1], *[num(v) for v in r[2:]]] for r in rows])
check("步子越小，预测越准：α = 0.001 时两者相差不到 0.001", abs(rows[0][2] - rows[0][3]) < 1e-3)
check("α = 0.1：预测能降 14.8（比总高度 13 还多），实际降 10.44", math.isclose(rows[2][2], -14.8) and math.isclose(rows[2][3], -10.44))
check("α = 1：跳到 (−1, −10)，高度 301——方向没错，这一步却变差了", rows[3][1] == "(-1, -10)" and rows[3][4] == 301)

print("\nα = 0.01 从 (1, 2) 出发连走 300 步：")
path = [np.array([x0, y0])]
for _ in range(300):
    path.append(path[-1] - alpha_small * np.array([2 * path[-1][0], 6 * path[-1][1]]))
path = np.array(path)
path_heights = f(path[:, 0], path[:, 1])
print("  路上的高度：" + "，".join(f"第 {k} 步 {num(path_heights[k])}" for k in (0, 1, 10, 30, 100)) + f"，第 300 步 {path_heights[300]:.1e}")
check("连走 300 步：每一步都比上一步低，最后高度不到 0.0001", bool(np.all(np.diff(path_heights) < 0)) and path_heights[-1] < 1e-4)
check("图上标的高度：第 10 步 4.15，第 30 步 0.59，第 100 步 0.018",
      round(float(path_heights[10]), 2) == 4.15 and round(float(path_heights[30]), 2) == 0.59 and round(float(path_heights[100]), 3) == 0.018)

print("\n梯度为零 ≠ 走到了最低点。三个函数在 0 处导数都是 0：")
rows = [["x²", 2 * 0.0, (-0.1) ** 2, 0.1**2, "两边都更高 → 最低点"],
        ["−x²", -2 * 0.0 + 0.0, -((-0.1) ** 2), -(0.1**2), "两边都更低 → 最高点"],
        ["x³", 3 * 0.0**2, (-0.1) ** 3, 0.1**3, "一边低一边高 → 都不是"]]
table(["函数", "0 处的导数", "左边 f(−0.1)", "右边 f(0.1)", "0 是什么"], rows)
check("x²、−x²、x³ 在 0 处导数都为 0，却是三种不同的情形",
      all(r[1] == 0 for r in rows) and rows[0][2] > 0 < rows[0][3] and rows[1][2] < 0 > rows[1][3] and rows[2][2] < 0 < rows[2][3])

# ---------------------------------------------------------------------------
banner("4b. 画图：figures/ch03_descent_path.png（小步一路下坡；大步跨到对面）与 figures/ch03_overview.png（3.0 节总览）")
big_jump = np.array([x0, y0]) - 1.0 * grad

fig = plt.figure(figsize=(9.4, 17.0))
gs = fig.add_gridspec(2, 1, height_ratios=[8.4, 3.9], hspace=0.34, left=0.12, right=0.95, top=0.915, bottom=0.095)
fig.suptitle("沿负梯度走小步，一步比一步低；步子太大，一步跨到对面", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)

ax = fig.add_subplot(gs[0])
data_axes(ax, "旋钮 x", "旋钮 y", grid=False)
ax.set_xlim(-0.75, 1.85)
ax.set_ylim(-0.45, 2.45)
ax.set_aspect("equal")
for height in range(1, 14, 2):         # 每隔 2 画一圈：1、3、5、…、13
    rx, ry = ring_xy(height)
    ax.plot(rx, ry, color=FAINT, lw=1.6)
    if height in (1, 5, 9, 13):
        ax.text(*ring_point(height, 98.6), f"{height}", fontsize=15, color=MUTED, ha="center", va="center", bbox=WHITE_BOX)
ax.plot(path[:, 0], path[:, 1], color=BLUE, lw=3, zorder=3)
arrow(ax, tuple(path[0]), tuple(path[1]), ORANGE, lw=4, zorder=6)
for k, (dx, dy, text) in {0: (0.06, 0.07, "出发 (1, 2)：高度 13"),
                          1: (0.08, -0.10, f"第 1 步 ({path[1][0]:g}, {path[1][1]:g})\n高度 {path_heights[1]:.4f}"),
                          10: (0.08, 0.0, f"第 10 步：高度 {path_heights[10]:.2f}"),
                          30: (0.08, 0.02, f"第 30 步：高度 {path_heights[30]:.2f}")}.items():
    ax.plot(*path[k], "o", color=ORANGE if k <= 1 else BLUE, ms=9, zorder=7)
    ax.text(path[k][0] + dx, path[k][1] + dy, text, fontsize=FS_SMALL, color=ORANGE if k <= 1 else BLUE,
            va="center", bbox=WHITE_BOX, zorder=8)
ax.plot(*path[100], "o", color=BLUE, ms=9, zorder=7)
ax.annotate(f"第 100 步：高度 {path_heights[100]:.3f}", xy=tuple(path[100]), xytext=(0.42, -0.27), fontsize=FS_SMALL, color=BLUE,
            va="center", arrowprops=dict(arrowstyle="-", lw=1.3, color=BLUE, shrinkA=2, shrinkB=5), zorder=8)   # 字放到路径下面，用引线连回去
ax.plot(0, 0, "*", color=INK, ms=15, zorder=6)
ax.text(-0.06, -0.10, "碗底", fontsize=FS_SMALL, color=INK, ha="right", va="top")
panel_title(fig, [ax], "① α = 0.01：从 (1, 2) 出发连走 300 步")
panel_note(fig, [ax], "灰圈每隔 2 一圈。每一步都垂直穿过等高线往里走：\n先顺着陡的 y 方向快速下降，再沿着缓的 x 方向慢慢滑向碗底。")

ax = fig.add_subplot(gs[1])
data_axes(ax, "旋钮 x（注意：这张图的范围比上图大得多）", "旋钮 y", grid=False)
ax.set_xlim(-18.5, 18.5)
ax.set_ylim(-11.8, 4.2)
ax.set_aspect("equal")
ax.set_anchor("N")
ax.set_yticks([-10, -5, 0])
for height in (75, 150, 225, f(*big_jump)):      # 灰圈每隔 75 一圈；最外面橙色那圈是落点所在的高度
    landing = height == f(*big_jump)
    rx, ry = ring_xy(height)
    ax.plot(rx, ry, color=ORANGE if landing else FAINT, lw=2.4 if landing else 1.6)
    ax.text(*ring_point(height, -25.8), f"{height:g}", fontsize=15, color=ORANGE if landing else MUTED, ha="center",
            va="center", bbox=WHITE_BOX)
arrow(ax, (x0, y0), tuple(big_jump), ORANGE, lw=4, zorder=6)
ax.plot(x0, y0, "o", color=ORANGE, ms=9, zorder=7)
ax.plot(*big_jump, "o", color=ORANGE, ms=9, zorder=7)
ax.plot(0, 0, "*", color=INK, ms=15, zorder=6)
ax.text(x0 + 0.8, y0 + 0.2, "出发 (1, 2)：高度 13", fontsize=FS_SMALL, color=ORANGE, va="bottom", bbox=WHITE_BOX)
ax.text(big_jump[0] + 0.9, big_jump[1] + 0.5, f"落点 (−1, −10)：高度 {f(*big_jump):g}", fontsize=FS_SMALL, color=ORANGE,
        va="bottom", bbox=WHITE_BOX)
panel_title(fig, [ax], "② α = 1：同一个方向，一步就跨过了碗底")
panel_note(fig, [ax], "方向没错（出发时确实朝着下坡），可这一步长到落在对面\n高度 301 的坡上。灰圈每隔 75 一圈，橙圈是 301。")
savefig(fig, "ch03_descent_path")
plt.close(fig)

fig, axes = lesson_figure(4, "旋钮再多也是这四步：量高度、逐个试、排清单、走一小步", panel_height=3.25, width=9.4)
HOT = dict(facecolor="#fff0d9", edgecolor=ORANGE, color=ORANGE)
ax = axes[0]
lesson_panel(ax, "① 两个旋钮 → 一个数（3.1 节）")
lesson_cells(ax, [[x0, y0]], 0.4, 1.55, 1.2, 0.95)
ax.text(1.0, 2.68, "旋钮 x", ha="center", fontsize=FS_SMALL, color=MUTED)
ax.text(2.2, 2.68, "旋钮 y", ha="center", fontsize=FS_SMALL, color=MUTED)
arrow(ax, (3.05, 2.02), (3.95, 2.02), MUTED, lw=2.5)
cell(ax, 4.2, 1.55, f"{f0:g}", width=1.4, height=0.95, **HOT)
ax.text(4.9, 2.68, "有多差", ha="center", fontsize=FS_SMALL, color=MUTED)
hand(ax, 6.05, 2.02, f"1² + 3 × 2² = {f0:g}")
note(ax, 0.4, 0.55, "旋钮叫参数；“有多差”这个数叫损失，越小越好。")
ax = axes[1]
lesson_panel(ax, "② 一次只拧一个旋钮（3.2 节）")
lesson_cells(ax, [[x0 + STEP, y0]], 0.4, 2.12, 1.2, 0.78, highlights=((0, 0),))
hand(ax, 3.05, 2.51, f"高度 {f0:g} → {f(x0 + STEP, y0):g}：x 的偏导数约 {df_dx:g}")
lesson_cells(ax, [[x0, y0 + STEP]], 0.4, 1.12, 1.2, 0.78, highlights=((0, 1),))
hand(ax, 3.05, 1.51, f"高度 {f0:g} → {f(x0, y0 + STEP):g}：y 的偏导数约 {df_dy:g}", color=GREEN)
note(ax, 0.4, 0.45, "橙色格子是多拧了 0.01 的那个旋钮，另一个按住不动。")
ax = axes[2]
lesson_panel(ax, "③ 每个旋钮一个偏导数，排成清单（3.3 节）")
lesson_cells(ax, [[df_dx, df_dy]], 0.4, 1.55, 1.7, 0.95)
ax.text(1.25, 2.68, "x 的偏导数", ha="center", fontsize=FS_SMALL, color=MUTED)
ax.text(2.95, 2.68, "y 的偏导数", ha="center", fontsize=FS_SMALL, color=MUTED)
hand(ax, 4.3, 2.02, f"这张清单叫梯度：({df_dx:g}, {df_dy:g})")
note(ax, 0.4, 0.55, "把清单当箭头看：它指向上坡最陡的方向。")
ax = axes[3]
lesson_panel(ax, "④ 朝反方向走一小步，再从 ① 来（3.4 节）")
hand(ax, 0.4, 2.62, f"(1, 2) − {alpha_small:g} × ({df_dx:g}, {df_dy:g}) =")
lesson_cells(ax, [[float(new_small[0]), float(new_small[1])]], 0.4, 1.12, 1.2, 0.95)
arrow(ax, (3.05, 1.59), (3.95, 1.59), MUTED, lw=2.5)
cell(ax, 4.2, 1.12, f"{f_small:.4f}", width=1.9, height=0.95, **HOT)
hand(ax, 6.45, 1.59, f"比 {f0:g} 低了", color=ORANGE)
note(ax, 0.4, 0.40, f"{alpha_small:g} 是学习率：乘在梯度前面的倍数。重复很多次，就是梯度下降。")
savefig(fig, "ch03_overview")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("5. 找一条直线：先用 3 个点手算一步，再让 50 个点的循环自己跑")
print("迷你版：3 个点 (0, 1)、(1, 4)、(2, 7)（教学构造：都在 y = 3x + 1 上，假装不知道）。从 w = 0、b = 0 出发：")
x3 = np.array([0.0, 1.0, 2.0])
y3 = np.array([1.0, 4.0, 7.0])


def mse(w, b, xs, ys):
    return float(np.mean((w * xs + b - ys) ** 2))             # 均方误差：误差平方的平均


def mse_grad(w, b, xs, ys):
    err = w * xs + b - ys                                     # 每个点的误差 e = 预测 − 真值
    return float(np.mean(2 * err * xs)), float(np.mean(2 * err))   # ∂L/∂w，∂L/∂b


def point_rows(w, b):
    pred = w * x3 + b
    err = pred - y3
    return [[x3[i], y3[i], pred[i], err[i], err[i] ** 2, 2 * err[i] * x3[i] + 0.0, 2 * err[i]] for i in range(3)]


table(["x", "真值 y", "预测 ŷ", "误差 e = ŷ − y", "e²", "2·e·x", "2·e"], point_rows(0.0, 0.0))
L_before = mse(0.0, 0.0, x3, y3)
dw3, db3 = mse_grad(0.0, 0.0, x3, y3)
print(f"损失 L = (1 + 16 + 49) / 3 = {L_before:g}")
print(f"∂L/∂w = (0 − 8 − 28) / 3 = {dw3:g}      ∂L/∂b = (−2 − 8 − 14) / 3 = {db3:g}")
check("3 个点：损失 22，∂L/∂w = −12，∂L/∂b = −8", L_before == 22 and dw3 == -12 and db3 == -8)

print("先不信公式，用“缩 h”量一遍（和 3.2 节同一个办法）：")
L_w = mse(0.01, 0.0, x3, y3)          # 只把 w 从 0 拧到 0.01：三个预测变成 0、0.01、0.02
L_b = mse(0.0, 0.01, x3, y3)          # 只把 b 从 0 拧到 0.01：三个预测都变成 0.01
print(f"  只动 w：损失 (1 + 15.9201 + 48.7204) / 3 = {L_w:.4f}，变化率 ({L_w:.4f} − 22) / 0.01 = {(L_w - L_before) / 0.01:.2f}，靠近 −12")
print(f"  只动 b：损失 (0.9801 + 15.9201 + 48.8601) / 3 = {L_b:.4f}，变化率 ({L_b:.4f} − 22) / 0.01 = {(L_b - L_before) / 0.01:.2f}，靠近 −8")
check("缩 h 手算：只动 w 损失 21.8802、变化率 −11.98；只动 b 损失 21.9201、变化率 −7.99",
      round(L_w, 4) == 21.8802 and round((L_w - L_before) / 0.01, 2) == -11.98
      and round(L_b, 4) == 21.9201 and round((L_b - L_before) / 0.01, 2) == -7.99)
h = 1e-6
dw3_num = (mse(h, 0.0, x3, y3) - L_before) / h
db3_num = (mse(0.0, h, x3, y3) - L_before) / h
print(f"  把 h 缩到 0.000001：{dw3_num:.4f}、{db3_num:.4f}")
check("链式法则推出的两个偏导数 = 缩 h 的数值结果", abs(dw3_num - dw3) < 1e-4 and abs(db3_num - db3) < 1e-4)

w3, b3 = 0.0 - 0.1 * dw3, 0.0 - 0.1 * db3
L_after = mse(w3, b3, x3, y3)
print(f"α = 0.1 更新一步：w = 0 − 0.1 × ({dw3:g}) = {w3:g}，b = 0 − 0.1 × ({db3:g}) = {b3:g}")
table(["x", "真值 y", "预测 ŷ", "误差 e = ŷ − y", "e²", "2·e·x", "2·e"], point_rows(w3, b3))
print(f"新损失 L = (0.04 + 4 + 14.44) / 3 = {L_after:g}")
check("更新到 w = 1.2、b = 0.8，损失 22 → 6.16", math.isclose(w3, 1.2) and math.isclose(b3, 0.8) and math.isclose(L_after, 6.16))
print(f"自测：按 3.4 节估算，这一步预测下降 0.1 × (12² + 8²) = {0.1 * (dw3**2 + db3**2):g}，实际下降 {L_before - L_after:g}")
check("预测下降 20.8，实际下降 15.84", math.isclose(0.1 * (dw3**2 + db3**2), 20.8) and math.isclose(L_before - L_after, 15.84))
w, b = 0.0, 0.0
for _ in range(200):
    dw, db = mse_grad(w, b, x3, y3)
    w, b = w - 0.1 * dw, b - 0.1 * db
print(f"同样的循环走 200 轮：w = {w:.4f}，b = {b:.4f}，损失 {mse(w, b, x3, y3):.2e}")
check("3 个点的循环找回了藏着的直线 y = 3x + 1", abs(w - 3) < 1e-3 and abs(b - 1) < 1e-3)

print("\n正式版：50 个点，大致落在 y = 2x + 1 附近（加了噪声）。旋钮还是 w、b，从 0 出发：")
# 下面两行是“改一改”第 1、2 条要改的地方。第 5、8 章的实验用同样的种子和写法，得到同一份数据。
ALPHA_LINE = 0.1  # TWEAK-1: 0.05
X_RANGE = (-1, 1)  # TWEAK-2: (-10, 10)
rng = np.random.default_rng(0)
X = rng.uniform(*X_RANGE, size=50)
Y = 2.0 * X + 1.0 + rng.normal(0, 0.1, size=50)

SNAPSHOTS = (0, 1, 2, 5, 10, 20, 50, 100, 200)
alpha = ALPHA_LINE
w, b = 0.0, 0.0
history, trail, rows, snapshots = [], [], [], {}
first_dw = None
for it in range(201):
    L = mse(w, b, X, Y)
    history.append(L)
    trail.append((w, b))
    if it in SNAPSHOTS:
        rows.append([it, w, b, L])
        snapshots[it] = (w, b, L)
    pred = w * X + b
    err = pred - Y
    dw = np.mean(2 * err * X)                   # ∂L/∂w
    db = np.mean(2 * err)                       # ∂L/∂b
    if first_dw is None:
        first_dw = float(dw)
    w -= alpha * dw                             # θ ← θ − α∇L
    b -= alpha * db
table(["迭代", "w", "b", "损失 L"], rows)      # 第 5、8 章引用了这张表里的 2.02617、1.00192 等数，精度不要动
w_best, b_best = np.polyfit(X, Y, 1)           # 这批数据上“最好的直线”（最小二乘的精确解），只用来核对
L_best = mse(w_best, b_best, X, Y)
print(f"核对：这 50 个点上最好的直线是 w = {w_best:.5f}、b = {b_best:.5f}，它的损失 {L_best:.8f} 就是压不掉的噪声")
check("学到 w ≈ 2、b ≈ 1（容差 0.05）", abs(w - 2.0) < 0.05 and abs(b - 1.0) < 0.05)
check("损失单调下降", all(history[i] >= history[i + 1] - 1e-12 for i in range(len(history) - 1)))
check("梯度下降走到的就是最好的那条直线（w、b 相差不到 0.01）", abs(w - w_best) < 1e-2 and abs(b - b_best) < 1e-2)
check("剩下的损失 ≈ 噪声的大小 0.1² = 0.01，压不掉", 0.005 < history[-1] < 0.015 and abs(history[-1] - L_best) < 1e-4)
check("出发时的损失接近理论值 4/3 + 1 + 0.01 ≈ 2.34（50 个点的抽样有出入）", abs(history[0] - (4 / 3 + 1 + 0.01)) < 0.5)
check("第 0 步的 ∂L/∂w 为负，所以第 1 步 w 变大", first_dw < 0 and math.isclose(rows[1][1], -ALPHA_LINE * first_dw))
check("正文引用的数：200 步后 w = 2.03、b = 1.00；损失 2.64 → 第 50 步 0.011 → 0.0099；最好的直线损失 0.00993373；第 1 步 w = 0.1496",
      round(w, 2) == 2.03 and round(b, 2) == 1.00 and round(history[0], 2) == 2.64 and round(history[50], 3) == 0.011
      and round(history[-1], 4) == 0.0099 and round(L_best, 8) == 0.00993373 and round(rows[1][1], 4) == 0.1496)

# 同一个均方误差，把平方展开成 w、b 的式子，才能一次算出整张“碗”的高度（画 5b 的图 ② 用）
m_xx, m_x, m_xy, m_y, m_yy = np.mean(X * X), np.mean(X), np.mean(X * Y), np.mean(Y), np.mean(Y * Y)


def loss_surface(w, b):
    return m_xx * w**2 + 2 * m_x * w * b + b**2 - 2 * m_xy * w - 2 * m_y * b + m_yy


check("展开后的式子和逐点算的均方误差是同一个数", math.isclose(loss_surface(0.0, 0.0), history[0])
      and math.isclose(loss_surface(1.2, 0.8), mse(1.2, 0.8, X, Y)))

# ---------------------------------------------------------------------------
banner("5b. 画图：figures/ch03_line_fit.png（直线贴近数据点 + (w, b) 碗上的路径 + 损失曲线）")
line_diverged = not (np.all(np.isfinite(history)) and np.all(np.isfinite(trail))) or history[-1] > history[0]
if line_diverged:
    print("  损失发散了（数字大到画不出来），这张图跳过。看上面九行表和第 6 节打印的“门槛”。")
else:
    trail_arr = np.array(trail)
    STEP_COLORS = [(0, MUTED), (5, BLUE), (20, GREEN), (200, ORANGE)]
    fig = plt.figure(figsize=(9.4, 21.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[5.4, 4.5, 3.8], hspace=0.46, left=0.13, right=0.95, top=0.93, bottom=0.075)
    fig.suptitle("找直线：直线贴近数据，就是在 (w, b) 的碗里走向碗底", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.988)

    ax = fig.add_subplot(gs[0])
    data_axes(ax, "数据 x", "数据 y")
    ax.plot(X, Y, "o", color=MUTED, ms=7, alpha=0.75, zorder=3)
    xs = np.array([-1.0, 1.0])
    for it, color in STEP_COLORS:
        lw_, lb_, _ = snapshots[it]
        ax.plot(xs, lw_ * xs + lb_, color=FAINT if it == 0 else color, lw=4 if it == 200 else 2.5, zorder=4)
        ax.text(1.03, lw_ * 1.0 + lb_, f"第 {it} 步", fontsize=FS_SMALL, color=color, va="center",
                fontweight="bold" if it == 200 else "normal")
    ax.set_xlim(-1.08, 1.38)
    ax.set_ylim(-1.6, 3.6)
    w200, b200, L200 = snapshots[200]
    ax.text(-1.02, 3.2, f"第 200 步：w = {w200:.2f}，b = {b200:.2f}", fontsize=FS_STEP, color=ORANGE, va="center")
    ax.text(-1.02, 2.72, "第 0 步：w = 0，b = 0（一条平线）", fontsize=FS_NOTE, color=MUTED, va="center")
    panel_title(fig, [ax], "① 这张画的是数据：50 个点，和第 0、5、20、200 步时的直线")
    panel_note(fig, [ax], "灰点是数据（已知、不能拧）。直线从平躺的第 0 步出发，越来越贴近这群点。")

    ax = fig.add_subplot(gs[1])
    data_axes(ax, "旋钮 w（斜率）", "旋钮 b（截距）", grid=False)
    ax.set_xlim(-0.45, 2.85)
    ax.set_ylim(-0.35, 1.55)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
    ax.set_yticks([0, 0.5, 1, 1.5])
    ww, bb = np.meshgrid(np.linspace(-0.45, 2.85, 300), np.linspace(-0.35, 1.55, 200))
    ax.contour(ww, bb, loss_surface(ww, bb), levels=np.arange(0.5, 8.1, 0.5), colors=FAINT, linewidths=1.6)
    for level in (0.5, 1.0, 1.5):      # 把高度标在 w = 2 那条竖线和等高线的下交点上：那里没有路径
        const = m_xx * 4 - 4 * m_xy + m_yy - level
        b_at = -(2 * m_x - m_y) - math.sqrt((2 * m_x - m_y) ** 2 - const)
        ax.text(2.0, b_at, f"{level:g}", fontsize=15, color=MUTED, ha="center", va="center", bbox=WHITE_BOX)
    ax.plot(trail_arr[:, 0], trail_arr[:, 1], color=BLUE, lw=3, zorder=3)
    LABELS = {0: (0.08, -0.13, "left"), 5: (-0.06, 0.15, "right"), 20: (0.0, 0.17, "center"), 200: (-0.05, -0.19, "right")}
    for it, color in STEP_COLORS:
        wi, bi, Li = snapshots[it]
        dx, dy, ha = LABELS[it]
        ax.plot(wi, bi, "o", color=color, ms=11, zorder=6)
        ax.text(wi + dx, bi + dy, f"第 {it} 步：损失 {Li:.2g}" if it else f"第 0 步：损失 {Li:.2f}", fontsize=FS_SMALL, color=color,
                ha=ha, va="center", bbox=WHITE_BOX, zorder=7)
    ax.plot(w_best, b_best, "*", color=INK, ms=24, zorder=5)      # 碗底；第 200 步的橙点就压在它上面
    panel_title(fig, [ax], "② 这张画的才是碗：桌面是两个旋钮 (w, b)，高度是损失 L")
    panel_note(fig, [ax], "灰圈：损失的等高线，每隔 0.5 一圈；★ 是碗底；圆点与 ① 里的直线同色。\n先顺着陡的 b 方向快走，再沿缓的 w 方向慢慢滑向碗底。")

    ax = fig.add_subplot(gs[2])
    data_axes(ax, "迭代次数（走了几步）", "损失 L（对数刻度）")
    ax.plot(history, color=BLUE, lw=3.5)
    ax.set_yscale("log")
    ax.set_ylim(0.004, 6)
    ax.set_xlim(-6, 206)
    ax.set_yticks([0.01, 0.1, 1])
    ax.set_yticklabels(["0.01", "0.1", "1"])
    ax.minorticks_off()
    ax.plot(0, history[0], "o", color=ORANGE, ms=10, zorder=5)
    ax.text(6, history[0], f"出发：{history[0]:.2f}", fontsize=FS_SMALL, color=ORANGE, va="center")
    ax.plot(50, history[50], "o", color=ORANGE, ms=10, zorder=5)
    ax.text(54, history[50] * 1.9, f"第 50 步：{history[50]:.3f}", fontsize=FS_SMALL, color=ORANGE, va="center")
    ax.axhline(history[-1], color=ORANGE, ls="--", lw=1.8)
    ax.text(203, history[-1] * 0.62, f"第 200 步：{history[-1]:.4f}——数据自带的噪声，压不掉", fontsize=FS_SMALL, color=ORANGE,
            va="center", ha="right")
    panel_title(fig, [ax], "③ 损失随迭代下降，最后停在噪声那一层")
    panel_note(fig, [ax], "纵轴是对数刻度：每往下一格（1 → 0.1 → 0.01），损失缩小到十分之一。")
    savefig(fig, "ch03_line_fit")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("6. 学习率：同一只碗、同一个起点，α 决定是靠近、来回跳，还是越跳越远")
print("一个旋钮的碗 L(w) = (w − 3)²，都从 w = 1 出发，只换 α：")
ALPHAS_1D = [0.1, 0.9, 1.0, 1.5]
trajectories = {}
for alpha in ALPHAS_1D:
    w, ws = 1.0, []
    for _ in range(5):
        ws.append(w)
        w = w - alpha * dL1(w)
    trajectories[alpha] = ws
table(["第几步", *[f"α={a:g} 的 w" for a in ALPHAS_1D]],
      [[k, *[trajectories[a][k] for a in ALPHAS_1D]] for k in range(5)])
print("\n换成“离碗底还有多远”（w − 3）看，规律就出来了——每一列都是上一行乘同一个数：")
table(["第几步", *[f"α={a:g}" for a in ALPHAS_1D]],
      [[k, *[trajectories[a][k] - 3 for a in ALPHAS_1D]] for k in range(5)])
print("这个数就是 1 − 2α：" + "，".join(f"α = {a:g} 时是 {num(1 - 2 * a)}" for a in ALPHAS_1D))
check("α = 0.1：1 → 1.4 → 1.72 → 1.976，同一侧稳稳靠近", np.allclose(trajectories[0.1][:4], [1, 1.4, 1.72, 1.976]))
check("α = 0.9：1 → 4.6 → 1.72 → 4.024，每步跨到对面但在靠近", np.allclose(trajectories[0.9][:4], [1, 4.6, 1.72, 4.024]))
check("α = 1：1 → 5 → 1 → 5，永远原地对跳", np.allclose(trajectories[1.0], [1, 5, 1, 5, 1]))
check("α = 1.5：1 → 7 → −5 → 19 → −29，损失 4 → 16 → 64 → 256 → 1024（发散）",
      np.allclose(trajectories[1.5], [1, 7, -5, 19, -29]) and np.allclose([L1(w) for w in trajectories[1.5]], [4, 16, 64, 256, 1024]))
check("离碗底的距离每步乘 1 − 2α：0.8、−0.8、−1、−2", all(
    np.allclose((np.array(trajectories[a][1:]) - 3) / (np.array(trajectories[a][:-1]) - 3), 1 - 2 * a) for a in ALPHAS_1D)
    and np.allclose([1 - 2 * a for a in ALPHAS_1D], [0.8, -0.8, -1, -2]))
check("自测：碗换成 5(w − 3)²，每步乘 1 − 10α；α = 0.2 时恰好乘 −1", math.isclose(1 - 10 * 0.2, -1))

steps_fast = math.ceil(math.log(0.01) / math.log(0.8))        # α = 0.1：每步乘 0.8
steps_slow = math.ceil(math.log(0.01) / math.log(0.998))      # α = 0.001：每步乘 0.998
print(f"\n太小也不行：把距离缩到原来的 1%，α = 0.1（每步乘 0.8）要 {steps_fast} 步；α = 0.001（每步乘 0.998）要 {steps_slow} 步")
check("α = 0.1 约 21 步，α = 0.001 约 2300 步", steps_fast == 21 and round(steps_slow, -2) == 2300
      and 0.8**21 <= 0.01 < 0.8**20)

print("\n碗越陡，能容忍的 α 越小。回到 f = x² + 3y²：x 每步乘 1 − 2α，y 每步乘 1 − 6α（3y² 的导数是 6y）：")
rows = []
for alpha in [0.01, 0.3, 0.4, 1.0]:
    pos = np.array([x0, y0])
    for _ in range(3):
        pos = pos - alpha * np.array([2 * pos[0], 6 * pos[1]])
    rows.append([alpha, 1 - 2 * alpha, 1 - 6 * alpha, f"({pos[0]:.4g}, {pos[1]:.4g})", f(*pos)])
table(["学习率 α", "x 每步乘", "y 每步乘", "走 3 步后的位置", "走 3 步后的高度"],
      [[f"{r[0]:g}", num(r[1]), num(r[2]), r[3], num(r[4])] for r in rows])
check("α = 0.01：x 乘 0.98、y 乘 0.94，正是 3.4 节那一步", math.isclose(rows[0][1], 0.98) and math.isclose(rows[0][2], 0.94))
check("α = 0.3：y 来回跳（乘 −0.8）但在缩小，高度下降", rows[1][4] < f0)
check("α = 0.4：x 方向很安全（乘 0.2，三步到 0.008），y 方向乘 −1.4 → 整体发散",
      math.isclose(rows[2][1], 0.2) and math.isclose(rows[2][2], -1.4) and rows[2][3].startswith("(0.008,") and rows[2][4] > f0)
check("门槛：x 方向 α < 1，y 方向 α < 1/3——碗形陡 3 倍，门槛小 3 倍", math.isclose(1 - 2 * 1.0, -1) and math.isclose(1 - 6 * (1 / 3), -1))
check("“陡 3 倍”说的是碗形：同样离碗底 1 格，y 方向的坡 6 是 x 方向的坡 2 的 3 倍；(1, 2) 处量到的 6 倍是因为 y 离碗底远一倍",
      (6 * 1) / (2 * 1) == 3 and df_dy / df_dx == 6 and y0 / x0 == 2)

print("\n回到第 5 节的 50 个点，固定走 50 步，只换 α：")


def run_line(alpha, steps):
    w, b = 0.0, 0.0
    for _ in range(steps):
        dw, db = mse_grad(w, b, X, Y)
        w, b = w - alpha * dw, b - alpha * db
    return mse(w, b, X, Y)


L_start = history[0]
ALPHAS_LINE = [0.01, 0.1, 0.5, 1.0, 1.6]
rows = []
for alpha in ALPHAS_LINE:
    L50 = run_line(alpha, 50)
    if not np.isfinite(L50) or L50 > 1e6:          # nan 和任何数比较都是 False，要单独判
        verdict = "飞出去了（发散）"
    elif L50 > L_start:
        verdict = "比出发时还高：在碗底两侧来回跳"
    elif L50 > 10 * L_best:
        verdict = "太慢，还在半山腰"
    else:
        verdict = "到碗底了"
    rows.append([alpha, L50, verdict])
table(["学习率 α", "50 步后的损失", "怎么样了"], rows, floatfmt=".4g")
hessian = 2 * np.array([[m_xx, m_x], [m_x, 1.0]])       # 这只碗在 w、b 两个方向上有多陡（以及它们怎样互相牵连）
alpha_limit = 2 / np.linalg.eigvalsh(hessian).max()
over = [a for a in ALPHAS_LINE if a > alpha_limit]
print(f"只看 b：b 动 1，每个误差都动 1，∂L/∂b 就动 {hessian[1, 1]:g}——和 (w − 3)² 的导数 2(w − 3) 一样陡，单看它门槛是 {2 / hessian[1, 1]:g}")
print(f"只看 w：w 动 1，∂L/∂w 动 {hessian[0, 0]:.4g}（= 2 × “x² 的平均”），单看它门槛是 {2 / hessian[0, 0]:.4g}")
print(f"w、b 互相牵连，精确的门槛由机器来算：α < {alpha_limit:.4f}。表里超过门槛的 α：" + "、".join(f"{a:g}" for a in over))
L50_at1, L200_at1 = run_line(1.0, 50), run_line(1.0, 200)
if not (np.isfinite(L50_at1) and np.isfinite(L200_at1)):
    trend = "早就飞出去了"
elif L200_at1 > L50_at1 > L_start:
    trend = "比出发时高，而且还在慢慢往上涨：缓慢发散"
else:
    trend = "在下降"
print(f"α = 1：走 50 步损失 {L50_at1:.4g}，走 200 步 {L200_at1:.4g}——{trend}")
check("α = 0.01 太慢；0.1 和 0.5 到了碗底", rows[0][2].startswith("太慢") and rows[1][2] == rows[2][2] == "到碗底了")
check("α = 1：50 步后的损失比出发时还高，200 步后更高（缓慢发散）", L200_at1 > L50_at1 > L_start)
check("α = 1.6：发散，损失超过 10 的 30 次方", rows[4][1] > 1e30)
check("只看 b 方向，这只碗和 (w − 3)² 一样陡（单看门槛 1）；w 方向缓得多（单看门槛约 2.9）；合起来的门槛略小于 1",
      hessian[1, 1] == 2 and round(2 / hessian[0, 0], 1) == 2.9 and 0.95 < alpha_limit < 1.0)
check("正文引用的数：门槛 0.9957；α = 1 走 50 步 3.24、走 200 步 43.5；α = 1.6 打印成 4.457e+34",
      round(alpha_limit, 4) == 0.9957 and round(L50_at1, 2) == 3.24 and round(L200_at1, 1) == 43.5 and f"{rows[4][1]:.4g}" == "4.457e+34")

# ---------------------------------------------------------------------------
banner("6b. 画图：figures/ch03_learning_rate.png（同一只碗上的三条轨迹）")
fig = plt.figure(figsize=(9.4, 17.6))
gs = fig.add_gridspec(3, 1, hspace=0.66, left=0.12, right=0.95, top=0.915, bottom=0.085)
fig.suptitle("同一只碗、同一个起点：学习率太大，就越跳越远", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
# 每个面板：α、颜色、小标题、灰字、横轴范围、纵轴范围、手算那行字的位置、四个“第几步”数字各自的偏移
PANELS = [(0.1, BLUE, "① α = 0.1：每步只挪一小段，稳稳靠近", "离碗底的距离每步 × 0.8：一直在同一侧，一次比一次近。",
           (0.2, 5.8), (-0.9, 8.6), (1.85, 7.5), [(-0.10, 0.30)] * 4),
          (0.9, GREEN, "② α = 0.9：每步都跨过碗底，但一次比一次近", "每步 × (−0.8)：负号 = 跨到对面；0.8 = 距离仍在缩小。",
           (0.2, 5.8), (-0.9, 8.6), (1.85, 7.5), [(0.0, 0.50)] * 4),
          (1.5, ORANGE, "③ α = 1.5：每步都跨过碗底，而且一次比一次远", "每步 × (−2)：距离翻倍，损失 4 → 16 → 64 → 256。这叫发散。",
           (-8.5, 22.5), (-42, 300), (-7.6, 262), [(0.0, -40), (0.0, 17), (0.0, 17), (0.0, 17)])]
for i, (alpha, color, title, caption, xlim, ylim, text_at, offsets) in enumerate(PANELS):
    ax = fig.add_subplot(gs[i])
    data_axes(ax, "旋钮 w", "损失 (w − 3)²")
    span = ylim[1] - ylim[0]
    ws = np.linspace(*xlim, 300)
    ax.plot(ws, L1(ws), color=FAINT, lw=3, zorder=2)
    ax.axvline(3, color=MUTED, ls=":", lw=1.8, zorder=1)
    ax.text(3.0 if i < 2 else 3.5, ylim[0] + 0.03 * span, "碗底 w = 3", fontsize=15, color=MUTED,
            ha="center" if i < 2 else "left", va="bottom", bbox=WHITE_BOX)
    traj = trajectories[alpha][:4]
    for k in range(3):
        arrow(ax, (traj[k], L1(traj[k])), (traj[k + 1], L1(traj[k + 1])), color, lw=2.6, zorder=5)
    for k, (wk, (dx, dy)) in enumerate(zip(traj, offsets)):
        ax.plot(wk, L1(wk), "o", color=color, ms=10, zorder=6)
        ax.text(wk + dx, L1(wk) + dy, f"{k}", fontsize=FS_SMALL, color=color, ha="center", va="bottom", fontweight="bold", zorder=7)
    ax.text(*text_at, "w：" + " → ".join(f"{wk:g}" for wk in traj).replace("-", "−"), fontsize=FS_STEP, color=color,
            va="center", bbox=WHITE_BOX, zorder=8)
    ax.text(text_at[0], text_at[1] - 0.13 * span, "圆点旁的数字 = 第几步", fontsize=15, color=MUTED, va="center",
            bbox=WHITE_BOX, zorder=8)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    panel_title(fig, [ax], title)
    panel_note(fig, [ax], caption)
savefig(fig, "ch03_learning_rate")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("7. 梯度裁剪：箭头太长就整体缩短，方向不变")


def clip_gradient(g, max_norm):
    """整体乘 min(1, 上限 / 长度)：没超长乘 1（原样），超长就缩到上限。"""
    scale = min(1.0, max_norm / float(np.linalg.norm(g)))
    return g * scale, scale


def pair(g) -> str:
    return f"({g[0]:g}, {g[1]:g})"              # 打印成正文里的写法 (3, 4)，不用 numpy 的 [3. 4.]


# 上限 c（项目里 max_grad_norm = 1.0）。“改一改”第 3 条改这一行。
MAX_NORM = 1.0  # TWEAK-3: 10.0
g_long = np.array([3.0, 4.0])                    # ‖g‖ = 5：第 2 章 2.3 节那个 3-4-5
g_short = np.array([0.3, 0.4])                   # ‖g‖ = 0.5：没超上限
rows = []
for g in (g_long, g_short):
    clipped, scale = clip_gradient(g, MAX_NORM)
    rows.append([pair(g), float(np.linalg.norm(g)), scale, pair(clipped), float(np.linalg.norm(clipped))])
table(["原梯度 g", "长度 ‖g‖", "缩放倍数 min(1, c/‖g‖)", "裁剪后", "裁剪后的长度"], rows)
g_clipped, scale_long = clip_gradient(g_long, MAX_NORM)
g_same, scale_short = clip_gradient(g_short, MAX_NORM)
print("手算（上限 1）：‖(3, 4)‖ = sqrt(9 + 16) = 5；min(1, 1/5) = 0.2；0.2 × (3, 4) = (0.6, 0.8)")
check("(3, 4) 长度 5，上限 1 → 乘 0.2 → (0.6, 0.8)，长度 1", math.isclose(scale_long, 0.2) and np.allclose(g_clipped, [0.6, 0.8])
      and math.isclose(float(np.linalg.norm(g_clipped)), 1.0))
check("裁剪不改方向：裁剪前后的单位向量相同，都是 (3, 4) 除以自己的长度 5",
      np.allclose(g_clipped / np.linalg.norm(g_clipped), g_long / 5))
check("(0.3, 0.4) 长度 0.5 没超上限：min(1, 2) = 1，原样通过", scale_short == 1.0 and np.array_equal(g_same, g_short))
check("自测：(6, 8) 长度 10，上限 5 → 乘 0.5 → (3, 4)；上限 20 → 乘 1，原样 (6, 8)",
      np.allclose(clip_gradient(np.array([6.0, 8.0]), 5.0)[0], [3, 4]) and clip_gradient(np.array([6.0, 8.0]), 5.0)[1] == 0.5
      and np.array_equal(clip_gradient(np.array([6.0, 8.0]), 20.0)[0], [6, 8]) and clip_gradient(np.array([6.0, 8.0]), 20.0)[1] == 1.0)

# ---------------------------------------------------------------------------
banner("7b. 画图：figures/ch03_gradient_clipping.png（箭头缩短、方向不变）")
if MAX_NORM != 1.0:
    print("  上限改过了：这张讲解图是照着“上限 1”画的，跳过。看上面那张表就行。")
else:
    fig = plt.figure(figsize=(9.4, 12.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[1.25, 1.0], hspace=0.46, wspace=0.08,
                          left=0.10, right=0.97, top=0.885, bottom=0.105)
    fig.suptitle("梯度裁剪：箭头太长就整体缩短，方向不变", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.975)
    circle_t = np.linspace(0, np.pi / 2, 100)

    ax = fig.add_subplot(gs[0, 0])
    data_axes(ax, "第 1 项", "第 2 项")
    ax.set_xlim(-0.15, 3.6)
    ax.set_ylim(-0.15, 4.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(4))
    ax.set_yticks(range(5))
    ax.plot(MAX_NORM * np.cos(circle_t), MAX_NORM * np.sin(circle_t), color=BLUE, ls="--", lw=2.2)
    ax.text(1.08, 0.16, "上限 c = 1", color=BLUE, fontsize=FS_SMALL)
    arrow(ax, (0, 0), tuple(g_long), FAINT, lw=4)
    arrow(ax, (0, 0), tuple(g_clipped), ORANGE, lw=5, zorder=6)
    ax.text(1.85, 3.22, "原来的 (3, 4)", color=MUTED, fontsize=FS_SMALL, ha="right", va="center", bbox=WHITE_BOX)
    ax.text(1.25, 0.72, f"裁剪后 {pair(g_clipped)}", color=ORANGE, fontsize=FS_SMALL, va="center", bbox=WHITE_BOX)
    ax_text = fig.add_subplot(gs[0, 1])
    lesson_panel(ax_text, xmax=10, ymax=10)
    hand(ax_text, 0.6, 8.3, "长度 = √(3² + 4²) = 5")
    hand(ax_text, 0.6, 6.5, f"倍数 = min(1, 1/5) = {scale_long:g}")
    hand(ax_text, 0.6, 4.7, f"{scale_long:g} × (3, 4) = {pair(g_clipped)}", color=ORANGE)
    hand(ax_text, 0.6, 2.9, f"新长度 = {float(np.linalg.norm(g_clipped)):g}", color=ORANGE)
    panel_title(fig, [ax], "① 太长：g = (3, 4)，长度 5，超过上限 1")
    panel_note(fig, [ax], "两项乘的是同一个数，所以箭头只变短、不转向（橙箭头压在灰箭头上）。")

    ax = fig.add_subplot(gs[1, 0])
    data_axes(ax, "第 1 项", "第 2 项")
    ax.set_xlim(-0.05, 1.2)
    ax.set_ylim(-0.05, 1.2)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.plot(MAX_NORM * np.cos(circle_t), MAX_NORM * np.sin(circle_t), color=BLUE, ls="--", lw=2.2)
    ax.text(0.60, 0.86, "上限 c = 1", color=BLUE, fontsize=FS_SMALL)
    arrow(ax, (0, 0), tuple(g_same), ORANGE, lw=5, zorder=6)
    ax.text(g_same[0] + 0.04, g_same[1] + 0.03, pair(g_same), color=ORANGE, fontsize=FS_SMALL, va="bottom")
    ax_text = fig.add_subplot(gs[1, 1])
    lesson_panel(ax_text, xmax=10, ymax=10)
    hand(ax_text, 0.6, 8.0, "长度 = √(0.3² + 0.4²) = 0.5")
    hand(ax_text, 0.6, 5.8, "倍数 = min(1, 1/0.5)")
    hand(ax_text, 0.6, 3.9, f"       = min(1, 2) = {scale_short:g}")
    hand(ax_text, 0.6, 1.8, "乘 1：一点没变", color=ORANGE)
    panel_title(fig, [ax], "② 没超上限：g = (0.3, 0.4)，长度 0.5，原样通过")
    panel_note(fig, [ax], "min(1, …) 保证倍数最大是 1：裁剪只会缩短箭头，从不拉长。")
    savefig(fig, "ch03_gradient_clipping")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("8. 映射到项目：正文引用的常数和源码行还在不在")
REPO = Path(__file__).resolve().parents[3]
cfg_path = REPO / "src" / "mjlab_microduck" / "tasks" / "microduck_velocity_env_cfg.py"
CFG_LINES = ["learning_rate=1.0e-3,", 'schedule="adaptive",', "desired_kl=0.01,", "max_grad_norm=1.0,"]
PPO_LINES = ["chain(self.actor.parameters(), self.critic.parameters()), lr=learning_rate",
             "if kl_mean > self.desired_kl * 2.0:",
             "self.learning_rate = max(1e-5, self.learning_rate / 1.5)",
             "elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:",
             "self.learning_rate = min(1e-2, self.learning_rate * 1.5)",
             "for param_group in self.optimizer.param_groups:",
             'param_group["lr"] = self.learning_rate',
             "self.optimizer.zero_grad()",
             "loss.backward()",
             "nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)",
             "nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)",
             "self.optimizer.step()"]

if cfg_path.is_file():
    cfg_text = cfg_path.read_text(encoding="utf-8")
    at = cfg_text.find("MicroduckRlCfg = RslRlOnPolicyRunnerCfg(")
    print("microduck_velocity_env_cfg.py 的 MicroduckRlCfg：", "  ".join(CFG_LINES))
    check("项目配置：初始学习率 0.001、自适应、目标 KL 0.01、梯度裁剪上限 1.0", at >= 0 and lines_in_order(cfg_text[at:], CFG_LINES))
else:
    print("  （没找到项目的 env cfg，跳过这一项）")
rsl_spec = importlib.util.find_spec("rsl_rl")
ppo_path = Path(rsl_spec.origin).parent / "algorithms" / "ppo.py" if rsl_spec and rsl_spec.origin else None
if ppo_path and ppo_path.is_file():
    ppo_text = ppo_path.read_text(encoding="utf-8")
    print("rsl_rl/algorithms/ppo.py：一个优化器同时管 actor 和 critic；update() 里调学习率（÷1.5 / ×1.5，夹在 1e-5 和 1e-2 之间）"
          "→ 清零 → 反向传播 → 两次裁剪 → 走一步")
    check(f"正文映射块引用的源码行（连同“一个优化器管两张网络”那一行，共 {len(PPO_LINES)} 行）都原样存在，且顺序一致",
          lines_in_order(ppo_text, PPO_LINES) and ppo_text.find("    def update(self)") < ppo_text.find(PPO_LINES[1]))
    check("优化器默认是 adam", 'optimizer: str = "adam"' in ppo_text)
else:
    print("  （当前 Python 环境里没有 rsl_rl，跳过源码核对；用 uv run 运行就会核对）")

done()
