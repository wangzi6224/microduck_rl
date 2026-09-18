"""第 2 章实验：向量、点积、范数、矩阵乘法，以及“一层神经网络就是一次矩阵乘”。

运行：uv run python docs/learn-zh/labs/ch02_vectors.py
纯 CPU，numpy + torch。
"""

import math

import numpy as np
import torch

from _common import banner, check, done, savefig, table, use_headless_matplotlib

np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------------------------------------
banner("1. 向量就是定长数组：加法、数乘、点积、长度")
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, -1.0, 0.5])
print("a =", a)
print("b =", b)
print("a + b       =", a + b)
print("2 * a       =", 2 * a)
dot_manual = sum(ai * bi for ai, bi in zip(a, b))
print("a·b (手算)  =", dot_manual, "  = 1·4 + 2·(-1) + 3·0.5")
print("a·b (numpy) =", a @ b)
norm_manual = math.sqrt(sum(ai * ai for ai in a))
print("‖a‖ (手算)  =", norm_manual, "  = sqrt(1² + 2² + 3²)")
print("‖a‖ (numpy) =", np.linalg.norm(a))
check("点积手算 = numpy", abs(dot_manual - a @ b) < 1e-12)
check("范数手算 = numpy", abs(norm_manual - np.linalg.norm(a)) < 1e-12)
unit = a / np.linalg.norm(a)
print("a 的单位向量 =", unit, "  长度 =", np.linalg.norm(unit))
cos_theta = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"a、b 夹角的余弦 = {cos_theta:.4f}  → 夹角 {math.degrees(math.acos(cos_theta)):.1f}°")

# ---------------------------------------------------------------------------
banner("2. 重力向量在“身体坐标系”里的投影（projected_gravity 的直觉）")
print("世界坐标系里重力永远是 g = (0, 0, -1)（单位向量）。")
print("机器人向前倾 θ 度，从机器人自己看，重力就“往前偏”了。")
rows = []
for deg in [0, 10, 30, 70]:
    th = math.radians(deg)
    # 绕 y 轴旋转 θ（前倾，头朝前下方）后，重力在身体系里的 (x, z) 分量
    # 身体“前”轴指向前下方，所以重力在它上面的投影是正的：g_x = +sin θ
    gx = math.sin(th)
    gz = -math.cos(th)
    rows.append([deg, gx, 0.0, gz, gx**2])
table(["前倾角°", "g_x(身体系)", "g_y", "g_z", "g_x²+g_y²"], rows)
print("最后一列正是 upright 奖励里的 xy_squared = sin²(tilt)；fell_over 终止阈值是 70°。")
print("往后仰 θ 度只是把 g_x 换成负号，平方后完全一样——奖励只看“歪多少”，不看“往哪歪”。")
check("g_x² + g_z² = 1（重力始终是单位向量）", all(abs(r[1] ** 2 + r[3] ** 2 - 1) < 1e-12 for r in rows))
check("70° 那一行 g_z ≈ -0.34（摔倒阈值）", abs(rows[-1][3] + 0.342) < 1e-3)

# ---------------------------------------------------------------------------
banner("2b. 把上面这张表画成图：figures/ch02_projected_gravity.png")
plt = use_headless_matplotlib()
from matplotlib.patches import Arc, Circle, Polygon  # noqa: E402

C_UP = "#1f77b4"      # 机身“上”轴
C_FWD = "#2ca02c"     # 机身“前”轴
C_G = "#222222"       # 重力
C_BODY = "#cfd8e3"    # 机身


def _rot(vec, up, fwd):
    """把“沿机身前 a、沿机身上 b”换算成画布坐标。"""
    a, b = vec
    return (a * fwd[0] + b * up[0], a * fwd[1] + b * up[1])


def draw_pose(ax, deg):
    """画一个倾斜 deg 度的机器人，以及重力在它两条身体轴上的分解。"""
    th = math.radians(deg)
    up = (math.sin(th), math.cos(th))       # 机身“上”轴（单位向量）
    fwd = (math.cos(th), -math.sin(th))     # 机身“前”轴（单位向量）
    gx, gz = math.sin(th), -math.cos(th)    # 重力在这两条轴上的读数

    # 世界的竖直方向：始终是画布的上下，和机器人姿势无关
    ax.axvline(0, color="#bbbbbb", lw=1, ls=":", zorder=0)

    # 机身：一个矩形 + 头 + 两只脚，整体跟着 θ 一起转
    body = [(-0.26, -0.55), (0.26, -0.55), (0.26, 0.5), (-0.26, 0.5)]
    ax.add_patch(Polygon([_rot(p, up, fwd) for p in body], closed=True,
                         facecolor=C_BODY, edgecolor="#8b9bb0", lw=1.2, zorder=1))
    ax.add_patch(Circle(_rot((0.0, 0.72), up, fwd), 0.2,
                        facecolor=C_BODY, edgecolor="#8b9bb0", lw=1.2, zorder=1))
    for foot_x in (-0.16, 0.16):
        ax.add_patch(Polygon([_rot(p, up, fwd) for p in
                              [(foot_x - 0.1, -0.55), (foot_x + 0.1, -0.55), (foot_x + 0.1, -0.72),
                               (foot_x - 0.1, -0.72)]], closed=True,
                             facecolor="#a9b7c8", edgecolor="#8b9bb0", lw=1, zorder=1))
    # 朝前的小鼻子，提醒“前”是哪一边
    ax.add_patch(Polygon([_rot(p, up, fwd) for p in [(0.2, 0.78), (0.42, 0.72), (0.2, 0.66)]],
                         closed=True, facecolor="#f0a02a", edgecolor="none", zorder=2))

    # 两条身体轴（机器人自己的“上”和“前”）
    ax.annotate("", xy=up, xytext=(0, 0), zorder=3,
                arrowprops=dict(arrowstyle="-|>", color=C_UP, lw=1.6))
    ax.annotate("", xy=fwd, xytext=(0, 0), zorder=3,
                arrowprops=dict(arrowstyle="-|>", color=C_FWD, lw=1.6))
    lbl_bg = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2)
    ax.text(up[0] * 1.22, up[1] * 1.22, "机身“上”", color=C_UP, fontsize=8.5,
            ha="center", va="center", bbox=lbl_bg, zorder=6)
    ax.text(fwd[0] * 1.24, fwd[1] * 1.24, "机身“前”", color=C_FWD, fontsize=8.5,
            ha="center", va="center", bbox=lbl_bg, zorder=6)

    # 重力：永远指向画布正下方，长度永远是 1
    ax.annotate("", xy=(0, -1), xytext=(0, 0), zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=C_G, lw=2.4))
    ax.text(-0.09, -1.05, "重力 g", color=C_G, fontsize=9, ha="right", va="center")

    # 分解：重力 = (沿“前”走 gx) + (沿“上”走 gz)
    px, py = fwd[0] * gx, fwd[1] * gx
    qx, qy = up[0] * gz, up[1] * gz
    if abs(gx) > 1e-6:
        ax.plot([0, px], [0, py], color=C_FWD, lw=5, alpha=0.8, solid_capstyle="round", zorder=3)
        ax.plot([px, 0], [py, -1], color="#999999", lw=1, ls="--", zorder=2)
        # 标签贴在分量中点，再朝“机身上”方向挪开一点，免得压住机身
        ax.text(px * 0.5 + up[0] * 0.22, py * 0.5 + up[1] * 0.22, f"g_x = {gx:+.2f}",
                color=C_FWD, fontsize=9.5, ha="center", va="center", bbox=lbl_bg, zorder=6)
    ax.plot([0, qx], [0, qy], color=C_UP, lw=5, alpha=0.8, solid_capstyle="round", zorder=3)
    ax.plot([qx, 0], [qy, -1], color="#999999", lw=1, ls="--", zorder=2)
    ax.text(qx * 0.5 - fwd[0] * 0.28, qy * 0.5 - fwd[1] * 0.28, f"g_z = {gz:+.2f}",
            color=C_UP, fontsize=9.5, ha="center", va="center", bbox=lbl_bg, zorder=6)

    # 倾角 θ：世界竖直方向 ↔ 机身“上”轴 之间的夹角
    if deg > 0:
        ax.add_patch(Arc((0, 0), 1.3, 1.3, angle=0, theta1=90 - deg, theta2=90,
                         color="#d62728", lw=1.4, zorder=3))
        mid = math.radians(90 - deg / 2)
        ax.text(0.82 * math.cos(mid), 0.82 * math.sin(mid), f"θ={deg}°",
                color="#d62728", fontsize=9, ha="center", va="center")

    ax.set_xlim(-1.5, 1.6)
    ax.set_ylim(-1.62, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    title = "① 站直 0°" if deg == 0 else f"{'②' if deg == 30 else '③'} 前倾 {deg}°"
    ax.set_title(f"{title}    传感器读数 ({gx:+.2f}, 0, {gz:+.2f})", fontsize=11, pad=6)
    note = {0: "重力全落在“上下”这一项：前后分量 = 0",
            30: "歪了：一部分重力跑到“前后”这一项上",
            70: "仓库判定：倾角过 70° 算摔倒"}[deg]
    ax.text(0, -1.22, f"偏差² = g_x² + g_y² = {gx**2:.2f}", fontsize=9.5, ha="center",
            va="top", color="#333333")
    ax.text(0, -1.42, note, fontsize=8.5, ha="center", va="top", color="#777777")


fig = plt.figure(figsize=(12.6, 8.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.32, wspace=0.12)
for col, deg in enumerate([0, 30, 70]):
    draw_pose(fig.add_subplot(gs[0, col]), deg)

deg_grid = np.linspace(0, 90, 361)
th_grid = np.radians(deg_grid)
marks = [0, 10, 30, 70]

ax = fig.add_subplot(gs[1, 0:2])
ax.plot(deg_grid, np.sin(th_grid), color=C_FWD, lw=2)
ax.plot(deg_grid, -np.cos(th_grid), color=C_UP, lw=2)
for d in marks:
    t = math.radians(d)
    ax.plot([d, d], [-1.05, 1.05], color="#dddddd", lw=1, zorder=0)
    ax.plot(d, math.sin(t), "o", color=C_FWD, ms=5)
    ax.plot(d, -math.cos(t), "o", color=C_UP, ms=5)
ax.axvline(70, color="#d62728", lw=1.2, ls="--")
ax.text(68.5, 1.04, "70° = 摔倒判定", color="#d62728", fontsize=9, ha="right", va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
ax.set_xlabel("倾角 θ（度）")
ax.set_ylabel("传感器读数")
ax.set_title("两个读数随倾角怎么变（站直时是 0 和 −1）", fontsize=10.5)
ax.set_xlim(0, 90)
ax.set_ylim(-1.15, 1.15)
ax.grid(alpha=0.25)
ax.text(62, 0.64, "g_x = sin θ（前后分量）", color=C_FWD, fontsize=9.5, ha="center", va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
ax.text(62, -0.28, "g_z = −cos θ（上下分量）", color=C_UP, fontsize=9.5, ha="center", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))

ax = fig.add_subplot(gs[1, 2])
err2 = np.sin(th_grid) ** 2
ax.plot(deg_grid, err2, color="#7f4fbf", lw=2)
ax.plot(deg_grid, np.exp(-err2 / 0.05), color="#e07b39", lw=2)
for d in marks:
    t = math.radians(d)
    ax.plot(d, math.sin(t) ** 2, "o", color="#7f4fbf", ms=5)
    ax.plot(d, math.exp(-math.sin(t) ** 2 / 0.05), "o", color="#e07b39", ms=5)
ax.annotate("倾 10° 就只剩 0.55 分", xy=(10, math.exp(-math.sin(math.radians(10)) ** 2 / 0.05)),
            xytext=(24, 0.80), fontsize=9, color="#b35c17",
            arrowprops=dict(arrowstyle="->", color="#b35c17", lw=1))
ax.axvline(70, color="#d62728", lw=1.2, ls="--")
ax.set_xlabel("倾角 θ（度）")
ax.set_xlim(0, 90)
ax.set_ylim(-0.05, 1.08)
ax.grid(alpha=0.25)
ax.set_title("同一件事换成分数：奖励为什么这么敏感", fontsize=10.5)
ax.text(79, 0.42, "偏差²\n= g_x²+g_y²", color="#7f4fbf", fontsize=9, ha="center", va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
ax.text(44, 0.16, "站直奖励\nexp(−偏差²/0.05)", color="#e07b39", fontsize=9, ha="center", va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))

fig.suptitle("投影重力 projected_gravity：重力从没变，变的是机器人自己的姿势", fontsize=13)
savefig(fig, "ch02_projected_gravity")

# ---------------------------------------------------------------------------
banner("2c. 两根身体轴的坐标是怎么来的：figures/ch02_axis_rotation.png")
TH = math.radians(30)
S, CO = math.sin(TH), math.cos(TH)     # 0.50, 0.87
UP30 = (S, CO)                          # 机身“上”轴转 30° 之后
FWD30 = (CO, -S)                        # 机身“前”轴转 30° 之后
G2 = (0.0, -1.0)                        # 世界系的重力（画布坐标：右=前，上=天）

print(f"sin30° = {S:.4f}   cos30° = {CO:.4f}")
print(f"机身“上”轴 = (sin30°, cos30°)  = ({UP30[0]:.2f}, {UP30[1]:.2f})")
print(f"机身“前”轴 = (cos30°, -sin30°) = ({FWD30[0]:.2f}, {FWD30[1]:.2f})")
dot_up = G2[0] * UP30[0] + G2[1] * UP30[1]
dot_fwd = G2[0] * FWD30[0] + G2[1] * FWD30[1]
print(f"g_z = g·上 = 0×{UP30[0]:.2f} + (-1)×{UP30[1]:.2f}  = {dot_up:+.2f}")
print(f"g_x = g·前 = 0×{FWD30[0]:.2f} + (-1)×({FWD30[1]:.2f}) = {dot_fwd:+.2f}")
check("g_z = -cos30°", abs(dot_up + CO) < 1e-12)
check("g_x = +sin30°", abs(dot_fwd - S) < 1e-12)
check("两根轴垂直（点积 = 0）", abs(UP30[0] * FWD30[0] + UP30[1] * FWD30[1]) < 1e-12)
check("两根轴都是单位向量", abs(UP30[0] ** 2 + UP30[1] ** 2 - 1) < 1e-12)


def _frame(ax, xlim, ylim, title):
    """一块干净的坐标纸：淡灰的横竖轴 + 标题。"""
    ax.axhline(0, color="#c9c9c9", lw=1, zorder=0)
    ax.axvline(0, color="#c9c9c9", lw=1, zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, fontsize=11, pad=8)
    ax.text(xlim[1] - 0.02, 0.04, "→ 前（画布向右）", fontsize=8, color="#999999", ha="right")
    ax.text(0.04, ylim[1] - 0.04, "↑ 天（画布向上）", fontsize=8, color="#999999", va="top")


def _ghost_body(ax, deg, alpha=0.18):
    """淡淡的机身轮廓，提醒“轴是钉在身上的”。"""
    th = math.radians(deg)
    up, fwd = (math.sin(th), math.cos(th)), (math.cos(th), -math.sin(th))
    pts = [(-0.17, -0.34), (0.17, -0.34), (0.17, 0.42), (-0.17, 0.42)]
    ax.add_patch(Polygon([(a * fwd[0] + b * up[0], a * fwd[1] + b * up[1]) for a, b in pts],
                         closed=True, facecolor="#7a8ba0", edgecolor="none",
                         alpha=alpha, zorder=0))
    ax.add_patch(Circle((0.62 * up[0], 0.62 * up[1]), 0.15, facecolor="#7a8ba0",
                        edgecolor="none", alpha=alpha, zorder=0))


def _arrow(ax, vec, color, lw=2.2, z=4):
    ax.annotate("", xy=vec, xytext=(0, 0), zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))


BOX = dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5)

fig = plt.figure(figsize=(12.4, 9.6))
axes = fig.subplots(2, 2)

# ---- ① 站直时两根轴的坐标 --------------------------------------------------
ax = axes[0][0]
_frame(ax, (-0.45, 1.5), (-0.75, 1.42), "① 先看站直：两根轴就是坐标纸的两个方向")
_ghost_body(ax, 0)
_arrow(ax, (0, 1), C_UP)
_arrow(ax, (1, 0), C_FWD)
ax.text(0.06, 1.06, '机身"上" = (0, 1)', color=C_UP, fontsize=10.5, ha="left", bbox=BOX)
ax.text(1.05, -0.14, '机身"前" = (1, 0)', color=C_FWD, fontsize=10.5, ha="right", bbox=BOX)
ax.text(0.52, -0.5, "括号里两个数的意思：\n（往右伸多少, 往上伸多少）\n每根轴长度都恒为 1",
        fontsize=9.5, ha="center", va="center", color="#444444")

# ---- ② 机身“上”轴转 30° ----------------------------------------------------
ax = axes[0][1]
_frame(ax, (-0.45, 1.5), (-0.75, 1.42), '② 机身"上"轴：从竖直方向歪开 30°')
_ghost_body(ax, 30)
ax.add_patch(Polygon([(0, 0), (UP30[0], 0), UP30], closed=True,
                     facecolor=C_UP, alpha=0.12, zorder=1))
_arrow(ax, UP30, C_UP)
ax.plot([UP30[0], UP30[0]], [0, UP30[1]], color=C_UP, ls="--", lw=1.2, zorder=2)
ax.plot([0, UP30[0]], [UP30[1], UP30[1]], color=C_UP, ls="--", lw=1.2, zorder=2)
ax.plot([0, UP30[0]], [0, 0], color=C_UP, lw=4.5, alpha=0.8, solid_capstyle="butt", zorder=3)
ax.plot([0, 0], [0, UP30[1]], color=C_UP, lw=4.5, alpha=0.8, solid_capstyle="butt", zorder=3)
ax.add_patch(Arc((0, 0), 0.9, 0.9, theta1=90 - 30, theta2=90, color="#d62728", lw=1.5, zorder=4))
ax.text(0.17, 0.52, "θ=30°", color="#d62728", fontsize=10, ha="left")
ax.text(UP30[0] / 2, -0.13, f"sin30° = {S:.2f}", color=C_UP, fontsize=10.5, ha="center", bbox=BOX)
ax.text(-0.06, UP30[1] / 2, f"cos30°\n= {CO:.2f}", color=C_UP, fontsize=10.5, ha="right",
        va="center", bbox=BOX)
ax.text(0.80, 1.06, "箭头长度仍是 1", color="#666666", fontsize=9, ha="left")
ax.text(0.52, -0.52, f"→ 机身\"上\" = (sin30°, cos30°) = ({UP30[0]:.2f}, {UP30[1]:.2f})",
        fontsize=10.5, ha="center", va="center", color=C_UP)

# ---- ③ 机身“前”轴转 30° ----------------------------------------------------
ax = axes[1][0]
_frame(ax, (-0.45, 1.5), (-1.02, 1.15), '③ 机身"前"轴：从水平方向低下 30°')
_ghost_body(ax, 30)
ax.add_patch(Polygon([(0, 0), (FWD30[0], 0), FWD30], closed=True,
                     facecolor=C_FWD, alpha=0.12, zorder=1))
_arrow(ax, FWD30, C_FWD)
ax.plot([FWD30[0], FWD30[0]], [0, FWD30[1]], color=C_FWD, ls="--", lw=1.2, zorder=2)
ax.plot([0, FWD30[0]], [FWD30[1], FWD30[1]], color=C_FWD, ls="--", lw=1.2, zorder=2)
ax.plot([0, FWD30[0]], [0, 0], color=C_FWD, lw=4.5, alpha=0.8, solid_capstyle="butt", zorder=3)
ax.plot([0, 0], [0, FWD30[1]], color=C_FWD, lw=4.5, alpha=0.8, solid_capstyle="butt", zorder=3)
ax.add_patch(Arc((0, 0), 0.9, 0.9, theta1=-30, theta2=0, color="#d62728", lw=1.5, zorder=4))
ax.text(0.50, -0.19, "θ=30°", color="#d62728", fontsize=10, ha="left")
ax.text(FWD30[0] / 2, 0.10, f"cos30° = {CO:.2f}", color=C_FWD, fontsize=10.5, ha="center", bbox=BOX)
ax.text(-0.06, FWD30[1] / 2, f"−sin30°\n= {-S:.2f}", color=C_FWD, fontsize=10.5, ha="right",
        va="center", bbox=BOX)
ax.text(-0.02, -0.80, "朝下 → 负号", color="#888888", fontsize=9, ha="left")
ax.text(0.52, -0.95, f"→ 机身\"前\" = (cos30°, −sin30°) = ({FWD30[0]:.2f}, {FWD30[1]:.2f})",
        fontsize=10.5, ha="center", va="center", color=C_FWD)

# ---- ④ 用这两根轴去拆重力 --------------------------------------------------
ax = axes[1][1]
_frame(ax, (-0.78, 1.55), (-2.05, 1.3), "④ 拿这两根轴去拆重力：各做一次点积")
_ghost_body(ax, 30, alpha=0.12)
_arrow(ax, UP30, C_UP, lw=1.8)
_arrow(ax, FWD30, C_FWD, lw=1.8)
_arrow(ax, G2, C_G, lw=2.6, z=5)
ax.text(-0.12, -0.98, "重力 g = (0, −1)", color=C_G, fontsize=10, ha="right", bbox=BOX)
ax.text(UP30[0] * 1.12, UP30[1] * 1.12, '机身"上"', color=C_UP, fontsize=9.5, ha="center", bbox=BOX)
ax.text(FWD30[0] * 1.14, FWD30[1] * 1.14, '机身"前"', color=C_FWD, fontsize=9.5, ha="center", bbox=BOX)
pz = (UP30[0] * dot_up, UP30[1] * dot_up)
px = (FWD30[0] * dot_fwd, FWD30[1] * dot_fwd)
ax.plot([0, pz[0]], [0, pz[1]], color=C_UP, lw=5, alpha=0.8, solid_capstyle="round", zorder=3)
ax.plot([0, px[0]], [0, px[1]], color=C_FWD, lw=5, alpha=0.8, solid_capstyle="round", zorder=3)
ax.plot([pz[0], G2[0]], [pz[1], G2[1]], color="#999999", ls="--", lw=1, zorder=2)
ax.plot([px[0], G2[0]], [px[1], G2[1]], color="#999999", ls="--", lw=1, zorder=2)
ax.text(-0.74, -1.20,
        f"g_z = g · 机身“上” = 0×{UP30[0]:.2f} + (−1)×{UP30[1]:.2f} = {dot_up:+.2f}\n"
        f"g_x = g · 机身“前” = 0×{FWD30[0]:.2f} + (−1)×({FWD30[1]:.2f}) = {dot_fwd:+.2f}",
        fontsize=10.5, ha="left", va="top", linespacing=1.6,
        bbox=dict(facecolor="#f5f7fa", edgecolor="#d0d7e0", pad=6))
ax.text(-0.74, -1.95,
        f"自检：两根轴垂直 → 上 · 前 = {UP30[0]:.2f}×{FWD30[0]:.2f} + {UP30[1]:.2f}×({FWD30[1]:.2f}) = 0",
        fontsize=9, ha="left", va="bottom", color="#666666")

fig.suptitle("身体转 30°，两根轴的坐标怎么来 —— 以及为什么读数就是两次点积", fontsize=13.5)
fig.subplots_adjust(hspace=0.22, wspace=0.06)
savefig(fig, "ch02_axis_rotation")

# ---------------------------------------------------------------------------
banner("3. 矩阵 × 向量 = 每一行做一次点积")
W = np.array([[1.0, 0.0, 2.0],
              [0.5, -1.0, 1.0]])
x = np.array([1.0, 2.0, 3.0])
row0 = W[0] @ x
row1 = W[1] @ x
print("W =\n", W)
print("x =", x)
print("第 0 行·x =", row0, "  第 1 行·x =", row1)
print("W @ x     =", W @ x)
check("矩阵乘 = 逐行点积", np.allclose(W @ x, [row0, row1]))
print("形状规则：(2×3) @ (3,) → (2,)   ——  输入 3 个数，输出 2 个数")

# ---------------------------------------------------------------------------
banner("4. 项目 actor 的第一层：61 维观测 → 512 个数")
torch.manual_seed(0)
layer = torch.nn.Linear(61, 512)          # 和 rsl_rl 里 MLP 的第一层完全一样
obs = torch.randn(61)                     # 假装这是一帧 61 维观测
out_torch = layer(obs)
W_np = layer.weight.detach().numpy()      # 形状 (512, 61)
b_np = layer.bias.detach().numpy()        # 形状 (512,)
out_np = W_np @ obs.numpy() + b_np        # y = W x + b
print("weight 形状 =", tuple(W_np.shape), " bias 形状 =", tuple(b_np.shape))
print("参数个数 = 512×61 + 512 =", 512 * 61 + 512)
print("前 5 个输出 (torch) =", out_torch[:5].detach().numpy())
print("前 5 个输出 (numpy) =", out_np[:5])
check("torch.nn.Linear == W@x + b", np.allclose(out_torch.detach().numpy(), out_np, atol=1e-5))

n_params = sum(p.numel() for p in layer.parameters())
check("参数个数 = 31744", n_params == 31744)

# ---------------------------------------------------------------------------
banner("5. 转置：把行变成列")
print("W.T =\n", W.T, "\n形状", W.shape, "→", W.T.shape)

done()
