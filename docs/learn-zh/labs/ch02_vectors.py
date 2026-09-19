"""第 2 章实验：向量、矩阵、张量、坐标轴与数据轴，以及神经网络里的矩阵乘。

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

# ---------------------------------------------------------------------------
banner("6. 沿哪条轴求和：dim=1 和 dim=0 算的不是一回事")
E = torch.tensor([[0.1, -0.2],        # A：(前后误差, 左右误差)
                  [-0.3, 0.4],        # B
                  [0.1, 0.1]])        # C
sq = E.square()                       # [3,2] 逐格平方，形状不变
per_robot = sq.sum(dim=1)             # 把第 1 轴（方向）加没 → 每只机器人一个数
per_axis = sq.sum(dim=0)              # 把第 0 轴（机器人）加没 → 每个方向一个数
print("E =\n", E.numpy(), "  形状", tuple(E.shape))
print("逐格平方 =\n", sq.numpy())
table(
    ["写法", "加没的是", "输出形状", "结果", "含义"],
    [
        ["sq.sum(dim=1)", "第 1 轴（方向）", str(tuple(per_robot.shape)), str(per_robot.numpy()), "每只机器人一个误差 ✓"],
        ["sq.sum(dim=0)", "第 0 轴（机器人）", str(tuple(per_axis.shape)), str(per_axis.numpy()), "全队统计量，不是奖励 ✗"],
    ],
)
check("dim=1 → 每只机器人一个数 [0.05, 0.25, 0.02]", np.allclose(per_robot.numpy(), [0.05, 0.25, 0.02]))
check("dim=0 → 每个方向一个数 [0.11, 0.21]", np.allclose(per_axis.numpy(), [0.11, 0.21]))

reward = torch.exp(-per_robot / 0.1)  # 第 1 章的钟形打分，σ² = 0.1
print("奖励 exp(−误差²/0.1) =", reward.numpy(), "  形状", tuple(reward.shape))
print("形状一路：[3,2] → [3,2] → [3] → [3]，进来三只，出去三个分数")
check("奖励 ≈ [0.606531, 0.082085, 0.818731]",
      np.allclose(reward.numpy(), [0.606531, 0.082085, 0.818731], atol=1e-6))

print("\n形状体检（表不是方的时候，写错立刻暴露）：")
act_diff = torch.zeros(2, 14)         # 两只环境 × 14 个关节的动作差
print("  [2,14].sum(dim=1) →", tuple(act_diff.sum(dim=1).shape), " ← 每只环境一个平滑代价 ✓")
print("  [2,14].sum(dim=0) →", tuple(act_diff.sum(dim=0).shape), " ← 每个关节跨机器人的合计 ✗")
check("[2,14] 沿 dim=1 求和 → [2]", tuple(act_diff.sum(dim=1).shape) == (2,))
big = torch.zeros(24, 4096, 61)       # 时间 × 机器人 × 观测项
print("  [24,4096,61].sum(dim=2) →", tuple(big.sum(dim=2).shape), " ← 每只机器人每一步一个数 ✓")
check("[24,4096,61] 沿 dim=2 求和 → [24,4096]", tuple(big.sum(dim=2).shape) == (24, 4096))

print("\n进阶折叠块里的批量矩阵乘（一行 = 一只机器人）：")
Xb = torch.tensor([[1.0, 2.0], [3.0, 4.0]])          # 2 只机器人，每只 2 项输入
Wb = torch.tensor([[2.0, 0.0], [1.0, -1.0], [0.0, 3.0]])  # 3 个输出 × 2 项输入
bb = torch.tensor([0.5, 0.0, -1.0])                  # 偏置（每一行都加同一份“运费”）
Y = Xb @ Wb.T
print("  X @ W.T =\n", Y.numpy(), "  形状", tuple(Y.shape))
print("  加偏置（广播）=\n", (Y + bb).numpy())
print("  X * X（逐格相乘，不是矩阵乘）=\n", (Xb * Xb).numpy())
check("XWᵀ = [[2,-1,6],[6,-1,12]]", np.allclose(Y.numpy(), [[2, -1, 6], [6, -1, 12]]))
check("广播加偏置 = [[2.5,-1,5],[6.5,-1,11]]", np.allclose((Y + bb).numpy(), [[2.5, -1, 5], [6.5, -1, 11]]))
check("X * X 逐格相乘 = [[1,4],[9,16]]", np.allclose((Xb * Xb).numpy(), [[1, 4], [9, 16]]))

# ---------------------------------------------------------------------------
banner("6b. 把两条轴画成图：figures/ch02_axes.png")
C_OK = "#2ca02c"      # 沿 dim=1：要的那种加法
C_BAD = "#d62728"     # 沿 dim=0：算出来不是奖励
C_EDGE = "#8fa3bd"

fig = plt.figure(figsize=(12.8, 5.0))
axL, axR = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.10})


def _cell(ax, x, y, txt, fc="#eef2f7"):
    ax.add_patch(plt.Rectangle((x, y), 0.92, 0.74, facecolor=fc, edgecolor=C_EDGE, lw=1.3))
    ax.text(x + 0.46, y + 0.37, txt, ha="center", va="center", fontsize=11.5)


# ---- 左：一张 [3,2] 的表，两种加法 -----------------------------------------
rows_lbl = ["A", "B", "C"]
cols_lbl = ["前后²", "左右²"]
sq_np = sq.numpy()
for j, cl in enumerate(cols_lbl):
    axL.text(j + 0.46, 3.05, cl, ha="center", va="bottom", fontsize=11, color="#555555")
for i, rl in enumerate(rows_lbl):
    y = 2 - i
    axL.text(-0.18, y + 0.37, rl, ha="right", va="center", fontsize=12, fontweight="bold")
    for j in range(2):
        _cell(axL, j, y, f"{sq_np[i][j]:.2f}")
    axL.annotate("", xy=(2.62, y + 0.37), xytext=(2.02, y + 0.37),
                 arrowprops=dict(arrowstyle="-|>", color=C_OK, lw=2.0))
    axL.text(2.72, y + 0.37, f"{per_robot[i]:.2f}", ha="left", va="center",
             fontsize=12, color=C_OK, fontweight="bold")
axL.text(4.85, 1.37, "sum(dim=1)\n[3,2] → [3]\n每只机器人一个数 ✓", ha="center", va="center",
         fontsize=10.5, color=C_OK,
         bbox=dict(boxstyle="round,pad=0.35", fc="#eefaee", ec=C_OK, lw=1.2))
for j in range(2):
    axL.annotate("", xy=(j + 0.46, -0.72), xytext=(j + 0.46, -0.06),
                 arrowprops=dict(arrowstyle="-|>", color=C_BAD, lw=2.0))
    axL.text(j + 0.46, -0.95, f"{per_axis[j]:.2f}", ha="center", va="top",
             fontsize=12, color=C_BAD, fontweight="bold")
axL.text(0.92, -1.62, "sum(dim=0)：[3,2] → [2]\n全队统计量，当奖励就错 ✗", ha="center", va="top",
         fontsize=10.5, color=C_BAD,
         bbox=dict(boxstyle="round,pad=0.35", fc="#fdeeee", ec=C_BAD, lw=1.2))
axL.set_title("一张 [3, 2] 的表：三只机器人 × 两个方向（已逐格平方）", fontsize=11.5, pad=14)
axL.set_xlim(-0.8, 6.4)
axL.set_ylim(-2.6, 3.6)
axL.set_aspect("equal")
axL.axis("off")

# ---- 右：三条轴的 [24, 4096, 61] -------------------------------------------
W_, H_, DX, DY = 3.6, 2.0, 0.46, 0.46
for k in (2, 1, 0):                       # 从后往前画三张表，代表 24 步里的三步
    shade = ["#dce6f2", "#e8eef7", "#f3f6fb"][k]
    axR.add_patch(plt.Rectangle((k * DX, k * DY), W_, H_, facecolor=shade,
                                edgecolor=C_EDGE, lw=1.3, zorder=3 - k))
axR.text(W_ / 2, H_ / 2, "一张 [4096, 61] 的表\n（这一步所有机器人的观测）",
         ha="center", va="center", fontsize=10, zorder=5)
axR.annotate("", xy=(W_, -0.30), xytext=(0, -0.30),
             arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=2.0))
axR.text(W_ / 2, -0.52, "第 2 轴：观测 61 项", ha="center", va="top", fontsize=10.5, color="#1f77b4")
axR.annotate("", xy=(-0.30, 0), xytext=(-0.30, H_),
             arrowprops=dict(arrowstyle="-|>", color="#7f4fbf", lw=2.0))
axR.text(-0.45, H_ / 2, "第 1 轴：4096 只机器人\n各自独立，不能串起来", ha="right", va="center",
         fontsize=10.5, color="#7f4fbf")
axR.annotate("", xy=(2 * DX + W_ + 0.30, 2 * DY + H_ + 0.30), xytext=(W_ + 0.16, H_ + 0.16),
             arrowprops=dict(arrowstyle="-|>", color="#e07b39", lw=2.2))
axR.text(2 * DX + W_ + 0.42, 2 * DY + H_ + 0.30, "第 0 轴：24 步时间\n有先后，第 12 章沿它递推",
         ha="left", va="center", fontsize=10.5, color="#e07b39")
axR.set_title("三条轴的张量 [24, 4096, 61]", fontsize=11.5, pad=14)
axR.set_xlim(-2.6, 8.4)
axR.set_ylim(-1.5, 4.4)
axR.set_aspect("equal")
axR.axis("off")

fig.suptitle("轴 = 表的一个方向；sum(dim=k) 把第 k 轴加没", fontsize=13.5)
savefig(fig, "ch02_axes")

# ---------------------------------------------------------------------------
banner("7. 数据有几层地址：shape、ndim、numel、索引与求和")
scalar = torch.tensor(7)
one_item = torch.tensor([7])
vector = torch.tensor([3, 4, 5])
row_vector = torch.tensor([[3, 4, 5]])
column_vector = torch.tensor([[3], [4], [5]])
T = torch.tensor([[[1, 2], [3, 4], [5, 6]],
                  [[7, 8], [9, 10], [11, 12]]])
table(
    ["数据", "shape（每条轴的长度）", "ndim（轴数）", "numel（数字总数）"],
    [[name, tuple(data.shape), data.ndim, data.numel()] for name, data in [
        ("标量 7", scalar), ("单元素向量 [7]", one_item),
        ("向量 [3,4,5]", vector), ("一行 [[3,4,5]]", row_vector),
        ("三行 [[3],[4],[5]]", column_vector), ("两张三行两列表 T", T),
    ]],
)
print("3 维向量的‘3’说的是分量数；1 维张量的‘1’说的是数据轴数。")
print("一个数也能放进一条轴：7 和 [7] 数字总数一样，形状不同。")
check("标量：shape=()、ndim=0、numel=1", scalar.shape == torch.Size([]) and scalar.ndim == 0 and scalar.numel() == 1)
check("单元素向量：shape=(1,)、ndim=1、numel=1", one_item.shape == (1,) and one_item.ndim == 1 and one_item.numel() == 1)
check("3 维向量用 1 维张量存：shape=(3,)", vector.shape == (3,) and vector.ndim == 1)
check("行向量和列向量都用 2 维张量存", row_vector.shape == (1, 3) and column_vector.shape == (3, 1))
check("T 有 3 条轴、12 个数", T.shape == (2, 3, 2) and T.ndim == 3 and T.numel() == 12)

print("\n给每条轴起名字：T[时间, 机器人, 速度项]，编号都从 0 开始。")
print("速度项 0 = 前后速度，速度项 1 = 左右速度；下面是方便手算的示意数值。")
for expression, selection in [("T[1]", T[1]), ("T[1,2]", T[1, 2]),
                              ("T[1,2,0]", T[1, 2, 0]), ("T[:,2,0]", T[:, 2, 0])]:
    print(f"  {expression:10s} = {selection.tolist()}，shape={tuple(selection.shape)}")
print("冒号 : 表示这一条轴的全部位置；普通整数索引选中一个位置，并去掉这条轴。")
check("地址 [1,2,0] 取到第二时刻、第三只机器人、第一项：11", T[1, 2, 0].item() == 11)
check("切片 [:,2,0] 保留全部时间：[5,11]", torch.equal(T[:, 2, 0], torch.tensor([5, 11])))
check("逐层选中后，形状 (3,2) → (2,) → ()", T[1].shape == (3, 2) and T[1, 2].shape == (2,) and T[1, 2, 0].shape == ())
check("一页切片保留页轴，数字与直接取页相同", T[1:2].shape == (1, 3, 2) and torch.equal(T[1:2][0], T[1]))
check("机器人 2 在两个时刻的记录 = [[5,6],[11,12]]", torch.equal(T[:, 2, :], torch.tensor([[5, 6], [11, 12]])))
check("自测地址 [0,1,1] = 4", T[0, 1, 1].item() == 4)
check("不指定 dim 的平方和为标量 0.32", sq.sum().ndim == 0 and math.isclose(sq.sum().item(), 0.32, abs_tol=1e-6))

print("\n沿一条轴求和：让该轴的编号变化，其余编号固定，再把取到的数相加。")
expected_sums = [
    torch.tensor([[8, 10], [12, 14], [16, 18]]),
    torch.tensor([[9, 12], [27, 30]]),
    torch.tensor([[3, 7, 11], [15, 19, 23]]),
]
for dimension, axis_name in enumerate(["时间", "机器人", "速度项"]):
    result = T.sum(dim=dimension)
    print(f"  T.sum(dim={dimension})：合并{axis_name}，shape={tuple(result.shape)}，结果={result.tolist()}")
    check(f"dim={dimension} 的结果与手算一致", torch.equal(result, expected_sums[dimension]))
print("这些求和用来观察轴；两种速度项相加不等于速度的大小。")
last_axis_sum = T.sum(dim=-1)
kept_axis_sum = T.sum(dim=-1, keepdim=True)
print("  dim=-1 表示最后一条轴：", last_axis_sum.tolist(), "shape=", tuple(last_axis_sum.shape))
print("  keepdim=True 留下长度为 1 的轴：", kept_axis_sum.tolist(), "shape=", tuple(kept_axis_sum.shape))
check("3 维张量的 dim=-1 就是 dim=2", torch.equal(last_axis_sum, T.sum(dim=2)))
check("keepdim 留下单元素轴，数字不变", kept_axis_sum.shape == (2, 3, 1) and torch.equal(kept_axis_sum.squeeze(-1), last_axis_sum))
check("矩阵配方的两项输出 = [7,1.5]", np.allclose(W @ x, [7, 1.5]))

# 2.4 进阶：R 的列是身体轴的世界坐标；反向换坐标用 R.T。
angle = math.radians(30)
rotation = np.array([[math.cos(angle), math.sin(angle)],
                     [-math.sin(angle), math.cos(angle)]])
body_gravity = rotation.T @ np.array([0.0, -1.0])
check("身体轴构成正交旋转，行列式为 1", np.allclose(rotation.T @ rotation, np.eye(2)) and math.isclose(np.linalg.det(rotation), 1.0))
check("30° 前倾的身体重力 = (0.5, -sqrt(3)/2)", np.allclose(body_gravity, [0.5, -math.sqrt(3) / 2]))
check("换回世界系，重力仍是 (0,-1)", np.allclose(rotation @ body_gravity, [0.0, -1.0]))

# ---------------------------------------------------------------------------
banner("7b. 零基础图解：从一个数，到一条记录、一张表、几页表")

# 每张图围绕一个问题；固定数据和上面的实验一致，避免图与手算对不上。
INK = "#243442"
MUTED = "#546574"
BLUE = "#2065a8"
GREEN = "#28745a"
ORANGE = "#b75b25"


def _lesson_panel(ax, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.text(0, 3.7, title, fontsize=20, fontweight="bold", color=INK, va="top")


def _lesson_cells(ax, values, left, bottom, width=1.0, height=0.7,
                  facecolor="#eaf1f8", highlights=()):
    """数字格按真实的二维行列绘制；highlights 是要突出显示的 (行, 列)。"""
    for i, values_row in enumerate(values):
        for j, value in enumerate(values_row):
            selected = (i, j) in highlights
            cell_y = bottom + (len(values) - i - 1) * height
            ax.add_patch(plt.Rectangle(
                (left + j * width, cell_y), width, height,
                facecolor="#fff0d9" if selected else facecolor,
                edgecolor=ORANGE if selected else "#8ea5b9", lw=2 if selected else 1.2,
            ))
            ax.text(left + (j + 0.5) * width, cell_y + height / 2,
                    f"{value:g}", fontsize=20, ha="center", va="center",
                    color=ORANGE if selected else INK, fontweight="bold" if selected else "normal")


fig, axes = plt.subplots(4, 1, figsize=(9, 13.8))
fig.subplots_adjust(top=0.93, hspace=0.14)
fig.suptitle("数字一样普通，摆放方式可以不同", fontsize=23, fontweight="bold", color=INK)
ax = axes[0]
_lesson_panel(ax, "① 标量：单独一个数")
_lesson_cells(ax, [[7]], 0.4, 1.55, 1.35, 1.05)
ax.text(3.5, 2.05, "shape = ()\n0 条轴 · 一共 1 个数", fontsize=20, va="center", color=BLUE, linespacing=1.8)
ax.text(0.4, 0.50, "不用先选行、列或页，直接就是 7。", fontsize=18, color=MUTED)
ax = axes[1]
_lesson_panel(ax, "② 向量：一条记录，按顺序放数")
_lesson_cells(ax, [[3, 4]], 0.4, 1.65, 1.25, 0.9)
ax.text(3.5, 2.05, "shape = (2,)\n1 条轴 · 长度 2 · 共 2 个数", fontsize=20, va="center", color=BLUE, linespacing=1.8)
ax.text(0.4, 0.55, "选第几个数就够了；逗号说明 shape 只有一项。", fontsize=18, color=MUTED)
ax = axes[2]
_lesson_panel(ax, "③ 矩阵：一张表，用行和列找数")
_lesson_cells(ax, [[1, 2], [3, 4], [5, 6]], 0.4, 0.95, 1.15, 0.70)
ax.text(3.5, 2.05, "shape = (3, 2)\n2 条轴 · 3 行 × 2 列\n一共 3 × 2 = 6 个数", fontsize=20, va="center", color=BLUE, linespacing=1.55)
ax.text(0.4, 0.30, "地址需要两项：先选行，再选列。", fontsize=18, color=MUTED)
ax = axes[3]
_lesson_panel(ax, "④ 三维张量：按页收好几张表")
_lesson_cells(ax, T[0].tolist(), 0.2, 1.1, 0.85, 0.60)
_lesson_cells(ax, T[1].tolist(), 2.2, 1.1, 0.85, 0.60)
ax.text(1.05, 0.78, "第 0 页", ha="center", fontsize=17, color=MUTED)
ax.text(3.05, 0.78, "第 1 页", ha="center", fontsize=17, color=MUTED)
ax.text(4.3, 2.10, "shape = (2, 3, 2)\n3 条轴：页、行、列\n共 2 × 3 × 2 = 12 个数", fontsize=20, va="center", color=BLUE, linespacing=1.55)
ax.text(0.2, 0.20, "在 PyTorch 里，上面四种数据都可以叫张量。", fontsize=18, color=MUTED)
savefig(fig, "ch02_number_to_tensor")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.2, 8.5))
fig.subplots_adjust(top=0.85, bottom=0.17, left=0.11, right=0.94)
fig.suptitle("向量 [3, 4]：往右 3 格，往上 4 格", fontsize=22, fontweight="bold", color=INK)
ax.set_xlim(-0.6, 5.2)
ax.set_ylim(-0.6, 5.2)
ax.set_aspect("equal")
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.tick_params(labelsize=16, length=0, pad=8)
ax.grid(color="#dae2eb", linewidth=1.2)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.annotate("", xy=(5.1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=1.9, color=MUTED))
ax.annotate("", xy=(0, 5.1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=1.9, color=MUTED))
ax.text(5.05, -0.25, "x 轴", fontsize=17, color=MUTED, ha="right")
ax.text(0.13, 5.0, "y 轴", fontsize=17, color=MUTED, va="top")
ax.annotate("", xy=(3, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=4, color=BLUE))
ax.annotate("", xy=(3, 4), xytext=(3, 0), arrowprops=dict(arrowstyle="-|>", lw=4, color=GREEN))
ax.annotate("", xy=(3, 4), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=4, color=ORANGE))
ax.text(1.50, -0.43, "先往右 3 格", fontsize=18, color=BLUE, ha="center")
ax.text(3.18, 1.85, "再往上\n4 格", fontsize=18, color=GREEN, linespacing=1.5)
ax.text(0.73, 2.45, "箭头长度 = 5", rotation=53.13, fontsize=18, color=ORANGE, ha="center", va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=3))
ax.plot(3, 4, "o", color=ORANGE, ms=10)
ax.text(3.17, 4.12, "终点 (3, 4)", fontsize=19, color=INK)
ax.text(-0.12, -0.15, "原点", fontsize=16, color=MUTED, ha="right", va="top")
fig.text(0.5, 0.080, "两个分量：x = 3，y = 4；它们描述同一根箭头。", ha="center", fontsize=18, color=INK)
fig.text(0.5, 0.032, "长度不是 3 + 4：直达的斜线比绕直角走更短。", ha="center", fontsize=18, color=MUTED)
savefig(fig, "ch02_vector_coordinates")
plt.close(fig)

fig, axes = plt.subplots(3, 1, figsize=(9, 10.6))
fig.subplots_adjust(top=0.91, hspace=0.15, bottom=0.08)
fig.suptitle("3 维向量，可以用 1 维张量存", fontsize=23, fontweight="bold", color=INK)
for ax, title in zip(axes, ["① 一维数组：[3, 4, 5]", "② 一行矩阵：[[3, 4, 5]]", "③ 一列矩阵：[[3], [4], [5]]"]):
    _lesson_panel(ax, title)
ax = axes[0]
_lesson_cells(ax, [vector.tolist()], 0.4, 1.75, 1.0, 0.85)
ax.text(0.9, 1.36, "位置 0", fontsize=15, ha="center", color=MUTED)
ax.text(1.9, 1.36, "位置 1", fontsize=15, ha="center", color=MUTED)
ax.text(2.9, 1.36, "位置 2", fontsize=15, ha="center", color=MUTED)
ax.text(4.2, 2.0, "shape = (3,)\n1 条轴，长度为 3", fontsize=20, va="center", color=BLUE, linespacing=1.7)
ax.text(0.4, 0.55, "画成一排只是方便看；它没有“行”和“列”两条轴。", fontsize=17, color=MUTED)
ax = axes[1]
_lesson_cells(ax, row_vector.tolist(), 0.8, 1.60, 1.0, 0.85)
ax.text(0.59, 2.02, "行 0", fontsize=16, ha="right", va="center", color=MUTED)
for j in range(3):
    ax.text(1.3 + j, 2.72, f"列 {j}", fontsize=16, ha="center", color=MUTED)
ax.text(4.6, 2.0, "shape = (1, 3)\n2 条轴：1 行 × 3 列", fontsize=20, va="center", color=BLUE, linespacing=1.7)
ax.text(0.4, 0.55, "长度为 1 的轴，仍然算一条轴。", fontsize=18, color=MUTED)
ax = axes[2]
_lesson_cells(ax, column_vector.tolist(), 1.0, 0.50, 1.15, 0.72)
for i in range(3):
    ax.text(0.8, 0.50 + (2 - i + 0.5) * 0.72, f"行 {i}", fontsize=16, ha="right", va="center", color=MUTED)
ax.text(1.58, 2.88, "列 0", fontsize=16, ha="center", color=MUTED)
ax.text(4.2, 2.0, "shape = (3, 1)\n2 条轴：3 行 × 1 列", fontsize=20, va="center", color=BLUE, linespacing=1.7)
fig.text(0.5, 0.025, "三个数没有变，变的是组织方式和取数地址。", ha="center", fontsize=19, color=INK)
savefig(fig, "ch02_vector_shapes")
plt.close(fig)

fig, axes = plt.subplots(3, 1, figsize=(9.4, 10.3))
fig.subplots_adjust(top=0.90, bottom=0.085, hspace=0.20)
fig.suptitle("矩阵乘向量：每行是一份配方", fontsize=23, fontweight="bold", color=INK)
for ax, title in zip(axes, ["输入 x：3 个数，顺序固定", "配方 0：第一行，算出第一个结果", "配方 1：第二行，算出第二个结果"]):
    _lesson_panel(ax, title)
_lesson_cells(axes[0], [x.tolist()], 1.6, 1.4, 1.8, 1.0)
for j, label in enumerate(["第 0 项", "第 1 项", "第 2 项"]):
    axes[0].text(2.5 + 1.8 * j, 2.72, label, ha="center", fontsize=18, color=MUTED)
axes[0].text(4.3, 0.62, "两份配方都使用这同一份输入", fontsize=18, color=MUTED, ha="center")
for row, ax in enumerate(axes[1:]):
    weights = W[row]
    _lesson_cells(ax, [weights.tolist()], 0.25, 2.0, 1.15, 0.78,
                  facecolor="#eaf1f8" if row == 0 else "#e9f2ed")
    ax.text(4.2, 2.37, f"W 的第 {row} 行", fontsize=19, color=BLUE if row == 0 else GREEN, va="center")
    expression = "1 × 1  +  0 × 2  +  2 × 3  =  7" if row == 0 else "0.5 × 1  +  (−1) × 2  +  1 × 3  =  1.5"
    ax.text(0.25, 1.17, expression, fontsize=21, color=BLUE if row == 0 else GREEN)
    ax.text(0.25, 0.30, "同一位置配对相乘 → 把三个乘积加起来", fontsize=18, color=MUTED)
fig.text(0.5, 0.053, "W 的 shape (2, 3)  @  x 的 shape (3,)  →  (2,)", fontsize=20, ha="center", color=INK)
fig.text(0.5, 0.009, "输入有 3 项，每份配方也要 3 项；2 份配方输出 [7, 1.5]。", fontsize=18, ha="center", color=MUTED)
savefig(fig, "ch02_matrix_recipe")
plt.close(fig)

fig = plt.figure(figsize=(9.2, 11.6))
fig.suptitle("三维张量：用三个编号找到一个数", fontsize=23, fontweight="bold", color=INK, y=0.97)
fig.text(0.5, 0.918, "T 的 shape = (2, 3, 2)", ha="center", fontsize=23, color=BLUE)
fig.text(0.5, 0.878, "轴 0：2 个时刻    轴 1：3 只机器人    轴 2：2 项速度", ha="center", fontsize=17, color=MUTED)
axes = fig.subplots(2, 1)
fig.subplots_adjust(top=0.825, bottom=0.19, left=0.10, right=0.96, hspace=0.35)
for page, ax in enumerate(axes):
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.35, 4.8)
    ax.axis("off")
    ax.text(0.0, 4.38, f"时刻 {page}：第 {page} 页", fontsize=21, fontweight="bold", color=ORANGE if page else INK)
    _lesson_cells(ax, T[page].tolist(), 3.1, 0.65, 2.1, 0.90,
                  highlights=((2, 0),) if page == 1 else ())
    for column, label in enumerate(["速度项 0\n前后", "速度项 1\n左右"]):
        ax.text(4.15 + column * 2.1, 3.62, label, fontsize=18, ha="center", color=BLUE, linespacing=1.4)
    for robot in range(3):
        ax.text(2.75, 0.65 + (2 - robot + 0.5) * 0.90, f"机器人 {robot}", fontsize=19, ha="right", va="center",
                color=ORANGE if page == 1 and robot == 2 else MUTED)
    if page == 1:
        ax.annotate("11 在这里", xy=(4.15, 1.10), xytext=(7.9, 0.15),
                    ha="center", fontsize=19, color=ORANGE,
                    arrowprops=dict(arrowstyle="->", lw=2, color=ORANGE, connectionstyle="angle3,angleA=0,angleB=-70"))
fig.text(0.5, 0.129, "T[1, 2, 0] = 11", ha="center", fontsize=26, color=ORANGE, fontweight="bold")
fig.text(0.5, 0.084, "选时刻 1 → 选机器人 2 → 选速度项 0", ha="center", fontsize=20, color=INK)
fig.text(0.5, 0.038, "编号从 0 开始；数据轴表示如何分类，不一定是空间方向。", ha="center", fontsize=17, color=MUTED)
savefig(fig, "ch02_tensor_address")
plt.close(fig)

done()
