"""附录 D 实验：三角函数从 0 补课——四张图 + 手算核对。

运行：uv run python docs/learn-zh/labs/appendix_d_trig.py
纯 CPU，只用标准库 math + numpy + matplotlib。

生成的图：
  figures/appd_angle_radian.png    角度是什么、弧度是什么
  figures/appd_right_triangle.png  直角三角形定义 sin/cos/tan；比值只跟角度有关
  figures/appd_unit_circle.png     单位圆 → 坐标就是 (cos, sin) → 转一圈画出波
  figures/appd_which_axis.png      从哪根轴量起；反过来查角度 arccos
"""

import math

import numpy as np

from _common import banner, check, done, savefig, table, use_headless_matplotlib

plt = use_headless_matplotlib()
from matplotlib.patches import Arc, Circle, Polygon, Wedge  # noqa: E402

C_SIN = "#d1495b"    # sin / 对边 / 竖直方向
C_COS = "#1f77b4"    # cos / 邻边 / 水平方向
C_HYP = "#2f3640"    # 斜边
C_ACC = "#e07b39"    # 强调色
C_GRAY = "#b5b5b5"
BOX = dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.6)


def clean(ax, xlim, ylim, title, axis_hint=True):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if axis_hint:
        ax.axhline(0, color=C_GRAY, lw=1, zorder=0)
        ax.axvline(0, color=C_GRAY, lw=1, zorder=0)
    ax.set_title(title, fontsize=11.5, pad=10)


def ray(ax, deg, r=1.0, **kw):
    th = math.radians(deg)
    ax.plot([0, r * math.cos(th)], [0, r * math.sin(th)], **kw)


# ===========================================================================
banner("1. 常用角度表（角度 / 弧度 / sin / cos / tan）")
rows = []
for deg in [0, 30, 45, 60, 90, 120, 180, 270]:
    th = math.radians(deg)
    tan = "∞（未定义）" if deg == 90 or deg == 270 else f"{math.tan(th):.4f}"
    rows.append([deg, f"{th:.4f}", f"{math.sin(th):.4f}", f"{math.cos(th):.4f}", tan])
table(["角度°", "弧度 rad", "sin", "cos", "tan"], rows)
print("记忆法：sin 0/30/45/60/90 = √0/2, √1/2, √2/2, √3/2, √4/2 = 0, 0.5, 0.707, 0.866, 1")
print("cos 就是把这一列倒过来读。")
check("sin30° = 0.5", abs(math.sin(math.radians(30)) - 0.5) < 1e-12)
check("cos60° = 0.5", abs(math.cos(math.radians(60)) - 0.5) < 1e-12)
check("sin²+cos² = 1（任取一个角）", abs(math.sin(1.234) ** 2 + math.cos(1.234) ** 2 - 1) < 1e-12)
check("sin(90°−θ) = cos θ", abs(math.sin(math.radians(60)) - math.cos(math.radians(30))) < 1e-12)
check("1 弧度 ≈ 57.2958°", abs(math.degrees(1.0) - 57.2958) < 1e-3)
check("arccos(0.342) ≈ 70°（摔倒阈值）", abs(math.degrees(math.acos(0.342)) - 70.0) < 0.05)

# ===========================================================================
banner("2. 图 D1：角度是什么、弧度是什么")
fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.9))

# ---- ① 角度 --------------------------------------------------------------
ax = axes[0]
clean(ax, (-1.55, 1.75), (-1.5, 1.6), "① 角度：从右边这条起始线，逆时针转了多少")
ax.add_patch(Circle((0, 0), 1, fill=False, color=C_GRAY, lw=1.6))
ax.add_patch(Wedge((0, 0), 0.52, 0, 30, facecolor=C_ACC, alpha=0.3, zorder=1))
for deg, style, col in [(0, "-", C_HYP), (30, "--", C_ACC), (45, "--", C_GRAY),
                        (60, "--", C_GRAY), (90, "-", C_HYP), (180, "-", C_HYP), (270, "-", C_HYP)]:
    ray(ax, deg, 1.0, ls=style, color=col, lw=1.8 if style == "-" else 1.2, zorder=2)
    th = math.radians(deg)
    ax.text(1.15 * math.cos(th), 1.15 * math.sin(th), f"{deg}°", fontsize=9.5,
            color=col if style == "-" else "#8a8a8a", ha="center", va="center")
ax.text(0.60, 0.16, "30°", color=C_ACC, fontsize=10.5)
R_DIR = 1.28
ax.add_patch(Arc((0, 0), 2 * R_DIR, 2 * R_DIR, theta1=105, theta2=165,
                 color="#9a9a9a", lw=1.4, zorder=3))
_a = math.radians(105)
ax.annotate("", xy=(R_DIR * math.cos(_a), R_DIR * math.sin(_a)),
            xytext=(R_DIR * math.cos(math.radians(112)), R_DIR * math.sin(math.radians(112))),
            zorder=3, arrowprops=dict(arrowstyle="-|>", color="#9a9a9a", lw=1.4))
ax.text(-1.06, 0.92, "逆时针\n= 正方向", fontsize=9, color="#8a8a8a", ha="center")
ax.text(0.52, -0.14, "起始线", fontsize=9, color="#9a9a9a", ha="center")
ax.text(0, -1.35, "转满一整圈 = 360°　　半圈 = 180°　　直角 = 90°",
        fontsize=10, ha="center", color="#444444")

# ---- ② 弧度 --------------------------------------------------------------
ax = axes[1]
clean(ax, (-1.55, 1.75), (-1.5, 1.6), "② 弧度：拿半径当尺子，去量圆周的弧长")
ax.add_patch(Circle((0, 0), 1, fill=False, color="#e2e2e2", lw=1.2))
for k in range(6):                      # 一整圈能放下 6 段“1 个半径长”的弧
    col = C_COS if k % 2 == 0 else "#7fb3d5"
    ax.add_patch(Arc((0, 0), 2, 2, theta1=math.degrees(k), theta2=math.degrees(k + 1),
                     color=col, lw=7, zorder=2))
    mid = k + 0.5
    ax.text(1.16 * math.cos(mid), 1.16 * math.sin(mid), str(k + 1), fontsize=9.5,
            color="#2c6ea3", ha="center", va="center")
ax.add_patch(Arc((0, 0), 2, 2, theta1=math.degrees(6), theta2=360, color=C_ACC, lw=7, zorder=2))
mid = (6 + 2 * math.pi) / 2
ax.text(1.20 * math.cos(mid), 1.20 * math.sin(mid), "0.28", fontsize=9.5,
        color=C_ACC, ha="center", va="center")
ax.plot([0, 1], [0, 0], color=C_HYP, lw=3.5, zorder=3)
ax.text(0.5, -0.13, "半径 r", fontsize=10, color=C_HYP, ha="center")
ax.annotate("这一段弧的长度 = 1 个半径\n→ 就叫 1 弧度 ≈ 57.3°",
            xy=(math.cos(0.5) * 1.02, math.sin(0.5) * 1.02), xytext=(0.30, 1.22),
            fontsize=9.5, color="#2c6ea3", ha="center",
            arrowprops=dict(arrowstyle="->", color="#2c6ea3", lw=1.2))
ax.text(0, -1.35, "整圈的弧长 = 2πr ≈ 6.28 r　→　整圈 = 2π 弧度 = 360°",
        fontsize=10, ha="center", color="#444444")
ax.text(0, -1.5, "所以 180° = π，　90° = π/2，　30° = π/6 ≈ 0.524",
        fontsize=9.5, ha="center", color="#777777")

fig.suptitle("角度和弧度：同一件事的两把尺子", fontsize=13.5)
fig.subplots_adjust(wspace=0.05)
savefig(fig, "appd_angle_radian")

# ===========================================================================
banner("3. 图 D2：直角三角形里的三个比值")
fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
DEG = 30
TH = math.radians(DEG)

# ---- ① 三条边的名字 -------------------------------------------------------
ax = axes[0]
clean(ax, (-0.35, 2.35), (-0.75, 1.55), "① 先认三条边（都是相对角 θ 说的）", axis_hint=False)
A, B = (0.0, 0.0), (1.75, 0.0)
C = (1.75, 1.75 * math.tan(TH))
ax.add_patch(Polygon([A, B, C], closed=True, facecolor="#f3f6f9", edgecolor="none", zorder=0))
ax.plot([A[0], B[0]], [A[1], B[1]], color=C_COS, lw=5, solid_capstyle="round", zorder=2)
ax.plot([B[0], C[0]], [B[1], C[1]], color=C_SIN, lw=5, solid_capstyle="round", zorder=2)
ax.plot([A[0], C[0]], [A[1], C[1]], color=C_HYP, lw=3, solid_capstyle="round", zorder=2)
ax.add_patch(Polygon([(B[0] - 0.11, 0), (B[0] - 0.11, 0.11), (B[0], 0.11)], closed=True,
                     facecolor="none", edgecolor="#888888", lw=1.2, zorder=3))
ax.add_patch(Arc(A, 0.8, 0.8, theta1=0, theta2=DEG, color=C_ACC, lw=1.8, zorder=3))
ax.text(0.47, 0.08, "θ", color=C_ACC, fontsize=12)
ax.text(0.88, -0.17, "邻边（紧挨着 θ）", color=C_COS, fontsize=10.5, ha="center")
ax.text(1.85, C[1] / 2, "对边\n（θ 对面）", color=C_SIN, fontsize=10.5, ha="left", va="center")
ax.text(0.80, 0.40, "斜边（最长、对着直角）", color=C_HYP, fontsize=10.5, rotation=DEG, ha="center")
ax.text(0.02, 1.34,
        "sin θ = 对边 ÷ 斜边\ncos θ = 邻边 ÷ 斜边\ntan θ = 对边 ÷ 邻边",
        fontsize=11.5, ha="left", va="top", linespacing=1.7,
        bbox=dict(facecolor="#fbfbfb", edgecolor="#dddddd", pad=6))
ax.text(1.0, -0.62, "换一个角来看，“对边”和“邻边”就互换——名字是相对 θ 的",
        fontsize=9, color="#888888", ha="center")

# ---- ② 比值只跟角度有关 ---------------------------------------------------
ax = axes[1]
clean(ax, (-0.35, 2.35), (-0.75, 1.55), "② 三角形放大一倍，比值一点不变", axis_hint=False)
for scale, alpha, lw in [(2.0, 0.35, 2.4), (1.0, 1.0, 4.0)]:
    P, Q = (scale * math.cos(TH), 0.0), (scale * math.cos(TH), scale * math.sin(TH))
    ax.plot([0, P[0]], [0, 0], color=C_COS, lw=lw, alpha=alpha, solid_capstyle="round", zorder=2)
    ax.plot([P[0], Q[0]], [0, Q[1]], color=C_SIN, lw=lw, alpha=alpha, solid_capstyle="round", zorder=2)
    ax.plot([0, Q[0]], [0, Q[1]], color=C_HYP, lw=lw * 0.6, alpha=alpha, solid_capstyle="round", zorder=2)
ax.add_patch(Arc((0, 0), 0.8, 0.8, theta1=0, theta2=DEG, color=C_ACC, lw=1.8, zorder=3))
ax.text(0.47, 0.08, "θ=30°", color=C_ACC, fontsize=10.5)
ax.text(math.cos(TH) + 0.05, math.sin(TH) / 2, "对边 0.50", color=C_SIN, fontsize=10, va="center")
ax.text(2 * math.cos(TH) + 0.05, 2 * math.sin(TH) / 2, "对边 1.00", color=C_SIN,
        fontsize=10, va="center", alpha=0.75)
ax.text(0.52, 0.34, "斜边 1", color=C_HYP, fontsize=10, rotation=DEG)
ax.text(1.32, 0.80, "斜边 2", color=C_HYP, fontsize=10, rotation=DEG, alpha=0.7)
ax.text(1.0, -0.40, "0.50 ÷ 1  =  1.00 ÷ 2  =  0.5", fontsize=12, ha="center", color="#333333")
ax.text(1.0, -0.62, "比值只认角度，不认大小 —— 所以它能被写成一个函数：sin30° = 0.5",
        fontsize=9.5, ha="center", color="#888888")

fig.suptitle("sin、cos、tan 就是直角三角形里的三个比值", fontsize=13.5)
fig.subplots_adjust(wspace=0.05)
savefig(fig, "appd_right_triangle")

# ===========================================================================
banner("4. 图 D3：单位圆 → 坐标 → 波形")
fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8), gridspec_kw=dict(width_ratios=[1, 1.35]))

# ---- ① 单位圆 -------------------------------------------------------------
ax = axes[0]
clean(ax, (-1.45, 1.55), (-1.45, 1.5), "① 斜边取 1：坐标直接就是 (cos θ, sin θ)")
ax.add_patch(Circle((0, 0), 1, fill=False, color=C_GRAY, lw=1.6))
px, py = math.cos(TH), math.sin(TH)
ax.plot([0, px], [0, py], color=C_HYP, lw=2.6, zorder=4)
ax.plot(px, py, "o", color=C_HYP, ms=7, zorder=5)
ax.plot([0, px], [0, 0], color=C_COS, lw=5.5, solid_capstyle="butt", zorder=3)
ax.plot([px, px], [0, py], color=C_SIN, lw=5.5, solid_capstyle="butt", zorder=3)
ax.plot([0, px], [py, py], color=C_SIN, ls="--", lw=1.1, zorder=2)
ax.plot([0, 0], [0, py], color=C_SIN, lw=5.5, alpha=0.45, solid_capstyle="butt", zorder=3)
ax.add_patch(Arc((0, 0), 0.62, 0.62, theta1=0, theta2=DEG, color=C_ACC, lw=1.8, zorder=4))
ax.text(0.36, 0.06, "θ", color=C_ACC, fontsize=12)
ax.text(px / 2, -0.15, f"cos θ = {px:.2f}", color=C_COS, fontsize=10.5, ha="center", bbox=BOX)
ax.text(px + 0.05, py / 2, f"sin θ\n= {py:.2f}", color=C_SIN, fontsize=10.5, va="center")
ax.text(px * 0.55, py * 0.72, "长度 1", color=C_HYP, fontsize=9.5, rotation=DEG)
ax.text(px + 0.04, py + 0.08, f"({px:.2f}, {py:.2f})", color=C_HYP, fontsize=10)
for qx, qy, s in [(0.36, 0.78, "cos +\nsin +"), (-0.62, 0.62, "cos −\nsin +"),
                  (-0.62, -0.62, "cos −\nsin −"), (0.62, -0.62, "cos +\nsin −")]:
    ax.text(qx, qy, s, fontsize=8.5, color="#aaaaaa", ha="center", va="center")
ax.text(0, -1.34, "半径 1 + 勾股定理　→　cos²θ + sin²θ = 1", fontsize=10.5,
        ha="center", color="#444444")

# ---- ② 波形 --------------------------------------------------------------
ax = axes[1]
deg = np.linspace(0, 360, 721)
rad = np.radians(deg)
ax.plot(deg, np.cos(rad), color=C_COS, lw=2.2)
ax.plot(deg, np.sin(rad), color=C_SIN, lw=2.2)
ax.axhline(0, color=C_GRAY, lw=1)
ax.axvline(30, color="#dddddd", lw=1.2, ls="--")
ax.plot(30, math.cos(TH), "o", color=C_COS, ms=7)
ax.plot(30, math.sin(TH), "o", color=C_SIN, ms=7)
ax.annotate(f"θ=30° 这一竖线上的两个高度，\n就是左图量出来的 {math.cos(TH):.2f} 和 {math.sin(TH):.2f}",
            xy=(30, math.sin(TH)), xytext=(44, -0.72), fontsize=9.5, color="#555555",
            arrowprops=dict(arrowstyle="->", color="#999999", lw=1.1))
ax.text(232, 0.86, "cos：从 1 出发", color=C_COS, fontsize=10.5, bbox=BOX)
ax.text(232, 0.58, "sin：从 0 出发", color=C_SIN, fontsize=10.5, bbox=BOX)
ax.set_xlim(0, 360)
ax.set_ylim(-1.25, 1.25)
ax.set_xticks([0, 90, 180, 270, 360])
ax.set_xticklabels(["0°", "90°", "180°", "270°", "360°"])
ax.set_yticks([-1, 0, 1])
ax.grid(alpha=0.22)
ax.set_title("② 让点绕圆走一圈，两个坐标各画出一条波", fontsize=11.5, pad=10)
ax.set_xlabel("转过的角度 θ")

fig.suptitle("单位圆：把三角形装进一个半径为 1 的圆，sin / cos 就变成坐标", fontsize=13.5)
fig.subplots_adjust(wspace=0.14, bottom=0.16)
fig.text(0.5, 0.035,
         "超过 90° 之后不用另学规则：点转到哪个象限，坐标的正负就是那个象限的正负",
         fontsize=9.5, ha="center", color="#888888")
savefig(fig, "appd_unit_circle")

# ===========================================================================
banner("5. 图 D4：从哪根轴量起；反过来查角度")
fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), gridspec_kw=dict(width_ratios=[1, 1.25]))

# ---- ① 同一根箭头，两种量法 ------------------------------------------------
ax = axes[0]
clean(ax, (-0.66, 1.06), (-1.02, 1.42), "① 同一根箭头，从哪根轴量起，公式就换个样子")
vx, vy = math.sin(TH), math.cos(TH)          # 从竖直方向歪开 30°
ax.annotate("", xy=(vx, vy), xytext=(0, 0), zorder=4,
            arrowprops=dict(arrowstyle="-|>", color=C_HYP, lw=2.6))
ax.plot([0, vx], [0, 0], color=C_COS, lw=5.5, solid_capstyle="butt", zorder=3)
ax.plot([0, 0], [0, vy], color=C_SIN, lw=5.5, solid_capstyle="butt", zorder=3)
ax.plot([vx, vx], [0, vy], color="#cccccc", ls="--", lw=1.1, zorder=2)
ax.plot([0, vx], [vy, vy], color="#cccccc", ls="--", lw=1.1, zorder=2)
ax.text(vx + 0.05, vy + 0.03, "一根长度 1 的箭头", color=C_HYP, fontsize=9.5, ha="left")

ax.add_patch(Arc((0, 0), 1.3, 1.3, theta1=60, theta2=90, color=C_SIN, lw=1.8, zorder=4))
ax.annotate("从竖直量起\nθ = 30°", xy=(0.65 * math.cos(math.radians(76)), 0.65 * math.sin(math.radians(76))),
            xytext=(-0.60, 1.16), fontsize=9.8, color=C_SIN, ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=C_SIN, lw=1.1))
ax.add_patch(Arc((0, 0), 0.62, 0.62, theta1=0, theta2=60, color=C_ACC, lw=1.8, zorder=4))
ax.annotate("从水平量起\n= 60°", xy=(0.31 * math.cos(math.radians(28)), 0.31 * math.sin(math.radians(28))),
            xytext=(0.62, 0.30), fontsize=9.8, color=C_ACC, ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=C_ACC, lw=1.1))

ax.text(vx / 2, -0.13, f"水平 {vx:.2f}", color=C_COS, fontsize=10.5, ha="center")
ax.text(-0.05, vy / 2, f"竖直\n{vy:.2f}", color=C_SIN, fontsize=10.5, ha="right", va="center")
ax.text(-0.63, -0.40,
        "从竖直量 θ=30°：(水平, 竖直) = (sin θ, cos θ)\n"
        "从水平量  60°：(水平, 竖直) = (cos 60°, sin 60°)\n"
        "两行数字完全相同，因为 sin θ = cos(90°−θ)",
        fontsize=9.6, ha="left", va="top", linespacing=1.7,
        bbox=dict(facecolor="#fbfbfb", edgecolor="#dddddd", pad=5))
ax.text(0.20, -0.96, "口诀：紧挨着角的那条边用 cos，角对面那条边用 sin",
        fontsize=9.3, ha="center", color="#666666")

# ---- ② 反过来查角度 -------------------------------------------------------
ax = axes[1]
d = np.linspace(0, 180, 721)
ax.plot(d, np.cos(np.radians(d)), color=C_COS, lw=2.2)
ax.axhline(0, color=C_GRAY, lw=1)
for val, col, note in [(0.866, "#8fb8d8", "cos θ = 0.87 → θ = 30°"),
                       (0.5, "#5b95c4", "cos θ = 0.50 → θ = 60°"),
                       (0.342, C_ACC, "cos θ = 0.34 → θ = 70°　←　摔倒阈值")]:
    a = math.degrees(math.acos(val))
    ax.plot([0, a], [val, val], color=col, ls="--", lw=1.3)
    ax.plot([a, a], [val, -1.12], color=col, ls="--", lw=1.3)
    ax.plot(a, val, "o", color=col, ms=6)
    ax.text(a + 3, val + 0.06, note, color=col if col != "#8fb8d8" else "#6f97b5", fontsize=9.5)
ax.set_xlim(0, 180)
ax.set_ylim(-1.15, 1.2)
ax.set_xticks([0, 30, 60, 70, 90, 120, 150, 180])
ax.set_xticklabels(["0°", "30°", "60°", "70°", "90°", "120°", "150°", "180°"], fontsize=9)
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.grid(alpha=0.22)
ax.set_title("② 反过来用：已知比值倒查角度，就是 arccos", fontsize=11.5, pad=10)
ax.set_xlabel("角度 θ")
ax.set_ylabel("cos θ")
ax.text(92, -0.95, "正着查：角度 → 比值　叫 cos\n倒着查：比值 → 角度　叫 arccos",
        fontsize=10, ha="left", va="center",
        bbox=dict(facecolor="#fbfbfb", edgecolor="#dddddd", pad=5))

fig.suptitle("两个最容易踩的坑：角从哪根轴量起；以及怎么倒着查回角度", fontsize=13.5)
fig.subplots_adjust(wspace=0.18)
savefig(fig, "appd_which_axis")

# ===========================================================================
banner("6. 把补课接回项目：倾角 ↔ projected_gravity")
rows = []
for deg in [0, 10, 30, 60, 70, 90]:
    th = math.radians(deg)
    rows.append([deg, f"{math.sin(th):+.3f}", f"{-math.cos(th):+.3f}",
                 f"{math.degrees(math.acos(min(1.0, -(-math.cos(th))))):.1f}"])
table(["倾角°", "g_x = sin θ", "g_z = −cos θ", "arccos(−g_z) 反算回来"], rows)
check("倾角 → g_z → 倾角 能对上", abs(math.degrees(math.acos(math.cos(math.radians(70)))) - 70) < 1e-6)

done()
