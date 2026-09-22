"""第 8 章实验：输入归一化——扁碗变圆、每列各自统计、在线均值与方差、训练时更新 / 推理时冻结、烘焙进同一个模块。

运行：uv run python docs/learn-zh/labs/ch08_normalization.py
纯 CPU，numpy + torch + matplotlib。第 3–6 节直接调用项目训练用的归一化器 rsl_rl 的 EmpiricalNormalization；
第 7 节只读几份源码的文字，不加载机器人、不训练。
小节编号与正文一一对应：实验第 K 节 = 正文 8.K 节（第 7 节对应「映射到项目」）。
正文“改一改”要改的三行都带 `# TWEAK-k:` 标记（第 1、3、5 节各一处）。
"""

import math
from pathlib import Path

import numpy as np
import torch
from rsl_rl.modules import EmpiricalNormalization

from _common import banner, check, done, lines_in_order, savefig, table
from _draw import (BLUE, CELL, CELL_HOT, FAINT, FS_NOTE, FS_SMALL, FS_STEP, FS_TICK, FS_TITLE, GREEN, INK, MUTED, ORANGE,
                   WHITE_BOX, arrow, cell, data_axes, hand, lesson_cells, lesson_figure, lesson_panel, note, panel_note,
                   panel_title, plt)

np.seterr(over="ignore", invalid="ignore")   # 发散时数字会大到 inf / nan——这正是要给读者看的现象
torch.manual_seed(0)
REPO = Path(__file__).resolve().parents[3]


def minus(s: str) -> str:
    return s.replace("-", "−")                   # 打印成正文里的减号


# ---------------------------------------------------------------------------
banner("1. 伏笔兑现：x 的范围放大 10 倍，同一个学习率就飞；归一化之后碗变圆")
X_RANGE = (-10, 10)  # TWEAK-1: (-100, 100)


def make_data(x_range):
    """和第 3 章实验第 5 节同一种造法、同一个种子：50 个点，大致落在 y = 2x + 1 上，带一点噪声。"""
    rng = np.random.default_rng(0)
    X = rng.uniform(*x_range, size=50)
    return X, 2.0 * X + 1.0 + rng.normal(0, 0.1, size=50)


def descend(X, Y, alpha, steps=100):
    """第 3 章 3.5 节的循环：从 w = 0、b = 0 出发，每步 w、b 各减 α × 偏导数。返回 w、b 的轨迹。"""
    w, b, path = 0.0, 0.0, [(0.0, 0.0)]
    for _ in range(steps):
        err = w * X + b - Y
        w, b = w - alpha * np.mean(2 * err * X), b - alpha * np.mean(2 * err)
        path.append((w, b))
    return np.array(path)


def loss(X, Y, w, b) -> float:
    return float(np.mean((w * X + b - Y) ** 2))


def hessian(X):
    """碗在 w、b 两个方向上有多陡：w 动 1，∂L/∂w 动 2 × “x² 的平均”；b 动 1，∂L/∂b 动 2（第 3 章 3.6 节）。"""
    return 2 * np.array([[np.mean(X ** 2), np.mean(X)], [np.mean(X), 1.0]])


def alpha_limit(X) -> float:
    return float(2 / np.linalg.eigvalsh(hessian(X)).max())   # 精确的门槛（w、b 互相牵连，交给机器算）


X_s, Y_s = make_data((-1, 1))
X_b, Y_b = make_data(X_RANGE)
mu_b, sd_b = float(X_b.mean()), float(X_b.std())
X_n = (X_b - mu_b) / sd_b                        # 第 4 章 4.7 节那一步：减均值、除标准差
rng_txt = minus(f"({X_RANGE[0]}, {X_RANGE[1]})")
runs = [("x ∈ (−1, 1)（第 3 章）", X_s, Y_s, 0.1), (f"x ∈ {rng_txt}", X_b, Y_b, 0.1),
        (f"x ∈ {rng_txt}，α 调小", X_b, Y_b, 0.02), (f"x ∈ {rng_txt}，归一化后", X_n, Y_b, 0.1)]
paths, rows = [], []
for name, X, Y, a in runs:
    p = descend(X, Y, a)
    paths.append(p)
    rows.append([name, a, p[-1, 0], p[-1, 1], loss(X, Y, *p[-1])])
table(["数据", "α", "w", "b", "100 步后的损失"], rows)
(w_s, b_s, L_s), (w_b, b_b, L_b), (w_02, b_02, L_02), (w_n, b_n, L_n) = [r[2:] for r in rows]
check("第 3 章那一行原样：w = 2.02442、b = 1.00206、损失 0.00993478",
      (round(w_s, 5), round(b_s, 5), round(L_s, 8)) == (2.02442, 1.00206, 0.00993478))
check(f"x ∈ {rng_txt}、α = 0.1：100 步后损失 {L_b:.3g}，发散", not np.isfinite(L_b) or L_b > 1e6)

m2_s, m2_b = float(np.mean(X_s ** 2)), float(np.mean(X_b ** 2))
print(f"只看 w：w 动 1，∂L/∂w 动 2 × “x² 的平均”。(−1, 1)：2 × {m2_s:.3f} ≈ {2 * m2_s:.2f}，门槛 ≈ {1 / m2_s:.1f}；"
      f"{rng_txt}：2 × {m2_b:.1f} ≈ {2 * m2_b:.0f}，门槛 ≈ {1 / m2_b:.2g}。只看 b：永远是 2，门槛 1")
print(f"精确的门槛：(−1, 1) 是 {alpha_limit(X_s):.4f}，{rng_txt} 是 {alpha_limit(X_b):.3g}")
check("x² 的平均：0.343 → 34.3（放大 10 倍，平方放大 100 倍）", round(m2_s, 3) == 0.343 and round(m2_b, 1) == 34.3)
check("字面值重算：2 × 0.343 ≈ 0.69，2 ÷ 0.69 ≈ 2.9；2 × 34.3 ≈ 69，2 ÷ 69 ≈ 0.029",
      round(2 * 0.343, 2) == 0.69 and round(2 / 0.69, 1) == 2.9 and round(2 * 34.3) == 69 and round(2 / 69, 3) == 0.029)
check("精确门槛：0.9957（第 3 章）→ 0.0292；单看 w 的 0.029 已经很准", round(alpha_limit(X_s), 4) == 0.9957
      and round(alpha_limit(X_b), 4) == 0.0292 and round(1 / m2_b, 3) == 0.029)
check("α = 0.1 是门槛 0.0292 的三倍多；字面值重算 2.9 ÷ 100 ≈ 0.029", 3 < 0.1 / alpha_limit(X_b) < 4
      and round(2.9 / 100, 3) == 0.029)
check("单看 w：每步乘 1 − 0.69α，乘数不小于 −1 ⇔ α ≤ 2 ÷ 0.69（边界上正好是 −1）", math.isclose(1 - 0.69 * (2 / 0.69), -1))
print(f"圈的宽窄：w 方向只比 b 方向窄 √{m2_b:.1f} ≈ {math.sqrt(round(m2_b, 1)):.2f} 倍（不是 {m2_b:.1f} 倍：高度跟距离的平方走）")
check("圈在 w 方向只比 b 方向窄 √34.3 ≈ 5.86 倍（5.86 × 5.86 ≈ 34.3）", round(math.sqrt(34.3), 2) == 5.86
      and round(5.86 * 5.86, 1) == 34.3 and abs(math.sqrt(m2_b) - 5.86) < 0.01)

slope_b, icpt_b = np.linalg.lstsq(np.c_[X_b, np.ones_like(X_b)], Y_b, rcond=None)[0]   # 这 50 个点上最好的直线
L_best = loss(X_b, Y_b, slope_b, icpt_b)
dist_w = paths[1][:4, 0] - slope_b              # α = 0.1：w 离碗底的距离，前 3 步
ratios = dist_w[1:] / dist_w[:-1]
print("α = 0.1 时 w 离碗底的距离：", "  →  ".join(f"{d:.1f}" for d in dist_w), "  相邻两个相除：",
      "、".join(f"{r:.2f}" for r in ratios))
check("w 离碗底的距离 −2.0 → 11.8 → −69.2 → 405.1，每步约乘 −5.9 = 1 − 0.1 × 69",
      [round(d, 1) for d in dist_w] == [-2.0, 11.8, -69.2, 405.1] and all(abs(r + 5.9) < 0.1 for r in ratios)
      and round(1 - 0.1 * 69, 1) == -5.9)
check("相邻两个相除是 −5.91、−5.85、−5.86，都在 −5.9 上下（b 也在动）；拿印出来的 −2.0、11.8、−69.2、405.1 相除也都约 −5.9",
      [round(float(r), 2) for r in ratios] == [-5.91, -5.85, -5.86]
      and all(round(q, 1) == -5.9 for q in (11.8 / -2.0, -69.2 / 11.8, 405.1 / -69.2)))
check("α = 0.1 第 1 步：w 从 0 跳到 13.8，离碗底 11.8", round(float(paths[1][1, 0]), 1) == 13.8 and round(float(dist_w[1]), 1) == 11.8)
print(f"α = 0.02：b 每步只乘 1 − 0.02 × 2 = 0.96；100 步后 b = {b_02:.3f}，离最佳的 {icpt_b:.4f} 还差 {icpt_b - b_02:.3f}"
      f"（0.96^100 = {0.96 ** 100:.4f}）")
check("α = 0.02：w 不飞了，可 100 步后 b 还差 0.017（= 起点差 1.0 × 0.96^100），损失 0.0102，没到碗底的 0.00993",
      round(icpt_b - b_02, 3) == 0.017 and round(0.96 ** 100, 3) == 0.017 and round(L_02, 4) == 0.0102
      and round(L_best, 5) == 0.00993 and L_02 > L_best)

print(f"归一化：μ = {mu_b:.3f}，σ = {sd_b:.3f}；x̃ 的平均 {np.mean(X_n):.1g}，x̃² 的平均 {np.mean(X_n ** 2):.6f}；"
      f"两个方向都是 {hessian(X_n)[0, 0]:g}，门槛 {alpha_limit(X_n):g}")
check(f"x ∈ {rng_txt} 的 μ = 0.537、σ = 5.829", round(mu_b, 3) == 0.537 and round(sd_b, 3) == 5.829)
check("归一化后 x̃ 的平均 0、x̃² 的平均 1 → 两个方向一样陡（都是 2），门槛正好 1",
      abs(np.mean(X_n)) < 1e-12 and math.isclose(np.mean(X_n ** 2), 1.0) and np.allclose(hessian(X_n), 2 * np.eye(2))
      and math.isclose(alpha_limit(X_n), 1.0))
check("归一化后 100 步的损失 0.00993373", round(L_n, 8) == 0.00993373)
check(f"……正好是这 50 个点上最好的直线的损失（最小二乘算出 {L_best:.8f}）", abs(L_n - L_best) < 1e-10)
slope_s, icpt_s = np.linalg.lstsq(np.c_[X_s, np.ones_like(X_s)], Y_s, rcond=None)[0]
check("两组 x 只差放大倍数：(−1, 1) 那组的碗底也是 0.00993373；第 3 章那一行 100 步后（0.00993478）还差一点点",
      abs(loss(X_s, Y_s, slope_s, icpt_s) - L_best) < 1e-12 and round(L_best, 8) == 0.00993373 and L_s > L_n)
X_s_n = (X_s - X_s.mean()) / X_s.std()
check("(−1, 1) 那串 x 和放大后的那串 x，归一化之后是同一串数：放大的倍数被消掉了", np.allclose(X_s_n, X_n))
print(f"归一化后的 w 不再是 2：它是“x̃ 动 1（= x 动一个 σ）时 y 动多少” = {slope_b:.4f} × {sd_b:.3f} = {slope_b * sd_b:.2f}")
check("w̃ = 11.67 = 2.0026 × 5.829（原来的斜率 × σ）", round(w_n, 2) == 11.67 and round(slope_b, 4) == 2.0026
      and round(2.0026 * 5.829, 2) == 11.67)
check("归一化那一行从 (0, 0) 出发，碗底在 (11.67, 2.08)：w 差 11.67、b 差 2.08（这时 b 的最好值就是 y 的平均）",
      round(w_n, 2) == 11.67 and round(b_n, 2) == 2.08 and round(float(Y_b.mean()), 2) == 2.08)

X_t = make_data((-0.1, 0.1))[0]                   # 自测：范围缩小 10 倍
m2_t = float(np.mean(X_t ** 2))
mult_t = 1 - 0.1 * 2 * m2_t
steps_t = math.log(0.1) / math.log(mult_t)
print(f"自测 (−0.1, 0.1)：x² 的平均 {m2_t:.5f}；2 × 0.00343 = 0.00686，门槛 2 ÷ 0.00686 ≈ {2 / 0.00686:.0f}；"
      f"α = 0.1 时 w 每步乘 {mult_t:.4f}，缩到十分之一要 {steps_t:.0f} 步")
check("自测：0.00343 → 2 × 0.00343 = 0.00686，门槛约 290；每步乘 0.9993，缩到十分之一要三千多步",
      round(m2_t, 5) == 0.00343 and round(2 * 0.00343, 5) == 0.00686 and round(2 / 0.00686, -1) == 290
      and round(mult_t, 4) == 0.9993 and 3000 < steps_t < 4000)


def quad_path(H, alpha, start, steps):
    """碗是二次的：离碗底的偏差 Δ 每步变成 Δ − α·H·Δ（和上面的循环完全等价）。"""
    out = [np.asarray(start, float)]
    for _ in range(steps):
        out.append(out[-1] - alpha * H @ out[-1])
    return np.array(out)


H_raw, H_nrm = hessian(X_b), hessian(X_n)
start = np.array([0.0 - slope_b, 0.0 - icpt_b])   # 起点 (0, 0) 离碗底 (2.0026, 1.0019) 有多远
p_fly = quad_path(H_raw, 0.1, start, 1)
p_slow = quad_path(H_raw, 0.02, start, 100)
p_round = quad_path(H_nrm, 0.1, start, 30)
shrink = np.linalg.norm(p_round[1:], axis=1) / np.linalg.norm(p_round[:-1], axis=1)
check("圆碗上 α = 0.1：每一步离碗底的距离都正好乘 0.8 = 1 − 0.1 × 2", np.allclose(shrink, 0.8))
check("二次碗的偏差递推和上面的循环走出同一条路（α = 0.02）", np.allclose(p_slow, paths[2] - [slope_b, icpt_b]))

# ---------------------------------------------------------------------------
banner("1b. 画图：figures/ch08_bowl_round.png（同样的高度差：扁碗的圈挤成一条缝，归一化后是正圆）")
if X_RANGE != (-10, 10):
    print("  X_RANGE 改过了：这张讲解图里的手算是照着 (−10, 10) 写的，跳过。看上面那张表就行。")
else:
    fig = plt.figure(figsize=(10.2, 13.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.92], hspace=0.42, wspace=0.06,
                          left=0.09, right=0.99, top=0.9, bottom=0.085)
    fig.suptitle("归一化把扁碗变圆：同一个 α，一个飞出去，一个直奔碗底", fontsize=FS_TITLE, fontweight="bold",
                 color=INK, y=0.975)
    g = np.linspace(-3.2, 3.2, 401)
    DW, DB = np.meshgrid(g, g)
    for row, (H, title, note_txt) in enumerate([
        (H_raw, f"① x ∈ (−10, 10) 原样：w 方向陡 {H_raw[0, 0] / 2:.1f} 倍，圈挤成一条缝",
         "α = 0.1：第 1 步 w 就冲出图外；α 小到 w 不飞（0.02），b 方向又每步只挪一点点。\n"
         f"灰圈：损失比碗底高 1、2、……、9（高度差相等）；w 方向只窄 √{m2_b:.1f} ≈ {math.sqrt(round(m2_b, 1)):.2f} 倍。"),
        (H_nrm, "② 归一化之后：两个方向一样陡，圈是正圆",
         "起点和 ① 同一处，同一个 α = 0.1：每一步离碗底的距离都乘 0.8，直奔碗底。\n"
         "灰圈的高度和 ① 一样；两幅图刻度相同。")]):
        ax = fig.add_subplot(gs[row, 0])
        data_axes(ax, "w 离碗底多远", "b 离碗底多远")
        Z = 0.5 * (H[0, 0] * DW ** 2 + 2 * H[0, 1] * DW * DB + H[1, 1] * DB ** 2)
        ax.contour(DW, DB, Z, levels=np.arange(1, 10), colors=FAINT, linewidths=1.3)
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
        ax.set_aspect("equal")
        ax.set_xticks(range(-3, 4))
        ax.set_yticks(range(-3, 4))
        ax.plot(0, 0, marker="*", ms=18, color=INK, zorder=7)
        ax.plot(*start, "o", ms=9, color=INK, zorder=7)
        ax.text(start[0] - (0.1 if row == 0 else 0.25), start[1] - 0.3, "起点" if row == 0 else "起点（和 ① 同一处）", color=INK,
                fontsize=FS_SMALL, ha="center" if row == 0 else "left", va="top",
                bbox=WHITE_BOX)
        ax_t = fig.add_subplot(gs[row, 1])
        lesson_panel(ax_t, xmax=10, ymax=10)
        if row == 0:
            (x0, y0), (x1, y1) = p_fly[0], p_fly[1]
            y_edge = y0 + (3.2 - x0) * (y1 - y0) / (x1 - x0)
            arrow(ax, (x0, y0), (3.2, y_edge), ORANGE, lw=3.5, zorder=6)
            ax.text(0.6, -1.55, f"α = 0.1：第 1 步\nw 跳到 {paths[1][1, 0]:.1f}（图外）\n离碗底 {dist_w[1]:.1f}", color=ORANGE,
                    fontsize=FS_SMALL, va="top", bbox=WHITE_BOX)
            ax.plot(p_slow[:, 0], p_slow[:, 1], "-o", color=BLUE, ms=4.5, lw=1.8, zorder=8)
            ax.text(0.35, 1.9, "α = 0.02：\n先左右跳几下，\n再沿 b 慢慢爬", color=BLUE, fontsize=FS_SMALL, va="center",
                    bbox=WHITE_BOX)
            hand(ax_t, 0.4, 9.2, f"只看 w：2 × {m2_b:.1f} ≈ {2 * m2_b:.0f}", color=INK)
            hand(ax_t, 0.4, 7.9, "只看 b：2", color=INK)
            hand(ax_t, 0.4, 6.1, "α = 0.1，w 每步乘", color=ORANGE)
            hand(ax_t, 0.4, 5.0, "1 − 0.1 × 69 = −5.9", color=ORANGE)
            note(ax_t, 0.4, 4.0, "大小超过 1 → 发散")
            hand(ax_t, 0.4, 2.5, "α = 0.02，b 每步乘", color=BLUE)
            hand(ax_t, 0.4, 1.4, "1 − 0.02 × 2 = 0.96", color=BLUE)
            note(ax_t, 0.4, 0.4, "每步只缩一点点 → 慢")
        else:
            ax.plot(p_round[:, 0], p_round[:, 1], "-o", color=GREEN, ms=5, lw=1.8, zorder=5)
            ax.text(-2.9, -2.2, "α = 0.1：一条直线\n走进碗底", color=GREEN, fontsize=FS_SMALL, va="center",
                    bbox=WHITE_BOX)
            hand(ax_t, 0.4, 9.2, "x̃² 的平均 = 1", color=INK)
            hand(ax_t, 0.4, 7.9, "只看 w：2 × 1 = 2", color=INK)
            hand(ax_t, 0.4, 6.6, "只看 b：2", color=INK)
            note(ax_t, 0.4, 5.5, "两个方向一样陡")
            hand(ax_t, 0.4, 3.9, "α = 0.1，两个方向都乘", color=GREEN)
            hand(ax_t, 0.4, 2.8, "1 − 0.1 × 2 = 0.8", color=GREEN)
            note(ax_t, 0.4, 1.7, "每步离碗底的距离 × 0.8")
        panel_title(fig, [ax], title)
        panel_note(fig, [ax], note_txt)
    savefig(fig, "ch08_bowl_round")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("2. 观测的各项差多少：先看 3 只机器人 × 2 项的迷你表，再看 61 项")
mini = np.array([[5.0, 0.2],                      # 机器人 A：关节速度 5 rad/s，投影重力前后分量 0.2
                 [1.0, 0.0],                      # 机器人 B
                 [-3.0, -0.2]])                   # 机器人 C   （为了好算编的数）
col_mean = mini.mean(axis=0)                      # 每一列跨机器人的平均：第 2 章 2.7 节的 dim=0
col_mean_t = torch.tensor(mini).mean(dim=0)
spread = np.abs(mini - col_mean).max(axis=0)
print("迷你表 [3, 2]：", minus(str(mini.tolist())))
print(f"每列的平均（dim=0）：{minus(str(col_mean.tolist()))}；每列离自己的平均最远：{spread.tolist()}，相差 {spread[0] / spread[1]:g} 倍；"
      f"六个数混在一起的平均：{mini.mean():g}")
check("每列平均：关节速度 (5 + 1 − 3) ÷ 3 = 1，重力分量 (0.2 + 0 − 0.2) ÷ 3 = 0；torch 的 mean(dim=0) 一样",
      np.allclose(col_mean, [1, 0]) and np.allclose(col_mean_t.numpy(), col_mean) and tuple(col_mean_t.shape) == (2,))
check("两列离自己的平均最远各是 4 和 0.2，差 20 倍；混着算的平均 (5 + 1 − 3 + 0.2 + 0 − 0.2) ÷ 6 = 0.5",
      np.allclose(spread, [4, 0.2]) and math.isclose(spread[0] / spread[1], 20) and math.isclose(mini.mean(), 0.5))

BLOCKS = [("身体角速度", 3, 3.0, "rad/s"), ("投影重力", 3, 1.0, "无单位"), ("关节角 − HOME", 14, 0.5, "rad"),
          ("关节速度", 14, 8.0, "rad/s"), ("上一步动作", 14, 0.5, "rad")]
COMMANDS = [("走的速度 前后", 0.4), ("走的速度 左右", 0.3), ("转的速度", 1.0),              # twist：m/s、m/s、rad/s
            ("头 neck_pitch", 0.05), ("头 head_pitch", 0.05), ("头 head_yaw", 0.07), ("头 head_roll", 0.015),
            ("身体 x", 0.005), ("身体 y", 0.005), ("身体 z", 0.005),                          # m
            ("身体 roll", 0.05), ("身体 pitch", 0.05), ("身体 yaw", 0.05)]                    # rad
table(["观测块", "列数", "典型幅度", "单位"], [[n, k, f"±{a:g}", u] for n, k, a, u in BLOCKS]
      + [["命令", len(COMMANDS), "±0.005 到 ±1", "m/s、rad/s、m、rad"]])
amps = np.concatenate([np.full(k, a) for _, k, a, _ in BLOCKS] + [np.array([a for _, a in COMMANDS])])
check("3 + 3 + 14 + 14 + 14 = 48 列本体感觉 + 13 列命令 = 61 列", sum(k for _, k, _, _ in BLOCKS) == 48
      and len(COMMANDS) == 13 and len(amps) == 61)
check("关节速度 ±8 是投影重力 ±1 的 8 倍、走路速度命令 ±0.4 的 20 倍、身体位置命令 ±0.005 的 1600 倍",
      8 / 1 == 8 and math.isclose(8 / 0.4, 20) and math.isclose(8 / 0.005, 1600) and math.isclose(amps.max() / amps.min(), 1600))
cfg_path = REPO / "src" / "mjlab_microduck" / "tasks" / "microduck_velocity_env_cfg.py"
CMD_LINES = ["joint_pos_action.scale = 1.0",
             "command.ranges.lin_vel_x = (-0.4, 0.4)", "command.ranges.lin_vel_y = (-0.3, 0.3)",
             "command.ranges.ang_vel_z = (-1.0, 1.0)",
             "(-0.05, 0.05),    # neck_pitch", "(-0.05, 0.05),    # head_pitch", "(-0.07, 0.07),    # head_yaw",
             "(-0.015, 0.015),  # head_roll", "(-0.005, 0.005),  # x (m)", "(-0.005, 0.005),  # y (m)",
             "(-0.005, 0.005),  # z (m)", "(-0.05, 0.05),    # roll (rad)", "(-0.05, 0.05),    # pitch (rad)",
             "(-0.05, 0.05),    # yaw (rad)"]
if cfg_path.is_file():
    check("命令 13 列的采样范围（训练开始时）和 env cfg 一致；动作的 scale = 1.0，所以“上一步动作”的单位就是 rad",
          lines_in_order(cfg_path.read_text(encoding="utf-8"), CMD_LINES))
    cfg_all = cfg_path.read_text(encoding="utf-8")
    HEAD_LAST = '{"step": 2000 * 24, "ranges": ((-1.10, 1.10),  (-1.10, 1.10),  (-1.40, 1.40),  (-0.31, 0.31))},'
    at_body = cfg_all.find('cfg.curriculum["body_pose_range"]')
    body_block = cfg_all[at_body:cfg_all.find("cfg.curriculum[", at_body + 10)]
    print("头的 4 个命令：训练开始时 ±0.015–±0.07 rad，第 2000 次迭代起放宽到 ±0.31–±1.4 rad；身体的 6 个只有一档，不变")
    check("命令范围会变：头的 4 个最后放宽到 ±1.10、±1.10、±1.40、±0.31 rad；身体姿态的课程只有一档（一直是开始时的范围）",
          lines_in_order(cfg_all, ['cfg.curriculum["head_pose_range"]', HEAD_LAST]) and at_body > 0
          and body_block.count('"step"') == 1)
else:
    print("  （没找到项目的 env cfg，跳过这一项）")

# ---------------------------------------------------------------------------
banner("3. 归一化：一列先减自己的平均，再除以自己的标准差（分母加一个小数 ε）")
EPS = 0.01  # TWEAK-2: 0.0
mu1, sd1 = 2.0, 4.0                               # 关节速度这一列攒出来的平均和标准差（rad/s）
z6, zm2, z2 = [(x - mu1) / (sd1 + EPS) for x in (6.0, -2.0, 2.0)]
print(minus(f"手算：(6 − 2) ÷ (4 + {EPS:g}) = 4 ÷ {sd1 + EPS:g} = {z6:.6f}；(−2 − 2) ÷ {sd1 + EPS:g} = {zm2:.6f}；(2 − 2) ÷ {sd1 + EPS:g} = {z2:g}"))
check("6 → 0.997506，−2 → −0.997506，2 → 0", round(z6, 6) == 0.997506 and round(zm2, 6) == -0.997506 and z2 == 0)
check("字面值重算：4 ÷ 4.01 = 0.997506", round(4 / 4.01, 6) == 0.997506)
norm1 = EmpiricalNormalization(1, eps=EPS)        # 项目用的归一化器，1 列
norm1.update(torch.tensor([[-2.0], [6.0]]))       # 喂两个读数：平均 2，标准差 4
out1 = norm1(torch.tensor([[6.0], [-2.0], [2.0]])).squeeze(1)
print(f"EmpiricalNormalization：攒出 μ = {norm1.mean.item():g}、σ = {norm1.std.item():g}；6、−2、2 → {minus(str([round(v, 6) for v in out1.tolist()]))}")
check("项目的归一化器算出同样的三个数", norm1.mean.item() == 2 and norm1.std.item() == 4
      and torch.allclose(out1, torch.tensor([z6, zm2, z2])))
check("它的 ε 默认就是 0.01（eps=1e-2）", EmpiricalNormalization(1).eps == 0.01)
z10, zm6 = (10 - mu1) / (sd1 + EPS), (-6 - mu1) / (sd1 + EPS)
check("不会被夹在 ±1 里：10 → 1.995012，−6 → −1.995012", round(z10, 6) == 1.995012 and round(zm6, 6) == -1.995012)
check("关节真的不动（读数 0）换算成 (0 − 2) ÷ 4.01 = −0.4988，不是 0", round((0 - mu1) / (sd1 + EPS), 4) == -0.4988)
const = np.array([3.0, 3.0, 3.0])                 # 自测：一列从来不变
z_const = (const - const.mean()) / (const.std() + EPS)
print(f"一列从来不变（3、3、3）：σ = {const.std():g}，归一化后 {z_const.tolist()}")
check("σ = 0 的列：加了 ε 不会除以 0，但分子也是 0，结果全是 0——没有信息", np.all(z_const == 0))
check("自测（天气）：(−20 − (−18)) ÷ 5 = −0.4，(2 − 14) ÷ 3 = −4", math.isclose((-20 + 18) / 5, -0.4)
      and math.isclose((2 - 14) / 3, -4))

sd_cols = mini.std(axis=0)                        # 除以条数 3（第 4 章 4.4 节的算法）
mini_n = (mini - col_mean) / (sd_cols + EPS)
print(f"迷你表各列：σ = {sd_cols[0]:.3f}、{sd_cols[1]:.4f}；归一化后 A 行 = ({mini_n[0, 0]:.3f}, {mini_n[0, 1]:.3f})，"
      f"B 行 = ({mini_n[1, 0]:g}, {mini_n[1, 1]:g})，C 行 = ({mini_n[2, 0]:.3f}, {mini_n[2, 1]:.3f})")
check("迷你表：σ = √(32/3) = 3.266、√(0.08/3) = 0.1633；A 行换算成 1.221 和 1.154",
      round(sd_cols[0], 3) == 3.266 and round(sd_cols[1], 4) == 0.1633 and round(mini_n[0, 0], 3) == 1.221
      and round(mini_n[0, 1], 3) == 1.154)
check("字面值重算：4 ÷ (3.266 + 0.01) = 1.221，0.2 ÷ (0.1633 + 0.01) = 1.154",
      round(4 / 3.276, 3) == 1.221 and round(0.2 / 0.1733, 3) == 1.154)
two = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
check("只挪零点、换刻度，不改形状：只有两个值的列，归一化后还是只有两个值",
      len(np.unique(np.round((two - two.mean()) / (two.std() + EPS), 12))) == 2)

gen = torch.Generator().manual_seed(0)            # 按上面的典型幅度造一批假观测：4096 行 × 61 列，每列在 ±幅度 里均匀散开
obs = (torch.rand(4096, 61, generator=gen, dtype=torch.float64) * 2 - 1) * torch.tensor(amps)
norm61 = EmpiricalNormalization(61, eps=EPS).double()
norm61.update(obs)
sd_before = norm61.std.numpy()
sd_after = norm61(obs).std(dim=0, unbiased=False).numpy()
theory_after = (amps / math.sqrt(3)) / (amps / math.sqrt(3) + EPS)   # 均匀散开时 σ = 幅度 / √3
small = sd_after[48 + 3:]
sd_jv, sd_body = float(sd_before[20:34].mean()), float(sd_before[55:58].mean())   # 关节速度 14 列、身体位置命令 3 列
print(f"61 列的 σ：关节速度那 14 列平均 {sd_jv:.2f}，身体位置命令那 3 列平均 {sd_body:.4f}，相差 {sd_jv / sd_body:.0f} 倍")
print(f"归一化后：{int((sd_after > 0.94).sum())} 列在 0.94 以上；头和身体的 10 列小命令在 {small.min():.2f}–{small.max():.2f}")
check(f"两头的 σ 相差约 1600 倍（实测 {sd_jv / sd_body:.0f}）；关节速度 ≈ 4.6，身体位置命令 ≈ 0.003",
      abs(sd_jv / sd_body / 1600 - 1) < 0.05 and round(sd_jv, 1) == 4.6 and round(sd_body, 3) == 0.003)
check("归一化后 51 列在 0.94 以上；10 列小命令被压到 0.22–0.80", int((sd_after > 0.94).sum()) == 51
      and round(small.min(), 2) == 0.22 and round(small.max(), 2) == 0.80)
check("均匀散开时，标准差约是幅度的 0.58 倍（1 ÷ √3）", round(1 / math.sqrt(3), 2) == 0.58)
sd_small = sd_before[48 + 3:]                     # 头 4 列 + 身体 6 列命令，换算前的 σ
print(f"10 列小命令换算前的 σ：{sd_small.min():.4f}–{sd_small.max():.4f}；比 ε = 0.01 还小的有 {int((sd_small < 0.01).sum())} 列，"
      f"最大的是 ε 的 {sd_small.max() / 0.01:.1f} 倍")
check("10 列小命令的 σ 只有 0.003–0.04：最小的比 ε = 0.01 还小，最大的也只有 ε 的 4 倍",
      round(float(sd_small.min()), 3) == 0.003 and round(float(sd_small.max()), 2) == 0.04 and sd_small.min() < 0.01
      and round(float(sd_small.max()) / 0.01) == 4)
check(f"和理论值 σ ÷ (σ + ε) 对得上（容差 0.01，当前 ε = {EPS:g}）", np.allclose(sd_after, theory_after, atol=0.01))

# ---------------------------------------------------------------------------
banner("3b. 画图：figures/ch08_one_column.png（一列换一把尺子：零点放在平均值，一格是一个标准差）")
fig, axes = lesson_figure(2, "一列输入换一把尺子：零点放在平均值，一格是一个标准差", panel_height=3.7, width=10.5)
ax = axes[0]
lesson_panel(ax, "① 同一个读数，两把尺子", xmax=10.5, ymax=4.4)
pos = lambda raw: 2.0 + (raw + 6) * 7.4 / 16      # 原始读数 → 图上的横坐标（两把尺子对齐）
for y_line, ticks, labels, unit, ruler in [(2.75, range(-6, 11, 2), [minus(str(t)) for t in range(-6, 11, 2)], "rad/s",
                                            "原来的尺子"),
                                           (0.75, [mu1 + k * (sd1 + EPS) for k in (-2, -1, 0, 1, 2)],
                                            ["−2", "−1", "0", "1", "2"], "几个 σ + ε", "新的尺子")]:
    ax.plot([pos(-6), pos(10)], [y_line, y_line], color=INK, lw=2)
    for t, lab in zip(ticks, labels):
        ax.plot([pos(t)] * 2, [y_line - 0.08, y_line + 0.08], color=INK, lw=1.5)
        ax.text(pos(t), y_line - 0.2, lab, fontsize=FS_TICK, color=MUTED, ha="center", va="top")
    ax.text(pos(10) + 0.12, y_line, unit, fontsize=FS_SMALL, color=MUTED, va="center")
    ax.text(0.0, y_line, ruler, fontsize=FS_SMALL, color=INK, va="center")
ax.annotate("", xy=(pos(6), 3.2), xytext=(pos(2), 3.2), arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=2))
ax.text(pos(4), 3.28, "σ = 4", color=ORANGE, fontsize=FS_SMALL, ha="center", va="bottom")
for raw, z, colr in [(-2.0, zm2, MUTED), (2.0, z2, ORANGE), (6.0, z6, BLUE)]:
    ax.plot([pos(raw), pos(mu1 + z * (sd1 + EPS))], [2.2, 0.95], ls="--", color=colr, lw=1.5)
    ax.plot(pos(raw), 2.75, "o", ms=11, color=colr, zorder=5)
    ax.plot(pos(mu1 + z * (sd1 + EPS)), 0.75, "o", ms=11, color=colr, zorder=5)
ax.text(pos(2) - 0.12, 1.6, "平均 μ = 2 → 0", color=ORANGE, fontsize=FS_SMALL, ha="right", va="center", bbox=WHITE_BOX)
ax.text(pos(6) + 0.15, 1.6, f"读数 6 → {z6:.4f}", color=BLUE, fontsize=FS_SMALL, va="center", bbox=WHITE_BOX)
ax = axes[1]
lesson_panel(ax, "② 手算：先减 μ，再除以 σ + ε", xmax=10.5, ymax=4.4)
hand(ax, 0.3, 3.1, f"读数 6：   (6 − 2) ÷ (4 + {EPS:g}) = 4 ÷ {sd1 + EPS:g} = {z6:.6f}", color=BLUE)
hand(ax, 0.3, 2.2, f"读数 −2：(−2 − 2) ÷ {sd1 + EPS:g} = {z6 * -1:.6f}".replace("-", "−"), color=MUTED)
hand(ax, 0.3, 1.3, f"读数 2：   (2 − 2) ÷ {sd1 + EPS:g} = 0", color=ORANGE)
note(ax, 0.3, 0.4, "减 2：比平常高多少；除以 4.01：这相当于几个平常的起伏。0 = “在平常值附近”，不是“不动”。")
savefig(fig, "ch08_one_column")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("3c. 画图：figures/ch08_obs_scales.png（61 列的标准差：换算前差 1600 倍，换算后大多在 1 附近）")
fig, axes = plt.subplots(2, 1, figsize=(11.5, 11.0), sharex=True)
fig.subplots_adjust(left=0.1, right=0.98, top=0.85, bottom=0.1, hspace=0.72)
fig.suptitle("61 列各用自己的尺子：换算前差 1600 倍，换算后大多在 1 附近", fontsize=FS_TITLE, fontweight="bold",
             color=INK, y=0.97)
colors = [BLUE, GREEN, MUTED, ORANGE, FAINT]
bar_colors = np.concatenate([[colors[i]] * k for i, (_, k, _, _) in enumerate(BLOCKS)] + [[INK] * 13])
idx = np.arange(1, 62)
edges = np.cumsum([0] + [k for _, k, _, _ in BLOCKS] + [13])
for ax, vals in [(axes[0], sd_before), (axes[1], sd_after)]:
    data_axes(ax, "观测的第几项（1–61）", "标准差（对数刻度）")
    ax.bar(idx, vals, color=bar_colors, width=0.8)
    ax.set_yscale("log")
    ax.set_ylim(0.001, 12)
    ax.set_yticks([0.001, 0.01, 0.1, 1, 10])
    ax.set_yticklabels(["0.001", "0.01", "0.1", "1", "10"])
    ax.set_xlim(0.2, 61.8)
    ax.tick_params(labelbottom=True)
    ax.axhline(1.0, color=INK, lw=1.2, ls="--", zorder=0)
    for e in edges[1:-1]:
        ax.axvline(e + 0.5, color=FAINT, lw=1.0, ls=":")
label_at = {"身体角速度": ("角速度", 0.4, "left"), "投影重力": ("重力", 5.9, "center")}   # 两个 3 列的窄块，名字错开放
for (name, k, _, _), lo, colr in zip(BLOCKS + [("命令", 13, 0, "")], edges[:-1], [BLUE, GREEN, MUTED, ORANGE, MUTED, INK]):
    txt, x_at, ha = label_at.get(name, (name, lo + (k + 1) / 2, "center"))
    axes[0].text(x_at, 1.03, txt, transform=axes[0].get_xaxis_transform(), fontsize=FS_TICK if k > 3 else 14,
                 color=colr, ha=ha, va="bottom", fontweight="bold")
arrow_kw = dict(arrowstyle="-|>", lw=1.8)
axes[0].annotate(f"关节速度 ≈ {sd_jv:.1f}", xy=(34.2, sd_jv), xytext=(38, 2.4), fontsize=FS_SMALL, color=ORANGE,
                 arrowprops=dict(color=ORANGE, **arrow_kw), bbox=WHITE_BOX)
axes[0].annotate(f"身体位置命令\n≈ {sd_body:.3f}", xy=(57, sd_body * 1.08), xytext=(57, 0.12), fontsize=FS_SMALL,
                 color=INK, ha="center", va="bottom", arrowprops=dict(color=INK, **arrow_kw), bbox=WHITE_BOX)
axes[1].annotate(f"σ 只有 {sd_small.min():.3f}–{sd_small.max():.2f} 的 10 列：\n被 ε 压到 {small.min():.2f}–{small.max():.2f}",
                 xy=(57, small.min() * 1.08), xytext=(42.5, 1.6), fontsize=FS_SMALL, color=INK, va="bottom", arrowprops=dict(color=INK, **arrow_kw),
                 bbox=WHITE_BOX)
for ax, ttl, nt in [
    (axes[0], "① 原样：每一列的标准差 σ（按典型幅度、训练开始时的命令范围造的示意数据）", f"关节速度 σ ≈ {sd_jv:.1f}，身体位置命令 σ ≈ {sd_body:.3f}：相差约 1600 倍。"),
    (axes[1], "② 每列减自己的 μ、除以自己的 σ + ε 之后（同一批示意数据）",
     f"51 列落在 0.94–1 之间；σ 只有 {sd_small.min():.3f}–{sd_small.max():.2f} 的 10 列小命令，"
     f"被 ε = {EPS:g} 压到 {small.min():.2f}–{small.max():.2f}。")]:
    panel_title(fig, [ax], ttl)
    panel_note(fig, [ax], nt + "\n纵轴是对数刻度：每往上一格，大 10 倍。")
savefig(fig, "ch08_obs_scales")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("4. 在线统计：每来一批，只把均值和方差往这批挪一点，不存旧数据")
old, new = np.array([0.0, 2.0]), np.array([4.0, 6.0])
n, m = len(old), len(new)
mu_o, mu_n, var_o, var_n = old.mean(), new.mean(), old.var(), new.var()
rate = m / (n + m)
mu_merged = mu_o + rate * (mu_n - mu_o)
both = np.concatenate([old, new])
dev = both - mu_merged
d = mu_n - mu_o
within, between = (n * var_o + m * var_n) / (n + m), n * m / (n + m) ** 2 * d ** 2
var_rsl = var_o + rate * (var_n - var_o + d * (mu_n - mu_merged))
print(f"旧的一批 [0, 2]：均值 {mu_o:g}、方差 {var_o:g}；新的一批 [4, 6]：均值 {mu_n:g}、方差 {var_n:g}")
print(f"新均值 = {mu_o:g} + ({m}/{n + m}) × ({mu_n:g} − {mu_o:g}) = {mu_merged:g}；四个数离 {mu_merged:g}：{minus(str(dev.tolist()))}，"
      f"平方的平均 = {np.mean(dev ** 2):g}")
print(f"批内 (2 × 1 + 2 × 1) ÷ 4 = {within:g}；批间 (2 × 2 ÷ 4²) × 4² = {between:g}；合起来 {within + between:g}。"
      f"rsl_rl 的写法：1 + 0.5 × (1 − 1 + 4 × (5 − 3)) = {var_rsl:g}")
check("新均值 1 + (2/4) × (5 − 1) = 3，和四个数直接平均一样", mu_merged == 3 and both.mean() == 3)
check("方差不是 (1 + 1) ÷ 2 = 1：离 3 的偏差 −3、−1、1、3，平方平均 (9 + 1 + 1 + 9) ÷ 4 = 5",
      (var_o + var_n) / 2 == 1 and np.array_equal(dev, [-3, -1, 1, 3]) and np.mean(dev ** 2) == 5)
check("5 = 批内 1 + 批间 4（worked_examples 的合并公式，等于全量方差）", within == 1 and between == 4
      and within + between == both.var() == 5)
check("rsl_rl 的递推写法代进去也是 5", var_rsl == 5)
new_alt = np.array([6.0, 8.0])                    # 三个 4 只是凑巧：新批换成 [6, 8]
d_alt = new_alt.mean() - mu_o
within_alt = (n * var_o + m * new_alt.var()) / (n + m)
between_alt = n * m / (n + m) ** 2 * d_alt ** 2
print(f"反例：新批换成 [6, 8]，N 还是 {n + m}，d = {d_alt:g}，批间 (2 × 2 ÷ 4²) × 6² = {between_alt:g}，"
      f"合并方差 {within_alt:g} + {between_alt:g} = {np.var(np.concatenate([old, new_alt])):g}")
check("三个 4 只是凑巧：新批换成 [6, 8]，N 仍是 4，d = 6，批间 (2 × 2 ÷ 4²) × 6² = 9，合并方差 1 + 9 = 10",
      d_alt == 6 and between_alt == 9 and within_alt == 1 and np.var(np.concatenate([old, new_alt])) == 10)
nm = EmpiricalNormalization(1)
nm.update(torch.tensor([[0.0], [2.0]]))
after_first = (nm.mean.item(), nm._var.item())
nm.update(torch.tensor([[4.0], [6.0]]))
print(f"EmpiricalNormalization：第一批后 μ = {after_first[0]:g}、方差 {after_first[1]:g}（初值 0 和 1 被整个换掉）；"
      f"第二批后 μ = {nm.mean.item():g}、方差 {nm._var.item():g}、σ = {nm.std.item():.3f}")
check("项目的类：第一批 rate = 1，初值被换掉（μ = 1、方差 1）；第二批后 μ = 3、方差 5、σ = √5 = 2.236",
      after_first == (1.0, 1.0) and nm.mean.item() == 3 and math.isclose(nm._var.item(), 5, rel_tol=1e-6)
      and round(nm.std.item(), 3) == 2.236)
nm.update(torch.tensor([[3.0], [3.0]]))           # 自测：第三批 [3, 3]
check("自测：第三批 [3, 3] 进来，均值还是 3，方差降到 20 ÷ 6 = 3.333", nm.mean.item() == 3
      and round(nm._var.item(), 3) == 3.333 and round(np.var([0, 2, 4, 6, 3, 3]), 3) == 3.333)

D = 4                                             # 更大的核对：5 批 × 64 行 × 4 列随机数
norm = EmpiricalNormalization(D)
count, mean, var, all_x = 0, torch.zeros(1, D), torch.ones(1, D), []
for _ in range(5):
    x = torch.randn(64, D) * torch.tensor([1.0, 8.0, 0.5, 3.0]) + torch.tensor([0.0, 2.0, -1.0, 5.0])
    all_x.append(x)
    norm.update(x)                                # rsl_rl
    count += x.shape[0]                           # 手写：和 normalization.py 的 update() 同一套公式
    r = x.shape[0] / count
    mean_x, var_x = x.mean(dim=0, keepdim=True), x.var(dim=0, unbiased=False, keepdim=True)
    delta = mean_x - mean
    mean = mean + r * delta
    var = var + r * (var_x - var + delta * (mean_x - mean))
X_all = torch.cat(all_x)
table(["列", "rsl_rl 均值", "手写均值", "全部攒齐再算", "rsl_rl 标准差", "全部攒齐再算"],
      [[j, norm.mean[j].item(), mean[0, j].item(), X_all[:, j].mean().item(), norm.std[j].item(),
        X_all[:, j].std(unbiased=False).item()] for j in range(D)])
check("手写的在线均值 = rsl_rl 的", torch.allclose(norm.mean, mean.squeeze(0), atol=1e-5))
check("在线算的均值、标准差 = 320 行攒齐了一次算", torch.allclose(norm.mean, X_all.mean(dim=0), atol=1e-4)
      and torch.allclose(norm.std, X_all.std(dim=0, unbiased=False), atol=1e-4))
print("项目里：每走一个环境步来一批（4096 只机器人各一行）；第 k 步 rate = 4096 ÷ (4096 × k) = 1/k。"
      f"1000 次迭代 × 24 步 = {1000 * 24:,} 批之后，rate 只剩 1/24,000")
check("1000 × 24 = 24,000；4096 × 24 = 98,304", 1000 * 24 == 24000 and 4096 * 24 == 98304)

# ---------------------------------------------------------------------------
banner("4b. 画图：figures/ch08_running_merge.png（合并两批：均值按条数加权，方差还要加上两批中心的距离）")
fig, axes = lesson_figure(3, "合并两批：均值按条数加权；方差 = 批内的散 + 两批中心的距离", panel_height=3.35, width=10.5)
X0 = lambda v: 2.3 + v * 0.85                     # 数轴 0..6 → 图上横坐标
for ax_i, ttl in zip(axes, ["① 均值：不用翻旧数据，也能算出 3", "② 方差：离 3 有多远，平方后平均 = 5",
                            "③ 5 从哪来：两批中心离 3 的 ±2，加上批内的 ±1"]):
    lesson_panel(ax_i, ttl, xmax=10.5, ymax=4.2)
ax = axes[0]
ax.plot([X0(-0.3), X0(6.4)], [1.9, 1.9], color=INK, lw=2)
for t in range(7):
    ax.text(X0(t), 1.65, str(t), fontsize=FS_TICK, color=MUTED, ha="center", va="top")
for v, c in [(0, BLUE), (2, BLUE), (4, GREEN), (6, GREEN)]:
    ax.plot(X0(v), 1.9, "o", ms=13, color=c, zorder=5)
for v, c, lab in [(1, BLUE, "旧批均值 1"), (5, GREEN, "新批均值 5")]:
    ax.plot(X0(v), 2.35, "v", ms=12, color=c)
    ax.text(X0(v), 2.6, lab, color=c, fontsize=FS_SMALL, ha="center", va="bottom")
ax.plot(X0(3), 0.95, "^", ms=14, color=ORANGE)
ax.text(X0(3) + 0.3, 0.95, "合并后的均值 3", color=ORANGE, fontsize=FS_SMALL, va="center")
hand(ax, 8.0, 2.55, "1 + (2/4) × (5 − 1)", color=ORANGE, fontsize=FS_SMALL)
hand(ax, 8.0, 1.85, "= 3", color=ORANGE)
note(ax, 0.0, 0.3, "旧批 [0, 2]、新批 [4, 6]：新均值 = 旧均值 + 这批占的比例 × 两个均值之差。")
ax = axes[1]
for k, (v, c) in enumerate([(0, BLUE), (2, BLUE), (4, GREEN), (6, GREEN)]):
    y = 3.05 - k * 0.6
    arrow(ax, (X0(3), y), (X0(v), y), c, lw=3)
    ax.text(X0(v) + (0.2 if v > 3 else -0.2), y, f"{minus(str(v - 3))}，平方 {(v - 3) ** 2}", color=c,
            fontsize=FS_SMALL, ha="left" if v > 3 else "right", va="center")
ax.plot([X0(3)] * 2, [1.0, 3.3], color=ORANGE, lw=1.5, ls="--")
ax.text(X0(3), 0.9, "3", color=ORANGE, fontsize=FS_TICK, ha="center", va="top")
hand(ax, 8.0, 2.55, "(9 + 1 + 1 + 9) ÷ 4", color=INK, fontsize=FS_SMALL)
hand(ax, 8.0, 1.85, "= 5", color=INK)
note(ax, 0.0, 0.3, "只把两批各自的方差平均：(1 + 1) ÷ 2 = 1——少了一大块。")
ax = axes[2]
rows3 = [["数", "离 3", "中心离 3", "离自己中心"], ["0", "−3", "−2", "−1"], ["2", "−1", "−2", "+1"],
         ["4", "1", "+2", "−1"], ["6", "3", "+2", "+1"]]
lesson_cells(ax, rows3, left=0.3, bottom=0.7, width=1.7, height=0.5, fontsize=FS_SMALL,
             highlights=[(i, 2) for i in range(1, 5)])
hand(ax, 7.25, 2.95, "离 3 = 中心离 3 + 离自己中心", color=INK, fontsize=FS_SMALL)
hand(ax, 7.25, 2.3, "中心离 3：平方都是 4", color=ORANGE, fontsize=FS_SMALL)
hand(ax, 7.25, 1.65, "离自己中心：平方都是 1", color=BLUE, fontsize=FS_SMALL)
hand(ax, 7.25, 1.0, "平方的平均：4 + 1 = 5", color=INK, fontsize=FS_SMALL)
note(ax, 0.0, 0.3, "每个数离 3 多远 = 它那批的中心离 3 多远（±2）+ 它离自己那批的中心多远（±1）。")
savefig(fig, "ch08_running_merge")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("5. 训练时更新、推理时冻结：train() / eval()；no_grad() 管的是另一件事")
FIRST_BATCH_SCALE = 1.0  # TWEAK-3: 100.0
gen = torch.Generator().manual_seed(1)
batches = [torch.randn(64, 1, generator=gen) * sd1 + mu1 for _ in range(40)]   # 关节速度那一列：平均 2、标准差 4
batches[0] = batches[0] * FIRST_BATCH_SCALE
nf = EmpiricalNormalization(1)
nf.train()                                        # 训练模式
track, rates = [], []
for k, bt in enumerate(batches[:30], start=1):
    nf.update(bt)
    track.append(nf.mean.item())
    rates.append(bt.shape[0] / nf.count.item())
mu30 = track[-1]
all30 = torch.cat(batches[:30]).mean().item()
print(f"训练模式 30 批：rate 依次是 {'、'.join('1' if r == 1 else f'1/{round(1 / r)}' for r in rates[:3])} …… 1/{round(1 / rates[-1])}；"
      f"μ 从 {track[0]:.2f} 走到 {mu30:.2f}（30 批攒齐了算是 {all30:.2f}）")
check("每批一样多（64 行）时 rate = 1、1/2、1/3……第 30 批只挪 1/30", np.allclose(rates, [1 / k for k in range(1, 31)]))
check("在线的 μ 就是 30 批攒齐了算的平均", math.isclose(mu30, all30, rel_tol=1e-5))
check(f"第 1 批 rate = 1：μ 直接等于这批的平均 1.55（当前 {track[0]:.2f}）", math.isclose(track[0], batches[0].mean().item(),
      rel_tol=1e-6) and round(track[0], 2) == 1.55)
check(f"30 批后 μ 停在 2 附近（当前 {mu30:.2f}，和真实平均 2 差不到 0.2）", abs(mu30 - 2) < 0.2)
frozen = nf.mean.clone()
nf.eval()                                         # 评估模式：冻结
for bt in batches[30:]:
    nf.update(bt * 100)                           # 喂离谱的数据
    track.append(nf.mean.item())
weird = [bt.mean().item() * 100 for bt in batches[30:]]
print(f"eval() 之后喂的 10 批放大了 100 倍，各批均值在 {min(weird):.0f}–{max(weird):.0f} 之间")
check("eval() 之后再喂 10 批放大 100 倍的数据，μ 一点没动", torch.equal(nf.mean, frozen))
check("这 10 批的均值都在 100 以上（图里画不下）", min(weird) > 100)
probe = EmpiricalNormalization(1)
probe.update(torch.tensor([[-2.0], [6.0]]))
m0 = probe.mean.clone()
probe(torch.tensor([[100.0]]))                    # 只做换算（forward），不调 update
check("换算本身（forward）从不改统计：只有 update() 改", torch.equal(probe.mean, m0))
with torch.no_grad():
    probe.update(torch.tensor([[100.0], [102.0]]))
changed_no_grad = not torch.equal(probe.mean, m0)
m1 = probe.mean.clone()
with torch.inference_mode():                      # 项目采数据时就包在这一层里
    probe.update(torch.tensor([[50.0], [52.0]]))
changed_inference = not torch.equal(probe.mean, m1)
probe.eval()
m2 = probe.mean.clone()
probe.update(torch.tensor([[-50.0], [-52.0]]))
table(["开关", "统计会不会变"], [["train() + no_grad()", "会" if changed_no_grad else "不会"],
                              ["train() + inference_mode()", "会" if changed_inference else "不会"],
                              ["eval()", "会" if not torch.equal(probe.mean, m2) else "不会"]])
check("⚠️ no_grad() / inference_mode() 不冻结统计：训练模式下照样更新；只有 eval() 冻结",
      changed_no_grad and changed_inference and torch.equal(probe.mean, m2))

# ---------------------------------------------------------------------------
banner("5b. 画图：figures/ch08_train_freeze.png（训练时每批挪一点、越挪越少；eval() 之后喂什么都不动）")
fig, ax = plt.subplots(figsize=(11, 6.4))
fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.2)
fig.suptitle("训练时每批挪一点、越挪越少；切到 eval() 之后，喂什么都不动", fontsize=FS_TITLE, fontweight="bold",
             color=INK, y=0.97)
data_axes(ax, "第几批数据", "这一列的 μ（rad/s）")
ks = np.arange(1, 41)
ax.plot(ks[:30], [bt.mean().item() for bt in batches[:30]], "o", ms=6, color=FAINT, label="这一批自己的平均")
ax.plot(ks[:30], track[:30], "-o", ms=5, lw=2.2, color=BLUE, label="在线统计的 μ（train）")
ax.plot(ks[29:], track[29:], "-", lw=3, color=ORANGE, label="冻结的 μ（eval）")
ax.axvline(30.5, color=INK, ls="--", lw=1.4)
lo, hi = ax.get_ylim()
span = hi - lo
ax.set_ylim(lo, hi + 0.35 * span)
ax.set_xlim(0.3, 42.5)
top = ax.get_ylim()[1]
ax.text(31.0, top - 0.04 * (top - lo), f"切到 eval()：\n后 10 批放大 100 倍，\n均值 {min(weird):.0f}–{max(weird):.0f}（图外），\nμ 一动不动",
        fontsize=FS_TICK, color=ORANGE, va="top", bbox=WHITE_BOX)
for k, (dx, dy) in {1: (0.2, -0.2), 2: (1.5, -0.14), 10: (0.6, 0.22), 30: (0.6, 0.22)}.items():
    ax.annotate(f"rate = {'1' if k == 1 else f'1/{k}'}", xy=(k, track[k - 1]), xytext=(k + dx, track[k - 1] + dy * span),
                fontsize=FS_TICK, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.2), bbox=WHITE_BOX)
ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])
ax.legend(loc="lower left", fontsize=FS_TICK, frameon=False, ncol=3, bbox_to_anchor=(0.0, -0.36))
panel_title(fig, [ax], "① 训练模式：第 k 批只挪 1/k    ② 评估模式：冻结")
savefig(fig, "ch08_train_freeze")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("6. 烘焙：归一化器和网络装进同一个模块；漏做一次、多做一次，网络看到的数都不对")
net = torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.ELU(), torch.nn.Linear(8, 1))
norm1.eval()                                      # 导出前冻结（第 5 节）
baked = torch.nn.Sequential(norm1, net)           # 归一化器 + 网络 = 一个新模块（烘焙）
x6 = torch.tensor([[6.0]])
with torch.no_grad():
    seen = {"恰好一次（烘焙好的整体）": norm1(x6).item(), "漏了（只导出网络）": x6.item(),
            "多了（先手工换算，再喂烘焙好的整体）": norm1(norm1(x6)).item()}
    same = torch.allclose(baked(x6), net(norm1(x6)))
table(["部署时的接法", "网络看到的数"], [[k, f"{v:g}" if v == round(v) else minus(f"{v:.4f}")] for k, v in seen.items()])
check("baked(x) 和 net(norm(x)) 完全一样", same)
twice = seen["多了（先手工换算，再喂烘焙好的整体）"]
check("漏了：网络看到 6，是训练时 0.997506 的约 6 倍", round(6 / seen["恰好一次（烘焙好的整体）"], 1) == 6.0)
check("多了：(0.997506 − 2) ÷ 4.01 ≈ −0.2500，正号变成负号", round(twice, 4) == -0.25 and twice < 0)
check("字面值重算：(0.997506 − 2) ÷ 4.01 四舍五入到 4 位是 −0.2500", round((0.997506 - 2) / 4.01, 4) == -0.25)
twice_exact = ((6 - 2) / 4.01 - 2) / 4.01
check("用没舍入的 4/4.01 算是 −0.249998（worked_examples 的 −0.249998445），和 torch 算的一致",
      round(twice_exact, 6) == -0.249998 and math.isclose(twice, twice_exact, rel_tol=1e-5))

# ---------------------------------------------------------------------------
banner("6b. 画图：figures/ch08_bake_paths.png（归一化要恰好做一次）")
fig, axes = lesson_figure(3, "归一化要恰好做一次：漏了、多了，网络看到的都不是训练时的数", panel_height=2.75, width=15.6)
rows_bp = [("① 烘焙好的 ONNX，喂原始读数：恰好一次",
            [("读数 6", ""), ("归一化器\nμ = 2，σ = 4", "onnx"), (minus(f"{seen['恰好一次（烘焙好的整体）']:.4f}"), "onnx-num"),
             ("网络", "onnx")], "✓", "和训练时看到的一样", GREEN),
           ("② 漏了：只导出网络", [("读数 6", ""), ("6", "num"), ("网络", "")], "✗", "比训练时大约 6 倍", ORANGE),
           ("③ 多了：运行时先手工换算，再喂烘焙好的 ONNX",
            [("读数 6", ""), ("手工换算", ""), ("0.9975", ""), ("归一化器", "onnx"), (minus(f"{twice:.4f}"), "onnx-num"),
             ("网络", "onnx")], "✗", "正号变成负号", ORANGE)]
slot = lambda k: 0.2 + k * 2.95
wbox = 2.25
for ax, (ttl, boxes, mark, verdict, vcol) in zip(axes, rows_bp):
    lesson_panel(ax, ttl, xmax=21.8, ymax=3.2)
    onnx_k = [k for k, (_, kind) in enumerate(boxes) if kind.startswith("onnx")]
    if onnx_k:
        x_lo, x_hi = slot(min(onnx_k)) - 0.15, slot(max(onnx_k)) + wbox + 0.15
        ax.add_patch(plt.Rectangle((x_lo, 0.62), x_hi - x_lo, 1.46, facecolor="none", edgecolor=BLUE, lw=1.8, ls="--"))
        ax.text(x_hi, 0.55, "ONNX 文件", color=BLUE, fontsize=FS_TICK, ha="right", va="top")
    for k, (txt, kind) in enumerate(boxes):
        cell(ax, slot(k), 0.8, txt, width=wbox, height=1.1, fontsize=14 if "\n" in txt else FS_STEP,
             facecolor=CELL_HOT if kind.endswith("num") else CELL)
        if k:
            arrow(ax, (slot(k - 1) + wbox + 0.04, 1.35), (slot(k) - 0.04, 1.35), MUTED, lw=2)
    x_v = slot(len(boxes) - 1) + wbox + 0.35
    hand(ax, x_v, 1.35, mark, color=vcol, fontsize=FS_TITLE)
    hand(ax, x_v + 0.7, 1.35, verdict, color=vcol, fontsize=FS_SMALL)
savefig(fig, "ch08_bake_paths")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("6c. 画图：figures/ch08_overview.png（8.0 节的总览图）")
fig, axes = lesson_figure(4, "", panel_height=2.55, width=11.5)
fig.suptitle("输入归一化：每一列换一把自己的尺子\n尺子边跑边攒、训练完冻结、和网络装在一起", fontsize=FS_TITLE,
             fontweight="bold", color=INK)
fig.subplots_adjust(top=0.9)
steps = [("① 原样：61 列大小差很多", [("关节速度\n±8 rad/s", 0), ("投影重力\n±1", 0), ("身体位置命令\n±0.005 m", 0)],
          "同一个学习率顾不了这么不齐的输入（8.1–8.2 节）"),
         ("② 归一化：每列减自己的 μ，除以自己的 σ + ε", [("读数 6", 0), ("减 μ：\n6 − 2 = 4", 0), ("除以 σ + ε：\n4 ÷ 4.01", 0),
                                                 (f"{z6:.4f}", 1)],
          "换算后，大多数列都在 1 附近（8.3 节）"),
         ("③ μ、σ 从哪来：训练时每批挪一点，训练完冻结", [("第 1 批\n挪 1", 0), ("第 2 批\n挪 1/2", 0), ("第 k 批\n挪 1/k", 0),
                                                 ("eval()\n不再动", 1)],
          "不存旧数据，只存均值、方差和条数（8.4 节）；部署时冻结（8.5 节）"),
         ("④ 烘焙：归一化器 + 网络 = 一个 ONNX 文件", [("原始观测", 0), ("归一化器", 1), ("网络", 1), ("动作", 0)],
          "真机只管喂原始读数，归一化恰好做一次（8.6 节）")]
for ax, (ttl, boxes, nt) in zip(axes, steps):
    lesson_panel(ax, ttl, xmax=12.5, ymax=3.1)
    xs = np.linspace(0.3, 9.6, len(boxes))
    for k, (x, (txt, hot)) in enumerate(zip(xs, boxes)):
        cell(ax, x, 1.05, txt, width=2.3, height=1.05, fontsize=FS_SMALL if "\n" in txt else FS_STEP,
             facecolor=CELL_HOT if hot else "#eaf1f8")
        if k and not ttl.startswith("①"):
            arrow(ax, (xs[k - 1] + 2.32, 1.58), (x - 0.02, 1.58), MUTED, lw=2)
    if ttl.startswith("④"):
        ax.add_patch(plt.Rectangle((xs[1] - 0.15, 0.9), xs[2] + 2.3 - xs[1] + 0.3, 1.35, facecolor="none",
                                   edgecolor=BLUE, lw=1.8, ls="--"))
    note(ax, 0.3, 0.45, nt)
savefig(fig, "ch08_overview")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("7. 映射到项目：正文引用的源码行和常数还在不在")
cfg_text = cfg_path.read_text(encoding="utf-8") if cfg_path.is_file() else ""
at = cfg_text.find("MicroduckRlCfg = RslRlOnPolicyRunnerCfg(")
if at >= 0:
    check("MicroduckRlCfg：actor 和 critic 各打开一个归一化器（obs_normalization=True 两次）；每次迭代走 24 步",
          lines_in_order(cfg_text[at:], ["actor=RslRlModelCfg(", "obs_normalization=True,", "critic=RslRlModelCfg(",
                                         "obs_normalization=True,", "num_steps_per_env=24,"]))
else:
    print("  （没找到项目的 env cfg，跳过这一项）")
for rel, text in [("export.py", "Export a trained checkpoint to ONNX, with the observation normalizer baked in."),
                  ("export.py", "This is the ONE path from a checkpoint to a deployable `.onnx`"),
                  ("AGENTS.md", "the normalizer must be baked into the ONNX.")]:
    p = REPO / ("src/mjlab_microduck/export.py" if rel == "export.py" else rel)
    if p.is_file():
        check(f"{rel} 里写着：{text}", text in p.read_text(encoding="utf-8"))
exp_py, exp_sh = REPO / "src" / "mjlab_microduck" / "export.py", REPO / "scripts" / "export.py"
if exp_py.is_file() and exp_sh.is_file():
    check("export.py 的 run_export() 先 get_inference_policy()（先冻结）再 export_policy_to_onnx；scripts/export.py 只是它的命令行外壳",
          lines_in_order(exp_py.read_text(encoding="utf-8"), ["def run_export(", "policy = runner.get_inference_policy(device=device)",
                                                              "runner.export_policy_to_onnx(path, filename)"])
          and "from mjlab_microduck.export import main" in exp_sh.read_text(encoding="utf-8"))
agents = REPO / "AGENTS.md"
if agents.is_file():
    check("AGENTS.md：仿真里回放看不出这个错（in-sim play hides the bug）",
          "in-sim play hides the bug (it applies the" in agents.read_text(encoding="utf-8"))

NORM_LINES = ["def __init__(self, shape: int | tuple[int, ...] | list[int], eps: float = 1e-2, until: int | None = None) -> None:",
              "def forward(self, x: torch.Tensor) -> torch.Tensor:", "return (x - self._mean) / (self._std + self.eps)",
              "def update(self, x: torch.Tensor) -> None:", "if not self.training:", "return",
              "count_x = x.shape[0]", "self.count += count_x", "rate = count_x / self.count",
              "var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)", "mean_x = torch.mean(x, dim=0, keepdim=True)",
              "delta_mean = mean_x - self._mean", "self._mean += rate * delta_mean",
              "self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))",
              "self._std = torch.sqrt(self._var)"]
MLP_LINES = ["if obs_normalization:", "self.obs_normalizer = EmpiricalNormalization(self.obs_dim)",
             "latent = torch.cat(obs_list, dim=-1)", "latent = self.obs_normalizer(latent)",
             "self.obs_normalizer.update(mlp_obs)", "class _OnnxMLPModel(nn.Module):", "x = self.obs_normalizer(x)",
             "out = self.mlp(x)"]
RUNNER_LINES = ["self.alg.train_mode()", "with torch.inference_mode():", 'for _ in range(self.cfg["num_steps_per_env"]):',
                "self.alg.process_env_step(obs, rewards, dones, extras)", "self.alg.eval_mode()",
                "onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)", "onnx_model.eval()"]
PPO_LINES = ["def process_env_step(", "self.actor.update_normalization(obs)", "self.critic.update_normalization(obs)"]
import rsl_rl  # noqa: E402  （开头已经用过它的子模块，这里只为拿到包所在的目录）
rsl_dir = Path(rsl_rl.__file__).parent
for rel, wanted, what in [("modules/normalization.py", NORM_LINES, "EmpiricalNormalization：ε 默认 0.01、换算、冻结、在线更新"),
                          ("models/mlp_model.py", MLP_LINES, "MLPModel：建归一化器、先归一化再进网络、导出版里排在最前"),
                          ("runners/on_policy_runner.py", RUNNER_LINES, "OnPolicyRunner：训练模式 + 不记求导账地采数据、每步更新；回放和导出前切 eval()"),
                          ("algorithms/ppo.py", PPO_LINES, "PPO.process_env_step()：actor、critic 的归一化器每步各更新一次")]:
    p = rsl_dir / rel
    if p.is_file():
        check(f"rsl_rl/{rel}：{what}（{len(wanted)} 行原样、顺序一致）", lines_in_order(p.read_text(encoding="utf-8"), wanted))
    else:
        print(f"  （没找到 rsl_rl/{rel}，跳过这一项；用 uv run 运行就会核对）")

done()
