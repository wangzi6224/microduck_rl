"""第 5 章实验：什么是“学习”——两张收据手算一步、四个零件、训练与超参数、torch 自动求导、
泛化与过拟合、验证集，最后核对项目里的训练主循环。

运行：uv run python docs/learn-zh/labs/ch05_linear_regression.py
纯 CPU，numpy + torch + matplotlib。第 7 节只读几份源码的文字，不加载机器人、不训练。
小节编号与正文对应：实验第 K 节 = 正文 5.K 节（K = 1…6；5.7 节没有要算的数），第 7 节对应「映射到项目」。
正文“改一改”要改的三行都带 `# TWEAK-k:` 标记（第 3、4、5 节各一处）。
"""

import importlib.util
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from _common import banner, check, done, lines_in_order, savefig, table
from _draw import (BLUE, FAINT, FS_NOTE, FS_SMALL, FS_STEP, FS_TICK, FS_TITLE, GREEN, INK, MUTED, ORANGE, WHITE_BOX,
                   data_axes, hand, lesson_cells, lesson_figure, lesson_panel, note, panel_note, panel_title, plt)

np.seterr(over="ignore", invalid="ignore")   # “改一改”把学习率调大时数会越滚越大——那正是要看的现象，不用 numpy 再警告
warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")   # 8 个旋钮穿 8 个点，本来就是故意的


def mean(v) -> float:
    return float(np.mean(v))


def minus(s: str) -> str:
    """图里的负号写成 −（和正文一致），不用键盘上的短横。"""
    return s.replace("-", "−")


# ---------------------------------------------------------------------------
banner("1. 一次学习：两张收据、两个旋钮，手算一步")
X2 = np.array([1.0, 2.0])          # 里程（公里）：两张收据，教学构造的数
Y2 = np.array([3.0, 5.0])          # 车费（元）：藏着的规则是“每公里 2 元 + 起步 1 元”，模型不知道
W0, B0, LR = 1.0, 0.0, 0.1         # 旋钮先随便猜 w = 1、b = 0；学习率 0.1


def step(w, b, x, y, alpha):
    """转一轮：预测 → 误差 → 损失 → 梯度（第 3 章 3.5 节那两列的平均）→ 两个旋钮同时拧。"""
    pred = w * x + b
    err = pred - y
    loss = mean(err ** 2)
    dw = mean(2 * err * x)
    db = mean(2 * err)
    return pred, err, loss, dw, db, w - alpha * dw, b - alpha * db


pred0, err0, L0, dw0, db0, w1, b1 = step(W0, B0, X2, Y2, LR)
pred1, err1, L1, dw1, db1, w2, b2 = step(w1, b1, X2, Y2, LR)
table(["里程 x", "车费 y", "预测 wx+b", "误差 e", "误差平方"],
      [[f"{x:g}", f"{y:g}", f"{p:g}", f"{e:g}", f"{e * e:g}"] for x, y, p, e in zip(X2, Y2, pred0, err0)])
print(f"损失 = (4 + 9) / 2 = {L0:g}")
print(f"∂L/∂w = (2×(−2)×1 + 2×(−3)×2) / 2 = {dw0:g}；∂L/∂b = (2×(−2) + 2×(−3)) / 2 = {db0:g}")
print(f"拧一步（学习率 {LR}）：w = 1 − 0.1×(−8) = {w1:g}，b = 0 − 0.1×(−5) = {b1:g}")
print(f"新预测 {pred1[0]:g}、{pred1[1]:g}；新误差 {err1[0]:g}、{err1[1]:g}；新损失 (0.49 + 0.81) / 2 = {L1:g}")
print(f"收据里没有的 1.5 公里：拧之前猜 {W0 * 1.5 + B0:g} 元，拧一步后猜 {w1 * 1.5 + b1:g} 元")
print(f"自测（第二步）：∂L/∂w = {dw1:g}，∂L/∂b = {db1:g}——误差变了，梯度就得重新量")
check("旧损失 (4 + 9)/2 = 6.5（worked_examples 冻结的数）", math.isclose(L0, 6.5) and (4 + 9) / 2 == 6.5)
check("梯度 −8、−5；按字面值重算：(−4 − 12)/2 = −8，(−4 − 6)/2 = −5",
      math.isclose(dw0, -8) and math.isclose(db0, -5) and (2 * -2 * 1 + 2 * -3 * 2) / 2 == -8 and (2 * -2 + 2 * -3) / 2 == -5)
check("拧一步：w = 1 − 0.1×(−8) = 1.8，b = 0 − 0.1×(−5) = 0.5",
      math.isclose(w1, 1.8) and math.isclose(b1, 0.5) and round(1 - 0.1 * -8, 10) == 1.8 and round(0 - 0.1 * -5, 10) == 0.5)
check("新预测 2.3、4.1，新误差 −0.7、−0.9，新损失 (0.49 + 0.81)/2 = 0.65",
      np.allclose(pred1, [2.3, 4.1]) and np.allclose(err1, [-0.7, -0.9]) and math.isclose(L1, 0.65)
      and round((0.49 + 0.81) / 2, 10) == 0.65)
check("1.5 公里：拧之前 1 × 1.5 + 0 = 1.5 元，拧一步后 1.8 × 1.5 + 0.5 = 3.2 元；藏着的规则答 2 × 1.5 + 1 = 4 元",
      math.isclose(W0 * 1.5 + B0, 1.5) and math.isclose(w1 * 1.5 + b1, 3.2) and round(1.8 * 1.5 + 0.5, 10) == 3.2
      and 2 * 1.5 + 1 == 4)
check("自测：第二步梯度 (−1.4 − 3.6)/2 = −2.5，(−1.4 − 1.8)/2 = −1.6（不是 −8、−5）",
      math.isclose(dw1, -2.5) and math.isclose(db1, -1.6)
      and round((2 * -0.7 * 1 + 2 * -0.9 * 2) / 2, 10) == -2.5 and round((2 * -0.7 + 2 * -0.9) / 2, 10) == -1.6)
_, _, L3a, _, _, w3a, b3a = step(0.0, 0.0, np.array([0.0, 1.0, 2.0]), np.array([1.0, 4.0, 7.0]), 0.1)
L3b = step(w3a, b3a, np.array([0.0, 1.0, 2.0]), np.array([1.0, 4.0, 7.0]), 0.1)[2]
print(f"同一个函数跑第 3 章 3.5 节的三个点：损失 {L3a:g} → 拧一步 (w, b) = ({w3a:g}, {b3a:g}) → 损失 {L3b:g}")
check("和第 3 章 3.5 节是同一件事：三个点那一轮也是 22 → (1.2, 0.8) → 6.16",
      math.isclose(L3a, 22) and math.isclose(w3a, 1.2) and math.isclose(b3a, 0.8) and math.isclose(L3b, 6.16))

# ---------------------------------------------------------------------------
banner("1b. 画图：figures/ch05_overview.png（5.0 节总览）和 figures/ch05_one_step.png（拧一步）")
fig, axes = lesson_figure(4, "学习转一轮：预测、打分、拧旋钮，损失 6.5 → 0.65", panel_height=3.1)
fig.subplots_adjust(right=0.84)
ax = axes[0]
lesson_panel(ax, "① 数据：两张收据（整个学习过程中一个数都不改）")
ax.text(5.3, 2.9, "第 1 张", fontsize=FS_SMALL, color=MUTED, ha="center", va="center")
ax.text(6.7, 2.9, "第 2 张", fontsize=FS_SMALL, color=MUTED, ha="center", va="center")
lesson_cells(ax, [[1, 2], [3, 5]], left=4.6, bottom=1.15, width=1.4, height=0.75)
hand(ax, 0.3, 2.275, "里程 x（公里）", color=INK)
hand(ax, 0.3, 1.525, "车费 y（元）", color=INK)
note(ax, 0.3, 0.45, "已知的例子。后面转多少轮，这四个数都原样不动。")
ax = axes[1]
lesson_panel(ax, f"② 模型：预测 = w × 里程 + b，旋钮先猜 w = {W0:g}、b = {B0:g}")
hand(ax, 0.3, 2.2, f"1 × 1 + 0 = {pred0[0]:g} 元")
hand(ax, 4.6, 2.2, f"1 × 2 + 0 = {pred0[1]:g} 元")
note(ax, 0.3, 0.8, "带旋钮的函数机器：旋钮拧到哪，预测就跟到哪。")
ax = axes[2]
lesson_panel(ax, "③ 损失：一个数说清“现在猜得有多差”")
hand(ax, 0.3, 2.45, minus(f"误差 = 预测 − 车费：1 − 3 = {err0[0]:g}，2 − 5 = {err0[1]:g}"))
hand(ax, 0.3, 1.5, f"平方再平均：(4 + 9) ÷ 2 = {L0:g}", color=ORANGE)
note(ax, 0.3, 0.5, "越小越好；下一步往哪边拧，就看它。")
ax = axes[3]
lesson_panel(ax, minus(f"④ 优化：梯度 ({dw0:g}, {db0:g})，往反方向各拧 {LR:g} 倍"))
hand(ax, 0.3, 2.45, minus(f"w = 1 − 0.1 × ({dw0:g}) = {w1:g}"), color=GREEN)
hand(ax, 0.3, 1.5, minus(f"b = 0 − 0.1 × ({db0:g}) = {b1:g}"), color=GREEN)
note(ax, 0.3, 0.5, f"带着新旋钮回到 ②：预测 {pred1[0]:g}、{pred1[1]:g} 元，损失降到 {L1:g}。")
top2, top4 = axes[1].get_position(), axes[3].get_position()
fig.add_artist(FancyArrowPatch((0.87, (top4.y0 + top4.y1) / 2), (0.87, (top2.y0 + top2.y1) / 2), transform=fig.transFigure,
                               connectionstyle="arc3,rad=0.28", arrowstyle="-|>", mutation_scale=26, lw=3.2, color=ORANGE))
fig.text(0.895, (top2.y1 + top4.y0) / 2, "下\n一\n轮", fontsize=FS_STEP, color=ORANGE, ha="center", va="center", linespacing=1.25)
savefig(fig, "ch05_overview")
plt.close(fig)

fig = plt.figure(figsize=(9, 12.6))
fig.suptitle("拧一步：直线向两张收据靠近，损失 6.5 → 0.65", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
ax_a = fig.add_axes([0.13, 0.575, 0.82, 0.30])
ax_b = fig.add_axes([0.13, 0.115, 0.82, 0.30])
xs = np.linspace(0, 3, 50)
for ax, w, b, pred, err, color in ((ax_a, W0, B0, pred0, err0, BLUE), (ax_b, w1, b1, pred1, err1, GREEN)):
    data_axes(ax, "里程 x（公里）", "车费 y（元）")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 7)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_yticks(range(8))
    if ax is ax_b:
        ax.plot(xs, W0 * xs + B0, color=FAINT, lw=2, ls="--")
        ax.text(2.95, 2.2, "拧之前", color=MUTED, fontsize=FS_SMALL, ha="right", va="top")
    ax.plot(xs, w * xs + b, color=color, lw=3)
    for xi, yi, pi, ei in zip(X2, Y2, pred, err):
        ax.plot([xi, xi], [pi, yi], color=ORANGE, lw=3.5, solid_capstyle="butt")
        ax.text(xi - 0.07, (pi + yi) / 2, minus(f"误差 {ei:g}"), color=ORANGE, fontsize=FS_SMALL, ha="right", va="center")
    ax.scatter(X2, Y2, s=110, color=INK, zorder=6)
    for xi, yi in zip(X2, Y2):
        ax.text(xi - 0.07, yi + 0.3, f"收据 ({xi:g}, {yi:g})", color=INK, fontsize=FS_SMALL, ha="right", bbox=WHITE_BOX)
ax_a.text(2.97, 2.1, "w = 1、b = 0", color=BLUE, fontsize=FS_SMALL, ha="right", va="top", bbox=WHITE_BOX)
hand(ax_a, 0.08, 6.4, "损失 = (4 + 9) ÷ 2 = 6.5", color=ORANGE, fontsize=FS_SMALL)
ax_b.text(2.97, 3.9, "w = 1.8、b = 0.5", color=GREEN, fontsize=FS_SMALL, ha="right", va="center", bbox=WHITE_BOX)
hand(ax_b, 0.08, 6.4, "损失 = (0.49 + 0.81) ÷ 2 = 0.65", color=ORANGE, fontsize=FS_SMALL)
ax_b.scatter([1.5], [w1 * 1.5 + b1], s=120, facecolor="white", edgecolor=GREEN, lw=2.5, zorder=7)
ax_b.text(1.58, w1 * 1.5 + b1 - 0.25, f"1.5 公里 → {w1 * 1.5 + b1:g} 元", color=GREEN, fontsize=FS_SMALL, va="top",
          bbox=WHITE_BOX)
panel_title(fig, [ax_a], "① 出发：w = 1、b = 0，两张收据都猜低了")
panel_note(fig, [ax_a], "橙色竖线是误差（预测 − 车费）：−2、−3。竖线越长，损失越大。")
panel_title(fig, [ax_b], "② 拧一步后：w = 1.8、b = 0.5，直线贴近了收据")
panel_note(fig, [ax_b], "误差缩成 −0.7、−0.9。空心点是收据里没有的 1.5 公里：\n"
                         "1.8 × 1.5 + 0.5 = 3.2 元——改的是旋钮，所以没见过的里程也有答案。")
savefig(fig, "ch05_one_step")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("2. 四个零件：数据、模型、损失、优化，各写成一个函数，拼回 5.1 节那一步")
data = {"x": X2, "y": Y2}                     # 零件一：数据（已知的例子，学习过程中不改）


def model(w, b, x):                           # 零件二：模型（形状是直线，旋钮是 w、b）
    return w * x + b


def loss_fn(pred, y):                         # 零件三：损失（均方误差：误差、平方、取平均）
    return mean((pred - y) ** 2)


def optimize(w, b, x, y, alpha):              # 零件四：优化（梯度下降：两个旋钮同时往下坡拧一步）
    err = model(w, b, x) - y
    return w - alpha * mean(2 * err * x), b - alpha * mean(2 * err)


L_before = loss_fn(model(W0, B0, data["x"]), data["y"])
w_new, b_new = optimize(W0, B0, data["x"], data["y"], LR)
L_after = loss_fn(model(w_new, b_new, data["x"]), data["y"])
print(f"数据 → 模型（w = {W0:g}, b = {B0:g}）→ 损失 {L_before:g} → 优化 → (w, b) = ({w_new:g}, {b_new:g}) → 损失 {L_after:g}")
check("四个零件拼起来 = 5.1 节那一步：6.5 → (1.8, 0.5) → 0.65",
      math.isclose(L_before, L0) and math.isclose(w_new, w1) and math.isclose(b_new, b1) and math.isclose(L_after, L1))

X4, Y4 = np.tile(X2, 2), np.tile(Y2, 2)       # 自测：两张收据各复制一份，变成 4 张


def sum_loss_and_grad(w, b, x, y):
    """把“取平均”换成“加起来”：损失和它对 w 的偏导数。"""
    err = model(w, b, x) - y
    return float(np.sum(err ** 2)), float(np.sum(2 * err * x))


s2, g2 = sum_loss_and_grad(W0, B0, X2, Y2)
s4, g4 = sum_loss_and_grad(W0, B0, X4, Y4)
m2, m4 = loss_fn(model(W0, B0, X2), Y2), loss_fn(model(W0, B0, X4), Y4)
gm2 = mean(2 * (model(W0, B0, X2) - Y2) * X2)
gm4 = mean(2 * (model(W0, B0, X4) - Y4) * X4)
table(["损失的算法", "2 张：损失", "2 张：∂L/∂w", "4 张：损失", "4 张：∂L/∂w"],
      [["加起来", f"{s2:g}", f"{g2:g}", f"{s4:g}", f"{g4:g}"], ["取平均", f"{m2:g}", f"{gm2:g}", f"{m4:g}", f"{gm4:g}"]])
check("自测：加起来 13 → 26、梯度 −16 → −32（都翻倍）；取平均 6.5 和 −8 不变",
      (s2, g2, s4, g4) == (13.0, -16.0, 26.0, -32.0) and (m2, gm2, m4, gm4) == (6.5, -8.0, 6.5, -8.0))
check("机器人每轮的数据：4096 只 × 每只 24 步 = 98,304 条记录", 4096 * 24 == 98_304)
check("actor 的旋钮：61→512→256→128→14 四层的权重和偏置 197,774 个，再加 14 个 σ = 197,788（第 6 章数给你看）",
      61 * 512 + 512 + 512 * 256 + 256 + 256 * 128 + 128 + 128 * 14 + 14 == 197_774 and 197_774 + 14 == 197_788)

# ---------------------------------------------------------------------------
banner("3. 训练：把这一圈转很多轮（学习率、轮数是人定的超参数）")
# 超参数：人定的数，训练过程中不对它们求梯度。“改一改”第 1 条改这一行。
ALPHA = 0.1  # TWEAK-1: 0.3
ROUNDS = 300
SNAP = (0, 1, 2, 3, 5, 10, 50, 100, 300)
w, b = W0, B0
rows, trail = [], []
for it in range(ROUNDS + 1):
    L = loss_fn(model(w, b, X2), Y2)
    trail.append((w, b, L))
    if it in SNAP:
        rows.append([it, f"{w:.4f}", f"{b:.4f}", f"{L:.6f}"])
    if it < ROUNDS:
        w, b = optimize(w, b, X2, Y2, ALPHA)
table(["迭代", "w", "b", "损失 L"], rows)
print(f"超参数（人定、不求梯度）：学习率 {ALPHA:g}、轮数 {ROUNDS}、起点 (w, b) = ({W0:g}, {B0:g})、模型的形状（直线）")
losses = np.array([t[2] for t in trail])
if not np.all(np.isfinite(losses)) or losses[-1] > losses[0]:
    lam_max = float(np.linalg.eigvalsh(2 * np.array([[np.mean(X2 ** 2), np.mean(X2)], [np.mean(X2), 1.0]])).max())
    print(f"  损失越转越大：学习率过了门槛（第 3 章 3.6 节）。这只碗的门槛约 {2 / lam_max:.4f}，由最陡的方向说了算。")
w_s = {it: tw for it, (tw, _, _) in enumerate(trail)}
check("第 2 轮：w = 1.8 − 0.1×(−2.5) = 2.05，b = 0.5 − 0.1×(−1.6) = 0.66，损失 (0.0841 + 0.0576)/2 = 0.07085",
      len(trail) > 2 and math.isclose(trail[2][0], 2.05) and math.isclose(trail[2][1], 0.66) and math.isclose(trail[2][2], 0.07085)
      and round(1.8 - 0.1 * -2.5, 10) == 2.05 and round(0.5 - 0.1 * -1.6, 10) == 0.66 and round((0.0841 + 0.0576) / 2, 10) == 0.07085)
p2 = model(trail[2][0], trail[2][1], X2) if len(trail) > 2 else np.array([np.nan, np.nan])
check("第 2 轮的新预测 2.71、4.76，新误差 −0.29、−0.24，平方 0.0841、0.0576",
      np.allclose(p2, [2.71, 4.76]) and np.allclose(p2 - Y2, [-0.29, -0.24]) and np.allclose((p2 - Y2) ** 2, [0.0841, 0.0576]))
check("损失每一轮都在变小（这个学习率没过门槛）；前三轮 6.5 → 0.013", bool(np.all(np.diff(losses) < 0)) and f"{trail[3][2]:.3f}" == "0.013")
check("w 先冲过 2：第 5 轮 w = 2.1551、b = 0.7410；最高点在第 6 轮（2.1553），之后慢慢退回",
      round(trail[5][0], 4) == 2.1551 and round(trail[5][1], 4) == 0.7410 and int(np.argmax([t[0] for t in trail])) == 6
      and round(trail[6][0], 4) == 2.1553 and trail[-1][0] < trail[6][0])
check("第 100 轮损失 0.000407；第 300 轮 w = 2.0021、b = 0.9966、损失 0.000001",
      f"{trail[100][2]:.6f}" == "0.000407" and f"{trail[300][0]:.4f}" == "2.0021" and f"{trail[300][1]:.4f}" == "0.9966"
      and f"{trail[300][2]:.6f}" == "0.000001")
e5 = np.array([2.1551 + 0.7410 - 3, 2 * 2.1551 + 0.7410 - 5])          # 进阶折叠：按表里印出的第 5 轮 w、b 重算
g5 = [mean(2 * (model(trail[k][0], trail[k][1], X2) - Y2) * X2) for k in (5, 10)]
print(f"第 5 轮（表里的数）：误差 {e5[0]:.4f}、{e5[1]:.4f}；∂L/∂w = {mean(2 * e5 * X2):.4f}，∂L/∂b = {mean(2 * e5):.4f}；"
      f"第 10 轮 ∂L/∂w = {g5[1]:.4f}（转正了）")
check("进阶：第 5 轮 2.8961、5.0512，误差 −0.1039、+0.0512；(−0.2078 + 0.2048)/2 = −0.0015，(−0.2078 + 0.1024)/2 = −0.0527",
      np.allclose(e5, [-0.1039, 0.0512]) and round(mean(2 * e5 * X2), 4) == -0.0015 and round(mean(2 * e5), 4) == -0.0527
      and round((-0.2078 + 0.2048) / 2, 4) == -0.0015 and round((-0.2078 + 0.1024) / 2, 4) == -0.0527)
check("进阶：按没舍入的旋钮，第 5 轮 ∂L/∂w 也几乎是 0（约 −0.002）；第 10 轮 ∂L/∂w 已经是正的，w 往回退",
      round(g5[0], 3) == -0.002 and g5[1] > 0)
check("理论值：藏着的规则 w = 2、b = 1 让两张收据的损失恰好是 0；300 轮后离它不到 0.01",
      loss_fn(model(2.0, 1.0, X2), Y2) == 0.0 and abs(trail[-1][0] - 2) < 0.01 and abs(trail[-1][1] - 1) < 0.01)
p_in, p_up = model(2.2, 0.7, X2), model(2.2, 1.2, X2)   # 沟里一组（一高一低）、坡上一组（两个都多 0.2）
L_in, L_up = loss_fn(p_in, Y2), loss_fn(p_up, Y2)
g_in = (mean(2 * (p_in - Y2) * X2), mean(2 * (p_in - Y2)))
print(f"互相补偿：(2.2, 0.7) 预测 {p_in[0]:g}、{p_in[1]:g}，损失 {L_in:.2f}；(2.2, 1.2) 预测 {p_up[0]:g}、{p_up[1]:g}，"
      f"损失 {L_up:.2f}；沟里的梯度 ({g_in[0]:.1f}, {abs(g_in[1]):.0f})")
check("沟里 (2.2, 0.7)：预测 2.9、5.1，误差 −0.1、0.1，损失 (0.01 + 0.01)/2 = 0.01",
      np.allclose(p_in, [2.9, 5.1]) and np.allclose(p_in - Y2, [-0.1, 0.1]) and math.isclose(L_in, 0.01)
      and round((0.01 + 0.01) / 2, 10) == 0.01)
check("坡上 (2.2, 1.2)：预测 3.4、5.6，误差 0.4、0.6，损失 (0.16 + 0.36)/2 = 0.26，是沟里那组的 26 倍",
      np.allclose(p_up, [3.4, 5.6]) and np.allclose(p_up - Y2, [0.4, 0.6]) and math.isclose(L_up, 0.26)
      and round((0.16 + 0.36) / 2, 10) == 0.26 and round(0.26 / 0.01, 6) == 26 and round(L_up / L_in, 6) == 26)
check("沟里的坡很缓：(2.2, 0.7) 处梯度 (0.1, 0)；按字面值 (−0.2 + 0.4)/2 = 0.1，学习率 0.1 一轮只拧 0.1 × 0.1 = 0.01",
      math.isclose(g_in[0], 0.1) and abs(g_in[1]) < 1e-12 and round((2 * -0.1 * 1 + 2 * 0.1 * 2) / 2, 10) == 0.1
      and round(0.1 * 0.1, 10) == 0.01)
H2 = 2 * np.array([[np.mean(X2 ** 2), np.mean(X2)], [np.mean(X2), 1.0]])   # 这只碗弯得多厉害（上面的门槛也由它算）
lam, vec = np.linalg.eigh(H2)
check("理论值：碗最缓的方向是“一个旋钮高、一个低”，最陡的是“两个一起高”，陡缓相差 40 多倍",
      vec[0, 0] * vec[1, 0] < 0 and vec[0, 1] * vec[1, 1] > 0 and 40 < lam[1] / lam[0] < 50)

# ---------------------------------------------------------------------------
banner("3b. 画图：figures/ch05_valley.png（旋钮先冲下陡坡，再沿着斜沟慢慢挪回 (2, 1)）")
VALLEY = 0.02                                          # 沟底：损失不到 0.02 的那一条窄带
path = np.array([(t[0], t[1]) for t in trail])
if ALPHA != 0.1 or not np.all(np.isfinite(path)):
    print("  学习率改过了：这张图是照着学习率 0.1 的那条路画的，跳过。看上面那张表就行。")
else:
    wg, bg = np.meshgrid(np.linspace(0.85, 2.55, 341), np.linspace(-0.12, 1.45, 315))
    lg = np.mean((wg[..., None] * X2 + bg[..., None] - Y2) ** 2, axis=-1)   # 每一组 (w, b) 在两张收据上的损失
    fig = plt.figure(figsize=(9, 10.8))
    fig.suptitle("w 冲过 2 再退回：碗底有一道斜着的缓沟", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
    ax = fig.add_axes([0.12, 0.22, 0.83, 0.65])
    data_axes(ax, "单价 w（元 / 公里）", "起步价 b（元）")
    ax.set_xlim(0.85, 2.55)
    ax.set_ylim(-0.12, 1.45)
    ax.set_aspect("equal")
    ax.contourf(wg, bg, lg, levels=[0, VALLEY], colors=["#f6d3b3"], zorder=1)
    cs = ax.contour(wg, bg, lg, levels=[1, 2, 3, 4, 5, 6], colors=FAINT, linewidths=1.5, zorder=2)
    ax.clabel(cs, fmt="%d", fontsize=13, inline_spacing=6,
              manual=[(1.64, 0.554), (1.45, 0.436), (1.303, 0.346), (1.18, 0.27), (1.072, 0.203), (0.974, 0.142)])
    ax.axvline(2.0, color=MUTED, ls="--", lw=1.6, zorder=2)
    ax.text(2.03, -0.1, "w = 2", color=MUTED, fontsize=FS_SMALL, va="bottom")
    ax.plot(path[:, 0], path[:, 1], color=BLUE, lw=2.4, zorder=4)
    marks = [0, 1, 2, 3, 5, 10, 50, 100, 300]           # 和 5.3 节那张表同样的几轮
    ax.scatter(path[marks, 0], path[marks, 1], s=45, color=BLUE, zorder=5)
    ax.scatter([path[5, 0]], [path[5, 1]], s=230, facecolor="none", edgecolor=ORANGE, lw=2.6, zorder=6)
    ax.scatter([2.0], [1.0], s=420, marker="*", color=ORANGE, edgecolor=INK, lw=1, zorder=7)
    ax.text(1.89, 1.0, "(2, 1)", color=INK, fontsize=FS_SMALL, ha="right", va="center")   # 不加白底：别在沟上挖出缺口
    ax.text(1.05, -0.01, "出发 (1, 0)", color=BLUE, fontsize=FS_SMALL, va="top", bbox=WHITE_BOX)
    ax.text(1.84, 0.47, "第 1 轮", color=BLUE, fontsize=FS_SMALL, va="top", bbox=WHITE_BOX)
    for (wv, bv), lv in (((2.2, 0.7), L_in), ((2.2, 1.2), L_up)):
        ax.scatter([wv], [bv], s=110, marker="D", color=GREEN, edgecolor="white", lw=1.0, zorder=8)
        ax.text(wv + 0.06, bv, f"({wv:g}, {bv:g})\n损失 {lv:.2f}", color=GREEN, fontsize=FS_SMALL, va="center",
                linespacing=1.25)
    ax.text(1.73, 1.37, f"沟底：损失不到 {VALLEY:g}", color=ORANGE, fontsize=FS_SMALL, ha="right", va="center", bbox=WHITE_BOX)
    panel_title(fig, [ax], "学习率 0.1，300 轮里旋钮走过的路（每个点是一组 w、b）")
    panel_note(fig, [ax], "细线是等高线，线上的数是那一圈的损失，相邻两圈差 1：离沟越远越密，坡越陡。\n"
                          f"橙色斜带是沟底（损失不到 {VALLEY:g}）：沿着它走，损失几乎不变。\n"
                          "蓝线头几步直冲下坡，进了沟就只能一点点挪；橙圈是表里的第 5 轮，在 w = 2 右边。")
    savefig(fig, "ch05_valley")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("4. 自动求导：只写损失怎么算，梯度交给 torch")
# “改一改”第 2 条改这一行：不告诉 torch 哪些是旋钮，看它报什么错。
REQUIRES_GRAD = True  # TWEAK-2: False
w = torch.tensor(1.0, requires_grad=REQUIRES_GRAD)   # 旋钮 w，初值 1；requires_grad=True：请 torch 记下它参与的每一步
b = torch.tensor(0.0, requires_grad=REQUIRES_GRAD)   # 旋钮 b，初值 0
x = torch.tensor([1.0, 2.0])                         # 数据：两张收据的里程
y = torch.tensor([3.0, 5.0])                         # 数据：两张收据的车费

L = torch.mean((w * x + b - y) ** 2)                 # 只写“损失怎么算”：预测、作差、平方、平均
autograd_ok = True
try:
    L.backward()                                     # 倒着把账算一遍：求出 ∂L/∂w、∂L/∂b，存进 w.grad、b.grad
except RuntimeError as err:
    autograd_ok = False
    print("  torch 报错：", str(err).splitlines()[0])
    print("  （没告诉 torch 哪些是旋钮，它就没有记账，backward() 无从算起。下面用到梯度的几项都会是 ✗。）")

if autograd_ok:
    print(f"torch 算的损失 L = {L.item():g}")
    table(["", "手算（5.1 节）", "torch：w.grad、b.grad"],
          [["∂L/∂w", f"{dw0:g}", f"{w.grad.item():g}"], ["∂L/∂b", f"{db0:g}", f"{b.grad.item():g}"]])
    check("backward() 算的就是 5.1 节手算的偏导数：−8、−5", w.grad.item() == dw0 == -8 and b.grad.item() == db0 == -5)
    with torch.no_grad():                            # 拧旋钮这两行不需要记账
        w -= 0.1 * w.grad                            # w = 1 − 0.1 × (−8) = 1.8
        b -= 0.1 * b.grad                            # b = 0 − 0.1 × (−5) = 0.5
    w.grad.zero_()                                   # 把 .grad 清零，准备下一轮
    b.grad.zero_()
    print(f"拧一步之后：w = {w.item():g}，b = {b.item():g}；清零之后 w.grad = {w.grad.item():g}")
    check("torch 拧一步也得 w = 1.8、b = 0.5；清零后 .grad 回到 0",
          math.isclose(w.item(), 1.8, rel_tol=1e-6) and math.isclose(b.item(), 0.5) and w.grad.item() == 0 == b.grad.item())
    L_again = torch.mean((w * x + b - y) ** 2)       # 旋钮改了，损失要自己重新算一遍
    print(f"拧完之后 L 里还是 {L.item():g}（torch 不会像 computed 那样自己重算）；重新算一遍才是 {L_again.item():.4g}")
    check("拧完旋钮，L 不会自己更新（还是 6.5）；重新算一遍才是 0.65",
          L.item() == 6.5 and math.isclose(L_again.item(), 0.65, rel_tol=1e-6))
    w_bad = torch.tensor(1.0, requires_grad=True)
    torch.mean((w_bad * x - y) ** 2).backward()
    try:
        w_bad -= 0.1 * w_bad.grad                    # 故意不包 no_grad，直接原地改旋钮
        inplace_msg = ""
    except RuntimeError as exc:
        inplace_msg = str(exc).splitlines()[0]
    print("不包 no_grad 就原地改旋钮：", inplace_msg or "（没有报错）")
    check("不包 no_grad 就原地改旋钮，torch 报错（它不许改正在记账的旋钮）", "in-place" in inplace_msg)

    # ⚠️ 忘了清零：新拿一对旋钮，连着 backward 两次、中间不清零
    w_acc = torch.tensor(1.0, requires_grad=True)
    b_acc = torch.tensor(0.0, requires_grad=True)
    for _ in range(2):
        torch.mean((w_acc * x + b_acc - y) ** 2).backward()
    print(f"连着 backward 两次、中间不清零：w.grad = {w_acc.grad.item():g}，b.grad = {b_acc.grad.item():g}")
    check("忘了清零，梯度会累加：−8 + (−8) = −16，−5 + (−5) = −10", w_acc.grad.item() == -16 and b_acc.grad.item() == -10)
    # 自测：每轮都忘了清零——第一轮照常拧，第二轮的 .grad 里还留着第一轮的 −8
    w_f = torch.tensor(1.0, requires_grad=True)
    b_f = torch.tensor(0.0, requires_grad=True)
    torch.mean((w_f * x + b_f - y) ** 2).backward()
    with torch.no_grad():
        w_f -= 0.1 * w_f.grad
        b_f -= 0.1 * b_f.grad
    torch.mean((w_f * x + b_f - y) ** 2).backward()   # 没清零就算第二轮
    print(f"每轮都忘了清零：第二轮的 w.grad = {w_f.grad.item():g}（本该是 −2.5）")
    check("自测：第二轮 w.grad = −8 + (−2.5) = −10.5，是该用的 −2.5 的 4 倍多",
          math.isclose(w_f.grad.item(), -10.5, rel_tol=1e-6) and -8 + -2.5 == -10.5 and 10.5 / 2.5 == 4.2)
else:
    check("backward() 算出 −8、−5（这次没算成）", False)

print("\n第 3 章 3.5 节的 50 个点：手写两行梯度 vs torch 自动求导，各走 200 轮")
rng = np.random.default_rng(0)                       # 与第 3 章同一份数据：同样的种子、同样的写法
X50 = rng.uniform(-1, 1, size=50)
Y50 = 2.0 * X50 + 1.0 + rng.normal(0, 0.1, size=50)
w_np, b_np = 0.0, 0.0
for _ in range(200):
    err50 = w_np * X50 + b_np - Y50
    w_np, b_np = w_np - 0.1 * np.mean(2 * err50 * X50), b_np - 0.1 * np.mean(2 * err50)   # 第 3 章手写的两行
w50, b50 = float(w_np), float(b_np)
if autograd_ok:
    Xt, Yt = torch.tensor(X50), torch.tensor(Y50)    # 双精度，和 numpy 一样，才能逐位比
    wt = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bt = torch.zeros((), dtype=torch.float64, requires_grad=True)
    for _ in range(200):
        L = torch.mean((wt * Xt + bt - Yt) ** 2)
        L.backward()
        with torch.no_grad():
            wt -= 0.1 * wt.grad
            bt -= 0.1 * bt.grad
        wt.grad.zero_()
        bt.grad.zero_()
    table(["", "w", "b"], [["torch 自动求导", wt.item(), bt.item()], ["手写梯度（第 3 章）", w50, b50]], floatfmt=".5f")
    check("torch 和手写两行走出同一条路：200 轮后相差不到 1e-12", abs(wt.item() - w50) < 1e-12 and abs(bt.item() - b50) < 1e-12)
    w50, b50 = wt.item(), bt.item()
else:
    print(f"  （torch 这一半跳过；手写版得到 w = {w50:.5f}、b = {b50:.5f}）")
    check("torch 和手写两行走出同一条路（这次没算成）", False)
check("和第 3 章 3.5 节九行表最后一行一致：w = 2.02617、b = 1.00192", round(w50, 5) == 2.02617 and round(b50, 5) == 1.00192)

# ---------------------------------------------------------------------------
banner("5. 泛化与过拟合：没见过的数据上准不准")
X_new = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])      # 50 个点里没有的 x（这批数据没有单位，x 就是一个数）
pred_new = [f"{w50 * xv + b50:.3f}" for xv in X_new]
table(["新的 x", "真值 2x+1", "模型预测"], [[f"{xv:g}", f"{2 * xv + 1:g}", p] for xv, p in zip(X_new, pred_new)])
check("新 x 上的误差都小于 0.03", max(abs(w50 * xv + b50 - (2 * xv + 1)) for xv in X_new) < 0.03)
check("按字面值重算：2.02617 × x + 1.00192，保留 3 位，和表里一样；x = 0.5 时 1.013085 + 1.00192 = 2.015005",
      pred_new == [f"{2.02617 * xv + 1.00192:.3f}" for xv in X_new] and round(2.02617 * 0.5, 6) == 1.013085
      and round(1.013085 + 1.00192, 6) == 2.015005)

print("\n迷你版：两张收据、三个旋钮 ŷ = b + w·x + c·x²")


def quad(b3, w3, c3, xv):
    return b3 + w3 * xv + c3 * xv ** 2


rows = []
for b3, w3, c3 in ((1, 2, 0), (17, -22, 8), (-7, 14, -4)):
    rows.append([f"({b3}, {w3}, {c3})", f"{loss_fn(quad(b3, w3, c3, X2), Y2):g}", f"{quad(b3, w3, c3, 1.5):g}"])
table(["(b, w, c)", "两张收据上的训练损失", "1.5 公里的预测（元）"], rows)
check("(1, 2, 0) 和 (17, −22, 8) 在两张收据上的训练损失都是 0", loss_fn(quad(1, 2, 0, X2), Y2) == 0 == loss_fn(quad(17, -22, 8, X2), Y2))
check("1.5 公里：1 + 2×1.5 = 4 元；17 − 22×1.5 + 8×2.25 = 17 − 33 + 18 = 2 元",
      quad(1, 2, 0, 1.5) == 4 and quad(17, -22, 8, 1.5) == 2 and 17 - 33 + 18 == 2)
check("自测：c = −4 → (b, w) = (1 + 2c, 2 − 3c) = (−7, 14)，训练损失 0，1.5 公里 −7 + 21 − 9 = 5 元",
      all(loss_fn(quad(1 + 2 * c, 2 - 3 * c, c, X2), Y2) == 0 for c in (-4, 0, 8))
      and quad(-7, 14, -4, 1.5) == 5 and -7 + 21 - 9 == 5)
check("理论值：c 取任何数，配上 b = 1 + 2c、w = 2 − 3c，两张收据的训练损失都是 0（抽 21 个 c 核对）",
      all(loss_fn(quad(1 + 2 * c, 2 - 3 * c, c, X2), Y2) < 1e-20 for c in np.linspace(-10, 10, 21)))

# “改一改”第 3 条改这一行：训练用的收据多到 40 张，8 个旋钮还管得住吗？
N_TRAIN = 8  # TWEAK-3: 40
print(f"\n正式版：{N_TRAIN} 张带随机误差的收据，直线（2 个旋钮）vs 曲线（8 个旋钮）")
SIGMA = 0.3                                          # 每张收据的随机误差（元）：堵车、等红灯
rng5 = np.random.default_rng(6)
noise_tr = rng5.normal(0, SIGMA, 8)                  # 先抽前 8 张训练收据的随机误差
x_va = rng5.uniform(0.5, 4.0, 40)                    # 40 张新收据：不拿来拧旋钮，只拿来打分
y_va = 2 * x_va + 1 + rng5.normal(0, SIGMA, 40)      # 考卷先定好：改了 N_TRAIN，这 40 张也一张不变
noise_tr = np.concatenate([noise_tr, rng5.normal(0, SIGMA, max(N_TRAIN - 8, 0))])[:N_TRAIN]
x_tr = np.linspace(0.5, 4.0, N_TRAIN)                # 训练用的收据：里程 0.5、1、1.5……4 公里
y_tr = 2 * x_tr + 1 + noise_tr
K_CURVE = 8


def fit(k):
    """k 个旋钮（最高到 x 的 k−1 次方），直接解出让训练损失最小的那组旋钮。"""
    return np.polyfit(x_tr, y_tr, k - 1)


def mse_of(p, xv, yv) -> float:
    return mean((np.polyval(p, xv) - yv) ** 2)


p_line, p_curve = fit(2), fit(K_CURVE)
print("训练收据的车费（元，保留 3 位）：", "、".join(f"{v:.3f}" for v in y_tr))
print(f"这些收据上最好的直线：w = {p_line[0]:.3f}，b = {p_line[1]:.3f}")
YS_TEXT = [2.316, 3.533, 3.234, 4.959, 6.304, 7.406, 8.196, 9.449]          # 正文 JS 里写的 8 个数
check("正文列出的 8 张车费和直线 w = 2.074、b = 1.008 就是实验里的数",
      [f"{v:.3f}" for v in y_tr] == [f"{v:.3f}" for v in YS_TEXT] and (f"{p_line[0]:.3f}", f"{p_line[1]:.3f}") == ("2.074", "1.008"))
js_loss = mean([(2.074 * xv + 1.008 - yv) ** 2 for xv, yv in zip(np.linspace(0.5, 4.0, 8), YS_TEXT)])
check(f"按字面值重算（控制台那三行）：用印出的 8 个车费和 2.074、1.008，得 {js_loss:.6f}——以 0.146 开头",
      0.146 <= js_loss < 0.147)
check("随机误差：1 公里付 3.533 元（规则 3 元），1.5 公里反而只付 3.234 元（规则 4 元）",
      YS_TEXT[1] > YS_TEXT[2] and 2 * 1.0 + 1 == 3 and 2 * 1.5 + 1 == 4)
tr_line, tr_curve = mse_of(p_line, x_tr, y_tr), mse_of(p_curve, x_tr, y_tr)
va_line, va_curve = mse_of(p_line, x_va, y_va), mse_of(p_curve, x_va, y_va)
table(["模型", f"训练损失（{N_TRAIN} 张）", "验证损失（40 张新收据）"],
      [["直线：2 个旋钮", f"{tr_line:.3f}", f"{va_line:.3f}"], [f"曲线：{K_CURVE} 个旋钮", f"{tr_curve:.3f}", f"{va_curve:.3f}"]])
worst = int(np.argmax(np.abs(np.polyval(p_curve, x_va) - y_va)))
xw, yw = float(x_va[worst]), float(y_va[worst])
cw, lw_ = float(np.polyval(p_curve, xw)), float(np.polyval(p_line, xw))
gap_shown = round(cw, 2) - round(yw, 2)            # 用印出来的两位小数相减，读者按计算器也得这个数
print(f"错得最多的一张新收据：{xw:.2f} 公里、实付 {yw:.2f} 元；曲线猜 {cw:.2f} 元（差 {gap_shown:.2f}），直线猜 {lw_:.2f} 元")
check("曲线穿过每一张训练收据：训练损失算到小数点后 9 位还是 0", tr_curve < 1e-9)
check("直线训练损失 0.146、验证损失 0.114；曲线验证损失 0.706",
      f"{tr_line:.3f}" == "0.146" and f"{va_line:.3f}" == "0.114" and f"{va_curve:.3f}" == "0.706")
check(f"曲线的验证损失（{va_curve:.3f}）是直线（{va_line:.3f}）的 3 倍以上", va_curve > 3 * va_line)
check("正文：新收据上曲线的损失是直线的 6 倍多", 6 < va_curve / va_line < 7)
check("错得最多的那张：0.67 公里、实付 1.89 元，曲线猜 4.16 元（4.16 − 1.89 = 2.27），直线猜 2.39 元",
      (f"{xw:.2f}", f"{yw:.2f}", f"{cw:.2f}", f"{gap_shown:.2f}", f"{lw_:.2f}") == ("0.67", "1.89", "4.16", "2.27", "2.39"))
check("这一张的平方误差：曲线 2.27² = 5.1529（约 5.15），直线 0.5² = 0.25，曲线是直线的 20 多倍",
      round(2.27 ** 2, 4) == 5.1529 and round(2.39 - 1.89, 2) == 0.5 and 20 < 5.1529 / 0.25 < 21)
check(f"理论值：直线的验证损失接近压不掉的随机误差 σ² = {SIGMA ** 2:.2f}（在它的一半到两倍之间）",
      0.5 * SIGMA ** 2 < va_line < 2 * SIGMA ** 2)

# ---------------------------------------------------------------------------
banner("5b. 画图：figures/ch05_overfit.png（穿过每张收据的曲线 vs 直线）")
grid = np.linspace(0.5, 4.0, 400)
c_grid = np.polyval(p_curve, grid)
if not np.all(np.isfinite(c_grid)):
    print("  曲线的数不是有限值，跳过这张图。")
else:
    y_lo, y_hi = 0.0, 12.0
    fig = plt.figure(figsize=(9, 12.8))
    fig.suptitle("穿过每一张收据的曲线，换一批收据就错得多", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
    ax_a = fig.add_axes([0.12, 0.575, 0.83, 0.30])
    ax_b = fig.add_axes([0.12, 0.115, 0.83, 0.30])
    for ax in (ax_a, ax_b):
        data_axes(ax, "里程 x（公里）", "车费 y（元）")
        ax.set_xlim(0.3, 4.2)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
        ax.set_yticks(range(0, 13, 2))
        ax.plot(grid, np.polyval(p_line, grid), color=BLUE, lw=3, zorder=3)
        ax.plot(grid, np.clip(c_grid, y_lo - 1, y_hi + 1), color=ORANGE, lw=3, zorder=4)
        lx, ly = (0.45, float(np.polyval(p_line, 0.45)) - 0.35) if ax is ax_a else (1.3, 2.35)   # ② 里让开橙圈那张
        ax.text(lx, ly, "直线：2 个旋钮", color=BLUE, fontsize=FS_SMALL, va="top", bbox=WHITE_BOX)
    ax_a.scatter(x_tr, y_tr, s=95, color=INK, zorder=6)
    ax_a.text(0.62, 4.75, f"曲线：{K_CURVE} 个旋钮", color=ORANGE, fontsize=FS_SMALL, va="bottom", bbox=WHITE_BOX)
    ax_a.text(0.4, 11.6, f"训练损失：直线 {tr_line:.3f}\n　　　　　曲线 {tr_curve:.3f}", fontsize=FS_SMALL, color=INK,
              va="top", linespacing=1.4, bbox=WHITE_BOX)
    ax_b.scatter(x_va, y_va, s=70, facecolor="white", edgecolor=MUTED, lw=1.8, zorder=6)
    ax_b.scatter([xw], [yw], s=120, facecolor="white", edgecolor=ORANGE, lw=2.8, zorder=7)
    ax_b.plot([xw, xw], [yw, cw], color=ORANGE, lw=2.2, ls=":", zorder=5)
    ax_b.text(xw - 0.05, cw + 0.45, f"差 {gap_shown:.2f} 元", color=ORANGE, fontsize=FS_SMALL, ha="left",
              va="bottom", bbox=WHITE_BOX)
    ax_b.text(0.4, 11.6, f"验证损失：直线 {va_line:.3f}\n　　　　　曲线 {va_curve:.3f}", fontsize=FS_SMALL, color=INK,
              va="top", linespacing=1.4, bbox=WHITE_BOX)
    panel_title(fig, [ax_a], f"① 训练用的 {N_TRAIN} 张收据：曲线穿过每一张")
    panel_note(fig, [ax_a], "曲线的训练损失是 0，比直线还“好”；可它在收据之间拐来拐去——拐的是随机误差。")
    panel_title(fig, [ax_b], "② 40 张没参加训练的新收据：直线准，曲线错得多")
    panel_note(fig, [ax_b], f"空心点是新收据。橙圈那张：曲线猜 {cw:.2f} 元，实付 {yw:.2f} 元；直线猜 {lw_:.2f} 元。")
    savefig(fig, "ch05_overfit")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("6. 验证集：旋钮从 1 个加到 8 个，训练损失和验证损失各怎么走")
print("迷你版：5.5 节打平的三组旋钮，多一张没拿来拧旋钮的收据（1.5 公里，4 元）来打分")
NEW_X, NEW_Y = 1.5, 4.0
rows, mini_val = [], []
for b3, w3, c3 in ((1, 2, 0), (17, -22, 8), (-7, 14, -4)):
    pv = quad(b3, w3, c3, NEW_X)
    mini_val.append((pv - NEW_Y) ** 2)
    rows.append([f"({b3}, {w3}, {c3})", f"{loss_fn(quad(b3, w3, c3, X2), Y2):g}", f"{pv:g}", f"{pv - NEW_Y:g}", f"{mini_val[-1]:g}"])
table(["(b, w, c)", "训练损失（2 张）", "新收据上的预测（元）", "误差", "这一张上的损失"], rows)
check("迷你验证集：训练损失都是 0；新收据上的损失 0、4、1，挑中的是 (1, 2, 0)——藏着的规则",
      mini_val == [0, 4, 1] and int(np.argmin(mini_val)) == 0)
print()
ks = list(range(1, K_CURVE + 1))
tr_k = [mse_of(fit(k), x_tr, y_tr) for k in ks]
va_k = [mse_of(fit(k), x_va, y_va) for k in ks]
table(["旋钮个数", "训练损失", "验证损失"], [[k, f"{t:.3f}", f"{v:.3f}"] for k, t, v in zip(ks, tr_k, va_k)])
best_k = ks[int(np.argmin(va_k))]
print(f"验证损失最低的是 {best_k} 个旋钮；训练损失最低的是 {ks[int(np.argmin(tr_k))]} 个")
check("旋钮越多，训练损失只降不升", all(a >= b_ - 1e-12 for a, b_ in zip(tr_k, tr_k[1:])))
check("验证损失最低的是 2 个旋钮（直线）；只看训练损失会挑 8 个", best_k == 2 and ks[int(np.argmin(tr_k))] == 8)
check("7 个旋钮时训练损失 0.008（读图段引用）", f"{tr_k[6]:.3f}" == "0.008")
check("1 个旋钮：训练 5.792、验证 5.121（两边都差）；2 个：0.146 和 0.114；8 个：0.000 和 0.706",
      [f"{tr_k[0]:.3f}", f"{va_k[0]:.3f}", f"{tr_k[1]:.3f}", f"{va_k[1]:.3f}", f"{tr_k[-1]:.3f}", f"{va_k[-1]:.3f}"]
      == ["5.792", "5.121", "0.146", "0.114", "0.000", "0.706"])

# ---------------------------------------------------------------------------
banner("6b. 画图：figures/ch05_train_vs_val.png（训练损失一路降，验证损失先降后升）")
tr_arr, va_arr = np.array(tr_k), np.array(va_k)
if not (np.all(np.isfinite(tr_arr)) and np.all(np.isfinite(va_arr))):
    print("  有不是有限值的损失，跳过这张图。")
else:
    FLOOR = 0.004                                   # 对数刻度画不出 0：训练损失为 0 的点画在这条底边上
    fig = plt.figure(figsize=(9, 8.0))
    fig.suptitle("旋钮越多，训练损失越低；验证损失在 2 个旋钮处最低", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
    ax = fig.add_axes([0.15, 0.25, 0.8, 0.59])
    data_axes(ax, "模型的旋钮个数", "损失（对数刻度）")
    ax.set_yscale("log")
    ax.set_ylim(FLOOR, 20)
    ax.set_xlim(0.6, K_CURVE + 0.4)
    ax.set_xticks(ks)
    ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    solid = tr_arr >= FLOOR                          # 能画在对数刻度上的训练损失
    ax.plot(np.array(ks)[solid], tr_arr[solid], color=BLUE, lw=3, marker="o", ms=9, zorder=4)
    for i in np.flatnonzero(~solid):                 # 训练损失（几乎）为 0：虚线连到底边，倒三角画在底边上
        if i > 0:
            ax.plot([ks[i - 1], ks[i]], [tr_arr[i - 1], FLOOR], color=BLUE, lw=2.5, ls=":", zorder=4)
        ax.scatter([ks[i]], [FLOOR], s=150, marker="v", color=BLUE, zorder=6, clip_on=False)
        ax.text(ks[i] + 0.14, FLOOR * 1.12, "0", color=BLUE, fontsize=FS_SMALL, va="bottom")
    ax.plot(ks, va_arr, color=ORANGE, lw=3, marker="s", ms=9, zorder=4)
    ax.text(1.18, math.sqrt(tr_arr[0] * va_arr[0]), f"训练 {tr_arr[0]:.3f}\n验证 {va_arr[0]:.3f}", color=INK,
            fontsize=FS_SMALL, va="center", linespacing=1.35, bbox=WHITE_BOX)
    ax.text(6.2, 0.06, "训练损失", color=BLUE, fontsize=FS_SMALL, va="bottom", bbox=WHITE_BOX)
    ax.text(5.6, 0.5, "验证损失", color=ORANGE, fontsize=FS_SMALL, va="bottom", bbox=WHITE_BOX)
    ib = best_k - 1
    ax.scatter([best_k], [va_arr[ib]], s=420, marker="*", color=ORANGE, edgecolor=INK, lw=1, zorder=6)
    ax.text(best_k + 0.2, va_arr[ib] * 0.42, f"验证最低：{best_k} 个旋钮\n训练 {tr_arr[ib]:.3f}，验证 {va_arr[ib]:.3f}",
            color=INK, fontsize=FS_SMALL, va="top", bbox=WHITE_BOX, linespacing=1.35)
    ax.text(ks[-1], va_arr[-1] * 1.4, f"{va_arr[-1]:.3f}", color=ORANGE, fontsize=FS_SMALL, ha="center", va="bottom",
            bbox=WHITE_BOX)
    panel_title(fig, [ax], f"同样 {N_TRAIN} 张训练收据、40 张验证收据，旋钮从 1 个加到 {K_CURVE} 个")
    panel_note(fig, [ax], "纵轴是对数刻度：每往上一格，损失乘 10（第 3 章 3.5 节）。\n"
                          "8 个旋钮时训练损失是 0：对数刻度画不出 0，倒三角画在底边。\n"
                          "训练损失只降不升；验证损失才说明学没学会。")
    savefig(fig, "ch05_train_vs_val")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("7. 映射到项目：正文引用的常数和源码行还在不在")
REPO = Path(__file__).resolve().parents[3]
cfg_path = REPO / "src" / "mjlab_microduck" / "tasks" / "microduck_velocity_env_cfg.py"
CFG_LINES = ["actor=RslRlModelCfg(", "hidden_dims=(512, 256, 128),", "algorithm=PpoWithSymmetryCfg(", "num_learning_epochs=5,",
             "num_mini_batches=4,", "learning_rate=1.0e-3,", "num_steps_per_env=24,", "max_iterations=50_000,"]
RUNNER_LINES = ["total_it = start_it + num_learning_iterations",
                "for it in range(start_it, total_it):",
                "with torch.inference_mode():",
                'for _ in range(self.cfg["num_steps_per_env"]):',
                "actions = self.alg.act(obs)",
                "obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))",
                "self.alg.process_env_step(obs, rewards, dones, extras)",
                "self.alg.compute_returns(obs)",
                "loss_dict = self.alg.update()"]
PPO_LINES = ["self.optimizer.zero_grad()", "loss.backward()", "self.optimizer.step()"]

if cfg_path.is_file():
    cfg_text = cfg_path.read_text(encoding="utf-8")
    at = cfg_text.find("MicroduckRlCfg = RslRlOnPolicyRunnerCfg(")
    print("microduck_velocity_env_cfg.py 的 MicroduckRlCfg：", "  ".join(CFG_LINES))
    check("项目的超参数：网络形状 512-256-128、每轮数据用 5 遍 × 每遍 4 小批、学习率 0.001、每轮每只走 24 步、最多 50,000 轮",
          at >= 0 and lines_in_order(cfg_text[at:], CFG_LINES))
else:
    print("  （没找到项目的 env cfg，跳过这一项）")
rsl_spec = importlib.util.find_spec("rsl_rl")
rsl_dir = Path(rsl_spec.origin).parent if rsl_spec and rsl_spec.origin else None
if rsl_dir and (rsl_dir / "runners" / "on_policy_runner.py").is_file():
    runner_text = (rsl_dir / "runners" / "on_policy_runner.py").read_text(encoding="utf-8")
    at = runner_text.find("    def learn(")
    print("rsl_rl/runners/on_policy_runner.py 的 learn()：一轮 = 收数据（不记账）→ 整理得分 → update()")
    check(f"映射块引用的 learn() 里的 {len(RUNNER_LINES)} 行都原样存在，且顺序一致", at >= 0 and lines_in_order(runner_text[at:], RUNNER_LINES))
    ppo_text = (rsl_dir / "algorithms" / "ppo.py").read_text(encoding="utf-8")
    check("update() 里仍是 清零 → backward → 拧一步（第 3 章 📍 见过）",
          lines_in_order(ppo_text[ppo_text.find("    def update(self)"):], PPO_LINES))
    storage_text = (rsl_dir / "storage" / "rollout_storage.py").read_text(encoding="utf-8")
    at_mb = storage_text.find("    def mini_batch_generator(")
    check("一轮拧 5 × 4 = 20 次：update() 对每个小批走一遍 清零 → backward → step，小批来自“5 遍 × 每遍 4 小批”的两层循环",
          lines_in_order(ppo_text[ppo_text.find("    def update(self)"):],
                         ["self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)",
                          "for batch in generator:"] + PPO_LINES)
          and at_mb >= 0 and lines_in_order(storage_text[at_mb:], ["for epoch in range(num_epochs):", "for i in range(num_mini_batches):"])
          and 5 * 4 == 20 and 4096 * 24 == 98_304)
else:
    print("  （当前 Python 环境里没有 rsl_rl，跳过源码核对；用 uv run 运行就会核对）")
agents_md = REPO / "AGENTS.md"
if agents_md.is_file():
    agents_text = agents_md.read_text(encoding="utf-8")
    check("AGENTS.md：训练命令用 4096 只机器人；步态的经验预算 4000–6000 轮；观测 61 维；14 个舵机",
          "--env.scene.num-envs 4096" in agents_text and "4000–6000" in agents_text and "61D" in agents_text
          and "14 Dynamixel XL330 servos" in agents_text)
else:
    print("  （没找到 AGENTS.md，跳过这一项）")
mjlab_spec = importlib.util.find_spec("mjlab")
train_py = Path(mjlab_spec.origin).parent / "scripts" / "train.py" if mjlab_spec and mjlab_spec.origin else None
if train_py and train_py.is_file():
    check("mjlab/scripts/train.py 把配置里的 max_iterations 交给 learn() 当轮数",
          "num_learning_iterations=cfg.agent.max_iterations" in train_py.read_text(encoding="utf-8"))
else:
    print("  （当前 Python 环境里没有 mjlab，跳过这一项）")

done()
