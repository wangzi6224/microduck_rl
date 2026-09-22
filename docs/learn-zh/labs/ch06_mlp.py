"""第 6 章实验：一个神经元、一层、叠成四层的多层感知机（MLP）、数旋钮、手写前向传播、critic。

运行：uv run python docs/learn-zh/labs/ch06_mlp.py
纯 CPU：numpy + torch + matplotlib；第 5–7 节用 rsl_rl 在 CPU 上建出项目真实的 actor / critic（不加载机器人、不训练）。
小节编号与正文一一对应：实验第 K 节 = 正文 6.K 节（第 8 节对应「映射到项目」）。
正文“改一改”要改的三行都带 `# TWEAK-k:` 标记（第 3 节两处、第 6 节一处）。
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from tensordict import TensorDict

from _common import banner, check, done, lines_in_order, num, savefig, table
from _draw import (BLUE, CELL, CELL_EDGE, CELL_HOT, FAINT, FS_NOTE, FS_SMALL, FS_STEP, FS_TICK, FS_TITLE, GREEN, INK,
                   MUTED, ORANGE, WHITE_BOX, arrow, cell, data_axes, hand, lesson_cells, lesson_figure, lesson_panel,
                   note, panel_note, panel_title, plt)

np.set_printoptions(precision=4, suppress=True)


def u(v, places=4) -> str:
    """贴进图和打印的数：num() 的写法（最多 places 位小数、去掉尾零），负号换成正规的“−”。"""
    return num(v, places).replace("-", "−")


def relu(z):
    """ReLU：负数一律变成 0，正数原样。"""
    return np.maximum(0.0, z)


def elu(z):
    """ELU：z ≥ 0 原样；z < 0 变成 e^z − 1（永远大于 −1）。np.minimum 只是防止大正数去算 exp 溢出。"""
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0, z, np.exp(np.minimum(z, 0.0)) - 1.0)


# ---------------------------------------------------------------------------
banner("1. 一个神经元：清单 × 价格，加上运费，再过一道门")
x_mini = np.array([0.2, -0.1])            # 教学构造的两个输入（为了好算，不是真实观测）
w_A, b_A = np.array([2.0, -1.0]), 0.1     # 店 A 的价格表和运费
w_B, b_B = np.array([-3.0, 1.0]), 0.0     # 店 B 的价格表和运费
dot_A, dot_B = float(w_A @ x_mini), float(w_B @ x_mini)
z_A, z_B = dot_A + b_A, dot_B + b_B
print(f"店 A：2 × 0.2 + (−1) × (−0.1) = {u(dot_A)}；加运费 0.1 → z = {u(z_A)}")
print(f"店 B：(−3) × 0.2 + 1 × (−0.1) = {u(dot_B)}；加运费 0 → z = {u(z_B)}")
check("店 A：0.4 + 0.1 = 0.5，加运费 0.1 → z = 0.6", math.isclose(dot_A, 0.5) and math.isclose(z_A, 0.6))
check("店 B：−0.6 − 0.1 = −0.7，运费 0 → z = −0.7", math.isclose(dot_B, -0.7) and math.isclose(z_B, -0.7))

elu_B = float(elu(z_B))
print(f"ReLU(0.6) = {u(float(relu(z_A)))}，ReLU(−0.7) = {u(float(relu(z_B)))}")
print(f"ELU(0.6) = {u(float(elu(z_A)))}；ELU(−0.7) = e^−0.7 − 1 = {math.exp(-0.7):.4f} − 1 = {u(elu_B)}"
      f"（浏览器控制台 Math.exp(-0.7) - 1 → {math.exp(-0.7) - 1!r}）")
check("ReLU：0.6 原样通过，−0.7 砍成 0", float(relu(z_A)) == z_A and float(relu(z_B)) == 0.0)
check("ELU：0.6 原样通过；−0.7 → e^−0.7 − 1 ≈ −0.5034", float(elu(z_A)) == z_A and round(elu_B, 4) == -0.5034)
check("字面值重算：e^−0.7 ≈ 0.4966，0.4966 − 1 = −0.5034", round(math.exp(-0.7), 4) == 0.4966 and round(0.4966 - 1, 4) == -0.5034)
check("控制台 Math.exp(-0.7) - 1 的全精度 −0.5034146962085905（正文和 JS 注释里的那个数）",
      repr(math.exp(-0.7) - 1) == "-0.5034146962085905")

Z_TABLE = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]
table(["门前的数 z", "ReLU(z)", "ELU(z)"], [[z, float(relu(z)), float(elu(z))] for z in Z_TABLE], floatfmt=".3f")
check("表里 ELU 的四个负数：−0.950、−0.865、−0.632、−0.393",
      [round(float(elu(z)), 3) for z in (-3, -2, -1, -0.5)] == [-0.95, -0.865, -0.632, -0.393])
check("手写的 ELU = torch.nn.ELU()（项目用的那道门）",
      np.allclose(elu(Z_TABLE), torch.nn.ELU()(torch.tensor(Z_TABLE)).numpy(), atol=1e-6))
check("手写的 ReLU = torch.nn.ReLU()", np.allclose(relu(np.array(Z_TABLE)), torch.nn.ReLU()(torch.tensor(Z_TABLE)).numpy()))
print(f"ELU(−10) = {u(float(elu(-10.0)), 5)}（控制台 Math.exp(-10) - 1 → {math.exp(-10) - 1!r}）")
check("ELU 越往左越贴近 −1，但到不了：z = −10 时是 −0.99995", round(float(elu(-10.0)), 5) == -0.99995 and float(elu(-10.0)) > -1)


def steps(f):
    """输入从 −2 到 −1、从 −1 到 0、从 0 到 1，每多 1，输出多多少（3 位小数）。"""
    return [round(float(f(b)) - float(f(a)), 3) for a, b in ((-2, -1), (-1, 0), (0, 1))]


print("输入每多 1，输出多多少（−2→−1、−1→0、0→1）：不装门", steps(lambda z: z), "；ReLU", steps(relu), "；ELU", steps(elu))
check("不装门（原样输出）：每一段都多 1——一条直线", steps(lambda z: z) == [1, 1, 1])
check("ReLU：0、0、1；ELU：0.233、0.632、1——每段不一样，不是直线", steps(relu) == [0, 0, 1] and steps(elu) == [0.233, 0.632, 1.0])
check("字面值重算：−0.632 − (−0.865) = 0.233", round(-0.632 - (-0.865), 3) == 0.233)
check("自测：z = −1.5 → ReLU 0，ELU = e^−1.5 − 1 ≈ 0.2231 − 1 = −0.7769；z = 1.5 → 两道门都原样 1.5",
      float(relu(-1.5)) == 0 and round(float(elu(-1.5)), 4) == -0.7769 and round(math.exp(-1.5), 4) == 0.2231
      and round(0.2231 - 1, 4) == -0.7769 and float(relu(1.5)) == float(elu(1.5)) == 1.5)

# ---------------------------------------------------------------------------
banner("1b. 画图：figures/ch06_neuron.png（一个神经元 = 一家店结账）")
fig, axes = lesson_figure(3, "一个神经元：清单 × 价格，加上运费，再过一道门", panel_height=3.1, width=10.0)
ax = axes[0]
lesson_panel(ax, "① 清单 × 价格：数量乘单价，再加起来", xmax=10, ymax=4.2)
for y, label, vals, calc, color in ((2.45, "清单 x", ["0.2", "−0.1"], "", INK),
                                    (1.6, "店 A 的价格", ["2", "−1"], f"2 × 0.2 + (−1) × (−0.1) = {u(dot_A)}", BLUE),
                                    (0.75, "店 B 的价格", ["−3", "1"], f"(−3) × 0.2 + 1 × (−0.1) = {u(dot_B)}", GREEN)):
    ax.text(0.2, y + 0.27, label, fontsize=FS_SMALL, color=color, va="center")
    lesson_cells(ax, [vals], 2.3, y, width=0.95, height=0.55)
    if calc:
        hand(ax, 4.5, y + 0.27, calc, color=color)
ax = axes[1]
lesson_panel(ax, "② 加运费：每家店再固定加一笔", xmax=10, ymax=4.2)
hand(ax, 0.3, 2.55, f"店 A：{u(dot_A)} + 0.1 = {u(z_A)}", color=BLUE)
hand(ax, 0.3, 1.65, f"店 B：{u(dot_B)} + 0 = {u(z_B)}", color=GREEN)
note(ax, 0.3, 0.6, "运费就是偏置 b（第 2 章 2.6 节），和清单无关。加完运费的数记作 z。")
ax = axes[2]
lesson_panel(ax, "③ 过一道门（ELU）：正数原样放行，负数压扁", xmax=10, ymax=4.2)
hand(ax, 0.3, 2.55, f"店 A：z = {u(z_A)}，不是负数 → 原样放行 → {u(float(elu(z_A)))}", color=BLUE)
hand(ax, 0.3, 1.65, f"店 B：z = {u(z_B)}，是负数 → $e^{{-0.7}} - 1$ = {u(elu_B)}", color=ORANGE)
note(ax, 0.3, 0.6, "负数过了这道门，永远不会低于 −1。每家店最后吐出一个数。")
savefig(fig, "ch06_neuron")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("1c. 画图：figures/ch06_gates.png（ReLU 和 ELU 的图像：都不是一条直线）")
fig = plt.figure(figsize=(9.4, 12.2))
gs = fig.add_gridspec(2, 1, hspace=0.66, left=0.13, right=0.96, top=0.885, bottom=0.105)
fig.suptitle("两道门都不是直线：正数原样通过，负数被改写", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.975)
zz = np.linspace(-3.3, 1.9, 400)
for row, (name, f, color) in enumerate((("ReLU", relu, BLUE), ("ELU", elu, ORANGE))):
    ax = fig.add_subplot(gs[row, 0])
    data_axes(ax, "门前的数 z", "过门后的数")
    ax.set_xlim(-3.3, 1.9)
    ax.set_ylim(-1.65, 1.9)
    ax.set_xticks([-3, -2, -1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.plot(zz, zz, color=FAINT, ls="--", lw=2.2)
    ax.text(-0.66 if name == "ReLU" else -1.3, -0.9 if name == "ReLU" else -1.5, "不装门：原样输出（一条直线）",
            color=MUTED, fontsize=FS_SMALL, va="center")
    ax.plot(zz, f(zz), color=color, lw=3.6)
    ax.plot([0.6], [0.6], "o", color=color, ms=10, zorder=6)
    ax.text(0.72, 0.3, f"{name}(0.6) = 0.6", color=color, fontsize=FS_SMALL, va="center", bbox=WHITE_BOX)
    ax.plot([-0.7], [float(f(-0.7))], "o", color=color, ms=10, zorder=6)
    if name == "ReLU":
        ax.text(-0.7, 0.28, "ReLU(−0.7) = 0", color=color, fontsize=FS_SMALL, ha="center", va="center", bbox=WHITE_BOX)
        panel_title(fig, [ax], "① ReLU：负数一律变成 0")
        panel_note(fig, [ax], "左边平、右边斜：两段直线在 0 处折开，整条线不是一根直线。")
    else:
        ax.text(-0.5, -0.66, f"ELU(−0.7) = {u(elu_B)}", color=color, fontsize=FS_SMALL, va="center", bbox=WHITE_BOX)
        ax.plot([-3.0], [float(elu(-3.0))], "o", color=color, ms=10, zorder=6)
        ax.text(-3.0, -0.62, f"ELU(−3) = {float(elu(-3.0)):.3f}".replace("-", "−"), color=color, fontsize=FS_SMALL,
                va="center", bbox=WHITE_BOX)
        ax.axhline(-1, color=GREEN, ls=":", lw=2.2)
        ax.text(-3.2, -1.19, "底线 −1，永远到不了", color=GREEN, fontsize=FS_SMALL, va="center")
        panel_title(fig, [ax], "② ELU：负数变成 $e^z - 1$，压在 −1 以上（项目用这道）")
        panel_note(fig, [ax], "负半边是一段弯曲线（在 0 处不折角，平滑地弯过去），越往左越贴近 −1；\n正半边和 ReLU 一样原样通过。")
savefig(fig, "ch06_gates")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("2. 层：一排店对着同一张清单，同时结账")
W1 = np.array([w_A, w_B])        # 两家店的价格表摞成 2 行 2 列（第 2 章 2.5 节）
b1 = np.array([b_A, b_B])        # 两家店的运费
z1 = W1 @ x_mini + b1
h1 = elu(z1)                     # 门对每一项各过一次
print("W₁ =", W1.tolist(), "  b₁ =", b1.tolist())
print(f"W₁x + b₁ = ({u(z1[0])}, {u(z1[1])})  → 逐项过门 → h₁ = ({u(h1[0])}, {u(h1[1])})")
check("矩阵写法 W₁x + b₁ 就是两家店各算一次：(0.6, −0.7)", np.allclose(z1, [z_A, z_B]))
check("门逐项过：h₁ = (0.6, −0.5034)", np.allclose(np.round(h1, 4), [0.6, -0.5034]))
lin = torch.nn.Linear(2, 2)
with torch.no_grad():
    lin.weight.copy_(torch.tensor(W1))
    lin.bias.copy_(torch.tensor(b1))
    z_torch = lin(torch.tensor(x_mini, dtype=torch.float32)).numpy()
check("torch.nn.Linear(2, 2) 装上同一张价格表和运费，也得 (0.6, −0.7)", np.allclose(z_torch, z1, atol=1e-6))
check("自测：店 B 的运费从 0 改成 1 → z = (0.6, 0.3)，都不是负数 → h₁ = (0.6, 0.3)",
      np.allclose(elu(W1 @ x_mini + np.array([0.1, 1.0])), [0.6, 0.3]))
first_layer = torch.nn.Linear(61, 512)
check("项目第一层：512 × 61 的价格表 + 512 个运费 = 31,744 个旋钮（第 2 章 2.6 节）",
      sum(p.numel() for p in first_layer.parameters()) == 31744)

# ---------------------------------------------------------------------------
banner("3. 把层叠起来：上一排报出的清单，是下一排的购物清单")
# 迷你网络用哪道门（项目用 ELU）。“改一改”第 1 条改这一行。
MINI_GATE = "elu"  # TWEAK-1: "relu"
gate = elu if MINI_GATE == "elu" else relu
GATE_NAME = {"elu": "ELU", "relu": "ReLU"}.get(MINI_GATE, MINI_GATE)
W2 = np.array([[1.0, -2.0]])     # 第二层只有一家店 C：价格 (1, −2)
b2 = np.array([0.0])             # 店 C 的运费 0
h1_mini = gate(W1 @ x_mini + b1)
mu_mini = float((W2 @ h1_mini + b2)[0])   # 最后一层不过门
print(f"第一层（门用 {GATE_NAME}）：h₁ = ({u(h1_mini[0])}, {u(h1_mini[1])})")
print(f"第二层：μ = 1 × {u(h1_mini[0])} + (−2) × ({u(h1_mini[1])}) + 0 = {mu_mini:.4f}（全精度 {mu_mini!r}）")
print(f"控制台核对：0.6 - 2 * (Math.exp(-0.7) - 1) → {0.6 - 2 * (math.exp(-0.7) - 1)!r}")
check("μ = 0.6 − 2 × (e^−0.7 − 1) = 1.606829（worked_examples 的“两层教学网络输出”是同一个数）", round(mu_mini, 6) == 1.606829)
check("字面值重算：0.6 + 2 × 0.5034 = 0.6 + 1.0068 = 1.6068", round(0.6 + 2 * 0.5034, 4) == 1.6068 and round(mu_mini, 4) == 1.6068)
check("自测：店 C 的价格换成 (1, 2) → 0.6 − 1.0068 = −0.4068",
      round(float(np.array([1.0, 2.0]) @ h1_mini), 4) == -0.4068 and round(0.6 - 1.0068, 4) == -0.4068)
check("最后一层不过门的理由：门前 −1.5 过 ELU 只剩 −0.7769；门前再负（−3 → −0.950，−10 → −0.99995）也只贴近 −1，到不了 −1.5",
      round(float(elu(-1.5)), 4) == -0.7769 and round(float(elu(-3.0)), 3) == -0.95 and round(float(elu(-10.0)), 5) == -0.99995
      and float(np.min(elu(np.linspace(-20, 0, 2001)))) > -1 > -1.5)

N_OBS, N_ACT = 61, 14
# 项目 actor 三个隐藏层的宽度（cfg 里 hidden_dims=(512, 256, 128)）。“改一改”第 2 条改这一行。
HIDDEN = (512, 256, 128)  # TWEAK-2: (64, 64)
SIZES = [N_OBS, *HIDDEN, N_ACT]
PROJECT_SIZES = [61, 512, 256, 128, 14]
print("项目 actor 的几串数：", " → ".join(str(s) for s in SIZES),
      f"（{len(SIZES)} 串数，{len(SIZES) - 1} 层 = {len(SIZES) - 1} 张价格表，其中 {len(HIDDEN)} 层是隐藏层）")
check("actor：61 → 512 → 256 → 128 → 14：5 串数、4 层（4 张价格表），其中 3 层是隐藏层", SIZES == PROJECT_SIZES)

# ---------------------------------------------------------------------------
banner("3b. 画图：figures/ch06_stack.png（四层堆叠，条的长度按数的个数成比例）")
if SIZES != PROJECT_SIZES:
    print("  隐藏层改过了：这张图是照项目的 61 → 512 → 256 → 128 → 14 画的，跳过。看上面那行就行。")
else:
    fig, ax = plt.subplots(figsize=(10.0, 11.8))
    fig.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.02)
    fig.suptitle("actor 的四层：61 个观测，一层层变成 14 个动作均值", fontsize=FS_TITLE, fontweight="bold", color=INK)
    lesson_panel(ax, xmax=10, ymax=12.6)
    names = ["输入 x：61 个观测", "隐藏层 1：h₁，512 个数", "隐藏层 2：h₂，256 个数", "隐藏层 3：h₃，128 个数", "输出 μ：14 个动作均值"]
    ops = ["× W₁（512 行 61 列）+ b₁，过门", "× W₂（256 行 512 列）+ b₂，过门", "× W₃（128 行 256 列）+ b₃，过门",
           "× W₄（14 行 128 列）+ b₄，不过门"]
    FULL = 6.8                                         # 512 个数画成 6.8 格长，其余按比例
    tops = []
    for k, n in enumerate(SIZES):
        y0 = 10.9 - 2.55 * k
        tops.append(y0)
        face, edge = (CELL, BLUE) if k == 0 else ((CELL_HOT, ORANGE) if k == len(SIZES) - 1 else (CELL, CELL_EDGE))
        ax.add_patch(plt.Rectangle((0.4, y0), FULL * n / 512, 0.5, facecolor=face, edgecolor=edge, lw=2))
        ax.text(0.4, y0 + 0.85, names[k], fontsize=FS_STEP, color=INK, va="center", fontweight="bold" if k in (0, 4) else None)
        if k < len(ops):
            arrow(ax, (0.9, y0 - 0.12), (0.9, y0 - 1.4), MUTED, lw=2.5)
            hand(ax, 1.3, y0 - 0.78, ops[k], color=ORANGE if k == len(ops) - 1 else BLUE, fontsize=FS_SMALL)
    ax.plot([7.75, 7.95, 7.95, 7.75], [tops[1] + 0.5, tops[1] + 0.5, tops[3], tops[3]], color=GREEN, lw=2.4)
    ax.text(8.15, (tops[1] + tops[3] + 0.5) / 2, "3 个隐藏层\n报出的\n3 串数", color=GREEN, fontsize=FS_SMALL, va="center")
    note(ax, 0.4, -0.35, "条的长度按数的个数成比例：512 最长，14 最短。一共 4 张价格表、5 串数。")
    ax.set_ylim(-0.75, 12.6)
    savefig(fig, "ch06_stack")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("4. 为什么要那道门：拆掉门，两层就能合并成一层")
z1_mini = W1 @ x_mini + b1
mu_nogate = float((W2 @ z1_mini + b2)[0])
W_merged = (W2 @ W1)[0]                 # 合并成的一家店的价格表
b_merged = float((W2 @ b1 + b2)[0])     # 它的运费
mu_merged = float(W_merged @ x_mini + b_merged)
print(f"拆掉门：μ = 1 × {u(z1_mini[0])} + (−2) × ({u(z1_mini[1])}) + 0 = {mu_nogate:.1f}")
print(f"合并店：价格 ({u(W_merged[0])}, {u(W_merged[1])})，运费 {u(b_merged)}；"
      f"{u(W_merged[0])} × 0.2 + ({u(W_merged[1])}) × (−0.1) + {u(b_merged)} = {mu_merged:.1f}")
check("拆掉门：μ = 1 × 0.6 + (−2) × (−0.7) = 0.6 + 1.4 = 2.0", math.isclose(mu_nogate, 2.0))
check("合并成一家店：价格 (1×2 + (−2)×(−3), 1×(−1) + (−2)×1) = (8, −3)，运费 1×0.1 + (−2)×0 + 0 = 0.1",
      np.allclose(W_merged, [8, -3]) and math.isclose(b_merged, 0.1))
check("合并店：8 × 0.2 + (−3) × (−0.1) + 0.1 = 1.6 + 0.3 + 0.1 = 2.0，和两层一样",
      math.isclose(mu_merged, 2.0) and math.isclose(mu_merged, mu_nogate))
x11 = np.array([1.0, 1.0])
check("自测：x = (1, 1)：两层拆掉门 1.1 + 4 = 5.1；合并店 8 − 3 + 0.1 = 5.1",
      math.isclose(float((W2 @ (W1 @ x11 + b1) + b2)[0]), 5.1) and math.isclose(float(W_merged @ x11 + b_merged), 5.1))
check(f"夹着门（{GATE_NAME}）：μ = {mu_mini:.4f}，不等于合并店的 2.0", not math.isclose(mu_mini, mu_merged))
rng = np.random.default_rng(0)
W1r, W2r, xr = rng.normal(size=(4, 3)), rng.normal(size=(2, 4)), rng.normal(size=3)
check("更大的随机例子（4×3 和 2×4 两张表）：不夹门时，先后乘 = 先合并成一张 2×3 的表再乘",
      np.allclose(W2r @ (W1r @ xr), (W2r @ W1r) @ xr))
check("同一个随机例子夹上 ELU，就不再等于那张合并的表", not np.allclose(W2r @ elu(W1r @ xr), (W2r @ W1r) @ xr))


def mini_net(x1, with_gate):
    """迷你网络：第 2 个输入固定为 −0.1，只拧第 1 个输入 x₁。"""
    z = W1 @ np.array([x1, -0.1]) + b1
    return float((W2 @ (gate(z) if with_gate else z) + b2)[0])


X1_ROW = [-2.0, -1.0, 0.0, 1.0, 2.0]
nog = [mini_net(v, False) for v in X1_ROW]
wg = [mini_net(v, True) for v in X1_ROW]
table(["x₁", "拆掉门", "夹着门"], [[f"{v:g}", a, b] for v, a, b in zip(X1_ROW, nog, wg)], floatfmt=".4f")
step_nog = [round(b - a, 3) for a, b in zip(nog, nog[1:])]
step_wg = [round(b - a, 3) for a, b in zip(wg, wg[1:])]
print("x₁ 每多 1，μ 多多少：拆掉门", step_nog, "；夹着门", step_wg)
check("拆掉门：每多 1 都多 8（一条直线；8 就是合并店的第 1 个价格）", step_nog == [8.0] * 4)
check("拆掉门那一列：−15.6、−7.6、0.4、8.4、16.4；夹着门那一列：−12.7776、−6.6347、0.3903、4.1099、6.1955",
      [round(v, 4) for v in nog] == [-15.6, -7.6, 0.4, 8.4, 16.4]
      and [round(v, 4) for v in wg] == [-12.7776, -6.6347, 0.3903, 4.1099, 6.1955])
check("夹着门：多 6.143、7.025、3.720、2.086，每段不一样（弯了）", step_wg == [6.143, 7.025, 3.72, 2.086])
check("字面值重算：−6.6347 − (−12.7776) = 6.143，0.3903 − (−6.6347) = 7.025",
      round(-6.6347 - (-12.7776), 3) == 6.143 and round(0.3903 - (-6.6347), 3) == 7.025)
wg4 = [round(v, 4) for v in wg]
check("字面值重算：用表里印出的 4 位小数相减，也得这四个数", [round(b - a, 3) for a, b in zip(wg4, wg4[1:])] == step_wg)

# ---------------------------------------------------------------------------
banner("4b. 画图：figures/ch06_gate_vs_nogate.png（拆掉门是直线，留着门才会弯）")
if MINI_GATE != "elu":
    print("  迷你网络的门改过了：这张图照 ELU 画、标的是 ELU 的数，跳过。看上面那张表就行。")
else:
    fig = plt.figure(figsize=(9.4, 12.6))
    gs = fig.add_gridspec(2, 1, hspace=0.66, left=0.13, right=0.96, top=0.885, bottom=0.10)
    fig.suptitle("拆掉门只剩一条直线，留着门才会弯", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.975)
    xs = np.linspace(-2.2, 2.2, 400)
    line_nog = np.array([mini_net(v, False) for v in xs])
    line_merged = W_merged[0] * xs + W_merged[1] * (-0.1) + b_merged
    curve_wg = np.array([mini_net(v, True) for v in xs])
    for row, (ys, pts, color) in enumerate(((line_nog, nog, BLUE), (curve_wg, wg, ORANGE))):
        ax = fig.add_subplot(gs[row, 0])
        data_axes(ax, "第 1 个输入 x₁（第 2 个输入固定为 −0.1）", "网络输出 μ")
        ax.set_xlim(-2.3, 2.3)
        ax.set_ylim(-19.0, 20.0)
        ax.set_xticks(X1_ROW)
        ax.set_yticks([-16, -8, 0, 8, 16])
        if row == 0:
            ax.plot(xs, ys, color=BLUE, lw=6, alpha=0.85)
            ax.plot(xs, line_merged, color=ORANGE, ls="--", lw=2.6)
            ax.text(-2.15, 17.0, "蓝：两层、拆掉门", color=BLUE, fontsize=FS_SMALL, va="center")
            ax.text(-2.15, 12.0, "橙虚线：合并成的一家店\n价格 (8, −3)，运费 0.1", color=ORANGE, fontsize=FS_SMALL, va="center")
        else:
            ax.plot(xs, line_nog, color=FAINT, ls="--", lw=2.2)
            ax.text(1.2, 16.2, "① 的直线（对照）", color=MUTED, fontsize=FS_SMALL, ha="center", va="center", bbox=WHITE_BOX)
            ax.plot(xs, ys, color=ORANGE, lw=4)
        ax.plot(X1_ROW, pts, "o", color=color, ms=9, zorder=6)
        for (xa, ya), (xb, yb) in zip(zip(X1_ROW, pts), zip(X1_ROW[1:], pts[1:])):
            step_label = f"+{u(yb - ya, 3)}" if row == 0 else f"+{yb - ya:.3f}"   # ② 固定 3 位小数，和正文 3.720 一致
            ax.text((xa + xb) / 2 + 0.06, (ya + yb) / 2 - 0.7, step_label, color=color, fontsize=FS_SMALL,
                    ha="left", va="top", bbox=WHITE_BOX)
        y02 = mini_net(0.2, row == 1)
        ax.plot([0.2], [y02], "D", color=INK, ms=9, zorder=7)
        ax.text(0.1, y02 + 1.1, f"x₁ = 0.2：μ = {y02:.1f}" if row == 0 else f"x₁ = 0.2：μ = {u(y02)}", color=INK, fontsize=FS_SMALL, ha="right", va="bottom", bbox=WHITE_BOX)
        if row == 0:
            panel_title(fig, [ax], "① 拆掉门：两层 = 一家店，一条直线")
            panel_note(fig, [ax], "蓝线和橙虚线完全重合。x₁ 每多 1，μ 都多 8：直线就是这个样子。")
        else:
            panel_title(fig, [ax], "② 留着门（ELU）：输出弯了")
            panel_note(fig, [ax], "每多 1，μ 多的量一段一段不一样：不是直线，哪一家店都合并不了它。\n两张图刻度相同。")
    savefig(fig, "ch06_gate_vs_nogate")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("5. 数一数旋钮：每一层 = 价格表的格子 + 运费")
mini_counts = [n_out * n_in + n_out for n_in, n_out in ((2, 2), (2, 1))]
check("迷你网络：第一层 2×2 + 2 = 6，第二层 1×2 + 1 = 3，一共 9 个旋钮", mini_counts == [6, 3] and sum(mini_counts) == 9)


def count(sizes):
    """每层 = 输出个数 × 输入个数（价格表）+ 输出个数（运费）。"""
    return [n_out * n_in + n_out for n_in, n_out in zip(sizes[:-1], sizes[1:])]


counts = count(SIZES)
rows = [[f"{i} → {o}", f"{o}×{i} = {o * i:,}", f"{o}", f"{c:,}", f"{o} × {i + 1}"]
        for (i, o), c in zip(zip(SIZES[:-1], SIZES[1:]), counts)]
table(["层", "价格表 W 的格子", "运费 b", "旋钮数", "= 输出 × (输入 + 1)"], rows + [["合计", "", "", f"{sum(counts):,}", ""]])
total_mlp = sum(counts)
check("四层：31,744 + 131,328 + 32,896 + 1,806 = 197,774", counts == [31744, 131328, 32896, 1806] and total_mlp == 197774)
check("每层 = 输出个数 × (输入个数 + 1)：512 × 62、256 × 513、128 × 257、14 × 129",
      [512 * 62, 256 * 513, 128 * 257, 14 * 129] == [31744, 131328, 32896, 1806])
check("中间那张 256 × 512 的表最大：131,328 / 197,774 ≈ 0.664，约三分之二", round(131328 / 197774, 3) == 0.664)

actor = MLPModel(TensorDict({"actor": torch.zeros(1, N_OBS)}, batch_size=[1]), {"actor": ["actor"]}, "actor", N_ACT,
                 HIDDEN, "elu", True, {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"})
n_mlp = sum(p.numel() for p in actor.mlp.parameters())
n_trainable = sum(p.numel() for p in actor.parameters() if p.requires_grad)
extra = [(name, p.numel()) for name, p in actor.named_parameters() if not name.startswith("mlp.")]
print(f"rsl_rl 照 cfg 建出的 actor：mlp 里 {n_mlp:,} 个；mlp 之外的可训练旋钮 {extra}；可训练合计 {n_trainable:,}")
check(f"rsl_rl 建出的 actor，MLP 的旋钮 = 上表合计 {total_mlp:,}", n_mlp == total_mlp)
check("正文的数：四层 MLP 正好 197,774 个", n_mlp == 197774)
check("mlp 之外只有 distribution.std_param：14 个 σ（第 4 章 4.7 节），初值都是 1.0；归一化那一步没有可训练的旋钮",
      extra == [("distribution.std_param", 14)] and bool(torch.all(actor.distribution.std_param == 1.0)))
check("actor 可训练的旋钮合计 197,774 + 14 = 197,788", n_trainable == 197788 and 197774 + 14 == 197788)
check("门没有旋钮：torch.nn.ELU() 里一个参数也没有", sum(p.numel() for p in torch.nn.ELU().parameters()) == 0)
c256 = count([61, 256, 256, 128, 14])
check("自测：第一层改成 61 → 256：15,872 + 65,792，比原来少 (31,744 − 15,872) + (131,328 − 65,792) = 81,408",
      c256[:2] == [15872, 65792] and 197774 - sum(c256) == 81408 and (31744 - 15872) + (131328 - 65792) == 81408)
check("自测：输入从 61 变 62：只多一列价格，512 个", sum(count([62, 512, 256, 128, 14])) - 197774 == 512)

# ---------------------------------------------------------------------------
banner("5b. 画图：figures/ch06_overview.png（6.0 节的总览：一家店 → 一排店 → 一排接一排）")
if MINI_GATE != "elu" or SIZES != PROJECT_SIZES:
    print("  门或隐藏层改过了：总览图标的是正文那组数，跳过。")
else:
    fig, axes = lesson_figure(4, "神经网络：一家店结账，一排店是一层，一层接一层", panel_height=2.8, width=10.0)

    def row_cells(ax, left, texts, width=0.95, hot=False):
        lesson_cells(ax, [texts], left, 1.75, width=width, height=0.62,
                     highlights=[(0, j) for j in range(len(texts))] if hot else ())

    def shop(ax, left, text, width=1.4):
        cell(ax, left, 1.75, text, width=width, height=0.62, facecolor=CELL, edgecolor=BLUE, color=BLUE, fontsize=FS_SMALL)

    def link(ax, x0, x1):
        arrow(ax, (x0, 2.06), (x1, 2.06), MUTED, lw=2.4)

    ax = axes[0]
    lesson_panel(ax, "① 一个神经元：一家店结账，吐出一个数", xmax=10, ymax=4)
    row_cells(ax, 0.3, ["0.2", "−0.1"])
    link(ax, 2.3, 3.2)
    shop(ax, 3.3, "店 A")
    link(ax, 4.8, 5.7)
    row_cells(ax, 5.8, [u(float(elu(z_A)))], hot=True)
    hand(ax, 0.3, 0.95, f"2 × 0.2 + (−1) × (−0.1) + 运费 0.1 = {u(z_A)}，过门后还是 {u(float(elu(z_A)))}", fontsize=FS_SMALL)
    note(ax, 0.3, 0.3, "6.1 节：清单 × 价格，加运费，再过一道门。")
    ax = axes[1]
    lesson_panel(ax, "② 一层：一排店对着同一张清单，吐出一串数", xmax=10, ymax=4)
    row_cells(ax, 0.3, ["0.2", "−0.1"])
    link(ax, 2.3, 3.2)
    shop(ax, 3.3, "店 A、店 B", width=2.0)
    link(ax, 5.4, 6.3)
    lesson_cells(ax, [[u(h1[0]), u(h1[1])]], 6.3, 1.75, width=1.5, height=0.62, highlights=[(0, 0), (0, 1)], fontsize=FS_SMALL)
    hand(ax, 0.3, 0.95, f"店 B：(−3) × 0.2 + 1 × (−0.1) + 0 = {u(z_B)}，过门 → {u(h1[1])}", fontsize=FS_SMALL)
    note(ax, 0.3, 0.3, "6.2 节：一排店就是一层，结果排成一张新清单。")
    ax = axes[2]
    lesson_panel(ax, "③ 叠起来：这串数又是下一家店的清单", xmax=10, ymax=4)
    ax.text(1.7, 2.62, "② 吐出的那串数", fontsize=FS_SMALL, color=MUTED, ha="center", va="center")
    lesson_cells(ax, [[u(h1[0]), u(h1[1])]], 0.3, 1.75, width=1.4, height=0.62, fontsize=FS_SMALL)
    link(ax, 3.2, 4.1)
    shop(ax, 4.2, "店 C", width=1.4)
    link(ax, 5.7, 6.6)
    lesson_cells(ax, [[u(mu_mini)]], 6.7, 1.75, width=1.4, height=0.62, highlights=[(0, 0)], fontsize=FS_SMALL)
    hand(ax, 0.2, 0.95, f"店 C：1 × {u(h1[0])} + (−2) × ({u(h1[1])}) + 0 = {u(mu_mini)}", fontsize=FS_SMALL)
    note(ax, 0.2, 0.3, "6.3 节：层接层；6.4 节看看拆掉门会怎样。")
    ax = axes[3]
    lesson_panel(ax, "④ 项目的 actor：同样的做法，放大成四层", xmax=10, ymax=4)
    for k, (n, lab) in enumerate(zip(SIZES, ["观测", "隐藏层 1", "隐藏层 2", "隐藏层 3", "动作均值"])):
        left = 0.3 + 2.0 * k
        lesson_cells(ax, [[str(n)]], left, 1.6, width=1.15, height=0.62, highlights=[(0, 0)] if k == 4 else ())
        ax.text(left + 0.575, 2.55, lab, fontsize=FS_SMALL, color=MUTED, ha="center", va="center")
        if k < 4:
            arrow(ax, (left + 1.22, 1.91), (left + 1.93, 1.91), MUTED, lw=2.4)
    hand(ax, 0.3, 0.95, f"4 张价格表 + 运费：一共 {total_mlp:,} 个旋钮（加上 14 个 σ 是 {total_mlp + 14:,}）", fontsize=FS_SMALL)
    note(ax, 0.3, 0.3, "6.5–6.7 节。格子里写的是每串有几个数，格子大小不按比例。")
    savefig(fig, "ch06_overview")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("6. 它就是一个函数：前向传播，手写四行")
torch.manual_seed(0)
net = MLP(input_dim=N_OBS, output_dim=N_ACT, hidden_dims=HIDDEN, activation="elu")
print(net)
linears = [m for m in net if isinstance(m, torch.nn.Linear)]
n_gates = sum(isinstance(m, torch.nn.ELU) for m in net)
check(f"rsl_rl 的 MLP：{len(HIDDEN) + 1} 张价格表（Linear）+ {len(HIDDEN)} 道门（ELU），挨着排",
      len(linears) == len(HIDDEN) + 1 and n_gates == len(HIDDEN) and len(net) == 2 * len(HIDDEN) + 1)
check("正文的网络：4 张价格表 + 3 道门，一共 7 台机器；每张表拆成“乘价格表”“加运费”两台就是 4 × 2 + 3 = 11 台",
      len(linears) == 4 and n_gates == 3 and len(net) == 7 and 2 * len(linears) + n_gates == 11)
Ws = [m.weight.detach().numpy() for m in linears]   # 把 torch 网络里的价格表和运费拷出来
bs = [m.bias.detach().numpy() for m in linears]


def forward_by_hand(obs):
    """正文 6.6 节那四行，写成循环（隐藏层改了也能用）：每张表之后过门，最后一张不过。"""
    h = obs
    for W, b in zip(Ws[:-1], bs[:-1]):
        h = elu(W @ h + b)        # 61 → 512 → 256 → 128，每次都过门
    return Ws[-1] @ h + bs[-1]    # 128 → 14，不过门


obs = torch.randn(N_OBS)
with torch.no_grad():
    out_torch = net(obs).numpy()
    again = net(obs).numpy()
out_np = forward_by_hand(obs.numpy())
print("torch 前 5 个输出:", out_torch[:5])
print("numpy 前 5 个输出:", out_np[:5])
check("numpy 手写 = rsl_rl 的 MLP（14 个输出逐个相差不到 0.00001）", float(np.max(np.abs(out_torch - out_np))) < 1e-5)
check("没有随机：同一个观测喂两次，输出逐位相同", np.array_equal(out_torch, again))
outs = []
with torch.no_grad():
    for k in range(3):
        o = torch.randn(N_OBS)
        outs.append(net(o).numpy())
        print(f"  观测 {k}: 输出前 3 项 = {outs[-1][:3]}")
check("三个不同的观测 → 三个不同的输出", not np.allclose(outs[0], outs[1]) and not np.allclose(outs[1], outs[2]))

# 一批机器人排成一张表：每一行一只，大家共用同一套旋钮（第 2 章 2.7 节的批量）。“改一改”第 3 条改这一行。
N_ROBOTS = 4  # TWEAK-3: 4096
batch = torch.randn(N_ROBOTS, N_OBS)
with torch.no_grad():
    out_batch = net(batch)
    rows_alone = torch.stack([net(batch[i]) for i in range(min(N_ROBOTS, 8))])
n_net = sum(p.numel() for p in net.parameters())
print(f"{N_ROBOTS} 只机器人排成形状 {list(batch.shape)} 的表 → 输出形状 {list(out_batch.shape)}；网络的旋钮仍是 {n_net:,} 个")
check(f"批量：[{N_ROBOTS}, 61] → [{N_ROBOTS}, 14]，每一行 = 单独喂这一行的结果",
      tuple(out_batch.shape) == (N_ROBOTS, N_ACT) and torch.allclose(out_batch[:len(rows_alone)], rows_alone, atol=1e-6))
check("正文的例子是 4 只机器人：输出形状 [4, 14]", tuple(out_batch.shape) == (4, 14))
check(f"旋钮个数不跟着机器人只数变：还是那一套 {total_mlp:,} 个", n_net == total_mlp)
X2 = np.array([[0.2, -0.1], [0.0, 0.0]])            # 迷你版：两只机器人，一行一只
mu2 = (gate(X2 @ W1.T + b1) @ W2.T + b2)[:, 0]      # 第 2 章 2.7 节：批量写成 X Wᵀ，每行加同一份运费
print(f"迷你批量：两只机器人 (0.2, −0.1)、(0, 0) → μ = ({u(mu2[0])}, {u(mu2[1])})")
check("迷你批量：两只机器人共用那 9 个旋钮，μ = (1.6068, 0.1)", np.allclose(np.round(mu2, 4), [1.6068, 0.1]))
X3 = np.vstack([X2, [0.0, 0.1]])                     # 自测：加第三只机器人 (0, 0.1)
z3 = X3 @ W1.T + b1
mu3 = (gate(z3) @ W2.T + b2)[:, 0]
print(f"加上第三只 (0, 0.1)：两家店 z = ({u(z3[2, 0])}, {u(z3[2, 1])})，μ = ({u(mu3[0])}, {u(mu3[1])}, {u(mu3[2])})")
check("自测：第三只 (0, 0.1) 的两个 z 是 (0, 0.1)，都不是负数，μ = −0.2；前两只不变",
      np.allclose(z3[2], [0.0, 0.1]) and math.isclose(float(mu3[2]), -0.2) and np.allclose(mu3[:2], mu2)
      and math.isclose(float((z3[2] @ W2.T + b2)[0]), -0.2))

# ---------------------------------------------------------------------------
banner("7. 第二张网络 critic：输入 76 个数，同样三个隐藏层，只吐 1 个数")
N_CRITIC_OBS = 76
critic = MLPModel(TensorDict({"critic": torch.zeros(1, N_CRITIC_OBS)}, batch_size=[1]), {"critic": ["critic"]}, "critic", 1,
                  HIDDEN, "elu", True)
critic_counts = count([N_CRITIC_OBS, *HIDDEN, 1])
n_critic = sum(p.numel() for p in critic.parameters() if p.requires_grad)
print("critic 各层旋钮：", [f"{c:,}" for c in critic_counts], f"合计 {n_critic:,}；最后一张表输出 {critic.mlp[-1].out_features} 个数")
check("critic 比 actor 多看 76 − 61 = 15 项", N_CRITIC_OBS - N_OBS == 15)
check("critic 最后一张表只有 1 家店：只吐 1 个数；它没有 σ", critic.mlp[-1].out_features == 1 and critic.distribution is None)
check("critic 第一层 512 × 77 = 39,424；最后一层 1 × 129 = 129", critic_counts[0] == 512 * 77 == 39424 and critic_counts[-1] == 129)
check("自测：critic 一共 39,424 + 131,328 + 32,896 + 129 = 203,777 个旋钮",
      n_critic == sum(critic_counts) == 203777 and 39424 + 131328 + 32896 + 129 == 203777)

# ---------------------------------------------------------------------------
banner("8. 映射到项目：正文引用的配置和源码行还在不在")
REPO = Path(__file__).resolve().parents[3]
cfg_path = REPO / "src" / "mjlab_microduck" / "tasks" / "microduck_velocity_env_cfg.py"
CFG_LINES = ["actor=RslRlModelCfg(", "hidden_dims=(512, 256, 128),", 'activation="elu",', "obs_normalization=True,",
             "distribution_cfg={", '"class_name": "GaussianDistribution",', '"init_std": 1.0,', '"std_type": "scalar",',
             "critic=RslRlModelCfg(", "hidden_dims=(512, 256, 128),", 'activation="elu",']
if cfg_path.is_file():
    cfg_text = cfg_path.read_text(encoding="utf-8")
    at = cfg_text.find("MicroduckRlCfg = RslRlOnPolicyRunnerCfg(")
    print("microduck_velocity_env_cfg.py 的 MicroduckRlCfg：actor / critic 都是 hidden_dims=(512, 256, 128)、activation=\"elu\"")
    check(f"项目配置：正文引用的 {len(CFG_LINES)} 行原样存在、顺序一致（两张网络的隐藏层、门、σ 的设置）",
          at >= 0 and lines_in_order(cfg_text[at:], CFG_LINES))
else:
    print("  （没找到项目的 env cfg，跳过这一项）")

MLP_LINES = ["last_activation: str | None = None,", "activation_mod = resolve_nn_activation(activation)",
             "layers.append(nn.Linear(input_dim, hidden_dims_processed[0]))", "layers.append(activation_mod)",
             "for layer_index in range(len(hidden_dims_processed) - 1):",
             "layers.append(nn.Linear(hidden_dims_processed[layer_index], hidden_dims_processed[layer_index + 1]))",
             "layers.append(activation_mod)", "if isinstance(output_dim, int):",
             "layers.append(nn.Linear(hidden_dims_processed[-1], output_dim))", "for layer in self:", "x = layer(x)"]
rsl_spec = importlib.util.find_spec("rsl_rl")
rsl_dir = Path(rsl_spec.origin).parent if rsl_spec and rsl_spec.origin else None
if rsl_dir and (rsl_dir / "modules" / "mlp.py").is_file():
    mlp_text = (rsl_dir / "modules" / "mlp.py").read_text(encoding="utf-8")
    print("rsl_rl/modules/mlp.py 的 MLP：第 1 张表 + 门 → 循环加中间的表 + 门 → 最后一张表（不接门）→ forward 依次过")
    check(f"正文映射块引用的 MLP 源码 {len(MLP_LINES)} 行都原样存在，且顺序一致", lines_in_order(mlp_text, MLP_LINES))
    others = {"utils/utils.py": '"elu": torch.nn.ELU(),',
              "models/mlp_model.py": "self.mlp = MLP(self._get_latent_dim(), mlp_output_dim, hidden_dims, activation)",
              "modules/distribution.py": "self.std_param = nn.Parameter(init_std * torch.ones(output_dim))"}
    for rel, line in others.items():
        path = rsl_dir / rel
        check(f"rsl_rl/{rel} 里还有：{line}", path.is_file() and line in path.read_text(encoding="utf-8"))
else:
    print("  （当前 Python 环境里没有 rsl_rl，跳过源码核对；用 uv run 运行就会核对）")
ch15 = REPO / "docs" / "learn-zh" / "labs" / "ch15_print_obs.py"
if ch15.is_file():
    check("critic 的 76 维由第 15 章的 GPU 实验建环境核对（那条检查还在）",
          'obs["critic"].shape[1] == 76' in ch15.read_text(encoding="utf-8"))

done()
