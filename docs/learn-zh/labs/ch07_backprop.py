"""第 7 章实验：计算图、反向传播、优化器（SGD）、滑动平均、Adam、mini-batch 与 epoch、一次完整的更新。

运行：uv run python docs/learn-zh/labs/ch07_backprop.py
纯 CPU，numpy + torch + matplotlib。第 8 节只读几份源码的文字，不加载机器人、不训练。
小节编号与正文一一对应：实验第 K 节 = 正文 7.K 节（第 8 节对应「映射到项目」）。
正文“改一改”要改的三行都带 `# TWEAK-k:` 标记（第 3、4、7 节各一处）。
"""

import importlib.util
import inspect
import math
from pathlib import Path

import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch

from _common import banner, check, done, lines_in_order, num, savefig, table
from _draw import (BLUE, CELL, CELL_HOT, FAINT, FS_NOTE, FS_SMALL, FS_STEP, FS_TITLE, GREEN, INK, MUTED, ORANGE, WHITE_BOX,
                   arrow, cell, data_axes, hand, lesson_panel, note, panel_note, panel_title, plt)

np.seterr(over="ignore", invalid="ignore")   # 发散时数会大得离谱——这正是要给读者看的现象，不用 numpy 再警告
F64 = torch.float64                           # 手算对照一律用 64 位小数：发散的那一行也不会先溢出成 inf


def fmt(v, places=2) -> str:
    """图里的数：最多 places 位小数、去尾零、用真正的减号。"""
    return num(v, places).replace("-", "−")


# ---------------------------------------------------------------------------
banner("1. 计算图：一行算式拆成三台小机器，数从左往右流（前向）")
X1, Y1 = 2.0, 5.0      # 例一的数据：输入 x、目标 y（记号同第 3 章 3.5 节：真值 y、预测 ŷ、误差 e = ŷ − y）
W0, B0 = 1.0, 0.0      # 两个旋钮现在的位置


def forward1(w, b, x=X1, y=Y1):
    """例一的三台机器：① ŷ = w·x + b　② e = ŷ − y　③ L = e²。"""
    y_hat = w * x + b
    e = y_hat - y
    return y_hat, e, e * e


yh1, e1, L1 = forward1(W0, B0)
print(f"一口气算：L = (w·x + b − y)² = ({W0:g}×{X1:g} + {B0:g} − {Y1:g})² = {(W0 * X1 + B0 - Y1) ** 2:g}")
print(f"拆成三台机器：ŷ = {W0:g}×{X1:g} + {B0:g} = {yh1:g} → e = {yh1:g} − {Y1:g} = {e1:g} → L = ({e1:g})² = {L1:g}")
check("例一前向：ŷ = 2，e = −3，L = 9，和一口气算的 (1×2 + 0 − 5)² 相同",
      (yh1, e1, L1) == (2, -3, 9) and (W0 * X1 + B0 - Y1) ** 2 == L1)
check("7.1 自测：b 改成 1 → ŷ = 3，e = −2，L = 4", forward1(1.0, 1.0) == (3, -2, 4))

X2, Y2, W1, W2 = 1.5, 2.0, 0.8, -0.5   # 例二：两层小网络（旧稿的数字原样保留）


def elu(z):
    """第 6 章的 ELU：正数原样过；负数压成 exp(z) − 1。"""
    return z if z > 0 else math.exp(z) - 1


def forward2(w1, w2, x=X2, y=Y2):
    """例二的五台机器：z = w1·x → h = ELU(z) → ŷ = w2·h → e = ŷ − y → L = e²。"""
    z = w1 * x
    h = elu(z)
    y_hat = w2 * h
    e = y_hat - y
    return z, h, y_hat, e, e * e


z2, h2, yh2, e2, L2 = forward2(W1, W2)
print(f"例二：z = {W1:g}×{X2:g} = {z2:.1f} → h = ELU({z2:.1f}) = {h2:.1f} → ŷ = {W2:g}×{h2:.1f} = {yh2:.1f}"
      f" → e = {yh2:.1f} − {Y2:g} = {e2:.1f} → L = ({e2:.1f})² = {L2:.2f}")
check("例二前向：z = 1.2，h = 1.2（正数原样过），ŷ = −0.6，e = −2.6，L = 6.76",
      all(math.isclose(a, b) for a, b in zip((z2, h2, yh2, e2, L2), (1.2, 1.2, -0.6, -2.6, 6.76))))

# --- 画流水线用的小工具（本章独有，所以写在这里） -------------------------------------------
UNIT, XMAX = 0.74, 17.0                   # 图内 1 个单位 ≈ 0.74 英寸，横竖一致，圆角框才不变形
CW, MW, GAP, BH = 1.15, 1.4, 0.24, 0.8    # 数的方框宽、机器的圆角框宽、箭头长、框高


def lesson_stack(heights, title):
    """竖着排几块高矮不同的分步面板（_draw.lesson_figure 的变体，每块高度用图内单位给）。"""
    fig_h = sum(heights) * UNIT + 0.8
    fig = plt.figure(figsize=(XMAX * UNIT, fig_h))
    gs = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.0,
                          top=1 - 0.8 / fig_h, bottom=0.0, left=0.0, right=1.0)
    axes = [fig.add_subplot(gs[i]) for i in range(len(heights))]
    fig.suptitle(title, fontsize=FS_TITLE, fontweight="bold", color=INK, y=1 - 0.15 / fig_h, va="top")
    return fig, axes


def machine_box(ax, x, y, text, color=GREEN):
    ax.add_patch(FancyBboxPatch((x, y), MW, BH, boxstyle="round,pad=0,rounding_size=0.25",
                                facecolor="white", edgecolor=color, lw=2.2, zorder=3))
    ax.text(x + MW / 2, y + BH / 2, text, ha="center", va="center", fontsize=FS_STEP if len(text) <= 4 else FS_SMALL,
            color=color, zorder=4)


def pipeline(ax, nums, machines, y, feeds=(), x0=0.25, faint=False):
    """一行流水线：数（方框，上方写名字）和机器（圆角框）交替。
    feeds = [(第几台机器, 文字, 宽度, 是不是旋钮)]：画在那台机器正上方、用箭头喂进去（旋钮蓝框，数据灰框）。
    返回 (每个数方框的中心 x, 每台机器的中心 x)。"""
    cx_num, cx_mac, x = [], [], x0
    for i, (name, val) in enumerate(nums):
        cell(ax, x, y, val, width=CW, height=BH, color=MUTED if faint else INK)
        ax.text(x + CW / 2, y + BH + 0.08, name, ha="center", va="bottom", fontsize=FS_SMALL, color=MUTED)
        cx_num.append(x + CW / 2)
        x += CW
        if i < len(machines):
            arrow(ax, (x, y + BH / 2), (x + GAP, y + BH / 2), FAINT, lw=2)
            machine_box(ax, x + GAP, y, machines[i], color=FAINT if faint else GREEN)
            cx_mac.append(x + GAP + MW / 2)
            x += GAP + MW
            arrow(ax, (x, y + BH / 2), (x + GAP, y + BH / 2), FAINT, lw=2)
            x += GAP
    for k, text, width, knob in feeds:
        color = BLUE if knob else MUTED
        cell(ax, cx_mac[k] - width / 2, y + BH + 0.62, text, width=width, height=0.6, facecolor="white",
             edgecolor=color, fontsize=FS_SMALL, color=color)
        arrow(ax, (cx_mac[k], y + BH + 0.62), (cx_mac[k], y + BH), color, lw=1.6)
    return cx_num, cx_mac


def rates_under(ax, cx_mac, y, rates, color=GREEN):
    """每台机器下面写它的本地汇率（绿字，可以两行）。"""
    for cx, lines in zip(cx_mac, rates):
        for j, text in enumerate(lines):
            ax.text(cx, y - 0.32 - 0.4 * j, text, ha="center", va="center", fontsize=FS_SMALL, color=color)


def grads_under(ax, cx_num, y_row, grads):
    """每个数方框正下方一个橙色格子：L 对这个数的倍数；格子之间画从右往左的橙箭头（反向）。"""
    shown = [(cx, g) for cx, g in zip(cx_num, grads) if g is not None]
    for cx, g in shown:
        cell(ax, cx - CW / 2, y_row, g, width=CW, height=0.62, facecolor=CELL_HOT, edgecolor=ORANGE, color=ORANGE)
    for (cx_a, _), (cx_b, _) in zip(shown[:-1], shown[1:]):
        arrow(ax, (cx_b - CW / 2, y_row + 0.31), (cx_a + CW / 2, y_row + 0.31), ORANGE, lw=2.2)


EX1_NUMS = [("x", fmt(X1)), ("ŷ", fmt(yh1)), ("e", fmt(e1)), ("L", fmt(L1))]
EX1_MACS = ["× w + b", "− y", "平方"]
EX1_FEEDS = [(0, f"w = {fmt(W0)}，b = {fmt(B0)}", 2.9, True), (1, f"y = {fmt(Y1)}", 1.3, False)]
EX2_NUMS = [("x", fmt(X2)), ("z", fmt(z2)), ("h", fmt(h2)), ("ŷ", fmt(yh2)), ("e", fmt(e2)), ("L", fmt(L2))]
EX2_MACS = ["× w₁", "ELU", "× w₂", "− y", "平方"]
EX2_FEEDS = [(0, f"w₁ = {fmt(W1)}", 1.6, True), (2, f"w₂ = {fmt(W2)}", 1.75, True), (3, f"y = {fmt(Y2)}", 1.3, False)]

# ---------------------------------------------------------------------------
banner("1b. 画图：figures/ch07_graph_forward.png（一行算式 → 三台机器 → 两层网络）")
fig, axes = lesson_stack([2.5, 4.3, 4.3], "计算图：一行算式拆成一台台小机器，数从左往右流")
lesson_panel(axes[0], "① 一行算式：L = (w·x + b − y)²", xmax=XMAX, ymax=2.5)
hand(axes[0], 0.3, 1.25, f"x = {fmt(X1)}，y = {fmt(Y1)}，w = {fmt(W0)}，b = {fmt(B0)}："
     f"({fmt(W0)} × {fmt(X1)} + {fmt(B0)} − {fmt(Y1)})² = ({fmt(e1)})² = {fmt(L1)}")
note(axes[0], 0.3, 0.45, "一口气就算完了，可里面其实藏着三步：乘再加、减、平方。")
lesson_panel(axes[1], "② 拆成三台机器：前一台的输出，就是后一台的输入", xmax=XMAX, ymax=4.3)
pipeline(axes[1], EX1_NUMS, EX1_MACS, y=1.05, feeds=EX1_FEEDS)
note(axes[1], 0.3, 0.4, "方框是数，圆角框是机器；蓝框是旋钮，灰框是数据，都从上面喂进机器。")
lesson_panel(axes[2], "③ 两层小网络：同样的机器，多串几台", xmax=XMAX, ymax=4.3)
pipeline(axes[2], EX2_NUMS, EX2_MACS, y=1.05, feeds=EX2_FEEDS)
note(axes[2], 0.3, 0.4, "ELU（第 6 章）：正数原样通过。第 6 章的大网络也是这样串起来的，只是每台机器一次处理很多个数。")
savefig(fig, "ch07_graph_forward")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("2. 反向传播：从 L 出发往回走，每过一台机器乘一次它的本地汇率")
rate_sq, rate_minus, rate_w, rate_b = 2 * e1, 1.0, X1, 1.0   # 平方：2e；减 y：1；乘加：对 w 是 x，对 b 是 1
g_e = 1.0 * rate_sq            # 从 L 出发：L 对自己的倍数是 1
g_yh = g_e * rate_minus
g_w, g_b = g_yh * rate_w, g_yh * rate_b
table(["走到", "乘上这台机器的本地汇率", "L 对它的倍数"],
      [["e", f"平方：2e = {e1 * 2:g}", g_e], ["ŷ", "减 y：1", g_yh], ["w", f"乘加：对 w 是 x = {X1:g}", g_w],
       ["b", "乘加：对 b 是 1", g_b]], floatfmt="g")
check("例一：∂L/∂w = (−6) × 1 × 2 = −12，∂L/∂b = (−6) × 1 × 1 = −6",
      (g_e, g_yh, g_w, g_b) == (-6, -6, -12, -6) and (-6) * 1 * 2 == -12 and (-6) * 1 * 1 == -6)
check("就是第 3 章 3.5 节表里那两列：2·e·x = −12，2·e = −6", 2 * e1 * X1 == g_w and 2 * e1 == g_b)
rows = []
for dw in (0.01, 0.001):
    Lw, Lb = forward1(W0 + dw, B0)[2], forward1(W0, B0 + dw)[2]
    rows.append([dw, Lw, (Lw - L1) / dw, Lb, (Lb - L1) / dw])
table(["拧多少", "w 拧后的 L", "变化率", "b 拧后的 L", "变化率"], rows, floatfmt=".6g")
check("缩步核对：w 拧 0.01 → L = 8.8804，(8.8804 − 9)/0.01 = −11.96；b 拧 0.01 → L = 8.9401，变化率 −5.99",
      round(rows[0][1], 4) == 8.8804 and round((8.8804 - 9) / 0.01, 2) == -11.96
      and round(rows[0][3], 4) == 8.9401 and round((8.9401 - 9) / 0.01, 2) == -5.99)
check("步子缩到 0.001：变化率 −11.996、−5.999，更靠近 −12、−6",
      round(rows[1][2], 3) == -11.996 and round(rows[1][4], 3) == -5.999)
w_t = torch.tensor(W0, dtype=F64, requires_grad=True)
b_t = torch.tensor(B0, dtype=F64, requires_grad=True)
((w_t * X1 + b_t - Y1) ** 2).backward()
check("autograd（torch 的自动求导）：w.grad = −12，b.grad = −6，和手算一样", (w_t.grad.item(), b_t.grad.item()) == (-12, -6))

# 例二：五台机器
r2_sq, r2_minus, r2_h, r2_w2, r2_elu, r2_w1 = 2 * e2, 1.0, W2, h2, (1.0 if z2 > 0 else math.exp(z2)), X2
g2_e = r2_sq
g2_yh = g2_e * r2_minus
g2_w2 = g2_yh * r2_w2            # 岔路：走到旋钮 w2
g2_h = g2_yh * r2_h              # 主路：继续往回走到 h
g2_z = g2_h * r2_elu
g2_w1 = g2_z * r2_w1
w1_t = torch.tensor(W1, dtype=F64, requires_grad=True)
w2_t = torch.tensor(W2, dtype=F64, requires_grad=True)
((w2_t * torch.nn.functional.elu(w1_t * X2) - Y2) ** 2).backward()
table(["", "手算链式法则", "autograd"], [["∂L/∂w2", g2_w2, w2_t.grad.item()], ["∂L/∂w1", g2_w1, w1_t.grad.item()]],
      floatfmt=".4g")
check("例二：∂L/∂w₂ = (−5.2) × 1 × 1.2 = −6.24；∂L/∂w₁ = (−5.2) × 1 × (−0.5) × 1 × 1.5 = 3.9",
      math.isclose(g2_w2, -6.24) and math.isclose(g2_w1, 3.9) and math.isclose(-5.2 * 1 * 1.2, -6.24)
      and math.isclose(-5.2 * 1 * (-0.5) * 1 * 1.5, 3.9))
check("例二：手算 = autograd", math.isclose(g2_w2, w2_t.grad.item()) and math.isclose(g2_w1, w1_t.grad.item()))
check("路上的倍数：L 对 h 是 (−5.2) × (−0.5) = 2.6，L 对 z 也是 2.6（ELU 在正数处的汇率是 1）",
      math.isclose(g2_h, 2.6) and math.isclose(g2_z, 2.6))
zn = torch.tensor(-1.2, dtype=F64, requires_grad=True)
torch.nn.functional.elu(zn).backward()
check("ELU 在负数处的汇率：z = −1.2 时是 exp(−1.2) = 0.3012", round(zn.grad.item(), 4) == 0.3012 == round(math.exp(-1.2), 4))
XA, XB = 2.0, 3.0                                   # 同一个 w 用在两条路上：ŷ = w·xa + w·xb
w_t = torch.tensor(1.0, dtype=F64, requires_grad=True)
(w_t * XA + w_t * XB).backward()
rate_two = ((1.01 * XA + 1.01 * XB) - (XA + XB)) / 0.01
check("两条路各贡献一份再相加：∂ŷ/∂w = 2 + 3 = 5（缩步量得 5，autograd 也是 5）",
      w_t.grad.item() == 5 and round(rate_two, 6) == 5)
check("7.2 自测：b = 1 时 e = −2，∂L/∂w = (−4) × 1 × 2 = −8，∂L/∂b = −4",
      2 * forward1(1.0, 1.0)[1] * X1 == -8 and 2 * forward1(1.0, 1.0)[1] == -4)

# ---------------------------------------------------------------------------
banner("2b. 画图：figures/ch07_graph_backward.png（本地汇率 → 往回乘 → 两层网络）")
fig, axes = lesson_stack([5.0, 5.6, 5.6], "反向传播：从 L 出发往回走，每过一台机器乘一次它的本地汇率")
EX1_RATES = [[f"对 w：x = {fmt(rate_w)}", f"对 b：{fmt(rate_b)}"], [fmt(rate_minus)], [f"2e = {fmt(rate_sq)}"]]
lesson_panel(axes[0], "① 每台机器只看自己：输入动一点，输出动几倍（本地汇率）", xmax=XMAX, ymax=5.0)
_, cm = pipeline(axes[0], EX1_NUMS, EX1_MACS, y=1.9, feeds=EX1_FEEDS)
rates_under(axes[0], cm, 1.9, EX1_RATES)
note(axes[0], 0.3, 0.45, "绿字是本地汇率：只用这台机器自己的输入，前向时就记下了。平方机器的输入是 e = −3，所以是 2e = −6。")
lesson_panel(axes[1], "② 从 L 往回乘：走到哪个数，就得到 L 对它的倍数", xmax=XMAX, ymax=5.6)
cn, cm = pipeline(axes[1], EX1_NUMS, EX1_MACS, y=2.5, feeds=EX1_FEEDS, faint=True)
rates_under(axes[1], cm, 2.5, EX1_RATES)
grads_under(axes[1], cn, 0.95, [None, fmt(g_yh), fmt(g_e), "1"])
hand(axes[1], 0.3, 0.42, f"∂L/∂w = ({fmt(g_yh)}) × {fmt(rate_w)} = {fmt(g_w)}；∂L/∂b = ({fmt(g_yh)}) × {fmt(rate_b)} = {fmt(g_b)}",
     color=ORANGE)
lesson_panel(axes[2], "③ 两层网络一样乘：最里面的 w₁ 要穿过五台机器", xmax=XMAX, ymax=5.6)
cn, cm = pipeline(axes[2], EX2_NUMS, EX2_MACS, y=2.5, feeds=EX2_FEEDS, faint=True)
rates_under(axes[2], cm, 2.5, [[f"对 w₁：x = {fmt(r2_w1)}"], [fmt(r2_elu)], [f"对 h：w₂ = {fmt(r2_h)}", f"对 w₂：h = {fmt(r2_w2)}"],
                               [fmt(r2_minus)], [f"2e = {fmt(r2_sq)}"]])
grads_under(axes[2], cn, 0.95, [None, fmt(g2_z), fmt(g2_h), fmt(g2_yh), fmt(g2_e), "1"])
hand(axes[2], 0.3, 0.42, f"∂L/∂w₂ = ({fmt(g2_yh)}) × {fmt(r2_w2)} = {fmt(g2_w2)}；∂L/∂w₁ = {fmt(g2_z)} × {fmt(r2_w1)} = {fmt(g2_w1)}",
     color=ORANGE)
savefig(fig, "ch07_graph_backward")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("3. 优化器：backward() 只把梯度写进 .grad，step() 才真正拧旋钮")
ALPHA_SGD = 0.01
w_new, b_new = W0 - ALPHA_SGD * g_w, B0 - ALPHA_SGD * g_b
yh_new, e_new, L_new = forward1(w_new, b_new)
print(f"手算 SGD 一步（α = {ALPHA_SGD}）：w = 1 − 0.01 × (−12) = {w_new:.2f}；b = 0 − 0.01 × (−6) = {b_new:.2f}")
print(f"新的前向：ŷ = {w_new:.2f} × 2 + {b_new:.2f} = {yh_new:.2f}，e = {e_new:.2f}，L = {L_new:.2f}")
w_t = torch.tensor(W0, dtype=F64, requires_grad=True)
b_t = torch.tensor(B0, dtype=F64, requires_grad=True)
opt = torch.optim.SGD([w_t, b_t], lr=ALPHA_SGD)
opt.zero_grad()
((w_t * X1 + b_t - Y1) ** 2).backward()
w_before_step, grad_seen = w_t.item(), w_t.grad.item()
opt.step()
print(f"torch：backward() 之后 w = {w_before_step:g}、w.grad = {grad_seen:g}；step() 之后 w = {w_t.item():.2f}、b = {b_t.item():.2f}")
check("backward() 之后 w 还是 1（只写了 .grad = −12）；step() 之后才变成 1.12", w_before_step == 1 and grad_seen == -12)
check("SGD 一步：w = 1 − 0.01 × (−12) = 1.12，b = 0 − 0.01 × (−6) = 0.06；torch.optim.SGD 同样",
      round(w_new, 6) == 1.12 and round(b_new, 6) == 0.06 and math.isclose(w_t.item(), 1.12) and math.isclose(b_t.item(), 0.06))
check("拧完再算一遍：ŷ = 1.12 × 2 + 0.06 = 2.30，e = −2.70，L = (−2.70)² = 7.29，比 9 小",
      round(yh_new, 6) == 2.3 and round(e_new, 6) == -2.7 and round(L_new, 6) == 7.29 == round(round(1.12 * 2 + 0.06 - 5, 2) ** 2, 6))
w_t = torch.tensor(W0, dtype=F64, requires_grad=True)
b_t = torch.tensor(B0, dtype=F64, requires_grad=True)
for _ in range(2):                                   # 不清零，连着 backward 两次
    ((w_t * X1 + b_t - Y1) ** 2).backward()
check("连着做两次“前向 + backward()”不清零：.grad 累加成 −24、−12（不是覆盖成 −12、−6）", (w_t.grad.item(), b_t.grad.item()) == (-24, -12))
loss_once = (w_t * X1 + b_t - Y1) ** 2
loss_once.backward()
try:                                                 # 同一个 loss 再往回走一次：前向记下的数已经扔了
    loss_once.backward()
    backward_twice_fails = False
except RuntimeError:
    backward_twice_fails = True
check("同一个 loss 不能 backward() 两次：往回走完一趟，前向记下的数就扔了，再来会报错（所以要连前向一起重做）", backward_twice_fails)
wq, bq = W0 - 0.05 * g_w, B0 - 0.05 * g_b
check("7.3 自测：α = 0.05 → w = 1.6，b = 0.3；ŷ = 3.5，e = −1.5，L = 2.25",
      (round(wq, 6), round(bq, 6)) == (1.6, 0.3) and tuple(round(v, 6) for v in forward1(wq, bq)) == (3.5, -1.5, 2.25))

# 扁碗 f = x² + 100y²：y 方向陡 100 倍（第 3 章 3.6 节的碗是 x² + 3y²）。从 (3, 2) 出发，SGD 走 100 步。
X_START, Y_START, STEEP = 3.0, 2.0, 100.0
ALPHA_FAST = 0.1
ALPHA_SLOW = 0.005  # TWEAK-1: 0.011


def bowl(x, y):
    return x ** 2 + STEEP * y ** 2


rows = []
for a in (ALPHA_FAST, ALPHA_SLOW):
    mx, my = 1 - 2 * a, 1 - 2 * STEEP * a          # 3.6 节的规则：每走一步，离碗底的距离各乘一个数
    x100, y100 = X_START * mx ** 100, Y_START * my ** 100
    rows.append([a, fmt(mx, 4), fmt(my, 4), f"{x100:.4g}", f"{y100:.4g}", f"{bowl(x100, y100):.4g}"])
table(["学习率 α", "x 每步乘", "y 每步乘", "100 步后 x", "100 步后 y", "100 步后高度"], rows)
print("门槛（3.6 节）：y 方向要 |1 − 200α| < 1，也就是 α < 0.01；x 方向要 |1 − 2α| < 1，α < 1。最陡的方向说了算。")
check("α = 0.1：y 每步乘 1 − 200 × 0.1 = −19，发散", 1 - 2 * STEEP * ALPHA_FAST == -19 and abs(float(rows[0][4])) > 1e100)
check("α = 0.005：y 每步乘 0（一步到底）；x 每步乘 0.99，0.99¹⁰⁰ = 0.366，3 × 0.366 = 1.098，高度 1.098² = 1.206",
      ALPHA_SLOW == 0.005 and round(0.99 ** 100, 3) == 0.366 and round(3 * 0.366, 3) == 1.098 == round(float(rows[1][3]), 3)
      and round(1.098 ** 2, 3) == 1.206 == round(float(rows[1][5]), 3))

# ---------------------------------------------------------------------------
banner("4. 滑动平均：今天只占一成，旧的平均占九成")
TEMPS = [26, 34, 28, 32, 30, 27, 33, 29, 31, 30]    # 教学构造的 10 天气温（°C），10 天平均正好 30
BETA = 0.9  # TWEAK-2: 0.5


def moving_average(values, beta, start):
    out, avg = [], start
    for v in values:
        avg = beta * avg + (1 - beta) * v              # 新平均 = β × 旧平均 + (1 − β) × 今天
        out.append(avg)
    return out


avg30 = moving_average(TEMPS, BETA, 30.0)
avg0 = moving_average(TEMPS, BETA, 0.0)
divisors = [1 - BETA ** (t + 1) for t in range(len(TEMPS))]
fixed = [a / d for a, d in zip(avg0, divisors)]
table(["第几天", "气温", "从 30 起步的平均", "从 0 起步的平均", "除以的数", "起步修正后"],
      [[t + 1, TEMPS[t], f"{avg30[t]:.2f}", f"{avg0[t]:.2f}", f"{divisors[t]:.3f}", f"{fixed[t]:.2f}"] for t in range(len(TEMPS))])
print(f"气温在 {min(TEMPS)}–{max(TEMPS)} 度之间跳；从 30 起步的平均只在 {min(avg30):.2f}–{max(avg30):.2f} 之间动。")
check("第 1 天：0.9 × 30 + 0.1 × 26 = 29.6（今天冷了 4 度，平均只降 0.4 度）",
      round(avg30[0], 6) == 29.6 == round(0.9 * 30 + 0.1 * 26, 6))
check("第 2 天：0.9 × 29.6 + 0.1 × 34 = 30.04；换个读法：29.6 + 0.1 × (34 − 29.6) = 29.6 + 0.44 = 30.04",
      round(avg30[1], 6) == 30.04 == round(0.9 * 29.6 + 0.1 * 34, 6) == round(29.6 + 0.1 * (34 - 29.6), 6) and round(0.1 * (34 - 29.6), 6) == 0.44)
check("气温跳了 8 度（26 到 34），平均一直在 29.6 到 30.07 之间",
      (min(TEMPS), max(TEMPS)) == (26, 34) and round(min(avg30), 2) == 29.6 and round(max(avg30), 2) == 30.07)
check("从 0 起步：第 1 天 0.9 × 0 + 0.1 × 26 = 2.6；除以 1 − 0.9 = 0.1，修正回 26",
      round(avg0[0], 6) == 2.6 and round(fixed[0], 6) == 26 and round(2.6 / 0.1, 6) == 26)
check("折叠：第 2 天 0.9 × 2.6 + 0.1 × 34 = 5.74；除以 1 − 0.9² = 0.19，得 5.74 ÷ 0.19 = 30.21",
      round(avg0[1], 6) == 5.74 and round(divisors[1], 6) == 0.19 and round(fixed[1], 2) == 30.21 == round(5.74 / 0.19, 2))
check("折叠：按份额求加权平均 (0.09 × 26 + 0.1 × 34) ÷ 0.19 = (2.34 + 3.4) ÷ 0.19 = 30.21，和 5.74 ÷ 0.19 一样",
      round(0.9 * 0.1, 6) == 0.09 and round(0.09 * 26, 6) == 2.34 and round((2.34 + 3.4) / 0.19, 2) == 30.21
      and round((0.09 * TEMPS[0] + 0.1 * TEMPS[1]) / 0.19, 2) == 30.21)
check("第 10 天：不修正的平均 19.60，还没爬到 20 度", round(avg0[9], 2) == 19.6 < 20)
check("正文的 JS reduce：从 30 起步，第 10 天的滑动平均约为 30.06", round(avg30[9], 2) == 30.06)
check("折叠：第 2 天来自数据的份额 0.9 × 0.1 + 0.1 = 0.19 = 1 − 0.9²，假的 0 占 0.81 = 0.9²；除数第 3 天 0.271，第 10 天 0.651",
      round(0.9 * 0.1 + 0.1, 6) == 0.19 == round(1 - 0.9 ** 2, 6) and round(0.9 ** 2, 6) == 0.81
      and [round(divisors[k], 3) for k in (0, 1, 2, 9)] == [0.1, 0.19, 0.271, 0.651])
check("7.4 自测：0.9 × 20 + 0.1 × 10 = 19；从 0 起步 0.1 × 10 = 1，除以 0.1 得 10",
      round(0.9 * 20 + 0.1 * 10, 6) == 19 and round(moving_average([10], 0.9, 0.0)[0], 6) == 1 and round(1 / 0.1, 6) == 10)
theory = [sum(BETA ** (t - i) * TEMPS[i] for i in range(t + 1)) / sum(BETA ** (t - i) for i in range(t + 1))
          for t in range(len(TEMPS))]
check(f"修正后 = 前 t 天的加权平均（越近权重越大，每往前一天权重乘 {BETA:g}）——换什么气温都成立",
      all(math.isclose(a, b) for a, b in zip(fixed, theory)))
check("记性：最近 10 步占总权重 1 − 0.9¹⁰ = 0.651；β = 0.999 时最近 1000 步占 1 − 0.999¹⁰⁰⁰ = 0.632",
      round(1 - 0.9 ** 10, 3) == 0.651 and round(1 - 0.999 ** 1000, 3) == 0.632)

# ---------------------------------------------------------------------------
banner("4b. 画图：figures/ch07_moving_average.png（又稳又慢；从 0 起步要修正）")
days = np.arange(0, len(TEMPS) + 1)
fig = plt.figure(figsize=(10, 11.6))
default_beta = BETA == 0.9
fig.suptitle("滑动平均：今天只占一成，平均线又稳又慢" if default_beta else f"滑动平均（β 改成了 {BETA:g}）：对照上面的表看",
             fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
ax1 = fig.add_axes([0.11, 0.575, 0.85, 0.30])
ax2 = fig.add_axes([0.11, 0.10, 0.85, 0.30])
for ax in (ax1, ax2):
    data_axes(ax, "第几天", "气温（°C）")
    ax.set_xlim(-0.3, len(TEMPS) + 0.3)
    ax.set_ylim(0, 38)
    ax.set_xticks(range(0, len(TEMPS) + 1))
    ax.set_yticks(range(0, 40, 10))
    ax.plot(days[1:], TEMPS, "o", color=BLUE, ms=9, zorder=5, label="每天的气温")
ax1.plot(days, [30.0] + avg30, "-", color=ORANGE, lw=3.2, label="滑动平均（从 30 起步）")
ax1.annotate(f"{BETA:g} × 30 + {1 - BETA:g} × {TEMPS[0]} = {fmt(avg30[0])}", xy=(1, avg30[0]), xytext=(1.6, 14),
             fontsize=FS_SMALL, color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6), bbox=WHITE_BOX)
ax1.legend(loc="lower right", fontsize=FS_SMALL, frameon=False)
panel_title(fig, [ax1], "① 从 30 起步：气温在 26 到 34 之间跳，平均线几乎不动" if default_beta
            else f"① 从 30 起步：平均线在 {min(avg30):.1f} 到 {max(avg30):.1f} 之间动")
panel_note(fig, [ax1], f"第 1 天比平均冷 4 度，平均{'只降' if default_beta else '降了'} {fmt(30 - avg30[0])} 度" + ("——一天的怪数据带不偏它。" if default_beta else "。"))
ax2.plot(days, [0.0] + avg0, "--", color=MUTED, lw=2.6, label="从 0 起步，不修正")
ax2.plot(days[1:], fixed, "-", color=ORANGE, lw=3.2, label="起步修正后")
ax2.annotate(f"{fmt(avg0[0])} ÷ {fmt(divisors[0], 3)} = {fmt(fixed[0])}", xy=(1, avg0[0]), xytext=(1.3, 17),
             fontsize=FS_SMALL, color=MUTED, arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.6), bbox=WHITE_BOX)
ax2.legend(loc="lower right", fontsize=FS_SMALL, frameon=False)
panel_title(fig, [ax2], "② 从 0 起步：不存在的旧平均被当成 0 度，头几天严重偏低")
panel_note(fig, [ax2], ("灰虚线第 10 天还没爬到 20 度。" if avg0[9] < 20 else f"灰虚线第 10 天是 {avg0[9]:.1f} 度。")
           + f"每天除以一个小于 1 的数（第 1 天除以 {fmt(divisors[0], 3)}）放大回来，\n橙线从第 1 天起就在 30 度上下。")
savefig(fig, "ch07_moving_average")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("5. Adam：两本滑动平均的账，每个旋钮自己定步长")
B1, B2, EPS = 0.9, 0.999, 1e-8


def adam_first_step(g):
    """Adam 第一步（两本账都从 0 起步）。返回 (m, v, 修正后的 m, 修正后的 v, 这一步是 α 的几倍)。"""
    m = B1 * 0 + (1 - B1) * g
    v = B2 * 0 + (1 - B2) * g * g
    m_hat, v_hat = m / (1 - B1), v / (1 - B2)
    return m, v, m_hat, v_hat, m_hat / (math.sqrt(v_hat) + EPS)


m, v, m_hat, v_hat, k = adam_first_step(2.0)
print(f"g = 2：m = 0.9 × 0 + 0.1 × 2 = {m:.1f}；v = 0.999 × 0 + 0.001 × 2² = {v:.3f}")
print(f"起步修正：{m:.1f} / (1 − 0.9) = {m_hat:g}；{v:.3f} / (1 − 0.999) = {v_hat:g}；这一步 = α × {m_hat:g} / √{v_hat:g} = {k:.6f} α")
check("冻结手算（worked_examples.py）：m = 0.2，v = 0.004；0.2 / (1 − 0.9) = 2；0.004 / (1 − 0.999) = 4",
      round(m, 9) == 0.2 and round(v, 9) == 0.004 and math.isclose(m_hat, 2) and math.isclose(v_hat, 4)
      and math.isclose(0.2 / (1 - 0.9), 2) and math.isclose(0.004 / (1 - 0.999), 4))
check("这一步 = α × 2 / √4 = α × 1：正好一个 α（ε 只差 10⁻⁸ 量级）", abs(k - 1) < 1e-7)
check("不修正会怎样：√0.004 = 0.06325，0.2 / 0.06325 = 3.162，第一步会是 α 的 3.16 倍",
      round(math.sqrt(0.004), 5) == 0.06325 and round(0.2 / 0.06325, 3) == 3.162 == round(0.2 / math.sqrt(0.004), 3))
m4, v4, mh4, vh4, k4 = adam_first_step(400.0)
check("g = 400：m = 40 → 400；v = 160 → 160,000，√160,000 = 400；这一步还是 α × 400 / 400 = α",
      math.isclose(m4, 40) and math.isclose(v4, 160) and math.isclose(mh4, 400) and math.isclose(vh4, 160000)
      and abs(k4 - 1) < 1e-7)
m3, v3, mh3, vh3, k3 = adam_first_step(-3.0)
check("7.5 自测：g = −3 → m = −0.3，v = 0.009 → −3、9 → 这一步 = α × (−3)/3 = −α，旋钮往大拧 α",
      math.isclose(m3, -0.3) and math.isclose(v3, 0.009) and math.isclose(mh3, -3) and math.isclose(vh3, 9)
      and abs(k3 + 1) < 1e-7)
ALPHA_ADAM = 0.3
p = torch.tensor([X_START, Y_START], dtype=F64, requires_grad=True)
opt = torch.optim.Adam([p], lr=ALPHA_ADAM)
opt.zero_grad()
bowl(p[0], p[1]).backward()
g_bowl = p.grad.clone()
opt.step()
hand_step = [X_START - ALPHA_ADAM * adam_first_step(g_bowl[0].item())[4], Y_START - ALPHA_ADAM * adam_first_step(g_bowl[1].item())[4]]
table(["", "x", "y"], [["梯度 g", g_bowl[0].item(), g_bowl[1].item()], ["手算 Adam 一步", hand_step[0], hand_step[1]],
                       ["torch Adam 一步", p[0].item(), p[1].item()]], floatfmt=".4g")
check("扁碗第一步：梯度 (2×3, 200×2) = (6, 400)，400 ÷ 6 ≈ 66.7，约 67 倍；Adam 两个方向都走 0.3 → (2.7, 1.7)，手算 = torch",
      g_bowl.tolist() == [6, 400] and round(400 / 6, 1) == 66.7 and round(400 / 6) == 67 and np.allclose(hand_step, [2.7, 1.7])
      and np.allclose(p.detach().numpy(), [2.7, 1.7]))


def run(opt_name, lr, steps=100):
    p = torch.tensor([X_START, Y_START], dtype=F64, requires_grad=True)
    opt = torch.optim.SGD([p], lr=lr) if opt_name == "sgd" else torch.optim.Adam([p], lr=lr)
    path = [p.detach().clone().numpy()]
    for _ in range(steps):
        opt.zero_grad()
        bowl(p[0], p[1]).backward()
        opt.step()
        path.append(p.detach().clone().numpy())
    return np.array(path)


paths = {f"SGD α={ALPHA_FAST:g}": run("sgd", ALPHA_FAST), f"SGD α={ALPHA_SLOW:g}": run("sgd", ALPHA_SLOW),
         f"Adam α={ALPHA_ADAM:g}": run("adam", ALPHA_ADAM)}
heights = {name: [float(bowl(*path[k])) for k in (10, 30, 60, 100)] for name, path in paths.items()}
table(["优化器", "10 步后高度", "30 步后", "60 步后", "100 步后"], [[n] + [f"{h:.4g}" for h in hs] for n, hs in heights.items()])
h_fast, h_slow, h_adam = heights.values()
y_fast = paths[f"SGD α={ALPHA_FAST:g}"][:, 1]
check("SGD α = 0.1：torch 走出的 y 是 2 → −38 → 722（每步乘 −19）", [round(float(v), 6) for v in y_fast[:3]] == [2, -38, 722])
check("SGD α = 0.1：10 步后高度已经 1.504e+28（科学计数法），越走越高", round(h_fast[0] / 1e28, 3) == 1.504 and h_fast[3] > h_fast[0])
check(f"SGD α = {ALPHA_SLOW:g}：torch 连走 100 步的高度 = 上面 0.99¹⁰⁰ 手算的 {rows[1][5]}", math.isclose(h_slow[3], float(rows[1][5]), rel_tol=1e-3))
check("SGD α = 0.005 的高度：10 步 7.361，30 步 4.924，60 步 2.694，100 步 1.206",
      [round(h, 3) for h in h_slow] == [7.361, 4.924, 2.694, 1.206])
check("Adam α = 0.3 的高度：10 步 24.24（比 SGD 还高），30 步 10.14，60 步 0.3982，100 步 0.0001204",
      [round(h_adam[0], 2), round(h_adam[1], 2), round(h_adam[2], 4), round(h_adam[3], 7)] == [24.24, 10.14, 0.3982, 0.0001204])
check("Adam 100 步后两个旋钮离碗底都不到 0.01 格", float(np.abs(paths[f"Adam α={ALPHA_ADAM:g}"][-1]).max()) < 0.01)
check("前 30 步 Adam 反而比 SGD α = 0.005 高（24.24 > 7.361，10.14 > 4.924）", h_adam[0] > h_slow[0] and h_adam[1] > h_slow[1])
m_a, v_a, m_b, v_b = 0.0, 0.0, 0.0, 0.0
for t, (ga, gb) in enumerate([(4.0, 4.0), (-4.0, 4.0)], start=1):   # a：+4 再 −4（来回跳）；b：+4 再 +4（一直同向）
    m_a, v_a = B1 * m_a + (1 - B1) * ga, B2 * v_a + (1 - B2) * ga * ga
    m_b, v_b = B1 * m_b + (1 - B1) * gb, B2 * v_b + (1 - B2) * gb * gb
k_a = (m_a / (1 - B1 ** 2)) / (math.sqrt(v_a / (1 - B2 ** 2)) + EPS)
k_b = (m_b / (1 - B1 ** 2)) / (math.sqrt(v_b / (1 - B2 ** 2)) + EPS)
print(f"来回跳（+4、−4）第 2 步：m = {m_a:.2f}，修正后 {m_a / (1 - B1 ** 2):.4f}；v = {v_a:.6f}，修正后 {v_a / (1 - B2 ** 2):.4f}；这一步 = {k_a:.4f} α")
check("折叠：+4、−4 第 2 步 m = −0.04 → ÷ 0.19 = −0.2105；v = 0.031984 → ÷ 0.001999 = 16.0，开方 4；这一步 −0.2105 ÷ 4 = −0.0526 α",
      round(m_a, 6) == -0.04 and round(m_a / 0.19, 4) == -0.2105 and round(v_a, 6) == 0.031984 and round(1 - B2 ** 2, 6) == 0.001999
      and round(v_a / (1 - B2 ** 2), 1) == 16.0 and round(-0.2105 / 4, 4) == -0.0526 == round(k_a, 4))
check("对照：+4、+4 一直同向，第 2 步修正后 m = 4，这一步还是 α", math.isclose(m_b / (1 - B1 ** 2), 4) and abs(k_b - 1) < 1e-6)
sig = inspect.signature(torch.optim.Adam.__init__).parameters
check("torch.optim.Adam 的默认值：β₁ = 0.9，β₂ = 0.999，ε = 1e-8", sig["betas"].default == (0.9, 0.999) and sig["eps"].default == 1e-8)

# ---------------------------------------------------------------------------
banner("5b. 画图：figures/ch07_optimizers.png（扁碗上三种走法）")
fig = plt.figure(figsize=(10.6, 15.6))
gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 0.95], hspace=0.62, wspace=0.05, left=0.09, right=0.99, top=0.915, bottom=0.07)
fig.suptitle("扁碗 x² + 100y²：SGD 要么飞、要么慢；Adam 每个方向自己定步长", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
gx, gy = np.meshgrid(np.linspace(-1.2, 3.8, 300), np.linspace(-2.6, 2.6, 300))
mx_s, my_s = 1 - 2 * ALPHA_SLOW, 1 - 2 * STEEP * ALPHA_SLOW
titles = [f"① SGD，α = {ALPHA_FAST:g}：陡的 y 方向一步跨过头",
          f"② SGD，α = {ALPHA_SLOW:g}：" + ("y 安全了，平的 x 方向爬不动" if abs(my_s) < 1 else "y 每步乘的数大小超过 1，也飞了"),
          f"③ Adam，α = {ALPHA_ADAM:g}：两个方向都走得动"]
texts = [[f"y 每步乘 1 − 200 × {ALPHA_FAST:g} = {fmt(1 - 2 * STEEP * ALPHA_FAST)}",
          f"y：{fmt(y_fast[0])} → {fmt(y_fast[1])} → {fmt(y_fast[2])} → …", f"10 步后高度 {h_fast[0]:.4g}"],
         [f"y 每步乘 1 − 200 × {ALPHA_SLOW:g} = {fmt(my_s, 4)}", f"x 每步乘 1 − 2 × {ALPHA_SLOW:g} = {fmt(mx_s, 4)}",
          f"100 步后 x = {float(rows[1][3]):.4g}", f"高度 {h_slow[3]:.4g}"],
         ["第 1 步：梯度 (6, 400)", "两个方向都走 0.3", "(3, 2) → (2.7, 1.7)", f"100 步后高度 {h_adam[3]:.4g}"]]
notes = ["箭头只画了第 1 步：从 y = 2 跳到 −38，远远出了图。",
         (f"100 个点挤在 y = 0 这条线上，每步只挪剩下那段 x 的 {fmt(2 * ALPHA_SLOW * 100, 2)}%。" if my_s == 0
          else f"y 每步乘 {fmt(my_s, 4)}：" + ("大小超过 1，越跳越远。" if abs(my_s) > 1 else "来回跳着缩小。")),
         "斜着冲下来，冲过碗底绕了一小圈，100 步后停在碗底。"]
for row, (name, path) in enumerate(paths.items()):
    ax = fig.add_subplot(gs[row, 0])
    data_axes(ax, "旋钮 x", "旋钮 y")
    ax.contour(gx, gy, bowl(gx, gy), levels=[1, 4, 9, 25, 100, 225, 400], colors=FAINT, linewidths=1.2)
    ax.set_xlim(-1.2, 3.8)
    ax.set_ylim(-2.6, 2.6)
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1, 2, 3])
    ax.set_yticks([-2, -1, 0, 1, 2])
    color = [MUTED, BLUE, ORANGE][row]
    if row == 0:
        arrow(ax, tuple(path[0]), (path[1][0], -2.6), color, lw=2.6)
    else:
        inside = path[np.all(np.isfinite(path), axis=1) & (np.abs(path[:, 1]) < 5)]
        ax.plot(inside[:, 0], inside[:, 1], "-o", color=color, ms=3.5, lw=1.4, zorder=5)
        arrow(ax, tuple(path[0]), tuple(path[1]), color, lw=2.6, zorder=6)
    ax.plot(*path[0], "o", color=INK, ms=9, zorder=7)
    ax.plot(0, 0, "*", color=GREEN, ms=18, zorder=7)
    ax.text(path[0][0] + 0.12, path[0][1] + 0.05, "出发 (3, 2)", fontsize=FS_SMALL, color=INK, va="bottom", bbox=WHITE_BOX)
    if row < 2:                                   # ③ 的路径正好绕着碗底转圈，字会被压住；读图段已说“绿星是碗底”
        ax.text(0.1, -0.45, "碗底", fontsize=FS_SMALL, color=GREEN, bbox=WHITE_BOX)
    ax_t = fig.add_subplot(gs[row, 1])
    lesson_panel(ax_t, xmax=10, ymax=10)
    for j, t in enumerate(texts[row]):
        hand(ax_t, 0.4, 8.6 - 2.2 * j, t, color=color if j == len(texts[row]) - 1 else BLUE, fontsize=FS_SMALL + 1)
    panel_title(fig, [ax, ax_t], titles[row])
    panel_note(fig, [ax, ax_t], notes[row])
savefig(fig, "ch07_optimizers")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("6. mini-batch 和 epoch：洗一次牌，切成几份，每份拧一次；全部轮一遍叫一个 epoch")
N_REC, N_PARTS, N_EPOCHS = 8, 4, 3
order = (torch.randperm(N_REC, generator=torch.Generator().manual_seed(35)) + 1).tolist()   # 记录编号 1–8 洗一次牌
size = N_REC // N_PARTS
parts = [order[i * size:(i + 1) * size] for i in range(N_PARTS)]
updates = [(ep + 1, i + 1, part) for ep in range(N_EPOCHS) for i, part in enumerate(parts)]   # 和项目一样：每遍用同一种切法
print("洗牌后的顺序：", order)
print("切成的 4 份：", parts)
for ep in range(N_EPOCHS):
    print(f"  第 {ep + 1} 遍：", "  ".join(f"更新 {ep * N_PARTS + i + 1}：{part}" for i, part in enumerate(parts)))
check("8 条记录每份 2 条：8 ÷ 4 = 2；过 3 遍：4 × 3 = 12 次更新", size == 2 and len(updates) == 12 == 4 * 3)
check("12 次更新一共看了 24 条次，但不同的记录只有 8 条：每条正好被看 3 次",
      sum(len(u[2]) for u in updates) == 24 and len({r for u in updates for r in u[2]}) == 8
      and all(sum(r in u[2] for u in updates) == 3 for r in range(1, 9)))
check("7.6 自测（旧稿原数）：切 2 份、每份 4 条、过 3 遍 → 2 × 3 = 6 次更新；记录仍只有 8 条", (8 // 2, 2 * 3) == (4, 6))
REC4 = [(2.0, 5.0), (1.0, 3.0), (2.0, 3.0), (1.0, 1.0)]     # 4 条记录（第 1 条就是 7.1 节那条），w = 1，b = 0
per = [2 * forward1(W0, B0, x, y)[1] * x for x, y in REC4]   # 每条记录自己的 ∂L/∂w = 2·e·x（7.2 节）
full, part_a, part_b = sum(per) / 4, sum(per[:2]) / 2, sum(per[2:]) / 2
xs_t, ys_t = torch.tensor([r[0] for r in REC4], dtype=F64), torch.tensor([r[1] for r in REC4], dtype=F64)
w_t = torch.tensor(W0, dtype=F64, requires_grad=True)
torch.mean((w_t * xs_t + B0 - ys_t) ** 2).backward()
print(f"每条记录的 ∂L/∂w：{[fmt(g) for g in per]}；全体平均 {fmt(full)}；前两条 {fmt(part_a)}；后两条 {fmt(part_b)}")
check("每条的 2·e·x：−12、−4、−4、0；全体平均 (−12 − 4 − 4 + 0)/4 = −5（torch 对 4 条的均方误差求导也是 −5）",
      per == [-12, -4, -4, 0] and full == -5 and w_t.grad.item() == -5)
check("两份各算一次：(−12 − 4)/2 = −8，(−4 + 0)/2 = −2——方向一样，大小在抖；两份再平均 (−8 − 2)/2 = −5",
      (part_a, part_b) == (-8, -2) and (part_a + part_b) / 2 == full)
ENVS, STEPS, MB, EP = 4096, 24, 4, 5
rec = ENVS * STEPS
check("项目：4096 × 24 = 98,304 条；÷ 4 = 24,576 条一份；4 份 × 5 遍 = 20 次更新",
      rec == 98304 and rec // MB == 24576 and MB * EP == 20)
check("7.6 自测：切 8 份 → 98,304 ÷ 8 = 12,288 条一份，8 × 5 = 40 次更新", rec // 8 == 12288 and 8 * EP == 40)

# ---------------------------------------------------------------------------
banner("6b. 画图：figures/ch07_minibatch.png（洗牌 → 切份 → 轮 3 遍 → 项目规模）")
fig, axes = lesson_stack([3.0, 3.3, 4.6, 2.5], "8 条记录、每份 2 条、过 3 遍：12 次更新，记录还是那 8 条")
lesson_panel(axes[0], "① 8 条记录，先洗一次牌", xmax=XMAX, ymax=3.0)
for i in range(N_REC):
    cell(axes[0], 0.4 + i * 0.85, 0.9, str(i + 1), width=0.85, height=0.8)
    cell(axes[0], 9.3 + i * 0.85, 0.9, str(order[i]), width=0.85, height=0.8)
arrow(axes[0], (7.4, 1.3), (9.1, 1.3), MUTED, lw=2.4)
axes[0].text(8.25, 1.55, "洗牌", ha="center", va="bottom", fontsize=FS_SMALL, color=MUTED)
note(axes[0], 0.4, 0.35, "洗牌只是打乱顺序，记录还是这 8 条。")
lesson_panel(axes[1], "② 每 2 条切成一份：4 个 mini-batch", xmax=XMAX, ymax=3.3)
for i, part in enumerate(parts):
    x0 = 0.4 + i * 4.1
    for j, r in enumerate(part):
        cell(axes[1], x0 + j * 0.85, 1.25, str(r), width=0.85, height=0.8)
    axes[1].text(x0 + 0.85, 0.85, f"第 {i + 1} 份", ha="center", va="center", fontsize=FS_SMALL, color=BLUE)
note(axes[1], 0.4, 0.3, "每一份算一次梯度、拧一次旋钮，这就是一次更新。")
lesson_panel(axes[2], "③ 4 份轮一遍 = 1 个 epoch；过 3 遍 = 12 次更新", xmax=XMAX, ymax=4.6)
for ep in range(N_EPOCHS):
    yy = 2.75 - ep * 0.95
    axes[2].text(0.4, yy + 0.35, f"第 {ep + 1} 遍", fontsize=FS_SMALL, color=GREEN, va="center")
    for i, part in enumerate(parts):
        cell(axes[2], 2.2 + i * 3.4, yy, f"更新 {ep * N_PARTS + i + 1}：{part[0]}, {part[1]}", width=3.2, height=0.72,
             fontsize=FS_SMALL, facecolor=CELL_HOT if ep == 0 and i == 0 else CELL)
note(axes[2], 0.4, 0.3, "每条记录被看了 3 次：一共 24 条次，但不是 24 条新的经历。")
lesson_panel(axes[3], "④ 项目里：同一件事，数大一点", xmax=XMAX, ymax=2.5)
hand(axes[3], 0.4, 1.45, "4096 × 24 = 98,304 条 → 切 4 份，每份 24,576 条 → 过 5 遍 → 4 × 5 = 20 次更新")
note(axes[3], 0.4, 0.5, "20 次更新，用的还是这 98,304 条。")
savefig(fig, "ch07_minibatch")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("7. 一次完整的更新：五行代码循环 420 次，小网络学会描一条波浪线")
torch.manual_seed(0)       # 和旧版实验同一个起点（旧版在这之前没用过随机数），所以数字能对上
X = torch.linspace(-math.pi, math.pi, 400).unsqueeze(1)
Y = torch.sin(X)
net = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.ELU(), torch.nn.Linear(32, 32), torch.nn.ELU(), torch.nn.Linear(32, 1))
opt = torch.optim.Adam(net.parameters(), lr=1e-2)
BATCH_SIZE = 64  # TWEAK-3: 400
EPOCHS = 60


def dec(v, sig=3) -> str:
    """三位有效数字、不用科学计数法：0.0000609 而不是 6.09e-05。"""
    return np.format_float_positional(v, precision=sig, unique=False, fractional=False, trim="k")


def full_loss():
    with torch.no_grad():
        return torch.mean((net(X) - Y) ** 2).item()


with torch.no_grad():
    snaps = {0: net(X).squeeze().numpy().copy()}
losses, n_updates = {0: full_loss()}, 0
for epoch in range(1, EPOCHS + 1):
    perm = torch.randperm(len(X))                         # 这个例子每遍都重新洗牌（项目只洗一次，见 📍）
    for i in range(0, len(X), BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        opt.zero_grad()                                   # ① 清掉上一次的梯度
        loss = torch.mean((net(X[idx]) - Y[idx]) ** 2)    # ② 用这一份数据前向，算损失
        loss.backward()                                   # ③ 反向传播：梯度写进每个旋钮的 .grad
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)   # ④ 梯度裁剪（第 3 章 3.7 节）
        opt.step()                                        # ⑤ Adam 拧旋钮
        n_updates += 1
    losses[epoch] = full_loss()
    if epoch in (1, EPOCHS):
        with torch.no_grad():
            snaps[epoch] = net(X).squeeze().numpy().copy()
per_epoch = math.ceil(len(X) / BATCH_SIZE)
SHOW = [0, 1, 5, 10, 20, 40, 60]
table(["过了几个 epoch", "更新了几次", "全部 400 个点的损失"],
      [["训练前" if e == 0 else e, e * per_epoch, dec(losses[e])] for e in SHOW if e <= EPOCHS])
print(f"400 个点、每份 {BATCH_SIZE} 个：每个 epoch {per_epoch} 次更新，{EPOCHS} 个 epoch 共 {n_updates} 次。")
check(f"每份 {BATCH_SIZE} 个点：400 ÷ {BATCH_SIZE} → 每个 epoch {per_epoch} 次更新，{EPOCHS} 个 epoch 共 {per_epoch * EPOCHS} 次",
      n_updates == per_epoch * EPOCHS)
check("正文的数：400 ÷ 64 = 6 份满的 + 1 份 16 个 → 每个 epoch 7 次，7 × 60 = 420 次", n_updates == 420 and 6 * 64 + 16 == 400)
check("损失：训练前 0.300 → 1 个 epoch 0.238 → 5 个 0.0752 → 10 个 0.0146 → 20 个 0.00152 → 40 个 0.0000551 → 60 个 0.0000609",
      [dec(losses[e]) for e in SHOW] == ["0.300", "0.238", "0.0752", "0.0146", "0.00152", "0.0000551", "0.0000609"])
print(f"开根号换回“差多少”：√{dec(losses[0])} = {math.sqrt(losses[0]):.2f}，√{dec(losses[EPOCHS])} = {math.sqrt(losses[EPOCHS]):.4f}")
check("正文的换算：√0.300 ≈ 0.55，√0.0000609 ≈ 0.0078（字面值和实际损失各算一遍）",
      round(math.sqrt(0.300), 2) == 0.55 and round(math.sqrt(0.0000609), 4) == 0.0078
      and round(math.sqrt(losses[0]), 2) == 0.55 and round(math.sqrt(losses[EPOCHS]), 4) == 0.0078)
check("60 个 epoch 后损失 < 0.01（换个随机种子也成立）", losses[EPOCHS] < 0.01)
check("到了碗底附近在轻轻晃：第 40 个 epoch 的损失比第 60 个还低一点", EPOCHS == 60 and losses[40] < losses[60])

# ---------------------------------------------------------------------------
banner("7b. 画图：figures/ch07_sin_fit.png（训练前 → 1 个 epoch → 60 个 epoch）")
fig = plt.figure(figsize=(10, 15.2))
fitted = losses[EPOCHS] < 0.001                        # “改一改”第 3 条之后可能描不好：标题跟着实际结果走
default_batch = BATCH_SIZE == 64
fig.suptitle(f"五行代码循环 {n_updates} 次：" + ("小网络把波浪线描了出来" if fitted else "波浪线还没描好"),
             fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
xs = X.squeeze().numpy()
stages = [(0, "① 训练前：输出和波浪线对不上", "旋钮还是随机的初始值，输出只是一条微微弯曲的斜线。"),
          (1, f"② 1 个 epoch 后（{per_epoch} 次更新）" + ("：开始往波浪线靠" if default_batch else ""),
           "只看过每个点一次：橙线变陡了，中间一段贴近了波浪线，两头还差得远。" if default_batch
           else f"只更新了 {per_epoch} 次，对照上面的损失表看。"),
          (EPOCHS, f"③ {EPOCHS} 个 epoch 后（{n_updates} 次更新）：" + ("几乎重合" if fitted else "还没描好"),
           "橙线压在灰线上，几乎看不出差别。" if fitted else f"损失 {dec(losses[EPOCHS])}，对照上面的损失表看。")]
for row, (ep, title, text) in enumerate(stages):
    ax = fig.add_axes([0.11, 0.72 - row * 0.30, 0.85, 0.17])
    data_axes(ax, "输入 x", "输出")
    ax.plot(xs, Y.squeeze().numpy(), color=FAINT, lw=5, label="目标：sin(x)")
    ax.plot(xs, snaps[ep], color=ORANGE, lw=2.6, label="网络的输出")
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-1.35, 1.35)
    ax.set_yticks([-1, 0, 1])
    ax.text(3.2, -1.2, f"损失 {dec(losses[ep])}", ha="right", fontsize=FS_SMALL, color=ORANGE, bbox=WHITE_BOX)
    if row == 0:
        ax.legend(loc="upper left", fontsize=FS_SMALL, frameon=False)
    panel_title(fig, [ax], title)
    panel_note(fig, [ax], text)
savefig(fig, "ch07_sin_fit")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("7c. 画图：figures/ch07_overview.png（7.0 节的总览：前向 → 反向 → 拧旋钮 → 换一份再来）")
fig, axes = lesson_stack([3.8, 3.9, 2.9, 2.9], "一次更新：前向算数，反向乘倍数，优化器拧旋钮，换一份数据再来")
lesson_panel(axes[0], f"① 前向：数从左往右流，算出损失 L = {fmt(L1)}", xmax=XMAX, ymax=3.8)
pipeline(axes[0], EX1_NUMS, EX1_MACS, y=0.55, feeds=EX1_FEEDS)
note(axes[0], 11.2, 0.95, "7.1 节")
lesson_panel(axes[1], "② 反向：从 L 往回乘，得到每个旋钮的梯度", xmax=XMAX, ymax=3.9)
cn, _ = pipeline(axes[1], EX1_NUMS, EX1_MACS, y=1.95, faint=True)
grads_under(axes[1], cn, 0.95, [None, fmt(g_yh), fmt(g_e), "1"])
hand(axes[1], 0.3, 0.45, f"∂L/∂w = {fmt(g_w)}，∂L/∂b = {fmt(g_b)}", color=ORANGE)
note(axes[1], 11.2, 1.26, "7.2 节")
lesson_panel(axes[2], f"③ 优化器照梯度拧旋钮（SGD，α = {ALPHA_SGD:g}）", xmax=XMAX, ymax=2.9)
hand(axes[2], 0.3, 1.5, f"w：1 − 0.01 × (−12) = {w_new:.2f}　b：0 − 0.01 × (−6) = {b_new:.2f}　损失 {fmt(L1)} → {L_new:.2f}")
note(axes[2], 0.3, 0.6, "7.3 节是 SGD；7.4、7.5 节换成会看历史的 Adam。")
lesson_panel(axes[3], "④ 换一份数据，再做一遍 ①②③", xmax=XMAX, ymax=2.9)
hand(axes[3], 0.3, 1.5, "8 条记录：4 份 × 3 遍 = 12 次更新　项目：4 份 × 5 遍 = 20 次", color=GREEN)
note(axes[3], 0.3, 0.6, "7.6 节。每一份数据走一遍 ①②③，就是一次更新。")
savefig(fig, "ch07_overview")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("8. 映射到项目：正文引用的常数和源码行还在不在")
REPO = Path(__file__).resolve().parents[3]
cfg_path = REPO / "src" / "mjlab_microduck" / "tasks" / "microduck_velocity_env_cfg.py"
CFG_LINES = ["num_learning_epochs=5,", "num_mini_batches=4,", "learning_rate=1.0e-3,", "max_grad_norm=1.0,", "num_steps_per_env=24,"]
if cfg_path.is_file():
    cfg_text = cfg_path.read_text(encoding="utf-8")
    at = cfg_text.find("MicroduckRlCfg = RslRlOnPolicyRunnerCfg(")
    print("microduck_velocity_env_cfg.py 的 MicroduckRlCfg：", "  ".join(CFG_LINES))
    check("项目配置：5 个 epoch、4 个 mini-batch、α 初始 0.001、裁剪上限 1.0、每只机器人 24 步",
          at >= 0 and lines_in_order(cfg_text[at:], CFG_LINES))
    agents = (REPO / "AGENTS.md").read_text(encoding="utf-8") if (REPO / "AGENTS.md").is_file() else ""
    check("训练命令里 4096 只机器人：--env.scene.num-envs 4096", "--env.scene.num-envs 4096" in agents)
    cfg_block = cfg_text[at:cfg_text.find("\n)\n", at)]
    check("项目的 actor、critic 都是一层接一层的普通网络（hidden_dims 512/256/128，没有 rnn）",
          at >= 0 and cfg_block.count("hidden_dims=(512, 256, 128),") == 2 and "rnn" not in cfg_block.lower())
else:
    print("  （没找到项目的 env cfg，跳过这一项）")
rsl_spec = importlib.util.find_spec("rsl_rl")
rsl_dir = Path(rsl_spec.origin).parent if rsl_spec and rsl_spec.origin else None
if rsl_dir and (rsl_dir / "algorithms" / "ppo.py").is_file():
    ppo_text = (rsl_dir / "algorithms" / "ppo.py").read_text(encoding="utf-8")
    PPO_LINES = ['optimizer: str = "adam",', "self.optimizer = resolve_optimizer(optimizer)(",
                 "chain(self.actor.parameters(), self.critic.parameters()), lr=learning_rate",
                 "generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)",
                 "for batch in generator:", "self.optimizer.zero_grad()", "loss.backward()",
                 "nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)",
                 "nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)", "self.optimizer.step()",
                 "num_updates = self.num_learning_epochs * self.num_mini_batches"]
    check(f"rsl_rl/algorithms/ppo.py：映射块引用的 {len(PPO_LINES)} 行都原样存在，且顺序一致", lines_in_order(ppo_text, PPO_LINES))
    check("检查点里连 Adam 的两本账一起存：save() 里有 optimizer_state_dict",
          '"optimizer_state_dict": self.optimizer.state_dict(),' in ppo_text)
    utils_text = (rsl_dir / "utils" / "utils.py").read_text(encoding="utf-8")
    check('rsl_rl/utils/utils.py 的 resolve_optimizer：名字 "adam" 换成 torch.optim.Adam',
          lines_in_order(utils_text, ["def resolve_optimizer(", '"adam": torch.optim.Adam,']))
    st_text = (rsl_dir / "storage" / "rollout_storage.py").read_text(encoding="utf-8")
    ST_LINES = ["def mini_batch_generator(", "batch_size = self.num_envs * self.num_transitions_per_env",
                "mini_batch_size = batch_size // num_mini_batches",
                "indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)",
                "for epoch in range(num_epochs):", "for i in range(num_mini_batches):", "start = i * mini_batch_size",
                "stop = (i + 1) * mini_batch_size", "batch_idx = indices[start:stop]", "yield RolloutStorage.Batch("]
    check(f"rsl_rl/storage/rollout_storage.py：mini_batch_generator 的 {len(ST_LINES)} 行都原样存在，且顺序一致",
          lines_in_order(st_text, ST_LINES))
    gen = st_text[st_text.find(ST_LINES[0]):st_text.find("def recurrent_mini_batch_generator")]
    check("只洗一次牌：randperm 在两层循环外面，循环里没有再洗", gen.count("randperm") == 1
          and gen.find("randperm") < gen.find("for epoch in range(num_epochs):"))
    check("update() 取 mini-batch 有两支：另一类网络走 recurrent_mini_batch_generator，普通网络走 mini_batch_generator",
          lines_in_order(ppo_text, ["if self.actor.is_recurrent or self.critic.is_recurrent:",
                                    "generator = self.storage.recurrent_mini_batch_generator(", "else:",
                                    "generator = self.storage.mini_batch_generator("]))
else:
    print("  （当前 Python 环境里没有 rsl_rl，跳过源码核对；用 uv run 运行就会核对）")

done()
