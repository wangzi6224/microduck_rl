"""第 12 章实验：优势估计——只看一步和一直看到终点两个极端、n 步回报、GAE 的展开与递推、往回传、偏差与方差、
超时自举、优势归一化；最后把迷你记录直接交给 rsl_rl 的 compute_returns，逐位核对。

运行：uv run python docs/learn-zh/labs/ch12_gae.py
纯 CPU，numpy + matplotlib；第 7、8 节用到 torch 和 rsl_rl（没装就跳过那几项）。第 8 节只读几份源码的文字、
调用一次 rsl_rl 的 compute_returns，不训练。第 5 节抽 200 万段 24 步的随机记录（几秒钟）。
小节编号与正文一一对应：实验第 K 节 = 正文 12.K 节（第 8 节对应「映射到项目」）。
正文“改一改”要改的三行都带 `# TWEAK-k:` 标记（第 3 节的 λ、第 4 节第 6 步的奖励、第 5 节的奖励起伏各一处）。
"""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from _common import banner, check, done, lines_in_order, num, savefig, table
from _draw import (BLUE, CELL, CELL_EDGE, CELL_HOT, FAINT, FS_NOTE, FS_SMALL, FS_STEP, FS_TICK, FS_TITLE, GREEN, INK,
                   MUTED, ORANGE, WHITE_BOX, arrow, cell, data_axes, hand, lesson_cells, lesson_figure, lesson_panel,
                   note, panel_note, panel_title, plt)

np.set_printoptions(precision=6, suppress=True)
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def sub(k) -> str:
    return str(k).translate(SUB)                       # 下标：sub(3) → ₃


def m(v, places=4) -> str:
    return num(v, places).replace("-", "−")            # 图里和手算串里用正文的减号 −


# ---------------------------------------------------------------------------
# 全章共用的两个函数：TD 误差，和 GAE 的倒序循环（和 rsl_rl 的 compute_returns 是同一个循环）
def td_errors(r, v, v_last, dones, gamma):
    """δ_t = r_t + γ·V(s_{t+1})·(1 − done_t) − V(s_t)。最后一步的“下一步价值”用 v_last（采集末尾的估计）。"""
    r, v, dones = (np.asarray(x, dtype=float) for x in (r, v, dones))
    v_next = np.concatenate([v[1:], np.asarray(v_last, dtype=float).reshape(1, *v.shape[1:])])
    return r + (1.0 - dones) * gamma * v_next - v


def gae(r, v, v_last, dones, gamma, lam):
    """沿第 0 轴（时间）从后往前：Â_t = δ_t + γλ·(1 − done_t)·Â_{t+1}；返回 (Â, returns = Â + V)。
    r、v、dones 的第 0 轴是时间；后面还可以跟一条“机器人”轴，每只机器人各算各的（第 4 节的 [8, 2]）。"""
    r, v, dones = (np.asarray(x, dtype=float) for x in (r, v, dones))
    steps = r.shape[0]
    adv = np.zeros(np.broadcast_shapes(r.shape, v.shape))
    running = np.zeros(adv.shape[1:])
    for t in reversed(range(steps)):
        v_next = np.asarray(v_last, dtype=float) if t == steps - 1 else v[t + 1]
        not_end = 1.0 - dones[t]
        delta = r[t] + not_end * gamma * v_next - v[t]
        running = delta + not_end * gamma * lam * running
        adv[t] = running
    return adv, adv + v


def gae_by_sum(delta, dones, gamma, lam):
    """按定义逐项加：Â_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + …，碰到回合结束就停。"""
    out = np.zeros(len(delta))
    for t in range(len(delta)):
        weight = 1.0
        for k in range(t, len(delta)):
            out[t] += weight * delta[k]
            if dones[k]:
                break
            weight *= gamma * lam
    return out


def discounted_returns(r, dones, gamma, v_last=0.0):
    """回报 G_t：真拿到的奖励从后往前打折累加（第 9 章 9.7 节）；采集末尾没结束的，用 v_last 补上。"""
    out, running = np.zeros(len(r)), v_last
    for t in reversed(range(len(r))):
        running = r[t] + (1 - dones[t]) * gamma * running
        out[t] = running
    return out


# ---------------------------------------------------------------------------
banner("1. 两个极端：只看一步（δ），还是一直看到终点（回报 − V）")
R3 = np.array([0.1, 0.2, 1.0])        # 三步的奖励（教学用的数）
V3 = np.array([0.5, 0.4, 0.3])        # critic 在这三步报的价值
D3 = np.array([0, 0, 1])              # 第 2 步之后回合终止：之后没有分，价值记 0
GAMMA_T = 0.9                         # 12.1–12.3 节为了好算用 0.9（项目是 0.99）
table(["t", "奖励 r", "critic 报的 V", "这一步之后"],
      [[t, num(R3[t]), num(V3[t]), "终止（之后记 0）" if D3[t] else "接着走"] for t in range(3)])
d3 = td_errors(R3, V3, 0.0, D3, GAMMA_T)
v_next3 = [V3[1], V3[2], 0.0]
for t in range(3):
    print(f"δ{sub(t)} = {num(R3[t])} + 0.9 × {num(v_next3[t])} − {num(V3[t])} = {m(d3[t])}")
G3 = discounted_returns(R3, D3, GAMMA_T)
print(f"只看一步：Â₀ = δ₀ = {m(d3[0])}")
print(f"一直看到终点：G₀ = 0.1 + 0.9 × 0.2 + 0.81 × 1 = {num(G3[0])}；G₀ − V₀ = {num(G3[0])} − 0.5 = {m(G3[0] - V3[0])}")
print(f"critic 说第 1 步还值 {num(V3[1])}，实际从第 1 步起拿到 G₁ = 0.2 + 0.9 × 1 = {num(G3[1])}")
check("三个 δ：−0.04、0.07、0.7（12.1 节手算）", np.allclose(d3, [-0.04, 0.07, 0.7]))
check("按印出来的数重算：0.1 + 0.9 × 0.4 − 0.5 = −0.04；0.2 + 0.9 × 0.3 − 0.4 = 0.07；1 + 0 − 0.3 = 0.7",
      round(0.1 + 0.9 * 0.4 - 0.5, 10) == -0.04 and round(0.2 + 0.9 * 0.3 - 0.4, 10) == 0.07
      and round(1 + 0 - 0.3, 10) == 0.7)
check("回报 G₀ = 1.09、G₁ = 1.1、G₂ = 1；G₀ − V₀ = 0.59",
      np.allclose(G3, [1.09, 1.1, 1.0]) and round(G3[0] - V3[0], 10) == 0.59
      and round(0.1 + 0.9 * 0.2 + 0.81 * 1, 10) == 1.09 and round(0.2 + 0.9 * 1, 10) == 1.1)
check("自测：第 1 步的动作，只看一步 Â₁ = δ₁ = 0.07；看到终点 G₁ − V₁ = 1.1 − 0.4 = 0.7",
      round(d3[1], 10) == 0.07 and round(G3[1] - V3[1], 10) == 0.7 and round(1.1 - 0.4, 10) == 0.7)

# ---------------------------------------------------------------------------
banner("2. n 步回报：先看 n 步真拿到的分，剩下的问 critic")


def n_step_parts(r, v, dones, gamma, n, t=0):
    """从第 t 步起：前 n 步真拿到的奖励（打过折）加起来；第 n 步之后请 critic 补。中途终止就不再补。"""
    real, disc = 0.0, 1.0
    for k in range(t, t + n):
        real += disc * r[k]
        disc *= gamma
        if dones[k]:
            return real, 0.0
    return real, disc * v[t + n]


parts = [n_step_parts(R3, V3, D3, GAMMA_T, n) for n in (1, 2, 3)]
nstep = [a + b for a, b in parts]
nstep_adv = [g - V3[0] for g in nstep]
table(["看几步", "真拿到的（打过折）", "critic 补的", "n 步回报", "减去 V₀"],
      [[n, num(a), num(b) if n < 3 else "0（已终止）", num(g), num(x)] for n, (a, b), g, x
       in zip((1, 2, 3), parts, nstep, nstep_adv)])
gain2, gain3 = nstep_adv[1] - nstep_adv[0], nstep_adv[2] - nstep_adv[1]
print(f"多看一步（1 → 2）：多出 {num(gain2)} = 0.9 × δ₁ = 0.9 × {num(d3[1])}")
print(f"再多看一步（2 → 3）：多出 {num(gain3)} = 0.81 × δ₂ = 0.81 × {num(d3[2])}")
print(f"换掉的那一块：0.9 × 0.4 换成 0.9 × (0.2 + 0.9 × 0.3) = {num(0.9 * (0.2 + 0.9 * 0.3))}，差 {num(0.9 * (0.2 + 0.9 * 0.3) - 0.9 * 0.4)}")
check("1、2、3 步回报：0.46、0.523、1.09；减去 V₀ 得 −0.04、0.023、0.59",
      np.allclose(nstep, [0.46, 0.523, 1.09]) and np.allclose(nstep_adv, [-0.04, 0.023, 0.59]))
check("按印出来的数重算：0.1 + 0.36 = 0.46；0.1 + 0.18 + 0.243 = 0.523；0.1 + 0.18 + 0.81 = 1.09",
      round(0.1 + 0.36, 10) == 0.46 and round(0.1 + 0.18 + 0.243, 10) == 0.523 and round(0.1 + 0.18 + 0.81, 10) == 1.09)
check("每多看一步，恰好多加一个打过折的 δ：0.023 = −0.04 + 0.9 × 0.07；0.59 = 0.023 + 0.81 × 0.7",
      math.isclose(gain2, GAMMA_T * d3[1]) and math.isclose(gain3, GAMMA_T ** 2 * d3[2])
      and round(-0.04 + 0.9 * 0.07, 10) == 0.023 and round(0.023 + 0.81 * 0.7, 10) == 0.59)
real2, est2 = n_step_parts(R3, np.array([0.5, 0.4, 1.0]), D3, GAMMA_T, 2)
check("自测：critic 在第 2 步要是报准了（1 而不是 0.3），看 2 步 = 0.1 + 0.18 + 0.81 × 1 − 0.5 = 0.59，和看到终点一样",
      round(real2 + est2 - V3[0], 10) == 0.59 and round(0.1 + 0.18 + 0.81 * 1 - 0.5, 10) == 0.59)
check("换掉的差额 0.9 × (0.2 + 0.27 − 0.4) = 0.9 × 0.07 = 0.063",
      round(0.9 * (0.2 + 0.27 - 0.4), 10) == 0.063 and round(0.9 * 0.07, 10) == 0.063)

# ---------------------------------------------------------------------------
banner("2b. 画图：figures/ch12_nstep_ladder.png（看 1 步、2 步、一直到终点）")
fig, axes = lesson_figure(3, "n 步回报：多看一步真的，就少信一步 critic", panel_height=2.95, width=11.6)
slot_x = [0.3 + 2.12 * i for i in range(4)]
rows = [  # 每一格：(文字, 种类)  种类 real = 真拿到的，est = critic 的估计，end = 终止之后
    [("0.1", "real"), ("0.9 × 0.4", "est")],
    [("0.1", "real"), ("0.9 × 0.2", "real"), ("0.81 × 0.3", "est")],
    [("0.1", "real"), ("0.9 × 0.2", "real"), ("0.81 × 1", "real"), ("终止：0", "end")],
]
titles = ["① 看 1 步：第 0 步是真的，之后全问 critic",
          "② 看 2 步：第 1 步也真走了，第 2 步起问 critic",
          "③ 一直看到终点：三步都是真拿到的"]
notes = ["蓝格是真拿到的奖励（乘过折扣），橙格是 critic 的估计；右边再减去 V₀ = 0.5。",
         f"critic 的 0.9 × 0.4 换成了真走一步：多出 0.9 × δ₁ = {num(gain2)}。",
         f"没有 critic 的份了：又多出 0.81 × δ₂ = {num(gain3)}。"]
face = {"real": "#dcebf7", "est": CELL_HOT, "end": "#eeeeee"}
edge = {"real": BLUE, "est": ORANGE, "end": FAINT}
color = {"real": BLUE, "est": ORANGE, "end": MUTED}
for ax, row, title, text, g, x in zip(axes, rows, titles, notes, nstep, nstep_adv):
    lesson_panel(ax, title, xmax=11.6, ymax=3.6)
    for i in range(4):
        ax.text(slot_x[i] + 0.86, 2.5, "终止之后" if i == 3 else f"第 {i} 步", fontsize=FS_SMALL - 2, color=MUTED,
                ha="center", va="center")
    for i, (label, kind) in enumerate(row):
        cell(ax, slot_x[i], 1.55, label, width=1.72, height=0.72, facecolor=face[kind], edgecolor=edge[kind],
             fontsize=FS_SMALL, color=color[kind])
        if i:
            ax.text(slot_x[i] - 0.2, 1.91, "+", fontsize=FS_STEP, color=INK, ha="center", va="center")
    hand(ax, 8.95, 2.12, f"= {num(g)}", color=INK)
    hand(ax, 8.95, 1.42, f"− 0.5 = {m(x)}", color=ORANGE)
    note(ax, 0.3, 0.72, text)
savefig(fig, "ch12_nstep_ladder")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("3. GAE：后面的 δ 全都加进来，每远一步多乘一次 γλ")
LAM_T = 0.8  # TWEAK-1: 1.0
GL_T = GAMMA_T * LAM_T
A3, RET3 = gae(R3, V3, 0.0, D3, GAMMA_T, LAM_T)
A3_sum = gae_by_sum(d3, D3, GAMMA_T, LAM_T)
print(f"γλ = 0.9 × {num(LAM_T)} = {num(GL_T)}；(γλ)² = {num(GL_T ** 2)}")
print(f"展开（按定义逐项加）：Â₀ = −0.04 + {num(GL_T)} × 0.07 + {num(GL_T ** 2)} × 0.7"
      f" = −0.04 + {num(GL_T * 0.07, 5)} + {num(GL_T ** 2 * 0.7, 5)} = {m(A3_sum[0], 5)}")
print(f"从后往前（递推）：Â₂ = 0.7；Â₁ = 0.07 + {num(GL_T)} × 0.7 = {m(A3[1], 5)}；"
      f"Â₀ = −0.04 + {num(GL_T)} × {num(A3[1], 5)} = {m(A3[0], 5)}")
check("GAE（γ = 0.9、λ = 0.8）：Â₂ = 0.7、Â₁ = 0.574、Â₀ = 0.37328（worked_examples 冻结）",
      np.allclose(A3, [0.37328, 0.574, 0.7]) and round(A3[0], 5) == 0.37328)
check("展开逐项加 = 从后往前递推（每一步都相同）", np.allclose(A3_sum, A3))
check("按印出来的数重算：0.07 + 0.72 × 0.7 = 0.574；−0.04 + 0.72 × 0.574 = 0.37328；−0.04 + 0.0504 + 0.36288 = 0.37328",
      round(0.07 + 0.72 * 0.7, 10) == 0.574 and round(-0.04 + 0.72 * 0.574, 10) == 0.37328
      and round(-0.04 + 0.0504 + 0.36288, 10) == 0.37328 and round(0.5184 * 0.7, 10) == 0.36288
      and round(0.72 * 0.72, 10) == 0.5184)
js_like = 0.0
for d in reversed([-0.04, 0.07, 0.7]):                # 和正文 JS 的 reduceRight 同样的顺序、同样的小数运算
    js_like = d + (0.9 * 0.8) * js_like
print(f"照 JS 的 reduceRight 算：{js_like!r}（小数存不精确，末尾多出一个 1）")
check("按印出来的数重算：0.72 × 0.7 = 0.504；0.72 × 0.574 = 0.41328；JS 控制台印 0.3732800000000001",
      round(0.72 * 0.7, 10) == 0.504 and round(0.72 * 0.574, 10) == 0.41328 and repr(js_like) == "0.3732800000000001")
mix_w = [1 - LAM_T, (1 - LAM_T) * LAM_T, LAM_T ** 2]
mix = sum(w * x for w, x in zip(mix_w, nstep_adv))
print(f"n 步回报的加权平均：权重 1 − λ = {num(mix_w[0])}、(1 − λ)λ = {num(mix_w[1])}、λ² = {num(mix_w[2])}，"
      f"加起来 {num(sum(mix_w))}")
print(f"  {num(mix_w[0])} × (−0.04) + {num(mix_w[1])} × 0.023 + {num(mix_w[2])} × 0.59 = {m(mix, 5)}")
check("n 步回报按 0.2、0.16、0.64 加权平均 = GAE 的 0.37328（进阶折叠）",
      np.allclose(mix_w, [0.2, 0.16, 0.64]) and math.isclose(sum(mix_w), 1.0) and math.isclose(mix, A3[0])
      and round(0.2 * -0.04 + 0.16 * 0.023 + 0.64 * 0.59, 10) == 0.37328)
check("折叠里的逐项：0.2 × (−0.04) = −0.008，0.16 × 0.023 = 0.00368，0.64 × 0.59 = 0.3776；"
      "δ₁ 分到 (0.16 + 0.64) × 0.9 = 0.72，δ₂ 分到 0.64 × 0.81 = 0.5184",
      round(0.2 * -0.04, 10) == -0.008 and round(0.16 * 0.023, 10) == 0.00368 and round(0.64 * 0.59, 10) == 0.3776
      and round(-0.008 + 0.00368 + 0.3776, 10) == 0.37328 and round((0.16 + 0.64) * 0.9, 10) == 0.72
      and round(0.64 * 0.81, 10) == 0.5184)
A3_half, _ = gae(R3, V3, 0.0, D3, GAMMA_T, 0.5)
check("自测：λ = 0.5（γλ = 0.45）：Â₁ = 0.07 + 0.45 × 0.7 = 0.385，Â₀ = −0.04 + 0.45 × 0.385 = 0.13325；展开 −0.04 + 0.0315 + 0.14175 也是",
      np.allclose(A3_half[:2], [0.13325, 0.385]) and round(0.07 + 0.45 * 0.7, 10) == 0.385
      and round(-0.04 + 0.45 * 0.385, 10) == 0.13325 and round(-0.04 + 0.0315 + 0.14175, 10) == 0.13325
      and round(0.45 * 0.07, 10) == 0.0315 and round(0.45 ** 2 * 0.7, 10) == 0.14175 and round(0.45 ** 2, 10) == 0.2025)
A3_l0, RET3_l0 = gae(R3, V3, 0.0, D3, GAMMA_T, 0.0)
A3_l1, RET3_l1 = gae(R3, V3, 0.0, D3, GAMMA_T, 1.0)
print(f"λ = 0：Â = {A3_l0}（就是 δ）；λ = 1：Â₀ = {num(A3_l1[0])} = G₀ − V₀（中间的 V 两两抵消）")
check("λ = 0 时 Â = δ；λ = 1 时 Â₀ = −0.04 + 0.9 × 0.07 + 0.81 × 0.7 = 0.59 = G₀ − V₀（望远镜消去）",
      np.allclose(A3_l0, d3) and math.isclose(A3_l1[0], G3[0] - V3[0])
      and round(-0.04 + 0.9 * 0.07 + 0.81 * 0.7, 10) == 0.59)
table(["t", "λ = 0：TD 目标", f"λ = {num(LAM_T)}：returns", "λ = 1：回报 G"],
      [[t, num(RET3_l0[t], 5), num(RET3[t], 5), num(RET3_l1[t], 5)] for t in range(3)])
check("returns = Â + V：0.87328、0.974、1（critic 的标签）", np.allclose(RET3, [0.87328, 0.974, 1.0])
      and round(0.37328 + 0.5, 10) == 0.87328 and round(0.574 + 0.4, 10) == 0.974)
check("λ = 0 的 returns 就是 TD 目标 r + γV(s')：0.46、0.47、1；λ = 1 的就是回报 G：1.09、1.1、1",
      np.allclose(RET3_l0, R3 + GAMMA_T * np.array(v_next3)) and np.allclose(RET3_l0, [0.46, 0.47, 1.0])
      and np.allclose(RET3_l1, G3))
# 项目的数
GAMMA, LAM = 0.99, 0.95
GL = GAMMA * LAM
print(f"项目：γλ = 0.99 × 0.95 = {num(GL)}；{num(GL)}¹¹ = {GL ** 11:.4f}，{num(GL)}¹² = {GL ** 12:.4f}，"
      f"{num(GL)}²³ = {GL ** 23:.4f}；只有 γ 时 0.99⁶⁸ = {0.99 ** 68:.4f}，0.99⁶⁹ = {0.99 ** 69:.4f}")
check("项目 γλ = 0.9405：往后第 11 步的 δ 还剩 0.5093，第 12 步 0.4790；第 23 步 0.2439",
      round(GL, 10) == 0.9405 and round(GL ** 11, 4) == 0.5093 and round(GL ** 12, 4) == 0.4790
      and round(GL ** 23, 4) == 0.2439)
check("只有 γ（λ = 1）时，要到第 69 步才打对折：0.99⁶⁸ = 0.5049 > 0.5 > 0.99⁶⁹ = 0.4998（第 9 章 9.8 节）",
      round(0.99 ** 68, 4) == 0.5049 and round(0.99 ** 69, 4) == 0.4998)

# ---------------------------------------------------------------------------
banner("3b. 画图：figures/ch12_gae_weights.png（n 步是硬切，GAE 是慢慢淡出）")
fig = plt.figure(figsize=(10.6, 13.4))
fig.suptitle("GAE：后面的 δ 不是硬切掉，而是一步一步淡出", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
ax = fig.add_axes([0.10, 0.625, 0.87, 0.26])
data_axes(ax, "", "乘在 δ 上的权重")
groups = [("看 1 步", [1, 0, 0], nstep_adv[0], BLUE), ("看 2 步", [1, 0.9, 0], nstep_adv[1], BLUE),
          ("看到终点", [1, 0.9, 0.81], nstep_adv[2], BLUE),
          (f"GAE，λ = {num(LAM_T)}", [1, GL_T, GL_T ** 2], A3[0], ORANGE)]
xt, xl = [], []
for gi, (name, ws, est, col) in enumerate(groups):
    for j, w in enumerate(ws):
        x = gi * 4.2 + j * 1.15
        ax.bar(x, w, width=0.95, color=col, alpha=0.85 if w else 0.0, edgecolor=col, linewidth=1.2)
        if w == 0:
            ax.plot([x - 0.4, x + 0.4], [0.004, 0.004], color=col, lw=2)
        ax.text(x, w + 0.03, num(w), ha="center", va="bottom", fontsize=14, color=INK)
        xt.append(x)
        xl.append(f"δ{sub(j)}")
    ax.text(gi * 4.2 + 1.15, -0.13, name, ha="center", va="top", fontsize=FS_SMALL, color=col, fontweight="bold")
    ax.text(gi * 4.2 + 1.15, -0.25, f"估出 {m(est, 5)}", ha="center", va="top", fontsize=FS_SMALL, color=INK)
ax.set_xticks(xt)
ax.set_xticklabels(xl)
ax.set_ylim(0, 1.18)
ax.set_xlim(-0.8, 3 * 4.2 + 3.1)
panel_title(fig, [ax], "① 三步账本（γ = 0.9）：每种估计给三个 δ 各乘多少")
panel_note(fig, [ax], f"n 步：前 n 个 δ 乘 γ 的次方，之后一刀切成 0。GAE：每往后一步再多乘一次 λ，\n"
                      f"权重 1、{num(GL_T)}、{num(GL_T ** 2)} 慢慢变小，估出来的数落在 −0.04 和 0.59 之间。")
ax = fig.add_axes([0.10, 0.10, 0.87, 0.29])
data_axes(ax, "往后第 k 步", "δ 的权重")
ks = np.arange(0, 81)
ax.plot(ks, GAMMA ** ks, color=FAINT, lw=3, label="λ = 1：只乘 γ = 0.99")
ax.plot(ks, GL ** ks, color=ORANGE, lw=3.2, label="λ = 0.95：乘 γλ = 0.9405")
ax.plot([0], [1], "o", color=BLUE, ms=11, zorder=6, label="λ = 0：只有第 0 步")
ax.axhline(0.5, color=MUTED, ls=":", lw=1.6)
ax.axvline(23, color=GREEN, ls="--", lw=2)
ax.text(23.8, 0.93, "24 步一批的最后一步（k = 23）", color=GREEN, fontsize=15, va="center")
ax.plot([11], [GL ** 11], "o", color=ORANGE, ms=9, zorder=6)
ax.text(12.2, GL ** 11 + 0.05, f"k = 11：{GL ** 11:.4f}", color=ORANGE, fontsize=15, bbox=WHITE_BOX)
ax.plot([23], [GL ** 23], "o", color=ORANGE, ms=9, zorder=6)
ax.text(24.2, GL ** 23 + 0.02, f"{GL ** 23:.4f}", color=ORANGE, fontsize=15, bbox=WHITE_BOX)
ax.plot([69], [GAMMA ** 69], "o", color=MUTED, ms=9, zorder=6)
ax.text(69, GAMMA ** 69 + 0.07, f"k = 69：{GAMMA ** 69:.4f}", color=MUTED, fontsize=15, ha="center", bbox=WHITE_BOX)
ax.set_xlim(-2, 82)
ax.set_ylim(0, 1.08)
ax.legend(loc="upper right", fontsize=15, frameon=False)
panel_title(fig, [ax], "② 项目的数（γ = 0.99、λ = 0.95）：权重每步乘 0.9405")
panel_note(fig, [ax], "橙线大约 11 步降到一半，灰线（只有 γ）要 69 步（第 9 章 9.8 节）；\n"
                      "24 步一批的最后一个 δ，在第 0 步那里还剩约四分之一的权重。")
savefig(fig, "ch12_gae_weights")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("3c. 画图：figures/ch12_overview.png（12.0 节的总览）")
fig, axes = lesson_figure(4, "优势估计：每步一个 δ，从后往前串，每传一步打一次折", panel_height=3.05, width=10.6)
ax = axes[0]
lesson_panel(ax, "① 记下三步：奖励 r、critic 报的价值 V", xmax=10.6, ymax=4.0)
ax.text(0.2, 2.35, "奖励 r", fontsize=FS_SMALL, color=INK, va="center")
ax.text(0.2, 1.55, "critic 的 V", fontsize=FS_SMALL, color=INK, va="center")
lesson_cells(ax, [list(R3), list(V3)], left=2.3, bottom=1.2, width=1.5, height=0.72)
ax.plot([6.8, 6.8], [1.05, 3.05], color=INK, lw=4)
ax.text(7.0, 2.35, "回合终止", fontsize=FS_SMALL, color=INK, va="center")
ax.text(7.0, 1.55, "之后记 0", fontsize=FS_SMALL, color=MUTED, va="center")
note(ax, 0.2, 0.45, "三个数是编的，为了好算；γ 先取 0.9。")
ax = axes[1]
lesson_panel(ax, "② 每一步算一个 δ（第 10 章的 TD 误差）", xmax=10.6, ymax=4.0)
ax.text(0.2, 2.25, "δ", fontsize=FS_STEP, color=INK, va="center")
lesson_cells(ax, [[m(x) for x in d3]], left=2.3, bottom=1.9, width=1.5, height=0.72)
hand(ax, 2.3, 1.2, f"δ₀ = 0.1 + 0.9 × 0.4 − 0.5 = {m(d3[0])}", fontsize=FS_SMALL + 1)
note(ax, 0.2, 0.45, "δ = 这一步比 critic 预计的好多少（负数：差多少）。")
ax = axes[2]
lesson_panel(ax, f"③ 从后往前串：每往前传一步，乘 γλ = {num(GL_T)}", xmax=10.6, ymax=4.0)
ax.text(0.2, 1.95, "Â", fontsize=FS_STEP, color=ORANGE, va="center")
lesson_cells(ax, [[num(x, 5) for x in A3]], left=2.3, bottom=1.6, width=1.5, height=0.72, highlights={(0, 0)})
for i in (2, 1):
    arrow(ax, (2.3 + 1.5 * i + 0.55, 2.5), (2.3 + 1.5 * (i - 1) + 0.95, 2.5), ORANGE, lw=2.4,
          connectionstyle="arc3,rad=0.45")
hand(ax, 7.0, 1.95, f"Â₁ = 0.07 + {num(GL_T)} × 0.7 = {num(A3[1], 5)}", fontsize=FS_SMALL - 1, color=ORANGE)
hand(ax, 2.3, 0.98, f"Â₀ = −0.04 + {num(GL_T)} × {num(A3[1], 5)} = {num(A3[0], 5)}", fontsize=FS_SMALL + 1, color=ORANGE)
note(ax, 0.2, 0.3, "后面的好消息传回来：第 0 步的 δ 是负的，Â 却是正的。")
ax = axes[3]
lesson_panel(ax, "④ 两个出口", xmax=10.6, ymax=4.0)
hand(ax, 0.3, 2.55, f"Â + V = {num(RET3[0], 5)}、{num(RET3[1], 5)}、{num(RET3[2], 5)}", color=GREEN)
note(ax, 6.6, 2.55, "→ critic 的标签", fontsize=FS_SMALL)
hand(ax, 0.3, 1.55, "Â 整批减平均、除以标准差", color=ORANGE)
note(ax, 6.6, 1.55, "→ 交给 actor", fontsize=FS_SMALL)
note(ax, 0.3, 0.55, "critic 学“这一步值多少”，actor 学“这个动作好不好”。")
savefig(fig, "ch12_overview")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("4. 往回传：8 步的记录，一次摔倒提前算到前两步头上；回合结束就断开")
FALL = -1.0  # TWEAK-2: 0.0
R8 = np.array([0.1, 0.1, 0.1, 1.0, 0.1, 0.1, FALL, 0.1])   # 第 6 步摔倒
V8 = np.array([0.5, 0.6, 0.7, 0.4, 0.3, 0.2, 0.1, 0.3])
D8 = np.array([0, 0, 0, 1, 0, 0, 1, 0])                    # 第 3 步、第 6 步之后回合结束
V8_LAST = 0.35                                             # 攒够 8 步时，critic 对“下一步”的估计
d8 = td_errors(R8, V8, V8_LAST, D8, GAMMA)
A8, RET8 = gae(R8, V8, V8_LAST, D8, GAMMA, LAM)
A8_l0, _ = gae(R8, V8, V8_LAST, D8, GAMMA, 0.0)
A8_l1, RET8_l1 = gae(R8, V8, V8_LAST, D8, GAMMA, 1.0)
after = {3: "终止", 6: "终止", 7: "攒够 8 步"}
table(["t", "r", "V", "这一步之后", "δ"],
      [[t, num(R8[t]), num(V8[t]), after.get(t, ""), num(d8[t], 6)] for t in range(8)])
table(["t", "Â（λ = 0，就是 δ）", "Â（λ = 0.95）", "Â（λ = 1）"],
      [[t, num(A8_l0[t], 6), num(A8[t], 6), num(A8_l1[t], 6)] for t in range(8)])
print(f"Â₆ = δ₆ = {m(A8[6], 6)}（第 6 步之后回合结束，后面的不接）")
print(f"Â₅ = −0.001 + 0.9405 × ({m(A8[6], 6)}) = {m(A8[5], 6)}")
print(f"Â₄ = −0.002 + 0.9405 × ({m(A8[5], 6)}) = {m(A8[4], 6)}")
print(f"Â₃ = δ₃ = {m(A8[3], 6)}（第 3 步之后回合结束，第 6 步的摔倒传不过来）")
check("8 步记录的 δ：0.194、0.193、−0.204、0.6、−0.002、−0.001、−1.1、0.1465",
      np.allclose(d8, [0.194, 0.193, -0.204, 0.6, -0.002, -0.001, -1.1, 0.1465]))
check("GAE（λ = 0.95）：0.694216、0.531862、0.3603、0.6、−0.975935、−1.03555、−1.1、0.1465",
      np.allclose(np.round(A8, 6), [0.694216, 0.531862, 0.3603, 0.6, -0.975935, -1.03555, -1.1, 0.1465]))
check("按印出来的数重算：−0.001 + 0.9405 × (−1.1) = −1.03555；−0.002 + 0.9405 × (−1.03555) = −0.975935；"
      "−0.204 + 0.9405 × 0.6 = 0.3603",
      round(-0.001 + 0.9405 * -1.1, 10) == -1.03555 and round(-0.002 + 0.9405 * -1.03555, 6) == -0.975935
      and round(-0.204 + 0.9405 * 0.6, 10) == 0.3603)
check("按定义逐项加（碰到结束就停）= 从后往前递推", np.allclose(gae_by_sum(d8, D8, GAMMA, LAM), A8))
check("回合边界挡住往回传：Â₃ = δ₃、Â₆ = δ₆（后面的 δ 一个也没加进来）", A8[3] == d8[3] and A8[6] == d8[6])
check("第 6 步的 −1.1 传到第 4 步乘了两次 0.9405：0.9405 × 0.9405 × (−1.1) ≈ −0.973",
      round(GL * GL * d8[6], 3) == -0.973 and round(0.9405 * 0.9405 * -1.1, 3) == -0.973)
check("自测：Â₁ = 0.193 + 0.9405 × 0.3603 = 0.193 + 0.33886 = 0.53186（0.9405 × 0.3603 = 0.33886215）",
      round(0.9405 * 0.3603, 10) == 0.33886215 and round(0.193 + 0.33886, 10) == 0.53186
      and round(0.193 + 0.9405 * 0.3603, 6) == 0.531862
      and round(A8[1], 6) == 0.531862)
check("自测：Â₀ = 0.194 + 0.9405 × 0.531862 = 0.194 + 0.500216 = 0.694216，和表里一样",
      round(0.9405 * 0.531862, 6) == 0.500216 and round(0.194 + 0.500216, 6) == 0.694216 and round(A8[0], 6) == 0.694216)
G8 = discounted_returns(R8, D8, GAMMA, v_last=V8_LAST)
check("λ = 1 时 returns = 真拿到的奖励打折加到回合结束（最后一段用 V = 0.35 补上）",
      np.allclose(RET8_l1, G8))
check("第 4 步分到的摔倒：λ = 0 分不到（−0.002），λ = 0.95 是 −0.9759，λ = 1 是 −1.0811",
      round(A8_l0[4], 4) == -0.002 and round(A8[4], 4) == -0.9759 and round(A8_l1[4], 4) == -1.0811)
# 一批里有很多只机器人：迷你版 [8, 2]
RB, VB, DB, VB_LAST = np.full(8, 0.1), np.full(8, 0.3), np.zeros(8), 0.3    # 机器人 B：一直稳稳地走
R82, V82, D82 = (np.stack([a, b], axis=1) for a, b in ((R8, RB), (V8, VB), (D8, DB)))
V82_LAST = np.array([V8_LAST, VB_LAST])
A82, RET82 = gae(R82, V82, V82_LAST, D82, GAMMA, LAM)
AB, _ = gae(RB, VB, VB_LAST, DB, GAMMA, LAM)
print(f"迷你版 [8, 2]：形状 {R82.shape}，第 0 轴 8 个时刻、第 1 轴 2 只机器人；沿时间倒着循环，每一步两只一起算")
check("[8, 2] 一起算，A 那一列 = 单独算 A，B 那一列 = 单独算 B", np.allclose(A82[:, 0], A8) and np.allclose(A82[:, 1], AB))
A_wrong, _ = gae(R82.reshape(-1), V82.reshape(-1), VB_LAST, D82.reshape(-1), GAMMA, LAM)
d_wrong_a0 = td_errors(R82.reshape(-1), V82.reshape(-1), VB_LAST, D82.reshape(-1), GAMMA)[0]
print(f"错误做法：按行展开成一条 16 步的长记录 A₀, B₀, A₁, B₁, …："
      f"A 第 0 步的 δ = 0.1 + 0.99 × 0.3 − 0.5 = {m(d_wrong_a0)}（正确是 0.194），Â₀ = {m(A_wrong[0], 6)}（正确是 {m(A8[0], 6)}）")
check("展开成一条长记录，就把 B 的 V = 0.3 当成了 A 下一步的价值：δ = −0.103，Â 全错",
      round(d_wrong_a0, 10) == -0.103 and round(0.1 + 0.99 * 0.3 - 0.5, 10) == -0.103
      and not np.allclose(A_wrong[0::2], A8))

# ---------------------------------------------------------------------------
banner("4b. 画图：figures/ch12_news_backward.png（摔倒的消息往回传，碰到回合结束就停）")
fig, axes = lesson_figure(2, "一次摔倒，往回传给前两步：每传一步乘 0.9405，碰到回合结束就停", panel_height=4.0, width=13.2)
cw, left = 1.32, 1.0
walls = [t for t in range(7) if D8[t]]
fell = "摔倒" if FALL < 0 else "回合结束"          # TWEAK-2 把摔倒的 −1 改成 0：标题跟着改，不写死
for ax, title in zip(axes, [f"① 每一步的 δ：第 6 步{fell}，δ₆ = {m(R8[6])} + 0 − {num(V8[6])} = {m(d8[6])}",
                            "② 从后往前串（λ = 0.95）：Â = δ + 0.9405 × 后一步的 Â"]):
    lesson_panel(ax, title, xmax=13.2, ymax=4.3)
    for t in walls:
        x = left + (t + 1) * cw
        ax.plot([x, x], [1.6, 3.0], color=INK, lw=4.5)
    x_end = left + 8 * cw
    ax.plot([x_end + 0.06, x_end + 0.06], [1.6, 3.0], color=GREEN, lw=2.5, ls="--")
ax = axes[0]
for t in range(8):
    ax.text(left + (t + 0.5) * cw, 3.3, f"t = {t}", fontsize=15, color=MUTED, ha="center", va="center")
for t in walls:
    ax.text(left + (t + 1) * cw, 1.45, "回合结束", fontsize=15, color=INK, ha="center", va="top")
ax.text(0.25, 2.3, "δ", fontsize=FS_STEP, color=INK, va="center")
lesson_cells(ax, [[m(x, 4) for x in d8]], left=left, bottom=1.95, width=cw, height=0.72, highlights={(0, 6)},
             fontsize=16)
ax.text(left + 8 * cw + 0.25, 2.55, "下一步用", fontsize=15, color=GREEN, va="center")
ax.text(left + 8 * cw + 0.25, 2.1, f"V = {num(V8_LAST)} 补", fontsize=15, color=GREEN, va="center")
note(ax, 0.25, 0.55, "第 4、5 步只拿到 0.1，δ 几乎是 0——只看一步，它们看不出两步之后要摔。")
ax = axes[1]
ax.text(0.25, 2.3, "Â", fontsize=FS_STEP, color=ORANGE, va="center")
lesson_cells(ax, [[m(x, 4) for x in A8]], left=left, bottom=1.95, width=cw, height=0.72,
             highlights={(0, 4), (0, 5), (0, 6)}, fontsize=16)
for t in range(7, 0, -1):
    if D8[t - 1]:
        continue                                   # 回合在 t − 1 之后结束：t 的 Â 传不到 t − 1
    arrow(ax, (left + (t + 0.3) * cw, 2.72), (left + (t - 0.3) * cw, 2.72), ORANGE, lw=2.2,
          connectionstyle="arc3,rad=0.5")
hand(ax, 0.25, 1.2, f"Â₅ = −0.001 + 0.9405 × ({m(A8[6], 4)}) = {m(A8[5], 5)}", fontsize=FS_SMALL)
hand(ax, 6.75, 1.2, f"Â₄ = −0.002 + 0.9405 × ({m(A8[5], 5)}) = {m(A8[4], 6)}", fontsize=FS_SMALL)
note(ax, 0.25, 0.45, "橙色弧线是往回传的路；回合结束的墙上没有弧线：第 6 步的摔倒算不到上一局（t ≤ 3）头上。")
savefig(fig, "ch12_news_backward")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("4c. 画图：figures/ch12_gae_lambda.png（同一段记录，三个 λ）")
fig = plt.figure(figsize=(10.4, 7.4))
fig.suptitle("同一段记录：λ 越大，摔倒的消息往回传得越远", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.975)
ax = fig.add_axes([0.11, 0.25, 0.86, 0.58])
data_axes(ax, "第 t 步", "优势估计 Â")
segments = [(0, 3), (4, 6), (7, 7)]                  # 三段各属于一局：线只在一局之内连
for lam_v, arr, col, mk in ((0.0, A8_l0, BLUE, "s"), (0.95, A8, ORANGE, "o"), (1.0, A8_l1, GREEN, "^")):
    for si, (a, b) in enumerate(segments):
        ax.plot(range(a, b + 1), arr[a:b + 1], marker=mk, ms=10, lw=2.6, color=col,
                label=f"λ = {num(lam_v)}" if si == 0 else None)
for t in walls:
    ax.axvspan(t + 0.44, t + 0.56, color=CELL_EDGE, alpha=0.6, lw=0)
    ax.text(t + 0.5, 1.01, "回合结束", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=15, color=INK)
ax.axhline(0, color=MUTED, lw=1.2)
for t in (4, 5):
    ax.annotate(m(A8_l0[t], 3), (t, A8_l0[t]), xytext=(t + 0.1, A8_l0[t] + 0.13), fontsize=14, color=BLUE)
    ax.annotate(m(A8[t], 3), (t, A8[t]), xytext=(t + 0.1, A8[t] + 0.13), fontsize=14, color=ORANGE, bbox=WHITE_BOX)
    ax.annotate(m(A8_l1[t], 3), (t, A8_l1[t]), xytext=(t + 0.1, A8_l1[t] - 0.24), fontsize=14, color=GREEN, bbox=WHITE_BOX)
ax.set_xticks(ts := np.arange(8))
lo, hi = min(A8_l1.min(), A8.min(), A8_l0.min()), max(A8_l1.max(), A8.max(), A8_l0.max())
ax.set_ylim(lo - 0.4, hi + 0.25)
ax.set_xlim(-0.4, 7.4)
ax.legend(loc="upper right", fontsize=15, frameon=False)
panel_note(fig, [ax], "蓝线（λ = 0）的第 4、5 步几乎是 0：只看一步，看不到后面的摔倒。橙线（0.95）、\n"
                      "绿线（1）把摔倒提前算给了第 4、5 步。线只在一局之内连：隔着回合结束的墙，互不相干。")
savefig(fig, "ch12_gae_lambda")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("5. 偏差与方差：λ 小，平均起来偏；λ 大，一次一次很吵")
# 构造的玩具：第 0 步两种走法各选一半；大步当场多拿，但第 3 步会因为身子晃少拿 0.3。
STEP0 = {"大步": 0.2, "小步": 0.1}
BASE, WOBBLE, WOBBLE_AT, T24 = 0.1, 0.3, 3, 24
NOISE = 0.5  # TWEAK-3: 0.1
V_WALK = BASE / (1 - GAMMA)                          # 一直稳稳地走（每步平均 0.1）的价值：0.1 + 0.99 × 0.1 + … = 10
mu_big = np.full(T24, BASE)
mu_big[0], mu_big[WOBBLE_AT] = STEP0["大步"], BASE - WOBBLE
mu_small = np.full(T24, BASE)
mu_small[0] = STEP0["小步"]
mu_avg = (mu_big + mu_small) / 2                     # critic 分不清两条路时，它能学到的只有两条路的平均


def backward_values(mu, v_end):
    """每一步的平均奖励从后往前打折累加：V_t = μ_t + γ·V_{t+1}。"""
    out, running = np.zeros(len(mu)), v_end
    for t in reversed(range(len(mu))):
        running = mu[t] + GAMMA * running
        out[t] = running
    return out


V_true_big = backward_values(mu_big, V_WALK)         # 走了大步之后，每一步真正的价值
V_critic = backward_values(mu_avg, V_WALK)           # critic 报的：两条路的平均（第 1–3 步看不出走的是哪条）
A_true = V_true_big[0] - V_critic[0]                 # 起点 critic 报得准（就是两条路的平均），所以这就是大步真正的优势
delta_mean = td_errors(mu_big, V_critic, V_WALK, np.zeros(T24), GAMMA)
print(f"大步这条路上，δ 的平均：第 0 步 {m(delta_mean[0])}，第 3 步 {m(delta_mean[3])}，其余都是 0")
print(f"大步真正的优势 A = 0.05 − 0.15 × 0.99³ = {m(A_true, 6)}（大步其实比平均差）")
check("δ 的平均：第 0 步 +0.05（大步当场多拿），第 3 步 −0.15（比 critic 预计的又少拿），其余为 0",
      round(delta_mean[0], 10) == 0.05 and round(delta_mean[3], 10) == -0.15
      and np.allclose(np.delete(delta_mean, [0, 3]), 0))
check("真正的优势 A = 0.05 − 0.15 × 0.99³ = −0.0955（按印出来的 0.9703 重算也是）",
      math.isclose(A_true, 0.05 - 0.15 * 0.99 ** 3) and round(A_true, 4) == -0.0955
      and round(0.05 - 0.15 * 0.9703, 4) == -0.0955)

LAMS = [0.0, 0.5, 0.8, 0.9, 0.95, 1.0]
N_AVG = 1000                                          # 平均多少次估计
mean_exact = {lam_v: gae(mu_big, V_critic, V_WALK, np.zeros(T24), GAMMA, lam_v)[0][0] for lam_v in LAMS}
std_exact = {lam_v: NOISE * math.sqrt(sum((GAMMA * lam_v) ** (2 * k) for k in range(T24))) for lam_v in LAMS}


def f4(x) -> str:
    return f"{0.0 if abs(x) < 5e-9 else x:.4f}"      # 4 位小数；−0.0000 这种浮点残渣印成 0.0000


def total_error(lam_v, n):
    """n 次估计取平均之后，离真值一般多远：√(偏差² + (标准差/√n)²)（进阶折叠里讲）。"""
    bias = gae(mu_big, V_critic, V_WALK, np.zeros(T24), GAMMA, lam_v)[0][0] - A_true
    sd = NOISE * math.sqrt(sum((GAMMA * lam_v) ** (2 * k) for k in range(T24)))
    return math.sqrt(bias ** 2 + sd ** 2 / n)


rows = [[num(lam_v, 2), f4(mean_exact[lam_v]), f4(mean_exact[lam_v] - A_true), f4(std_exact[lam_v]),
         f4(total_error(lam_v, N_AVG))] for lam_v in LAMS]
table(["λ", "平均估成", "偏差", "一次估计的标准差", f"{N_AVG} 次平均后离真值"], rows)
# 抽样核对：200 万段 24 步的记录（分 20 块，每块 10 万段），每段只看第 0 步的 Â
rng = np.random.default_rng(12)
N_CHUNK, CHUNK = 20, 100_000
sums = {lam_v: 0.0 for lam_v in LAMS}
sqs = {lam_v: 0.0 for lam_v in LAMS}
first_chunk, group_means = {}, {lam_v: [] for lam_v in LAMS}
for c in range(N_CHUNK):
    rewards = mu_big[:, None] + NOISE * rng.standard_normal((T24, CHUNK))
    for lam_v in LAMS:
        a0 = gae(rewards, V_critic[:, None], V_WALK, np.zeros((T24, 1)), GAMMA, lam_v)[0][0]
        sums[lam_v] += a0.sum()
        sqs[lam_v] += (a0 ** 2).sum()
        group_means[lam_v].append(a0.reshape(-1, N_AVG).mean(axis=1))
        if c == 0:
            first_chunk[lam_v] = a0
n_all = N_CHUNK * CHUNK
sim_mean = {lam_v: sums[lam_v] / n_all for lam_v in LAMS}
sim_std = {lam_v: math.sqrt(sqs[lam_v] / n_all - sim_mean[lam_v] ** 2) for lam_v in LAMS}
group_means = {lam_v: np.concatenate(v) for lam_v, v in group_means.items()}
sim_total = {lam_v: float(np.sqrt(np.mean((group_means[lam_v] - A_true) ** 2))) for lam_v in LAMS}
table(["λ", "抽样：平均", "抽样：标准差", f"抽样：{N_AVG} 次平均后离真值"],
      [[num(lam_v, 2), f"{sim_mean[lam_v]:.4f}", f"{sim_std[lam_v]:.4f}", f"{sim_total[lam_v]:.4f}"] for lam_v in LAMS])
print(f"（抽了 {n_all:,} 段记录；“{N_AVG} 次平均”的那一列，是把它们每 {N_AVG} 段分一组、"
      f"{len(group_means[1.0]):,} 组的平均值离真值的方均根）")
check(f"抽样的平均、标准差和上表的精确值对得上（平均差不到 5 个“标准差/√{n_all:,}”，标准差差不到 1%）",
      all(abs(sim_mean[x] - mean_exact[x]) < 5 * std_exact[x] / math.sqrt(n_all) for x in LAMS)
      and all(abs(sim_std[x] / std_exact[x] - 1) < 0.01 for x in LAMS))
check(f"抽样的“{N_AVG} 次平均后离真值”和精确值差不到 6%",
      all(abs(sim_total[x] / total_error(x, N_AVG) - 1) < 0.06 for x in LAMS))
check("偏差随 λ 变大一路变小、标准差一路变大（λ = 0 的 0.5 就是一步奖励自己的起伏）",
      all(mean_exact[a] - A_true > mean_exact[b] - A_true for a, b in zip(LAMS, LAMS[1:]))
      and all(std_exact[a] < std_exact[b] for a, b in zip(LAMS, LAMS[1:])) and math.isclose(std_exact[0.0], NOISE))
check("λ = 0 平均估成 +0.05：连正负号都错了（真值是负的）；λ = 1 平均正好是真值",
      mean_exact[0.0] > 0 > A_true and math.isclose(mean_exact[1.0], A_true))
check("表里的数：λ = 0 偏差 0.1455、标准差 0.5；λ = 0.95 平均 −0.0748、偏差 0.0208、标准差 1.4322；λ = 1 标准差 2.1927",
      f"{mean_exact[0.0] - A_true:.4f}" == "0.1455" and f"{std_exact[0.0]:.4f}" == "0.5000"
      and f"{mean_exact[0.95]:.4f}" == "-0.0748" and f"{mean_exact[0.95] - A_true:.4f}" == "0.0208"
      and f"{std_exact[0.95]:.4f}" == "1.4322" and f"{std_exact[1.0]:.4f}" == "2.1927")
sq_sum = sum(0.9801 ** k for k in range(24))
check(f"进阶折叠：1 + 0.9801 + 0.9801² + …（24 项）= {sq_sum:.2f}，0.5 × √19.23 ≈ 2.19",
      round(sq_sum, 2) == 19.23 and round(0.5 * math.sqrt(19.23), 2) == 2.19 and round(0.99 ** 2, 10) == 0.9801)
check("正文的约数：2.1927 ÷ 31.6 ≈ 0.069，0.5 ÷ 31.6 ≈ 0.016；偏差 0.05 − (−0.0955) = 0.1455；2.19 是 0.0955 的二十多倍",
      round(2.1927 / 31.6, 3) == 0.069 and round(0.5 / 31.6, 3) == 0.016 and round(0.05 + 0.0955, 10) == 0.1455
      and 20 < 2.19 / 0.0955 < 30)
check("进阶折叠：0.1455² ≈ 0.02117，(0.5 ÷ √1000)² = 0.00025，√(0.02117 + 0.00025) ≈ 0.1464",
      round(0.1455 ** 2, 5) == 0.02117 and round((0.5 / math.sqrt(1000)) ** 2, 5) == 0.00025
      and round(math.sqrt(0.02117 + 0.00025), 4) == 0.1464)
check("√1000 ≈ 31.6；0.99³ = 0.9703，0.9405³ = 0.8319", round(math.sqrt(1000), 1) == 31.6
      and round(0.99 ** 3, 4) == 0.9703 and round(GL ** 3, 4) == 0.8319)
check("按印出来的数重算：0.05 − 0.15 × 0.8319 = −0.0748；2.1927 ÷ √1000 = 0.0693；√(0.0208² + (1.4322 ÷ √1000)²) = 0.0498",
      round(0.05 - 0.15 * 0.8319, 4) == -0.0748 and round(2.1927 / math.sqrt(1000), 4) == 0.0693
      and round(math.sqrt(0.0208 ** 2 + (1.4322 / math.sqrt(1000)) ** 2), 4) == 0.0498)
check("正文引用：平均 1000 次后 λ = 0 离真值 0.1464、λ = 0.95 是 0.0498，六行里 λ = 0.95 那一行最小",
      f"{total_error(0.0, N_AVG):.4f}" == "0.1464" and f"{total_error(0.95, N_AVG):.4f}" == "0.0498"
      and min(LAMS, key=lambda x: total_error(x, N_AVG)) == 0.95)
grid = [i / 100 for i in range(101)]
best = {n: min(grid, key=lambda x: total_error(x, n)) for n in (1, 100, 1000, 10000)}
table(["平均几次", "最好的 λ", "那时离真值", "λ = 0.95 时离真值"],
      [[f"{n:,}", num(best[n], 2), f"{total_error(best[n], n):.4f}", f"{total_error(0.95, n):.4f}"] for n in best])
check("最好的 λ 随样本变多往 1 挪：平均 1 次时 0，100 次 0.83，1000 次 0.94，10000 次 0.98",
      [best[n] for n in (1, 100, 1000, 10000)] == [0.0, 0.83, 0.94, 0.98])

# ---------------------------------------------------------------------------
banner("5b. 画图：figures/ch12_bias_variance.png（一次估计、1000 次平均、总误差）")
fig = plt.figure(figsize=(10.6, 18.0))
fig.suptitle("λ 小：平均起来偏；λ 大：一次一次很吵", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.99)
lam_color = {0.0: BLUE, 0.5: FAINT, 0.8: FAINT, 0.9: FAINT, 0.95: ORANGE, 1.0: GREEN}
N_DOTS = 2000


def strip(ax, data_by_lam, xlim):
    """每种 λ 一行：每个点是一个数，上下随机抖开一点免得叠在一起；黑竖线是这一行的平均。"""
    jitter = np.random.default_rng(3)
    for i, lam_v in enumerate(LAMS):
        vals = data_by_lam[lam_v][:N_DOTS]
        ax.scatter(vals, i + jitter.uniform(-0.3, 0.3, len(vals)), s=5, color=lam_color[lam_v], alpha=0.3,
                   linewidths=0, zorder=3)
        ax.plot([vals.mean()] * 2, [i - 0.4, i + 0.4], color=INK, lw=2.8, zorder=5)
    ax.set_yticks(range(len(LAMS)))
    ax.set_yticklabels([f"λ = {num(x)}" for x in LAMS])
    ax.set_ylim(-0.6, len(LAMS) - 0.4)
    ax.set_xlim(*xlim)
    ax.axvline(A_true, color=INK, ls="--", lw=2, zorder=4)
    ax.axvline(0, color=MUTED, ls=":", lw=1.8, zorder=4)


ax = fig.add_axes([0.14, 0.715, 0.82, 0.20])
data_axes(ax, "一次估计 Â₀", "")
half1 = 3.2 * max(std_exact.values())
strip(ax, first_chunk, (A_true - half1, A_true + half1))
panel_title(fig, [ax], "① 一次估计：六种 λ 都铺得很开，看不出谁偏")
panel_note(fig, [ax], f"每个点是一次估计（每行 {N_DOTS} 个）；黑竖线是这一行的平均，黑虚线是真值 {m(A_true, 4)}，\n"
                      "点线是 0。这么吵，六条平均线挤在一起，看不出谁偏。")
ax = fig.add_axes([0.14, 0.40, 0.82, 0.20])
data_axes(ax, f"{N_AVG} 次估计的平均", "")
lo2 = min(min(np.percentile(g, 0.5) for g in group_means.values()), A_true)
hi2 = max(max(np.percentile(g, 99.5) for g in group_means.values()), 0.0)
pad = 0.08 * (hi2 - lo2)
strip(ax, group_means, (lo2 - pad, hi2 + pad))
panel_title(fig, [ax], f"② {N_AVG} 次估计取一次平均：吵声小了 √{N_AVG} ≈ 31.6 倍，偏差一点没少")
panel_note(fig, [ax], f"每个点是 {N_AVG} 次估计的平均。偏差露出来了：λ = 0 整团在 0 的右边，平均起来说“大步更好”，\n"
                      "方向都反了；λ = 1 正对真值，却散得最开；λ = 0.95 偏一点、窄一些。")
ax = fig.add_axes([0.14, 0.085, 0.82, 0.20])
data_axes(ax, "λ", "离真值多远")
lam_grid = np.linspace(0, 1, 101)
bias_curve = [abs(gae(mu_big, V_critic, V_WALK, np.zeros(T24), GAMMA, x)[0][0] - A_true) for x in lam_grid]
sd_curve = [NOISE * math.sqrt(sum((GAMMA * x) ** (2 * k) for k in range(T24))) / math.sqrt(N_AVG) for x in lam_grid]
tot_curve = [total_error(x, N_AVG) for x in lam_grid]
ax.plot(lam_grid, bias_curve, color=ORANGE, lw=3, label="偏差（多抽也不变）")
ax.plot(lam_grid, sd_curve, color=BLUE, lw=3, label=f"标准差 ÷ √{N_AVG}")
ax.plot(lam_grid, tot_curve, color=INK, lw=3.2, label="合起来离真值多远")
b = best[N_AVG]
ax.plot([b], [total_error(b, N_AVG)], "o", color=INK, ms=11, zorder=6)
ax.annotate(f"最低点 λ = {num(b, 2)}", (b, total_error(b, N_AVG)), xytext=(b - 0.3, total_error(b, N_AVG) - 0.012),
            fontsize=15, color=INK, va="center", arrowprops=dict(arrowstyle="-", color=INK, lw=1.2), bbox=WHITE_BOX)
ax.set_xlim(0, 1.02)
ax.set_ylim(0, max(max(tot_curve), max(bias_curve)) * 1.12)
ax.legend(loc="center left", fontsize=14, frameon=False, bbox_to_anchor=(0.0, 0.52))
panel_title(fig, [ax], f"③ 平均 {N_AVG} 次之后：偏差往下走，吵声往上走，合起来有个最低点")
panel_note(fig, [ax], f"最低点的位置随“平均几次”挪：只看 1 次时是 λ = {num(best[1], 2)}，100 次时 {num(best[100], 2)}，"
                      f"10000 次时 {num(best[10000], 2)}。")
savefig(fig, "ch12_bias_variance")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("6. 三种边界：终止记 0；超时把 γV 补进奖励；24 步攒够用 critic 补")
r_to, v_to = 0.1, 0.8                                 # 超时那一步：原始奖励 0.1，critic 估 V(s_t) = 0.8
r_fixed = r_to + GAMMA * v_to
d_fix = r_fixed + 0.0 - v_to
d_nofix = r_to + 0.0 - v_to
print(f"超时那一步：rsl_rl 把奖励改成 r + γV = 0.1 + 0.99 × 0.8 = {num(r_fixed)}")
print(f"  改过之后：δ = {num(r_fixed)} + 0 − 0.8 = {m(d_fix)}；不改（当成终止）：δ = 0.1 + 0 − 0.8 = {m(d_nofix)}")
check("超时补分：0.1 + 0.99 × 0.8 = 0.892（worked_examples 冻结）", round(r_fixed, 10) == 0.892
      and round(0.1 + 0.99 * 0.8, 10) == 0.892)
check("补了：δ = 0.092，和“回合没停、下一步也值 0.8”时的 0.1 + 0.99 × 0.8 − 0.8 一样；不补：δ = −0.7",
      round(d_fix, 10) == 0.092 and round(0.1 + 0.99 * 0.8 - 0.8, 10) == 0.092 and round(d_nofix, 10) == -0.7)
RT, VT, DT = np.full(3, 0.1), np.full(3, 0.8), np.array([0, 0, 1])       # 超时前最后 3 步，最后一步到点
A_nofix, _ = gae(RT, VT, 0.0, DT, GAMMA, LAM)
A_fix, _ = gae(RT + np.array([0, 0, GAMMA * v_to]), VT, 0.0, DT, GAMMA, LAM)
table(["超时前的最后 3 步", "Â（不补，当成终止）", "Â（补上 γV）"],
      [[f"倒数第 {3 - t} 步", num(A_nofix[t], 4), num(A_fix[t], 4)] for t in range(3)])
check("不补：最后 3 步的 Â 是 −0.4407、−0.5664、−0.7，冤枉一路往回传；补上：0.2599、0.1785、0.092",
      np.allclose(np.round(A_nofix, 4), [-0.4407, -0.5664, -0.7]) and np.allclose(np.round(A_fix, 4), [0.2599, 0.1785, 0.092]))
d_fix_all = td_errors(RT + np.array([0, 0, GAMMA * v_to]), VT, 0.0, DT, GAMMA)
check("补上之后，超时前 3 步的 δ 都是 0.092（和前面每一步一样），Â 都是正的",
      np.allclose(d_fix_all, 0.092) and bool((A_fix > 0).all()))
d_ideal_07 = 0.1 + GAMMA * 0.7 - 0.8
print(f"进阶：假如下一步其实只值 0.7：理想的 δ = 0.1 + 0.99 × 0.7 − 0.8 = {m(d_ideal_07)}，rsl_rl 给的是 0.092，"
      f"差 {num(0.092 - d_ideal_07)} = 0.99 × (0.8 − 0.7)")
check("rsl_rl 拿 V(s_t) 顶替 V(s_{t+1})：两者差 0.1，δ 就差 0.99 × 0.1 = 0.099",
      round(d_ideal_07, 10) == -0.007 and round(0.092 - d_ideal_07, 10) == 0.099)
d7_as_end = R8[7] + 0.0 - V8[7]
print(f"24 步（迷你版 8 步）攒够时：δ₇ = 0.1 + 0.99 × 0.35 − 0.3 = {num(d8[7])}；要是错当成终止：0.1 − 0.3 = {m(d7_as_end)}")
check("采集边界用 critic 补：δ₇ = 0.1465；错当成终止就成了 −0.2",
      round(d8[7], 10) == 0.1465 and round(d7_as_end, 10) == -0.2 and round(0.1 + 0.99 * 0.35 - 0.3, 10) == 0.1465)

# ---------------------------------------------------------------------------
banner("6b. 画图：figures/ch12_boundaries.png（三种边界各怎么处理）")
fig, axes = lesson_figure(3, "三种边界：终止记 0，超时补进奖励，攒够 24 步用 critic 补", panel_height=3.4, width=11.6)
specs = [
    ("① 终止（摔倒）：之后真的没分了，记 0", [num(R8[4]), num(R8[5]), m(R8[6])], "wall", ("摔倒", "之后记 0"),
     f"δ₆ = {m(R8[6])} + 0 − {num(V8[6])} = {m(d8[6])}（12.4 节那段记录的第 6 步）", "这一局到此为止：Â 不从新的一局往回串。"),
    ("② 超时（20 秒到）：之后本来还有分，补进这一步的奖励", ["0.1", "0.1", "0.1 → 0.892"], "wall", ("到点", "回合重来"),
     f"0.1 + 0.99 × 0.8 = {num(r_fixed)}；δ = {num(r_fixed)} + 0 − 0.8 = {num(d_fix)}", "不补就是 0.1 − 0.8 = −0.7：好好走路也被当成出事。"),
    ("③ 攒够 24 步（采集边界）：回合没完，用 critic 补", ["0.1", "0.1", "0.1"], "dash", ("攒够", "回合继续"),
     f"δ₇ = 0.1 + 0.99 × 0.35 − 0.3 = {num(d8[7])}（8 步迷你版的最后一步）", "下一批从这里接着记；这一批的最后一步，用 critic 现估的下一步补上。"),
]
for ax, (title, vals, kind, tags, calc, text) in zip(axes, specs):
    lesson_panel(ax, title, xmax=11.6, ymax=4.0)
    ax.text(0.2, 2.25, "r", fontsize=FS_STEP, color=INK, va="center")
    x = 0.8
    for i, v in enumerate(vals):
        hot = i == 2
        w = 2.7 if "→" in v else 1.9
        cell(ax, x, 1.9, v, width=w, height=0.72, facecolor=CELL_HOT if hot else CELL,
             edgecolor=ORANGE if hot else CELL_EDGE, fontsize=FS_SMALL, color=ORANGE if hot else INK)
        x += w + 0.1
    x += 0.05
    if kind == "wall":
        ax.plot([x, x], [1.6, 2.95], color=INK, lw=5)
        ax.text(x + 0.2, 2.6, tags[0], fontsize=FS_SMALL, color=INK, va="center")
        ax.text(x + 0.2, 2.05, tags[1], fontsize=FS_SMALL - 1, color=MUTED, va="center")
        cell(ax, 9.7, 1.9, "新一局", width=1.7, height=0.72, facecolor="#eeeeee", edgecolor=FAINT, fontsize=FS_SMALL,
             color=MUTED)
        ax.text(10.55, 2.95, "不接", fontsize=FS_SMALL - 1, color=MUTED, ha="center", va="center")
    else:
        ax.plot([x, x], [1.6, 2.95], color=GREEN, lw=3, ls="--")
        ax.text(x + 0.2, 2.6, tags[0], fontsize=FS_SMALL, color=GREEN, va="center")
        ax.text(x + 0.2, 2.05, tags[1], fontsize=FS_SMALL - 1, color=MUTED, va="center")
        cell(ax, 9.7, 1.9, "V = 0.35", width=1.7, height=0.72, facecolor="#e3f1ea", edgecolor=GREEN, fontsize=FS_SMALL,
             color=GREEN)
        ax.text(10.55, 2.95, "critic 估的下一步", fontsize=FS_SMALL - 2, color=GREEN, ha="center", va="center")
    hand(ax, 0.2, 1.1, calc, fontsize=FS_SMALL + 1, color=BLUE)
    note(ax, 0.2, 0.4, text)
savefig(fig, "ch12_boundaries")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("7. 优势归一化：整批减平均、除以标准差（+ 1e-8）")
toy = np.array([0.3, 0.5, 0.7])                       # 为了好算编的三个优势
toy_std1 = float(np.std(toy, ddof=1))                 # torch 的 .std() 默认除以 n − 1（第 4 章 4.4 节的进阶折叠）
z = (toy - toy.mean()) / (toy_std1 + 1e-8)
z100 = (100 * toy - (100 * toy).mean()) / (np.std(100 * toy, ddof=1) + 1e-8)
flat = np.full(3, 0.5)
z_flat = (flat - flat.mean()) / (np.std(flat, ddof=1) + 1e-8)
with np.errstate(invalid="ignore"):
    z_flat_noeps = (flat - flat.mean()) / np.std(flat, ddof=1)
print(f"[0.3, 0.5, 0.7]：平均 {num(toy.mean())}，标准差（除以 n − 1）√((0.04 + 0 + 0.04) ÷ 2) = {num(toy_std1)} → {np.round(z, 6)}")
print(f"奖励单位放大 100 倍 [30, 50, 70] → {np.round(z100, 6)}（一模一样）")
print(f"三个一样的 [0.5, 0.5, 0.5]：标准差 0，加了 1e-8 → {np.round(z_flat, 6)}；不加 → {z_flat_noeps}")
check("手算：离平均 −0.2、0、0.2，平方加起来 0.08，÷ 2 = 0.04，开根号 0.2；(0.3 − 0.5) ÷ 0.2 = −1；[30, 50, 70] 平均 50、标准差 20",
      round(0.04 + 0 + 0.04, 10) == 0.08 and round(0.08 / 2, 10) == 0.04 and math.isclose(math.sqrt(0.04), 0.2)
      and round((0.3 - 0.5) / 0.2, 10) == -1 and math.isclose(float(np.mean(100 * toy)), 50)
      and math.isclose(float(np.std(100 * toy, ddof=1)), 20))
check("[0.3, 0.5, 0.7] → 平均 0.5、标准差 0.2 → [−1, 0, 1]；放大 100 倍结果不变",
      math.isclose(toy_std1, 0.2) and np.allclose(z, [-1, 0, 1]) and np.allclose(z100, z))
check("标准差为 0 时，分母上的 1e-8 让结果是 0 而不是 nan（不加就是 0 ÷ 0）",
      np.allclose(z_flat, 0) and np.all(np.isnan(z_flat_noeps)))
check("除以 n 的标准差是 0.1633（第 8 章也出现过这个数），和 torch 默认的 0.2 不同", round(float(np.std(toy)), 4) == 0.1633)
z123 = (np.array([1.0, 2, 3]) - 2) / (np.std([1.0, 2, 3], ddof=1) + 1e-8)
z_big = (np.array([10.0, 20, 30]) - 20) / (np.std([10.0, 20, 30], ddof=1) + 1e-8)
check("自测：[1, 2, 3] 平均 2、标准差 √((1 + 0 + 1) ÷ 2) = 1 → [−1, 0, 1]；放大 10 倍 [10, 20, 30] 还是 [−1, 0, 1]",
      np.allclose(z123, [-1, 0, 1]) and np.allclose(z_big, [-1, 0, 1]))
Z3 = (A3 - A3.mean()) / (np.std(A3, ddof=1) + 1e-8)
wrong_labels = V3 + Z3
print(f"三步账本：Â = {np.round(A3, 5)} → 归一化 {np.round(Z3, 4)}")
print(f"critic 的标签用没归一化的 Â：{np.round(RET3, 5)}；错把归一化后的加回 V：{np.round(wrong_labels, 4)}")
check("账本的 Â：平均 0.54909、标准差 0.16478；(0.37328 − 0.54909) ÷ 0.16478 ≈ −1.067",
      round(float(A3.mean()), 5) == 0.54909 and round(float(np.std(A3, ddof=1)), 5) == 0.16478
      and round((0.37328 - 0.54909) / 0.16478, 3) == -1.067)
check("账本的 Â 归一化后是 −1.067、0.1512、0.9158；错加回 V，第 0 步的标签从 0.87328 变成 0.5 + (−1.067) = −0.567",
      np.allclose(np.round(Z3, 4), [-1.067, 0.1512, 0.9158]) and round(wrong_labels[0], 3) == -0.567
      and round(0.5 + -1.067, 3) == -0.567)
torch_spec = importlib.util.find_spec("torch")
if torch_spec:
    import torch

    t_toy = torch.tensor(toy, dtype=torch.float64)
    t_z = (t_toy - t_toy.mean()) / (t_toy.std() + 1e-8)
    t_a8 = torch.tensor(A8, dtype=torch.float64)
    t_a8n = (t_a8 - t_a8.mean()) / (t_a8.std() + 1e-8)
    print(f"torch：tensor([0.3, 0.5, 0.7]).std() = {float(t_toy.std()):g}；8 步的 Â 归一化后平均 {float(t_a8n.mean()):.1e}，"
          f"标准差 {float(t_a8n.std()):g}")
    check("torch 的 .std() 默认除以 n − 1：0.2；手写的归一化和 torch 一致", math.isclose(float(t_toy.std()), 0.2)
          and np.allclose(t_z.numpy(), z))
    check("归一化后平均是 0、标准差是 1（8 步记录的 Â）", abs(float(t_a8n.mean())) < 1e-12
          and math.isclose(float(t_a8n.std()), 1.0, rel_tol=1e-6))
else:
    print("  （没装 torch，跳过 torch 的核对）")

# ---------------------------------------------------------------------------
banner("8. 映射到项目：正文引用的常数和源码行还在不在；把 [8, 2] 直接交给 rsl_rl 算一遍")
REPO = Path(__file__).resolve().parents[3]
cfg_path = REPO / "src" / "mjlab_microduck" / "tasks" / "microduck_velocity_env_cfg.py"
CFG_LINES = ["gamma=0.99,", "lam=0.95,", "num_steps_per_env=24,"]
if cfg_path.is_file():
    cfg_text = cfg_path.read_text(encoding="utf-8")
    at = cfg_text.find("MicroduckRlCfg = RslRlOnPolicyRunnerCfg(")
    print("microduck_velocity_env_cfg.py 的 MicroduckRlCfg：", "  ".join(CFG_LINES))
    check("项目：γ = 0.99、λ = 0.95、每只机器人攒 24 步", at >= 0 and lines_in_order(cfg_text[at:], CFG_LINES))
    check("项目没有打开“每个 mini-batch 各自归一化优势”", "normalize_advantage_per_mini_batch" not in cfg_text)
    check("项目：熵奖励系数 c = 0.01（12.7 节说的损失里的另一项）", lines_in_order(cfg_text[at:], ["entropy_coef=0.01,", "gamma=0.99,"]))
else:
    print("  （没找到项目的 env cfg，跳过这一项）")
patch_path = REPO / "src" / "mjlab_microduck" / "tasks" / "mdp.py"
PATCH_LINES = ["_orig_compute_returns = _PPO.compute_returns",
               "def _safe_compute_returns(self, obs) -> None:",
               "_orig_compute_returns(self, obs)",
               "torch.nan_to_num_(st.advantages, nan=0.0, posinf=0.0, neginf=0.0)",
               "torch.nan_to_num_(st.returns,    nan=0.0, posinf=0.0, neginf=0.0)",
               "_PPO.compute_returns = _safe_compute_returns"]
if patch_path.is_file():
    check("项目的 mdp.py 给 compute_returns 包了一层：原样算完，再把 NaN / 无穷大换成 0",
          lines_in_order(patch_path.read_text(encoding="utf-8"), PATCH_LINES))
rsl_spec = importlib.util.find_spec("rsl_rl")
rsl_root = Path(rsl_spec.origin).parent if rsl_spec and rsl_spec.origin else None
PPO_LINES = ["lam: float = 0.95,",
             "normalize_advantage_per_mini_batch: bool = False,",
             "self.transition.values = self.critic(obs).detach()",
             "self.transition.rewards = rewards.clone()",
             "self.transition.dones = dones",
             'if "time_outs" in extras:',
             "self.transition.rewards += self.gamma * torch.squeeze(",
             'self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),',
             "st = self.storage",
             "last_values = self.critic(obs).detach()",
             "advantage = 0",
             "for step in reversed(range(st.num_transitions_per_env)):",
             "next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]",
             "next_is_not_terminal = 1.0 - st.dones[step].float()",
             "delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]",
             "advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage",
             "st.returns[step] = advantage + st.values[step]",
             "st.advantages = st.returns - st.values",
             "if not self.normalize_advantage_per_mini_batch:",
             "st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)"]
STORAGE_LINES = ["self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)",
                 "self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()",
                 "self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)",
                 "self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)",
                 "self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)"]
RUNNER_LINES = ["self.alg.process_env_step(obs, rewards, dones, extras)", "self.alg.compute_returns(obs)"]
if rsl_root and (rsl_root / "algorithms" / "ppo.py").is_file():
    ppo_text = (rsl_root / "algorithms" / "ppo.py").read_text(encoding="utf-8")
    print("rsl_rl/algorithms/ppo.py：act() 存 V(s_t) → process_env_step() 超时补 γV → compute_returns() 倒着算 δ、Â、returns，再归一化 Â")
    check(f"正文映射块引用的 ppo.py 源码行（{len(PPO_LINES)} 行，含默认 λ = 0.95）都原样存在，且顺序一致",
          lines_in_order(ppo_text, PPO_LINES) and ppo_text.find("    def act(") < ppo_text.find(PPO_LINES[2])
          < ppo_text.find("    def process_env_step(") < ppo_text.find(PPO_LINES[5])
          < ppo_text.find("    def compute_returns(") < ppo_text.find(PPO_LINES[8]))
    body = ppo_text[ppo_text.find("    def compute_returns("):ppo_text.find("    def update(")].splitlines()[1:]
    code_lines = [x for x in body if x.strip() and not x.strip().startswith(("#", '"""'))]
    check(f"compute_returns() 去掉注释、空行和说明文字，正好 12 行代码（现在 {len(code_lines)} 行）", len(code_lines) == 12)
    check("损失 = surrogate + value_loss_coef × value_loss − entropy_coef × 熵：熵那一项不跟着 Â 的大小变（12.7 节）",
          "loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()" in ppo_text)
    try:
        import importlib.metadata as _md
        rsl_version = _md.version("rsl-rl-lib")
    except Exception:
        rsl_version = "?"
    check(f"装的 rsl_rl 是 5.0.1（现在 {rsl_version}）", rsl_version == "5.0.1")
    storage_text = (rsl_root / "storage" / "rollout_storage.py").read_text(encoding="utf-8")
    check("存储的形状：奖励、done、价值、returns、优势都是 [24, 4096, 1]（时间、机器人、1 个数）",
          lines_in_order(storage_text, STORAGE_LINES))
    runner_text = (rsl_root / "runners" / "on_policy_runner.py").read_text(encoding="utf-8")
    check("runner 先一步一步 process_env_step，攒够后再 compute_returns", lines_in_order(runner_text, RUNNER_LINES))
else:
    print("  （当前 Python 环境里没有 rsl_rl，跳过源码核对；用 uv run 运行就会核对）")
mj_spec = importlib.util.find_spec("mjlab")
mj_root = Path(mj_spec.origin).parent if mj_spec and mj_spec.origin else None
if mj_root and (mj_root / "rl" / "vecenv_wrapper.py").is_file():
    wrap_text = (mj_root / "rl" / "vecenv_wrapper.py").read_text(encoding="utf-8")
    env_text = (mj_root / "envs" / "manager_based_rl_env.py").read_text(encoding="utf-8")
    src_dir = REPO / "src"
    proj_sets_horizon = src_dir.is_dir() and any("is_finite_horizon" in p.read_text(encoding="utf-8") for p in src_dir.rglob("*.py"))
    check("mjlab 每一步都把 time_outs 放进 extras：is_finite_horizon 默认 False，项目没改它",
          "if not self.cfg.is_finite_horizon:" in wrap_text and "is_finite_horizon: bool = False" in env_text
          and not proj_sets_horizon)
    mjcfg_text = (mj_root / "rl" / "config.py").read_text(encoding="utf-8")
    check("mjlab：dones = 终止或超时；超时另记在 time_outs 里；mjlab 的默认 λ 也是 0.95",
          lines_in_order(wrap_text, ["term_or_trunc = terminated | truncated", "dones = term_or_trunc.to(dtype=torch.long)",
                                     'extras["time_outs"] = truncated']) and "lam: float = 0.95" in mjcfg_text)
else:
    print("  （当前 Python 环境里没有 mjlab，跳过这一项）")
if rsl_root and torch_spec:
    try:
        from rsl_rl.algorithms.ppo import PPO
    except Exception as err:   # 缺依赖时只跳过这一项
        PPO = None
        print(f"  （导入 rsl_rl 的 PPO 失败：{type(err).__name__}，跳过这一项）")
    if PPO is not None:
        f64 = dict(dtype=torch.float64)
        st = SimpleNamespace(num_transitions_per_env=8,
                             rewards=torch.tensor(R82[..., None], **f64), values=torch.tensor(V82[..., None], **f64),
                             dones=torch.tensor(D82[..., None], dtype=torch.uint8),
                             returns=torch.zeros(8, 2, 1, **f64), advantages=torch.zeros(8, 2, 1, **f64))
        fake = SimpleNamespace(storage=st, critic=lambda obs: torch.tensor(V82_LAST[:, None], **f64),
                               gamma=GAMMA, lam=LAM, normalize_advantage_per_mini_batch=False)
        PPO.compute_returns(fake, None)                  # 就是训练时那一个函数，只是存储换成了 [8, 2, 1] 的迷你版
        raw = (st.returns - st.values).squeeze(-1).numpy()
        normed = st.advantages.squeeze(-1).numpy()
        want = (A82 - A82.mean()) / (np.std(A82, ddof=1) + 1e-8)
        gap = max(np.abs(raw - A82).max(), np.abs(st.returns.squeeze(-1).numpy() - RET82).max(), np.abs(normed - want).max())
        print(f"rsl_rl 的 compute_returns 算 [8, 2]：A 的 Â₀ = {raw[0, 0]:.6f}，Â₄ = {raw[4, 0]:.6f}；B 的 Â₀ = {raw[0, 1]:.6f}"
              f"（和手写的 gae() 最多差 {gap:.0e}：rsl_rl 有一步用 32 位小数算，末几位有舍入）")
        check("rsl_rl 的 compute_returns 和本章手写的 gae()：returns、Â 相同（差不到 1e-6）",
              np.allclose(raw, A82, atol=1e-6, rtol=0) and np.allclose(st.returns.squeeze(-1).numpy(), RET82, atol=1e-6, rtol=0))
        check("它最后整批（16 个数）一起归一化，和第 7 节的做法一样（除以 n − 1 的标准差，加 1e-8）",
              np.allclose(normed, want, atol=1e-6, rtol=0))

done()
