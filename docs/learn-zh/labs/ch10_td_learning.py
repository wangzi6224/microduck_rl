"""第 10 章实验：价值、贝尔曼方程、TD 学习、动作价值与优势，以及 critic 的一步训练。

运行：uv run python docs/learn-zh/labs/ch10_td_learning.py
纯 CPU，numpy + matplotlib；第 5 节顺手用 torch 求一次导数（没装 torch 就跳过那一项）。第 6 节只读几份源码的文字，不训练。
走廊游戏和第 9 章共用：5 格（0–4），终点是第 4 格；动作 ±1；80% 按指令走，20% 打滑走反；
走进终点得 +1，否则每步 −0.05。本章 γ = 0.9，固定策略：70% 向右、30% 向左。
小节编号与正文一一对应：实验第 K 节 = 正文 10.K 节（第 6 节对应「映射到项目」）。
第 3 节另外换 10 组随机数把走廊上的 TD 重跑一遍（约 1 秒），核对“后一半回合的平均偏高”不是运气。
正文“改一改”要改的三行都带 `# TWEAK-k:` 标记（开头的策略、第 3 节走廊的 α、第 4 节的策略概率各一处）。
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

from _common import banner, check, done, lines_in_order, savefig, table
from _draw import (BLUE, CELL, CELL_EDGE, CELL_HOT, FS_NOTE, FS_SMALL, FS_STEP, FS_TITLE, GREEN, INK, MUTED,
                   ORANGE, WHITE_BOX, arrow, cell, data_axes, hand, lesson_cells, lesson_figure, lesson_panel, note,
                   panel_note, panel_title, plt)

# ---------------------------------------------------------------------------
# 走廊游戏（和第 9 章共用，设定不改）
N, GOAL = 5, 4                        # 格子 0–4，第 4 格是终点
SLIP = 0.2                            # 20% 打滑：往指令的反方向走
GOAL_REWARD, STEP_REWARD = 1.0, -0.05
GAMMA = 0.9                           # 本章的折扣 γ（项目里是 0.99，见第 6 节）
# 固定策略：每一格都以这个概率选“向右”。“改一改”第 1 条改这一行。
P_RIGHT = 0.7  # TWEAK-1: 1.0


def next_cell(s, move):
    return min(max(s + move, 0), N - 1)          # 撞墙就停在原地：第 0 格往左还是第 0 格


def reward_into(s2):
    return GOAL_REWARD if s2 == GOAL else STEP_REWARD


def step(s, a, rng):
    move = a if rng.uniform() < 1 - SLIP else -a
    s2 = next_cell(s, move)
    return s2, reward_into(s2), s2 == GOAL


def policy(rng):
    return +1 if rng.uniform() < P_RIGHT else -1


def discounted_return(rewards):
    G = 0.0
    for r in reversed(rewards):                   # 从最后一步往前：G = r + γ × G（JS 的 reduceRight）
        G = r + GAMMA * G
    return G


def run_episode(s0, rng, max_steps=200):
    s, rewards = s0, []
    for _ in range(max_steps):
        s, r, finished = step(s, policy(rng), rng)
        rewards.append(r)
        if finished:
            break
    return rewards


def solve_exact():
    """完整的贝尔曼方程 V(s) = Σ_a π(a|s) Σ_s' P(s'|s,a)[r + γV(s')]，写成方程组 (I − γM)V = b 直接解。"""
    P = np.zeros((N, 2, N))
    for s in range(GOAL):
        for ai, a in enumerate((-1, +1)):
            for move, prob in ((a, 1 - SLIP), (-a, SLIP)):
                P[s, ai, next_cell(s, move)] += prob
    pi = np.array([1 - P_RIGHT, P_RIGHT])
    M, b = np.zeros((N, N)), np.zeros(N)
    for s in range(GOAL):
        for ai in range(2):
            for s2 in range(N):
                M[s, s2] += pi[ai] * P[s, ai, s2]
                b[s] += pi[ai] * P[s, ai, s2] * reward_into(s2)
    return np.linalg.solve(np.eye(N) - GAMMA * M, b)


V_exact = solve_exact()                           # 第 2 节再用两种办法核对它；1b 的图先借它画一条参考线


def corridor(ax, x0, y0, values, w=1.5, h=0.8, fmt="{:.3f}", hot=(), names=True):
    """画一条 5 格走廊：格子上方写第几格，格里写数；终点格绿边，hot 里的格子橙色。"""
    for s in range(N):
        x = x0 + s * w
        edge = GREEN if s == GOAL else (ORANGE if s in hot else CELL_EDGE)
        ax.add_patch(plt.Rectangle((x, y0), w, h, facecolor=CELL_HOT if s in hot else CELL, edgecolor=edge,
                                   lw=2.4 if (s == GOAL or s in hot) else 1.2))
        text = values[s] if isinstance(values[s], str) else fmt.format(values[s]).replace("-", "−")
        ax.text(x + w / 2, y0 + h / 2, text, fontsize=FS_STEP, ha="center", va="center", color=INK)
        if names:
            ax.text(x + w / 2, y0 + h + 0.1, "终点" if s == GOAL else f"第 {s} 格", fontsize=FS_SMALL,
                    ha="center", va="bottom", color=GREEN if s == GOAL else MUTED)


def box(ax, x, y, text, w=1.7, h=0.8, edge=CELL_EDGE, face=CELL):
    cell(ax, x, y, text, width=w, height=h, facecolor=face, edgecolor=edge)


def f3(v):
    return f"{v:.3f}".replace("-", "−")


# ---------------------------------------------------------------------------
banner("1. 价值：一个回合一个回报；从同一格出发跑很多回合，回报取平均（蒙特卡洛）")
episode_one = [1.0]                               # 第 3 格 → 终点：一步就到
episode_two = [-0.05, -0.05, 1.0]                 # 第 3 格 → 第 2 格（打滑）→ 第 3 格 → 终点
G_one, G_two = discounted_return(episode_one), discounted_return(episode_two)
backward, running = [], 0.0
for r in reversed(episode_two):
    running = r + GAMMA * running
    backward.append(running)
print("回合一：奖励 [1]，回报 = 1")
print("回合二：奖励 [−0.05, −0.05, 1]，从后往前：1 → −0.05 + 0.9 × 1 = 0.85 → −0.05 + 0.9 × 0.85 = 0.715")
print(f"两个回合的平均：(1 + 0.715) / 2 = {(G_one + G_two) / 2:g}")
check("回合一的回报是 1", G_one == 1.0)
check("回合二从后往前：1 → 0.85 → 0.715（每个中间数都是“从那一步起的回报”）", np.allclose(backward, [1.0, 0.85, 0.715]))
check("回合二按定义逐项加：−0.05 + 0.9 × (−0.05) + 0.81 × 1 = 0.715",
      math.isclose(-0.05 + 0.9 * -0.05 + 0.81 * 1, 0.715) and math.isclose(G_two, 0.715))
check("两个回合的平均 (1 + 0.715) / 2 = 0.8575", math.isclose((G_one + G_two) / 2, 0.8575))
check("自测：从第 2 格出发、两步没打滑，奖励 [−0.05, 1]，回报 −0.05 + 0.9 × 1 = 0.85 = 回合二倒着算的中间数",
      math.isclose(discounted_return([-0.05, 1.0]), 0.85) and math.isclose(backward[1], 0.85))

N_MC = 3000
rng_mc = np.random.default_rng(0)
mc_eps = {s: [run_episode(s, rng_mc) for _ in range(N_MC)] for s in range(GOAL)}
mc_returns = {s: np.array([discounted_return(ep) for ep in mc_eps[s]]) for s in range(GOAL)}
mc_steps = {s: np.array([len(ep) for ep in mc_eps[s]]) for s in range(GOAL)}   # 走了几步才到终点：10.2 节末尾要用
V_mc = np.array([mc_returns[s].mean() for s in range(GOAL)] + [0.0])
se_mc = np.array([mc_returns[s].std() / math.sqrt(N_MC) for s in range(GOAL)] + [0.0])
print(f"\n从每一格出发各跑 {N_MC} 个回合，回报取平均：")
table(["出发的格子", "回报的平均", "一般差多少 σ/√n", "平均走几步到终点"],
      [[s, V_mc[s], se_mc[s], mc_steps[s].mean() if s < GOAL else 0.0] for s in range(N)], floatfmt=".3f")
check(f"从第 0 格出发平均走 {mc_steps[0].mean():.2f} 步才到终点（正文的 11 步；70% 向右，路上来回蹭）",
      round(float(mc_steps[0].mean())) == 11)
check("越靠近终点，回报的平均越高（第 0 格 < 第 1 格 < 第 2 格 < 第 3 格）", bool(np.all(np.diff(V_mc[:GOAL]) > 0)))
check("终点格记 0：到了就结束，之后没有奖励", V_mc[GOAL] == 0.0)
check("3 位小数：0.138、0.265、0.452、0.758；σ/√n 都不超过 0.007",
      [round(float(v), 3) for v in V_mc[:GOAL]] == [0.138, 0.265, 0.452, 0.758] and bool(np.all(np.round(se_mc, 3) <= 0.007)))

# ---------------------------------------------------------------------------
banner("1b. 画图：figures/ch10_mc_value.png（一个回合一个回报 → 很多回合取平均）")
fig = plt.figure(figsize=(9.8, 13.2))
fig.suptitle("价值：从这一格出发，很多回合回报的平均", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
ax = fig.add_axes([0.04, 0.69, 0.93, 0.265])
lesson_panel(ax, "① 同样从第 3 格出发，两个回合的回报不一样", xmax=10, ymax=5.4)
corridor(ax, 1.4, 3.35, ["", "", "", "起点", ""], w=1.4, h=0.75, hot=(3,))
hand(ax, 0.1, 2.55, "回合一：第 3 格 → 终点。奖励 [1]，回报 = 1")
hand(ax, 0.1, 1.85, "回合二：第 3 格 → 第 2 格（打滑）→ 第 3 格 → 终点", color=GREEN)
hand(ax, 0.1, 1.2, "  奖励 [−0.05, −0.05, 1]，从后往前：1 → 0.85 → 0.715", color=GREEN)
note(ax, 0.1, 0.45, "路上打不打滑是随机的，所以回报也是随机的：同一格出发，这次 1，下次 0.715。")

ax2 = fig.add_axes([0.14, 0.455, 0.82, 0.19])
data_axes(ax2, "回合数", "回报的平均")
n_axis = np.arange(1, N_MC + 1)
running_mean = np.cumsum(mc_returns[3]) / n_axis
ax2.plot(n_axis, running_mean, color=BLUE, lw=2.4, zorder=4)
ax2.axhline(V_exact[3], color=GREEN, ls="--", lw=2.2, zorder=3)
ax2.set_xlim(0, N_MC)
ax2.set_ylim(0.55, 1.05)
ax2.text(N_MC * 0.62, V_exact[3] - 0.07, f"精确值 {f3(V_exact[3])}（10.2 节算）", color=GREEN, fontsize=FS_SMALL, bbox=WHITE_BOX)
ax2.text(N_MC * 0.98, running_mean[-1] + 0.05, f"{N_MC} 个回合的平均 {f3(running_mean[-1])}", color=BLUE,
         fontsize=FS_SMALL, ha="right", bbox=WHITE_BOX)
panel_title(fig, [ax2], f"② 从第 3 格出发跑 {N_MC} 个回合：回报的平均越来越稳")
panel_note(fig, [ax2], "前几十个回合，平均忽高忽低；回合越多，平均越稳（第 4 章 4.5 节“多抽几次取平均”）。")

ax3 = fig.add_axes([0.04, 0.09, 0.93, 0.24])
lesson_panel(ax3, "③ 每一格都这样估：越靠近终点，价值越高", xmax=10, ymax=4.0)
corridor(ax3, 1.4, 1.75, list(V_mc), w=1.4, h=0.8)
note(ax3, 0.1, 0.95, f"每格 {N_MC} 个回合取平均。终点之后没有分，记 0。")
savefig(fig, "ch10_mc_value")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("2. 贝尔曼方程：这一格的价值 = 这一步的奖励 + 0.9 × 下一格的价值（按概率加权）")
# 甲、乙两格，确定的路线：甲 →(+0.1) 乙 →(+2) 终点
V_yi = 2 + GAMMA * 0
V_jia = 0.1 + GAMMA * V_yi
G_jia_direct = 0.1 + GAMMA * 2                   # 从甲出发，直接写整局回报
print(f"乙：2 + 0.9 × 0 = {V_yi:g}；甲：0.1 + 0.9 × 2 = {V_jia:g}；从甲直接写整局回报：0.1 + 0.9 × 2 = {G_jia_direct:g}")
check("乙的价值 2 + 0.9 × 0 = 2", V_yi == 2)
check("甲的价值 0.1 + 0.9 × 2 = 1.9", math.isclose(V_jia, 1.9))
check("拆成“这一步 + 0.9 × 乙的价值”，和直接写整局回报是同一个数：没有凭空加分", math.isclose(V_jia, G_jia_direct))
# 甲走一步有 20% 摔倒（这一回合就此结束，之后没有分）
P_REACH = 0.8
V_jia_risky = P_REACH * (0.1 + GAMMA * V_yi) + (1 - P_REACH) * (0.1 + GAMMA * 0)
print(f"20% 摔倒：0.8 × (0.1 + 0.9 × 2) + 0.2 × (0.1 + 0.9 × 0) = 0.8 × 1.9 + 0.2 × 0.1 = {V_jia_risky:g}")
check("贝尔曼期望 0.8 × (0.1 + 0.9 × 2) + 0.2 × 0.1 = 1.54", math.isclose(V_jia_risky, 1.54))
check("字面值：0.8 × 1.9 = 1.52，0.2 × 0.1 = 0.02，1.52 + 0.02 = 1.54",
      math.isclose(0.8 * 1.9, 1.52) and math.isclose(0.2 * 0.1, 0.02) and math.isclose(1.52 + 0.02, 1.54))
N_TWO_MC = 20000
rng_two = np.random.default_rng(2)
G_two_samples = np.where(rng_two.uniform(size=N_TWO_MC) < P_REACH, 0.1 + GAMMA * 2, 0.1)
tol_two = 4 * G_two_samples.std() / math.sqrt(N_TWO_MC)
print(f"从甲出发抽 {N_TWO_MC} 个回合（每回合的回报不是 1.9 就是 0.1），平均 = {G_two_samples.mean():.3f}")
check(f"抽样平均和 {V_jia_risky:g} 相差不到 4 × σ/√n = {tol_two:.4f}", abs(G_two_samples.mean() - V_jia_risky) < tol_two)
check("抽样平均（3 位小数）是 1.535，和 1.54 差不到 0.01",
      round(float(G_two_samples.mean()), 3) == 1.535 and abs(float(G_two_samples.mean()) - 1.54) < 0.01)

# 走廊：“选哪个动作”和“打不打滑”合起来，每一步实际往右走一格的概率
p_right = P_RIGHT * (1 - SLIP) + (1 - P_RIGHT) * SLIP
p_left = 1 - p_right
print(f"\n走廊里实际往右走一格：{P_RIGHT:g} × {1 - SLIP:g} + {1 - P_RIGHT:g} × {SLIP:g} = {p_right:.2f}；实际往左：{p_left:.2f}")
check("字面值：0.7 × 0.8 + 0.3 × 0.2 = 0.56 + 0.06 = 0.62", math.isclose(0.7 * 0.8 + 0.3 * 0.2, 0.62))
check(f"实际往右 {p_right:.2f}、往左 {p_left:.2f}（正文的 0.62、0.38）", math.isclose(p_right, 0.62) and math.isclose(p_left, 0.38))


def bellman_rhs(V, s):
    right, left = next_cell(s, +1), next_cell(s, -1)
    return p_right * (reward_into(right) + GAMMA * V[right]) + p_left * (reward_into(left) + GAMMA * V[left])


rows = [[s, V_exact[s], bellman_rhs(V_exact, s)] for s in range(GOAL)]
print("把精确值代进每一格的方程，左右两边：")
table(["格子", "左边：这一格的价值", "右边：往右 + 往左按概率加权"], rows, floatfmt=".3f")
check("四格的贝尔曼方程都成立（左边 = 右边）", all(abs(r[1] - r[2]) < 1e-12 for r in rows))
check("精确值（3 位小数）是 0.140、0.255、0.460、0.758，终点 0",
      [round(float(v), 3) for v in V_exact] == [0.140, 0.255, 0.460, 0.758, 0.0])
check("字面值：0.62 × (1 + 0.9 × 0) + 0.38 × (−0.05 + 0.9 × 0.460) = 0.758",
      round(0.62 * (1 + 0.9 * 0) + 0.38 * (-0.05 + 0.9 * 0.460), 3) == 0.758)
check("字面值逐步：0.9 × 0.460 = 0.414，−0.05 + 0.414 = 0.364，0.38 × 0.364 = 0.13832，0.62 + 0.13832 = 0.75832",
      math.isclose(0.9 * 0.460, 0.414) and math.isclose(-0.05 + 0.414, 0.364) and math.isclose(0.38 * 0.364, 0.13832)
      and math.isclose(0.62 + 0.13832, 0.75832))

# 反复代入：先全填 0；每一遍都用上一遍的数，把第 0–3 格按贝尔曼方程重算一次（终点永远是 0）
N_SWEEPS = 300
V = np.zeros(N)
sweeps = [V.copy()]
for _ in range(N_SWEEPS):
    V = np.array([bellman_rhs(V, s) for s in range(GOAL)] + [0.0])
    sweeps.append(V)
SHOW_SWEEPS = [0, 1, 2, 3, 4, 10, 30, 100]
print("\n先全填 0，一遍一遍重算：")
table(["第几遍"] + [f"第 {s} 格" for s in range(GOAL)] + ["终点"], [[k] + list(sweeps[k]) for k in SHOW_SWEEPS],
      floatfmt=".3f")
check("手算第 1 遍：第 3 格 0.62 × 1 + 0.38 × (−0.05) = 0.601，其余三格都是 −0.05",
      np.allclose(sweeps[1], [-0.05, -0.05, -0.05, 0.601, 0.0]) and math.isclose(0.38 * 0.05, 0.019)
      and math.isclose(0.62 - 0.019, 0.601))
check("字面值：第 2 遍第 2 格 0.62 × (−0.05 + 0.9 × 0.601) + 0.38 × (−0.05 + 0.9 × (−0.05)) = 0.268",
      round(0.62 * (-0.05 + 0.9 * 0.601) + 0.38 * (-0.05 + 0.9 * -0.05), 3) == 0.268 and round(sweeps[2][2], 3) == 0.268)
check("字面值：−0.05 + 0.9 × 0.601 = 0.4909，0.62 × 0.4909 − 0.38 × 0.095 = 0.304358 − 0.0361 ≈ 0.268",
      math.isclose(-0.05 + 0.9 * 0.601, 0.4909) and math.isclose(0.62 * 0.4909, 0.304358) and math.isclose(0.38 * 0.095, 0.0361)
      and round(0.304358 - 0.0361, 3) == 0.268)
check("自测：折扣改成 0.5，甲 = 0.1 + 0.5 × 2 = 1.1", math.isclose(0.1 + 0.5 * 2, 1.1))
check("第 k 遍（k = 1…4），离终点 k 格的那一格第一次变大；比它更远的格子还在变小",
      all(sweeps[k][GOAL - k] > sweeps[k - 1][GOAL - k] and all(sweeps[k][s] < sweeps[k - 1][s] for s in range(GOAL - k))
          for k in range(1, GOAL + 1)))
check("正文引用：第 3 遍第 1 格 0.067；第 0 格第 3 遍 −0.136，第 4 遍 −0.059（还是负的，但变大了）",
      round(float(sweeps[3][1]), 3) == 0.067 and round(float(sweeps[3][0]), 3) == -0.136
      and round(float(sweeps[4][0]), 3) == -0.059)
check("正文引用：第 3 格第 2 遍不升反降，0.601 → 0.584（它用的 V(2) 是上一遍的 −0.05，比 0 还差），第 3 遍回到 0.693",
      round(float(sweeps[2][3]), 3) == 0.584 and sweeps[2][3] < sweeps[1][3] and round(float(sweeps[3][3]), 3) == 0.693)
check("字面值：0.38 × (−0.05 + 0.9 × (−0.05)) = 0.38 × (−0.095) = −0.0361，0.62 − 0.0361 = 0.5839",
      math.isclose(-0.05 + 0.9 * -0.05, -0.095) and math.isclose(0.38 * 0.095, 0.0361)
      and math.isclose(0.62 - 0.0361, 0.5839) and round(0.5839, 3) == 0.584)
check("第 100 遍的 3 位小数已经和精确值一样", np.array_equal(np.round(sweeps[100], 3), np.round(V_exact, 3)))
check("第 30 遍只有第 1 格还差最后一位（0.254 对 0.255）",
      [round(float(v), 3) for v in sweeps[30]] == [0.140, 0.254, 0.460, 0.758, 0.0])
check(f"算 {N_SWEEPS} 遍，和直接解方程组（完整写法 Σπ ΣP）的结果相差不到 1e-9", np.max(np.abs(sweeps[-1] - V_exact)) < 1e-9)


def sweep_values(p_choose_right, gamma, n_sweeps):
    """从全 0 反复代入 n_sweeps 遍（和上面同一个算法，策略和折扣可以换）。"""
    pr = p_choose_right * (1 - SLIP) + (1 - p_choose_right) * SLIP
    V = np.zeros(N)
    for _ in range(n_sweeps):
        V = np.array([pr * (reward_into(next_cell(s, +1)) + gamma * V[next_cell(s, +1)])
                      + (1 - pr) * (reward_into(next_cell(s, -1)) + gamma * V[next_cell(s, -1)]) for s in range(GOAL)] + [0.0])
    return V


V_ch09 = sweep_values(1.0, 0.99, 50)
print(f"\n第 9 章的设定（总是按 →、γ = 0.99、走满 50 步就超时）：从全 0 代入 50 遍，第 0 格 = {V_ch09[0]:.4f}")
check("按第 9 章的设定代入 50 遍，第 0 格正好是那一章“总是按 →”的精确值 0.7015", round(float(V_ch09[0]), 4) == 0.7015)
# “第 k 遍 = 最多再走 k 步时的平均回报”：拿截断的回合实打实抽一遍（最多走 4 步），和第 4 遍对照
rng_cut = np.random.default_rng(4)
cut_returns = np.array([discounted_return(run_episode(0, rng_cut, max_steps=4)) for _ in range(20000)])
tol_cut = 4 * cut_returns.std() / math.sqrt(len(cut_returns))
print(f"从第 0 格出发、最多走 4 步就停：抽 20000 次，回报的平均 = {cut_returns.mean():.3f}；第 4 遍第 0 格 = {sweeps[4][0]:.3f}")
check(f"第 k 遍 = 最多再走 k 步的平均回报：最多走 4 步的抽样平均和第 4 遍相差不到 4 × σ/√n = {tol_cut:.4f}",
      abs(cut_returns.mean() - sweeps[4][0]) < tol_cut)
check("第 1 遍就是“最多走一步”：第 3 格 0.62 × 1 + 0.38 × (−0.05) = 0.601；第 0–2 格一步够不着终点，都是 −0.05（自测 ③）",
      math.isclose(0.62 * 1 + 0.38 * -0.05, 0.601) and np.allclose(sweeps[1][:GOAL], [-0.05, -0.05, -0.05, 0.601]))
check(f"收敛的道理：γ^30 = {GAMMA ** 30:.3f}，γ^60 = {GAMMA ** 60:.4f} = 0.042 × 0.042（往后的步子能加进来的分越来越少）",
      round(GAMMA ** 30, 3) == 0.042 and round(GAMMA ** 60, 4) == 0.0018 and round(0.042 * 0.042, 4) == 0.0018)
# 两套设定的差距（正文 10.2 节末尾）
check(f"同一个第 0 格：本章 {V_exact[0]:.3f}，第 9 章的设定 {V_ch09[0]:.4f}，差约五倍",
      4.5 < V_ch09[0] / V_exact[0] < 5.5)
check("字面值：0.9² = 0.81，0.81² = 0.6561（γ = 0.9 时四步后的 +1）；0.99⁴ = 0.9606",
      math.isclose(0.9 ** 2, 0.81) and math.isclose(0.81 ** 2, 0.6561) and round(0.99 ** 4, 4) == 0.9606)

print("\n10.1 节的蒙特卡洛估计和精确值比一比：")
table(["格子", "蒙特卡洛", "精确值", "差"],
      [[s, V_mc[s], V_exact[s], round(float(V_mc[s] - V_exact[s]), 3) + 0.0] for s in range(N)],   # + 0.0：不打印 −0.000
      floatfmt=".3f")
check(f"每一格的差都在 4 × σ/√n 以内（σ/√n 最大的一格是 {se_mc.max():.3f}）", bool(np.all(np.abs(V_mc - V_exact) <= 4 * se_mc)))
check(f"蒙特卡洛一共跑了 {GOAL} × {N_MC} = {GOAL * N_MC} 个回合（正文的 12000）", GOAL * N_MC == 12000)
check("差得最多的是第 1 格：0.265 − 0.255 = 0.010", int(np.argmax(np.abs(V_mc - V_exact))) == 1
      and round(float(np.max(np.abs(V_mc - V_exact))), 3) == 0.010)
ratio_1 = float(abs(V_mc[1] - V_exact[1]) / se_mc[1])
check(f"第 1 格的差是它 σ/√n 的 {ratio_1:.2f} 倍，约一倍半（字面值 0.010 ÷ 0.007 = {0.010 / 0.007:.2f}）",
      1.3 < ratio_1 < 1.7 and 1.3 < 0.010 / 0.007 < 1.7)

# ---------------------------------------------------------------------------
banner("2b. 画图：figures/ch10_bellman_ledger.png（两格的账 → 按概率加权 → 走廊的一格）")
fig, axes = lesson_figure(3, "贝尔曼方程：这一格的价值 = 这一步的奖励 + 0.9 × 下一格的价值", panel_height=3.9, width=10.4)
ax = axes[0]
lesson_panel(ax, "① 确定的路线：乙的价值 2，甲的价值 1.9", xmax=10.4, ymax=4.6)
box(ax, 0.3, 2.05, "甲")
box(ax, 4.0, 2.05, "乙")
box(ax, 7.7, 2.05, "终点", w=2.3, edge=GREEN)
arrow(ax, (2.0, 2.45), (4.0, 2.45), INK, lw=2.4)
arrow(ax, (5.7, 2.45), (7.7, 2.45), INK, lw=2.4)
ax.text(3.0, 2.95, "+0.1", fontsize=FS_STEP, ha="center", color=INK)
ax.text(6.7, 2.95, "+2", fontsize=FS_STEP, ha="center", color=INK)
hand(ax, 0.3, 1.45, "乙 = 2 + 0.9 × 0 = 2          （终点之后没有分，记 0）")
hand(ax, 0.3, 0.85, f"甲 = 0.1 + 0.9 × 2 = {V_jia:g}")
note(ax, 0.3, 0.3, "从甲直接写整局回报，也是 0.1 + 0.9 × 2 = 1.9：同一笔账，拆成两段。")
ax = axes[1]
lesson_panel(ax, "② 甲走一步有 20% 摔倒：两种未来按概率加权", xmax=10.4, ymax=4.6)
box(ax, 0.3, 1.85, "甲")
box(ax, 3.9, 2.75, "乙：2")
box(ax, 3.9, 0.95, "摔倒：0", edge=ORANGE)
arrow(ax, (2.0, 2.35), (3.9, 3.1), INK, lw=2.4)
arrow(ax, (2.0, 2.15), (3.9, 1.4), ORANGE, lw=2.4)
ax.text(2.1, 3.2, "80%，+0.1", fontsize=FS_SMALL, color=INK)
ax.text(2.1, 0.95, "20%，+0.1", fontsize=FS_SMALL, color=ORANGE)
hand(ax, 5.95, 3.3, "0.8 × (0.1 + 0.9 × 2)", fontsize=FS_NOTE)
hand(ax, 5.95, 2.6, "+ 0.2 × (0.1 + 0.9 × 0)", fontsize=FS_NOTE)
hand(ax, 5.95, 1.95, "= 0.8 × 1.9 + 0.2 × 0.1", fontsize=FS_NOTE)
hand(ax, 5.95, 1.3, f"= {V_jia_risky:g}", color=ORANGE)
note(ax, 0.3, 0.3, "期望就是按概率加权（第 4 章 4.3 节）：不是挑一条未来当成结局。")
ax = axes[2]
lesson_panel(ax, "③ 走廊的第 3 格：同一个算法", xmax=10.4, ymax=4.6)
box(ax, 0.3, 2.05, f"第 2 格：{f3(V_exact[2])}", w=2.5)
box(ax, 4.1, 2.05, "第 3 格")
box(ax, 7.7, 2.05, "终点：0", w=2.3, edge=GREEN)
arrow(ax, (4.1, 2.45), (2.8, 2.45), INK, lw=2.4)
arrow(ax, (5.8, 2.45), (7.7, 2.45), INK, lw=2.4)
ax.text(3.45, 2.95, f"{p_left:.0%}，−0.05", fontsize=FS_SMALL, ha="center", color=INK)
ax.text(6.75, 2.95, f"{p_right:.0%}，+1", fontsize=FS_SMALL, ha="center", color=INK)
hand(ax, 0.3, 1.35, f"{p_right:.2f} × (1 + 0.9 × 0) + {p_left:.2f} × (−0.05 + 0.9 × {f3(V_exact[2])}) = {f3(V_exact[3])}",
     fontsize=FS_NOTE)
note(ax, 0.3, 0.55, f"{p_right:.0%} = {P_RIGHT:g} × {1 - SLIP:g} + {1 - P_RIGHT:g} × {SLIP:g}：选了向右没打滑，或者选了向左打了滑。")
savefig(fig, "ch10_bellman_ledger")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("2c. 画图：figures/ch10_value_sweeps.png（价值从终点往回传）")
fig, axes = lesson_figure(1, "价值从终点往回传：每重算一遍，+1 的消息往回走一格", panel_height=11.6, width=10.4)
fig.subplots_adjust(top=0.935, bottom=0.02)
ax = axes[0]
lesson_panel(ax, "", xmax=10.4, ymax=11.6)
grid = [list(sweeps[k]) for k in SHOW_SWEEPS] + [list(V_exact)]
labels = [f"第 {k} 遍" for k in SHOW_SWEEPS] + ["解方程组"]
hot = [(k, GOAL - k) for k in range(1, GOAL + 1)]
left, height, width = 2.65, 0.66, 1.5
bottom = 4.4
lesson_cells(ax, grid, left=left, bottom=bottom, width=width, height=height, fmt="{:.3f}", highlights=hot, fontsize=FS_NOTE)
for i, text in enumerate(labels):
    y = bottom + (len(grid) - i - 1) * height + height / 2
    ax.text(left - 0.15, y, text, fontsize=FS_SMALL, ha="right", va="center", color=GREEN if i == len(grid) - 1 else MUTED)
for s in range(N):
    ax.text(left + (s + 0.5) * width, bottom + len(grid) * height + 0.12, "终点" if s == GOAL else f"第 {s} 格",
            fontsize=FS_SMALL, ha="center", va="bottom", color=GREEN if s == GOAL else MUTED)
ax.text(0.1, 11.25, "先全填 0，每一遍都用上一遍的数，把 0–3 格按贝尔曼方程重算一次：", fontsize=FS_NOTE, color=INK, va="center")
fg = lambda v: f"{v:g}".replace("-", "−")               # 手算行跟着“改一改”变，不手抄
hand(ax, 0.1, 3.8, f"第 1 遍，第 3 格：{p_right:.2f} × (1 + 0.9 × 0)", fontsize=FS_NOTE)
hand(ax, 0.1, 3.25, f"                         + {p_left:.2f} × (−0.05 + 0.9 × 0) = {sweeps[1][3]:.3f}", fontsize=FS_NOTE)
hand(ax, 0.1, 2.55, f"第 2 遍，第 2 格：{p_right:.2f} × (−0.05 + 0.9 × {fg(sweeps[1][3])})", fontsize=FS_NOTE, color=GREEN)
hand(ax, 0.1, 2.0, f"                         + {p_left:.2f} × (−0.05 + 0.9 × ({fg(sweeps[1][1])})) = {sweeps[2][2]:.3f}",
     fontsize=FS_NOTE, color=GREEN)
note(ax, 0.1, 0.95, "橙色格：离终点 k 格的那一格，第 k 遍第一次变大——+1 的消息刚传到。\n算到第 100 遍，3 位小数已经不再变，和直接解方程组的结果一样。",
     fontsize=FS_NOTE, linespacing=1.4)
savefig(fig, "ch10_value_sweeps")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("3. TD 学习：走一步，把估计往 TD 目标挪一点")
est_jia, est_yi, r_seen, ALPHA_HAND = 1.0, 1.5, 0.1, 0.1
td_target = r_seen + GAMMA * est_yi
td_error = td_target - est_jia
new_jia = est_jia + ALPHA_HAND * td_error
print(f"TD 目标 = 0.1 + 0.9 × 1.5 = {td_target:g}；δ = {td_target:g} − 1.0 = {td_error:g}；"
      f"新估计 = 1.0 + 0.1 × {td_error:g} = {new_jia:g}")
check("一次 TD 更新：1 + 0.1 × (0.1 + 0.9 × 1.5 − 1) = 1.045", math.isclose(new_jia, 1.045))
check("字面值：0.1 + 0.9 × 1.5 = 1.45，1.45 − 1.0 = 0.45，1.0 + 0.1 × 0.45 = 1.045",
      math.isclose(0.1 + 0.9 * 1.5, 1.45) and math.isclose(1.45 - 1.0, 0.45) and math.isclose(1.0 + 0.1 * 0.45, 1.045))

# 回到确定的路线 甲 → 乙 → 终点，反复走：每一回合先修甲、再修乙
N_TWO = 80
two = {"甲": est_jia, "乙": est_yi}
two_rows = [[0, float("nan"), est_jia, est_yi]]
for ep in range(1, N_TWO + 1):
    target_jia = 0.1 + GAMMA * two["乙"]                     # 甲的 TD 目标：一半事实（0.1），一半猜测（乙此刻的估计）
    two["甲"] += ALPHA_HAND * (target_jia - two["甲"])
    two["乙"] += ALPHA_HAND * ((2 + GAMMA * 0) - two["乙"])  # 乙的 TD 目标全是事实：2 + 0.9 × 0
    two_rows.append([ep, target_jia, two["甲"], two["乙"]])
table(["回合", "甲的 TD 目标", "甲的估计", "乙的估计"], [two_rows[k] for k in (1, 2, 3, 10, 30, 80)], floatfmt=".3f")
check("第 2 回合：甲的 TD 目标 0.1 + 0.9 × 1.55 = 1.495，甲的估计 1.045 + 0.1 × (1.495 − 1.045) = 1.09",
      math.isclose(two_rows[2][1], 1.495) and math.isclose(two_rows[2][2], 1.09)
      and math.isclose(0.1 + 0.9 * 1.55, 1.495) and math.isclose(1.045 + 0.1 * (1.495 - 1.045), 1.09))
check("自测：第 3 回合 0.1 + 0.9 × 1.595 = 1.5355，1.5355 − 1.09 = 0.4455，1.09 + 0.1 × 0.4455 = 1.13455",
      math.isclose(two_rows[3][1], 1.5355) and math.isclose(two_rows[3][2], 1.13455) and math.isclose(0.1 + 0.9 * 1.595, 1.5355)
      and math.isclose(1.5355 - 1.09, 0.4455) and math.isclose(1.09 + 0.1 * 0.4455, 1.13455))
check("甲的 TD 目标一开始偏低（1.45 < 真值 1.9），之后回合回合往上走", two_rows[1][1] < 1.9 and
      all(two_rows[k + 1][1] > two_rows[k][1] for k in range(1, N_TWO)))
check(f"走 {N_TWO} 个回合：乙的估计离 2 不到 0.001，甲的估计离 1.9 不到 0.01",
      abs(two["乙"] - 2) < 1e-3 and abs(two["甲"] - 1.9) < 1e-2)

# 走廊：从全 0 开始，每一步都修一次
ALPHA_TD = 0.1  # TWEAK-2: 0.01
N_TD = 4000


def run_td(alpha, seed, n_ep=N_TD):
    """走廊上从全 0 开始做 TD。返回：每个回合结束时记下的估计（第 0 行是开始前）、
    后一半回合里“每走一步记一次”的平均，以及有几个回合是走进终点结束的。"""
    rng = np.random.default_rng(seed)
    V = np.zeros(N)
    ends, step_sum, step_count, n_finished = [V.copy()], np.zeros(N), 0, 0
    for ep in range(n_ep):
        s = int(rng.integers(0, GOAL))                      # 每个回合从 0–3 格里随便挑一格出发
        for _ in range(200):                                # 200 步只是保险，实际远远用不到（下面核对）
            s2, r, finished = step(s, policy(rng), rng)
            target = r + (0.0 if finished else GAMMA * V[s2])  # 进了终点，下一格的价值记 0
            V[s] += alpha * (target - V[s])                    # 往 TD 目标挪 α 那么多
            s = s2
            if ep >= n_ep // 2:
                step_sum += V
                step_count += 1
            if finished:
                n_finished += 1
                break
        ends.append(V.copy())                               # 回合刚结束时记一次：下面的表、图都用它
    return np.array(ends), step_sum / step_count, n_finished


trace, step_mean, n_finished = run_td(ALPHA_TD, 1)
tail_mean = trace[N_TD // 2:].mean(axis=0)                  # 后一半回合（第 2000 个回合以后）结束时记下的估计，取平均
first_positive = [int(np.argmax(trace[:, s] > 0)) for s in range(GOAL)]
SHOW_TD = [1, 10, 100, 1000, 4000]
print(f"\n走廊上从全 0 开始做 TD（α = {ALPHA_TD:g}），走完若干回合时的估计：")
table(["回合数"] + [f"第 {s} 格" for s in range(GOAL)],
      [[k] + list(trace[k, :GOAL]) for k in SHOW_TD] + [["精确值"] + list(V_exact[:GOAL]), ["后一半的平均"] + list(tail_mean[:GOAL])],
      floatfmt=".3f")
print("每格的估计第一次变成正数，是走完第几个回合时：" + "、".join(f"第 {s} 格 {first_positive[s]}" for s in range(GOAL)))
TAIL_TOL = 0.04
check(f"TD 的估计在精确值附近晃：后一半回合的平均，每格离精确值不到 {TAIL_TOL}（α = {ALPHA_TD:g}）",
      bool(np.all(np.abs(tail_mean - V_exact) < TAIL_TOL)))
check("正文引用：第 1000 个回合第 3 格晃到 0.870，第 4000 个回合回到 0.711",
      round(float(trace[1000, 3]), 3) == 0.870 and round(float(trace[4000, 3]), 3) == 0.711)
check("正文引用：后一半的平均 0.120、0.250、0.474、0.785；第 3 格高 0.785 − 0.758 = 0.027，第 0 格低 0.120 − 0.140 = −0.020",
      [round(float(v), 3) for v in tail_mean[:GOAL]] == [0.120, 0.250, 0.474, 0.785]
      and round(float(tail_mean[3] - V_exact[3]), 3) == 0.027 and round(float(tail_mean[0] - V_exact[0]), 3) == -0.020
      and math.isclose(0.785 - 0.758, 0.027) and math.isclose(0.120 - 0.140, -0.020))
# 为什么平均起来还偏一点（正文的进阶块）
pull = ALPHA_TD * (1 - V_exact[3])
print(f"每个回合最后一步都是第 3 格走进终点（{n_finished}/{N_TD} 个回合）：TD 目标 1 把第 3 格往上拉约 "
      f"{ALPHA_TD:g} × (1 − {V_exact[3]:.3f}) = {pull:.4f}")
print(f"后一半回合里改成每走一步记一次、再取平均：第 3 格比精确值高 {step_mean[3] - V_exact[3]:.3f}，"
      f"第 0 格低 {V_exact[0] - step_mean[0]:.3f}（回合结束时记：高 {tail_mean[3] - V_exact[3]:.3f}、低 {V_exact[0] - tail_mean[0]:.3f}）")
check("每个回合都是走进终点结束的（没有一个用到 200 步的保险），而终点只能从第 3 格走进去", n_finished == N_TD)
check("字面值：0.1 × (1 − 0.758) = 0.0242 ≈ 0.024", math.isclose(0.1 * (1 - 0.758), 0.0242) and round(0.0242, 3) == 0.024)
check("正文引用：每走一步记一次再平均，第 3 格只高 0.004（回合结束时记是 0.027），第 0 格照样低 0.019",
      round(float(step_mean[3] - V_exact[3]), 3) == 0.004 and step_mean[3] - V_exact[3] < (tail_mean[3] - V_exact[3]) / 3
      and round(float(V_exact[0] - step_mean[0]), 3) == 0.019)
# α 越小，两份偏差一起缩（正文 10.3 节的进阶块）。α 小学得慢，所以回合数跟着翻倍；每档换 3 组随机数取平均
ladder = []
for alpha_k, n_ep_k in ((0.1, 4000), (0.05, 8000), (0.025, 16000)):
    runs = [run_td(alpha_k, sd, n_ep=n_ep_k) for sd in (1, 2, 3)]
    tail_k = float(np.mean([r[0][n_ep_k // 2:, 3].mean() for r in runs])) - V_exact[3]
    step_k = float(np.mean([r[1][3] for r in runs])) - V_exact[3]
    zero_k = float(np.mean([r[1][0] for r in runs])) - V_exact[0]
    ladder.append([alpha_k, n_ep_k, tail_k, step_k, tail_k - step_k, alpha_k * (1 - V_exact[3]), zero_k])
print("\nα 换三档（每档 3 组随机数取平均），两份偏差一起缩：")
table(["α", "回合数", "第 3 格回合末记", "第 3 格每步记", "两者之差", "α × (1 − 0.758)"],
      [row[:6] for row in ladder], floatfmt=".3f")
print("第 0 格“每步记”的偏差：" + "、".join(f"{row[6]:+.3f}" for row in ladder))
check("“什么时候看”那一份 ≈ α × (1 − 0.758)：三档都差不到 0.003，且随 α 一起缩小",
      all(abs(row[4] - row[5]) < 0.003 for row in ladder)
      and ladder[0][4] > ladder[1][4] > ladder[2][4] > 0)
check("α 固定那一份（第 0 格每步记）也随 α 一起缩小：三档依次变小，而且都是负的",
      ladder[0][6] < ladder[1][6] < ladder[2][6] < 0)
check(f"正文引用的三行：第 3 格回合末记 {'、'.join(f'{r[2]:.3f}' for r in ladder)}；"
      f"每步记 {'、'.join(f'{r[3]:.3f}' for r in ladder)}；两者之差 {'、'.join(f'{r[4]:.3f}' for r in ladder)}；"
      f"α × (1 − 0.758) {'、'.join(f'{r[5]:.3f}' for r in ladder)}；第 0 格每步记 {'、'.join(f'{r[6]:.3f}' for r in ladder)}",
      [round(r[2], 3) for r in ladder] == [0.028, 0.013, 0.008]
      and [round(r[3], 3) for r in ladder] == [0.006, 0.002, 0.003]
      and [round(r[4], 3) for r in ladder] == [0.022, 0.010, 0.005]
      and [round(r[5], 3) for r in ladder] == [0.024, 0.012, 0.006]
      and [round(r[6], 3) for r in ladder] == [-0.018, -0.010, -0.003])

seed_gaps = [float(run_td(ALPHA_TD, sd)[0][N_TD // 2:, 3].mean() - V_exact[3]) for sd in range(2, 12)]
print(f"换 10 组随机数重跑（α = {ALPHA_TD:g}），第 3 格后一半的平均减精确值：" + "  ".join(f"{g:+.3f}" for g in seed_gaps))
check(f"换 10 组随机数重跑，第 3 格后一半的平均每一次都高于精确值（α = {ALPHA_TD:g}）：不是运气", all(g > 0 for g in seed_gaps))
check("正文引用：10 组随机数里，第 3 格高出 0.018 到 0.040", round(min(seed_gaps), 3) == 0.018 and round(max(seed_gaps), 3) == 0.040)
check("第 3 格最先变正，然后依次是第 2、1、0 格：+1 的消息也是从终点往回传",
      0 < first_positive[3] < first_positive[2] < first_positive[1] < first_positive[0])
check("正文引用：第 3、2、1、0 格依次在第 1、3、17、42 个回合变正；第 1 个回合走完第 3 格是 0.095",
      first_positive[::-1] == [1, 3, 17, 42] and round(float(trace[1, 3]), 3) == 0.095)
check("正文引用：第 100 个回合时四格是 0.130、0.246、0.469、0.709",
      [round(float(v), 3) for v in trace[100, :GOAL]] == [0.130, 0.246, 0.469, 0.709])
check("正文引用：第 1 个回合只有第 3 格变正（0.095），另外三格还是负的；第 10 个回合第 2 格跟上（0.143）",
      round(float(trace[1, 3]), 3) == 0.095 and bool(np.all(trace[1, :3] < 0)) and round(float(trace[10, 2]), 3) == 0.143)
check("自测②：第 1000 个回合第 3 格 0.870，比精确值高 0.870 − 0.758 = 0.112；第 4000 个回合 0.711 又低了",
      math.isclose(0.870 - 0.758, 0.112) and round(float(trace[1000, 3] - V_exact[3]), 3) == 0.112
      and trace[4000, 3] < V_exact[3])

# ---------------------------------------------------------------------------
banner("3b. 画图：figures/ch10_td_learning.png（TD 目标先错后对；走廊上四格一个个学准）")
fig = plt.figure(figsize=(9.8, 13.0))
fig.suptitle("TD 学习：TD 目标用了自己的估计，估计变准，目标才跟着变准", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
ax1 = fig.add_axes([0.13, 0.575, 0.7, 0.3])
data_axes(ax1, "回合数", "估计的价值")
eps = np.arange(0, N_TWO + 1)
arr = np.array(two_rows)
ax1.axhline(2.0, color=GREEN, ls=":", lw=2)
ax1.axhline(1.9, color=BLUE, ls=":", lw=2)
ax1.plot(eps, arr[:, 3], color=GREEN, lw=2.6, label="乙的估计")
ax1.plot(eps, arr[:, 2], color=BLUE, lw=2.6, label="甲的估计")
ax1.plot(eps[1:], arr[1:, 1], color=ORANGE, lw=2.2, ls="--", label="甲的 TD 目标")
ax1.set_xlim(0, N_TWO)
ax1.set_ylim(0.95, 2.1)
ax1.text(N_TWO + 1.5, 2.0, "乙的真值 2", color=GREEN, fontsize=FS_SMALL, va="center")
ax1.text(N_TWO + 1.5, 1.88, "甲的真值 1.9", color=BLUE, fontsize=FS_SMALL, va="center")
ax1.annotate(f"第 1 回合：TD 目标 {two_rows[1][1]:g}", xy=(1, two_rows[1][1]), xytext=(12, 1.2), fontsize=FS_SMALL,
             color=ORANGE, arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.8))
ax1.legend(loc="lower right", fontsize=FS_SMALL, frameon=False)
panel_title(fig, [ax1], "① 甲 → 乙 → 终点，反复走：乙先学准，甲的 TD 目标才跟上")
panel_note(fig, [ax1], "乙的 TD 目标是 2 + 0.9 × 0，全是事实；甲的 TD 目标 0.1 + 0.9 × 乙的估计，\n一半是猜测——乙还没学准时，它就偏低。")

ax2 = fig.add_axes([0.13, 0.105, 0.7, 0.3])
data_axes(ax2, "回合数（对数刻度：1、10、100、1000 等距）", "估计的价值")
ks = np.arange(1, N_TD + 1)
colors = [MUTED, BLUE, GREEN, ORANGE]
for s in range(GOAL):
    ax2.plot(ks, trace[1:, s], color=colors[s], lw=2.0)
    ax2.axhline(V_exact[s], color=colors[s], ls=":", lw=1.8)
    ax2.text(N_TD * 1.25, V_exact[s], f"第 {s} 格 {f3(V_exact[s])}", color=colors[s], fontsize=FS_SMALL, va="center")
    if first_positive[s] > 0:
        ax2.plot([first_positive[s]], [trace[first_positive[s], s]], "o", color=colors[s], ms=11, mec="white", mew=1.5, zorder=6)
ax2.set_xscale("log")
ax2.set_xlim(0.85, N_TD)                                # 从 0.85 起：x = 1 的圆点不被左轴切掉
ax2.set_xticks([1, 10, 100, 1000])
ax2.set_xticklabels(["1", "10", "100", "1000"])
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}".replace("-", "−")))
if np.all(np.isfinite(trace)):
    ax2.set_ylim(min(-0.25, float(trace.min()) - 0.05), max(1.0, float(trace.max()) + 0.05))
panel_title(fig, [ax2], f"② 走廊上从全 0 开始，α = {ALPHA_TD:g}：离终点越近的格子，越早学准")
panel_note(fig, [ax2], f"圆点：每格的估计第一次变成正数。虚线是 10.2 节的精确值；\n学准以后，估计仍在虚线附近晃，不会停在虚线上（α 一直是 {ALPHA_TD:g}）。")
savefig(fig, "ch10_td_learning")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("4. 动作价值与优势：先做这个动作、之后照策略走，比平均好多少")
PI_RIGHT_HERE = 0.75  # TWEAK-3: 0.5
Q_here = {"向右": 2.0, "向左": 0.0}
V_here = PI_RIGHT_HERE * Q_here["向右"] + (1 - PI_RIGHT_HERE) * Q_here["向左"]
A_here = {a: q - V_here for a, q in Q_here.items()}
mean_A_here = PI_RIGHT_HERE * A_here["向右"] + (1 - PI_RIGHT_HERE) * A_here["向左"]
print(f"V = {PI_RIGHT_HERE:g} × 2 + {1 - PI_RIGHT_HERE:g} × 0 = {V_here:g}；A(向右) = 2 − {V_here:g} = {A_here['向右']:+g}，"
      f"A(向左) = 0 − {V_here:g} = {A_here['向左']:+g}；按策略加权的平均 = {mean_A_here:g}")
check("V = 0.75 × 2 + 0.25 × 0 = 1.5", math.isclose(V_here, 1.5))
check("优势：向右 2 − 1.5 = +0.5，向左 0 − 1.5 = −1.5", math.isclose(A_here["向右"], 0.5) and math.isclose(A_here["向左"], -1.5))
check("按策略加权的优势均值 0.75 × 0.5 + 0.25 × (−1.5) = 0.375 − 0.375 = 0", abs(mean_A_here) < 1e-12
      and abs(0.75 * 0.5 + 0.25 * -1.5) < 1e-12 and math.isclose(0.75 * 0.5, 0.375) and math.isclose(0.25 * 1.5, 0.375))
check("不加权、直接相加不是 0：0.5 + (−1.5) = −1", math.isclose(0.5 + -1.5, -1.0))

Q_push = {"迈一步稳住": -1.0, "硬扛不动": -5.0}          # 被推了一把：怎么做都要扣分
V_push = 0.5 * Q_push["迈一步稳住"] + 0.5 * Q_push["硬扛不动"]
A_push = {a: q - V_push for a, q in Q_push.items()}
print(f"被推了一把：V = 0.5 × (−1) + 0.5 × (−5) = {V_push:g}；A(迈一步稳住) = {A_push['迈一步稳住']:+g}，A(硬扛不动) = {A_push['硬扛不动']:+g}")
check("V = −3；迈一步稳住的优势 −1 − (−3) = +2，硬扛不动 −5 − (−3) = −2",
      V_push == -3 and A_push["迈一步稳住"] == 2 and A_push["硬扛不动"] == -2)


def q_value(V, s, a):
    want, slip = next_cell(s, a), next_cell(s, -a)
    return (1 - SLIP) * (reward_into(want) + GAMMA * V[want]) + SLIP * (reward_into(slip) + GAMMA * V[slip])


Q2 = {a: q_value(V_exact, 2, a) for a in (-1, +1)}
V2 = (1 - P_RIGHT) * Q2[-1] + P_RIGHT * Q2[+1]
A2 = {a: Q2[a] - V2 for a in (-1, +1)}
print("\n走廊第 2 格（用 10.2 节的精确值）：")
table(["动作", "策略选它的概率", "Q(2, a)", "A(2, a) = Q − V"],
      [["向左", 1 - P_RIGHT, Q2[-1], A2[-1]], ["向右", P_RIGHT, Q2[+1], A2[+1]]], floatfmt=".3f")
print(f"V(2) = {1 - P_RIGHT:g} × {Q2[-1]:.3f} + {P_RIGHT:g} × {Q2[+1]:.3f} = {V2:.3f}")
check("V(2) 就是两个 Q 按策略加权：和 10.2 节的精确值一样", abs(V2 - V_exact[2]) < 1e-12)
check("第 2 格的优势按策略加权，平均也是 0", abs((1 - P_RIGHT) * A2[-1] + P_RIGHT * A2[+1]) < 1e-12)
check("3 位小数：Q(2, 向左) = 0.270，Q(2, 向右) = 0.542，优势 −0.190 和 +0.082",
      [round(Q2[-1], 3), round(Q2[+1], 3), round(A2[-1], 3), round(A2[+1], 3)] == [0.270, 0.542, -0.190, 0.082])
check("字面值：0.8 × (−0.05 + 0.9 × 0.255) + 0.2 × (−0.05 + 0.9 × 0.758) = 0.270；"
      "0.8 × (−0.05 + 0.9 × 0.758) + 0.2 × (−0.05 + 0.9 × 0.255) = 0.542",
      round(0.8 * (-0.05 + 0.9 * 0.255) + 0.2 * (-0.05 + 0.9 * 0.758), 3) == 0.270
      and round(0.8 * (-0.05 + 0.9 * 0.758) + 0.2 * (-0.05 + 0.9 * 0.255), 3) == 0.542)
check("字面值：0.3 × 0.270 + 0.7 × 0.542 = 0.4604 ≈ 0.460；0.270 − 0.460 = −0.190；0.542 − 0.460 = 0.082",
      round(0.3 * 0.270 + 0.7 * 0.542, 4) == 0.4604 and math.isclose(0.270 - 0.460, -0.190) and math.isclose(0.542 - 0.460, 0.082))
check("字面值逐步：0.9 × 0.758 = 0.6822 → 0.6322；0.9 × 0.255 = 0.2295 → 0.1795；0.8 × 0.6322 + 0.2 × 0.1795 = 0.50576 + 0.0359 = 0.54166；"
      "0.8 × 0.1795 + 0.2 × 0.6322 = 0.1436 + 0.12644 = 0.27004；0.081 + 0.3794 = 0.4604",
      all(math.isclose(a, b) for a, b in [(0.9 * 0.758, 0.6822), (-0.05 + 0.6822, 0.6322), (0.9 * 0.255, 0.2295),
          (-0.05 + 0.2295, 0.1795), (0.8 * 0.6322, 0.50576), (0.2 * 0.1795, 0.0359), (0.50576 + 0.0359, 0.54166),
          (0.8 * 0.1795, 0.1436), (0.2 * 0.6322, 0.12644), (0.1436 + 0.12644, 0.27004), (0.3 * 0.270, 0.081),
          (0.7 * 0.542, 0.3794), (0.081 + 0.3794, 0.4604)]))

# 一次 TD 误差 δ 是优势的一次抽样：第 2 格向右，没打滑到第 3 格，打滑到第 1 格
delta_ok = STEP_REWARD + GAMMA * V_exact[3] - V_exact[2]
delta_slip = STEP_REWARD + GAMMA * V_exact[1] - V_exact[2]
print(f"第 2 格向右：没打滑 δ = {delta_ok:+.3f}，打滑 δ = {delta_slip:+.3f}；0.8 × {delta_ok:.3f} + 0.2 × ({delta_slip:.3f}) = "
      f"{0.8 * delta_ok + 0.2 * delta_slip:.3f} = A(2, 向右)")
check("δ 按 80% / 20% 加权的平均，正好等于 A(2, 向右)", abs((1 - SLIP) * delta_ok + SLIP * delta_slip - A2[+1]) < 1e-12)
check("字面值：−0.05 + 0.9 × 0.758 − 0.460 = 0.1722；−0.05 + 0.9 × 0.255 − 0.460 = −0.2805；"
      "0.8 × 0.1722 + 0.2 × (−0.2805) = 0.0817 ≈ 0.082",
      math.isclose(-0.05 + 0.9 * 0.758 - 0.460, 0.1722) and math.isclose(-0.05 + 0.9 * 0.255 - 0.460, -0.2805)
      and math.isclose(0.8 * 0.1722, 0.13776) and math.isclose(0.2 * -0.2805, -0.0561)
      and math.isclose(0.13776 - 0.0561, 0.08166) and round(0.08166, 3) == 0.082)

# ---------------------------------------------------------------------------
banner("4b. 画图：figures/ch10_q_advantage.png（Q 的条形、V 的虚线，优势 = 两者之差）")


def q_bars(ax, names, probs, qs, v, fmt, ylim):
    data_axes(ax, "", "价值")
    xs = np.arange(len(qs))
    ax.bar(xs, qs, width=0.46, color=[BLUE, GREEN], alpha=0.85, zorder=3)
    ax.axhline(0, color=CELL_EDGE, lw=1.4, zorder=2)
    ax.axhline(v, color=ORANGE, ls="--", lw=2.4, zorder=4)
    ax.text(2.02, v, f"V = {fmt(v)}", color=ORANGE, fontsize=FS_SMALL, va="bottom", ha="right", bbox=WHITE_BOX, zorder=7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}".replace("-", "−")))
    span = ylim[1] - ylim[0]
    for x, q in zip(xs, qs):
        ax.text(x, q + (0.04 * span if q >= 0 else -0.04 * span), f"Q = {fmt(q)}", ha="center",
                va="bottom" if q >= 0 else "top", fontsize=FS_SMALL, color=INK, bbox=WHITE_BOX, zorder=6)
        if abs(q - v) > 1e-9:
            arrow(ax, (x + 0.3, v), (x + 0.3, q), ORANGE, lw=2.6, zorder=6)
        ax.text(x + 0.36, (v + q) / 2, f"A = {fmt(q - v, sign=True)}", color=ORANGE, fontsize=FS_SMALL, va="center",
                bbox=WHITE_BOX, zorder=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n}\n（策略选它 {p:.0%}）" for n, p in zip(names, probs)], fontsize=FS_SMALL, color=INK)
    ax.set_xlim(-0.55, 2.05)
    ax.set_ylim(*ylim)


def fmt_g(v, sign=False):
    return (f"{v:+g}" if sign else f"{v:g}").replace("-", "−")


def fmt_3(v, sign=False):
    return (f"{v:+.3f}" if sign else f"{v:.3f}").replace("-", "−")


fig = plt.figure(figsize=(9.8, 15.4))
fig.suptitle("优势 = 动作价值 − 价值：这个动作比平均好多少", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.99)
ax = fig.add_axes([0.14, 0.715, 0.8, 0.19])
q_bars(ax, ["向右", "向左"], [PI_RIGHT_HERE, 1 - PI_RIGHT_HERE], [Q_here["向右"], Q_here["向左"]], V_here, fmt_g, (-0.4, 2.6))
panel_title(fig, [ax], f"① 编的一格：Q 是 2 和 0，按策略加权 V = {PI_RIGHT_HERE:g} × 2 + {1 - PI_RIGHT_HERE:g} × 0 = {V_here:g}")
panel_note(fig, [ax], "橙色箭头从虚线指到柱顶：朝上是正优势，朝下是负优势。\n"
           f"按策略加权：{PI_RIGHT_HERE:g} × {fmt_g(A_here['向右'])} + {1 - PI_RIGHT_HERE:g} × ({fmt_g(A_here['向左'])}) = "
           f"{fmt_g(round(mean_A_here, 12) + 0.0)}。")
ax = fig.add_axes([0.14, 0.40, 0.8, 0.19])
q_bars(ax, ["向右", "向左"], [P_RIGHT, 1 - P_RIGHT], [Q2[+1], Q2[-1]], V2, fmt_3, (0, 0.68))
panel_title(fig, [ax], "② 走廊的第 2 格：向右的优势为正，向左为负")
panel_note(fig, [ax], "两根柱都是正的：向左也能拿到正的价值，只是比平均差。")
ax = fig.add_axes([0.14, 0.085, 0.8, 0.19])
q_bars(ax, ["迈一步稳住", "硬扛不动"], [0.5, 0.5], [Q_push["迈一步稳住"], Q_push["硬扛不动"]], V_push, fmt_g, (-6.2, 0.9))
panel_title(fig, [ax], "③ 被推了一把：两个动作都要扣分，扣得少的那个优势为正")
panel_note(fig, [ax], "优势为正 = 比这个处境的平均好，不等于“这一步得正分”。")
savefig(fig, "ch10_q_advantage")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("4c. 画图：figures/ch10_overview.png（10.0 节的总览）")
fig, axes = lesson_figure(4, "一格值多少：往回算出来，边走边修，再比出哪个动作更好", panel_height=3.0, width=10.2)
ax = axes[0]
lesson_panel(ax, "① 价值：从这一格出发，很多回合回报的平均", xmax=10.2, ymax=4.0)
corridor(ax, 1.0, 1.35, list(V_exact), w=1.6, h=0.8)
note(ax, 0.1, 0.45, "越靠近终点越值钱；终点之后没有分，记 0。")
ax = axes[1]
lesson_panel(ax, "② 贝尔曼方程：这一步的奖励 + 0.9 × 下一格的价值", xmax=10.2, ymax=4.0)
box(ax, 0.3, 1.75, "甲", w=1.4, h=0.7)
box(ax, 3.6, 1.75, "乙", w=1.4, h=0.7)
box(ax, 6.9, 1.75, "终点", w=1.8, h=0.7, edge=GREEN)
arrow(ax, (1.7, 2.1), (3.6, 2.1), INK, lw=2.2)
arrow(ax, (5.0, 2.1), (6.9, 2.1), INK, lw=2.2)
ax.text(2.65, 2.5, "+0.1", fontsize=FS_SMALL, ha="center", color=INK)
ax.text(5.95, 2.5, "+2", fontsize=FS_SMALL, ha="center", color=INK)
hand(ax, 0.3, 1.15, f"甲的价值 = 0.1 + 0.9 × 乙的价值 2 = {V_jia:g}")
note(ax, 0.3, 0.45, "一整局的账，拆成眼前这一步 + 打折后的剩下。")
ax = axes[2]
lesson_panel(ax, "③ TD 学习：走一步，就往 TD 目标挪一点", xmax=10.2, ymax=4.0)
x_of = lambda v: 0.8 + (v - 0.95) / (1.55 - 0.95) * 8.6    # 1.0、1.045、1.45 按真实比例摆在数轴上
ax.plot([x_of(0.95), x_of(1.55)], [2.3, 2.3], color=CELL_EDGE, lw=2)
for v, text, color in ((est_jia, "旧估计 1.0", INK), (new_jia, f"新估计 {new_jia:g}", BLUE), (td_target, f"TD 目标 {td_target:g}", ORANGE)):
    ax.plot([x_of(v)], [2.3], "o", color=color, ms=9, zorder=5)
ax.text(x_of(est_jia), 1.8, "旧估计 1.0", fontsize=FS_SMALL, color=INK, ha="center", va="top")
ax.text(x_of(new_jia) + 0.1, 2.55, f"新估计 {new_jia:g}", fontsize=FS_SMALL, color=BLUE, ha="left")
ax.text(x_of(td_target), 2.55, f"TD 目标 {td_target:g}", fontsize=FS_SMALL, color=ORANGE, ha="center")
arrow(ax, (x_of(est_jia), 2.0), (x_of(new_jia), 2.0), BLUE, lw=2.6)
hand(ax, 3.0, 1.2, f"1.0 + 0.1 × ({td_target:g} − 1.0) = {new_jia:g}")
note(ax, 0.3, 0.45, "不用等回合结束，每走一步就修一次；只挪差距的一成。")
ax = axes[3]
lesson_panel(ax, "④ 优势：这个动作比平均好多少", xmax=10.2, ymax=4.0)
base, scale = 0.35, 1.0
for x, q, color, name in ((1.2, Q_here["向右"], BLUE, "向右"), (3.2, Q_here["向左"], GREEN, "向左")):
    ax.add_patch(plt.Rectangle((x, base), 1.0, max(q * scale, 0.02), facecolor=color, alpha=0.85, edgecolor="none"))
    ax.text(x + 0.5, base + q * scale + 0.12, fmt_g(q), ha="center", va="bottom", fontsize=FS_SMALL, color=INK)
    ax.text(x + 0.5, base - 0.08, name, ha="center", va="top", fontsize=FS_SMALL, color=MUTED)
ax.plot([0.9, 4.5], [base + V_here * scale] * 2, color=ORANGE, ls="--", lw=2.2)
ax.text(4.6, base + V_here * scale, f"平均 {V_here:g}", color=ORANGE, fontsize=FS_SMALL, va="center")
hand(ax, 6.0, 2.4, f"2 − {V_here:g} = {fmt_g(A_here['向右'], sign=True)}", color=BLUE)
hand(ax, 6.0, 1.6, f"0 − {V_here:g} = {fmt_g(A_here['向左'], sign=True)}", color=GREEN)
note(ax, 6.0, 0.8, "比平均好为正，差为负。", fontsize=FS_NOTE)
savefig(fig, "ch10_overview")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("5. critic：把 (估计 − 标签)² 往下压一步，就是 TD 的那一步")
v0, label, LR = 1.0, 1.45, 0.05                  # 估计 1.0；标签 = 10.3 节的 TD 目标 1.45；学习率 0.05
loss0 = (v0 - label) ** 2
grad_rule = 2 * (v0 - label)                     # (v − y)² 对 v 的导数 = 2(v − y)
h = 1e-6
grad_fd = ((v0 + h - label) ** 2 - (v0 - h - label) ** 2) / (2 * h)   # 第 1 章的“缩 h”
v1 = v0 - LR * grad_rule                          # 第 3 章 3.4 节的一步梯度下降
print(f"损失 (1.0 − 1.45)² = {loss0:.4f}；导数 2 × (1.0 − 1.45) = {grad_rule:g}（缩 h 量得 {grad_fd:.6f}）；"
      f"1.0 − 0.05 × ({grad_rule:g}) = {v1:g}")
check("损失 (1.0 − 1.45)² = 0.45² = 0.2025", math.isclose(loss0, 0.2025) and math.isclose(0.45 ** 2, 0.2025))
check("导数 2 × (1.0 − 1.45) = −0.9，缩 h 量出来一样", math.isclose(grad_rule, -0.9) and abs(grad_fd - grad_rule) < 1e-6)
check("学习率 0.05 走一步：1.0 − 0.05 × (−0.9) = 1.045，和 10.3 节 α = 0.1 的 TD 更新同一个数",
      math.isclose(v1, 1.045) and math.isclose(v1, new_jia) and math.isclose(0.05 * 0.9, 0.045) and math.isclose(1.0 + 0.045, 1.045))
check("对得上的原因：平方的导数多出一个 2，0.05 × 2 = 0.1 = α", math.isclose(LR * 2, ALPHA_HAND))
if importlib.util.find_spec("torch"):
    import torch

    v_t = torch.tensor(v0, requires_grad=True)
    ((v_t - label) ** 2).backward()                # 第 5 章 5.4 节的自动求导
    print(f"torch 自动求导：{float(v_t.grad):.4f}")
    check("torch 自动求导也得 −0.9", abs(float(v_t.grad) + 0.9) < 1e-6)
else:
    print("  （没装 torch，跳过自动求导那一项；用 uv run 运行就会有）")
critic_knobs = 512 * (76 + 1) + 256 * (512 + 1) + 128 * (256 + 1) + 1 * (128 + 1)
print(f"critic 的旋钮（第 6 章 6.7 节）：512 × 77 + 256 × 513 + 128 × 257 + 1 × 129 = {critic_knobs:,}")
check("critic 一共 203,777 个旋钮", critic_knobs == 203_777)

# ---------------------------------------------------------------------------
banner("6. 映射到项目：正文引用的常数和源码行还在不在")
REPO = Path(__file__).resolve().parents[3]
cfg_path = REPO / "src" / "mjlab_microduck" / "tasks" / "microduck_velocity_env_cfg.py"
CFG_LINES = ["critic=RslRlModelCfg(", "hidden_dims=(512, 256, 128),", 'activation="elu",', "obs_normalization=True,",
             "value_loss_coef=1.0,", "use_clipped_value_loss=True,", "clip_param=0.2,", "gamma=0.99,", "lam=0.95,",
             "num_steps_per_env=24,"]
if cfg_path.is_file():
    cfg_text = cfg_path.read_text(encoding="utf-8")
    at = cfg_text.find("MicroduckRlCfg = RslRlOnPolicyRunnerCfg(")
    print("microduck_velocity_env_cfg.py 的 MicroduckRlCfg：", "  ".join(CFG_LINES))
    check("critic 三层 512、256、128，ELU，有归一化；value_loss_coef 1.0；γ = 0.99；λ = 0.95；每次 24 步",
          at >= 0 and lines_in_order(cfg_text[at:], CFG_LINES))
    check("microduck 没有改回合长度（用 mjlab 模板的 20 秒）", "episode_length_s" not in cfg_text)
    check("机身速度只给 critic：actor 删掉 base_lin_vel，critic 另加一个；两边都删掉地形扫描；脚底传感器只认左右两只脚",
          lines_in_order(cfg_text, ['del cfg.observations["actor"].terms["base_lin_vel"]',
                                    'del cfg.observations["critic"].terms["height_scan"]',
                                    'cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg('])
          and 'pattern=r"^(left_foot_collision|right_foot_collision)$"' in cfg_text)
else:
    print("  （没找到项目的 env cfg，跳过这一项）")

PPO_LINES = ["self.transition.values = self.critic(obs).detach()",
             'if "time_outs" in extras:',
             "self.transition.rewards += self.gamma * torch.squeeze(",
             "last_values = self.critic(obs).detach()",
             "for step in reversed(range(st.num_transitions_per_env)):",
             "next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]",
             "next_is_not_terminal = 1.0 - st.dones[step].float()",
             "delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]",
             "values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])",
             "value_losses = (values - batch.returns).pow(2)",
             "value_loss = torch.max(value_losses, value_losses_clipped).mean()",
             "loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()"]
CRITIC_BUILD = 'critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)'
rsl_spec = importlib.util.find_spec("rsl_rl")
ppo_path = Path(rsl_spec.origin).parent / "algorithms" / "ppo.py" if rsl_spec and rsl_spec.origin else None
if ppo_path and ppo_path.is_file():
    ppo_text = ppo_path.read_text(encoding="utf-8")
    print("rsl_rl/algorithms/ppo.py：act() 存下 V(s_t) → compute_returns() 算 δ → update() 算 critic 的损失，加进总损失")
    check(f"正文映射块引用的源码行（连同到点补估计的两行，共 {len(PPO_LINES)} 行）都原样存在，且顺序一致",
          lines_in_order(ppo_text, PPO_LINES) and ppo_text.find("    def act(") < ppo_text.find(PPO_LINES[0])
          < ppo_text.find("    def process_env_step(") < ppo_text.find(PPO_LINES[1])
          < ppo_text.find("    def compute_returns(") < ppo_text.find(PPO_LINES[3]) < ppo_text.find("    def update(")
          < ppo_text.find(PPO_LINES[8]))
    check("construct_algorithm() 里 critic 的输出只有 1 个数", CRITIC_BUILD in ppo_text)
else:
    print("  （当前 Python 环境里没有 rsl_rl，跳过源码核对；用 uv run 运行就会核对）")

mj_spec = importlib.util.find_spec("mjlab")
mj_root = Path(mj_spec.origin).parent if mj_spec and mj_spec.origin else None
if mj_root and (mj_root / "rl" / "vecenv_wrapper.py").is_file():
    wrap_text = (mj_root / "rl" / "vecenv_wrapper.py").read_text(encoding="utf-8")
    vel_text = (mj_root / "tasks" / "velocity" / "velocity_env_cfg.py").read_text(encoding="utf-8")
    check("mjlab 的 dones = 摔倒（terminated）或到点（truncated）；到点另记在 time_outs 里",
          lines_in_order(wrap_text, ["term_or_trunc = terminated | truncated", 'extras["time_outs"] = truncated']))
    check("mjlab 模板：物理步 0.005 秒、每 4 个物理步一个环境步、回合 20 秒 → 20 ÷ 0.02 = 1000 步",
          lines_in_order(vel_text, ["timestep=0.005,", "decimation=4,", "episode_length_s=20.0,"])
          and round(20.0 / (0.005 * 4)) == 1000)
    obs_text = (mj_root / "tasks" / "velocity" / "mdp" / "observations.py").read_text(encoding="utf-8")
    check("mjlab 模板的 critic 组 = actor 的全部 + 四项脚部观测（离地高度、悬空时长、着没着地、受力），而且不加噪声",
          lines_in_order(vel_text, ["critic_terms = {", "**actor_terms,", '"foot_height": ObservationTermCfg(',
                                    '"foot_air_time": ObservationTermCfg(', '"foot_contact": ObservationTermCfg(',
                                    '"foot_contact_forces": ObservationTermCfg(', '"critic": ObservationGroupCfg(',
                                    "enable_corruption=False,"])
          and "forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]" in obs_text)
    check("多出的 15 个数：速度 3 + 离地高度 2 + 悬空时长 2 + 着没着地 2 + 受力 2 × 3 = 15；61 + 15 = 76",
          3 + 2 + 2 + 2 + 2 * 3 == 15 and 61 + 15 == 76 and 3 + 6 + 6 == 15)
else:
    print("  （当前 Python 环境里没有 mjlab，跳过这一项；用 uv run 运行就会核对）")

done()
