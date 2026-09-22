"""第 9 章实验：5 格的“走廊游戏”——用最小的例子看清 智能体 / 环境 / 策略 / 回合 / 终止与超时 / 回报 / 折扣 / 目标。

运行：uv run python docs/learn-zh/labs/ch09_gridworld.py
纯 CPU，numpy + matplotlib。第 8 节只读几份源码的文字，不加载机器人、不训练。
小节顺序与正文一致：1 ↔ 9.2，2 ↔ 9.4，3 ↔ 9.5，4 ↔ 9.6，5 ↔ 9.7，6 ↔ 9.8，7 ↔ 9.9（外加 9.0 的总览图），8 ↔「映射到项目」。
正文“改一改”要改的三行都带 `# TWEAK-k:` 标记（第 1 节两处、第 6 节一处）。
走廊的设定和第 10 章的实验 ch10_td_learning.py 相同，不要改。
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

from _common import banner, check, done, lines_in_order, num, savefig, table
from _draw import (BLUE, CELL, CELL_EDGE, CELL_HOT, FAINT, FS_NOTE, FS_SMALL, FS_STEP, FS_TICK, FS_TITLE, GREEN, INK,
                   MUTED, ORANGE, WHITE_BOX, arrow, cell, data_axes, hand, lesson_cells, lesson_figure, lesson_panel,
                   note, panel_note, panel_title, plt)

# ---------------------------------------------------------------------------
banner("1. 走廊的规则：智能体按一下，环境掷一次骰子、给一个分（9.2 节）")
N_CELLS = 5            # 格子 0、1、2、3、4
START, GOAL = 0, 4     # 从第 0 格出发，第 4 格是终点
GOAL_REWARD = 1.0      # 踩上终点：+1，这一局结束
STEP_REWARD = -0.05  # TWEAK-1: 0.0
P_SLIP = 0.2  # TWEAK-3: 0.5
TIME_LIMIT = 50        # 一局最多 50 步（9.6 节：走满 50 步还没到终点，就按超时收场）
RIGHT, LEFT = +1, -1
ARROW_OF = {RIGHT: "→", LEFT: "←"}


def step(s, a, rng):
    """环境的一步：智能体在第 s 格按了 a（+1 = →，−1 = ←）。
    掷一次骰子：1 − P_SLIP 的可能照办，P_SLIP 的可能打滑走反；撞墙就原地不动。返回 (下一格, 奖励, 结束了吗, 打滑了吗)。"""
    slipped = not (rng.uniform() < 1 - P_SLIP)
    move = -a if slipped else a
    s_next = min(max(s + move, 0), N_CELLS - 1)
    ended = s_next == GOAL
    return s_next, (GOAL_REWARD if ended else STEP_REWARD), ended, slipped


def outcomes(s, a):
    """一步的全部可能结果（不抽样，直接列出来）：[(下一格, 概率, 奖励, 结束了吗)]。"""
    rows = []
    for move, p in ((a, 1 - P_SLIP), (-a, P_SLIP)):
        s_next = min(max(s + move, 0), N_CELLS - 1)
        rows.append((s_next, p, GOAL_REWARD if s_next == GOAL else STEP_REWARD, s_next == GOAL))
    return rows


rows = []
for s, a, what in ((1, RIGHT, "中间"), (3, RIGHT, "终点前一格"), (0, LEFT, "墙边")):
    for s_next, p, r, ended in outcomes(s, a):
        rows.append([f"第 {s} 格（{what}）", ARROW_OF[a], f"第 {s_next} 格", num(p, 2), num(r, 2), "是" if ended else "否"])
table(["在哪", "按了", "结果", "概率", "奖励", "这一局结束？"], rows)
one = outcomes(1, RIGHT)
check("在第 1 格按 →：80% 到第 2 格、20% 打滑退回第 0 格，两种都拿 −0.05",
      one == [(2, 0.8, -0.05, False), (0, 0.2, -0.05, False)])
check("在第 3 格按 →：80% 踩上终点拿 +1、这一局结束；20% 退回第 2 格拿 −0.05",
      outcomes(3, RIGHT) == [(4, 0.8, 1.0, True), (2, 0.2, -0.05, False)])
check("在第 0 格按 ←：撞墙原地不动（80%），打滑反而到第 1 格（20%）", outcomes(0, LEFT) == [(0, 0.8, -0.05, False), (1, 0.2, -0.05, False)])

rng = np.random.default_rng(1)
n_try = 10_000
arrived = sum(step(1, RIGHT, rng)[0] == 2 for _ in range(n_try)) / n_try
p_told = 1 - P_SLIP
se = math.sqrt(p_told * (1 - p_told) / n_try)
print(f"在第 1 格按 → {n_try} 次，真到了第 2 格的比例：{arrived:.4f}（规则写的是 {p_told:g}）")
check(f"抽样比例靠近规则里的 {p_told:g}（差不到 4 倍标准误差 {4 * se:.4f}，第 4 章 4.5 节的大数定律）", abs(arrived - p_told) < 4 * se)

# ---------------------------------------------------------------------------
banner("1b. 画图：figures/ch09_loop.png（一圈：看一眼 → 做动作 → 环境回一个分和下一眼）")


def loop_panel(ax, title, seen, act, back, env_line, agent_line, foot):
    lesson_panel(ax, title, xmax=10, ymax=5)
    for x0, head, sub in ((0.25, "智能体", agent_line), (6.85, "环境", env_line)):
        ax.add_patch(plt.Rectangle((x0, 1.7), 2.9, 1.3, facecolor=CELL, edgecolor=CELL_EDGE, lw=1.4))
        ax.text(x0 + 1.45, 2.58, head, fontsize=FS_STEP, fontweight="bold", color=INK, ha="center", va="center")
        ax.text(x0 + 1.45, 2.05, sub, fontsize=FS_SMALL - 2, color=MUTED, ha="center", va="center")
    arrow(ax, (3.2, 2.72), (6.8, 2.72), BLUE, lw=3)
    arrow(ax, (6.8, 1.98), (3.2, 1.98), GREEN, lw=3)
    hand(ax, 0.25, 3.72, seen, color=INK, fontsize=FS_SMALL)
    hand(ax, 5.0, 3.25, act, color=BLUE, fontsize=FS_SMALL, ha="center")
    hand(ax, 5.0, 1.3, back, color=GREEN, fontsize=FS_SMALL, ha="center")
    note(ax, 0.25, 0.5, foot)


fig, axes = lesson_figure(2, "一步 = 一圈：智能体看一眼、做一个动作，环境回一个分和下一眼", panel_height=4.1, width=9.6)
loop_panel(axes[0], "① 走廊里：t = 1 这一圈", "看到 $o_1$：我在第 1 格", "动作 $a_1$：按 →",
           f"奖励 $r_1 = {num(STEP_REWARD, 2)}$    下一眼 $o_2$：在第 2 格", "走廊 + 打滑 + 记分", "做决定的那个",
           f"环境掷骰子：{(1 - P_SLIP) * 100:.0f}% 照办到第 2 格，{P_SLIP * 100:.0f}% 打滑退回第 0 格。画的是照办那一支。")
loop_panel(axes[1], "② 机器人里：每 0.02 秒一圈", "看到 $o_t$：61 个数", "动作 $a_t$：14 个舵机的目标角度",
           "奖励 $r_t$：1 个数    下一眼 $o_{t+1}$：61 个数", "MuJoCo 仿真 + 打分", "actor 网络（第 6 章）",
           "仿真往前算 4 小步，每小步 0.005 秒，合 0.02 秒；一秒钟转 50 圈。")
savefig(fig, "ch09_loop")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("2. 策略：看到什么就怎么做的规则；带随机，才会探索（9.4 节）")
P_MOSTLY = 0.9   # “九成按 →”这个策略按 → 的概率（故意不用 0.8，免得和地面的 80% 混在一起）


def always_right(s, rng):
    return RIGHT


def mostly_right(s, rng):
    return RIGHT if rng.uniform() < P_MOSTLY else LEFT


def coin_flip(s, rng):
    return RIGHT if rng.uniform() < 0.5 else LEFT


POLICIES = [("总是按 →", always_right, 1.0), ("九成按 →", mostly_right, P_MOSTLY), ("随机乱按", coin_flip, 0.5)]
rows = []
tries_left = {}
for name, policy, p_right in POLICIES:
    rng = np.random.default_rng(2)
    lefts = sum(policy(0, rng) == LEFT for _ in range(100))
    tries_left[name] = lefts
    rows.append([name, num(p_right, 2), f"{100 * (1 - p_right):.0f}", lefts])
table(["策略", "按 → 的概率", "100 次里按 ← 的期望次数", "这次实际按 ← 的次数"], rows)
check("总是按 →：100 次里一次 ← 也没试过", tries_left["总是按 →"] == 0)
check("九成按 →：期望 100 × 0.1 = 10 次 ←；这次抽到 " + str(tries_left["九成按 →"]) + " 次（落在 4 到 16 之间）",
      4 <= tries_left["九成按 →"] <= 16)

p_move_right = P_MOSTLY * (1 - P_SLIP) + (1 - P_MOSTLY) * P_SLIP
by_obey, by_slip = P_MOSTLY * (1 - P_SLIP), (1 - P_MOSTLY) * P_SLIP   # 按 → 且照办；按 ← 却打滑
print(f"九成按 →：真往右挪的概率 = {P_MOSTLY:g} × {1 - P_SLIP:g} + {1 - P_MOSTLY:g} × {P_SLIP:g} = {by_obey:.2f} + {by_slip:.2f} = "
      f"{p_move_right:.2f}")
check("两层随机叠起来：0.9 × 0.8 + 0.1 × 0.2 = 0.72 + 0.02 = 0.74", round(p_move_right, 2) == 0.74
      and round(0.72 + 0.02, 2) == 0.74)
rng = np.random.default_rng(3)
moved = 0
for _ in range(n_try):
    s_next = step(1, mostly_right(1, rng), rng)[0]
    moved += s_next == 2
se = math.sqrt(p_move_right * (1 - p_move_right) / n_try)
print(f"在第 1 格用“九成按 →”走 {n_try} 步，真往右挪的比例：{moved / n_try:.4f}")
check(f"抽样比例靠近 {p_move_right:.2f}（差不到 4 倍标准误差 {4 * se:.4f}）", abs(moved / n_try - p_move_right) < 4 * se)
check("正文引用的这一次抽样：100 次里按 ← 的次数 0、7、49；站在第 1 格试 10,000 次，真往右挪的比例 0.7398（种子固定）",
      tries_left == {"总是按 →": 0, "九成按 →": 7, "随机乱按": 49} and round(moved / n_try, 4) == 0.7398)
check("自测：九成按 → 按 1000 次，期望 1000 × 0.1 = 100 次 ←", round(1000 * (1 - P_MOSTLY)) == 100)
p_right_always = 1.0 * (1 - P_SLIP)
p_right_coin = 0.5 * (1 - P_SLIP) + 0.5 * P_SLIP
print(f"自测：总是按 → 真往右挪 1 × {1 - P_SLIP:g} = {p_right_always:.2f}；"
      f"随机乱按 0.5 × {1 - P_SLIP:g} + 0.5 × {P_SLIP:g} = {p_right_coin:.2f}")
check("自测：总是按 → 真往右挪 1 × 0.8 = 0.8；随机乱按 0.5 × 0.8 + 0.5 × 0.2 = 0.4 + 0.1 = 0.5",
      round(p_right_always, 2) == 0.8 and round(p_right_coin, 2) == 0.5 and round(0.4 + 0.1, 2) == 0.5)

# ---------------------------------------------------------------------------
banner("3. 回合与轨迹：“总是按 →”跑两局，把每一步记下来（9.5 节）")


def run_episode(policy, rng, limit=TIME_LIMIT):
    """跑一局：返回轨迹（每一步一行：t, s_t, a_t, r_t, s_{t+1}, 打滑了吗）和结束方式。"""
    s, traj = START, []
    for t in range(limit):
        a = policy(s, rng)
        s_next, r, ended, slipped = step(s, a, rng)
        traj.append((t, s, a, r, s_next, slipped))
        s = s_next
        if ended:
            return traj, "终止"
    return traj, "超时"


def show(traj):
    table(["t", "s_t", "a_t", "r_t", "s_{t+1}"], [[t, s, ARROW_OF[a], num(r, 2), s2] for t, s, a, r, s2, _ in traj])


traj_a, end_a = run_episode(always_right, np.random.default_rng(0))   # 第一局
traj_b, end_b = run_episode(always_right, np.random.default_rng(2))   # 第二局
print("第一局：")
show(traj_a)
print("第二局：")
show(traj_b)
rewards_a = [x[3] for x in traj_a]
rewards_b = [x[3] for x in traj_b]
slips_b = [x[0] for x in traj_b if x[5]]
print(f"第一局 {len(traj_a)} 步、打滑 {sum(x[5] for x in traj_a)} 次；第二局 {len(traj_b)} 步、打滑在 t = {slips_b}")
check("第一局：4 步到终点，一次没打滑", len(traj_a) == 4 and not any(x[5] for x in traj_a) and end_a == "终止")
check("第二局：6 步到终点，只在 t = 2 打滑一次（从第 2 格退回第 1 格）",
      len(traj_b) == 6 and slips_b == [2] and traj_b[2][1] == 2 and traj_b[2][4] == 1 and end_b == "终止")
n_states = len(traj_b) + 1
print(f"第二局的轨迹：状态 s_0…s_{len(traj_b)} 共 {n_states} 个，动作 {len(traj_b)} 个，奖励 {len(traj_b)} 个")
check("走了 6 步的一局：7 个状态、6 个动作、6 个奖励（最后那个状态是终点，没有再做动作）", n_states == 7 and len(rewards_b) == 6)
check("自测：第一局 4 步 → 5 个状态（第 0 到第 4 格）、4 个动作、4 个奖励（−0.05、−0.05、−0.05、+1）",
      len(traj_a) + 1 == 5 and [x[1] for x in traj_a] + [traj_a[-1][4]] == [0, 1, 2, 3, 4] and rewards_a == [-0.05, -0.05, -0.05, 1.0])

# ---------------------------------------------------------------------------
banner("3b. 画图：figures/ch09_trajectory.png（同一个策略、同一个起点，两局两条轨迹）")


def trajectory_panel(ax, traj, title, foot):
    rows_n = len(traj)
    ymax = rows_n + 2.4
    lesson_panel(ax, title, xmax=10, ymax=ymax)
    top = ymax - 1.35                         # 表头那一行的中心
    for j in range(N_CELLS):
        ax.text(1.65 + j, top, f"{j} 终点" if j == GOAL else f"{j}", fontsize=FS_SMALL, color=GREEN if j == GOAL else MUTED,
                ha="center", va="center")
    ax.text(6.95, top, "按了", fontsize=FS_SMALL, color=MUTED, ha="center", va="center")
    ax.text(8.45, top, "奖励", fontsize=FS_SMALL, color=MUTED, ha="center", va="center")
    for t, s, a, r, s2, slipped in traj:
        y = top - 0.95 * (t + 1)
        ax.text(0.15, y, f"t = {t}", fontsize=FS_SMALL, color=INK, va="center")
        for j in range(N_CELLS):
            edge = GREEN if j == GOAL else CELL_EDGE
            ax.add_patch(plt.Rectangle((1.2 + j, y - 0.34), 0.9, 0.68, facecolor=CELL_HOT if j == s else CELL,
                                       edgecolor=edge, lw=1.8 if j == GOAL else 1.1))
        ax.plot(1.65 + s, y, "o", color=INK, ms=10, zorder=6)
        color = ORANGE if slipped else BLUE
        arrow(ax, (1.65 + s + (0.18 if s2 > s else -0.18), y), (1.65 + s2 - (0.2 if s2 > s else -0.2), y), color, lw=3,
              zorder=7)
        if slipped:                           # 标在同一行“按了”那一列：按的是 →，却打滑了
            ax.text(7.3, y, "打滑", fontsize=FS_SMALL - 2, color=ORANGE, ha="left", va="center", zorder=8)
        ax.text(6.95, y, ARROW_OF[a], fontsize=FS_STEP, color=INK, ha="center", va="center")
        ax.text(8.45, y, "+1" if r > 0 else num(r, 2).replace("-", "−"), fontsize=FS_STEP,
                color=GREEN if r > 0 else INK, ha="center", va="center")
    note(ax, 0.15, 0.45, foot)


if max(len(traj_a), len(traj_b)) > 9:
    print("  这两局太长（多半是改了打滑概率），轨迹图只照着正文那两局画，跳过。看上面的表就行。")
else:
    heights = [len(traj_a) + 2.4, len(traj_b) + 2.4]
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 0.62 * sum(heights) + 1.3), gridspec_kw={"height_ratios": heights})
    fig.subplots_adjust(top=1 - 1.05 / (0.62 * sum(heights) + 1.3), hspace=0.08, left=0.03, right=0.99, bottom=0.01)
    fig.suptitle("同一个策略、同一个起点：两局走出两条不同的轨迹", fontsize=FS_TITLE, fontweight="bold", color=INK)
    trajectory_panel(axes[0], traj_a, f"① 第一局：每次都按 →，一次没打滑，{len(traj_a)} 步到终点",
                     "每一行是一步：圆点是这一步开始时在哪格，箭头是这一步实际往哪走。")
    trajectory_panel(axes[1], traj_b, f"② 第二局：还是每次都按 →，t = 2 打滑退了一格，{len(traj_b)} 步到终点",
                     "策略一模一样，环境的骰子不同，轨迹就不同：多走了 2 步，多扣了 2 次分。")
    savefig(fig, "ch09_trajectory")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("4. 终止与超时：一局有两种收场（9.6 节）")
rng = np.random.default_rng(4)
endings = {"终止": 0, "超时": 0}
last_cell_on_timeout = [0] * N_CELLS
for _ in range(2000):
    traj, how = run_episode(coin_flip, rng)
    endings[how] += 1
    if how == "超时":
        last_cell_on_timeout[traj[-1][4]] += 1
print(f"“随机乱按”跑 2000 局：踩上终点而终止 {endings['终止']} 局，走满 {TIME_LIMIT} 步被超时截断 {endings['超时']} 局")
print("超时那几局，截断时站在哪一格：", "  ".join(f"第 {j} 格 {c} 局" for j, c in enumerate(last_cell_on_timeout) if j != GOAL))
check("随机乱按 2000 局：1878 局终止、122 局超时（种子固定时的这一次）", endings == {"终止": 1878, "超时": 122})
check("超时收场的局，一局也没站在终点上（站在终点就是终止了）", last_cell_on_timeout[GOAL] == 0)
check("超时收场时正站在第 3 格（离终点一步）的有 14 局", last_cell_on_timeout[3] == 14)
g_z_70 = -math.cos(math.radians(70))
print(f"机器人倾斜 70° 时，投影重力的第 3 项 g_z = −cos 70° = {g_z_70:.3f}（站直时是 −1）")
check("倾斜 70° 时 g_z ≈ −0.34（第 2 章 2.4 节的摔倒判据）", round(g_z_70, 2) == -0.34)
episode_steps = math.ceil(20.0 / (4 * 0.005))
print(f"机器人：一局最长 20 秒 ÷ 每步 0.02 秒 = {episode_steps} 步；而每只机器人攒够 24 步就训练一次：{episode_steps} ÷ 24 = "
      f"{episode_steps / 24:.2f}")
check("机器人一局最长 20 ÷ 0.02 = 1000 步；1000 ÷ 24 = 41.67，攒数据的 24 步和一局的边界本来就对不齐",
      episode_steps == 1000 and round(episode_steps / 24, 2) == 41.67)

# ---------------------------------------------------------------------------
banner("5. 回报：从这一步往后的分，打着折加起来（9.7 节）")


def discounted_return(rewards, gamma):
    """从最后一步往前加：总分 = 这一步的分 + γ × 后面的总分（JS 里的 reduceRight）。"""
    later = 0.0
    for r in reversed(rewards):
        later = r + gamma * later
    return later


def forward_return(rewards, gamma):
    """从前往后逐项打折：r_0 + γ r_1 + γ² r_2 + …"""
    return sum(gamma ** k * r for k, r in enumerate(rewards))


three = [0.0, 0.0, 1.0]
G_three = [discounted_return(three[t:], 0.9) for t in range(3)]
print("三步经历 [0, 0, 1]，每远一步乘 0.9：")
print(f"  从后往前：G_2 = 1；G_1 = 0 + 0.9 × 1 = {G_three[1]:g}；G_0 = 0 + 0.9 × 0.9 = {G_three[0]:g}")
print(f"  从前往后：0 + 0.9 × 0 + 0.9² × 1 = {forward_return(three, 0.9):g}")
check("三步 [0, 0, 1]、γ = 0.9：G_0 = 0.81，G_1 = 0.9，G_2 = 1（worked_examples 同一个数）",
      [round(g, 10) for g in G_three] == [0.81, 0.9, 1.0])
check("从前往后逐项打折 = 从后往前一步步加（两种算法同一个数）", math.isclose(forward_return(three, 0.9), G_three[0]))
check("自测 [0, 1, 0]、γ = 0.9：G_0 = 0.9，G_1 = 1，G_2 = 0",
      [round(discounted_return([0, 1, 0][t:], 0.9), 10) for t in range(3)] == [0.9, 1.0, 0.0])

rows = []
G0_a = {}
for g in (1.0, 0.99, 0.9, 0.5):
    G0_a[g] = discounted_return(rewards_a, g)
    rows.append([num(g, 2), f"{G0_a[g]:.6g}"])
table(["γ", "第一局的 G_0"], rows)
if end_a == "终止":
    last = len(rewards_a) - 1
    print(f"每一行的最后一步都是 G_{last} = {discounted_return(rewards_a[last:], 0.5):g}：终点那一步之后没有别的分了")
check("第一局 [−0.05, −0.05, −0.05, 1]：γ = 1 → 0.85，0.99 → 0.821794，0.9 → 0.5935，0.5 → 0.0375",
      [round(G0_a[g], 6) for g in (1.0, 0.99, 0.9, 0.5)] == [0.85, 0.821794, 0.5935, 0.0375])
check("字面值重算 γ = 0.9：−0.05 − 0.045 − 0.0405 + 0.729 = 0.5935", round(-0.05 - 0.045 - 0.0405 + 0.729, 4) == 0.5935)
check("字面值重算 γ = 1：−0.05 × 3 + 1 = 0.85", round(-0.05 * 3 + 1, 2) == 0.85)
check("四个分分别乘 1、0.9、0.81、0.729（0.9 的 0、1、2、3 次方）", [round(0.9 ** k, 3) for k in range(4)] == [1.0, 0.9, 0.81, 0.729])
G0_js = 0 + 0.9 * (0 + 0.9 * (1 + 0.9 * 0))   # JS 的 reduceRight 按同样的顺序算
check("reduceRight 一步步算出的数打印出来正好是 0.81（浏览器和 Python 用的是同一种小数）", repr(G0_js) == "0.81")

# ---------------------------------------------------------------------------
banner("5b. 画图：figures/ch09_return_by_hand.png（三步经历 [0, 0, 1] 的回报，两种算法）")
fig, axes = lesson_figure(3, "回报：从这一步往后的分，每远一步多打一次 0.9 折", panel_height=3.35, width=9.6)
ax = axes[0]
lesson_panel(ax, "① 从第 0 步看：越远的分，多乘几次 0.9")
for j in range(3):
    ax.text(3.55 + j * 1.5, 2.95, f"第 {j} 步", fontsize=FS_SMALL, color=MUTED, ha="center", va="center")
labels = ["这一步的分", "要乘的折扣", "打折之后"]
weights = [0.9 ** j for j in range(3)]
lesson_cells(ax, [three, weights, [w * r for w, r in zip(weights, three)]], left=2.8, bottom=0.55, width=1.5, height=0.72,
             highlights={(2, 2)}, fmt="{:g}")
for i, lab in enumerate(labels):
    ax.text(2.65, 0.55 + (2 - i) * 0.72 + 0.36, lab, fontsize=FS_SMALL, color=INK, ha="right", va="center")
hand(ax, 7.45, 1.95, "0 + 0 + 0.81", color=BLUE)
hand(ax, 7.45, 1.25, f"= {G_three[0]:g}", color=ORANGE)
note(ax, 0.3, 0.12, "第 k 步的分乘 k 次 0.9：第 0 步乘 1（不打折），第 2 步乘 0.9 × 0.9 = 0.81。")

ax = axes[1]
lesson_panel(ax, "② 从最后一步往前：这一步的分 + 0.9 × 后面的总分")
lesson_cells(ax, [[num(g, 2) for g in G_three]], left=0.3, bottom=1.35, width=1.2, height=0.8, highlights={(0, 0)})
for j in range(3):
    ax.text(0.9 + j * 1.2, 2.5, f"第 {j} 步", fontsize=FS_SMALL, color=MUTED, ha="center", va="center")
for j in (2, 1):
    arrow(ax, (0.9 + j * 1.2 - 0.05, 1.12), (0.9 + (j - 1) * 1.2 + 0.05, 1.12), GREEN, lw=2.5,
          connectionstyle="arc3,rad=-0.45")
hand(ax, 4.35, 2.75, "第 2 步：1（后面没有了）", color=INK)
hand(ax, 4.35, 1.85, f"第 1 步：0 + 0.9 × 1 = {G_three[1]:g}", color=GREEN)
hand(ax, 4.35, 0.95, f"第 0 步：0 + 0.9 × 0.9 = {G_three[0]:g}", color=ORANGE)
note(ax, 0.3, 0.25, "每往前一步，只做一次“后面的总分 × 0.9，再加上这一步的分”，不用从头加。")

ax = axes[2]
lesson_panel(ax, "③ 眼前 0 分，回报却是 0.81")
lesson_cells(ax, [three, [num(g, 2) for g in G_three]], left=3.6, bottom=1.25, width=1.3, height=0.72, highlights={(1, 0)})
for i, lab in enumerate(["当场拿到的分", "从这一步起的回报"]):
    ax.text(3.45, 1.25 + (1 - i) * 0.72 + 0.36, lab, fontsize=FS_SMALL, color=INK, ha="right", va="center")
for j in range(3):
    ax.text(4.25 + j * 1.3, 2.95, f"第 {j} 步", fontsize=FS_SMALL, color=MUTED, ha="center", va="center")
note(ax, 0.3, 0.55, "第 0 步当场一分没拿，可它离那个 +1 只差两步：回报把这份“铺路”的功劳记下了。")
savefig(fig, "ch09_return_by_hand")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("6. 折扣因子 γ 看多远：每过一步乘一次 γ（9.8 节）")
GAMMA = 0.99  # TWEAK-2: 0.9
STEP_SECONDS = 0.02    # 50 Hz：一步 0.02 秒
rows = []
for k in (1, 10, 50, 69, 100, 200, 500):
    rows.append([k, num(k * STEP_SECONDS, 2), f"{GAMMA ** k:.4g}", f"{0.9 ** k:.3g}"])
table(["往后第 k 步", "= 几秒", f"{GAMMA:g}^k", "0.9^k"], rows)
half_life = next(k for k in range(1, 10_000) if GAMMA ** k <= 0.5)
half_life_09 = next(k for k in range(1, 10_000) if 0.9 ** k <= 0.5)
print(f"γ = {GAMMA:g}：第 {half_life} 步第一次不到一半（{GAMMA ** half_life:.4f}），合 {half_life * STEP_SECONDS:.2f} 秒；"
      f"γ = 0.9：第 {half_life_09} 步（{0.9 ** half_life_09:.3f}），合 {half_life_09 * STEP_SECONDS:.2f} 秒")
check("0.99 的 69 次方 = 0.4998（约一半），69 步 = 1.38 秒", round(0.99 ** 69, 4) == 0.4998 and half_life == 69
      and round(69 * STEP_SECONDS, 2) == 1.38)
check("0.99^100 = 0.366（100 步 = 2 秒），0.99^200 = 0.134，0.99^500 = 0.0066",
      round(GAMMA ** 100, 3) == 0.366 and round(GAMMA ** 200, 3) == 0.134 and round(GAMMA ** 500, 4) == 0.0066)
check("0.9^7 = 0.478（7 步 = 0.14 秒就不到一半），0.9^10 = 0.349", round(0.9 ** 7, 3) == 0.478 and half_life_09 == 7
      and round(0.9 ** 10, 3) == 0.349)
check("0.99^10 = 0.904，0.99^50 = 0.605；γ = 0.9 时 50 步（1 秒）后只算 0.00515",
      round(0.99 ** 10, 3) == 0.904 and round(0.99 ** 50, 3) == 0.605 and round(0.9 ** 50, 5) == 0.00515)

print("每一步都拿 1 分、一直拿下去，打折后的总分（前 n 项加起来）：")
partial = {n: sum(GAMMA ** k for k in range(n)) for n in (1, 2, 100, 500, 2000)}
print("  " + "   ".join(f"前 {n} 项 {v:.2f}" for n, v in partial.items()) + f"   → 靠近 1/(1 − γ) = {1 / (1 - GAMMA):.0f}")
check("1 + 0.99 = 1.99；前 100 项 63.40；前 500 项 99.34；越加越靠近 1/(1 − 0.99) = 100",
      round(partial[2], 2) == 1.99 and round(partial[100], 2) == 63.40 and round(partial[500], 2) == 99.34
      and round(1 / (1 - 0.99)) == 100 and abs(partial[2000] - 100) < 0.001)
check("γ = 0.5 的迷你版：1 + 0.5 + 0.25 + 0.125 = 1.875，再加下去靠近 1/(1 − 0.5) = 2",
      1 + 0.5 + 0.25 + 0.125 == 1.875 and abs(sum(0.5 ** k for k in range(60)) - 2) < 1e-12)
check("1/(1 − 0.9) = 10；自测 γ = 0.95：1/(1 − 0.95) = 20", round(1 / (1 - 0.9)) == 10 and round(1 / (1 - 0.95)) == 20)
check("自测 γ = 0.95：0.95^13 = 0.513，0.95^14 = 0.488 —— 13 到 14 步之间剩一半（约 0.27 秒）",
      round(0.95 ** 13, 3) == 0.513 and round(0.95 ** 14, 3) == 0.488 and round(13.5 * STEP_SECONDS, 2) == 0.27)
g100 = math.sqrt(0.99)
print(f"换成 100 Hz、想保持“每秒打的折”不变：γ_100 = √0.99 = {g100:.5f}；它的 200 次方 = {g100 ** 200:.3f}，和 0.99^100 一样")
check("√0.99 = 0.99499；0.99499^200 = 0.366 = 0.99^100（2 秒在 100 Hz 下是 200 步）",
      round(g100, 5) == 0.99499 and round(0.99499 ** 200, 3) == 0.366 and round(0.99 ** 100, 3) == 0.366)
check("字面值重算：1/e = 0.3679，和 0.99^100 = 0.366 差不多（第 1 章 1.8 节那张表）", round(math.exp(-1), 4) == 0.3679)
check("γ 越接近 1，第 1/(1 − γ) 步的折扣越接近 1/e：0.9^10 = 0.349，0.99^100 = 0.366，0.999^1000 = 0.3677",
      round(0.9 ** 10, 3) == 0.349 and round(0.99 ** 100, 3) == 0.366 and round(0.999 ** 1000, 4) == 0.3677)
check("100 Hz 走两步 = 50 Hz 走一步：0.99499 × 0.99499 ≈ 0.99", round(0.99499 * 0.99499, 4) == 0.99)

# ---------------------------------------------------------------------------
banner("6b. 画图：figures/ch09_discount.png（γ^k 怎样随步数衰减）")
fig = plt.figure(figsize=(9.6, 13.2))
fig.suptitle(f"γ = {GAMMA:g}：每过一步乘一次 {GAMMA:g}，{half_life} 步后剩一半", fontsize=FS_TITLE, fontweight="bold",
             color=INK, y=0.985)
ax1 = fig.add_axes([0.13, 0.575, 0.83, 0.30])
k_all = np.arange(0, 501)
ax1.plot(k_all, GAMMA ** k_all, color=BLUE, lw=3.2, label=f"γ = {GAMMA:g}")
ax1.plot(k_all, 0.9 ** k_all, color=ORANGE, lw=2.6, label="γ = 0.9")
data_axes(ax1, "往后第 k 步（括号里是秒：50 Hz，一步 0.02 秒）", "这一步的分还算几成")
ax1.set_xlim(0, 510)
ax1.set_ylim(0, 1.05)
ax1.set_xticks(range(0, 501, 100), [f"{k}\n({num(k * STEP_SECONDS, 2)} 秒)" for k in range(0, 501, 100)])
ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
for k, dx, dy, fmt in ((half_life, 22, 0.1, ".4f"), (100, 30, 0.02, ".3f"), (200, 12, 0.1, ".3f"), (500, -30, 0.12, ".4f")):
    v = GAMMA ** k
    ax1.plot([k], [v], "o", color=BLUE, ms=9, zorder=6)
    ax1.text(k + dx, v + dy, f"第 {k} 步：{v:{fmt}}", fontsize=FS_SMALL - 1, color=BLUE, ha="left" if dx > 0 else "right",
             va="center", bbox=WHITE_BOX)
ax1.axhline(0.5, color=FAINT, ls="--", lw=1.4, zorder=1)
ax1.legend(fontsize=FS_SMALL, frameon=False, loc="upper right")
panel_title(fig, [ax1], f"① 往后 500 步（10 秒）：γ = {GAMMA:g} 慢慢往下走")
panel_note(fig, [ax1], f"100 步（2 秒）后还剩 {GAMMA ** 100:.3f}，200 步后 {GAMMA ** 200:.3f}，500 步后 {GAMMA ** 500:.4f}：\n"
                        "越来越小，但不是到某一步突然变成 0。")

ax2 = fig.add_axes([0.13, 0.12, 0.83, 0.25])
k_few = np.arange(0, 31)
ax2.plot(k_few, GAMMA ** k_few, "-o", color=BLUE, lw=2.6, ms=5, label=f"γ = {GAMMA:g}")
ax2.plot(k_few, 0.9 ** k_few, "-o", color=ORANGE, lw=2.4, ms=5, label="γ = 0.9")
data_axes(ax2, "往后第 k 步（只画前 30 步，括号里是秒）", "还算几成")
ax2.set_xlim(0, 30.5)
ax2.set_ylim(0, 1.05)
ax2.set_xticks(range(0, 31, 5), [f"{k}\n({num(k * STEP_SECONDS, 2)} 秒)" for k in range(0, 31, 5)])
ax2.set_yticks([0, 0.5, 1.0])
ax2.axhline(0.5, color=FAINT, ls="--", lw=1.4, zorder=1)
ax2.plot([7], [0.9 ** 7], "o", color=ORANGE, ms=11, zorder=6)
ax2.text(6.4, 0.9 ** 7 - 0.13, f"第 7 步：{0.9 ** 7:.3f}", fontsize=FS_SMALL - 1, color=ORANGE, ha="right", va="center",
         bbox=WHITE_BOX)
ax2.plot([10], [GAMMA ** 10], "o", color=BLUE, ms=11, zorder=6)
ax2.text(10.8, GAMMA ** 10 - 0.12, f"第 10 步：{GAMMA ** 10:.3f}", fontsize=FS_SMALL - 1, color=BLUE, va="center",
         bbox=WHITE_BOX)
ax2.legend(fontsize=FS_SMALL, frameon=False, loc="upper right")
panel_title(fig, [ax2], "② 放大最前面 30 步：γ = 0.9 掉得快得多")
panel_note(fig, [ax2], "横轴只画前 30 步，是 ① 最左边那一小段的放大。\nγ = 0.9 时，7 步（0.14 秒）以后的分就不到一半了。")
savefig(fig, "ch09_discount")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("7. 目标：比的是很多局的平均回报（9.9 节）")


def exact_average(p_right, gamma, limit=TIME_LIMIT):
    """不抽样的精确值：按概率一步一步往回推（还剩 h 步时，从每一格出发的平均回报和平均步数）。
    这是第 10 章（价值函数）要教的算法，这里只拿来核对抽样平均。"""
    value, length = np.zeros(N_CELLS), np.zeros(N_CELLS)
    for _ in range(limit):
        new_value, new_length = np.zeros(N_CELLS), np.zeros(N_CELLS)
        for s in range(N_CELLS):
            if s == GOAL:
                continue
            for a, p_a in ((RIGHT, p_right), (LEFT, 1 - p_right)):
                for s_next, p, r, ended in outcomes(s, a):
                    later_value = 0.0 if ended else value[s_next]
                    later_length = 0.0 if ended else length[s_next]
                    new_value[s] += p_a * p * (r + gamma * later_value)
                    new_length[s] += p_a * p * (1 + later_length)
        value, length = new_value, new_length
    return float(value[START]), float(length[START])


N_EPISODES = 2000
results = {}
rows = []
for name, policy, p_right in POLICIES:
    rng = np.random.default_rng(1)
    returns, lengths = [], []
    for _ in range(N_EPISODES):
        traj, _ = run_episode(policy, rng)
        returns.append(discounted_return([x[3] for x in traj], GAMMA))
        lengths.append(len(traj))
    returns = np.array(returns)
    exact_g, exact_len = exact_average(p_right, GAMMA)
    results[name] = (returns, float(np.mean(lengths)), exact_g, exact_len, np.array(lengths))
    rows.append([name, f"{returns.mean():.3f}", f"{np.mean(lengths):.2f}"])
table(["策略", "2000 局的平均回报", "平均每局几步"], rows)
for name, (returns, mean_len, exact_g, exact_len, _) in results.items():
    se = returns.std() / math.sqrt(N_EPISODES)
    print(f"  {name}：不抽样的精确值 平均回报 {exact_g:.4f}、平均 {exact_len:.2f} 步；抽样平均的标准误差 {se:.4f}")
    check(f"{name}：2000 局的平均 {returns.mean():.4f} 靠近精确值 {exact_g:.4f}（差不到 4 倍标准误差 {4 * se:.4f}）",
          abs(returns.mean() - exact_g) < 4 * se)
means = {name: round(float(v[0].mean()), 3) for name, v in results.items()}
check("正文引用的三个平均：总是按 → 0.703、九成按 → 0.639、随机乱按 0.015（种子固定时的这一次）",
      means == {"总是按 →": 0.703, "九成按 →": 0.639, "随机乱按": 0.015})
check("平均每局步数：6.09、7.25、18.99", [round(v[1], 2) for v in results.values()] == [6.09, 7.25, 18.99])
check("精确值：总是按 → 0.7015、平均 6.11 步（比抽样多一位，给 9.9 节引用）",
      round(results["总是按 →"][2], 4) == 0.7015 and round(results["总是按 →"][3], 2) == 6.11)
returns_right = results["总是按 →"][0]
first_right = returns_right[0]
print(f"总是按 →：第 1 局的回报 {first_right:.3f}；2000 局里最高 {returns_right.max():.3f}、最低 {returns_right.min():.3f}")
check("单独一局说明不了什么：“总是按 →”的回报从最低 −0.472 到最高 0.822 都有",
      round(returns_right.min(), 3) == -0.472 and round(returns_right.max(), 3) == 0.822)
lengths_right = results["总是按 →"][4]
n_four = int(np.sum(lengths_right == 4))
p_four = (1 - P_SLIP) ** 4
n_negative = int(np.sum(returns_right < 0))
print(f"总是按 →：4 步就到（四步都照办）的局 {n_four} 局，占 {n_four / N_EPISODES:.4f}；"
      f"理论上 {1 - P_SLIP:g} × {1 - P_SLIP:g} × {1 - P_SLIP:g} × {1 - P_SLIP:g} = {p_four:.4f}；回报是负的局 {n_negative} 局")
check("4 步就到的局：0.8 × 0.8 × 0.8 × 0.8 = 0.4096，约四成；2000 局里有 829 局；回报是负的有 6 局（种子固定）",
      round(0.8 * 0.8 * 0.8 * 0.8, 4) == 0.4096 and n_four == 829 and n_negative == 6)
se_four = math.sqrt(p_four * (1 - p_four) / N_EPISODES)
check(f"4 步就到的比例 {n_four / N_EPISODES:.4f} 靠近理论值 {p_four:.4f}（差不到 4 倍标准误差 {4 * se_four:.4f}）",
      abs(n_four / N_EPISODES - p_four) < 4 * se_four)
G0_b = discounted_return(rewards_b, GAMMA)
geo5 = sum(GAMMA ** k for k in range(5))
if len(rewards_b) == 6:
    g_ = f"{GAMMA:g}"
    print((f"第二局（6 步）：G_0 = {STEP_REWARD:g} × (1 + {g_} + {g_}² + {g_}³ + {g_}⁴) + {g_}⁵ × 1 = "
           f"{STEP_REWARD:g} × {geo5:.3f} + {GAMMA ** 5:.3f} = {G0_b:.3f}").replace("-", "−"))
else:
    print(f"第二局走了 {len(rewards_b)} 步：G_0 = {G0_b:.3f}".replace("-", "−"))
check("第一局（4 步）的 G_0 = 0.822；精确值 0.7015 和 2000 局的平均 0.703 相差不到 0.002",
      round(G0_a[0.99], 3) == 0.822 and abs(0.703 - 0.7015) < 0.002
      and abs(results["总是按 →"][0].mean() - results["总是按 →"][2]) < 0.002)
check("第二局 G_0 = −0.05 × 4.901 + 0.951 = 0.706（4.901、0.951 都按实算的值核对；字面值重算也是 0.706）",
      round(G0_b, 3) == 0.706 and round(geo5, 3) == 4.901 and round(GAMMA ** 5, 3) == 0.951
      and round(-0.05 * 4.901 + 0.951, 3) == 0.706)

# ---------------------------------------------------------------------------
banner("7b. 画图：figures/ch09_reward_return_value.png（奖励 / 回报 / 很多局回报的平均）")
fig = plt.figure(figsize=(9.6, 15.2))
fig.suptitle("奖励、回报、价值：一步的分、一局的总分、很多局的平均", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.99)
ax1 = fig.add_axes([0.14, 0.715, 0.82, 0.18])
t_b = np.arange(len(rewards_b))
ax1.bar(t_b, rewards_b, width=0.6, color=[GREEN if r > 0 else ORANGE for r in rewards_b], zorder=3)
data_axes(ax1, "第几步 t", "奖励 $r_t$")
ax1.set_xticks(t_b)
ax1.set_ylim(-0.25, 1.2)
ax1.set_yticks([0, 0.5, 1.0])
ax1.axhline(0, color=CELL_EDGE, lw=1.2, zorder=2)
for t, r in zip(t_b, rewards_b):
    ax1.text(t, r + (0.08 if r > 0 else -0.1), "+1" if r > 0 else num(r, 2).replace("-", "−"), fontsize=FS_SMALL - 2,
             color=GREEN if r > 0 else ORANGE, ha="center", va="center")
panel_title(fig, [ax1], f"① 奖励：一步一个分（第二局，{len(rewards_b)} 步）")
panel_note(fig, [ax1], f"前 {len(rewards_b) - 1} 步各扣 {num(-STEP_REWARD, 2)}，最后一步踩上终点拿 +1。")

ax2 = fig.add_axes([0.06, 0.435, 0.9, 0.13])
lesson_panel(ax2, xmax=10, ymax=2.6)
g_ = f"{GAMMA:g}"
hand(ax2, 0.2, 1.75, rf"$G_0 = {STEP_REWARD:g} \times (1 + {g_} + {g_}^2 + {g_}^3 + {g_}^4) + {g_}^5 \times 1$", color=BLUE)
hand(ax2, 0.75, 0.85, f"= {STEP_REWARD:g} × {geo5:.3f} + {GAMMA ** 5:.3f} = {G0_b:.3f}".replace("-", "−"), color=ORANGE)
panel_title(fig, [ax2], f"② 回报：把这一局的分打着折加成一个数（γ = {GAMMA:g}）")
panel_note(fig, [ax2], f"这是这一局的回报。换一局（第一局只走 {len(rewards_a)} 步）回报就不同："
                        f"{discounted_return(rewards_a, GAMMA):.3f}。")

ax3 = fig.add_axes([0.14, 0.075, 0.82, 0.25])
rounded = np.round(returns_right, 6)
values, counts = np.unique(rounded, return_counts=True)
ax3.bar(values, counts, width=0.036, color=BLUE, zorder=3)     # 相邻两种步数的回报差约 0.058，柱宽取它的六成
data_axes(ax3, "这一局的回报 $G_0$", f"局数（共 {N_EPISODES} 局）")
top = int(counts.max())
ax3.set_ylim(0, top * 1.28)
mean_right = float(returns_right.mean())
ax3.axvline(mean_right, color=INK, ls="--", lw=2, zorder=4)
ax3.text(mean_right - 0.03, top * 0.8, f"虚线：{N_EPISODES} 局的平均 {mean_right:.3f}", fontsize=FS_SMALL - 1, color=INK,
         ha="right", va="center", bbox=WHITE_BOX, zorder=5)
for n_label, who in ((len(rewards_a), "第一局"), (len(rewards_b), "第二局")):
    hit = lengths_right == n_label                  # 这两根柱子：柱顶标步数，接上 ①② 里的两局
    if hit.any():
        v = float(rounded[hit][0])
        c = int(np.sum(rounded == v))
        if who == "第一局":                          # 最高的那根：字直接写在柱顶
            ax3.text(v, c + top * 0.03, f"{n_label} 步\n（{who}）", fontsize=FS_SMALL - 2, color=ORANGE, ha="center",
                     va="bottom", linespacing=1.1, bbox=WHITE_BOX, zorder=6)
        else:                                       # 这根正好压着平均虚线：字写在左边，箭头指过去
            ax3.annotate(f"{n_label} 步（{who}）", xy=(v, c), xytext=(v - 0.18, c + top * 0.12), fontsize=FS_SMALL - 2,
                         color=ORANGE, ha="right", va="center", bbox=WHITE_BOX, zorder=6,
                         arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6))
negatives = values[values < 0]
if len(negatives):                                  # 负回报的局只有零星几局，柱子矮得看不见：在横轴上画小点
    ax3.plot(negatives, np.zeros_like(negatives), "o", color=ORANGE, ms=7, zorder=6, clip_on=False)
    ax3.annotate(f"回报是负的：{n_negative} 局，最低 {returns_right.min():.3f}\n（柱子矮得看不见，用小点标出）".replace("-", "−"),
                 xy=(float(negatives.min()), top * 0.02), xytext=(float(negatives.min()), top * 0.42),
                 fontsize=FS_SMALL - 2, color=ORANGE, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6))
ax3.set_xlim(min(-0.1, float(returns_right.min()) - 0.06), 0.95)
panel_title(fig, [ax3], "③ 同一个策略跑 2000 局：回报的平均（第 10 章叫它“价值”）")
panel_note(fig, [ax3], "每根柱子是一种步数（步数一样，回报就一样），柱高是有几局。\n"
                        f"平均 {mean_right:.3f}：从起点出发、每次都按 →，平均能拿这么多。")
savefig(fig, "ch09_reward_return_value")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("7c. 画图：figures/ch09_overview.png（9.0 节的总览：一步、一局、一个总分、很多局的平均）")
if len(rewards_a) > 6:
    print("  第一局太长（多半是改了打滑概率），总览图只照着正文那一局画，跳过。")
else:
    fig, axes = lesson_figure(4, "强化学习：一步一个分，一局一个总分，比的是很多局的平均", panel_height=2.95, width=9.6)
    ax = axes[0]
    lesson_panel(ax, "① 一步：按一下，环境掷骰子，再给一个分")
    for j in range(N_CELLS):
        cell(ax, 0.4 + 1.25 * j, 0.95, f"{j}", width=1.05, height=0.8, facecolor=CELL_HOT if j == 1 else CELL,
             edgecolor=GREEN if j == GOAL else CELL_EDGE)
    ax.text(0.4 + 0.525, 0.55, "起点", fontsize=FS_SMALL - 2, color=MUTED, ha="center", va="center")
    ax.text(0.4 + 1.25 * GOAL + 0.525, 0.55, "终点 +1", fontsize=FS_SMALL - 2, color=GREEN, ha="center", va="center")
    arrow(ax, (1.65 + 0.6, 1.85), (2.9 + 0.45, 1.85), BLUE, lw=2.6, connectionstyle="arc3,rad=-0.5")
    arrow(ax, (1.65 + 0.4, 1.85), (0.4 + 0.6, 1.85), ORANGE, lw=2.6, connectionstyle="arc3,rad=0.5")
    ax.text(3.45, 2.5, f"按 →：{(1 - P_SLIP) * 100:.0f}% 照办", fontsize=FS_SMALL - 1, color=BLUE, ha="left", va="center")
    ax.text(0.85, 2.5, f"{P_SLIP * 100:.0f}% 打滑", fontsize=FS_SMALL - 1, color=ORANGE, ha="right", va="center")
    hand(ax, 6.9, 1.35, f"没到终点：{num(STEP_REWARD, 2).replace('-', '−')}", color=INK, fontsize=FS_SMALL)
    ax = axes[1]
    lesson_panel(ax, "② 一局：从起点走到终点，每一步记一个分")
    shown = ["+1" if r > 0 else num(r, 2).replace("-", "−") for r in rewards_a]
    lesson_cells(ax, [shown], left=0.4, bottom=1.25, width=1.35, height=0.8, highlights={(0, len(shown) - 1)})
    for j in range(len(shown)):
        ax.text(0.4 + 1.35 * (j + 0.5), 2.4, f"第 {j} 步", fontsize=FS_SMALL - 1, color=MUTED, ha="center", va="center")
    note(ax, 6.2, 1.65, f"这一局 {len(shown)} 步，\n一次没打滑")
    ax = axes[2]
    lesson_panel(ax, "③ 一个总分：越远的分，打越多的折")
    hand(ax, 0.4, 2.35, f"直接加：{STEP_REWARD:g} × {len(rewards_a) - 1} + 1 = {G0_a[1.0]:.2f}".replace("-", "−"), color=INK)
    hand(ax, 0.4, 1.35, f"每远一步多乘一次 0.99：{G0_a[0.99]:.3f}", color=BLUE)
    note(ax, 0.4, 0.45, "打折以后，同样拿到 +1，拿得越晚，算进总分的越少。")
    ax = axes[3]
    lesson_panel(ax, "④ 很多局取平均：平均总分高的走法更好")
    for i, name in enumerate(("总是按 →", "随机乱按")):
        cell(ax, 0.4 + 4.6 * i, 1.3, f"{name}", width=2.3, height=0.85, facecolor=CELL)
        cell(ax, 2.7 + 4.6 * i, 1.3, f"{results[name][0].mean():.3f}", width=1.5, height=0.85,
             facecolor=CELL_HOT if i == 0 else CELL, color=ORANGE if i == 0 else INK)
    note(ax, 0.4, 0.5, f"各跑 {N_EPISODES} 局，每局的总分都不一样；比的是平均。")
    savefig(fig, "ch09_overview")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("8. 映射到项目：正文引用的常数和源码行还在不在")
REPO = Path(__file__).resolve().parents[3]
print(f"一步 = 4 个物理小步 × 0.005 秒 = {4 * 0.005:g} 秒（50 Hz）；一局最长 20 秒 = {math.ceil(20.0 / (4 * 0.005))} 步")
check("4 × 0.005 = 0.02 秒一步，1 秒 50 步；20 秒 = 1000 步", math.isclose(4 * 0.005, 0.02) and round(1 / 0.02) == 50
      and math.ceil(20.0 / (4 * 0.005)) == 1000)
cfg_path = REPO / "src" / "mjlab_microduck" / "tasks" / "microduck_velocity_env_cfg.py"
CFG_LINES = ['cfg.terminations["nan_state"] = TerminationTermCfg(', "time_out=False,",
             'del cfg.observations["actor"].terms["base_lin_vel"]',
             'cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(',
             "MicroduckRlCfg = RslRlOnPolicyRunnerCfg(", "gamma=0.99,", "num_steps_per_env=24,"]
if cfg_path.is_file():
    cfg_text = cfg_path.read_text(encoding="utf-8")
    print("microduck_velocity_env_cfg.py：数值出错也算终止；前进速度只给 critic、不给 actor；γ = 0.99；每 24 步训练一次")
    check("项目配置：nan_state（time_out=False）→ actor 删掉 base_lin_vel → critic 加上 → MicroduckRlCfg 的 gamma=0.99、"
          "num_steps_per_env=24", lines_in_order(cfg_text, CFG_LINES))
else:
    print("  （没找到项目的 env cfg，跳过这一项）")
mjlab_spec = importlib.util.find_spec("mjlab")
mjlab_dir = Path(mjlab_spec.origin).parent if mjlab_spec and mjlab_spec.origin else None
if mjlab_dir and (mjlab_dir / "tasks" / "velocity" / "velocity_env_cfg.py").is_file():
    vel_text = (mjlab_dir / "tasks" / "velocity" / "velocity_env_cfg.py").read_text(encoding="utf-8")
    VEL_LINES = ['"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),', '"fell_over": TerminationTermCfg(',
                 "func=mdp.bad_orientation,", 'params={"limit_angle": math.radians(70.0)},',
                 '"out_of_terrain_bounds": TerminationTermCfg(', "func=mdp.out_of_terrain_bounds,", "time_out=True,",
                 "timestep=0.005,", "decimation=4,", "episode_length_s=20.0,"]
    at = vel_text.find("def make_velocity_env_cfg(")
    print("mjlab/tasks/velocity/velocity_env_cfg.py 的 make_velocity_env_cfg()：三种收场 + 0.005 秒 × 4 + 20 秒")
    check(f"模板里的终止表和三个时间常数（共 {len(VEL_LINES)} 行）原样存在、顺序一致", at >= 0 and lines_in_order(vel_text[at:], VEL_LINES))
    term_text = (mjlab_dir / "envs" / "mdp" / "terminations.py").read_text(encoding="utf-8")
    check("mjlab/envs/mdp/terminations.py：time_out 比的是步数到没到上限；bad_orientation 算 arccos(−g_z) 和 70° 比",
          lines_in_order(term_text, ["def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:",
                                     "return env.episode_length_buf >= env.max_episode_length", "def bad_orientation(",
                                     "return torch.acos(-projected_gravity[:, 2]).abs() > limit_angle"]))
    env_text = (mjlab_dir / "envs" / "manager_based_rl_env.py").read_text(encoding="utf-8")
    check("mjlab/envs/manager_based_rl_env.py：一步 = 物理小步 × decimation；一局的步数 = 秒数 ÷ 一步的秒数，向上取整",
          lines_in_order(env_text, ["return self.cfg.sim.mujoco.timestep * self.cfg.decimation",
                                    "return math.ceil(self.max_episode_length_s / self.step_dt)"]))
else:
    print("  （当前 Python 环境里没有 mjlab，跳过源码核对；用 uv run 运行就会核对）")
rsl_spec = importlib.util.find_spec("rsl_rl")
ppo_path = Path(rsl_spec.origin).parent / "algorithms" / "ppo.py" if rsl_spec and rsl_spec.origin else None
if ppo_path and ppo_path.is_file():
    ppo_text = ppo_path.read_text(encoding="utf-8")
    check("rsl_rl/algorithms/ppo.py：超时的那一步，用 critic 的估计补上“后面本来还有的分”（第 12 章细讲）",
          lines_in_order(ppo_text, ['if "time_outs" in extras:', "self.transition.rewards += self.gamma * torch.squeeze(",
                                    'self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),']))
else:
    print("  （当前 Python 环境里没有 rsl_rl，跳过源码核对；用 uv run 运行就会核对）")

done()
