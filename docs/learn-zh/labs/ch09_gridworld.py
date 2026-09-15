"""第 9 章实验：一个 5 格的“走廊世界”——用最小的例子看清 状态/动作/奖励/回合/回报/折扣。

运行：uv run python docs/learn-zh/labs/ch09_gridworld.py
纯 CPU。
"""

import numpy as np

from _common import banner, check, done, table

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
banner("1. 环境：5 个格子，从 0 出发，走到 4 拿到 +1 结束；每走一步扣 0.05")
N_STATES = 5
GOAL = 4


def step(s, a):
    """a = +1 向右, -1 向左。环境有点滑：80% 按你说的走，20% 反着走。"""
    move = a if rng.uniform() < 0.8 else -a
    s2 = int(np.clip(s + move, 0, N_STATES - 1))
    done = s2 == GOAL
    r = 1.0 if done else -0.05
    return s2, r, done


print("状态 s ∈ {0,1,2,3,4}，动作 a ∈ {−1,+1}，终点 4。")
print("转移 P(s'|s,a)：80% 如愿，20% 打滑反向。这就是“环境的随机性”。")

# ---------------------------------------------------------------------------
banner("2. 跑一个回合（策略：总是向右）")
s, t, traj = 0, 0, []
while True:
    a = +1
    s2, r, d = step(s, a)
    traj.append((t, s, a, r, s2))
    s, t = s2, t + 1
    if d or t >= 50:
        break
table(["t", "s_t", "a_t", "r_t", "s_{t+1}"], [list(x) for x in traj])
print(f"回合长度 T = {len(traj)} 步。这一串 (s, a, r) 就是一条“轨迹” τ。")

# ---------------------------------------------------------------------------
banner("3. 回报 G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …（从第 t 步往后的“折扣总分”）")
rewards = np.array([x[3] for x in traj])


def returns(rews, gamma):
    G = np.zeros(len(rews))
    running = 0.0
    for t in reversed(range(len(rews))):        # 从后往前累加最省事：G_t = r_t + γ G_{t+1}
        running = rews[t] + gamma * running
        G[t] = running
    return G


rows = []
for g in (1.0, 0.99, 0.9, 0.5):
    G = returns(rewards, g)
    rows.append([g, G[0], G[len(G) // 2], G[-1]])
table(["γ", "G_0 (开头)", "G_中间", "G_末尾"], rows)
print("γ 越小，远处的 +1 折得越狠，开头那一步看到的回报就越小。")
G099 = returns(rewards, 0.99)
manual = sum((0.99**k) * rewards[k] for k in range(len(rewards)))
check("G_0 手算 Σγ^k r_k = 从后往前递推", abs(G099[0] - manual) < 1e-12)
check("G_末尾 = 最后一步的奖励 1.0", G099[-1] == 1.0)

# ---------------------------------------------------------------------------
banner("4. 折扣因子 γ 的“视界”：γ^k 衰减表，项目 γ = 0.99")
rows = []
for k in (1, 10, 50, 100, 200, 500):
    rows.append([k, k * 0.02, 0.99**k, 0.9**k])
table(["k 步后", "= 几秒 (50 Hz)", "0.99^k", "0.9^k"], rows)
print(f"1/(1−γ) = {1/(1-0.99):.0f} 步 ≈ 2 秒：γ=0.99 时，2 秒后的奖励只剩 e^{-1} ≈ 37%，10 秒后几乎为 0。")
check("100 步后 0.99^100 ≈ 0.366", abs(0.99**100 - 0.366) < 0.001)

# ---------------------------------------------------------------------------
banner("5. 两种策略、各跑 2000 回合，比平均回报（G_0 的期望，用样本平均估）")


def run_episode(policy):
    s, t, rews = 0, 0, []
    while True:
        a = policy(s)
        s, r, d = step(s, a)
        rews.append(r)
        t += 1
        if d or t >= 50:
            return returns(np.array(rews), 0.99)[0], t


for name, pol in [("总是向右", lambda s: +1), ("随机乱走", lambda s: rng.choice([-1, +1]))]:
    Gs, Ts = zip(*[run_episode(pol) for _ in range(2000)])
    print(f"  {name}: 平均回报 {np.mean(Gs):.3f}，平均回合长度 {np.mean(Ts):.1f} 步")
print("“好策略”= 平均回报高的策略。RL 的目标就是找到它。")

done()
