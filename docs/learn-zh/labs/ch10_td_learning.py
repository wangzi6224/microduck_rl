"""第 10 章实验：价值函数 V(s)、贝尔曼方程、蒙特卡洛 vs TD、优势 A = Q − V。
用第 9 章的走廊世界，把 V 用三种方法算出来对比。

运行：uv run python docs/learn-zh/labs/ch10_td_learning.py
纯 CPU。
"""

import numpy as np

from _common import banner, check, done, table

rng = np.random.default_rng(0)
N, GOAL, GAMMA = 5, 4, 0.9          # 用 γ=0.9 让数字看得清


def step(s, a):
    move = a if rng.uniform() < 0.8 else -a
    s2 = int(np.clip(s + move, 0, N - 1))
    done = s2 == GOAL
    return s2, (1.0 if done else -0.05), done


def policy(s):
    """固定策略 π：70% 向右，30% 向左。我们要算的是“在这个策略下每个格子值多少”。"""
    return +1 if rng.uniform() < 0.7 else -1


# ---------------------------------------------------------------------------
banner("1. 方法 A：蒙特卡洛。从每个格子出发跑很多回合，回报取平均")


def run_from(s0, max_t=200):
    s, rews = s0, []
    for _ in range(max_t):
        s, r, d = step(s, policy(s))
        rews.append(r)
        if d:
            break
    G = 0.0
    for r in reversed(rews):
        G = r + GAMMA * G
    return G


V_mc = np.array([np.mean([run_from(s) for _ in range(3000)]) for s in range(N - 1)] + [0.0])
print("V_MC =", np.round(V_mc, 3), "（终点格值 0：到了就结束，之后没有奖励）")

# ---------------------------------------------------------------------------
banner("2. 方法 B：贝尔曼方程精确解。V(s) = Σ_a π(a|s) Σ_s' P(s'|s,a) [ r + γ V(s') ]")
# 先把转移概率表写出来：P[s, a_idx, s'] 和 R[s, a_idx, s']
A = [-1, +1]
P = np.zeros((N, 2, N))
R = np.zeros((N, 2, N))
for s in range(N - 1):
    for ai, a in enumerate(A):
        for move, prob in ((a, 0.8), (-a, 0.2)):
            s2 = int(np.clip(s + move, 0, N - 1))
            P[s, ai, s2] += prob
            R[s, ai, s2] = 1.0 if s2 == GOAL else -0.05
pi = np.array([0.3, 0.7])       # π(左), π(右)

# 贝尔曼方程是一组线性方程：V = b + γ M V  →  (I − γM) V = b
M = np.zeros((N, N))
b = np.zeros(N)
for s in range(N - 1):
    for ai in range(2):
        for s2 in range(N):
            M[s, s2] += pi[ai] * P[s, ai, s2]
            b[s] += pi[ai] * P[s, ai, s2] * R[s, ai, s2]
V_exact = np.linalg.solve(np.eye(N) - GAMMA * M, b)
V_exact[GOAL] = 0.0
print("V_exact =", np.round(V_exact, 3))
table(["格子 s", "蒙特卡洛 V", "贝尔曼精确 V", "差"], [[s, V_mc[s], V_exact[s], V_mc[s] - V_exact[s]] for s in range(N)])
check("蒙特卡洛 ≈ 精确解（误差 < 0.03）", np.max(np.abs(V_mc - V_exact)) < 0.03)

# 验证贝尔曼方程在精确解上成立：左边 V(s) = 右边 Σ...
s = 1
rhs = sum(pi[ai] * sum(P[s, ai, s2] * (R[s, ai, s2] + GAMMA * V_exact[s2]) for s2 in range(N)) for ai in range(2))
print(f"格子 1：V(1) = {V_exact[1]:.4f}，贝尔曼右边 = {rhs:.4f}")
check("贝尔曼方程成立", abs(rhs - V_exact[1]) < 1e-10)

# ---------------------------------------------------------------------------
banner("3. 方法 C：TD 学习。不等回合结束，每走一步就用 r + γV(s') 修正 V(s)")
V_td = np.zeros(N)
alpha = 0.05
rows = []
for ep in range(4000):
    s = rng.integers(0, N - 1)
    for _ in range(200):
        s2, r, d = step(s, policy(s))
        target = r + (0.0 if d else GAMMA * V_td[s2])      # 用“下一步的估计”当目标（自举）
        delta = target - V_td[s]                            # TD 误差 δ
        V_td[s] += alpha * delta                            # 往目标挪一点
        s = s2
        if d:
            break
    if ep in (0, 10, 100, 1000, 3999):
        rows.append([ep] + list(np.round(V_td[:4], 3)))
table(["回合数", "V(0)", "V(1)", "V(2)", "V(3)"], rows)
print("精确值     ", np.round(V_exact[:4], 3))
check("TD 收敛到精确解附近（误差 < 0.05）", np.max(np.abs(V_td - V_exact)) < 0.05)
print("TD 每一步都能学，不用等回合结束；代价是目标里用了自己的估计（“自举”），早期会偏。")

# ---------------------------------------------------------------------------
banner("4. Q(s,a) 和优势 A(s,a) = Q(s,a) − V(s)：在格子 2，向右比向左好多少")
s = 2
Q = np.array([sum(P[s, ai, s2] * (R[s, ai, s2] + GAMMA * V_exact[s2]) for s2 in range(N)) for ai in range(2)])
V_s = pi @ Q
table(["动作", "Q(2,a)", "A(2,a) = Q − V"], [["向左", Q[0], Q[0] - V_s], ["向右", Q[1], Q[1] - V_s]])
print(f"V(2) = π-加权的 Q = {V_s:.4f}；优势为正的动作“比平均好”，为负的“比平均差”。")
check("V = Σ π(a) Q(s,a)", abs(V_s - V_exact[2]) < 1e-10)
check("优势按 π 加权平均为 0", abs(pi @ (Q - V_s)) < 1e-12)

done()
