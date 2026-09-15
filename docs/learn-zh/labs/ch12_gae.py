"""第 12 章实验：n 步回报、TD 误差 δ、GAE 公式，与 rsl_rl 的 compute_returns 逻辑逐位对比；
λ 的两个极端；time_out 自举；优势归一化。

运行：uv run python docs/learn-zh/labs/ch12_gae.py
纯 CPU。
"""

import numpy as np
import torch

from _common import banner, check, done, savefig, table, use_headless_matplotlib

torch.manual_seed(0)
rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True)
GAMMA, LAM = 0.99, 0.95

# ---------------------------------------------------------------------------
banner("1. 造一段 T=8 步的数据：奖励 r_t、critic 的估值 V_t、是否结束 done_t")
T = 8
r = np.array([0.1, 0.1, 0.1, 1.0, 0.1, 0.1, -1.0, 0.1])
V = np.array([0.5, 0.6, 0.7, 0.4, 0.3, 0.2, 0.1, 0.3])
dones = np.array([0, 0, 0, 1, 0, 0, 1, 0])  # 第 3 步和第 6 步回合结束
V_last = 0.35                               # 第 8 步（rollout 之后那一步）的估值
table(["t", "r_t", "V_t", "done_t"], [[t, r[t], V[t], dones[t]] for t in range(T)])

# ---------------------------------------------------------------------------
banner("2. TD 误差 δ_t = r_t + γ V_{t+1} − V_t（回合结束时 V_{t+1} 当 0）")
V_next = np.append(V[1:], V_last)
delta = r + (1 - dones) * GAMMA * V_next - V
table(["t", "δ_t"], [[t, delta[t]] for t in range(T)])

# ---------------------------------------------------------------------------
banner("3. GAE：Â_t = δ_t + (γλ) δ_{t+1} + (γλ)² δ_{t+2} + …（遇到 done 截断）")


def gae_explicit(delta, done, gamma, lam):
    """按定义直接求和：对每个 t，往后累加 (γλ)^k δ_{t+k}，碰到 done 停止。"""
    T = len(delta)
    A = np.zeros(T)
    for t in range(T):
        coef, acc = 1.0, 0.0
        for k in range(t, T):
            acc += coef * delta[k]
            if done[k]:
                break
            coef *= gamma * lam
        A[t] = acc
    return A


def gae_rsl_rl(r, V, done, V_last, gamma, lam):
    """和 rsl_rl/algorithms/ppo.py compute_returns 的循环一模一样：从后往前递推。"""
    T = len(r)
    returns = np.zeros(T)
    advantage = 0.0
    for step in reversed(range(T)):
        next_values = V_last if step == T - 1 else V[step + 1]
        next_is_not_terminal = 1.0 - done[step]
        delta = r[step] + next_is_not_terminal * gamma * next_values - V[step]
        advantage = delta + next_is_not_terminal * gamma * lam * advantage
        returns[step] = advantage + V[step]
    advantages = returns - V
    return advantages, returns


A_def = gae_explicit(delta, dones, GAMMA, LAM)
A_rsl, ret_rsl = gae_rsl_rl(r, V, dones, V_last, GAMMA, LAM)
table(["t", "Â (按定义求和)", "Â (rsl_rl 递推)", "returns = Â + V"], [[t, A_def[t], A_rsl[t], ret_rsl[t]] for t in range(T)])
check("按定义求和 = rsl_rl 递推", np.allclose(A_def, A_rsl))
print("递推 Â_t = δ_t + γλ Â_{t+1} 和逐项求和是同一个东西，只是从后往前算一遍就够，O(T)。")

# ---------------------------------------------------------------------------
banner("4. λ 的两个极端")
A_l0, _ = gae_rsl_rl(r, V, dones, V_last, GAMMA, 0.0)
A_l1, ret_l1 = gae_rsl_rl(r, V, dones, V_last, GAMMA, 1.0)
# λ=1 时 returns 应该等于“蒙特卡洛回报”（真实折扣总和 + 末尾自举）
G = np.zeros(T); running = V_last
for t in reversed(range(T)):
    running = r[t] + (1 - dones[t]) * GAMMA * running
    G[t] = running
table(["t", "λ=0: Â=δ_t", "λ=1: Â", "λ=1: returns", "蒙特卡洛回报 G_t"], [[t, A_l0[t], A_l1[t], ret_l1[t], G[t]] for t in range(T)])
check("λ=0 时 Â = δ（一步 TD）", np.allclose(A_l0, delta))
check("λ=1 时 returns = 蒙特卡洛回报", np.allclose(ret_l1, G))

# ---------------------------------------------------------------------------
banner("5. 方差随 λ 增大：奖励带噪声时，不同 λ 的 Â_0 散得多开（critic 用真值）")
T2, trials = 24, 300
V_true = np.array([sum(GAMMA**k for k in range(T2 - t)) for t in range(T2)])   # 每步奖励恰好 1 时的真实回报
rows = []
for lam in (0.0, 0.5, 0.95, 1.0):
    ests = []
    for _ in range(trials):
        rr = 1.0 + rng.normal(0, 0.5, T2)                   # 奖励噪声
        A_, _ = gae_rsl_rl(rr, V_true, np.zeros(T2), 0.0, GAMMA, lam)
        ests.append(A_[0])
    rows.append([lam, np.mean(ests), np.std(ests)])
table(["λ", "Â_0 的均值", "Â_0 的标准差"], rows)
check("λ 越大标准差越大", rows[0][2] < rows[1][2] < rows[2][2] < rows[3][2])
print("λ=0 只用一步真实奖励，其余全靠 critic → 噪声最小，但 critic 一错它就跟着错。")
print("λ=1 把 24 步真实奖励全加起来 → 不依赖 critic 的中间估值，但 24 步噪声全进来了。0.95 是折中。")

# ---------------------------------------------------------------------------
banner("6. time_out 自举：超时截断 ≠ 失败终止")
# 超时那一步，rsl_rl 把 γ·V(s) 加回奖励（ppo.py process_env_step），等价于“回合其实还能继续”
r_t, V_t, gamma = 0.1, 0.8, GAMMA
print(f"某一步超时：原始 r = {r_t}，critic 估 V = {V_t}")
print(f"  rsl_rl 修正后的 r = r + γV = {r_t + gamma * V_t:.4f}")
print("  然后 done=1 截断。效果：这一步的 δ = (r + γV) + 0 − V(s_t)，好像后面还有 V 那么多分——不惩罚“只是时间到了”。")
print("  摔倒（fell_over）没有这一步：δ = r + 0 − V，后面真的没分了。")

# ---------------------------------------------------------------------------
banner("7. 优势归一化：(Â − mean) / (std + 1e-8)")
A_norm = (A_rsl - A_rsl.mean()) / (A_rsl.std() + 1e-8)
print("归一化后均值 =", round(A_norm.mean(), 6), " 标准差 =", round(A_norm.std(), 4))
check("归一化后均值 0、标准差 1", abs(A_norm.mean()) < 1e-8 and abs(A_norm.std() - 1) < 1e-6)
print("这样无论奖励的绝对尺度是 0.01 还是 100，送进 PPO 的“优势”都在 ±1 量级，学习率才有统一的意义。")

plt = use_headless_matplotlib()
fig, ax = plt.subplots(figsize=(6, 3.4))
for lam, c in ((0.0, "tab:red"), (0.95, "tab:blue"), (1.0, "tab:green")):
    A_, _ = gae_rsl_rl(r, V, dones, V_last, GAMMA, lam)
    ax.plot(range(T), A_, "o-", color=c, label=f"λ={lam}")
ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("t"); ax.set_ylabel("Â_t"); ax.legend(); ax.grid(alpha=0.3)
ax.set_title("同一段数据，不同 λ 的优势估计")
savefig(fig, "ch12_gae_lambda")

done()
