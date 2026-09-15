"""第 11 章实验：策略梯度。log-derivative trick 的数值验证、REINFORCE、基线降方差、高斯策略学习。

任务：一维“推小车”。观测 s ∈ (−1, 1) 是小车离目标的位置，动作 a 是推力，
      推完小车到 s + a，奖励 r = −(s + a)²。最优动作是 a = −s。
      这是一个“一步就结束”的最小 RL 问题（contextual bandit）。

运行：uv run python docs/learn-zh/labs/ch11_reinforce.py
纯 CPU。
"""

import math

import numpy as np
import torch

from _common import banner, check, done, savefig, table, use_headless_matplotlib

torch.manual_seed(0)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
banner("1. log-derivative trick 数值验证：∇_θ E[R] = E[R · ∇_θ log π(a)]")
# 策略：a ~ N(μ, σ=0.5)，只有一个旋钮 μ。奖励 R(a) = −(a − 1)²（目标 a = 1）。
sigma = 0.5


def R(a):
    return -((a - 1.0) ** 2)


def J(mu, n=400_000):
    a = mu + sigma * rng.standard_normal(n)
    return R(a).mean()


mu0 = 0.3
# 方法 1：有限差分（笨办法，但不依赖任何技巧）
h = 1e-2
grad_fd = (J(mu0 + h) - J(mu0 - h)) / (2 * h)
# 方法 2：log-derivative trick。∇_μ log N(a; μ, σ) = (a − μ)/σ²
a = mu0 + sigma * rng.standard_normal(400_000)
grad_lr = (R(a) * (a - mu0) / sigma**2).mean()
# 理论值：E[R] = −((μ−1)² + σ²)，导数 = −2(μ−1)
grad_true = -2 * (mu0 - 1.0)
table(["", "∂J/∂μ"], [["有限差分", grad_fd], ["log-derivative trick", grad_lr], ["理论值 −2(μ−1)", grad_true]])
check("log-derivative trick ≈ 理论值", abs(grad_lr - grad_true) < 0.05)
print("要点：我们没有对“采样”求导（那不可导），而是给每个样本的奖励乘上 ∇log π，再取平均。")

# ---------------------------------------------------------------------------
banner("2. 基线不改变期望，但大幅降低方差")
n_trials, n_samp = 200, 64
ests_nb, ests_b = [], []
for _ in range(n_trials):
    a = mu0 + sigma * rng.standard_normal(n_samp)
    score = (a - mu0) / sigma**2
    ests_nb.append((R(a) * score).mean())                  # 无基线
    ests_b.append(((R(a) - R(a).mean()) * score).mean())   # 减去平均奖励当基线
table(["估计器（每次 64 个样本）", "均值", "标准差"], [["R·∇logπ", np.mean(ests_nb), np.std(ests_nb)], ["(R − b)·∇logπ", np.mean(ests_b), np.std(ests_b)]])
check("两者均值都 ≈ 理论值", abs(np.mean(ests_nb) - grad_true) < 0.15 and abs(np.mean(ests_b) - grad_true) < 0.15)
check("有基线的标准差更小", np.std(ests_b) < np.std(ests_nb))
print("为什么均值不变：E[b · ∇logπ] = b · ∇ E[1] = b · 0 = 0。（第 4 章的期望线性。）")

# ---------------------------------------------------------------------------
banner("3. REINFORCE 训练一个高斯策略：μ_θ(s) 是一个小网络，σ 是一个可学参数")
net = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.ELU(), torch.nn.Linear(16, 1))
log_std = torch.nn.Parameter(torch.zeros(1))          # σ = exp(log_std)，初值 1.0
opt = torch.optim.Adam(list(net.parameters()) + [log_std], lr=3e-3)
rows = []
for it in range(1501):
    s = torch.rand(256, 1) * 2 - 1                     # 256 个随机观测
    mu = net(s)
    std = log_std.exp().expand_as(mu)
    dist = torch.distributions.Normal(mu, std)
    a = dist.sample()                                  # 采样动作（探索）
    r = -((s + a) ** 2).squeeze(1)                     # 奖励
    logp = dist.log_prob(a).squeeze(1)                 # log π(a|s)
    adv = r - r.mean()                                 # 减基线
    loss = -(adv.detach() * logp).mean()               # ★ 策略梯度损失：−E[A · log π]
    opt.zero_grad(); loss.backward(); opt.step()
    if it in (0, 50, 200, 500, 1000, 1500):
        with torch.no_grad():
            test_s = torch.tensor([[-0.8], [0.0], [0.6]])
            rows.append([it, r.mean().item(), std[0].item()] + [net(x.unsqueeze(0)).item() for x in test_s])
table(["迭代", "平均奖励", "σ", "μ(s=−0.8)", "μ(s=0)", "μ(s=0.6)"], rows)
print("最优是 μ(s) = −s：μ(−0.8)→0.8，μ(0)→0，μ(0.6)→−0.6。")
check("学到 μ(s) ≈ −s（误差 < 0.1）", abs(rows[-1][3] - 0.8) < 0.1 and abs(rows[-1][5] + 0.6) < 0.1)
check("σ 在训练中缩小（越来越确定）", rows[-1][2] < rows[0][2])

plt = use_headless_matplotlib()
fig, ax = plt.subplots(figsize=(5.5, 3.4))
ss = torch.linspace(-1, 1, 100).unsqueeze(1)
with torch.no_grad():
    ax.plot(ss, net(ss), label="学到的 μ(s)")
ax.plot(ss, -ss, "--", label="最优 −s")
ax.set_xlabel("观测 s"); ax.set_ylabel("动作均值"); ax.legend(); ax.grid(alpha=0.3)
ax.set_title("REINFORCE 学到的策略")
savefig(fig, "ch11_reinforce_policy")

# ---------------------------------------------------------------------------
banner("4. 熵奖励：加上 −c·H 项后 σ 不会缩到 0")
print(f"训练结束时 σ = {log_std.exp().item():.3f}；14 维高斯的熵公式 H = Σ ½ln(2πeσ²)（第 4 章）。")
print("项目里 entropy_coef=0.01：损失里减去 0.01×熵，鼓励 σ 别太快缩小。")

done()
