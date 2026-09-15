"""第 4 章实验：期望、方差、大数定律、均匀分布、高斯分布、log-prob、熵。

运行：uv run python docs/learn-zh/labs/ch04_probability.py
纯 CPU，numpy + torch。
"""

import math

import numpy as np
import torch

from _common import banner, check, done, savefig, table, use_headless_matplotlib

torch.manual_seed(0)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
banner("1. 期望 = 概率加权平均：掷一个骰子")
values = np.arange(1, 7)
probs = np.full(6, 1 / 6)
E = np.sum(values * probs)
Var = np.sum(probs * (values - E) ** 2)
print("E[X]   = Σ x·p(x) = (1+2+3+4+5+6)/6 =", E)
print("Var[X] = Σ p(x)·(x−E)² =", Var, " 标准差 σ =", math.sqrt(Var))
check("骰子期望 3.5", abs(E - 3.5) < 1e-12)

# ---------------------------------------------------------------------------
banner("2. 大数定律：样本均值 → 期望（项目 DR 的电池电压 vin ~ U(6.5, 8.2)）")
lo, hi = 6.5, 8.2
E_true = (lo + hi) / 2
Var_true = (hi - lo) ** 2 / 12
rows = []
for n in [1, 10, 100, 1000, 10000, 100000]:
    s = rng.uniform(lo, hi, size=n)
    rows.append([n, s.mean(), s.mean() - E_true, s.var()])
table(["样本数 n", "样本均值", "与期望的差", "样本方差"], rows)
print(f"理论：E = (a+b)/2 = {E_true}，Var = (b−a)²/12 = {Var_true:.4f}")
check("十万个样本的均值误差 < 0.01", abs(rows[-1][2]) < 0.01)

# ---------------------------------------------------------------------------
banner("3. 高斯分布密度：手写公式 vs torch")


def gauss_pdf(x, mu, sigma):
    return 1.0 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-((x - mu) ** 2) / (2 * sigma**2))


mu, sigma = 0.0, 1.0
rows = []
for x in [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]:
    p = gauss_pdf(x, mu, sigma)
    p_t = torch.distributions.Normal(mu, sigma).log_prob(torch.tensor(x)).exp().item()
    rows.append([x, p, p_t, math.log(p)])
table(["x", "p(x) 手写", "p(x) torch", "log p(x)"], rows)
check("手写高斯 = torch.Normal", all(abs(r[1] - r[2]) < 1e-6 for r in rows))

# ---------------------------------------------------------------------------
banner("4. 项目 actor 的高斯策略：14 维动作，std 是可学参数（init_std=1.0）")
mean = torch.randn(14) * 0.3           # 假装是网络输出的 14 个均值
std = torch.ones(14)                   # distribution.py:157  std_param = 1.0 * ones(14)
dist = torch.distributions.Normal(mean, std)
eps = torch.randn(14)
action = mean + std * eps              # 采样 = 均值 + 标准差 × 标准正态噪声
print("mean[:4]   =", mean[:4].numpy())
print("eps[:4]    =", eps[:4].numpy())
print("action[:4] =", action[:4].numpy())

# log-prob 逐维相加（distribution.py:217  .sum(dim=-1)）
logp_manual = sum(math.log(gauss_pdf(action[i].item(), mean[i].item(), std[i].item())) for i in range(14))
logp_torch = dist.log_prob(action).sum(-1).item()
table(["", "log π(a|s)"], [["手写 Σ log p_i", logp_manual], ["torch .sum(-1)", logp_torch]])
check("14 维 log-prob 手写 = torch", abs(logp_manual - logp_torch) < 1e-5)
print("为什么 14 个概率相加而不是相乘？因为取了 log：log(p₁·p₂…) = log p₁ + log p₂ + …")

# 熵：一维高斯 H = ½ ln(2πe σ²)，14 维独立就相加
H_manual = 14 * 0.5 * math.log(2 * math.pi * math.e * 1.0**2)
H_torch = dist.entropy().sum(-1).item()
table(["", "熵 H"], [["手写 14·½ln(2πeσ²)", H_manual], ["torch .entropy().sum(-1)", H_torch]])
check("熵手写 = torch", abs(H_manual - H_torch) < 1e-5)
rows = [[s, 0.5 * math.log(2 * math.pi * math.e * s**2)] for s in [0.1, 0.5, 1.0, 2.0]]
table(["σ", "一维熵"], rows)
print("σ 越大熵越大 → “更随机、更爱探索”。wandb 里的 Mean noise std 就是这个 σ。")

# ---------------------------------------------------------------------------
banner("5. 条件概率：π(a|s) 的“|”读作“在给定 s 的条件下”")
print("同一个策略网络，不同的观测 s 给出不同的均值 μ(s)，动作分布就随 s 变：")
for k in range(2):
    s = torch.randn(61)
    mu_s = torch.tanh(s[:14] * 0.5)      # 用一个随便的函数模拟 μ(s)
    print(f"  观测 s{k}: μ(s)[:3] = {mu_s[:3].numpy()}")

banner("6. 画图：高斯分布的 68% / 95%")
plt = use_headless_matplotlib()
fig, ax = plt.subplots(figsize=(6, 3.4))
xs = np.linspace(-4, 4, 400)
pdf = np.exp(-xs**2 / 2) / math.sqrt(2 * math.pi)
ax.plot(xs, pdf, color="black")
ax.fill_between(xs, pdf, where=np.abs(xs) <= 1, color="tab:blue", alpha=0.45, label="μ±1σ 内：68%")
ax.fill_between(xs, pdf, where=(np.abs(xs) > 1) & (np.abs(xs) <= 2), color="tab:blue", alpha=0.2, label="1σ 到 2σ：再加 27%")
ax.set_xlabel("x（这里 μ=0, σ=1）"); ax.set_ylabel("密度 p(x)")
ax.set_title("高斯分布：钟形，面积总共是 1")
ax.legend(); ax.grid(alpha=0.3)
savefig(fig, "ch04_gaussian_68_95")

done()
