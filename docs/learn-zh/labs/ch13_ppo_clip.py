"""第 13 章实验：PPO 的裁剪目标。比率 r = π_new/π_old、裁剪的形状、被裁剪比例、价值损失、总损失、
自适应 KL 学习率——全部用合成数据一步步算出来，并与 rsl_rl ppo.py 的写法逐行对应。

运行：uv run python docs/learn-zh/labs/ch13_ppo_clip.py
纯 CPU。
"""

import numpy as np
import torch

from _common import banner, check, done, savefig, table, use_headless_matplotlib

torch.manual_seed(0)
np.set_printoptions(precision=4, suppress=True)
CLIP, ENT_COEF, VF_COEF = 0.2, 0.01, 1.0

# ---------------------------------------------------------------------------
banner("1. 比率 r = π_new(a|s) / π_old(a|s) = exp(log π_new − log π_old)")
old_logp = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0])
new_logp = torch.tensor([-1.0, -0.8, -1.4, -0.5, -1.2])
ratio = torch.exp(new_logp - old_logp)
table(["log π_old", "log π_new", "ratio"], [[o.item(), n.item(), rt.item()] for o, n, rt in zip(old_logp, new_logp, ratio)])
print("ratio > 1：新策略更爱这个动作；< 1：更不爱。ratio = 1：没变。")

# ---------------------------------------------------------------------------
banner("2. 裁剪目标 L = min( r·Â,  clip(r, 1−ε, 1+ε)·Â )，ε = 0.2")


def clipped_objective(ratio, adv, eps=CLIP):
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    return torch.min(unclipped, clipped)


rows = []
for rt in (0.5, 0.8, 1.0, 1.2, 1.5):
    rt_t = torch.tensor(rt)
    rows.append([rt, clipped_objective(rt_t, torch.tensor(1.0)).item(), clipped_objective(rt_t, torch.tensor(-1.0)).item()])
table(["ratio", "Â=+1 时的目标", "Â=−1 时的目标"], rows)
print("Â>0（好动作）：ratio 涨到 1.2 以后目标不再涨——“别再更爱它了”。")
print("Â<0（坏动作）：ratio 跌到 0.8 以后目标不再涨——“别再更讨厌它了”。")

plt = use_headless_matplotlib()
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
rr = torch.linspace(0.4, 1.6, 200)
for ax, adv, title in ((axes[0], 1.0, "Â = +1（好动作）"), (axes[1], -1.0, "Â = −1（坏动作）")):
    ax.plot(rr, rr * adv, "--", color="gray", label="不裁剪 r·Â")
    ax.plot(rr, clipped_objective(rr, torch.tensor(adv)), color="tab:blue", lw=2, label="PPO 目标")
    ax.axvline(1 - CLIP, color="r", lw=0.5); ax.axvline(1 + CLIP, color="r", lw=0.5)
    ax.set_xlabel("ratio"); ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=8)
savefig(fig, "ch13_ppo_clip")

# ---------------------------------------------------------------------------
banner("3. rsl_rl 的写法：取负号变成“损失”，min 变 max")
adv = torch.tensor([1.0, 0.5, -0.5, 2.0, -1.0])
# ppo.py:297-302
surrogate = -adv * ratio
surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP)
surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
# 教科书写法
objective = clipped_objective(ratio, adv).mean()
table(["", "值"], [["教科书目标（要最大化）", objective.item()], ["rsl_rl surrogate_loss（要最小化）", surrogate_loss.item()]])
check("rsl_rl 的 max(−…) = −教科书的 min(…)", abs(surrogate_loss.item() + objective.item()) < 1e-6)

# ---------------------------------------------------------------------------
banner("4. 一个 mini-batch 里有多少样本被裁剪了")
N = 10_000
old_lp = torch.randn(N) * 0.5 - 20
new_lp = old_lp + torch.randn(N) * 0.15          # 新策略比旧策略稍微变了一点
ratio_b = torch.exp(new_lp - old_lp)
adv_b = torch.randn(N)
clipped_mask = ((ratio_b > 1 + CLIP) & (adv_b > 0)) | ((ratio_b < 1 - CLIP) & (adv_b < 0))
print(f"ratio 的范围：{ratio_b.min():.3f} ~ {ratio_b.max():.3f}，均值 {ratio_b.mean():.3f}")
print(f"代理项处于平坦区的样本比例：{clipped_mask.float().mean()*100:.1f}%")
print("高比例提示需核对 KL 与更新幅度；低比例也可能只是更新保守，不能单独判断好坏。")

# ---------------------------------------------------------------------------
banner("5. 价值损失（clipped）+ 熵 → 总损失，和 ppo.py:304-313 一样")
values_old = torch.tensor([0.5, 0.6, 0.7, 0.4, 0.3])      # rollout 时 critic 的估值
values_new = torch.tensor([0.55, 0.9, 0.65, 0.1, 0.35])   # 更新后 critic 的估值
returns = torch.tensor([0.6, 0.7, 0.6, 0.5, 0.2])         # 第 12 章的 returns = Â + V
value_clipped = values_old + (values_new - values_old).clamp(-CLIP, CLIP)
value_losses = (values_new - returns).pow(2)
value_losses_clipped = (value_clipped - returns).pow(2)
value_loss = torch.max(value_losses, value_losses_clipped).mean()
entropy = torch.tensor([19.8, 19.7, 19.9, 19.6, 19.8])    # 14 维高斯、σ≈1 时的熵（第 4 章）
loss = surrogate_loss + VF_COEF * value_loss - ENT_COEF * entropy.mean()
table(["项", "值", "系数"], [["surrogate_loss", surrogate_loss.item(), 1], ["value_loss", value_loss.item(), VF_COEF], ["entropy (减去)", entropy.mean().item(), ENT_COEF], ["总损失 loss", loss.item(), ""]])
check("总损失 = surrogate + 1.0·value − 0.01·entropy", abs(loss.item() - (surrogate_loss.item() + value_loss.item() - 0.01 * entropy.mean().item())) < 1e-6)

# ---------------------------------------------------------------------------
banner("6. 自适应 KL 学习率（ppo.py:280-284）：KL 太大 ÷1.5，太小 ×1.5")
desired_kl = 0.01
lr = 1e-3
rows = []
for kl in (0.03, 0.025, 0.012, 0.004, 0.003, 0.02):
    if kl > desired_kl * 2.0:
        lr = max(1e-5, lr / 1.5)
    elif kl < desired_kl / 2.0 and kl > 0.0:
        lr = min(1e-2, lr * 1.5)
    rows.append([kl, lr])
table(["这次更新测到的 KL", "调整后的学习率"], rows)
print("KL 是“新旧两口钟差多远”的尺子（两个高斯之间有闭式公式，torch 直接算）。目标 0.01，允许在 0.005–0.02 之间晃。")

# 两个一维高斯的 KL，手算 vs torch
mu1, s1, mu2, s2 = 0.0, 1.0, 0.3, 0.8
kl_manual = np.log(s2 / s1) + (s1**2 + (mu1 - mu2) ** 2) / (2 * s2**2) - 0.5
kl_torch = torch.distributions.kl_divergence(torch.distributions.Normal(mu1, s1), torch.distributions.Normal(mu2, s2)).item()
table(["", "KL(old‖new)"], [["手算公式", kl_manual], ["torch", kl_torch]])
check("KL 手算 = torch", abs(kl_manual - kl_torch) < 1e-6)

# ---------------------------------------------------------------------------
banner("7. 样本量算术：4096 env × 24 步 = 98,304；÷4 = 24,576 每批；×5 epoch = 20 次更新")
n_env, T, n_mb, n_ep = 4096, 24, 4, 5
print(f"batch = {n_env*T:,}，mini_batch = {n_env*T//n_mb:,}，每次 rollout 更新 {n_mb*n_ep} 次")
check("98304 / 4 = 24576", n_env * T // n_mb == 24576)

done()
