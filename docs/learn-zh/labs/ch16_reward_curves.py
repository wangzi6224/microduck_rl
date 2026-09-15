"""第 16 章实验：画出 velocity 任务里每一种奖励核的曲线（CPU），并用 4 个环境跑 50 步验证每一项的符号（🖥️GPU 部分可选）。

运行：uv run python docs/learn-zh/labs/ch16_reward_curves.py            # 只画曲线（CPU）
      uv run python docs/learn-zh/labs/ch16_reward_curves.py --env      # 再建环境跑 50 步（需要 GPU）
"""

import math
import sys

import numpy as np

from _common import banner, check, done, savefig, table, use_headless_matplotlib

# ---------------------------------------------------------------------------
banner("1. 从真实 cfg 读出全部奖励项：名字、权重、关键参数")
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg  # noqa: E402

cfg = make_microduck_velocity_env_cfg()
rows = []
for name, term in cfg.rewards.items():
    keyp = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in term.params.items() if k in ("std", "threshold_min", "threshold_max", "target_height", "tau_s", "nominal_height")}
    fn = term.func.__name__ if hasattr(term.func, "__name__") else type(term.func).__name__
    rows.append([name, fn, term.weight, str(keyp)])
table(["奖励项", "函数", "权重", "关键参数"], rows)
pos = [n for n, t in cfg.rewards.items() if t.weight > 0]
neg = [n for n, t in cfg.rewards.items() if t.weight < 0]
zero = [n for n, t in cfg.rewards.items() if t.weight == 0]
print(f"正权重 {len(pos)} 项（任务）、负权重 {len(neg)} 项（代价）、权重 0 {len(zero)} 项（由课程打开或仅占位）")
check("共 16 项", len(cfg.rewards) == 16)

# ---------------------------------------------------------------------------
banner("2. 五种“核”的形状")
plt = use_headless_matplotlib()
fig, axes = plt.subplots(2, 3, figsize=(11, 6))
ax = axes.flat

e = np.linspace(0, 1.0, 300)
for s2, lab in ((0.1, "lin vel std²=0.1"), (0.5, "ang vel std²=0.5"), (0.05, "upright std²=0.05")):
    ax[0].plot(e, np.exp(-e**2 / s2), label=lab)
ax[0].set_title("高斯核 exp(−err²/std²)"); ax[0].set_xlabel("误差"); ax[0].legend(fontsize=7)

err = np.linspace(-1.5, 1.5, 300)
ax[1].plot(err, np.exp(-(err / 0.5) ** 2), label="head_pose std=0.5")
ax[1].set_title("头部跟踪 exp(−(err/std)²)（同一种核，写法不同）"); ax[1].set_xlabel("误差 rad"); ax[1].legend(fontsize=7)

x = np.linspace(-1, 1, 300)
ax[2].plot(x, x**2, label="平方 (action_rate, body_ang_vel)")
ax[2].plot(x, np.abs(x), label="绝对值 L1 (head_pose_bias)")
ax[2].set_title("代价核：平方 vs 绝对值"); ax[2].legend(fontsize=7)

t = np.linspace(0, 0.5, 300)
ax[3].plot(t, ((t > 0.125) & (t < 0.3)).astype(float))
ax[3].set_title("air_time：盒式指示器 (0.125, 0.3) s"); ax[3].set_xlabel("脚离地时间 s")

q = np.linspace(-1.2, 1.2, 300); lo, hi = -0.9, 0.9
ax[4].plot(q, np.clip(lo - q, 0, None) + np.clip(q - hi, 0, None))
ax[4].set_title("dof_pos_limits：软限位外的铰链损失"); ax[4].set_xlabel("关节角 (软限位 ±0.9)")

h = np.linspace(0, 0.06, 300)
ax[5].plot(h, np.abs(h - 0.02), label="|h − 0.02| × 脚速")
ax[5].set_title("foot_clearance：离目标高度的距离（乘脚速）"); ax[5].set_xlabel("脚高 m"); ax[5].legend(fontsize=7)
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
savefig(fig, "ch16_reward_kernels")

# ---------------------------------------------------------------------------
banner("3. 数一数：满分时每步能拿多少（权重 × 核最大值 × dt）")
dt = 0.02
rows = []
for name, mx in (("track_linear_velocity", 1.0), ("track_angular_velocity", 1.0), ("upright", 1.0), ("pose", 1.0), ("air_time", 2.0), ("head_pose_tracking", 1.0)):
    w = cfg.rewards[name].weight
    rows.append([name, w, mx, w * mx * dt, w * mx * dt * 1000])
table(["任务项", "权重", "核最大值", "每步最多", "1000 步(20 s)最多"], rows)
print("总分上限约", round(sum(r[3] for r in rows) * 1000, 1), "分/回合。日志里 Mean reward 涨到几十就是在往这个上限逼近。")

# ---------------------------------------------------------------------------
if "--env" in sys.argv:
    banner("4. 🖥️ 建 4 个环境跑 50 步，看每一项的符号（AGENTS.md 的不变量：惩罚项 ≤ 0）")
    print("reward_manager._step_reward 存的是 函数值 × 权重（没乘 dt），这里把 50 步、4 个环境的平均累加起来。")
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    ecfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck", play=True)
    ecfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(ecfg, device="cuda:0")
    env.reset()
    sums = {}
    for _ in range(50):
        act = torch.randn(4, 14, device="cuda:0") * 0.05
        env.step(act)
        rm = env.reward_manager
        for i, name in enumerate(rm.active_terms):
            sums[name] = sums.get(name, 0.0) + rm._step_reward[:, i].mean().item()
    rows = [[n, v, cfg.rewards[n].weight] for n, v in sums.items()]
    table(["奖励项", "50 步累计（×权重，未乘 dt）", "权重"], rows)
    bad = [n for n, v in sums.items() if cfg.rewards[n].weight < 0 and v > 1e-9]
    check("负权重项全部 ≤ 0", not bad)
    bad2 = [n for n, v in sums.items() if n == "head_pose_bias" and v > 1e-9]
    check("self-negating 的 head_pose_bias ≤ 0", not bad2)
    env.close()
else:
    print("\n（加 --env 参数可再建环境跑 50 步验证符号，需要 GPU）")

done()
