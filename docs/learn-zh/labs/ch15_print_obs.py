"""第 15 章实验（🖥️GPU）：建 4 个 velocity 环境，打印 61 维 actor 观测和 76 维 critic 观测的每一块：名字、维数、一帧真实数值。

运行：uv run python docs/learn-zh/labs/ch15_print_obs.py
需要 CUDA GPU（mjlab 用 MuJoCo Warp）。没有 GPU 的读者跑 CPU 伴生实验 ch15_obs_layout.py：
它不建环境，从 cfg 推出同样的块表并核对正文的每个常数（两个实验互不依赖）。
正文引用的是本实验的一次存档输出：第 2 节的表在 15.1 节，第 3 节的一帧在 15.2、15.3、15.5 节，
第 4 节的 critic 表在 15.8 节，第 5 节的噪声与延迟在 15.4 节。
第 3 节的随机动作和环境都没有固定种子，每次跑出来的那一帧数字不同；块的布局和第 2、4、5 节的表不变。
"""
# LAB_REQUIRES: gpu

import numpy as np
import torch

from _common import banner, check, done, table

np.set_printoptions(precision=3, suppress=True, linewidth=120)

# ---------------------------------------------------------------------------
banner("1. 建环境：Mjlab-Velocity-Flat-MicroDuck，4 个并行环境")
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck", play=True)
cfg.scene.num_envs = 4
env = ManagerBasedRlEnv(cfg, device="cuda:0")
obs, _ = env.reset()
print("观测组:", list(obs.keys()))
print("actor 组形状:", tuple(obs["actor"].shape), " critic 组形状:", tuple(obs["critic"].shape))
check("actor 观测 61 维", obs["actor"].shape[1] == 61)
check("critic 观测 76 维", obs["critic"].shape[1] == 76)
check("控制周期 0.02 s (50 Hz)", abs(env.step_dt - 0.02) < 1e-9)

# ---------------------------------------------------------------------------
banner("2. actor 的 8 块：名字、维数、在 61 维里的起止位置")
om = env.observation_manager
rows, start = [], 0
for name, dim in zip(om.active_terms["actor"], om.group_obs_term_dim["actor"]):
    d = int(np.prod(dim))
    rows.append([name, d, f"[{start}:{start + d})"])
    start += d
table(["观测块", "维数", "索引区间"], rows)
check("8 块加起来 61", start == 61)

# ---------------------------------------------------------------------------
banner("3. 走 30 步（随机小动作），打印第 0 号机器人的一帧 61 维观测，按块标注")
for _ in range(30):
    act = torch.randn(4, 14, device="cuda:0") * 0.05
    obs, rew, terminated, truncated, extras = env.step(act)
frame = obs["actor"][0].cpu().numpy()
start = 0
for name, dim in zip(om.active_terms["actor"], om.group_obs_term_dim["actor"]):
    d = int(np.prod(dim))
    print(f"  {name:18s} [{start:2d}:{start + d:2d})  {frame[start:start + d]}")
    start += d
print("  单帧奖励 (4 个环境):", rew.cpu().numpy())

# ---------------------------------------------------------------------------
banner("4. critic 多出来的块：只有仿真才知道的“特权信息”")
rows, start = [], 0
actor_names = set(om.active_terms["actor"])
for name, dim in zip(om.active_terms["critic"], om.group_obs_term_dim["critic"]):
    d = int(np.prod(dim))
    rows.append([name, d, "" if name in actor_names else "★ 仅 critic"])
    start += d
table(["critic 观测块", "维数", ""], rows)
check("critic 各块加起来 76", start == 76)

# ---------------------------------------------------------------------------
banner("5. 噪声与延迟：从 cfg 里读出来")
rows = []
for name in om.active_terms["actor"]:
    t = om.get_term_cfg("actor", name)
    noise = getattr(t.noise, "n_max", None) if t.noise is not None else None
    delay = f"{t.delay_min_lag}-{t.delay_max_lag}" if getattr(t, "delay_max_lag", 0) else "无"
    rows.append([name, f"±{noise}" if noise is not None else "无", delay])
table(["观测块", "均匀噪声幅度", "延迟 (步)"], rows)
print("critic 组 enable_corruption=False：同样的块，critic 看到的是无噪声、无延迟的真值。")

env.close()
done()
