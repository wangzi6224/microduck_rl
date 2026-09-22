"""第 14 章实验（🖥️GPU）：跑一次 64 env × 5 iteration 的冒烟训练，把日志里的每一行翻译成“它是哪个公式算出来的”。

运行：uv run python docs/learn-zh/labs/ch14_smoke_train.py
需要 CUDA GPU，约 1 分钟（首次会编译 warp 内核，更久）。
日志用 tensorboard 而不是 wandb，免登录。
小节对应正文：1 → 14.10 节（冒烟测试），2 → 14.4 节（一块日志），3 → 14.6 节（惩罚项 ≤ 0），4 → 14.7 节（其余几组），5 → 14.9 节（产物）。
正文里的手算、从配置读出的常数和全部讲解图，由 CPU 伴生实验 ch14_loop_by_hand.py 复核和生成；
它会打开这里产出的运行目录（logs/rsl_rl/velocity/*learnzh-ch14*）核对检查点。注意：第 2 节只认 "Learning iteration 4/5" 那一块，
想多跑几圈就直接用训练命令，不要改这里的 5。
"""
# LAB_REQUIRES: gpu

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from _common import banner, check, done, table

REPO = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
banner("1. 启动冒烟训练：uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5")
cmd = [
    "uv", "run", "train", "Mjlab-Velocity-Flat-MicroDuck",
    "--env.scene.num-envs", "64",
    "--agent.max-iterations", "5",
    "--agent.logger", "tensorboard",
    "--agent.run-name", "learnzh-ch14",
]
print(" ".join(cmd))
proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=900)
log = proc.stdout + proc.stderr
check("训练进程退出码 0", proc.returncode == 0)

# ---------------------------------------------------------------------------
banner("2. 每次 iteration 打印的块：抓最后一次（iteration 4/5）")
blocks = re.split(r"#{20,}", log)
last = [b for b in blocks if "Learning iteration 4/5" in b][-1]
lines = [ln.rstrip() for ln in last.splitlines() if ln.strip()]


def grab(key):
    for ln in lines:
        if key in ln:
            return ln.split(":", 1)[1].strip()
    return "?"


explain = [
    ("Total steps", "累计环境步数 = iteration × num_envs × 24（64 env：每次 1536）"),
    ("Collection time", "跑 24 步仿真、收一批数据的时间"),
    ("Learning time", "20 次 PPO 更新（5 遍 × 4 批）的时间（第 13 章）"),
    ("Mean value loss", "critic 的损失：(V − returns)² 的平均，项目里还套了一层裁剪（第 10 章 10.5、第 13 章）"),
    ("Mean surrogate loss", "−L^CLIP 的平均（第 13 章；负数正常）"),
    ("Mean entropy loss", "14 维高斯的熵，σ≈1 时 ≈ 19.86（第 4 章 4.9）"),
    ("Mean reward", "回合内奖励之和的平均（16 项加权和 × 0.02，第 16 章）"),
    ("Mean episode length", "平均回合长度（步）。摔得快就短"),
    ("Mean action std", "14 个 σ 的平均（第 4 章 4.7），初值 1.0"),
]
rows = [[k, grab(k), why] for k, why in explain]
table(["日志行", "本次的值", "它是什么"], rows)

# ---------------------------------------------------------------------------
banner("3. Episode_Reward/*：每一项奖励的加权贡献。惩罚项必须 ≤ 0")
rew_rows = []
bad = []
for ln in lines:
    m = re.match(r"\s*Episode_Reward/(\w+):\s*(-?[\d.]+)", ln)
    if m:
        name, val = m.group(1), float(m.group(2))
        rew_rows.append([name, val])
        if name in ("body_ang_vel", "angular_momentum", "dof_pos_limits", "action_rate_l2",
                    "foot_clearance", "foot_swing_height", "foot_slip", "self_collisions", "head_pose_bias") and val > 0:
            bad.append(name)
table(["奖励项", "加权后的值"], rew_rows)
check("所有惩罚项 ≤ 0（AGENTS.md 的不变量）", not bad)

# ---------------------------------------------------------------------------
banner("4. 其余几组")
for key in ("Curriculum/", "Episode_Termination/", "Metrics/twist/"):
    print(f"  {key}")
    for ln in lines:
        if key in ln:
            print("   ", ln.strip())
print("  Curriculum/*：课程当前阶段的取值（第 18 章）。Episode_Termination/*：本次结束原因的计数。")

# ---------------------------------------------------------------------------
banner("5. 产物")
run_dirs = sorted((REPO / "logs" / "rsl_rl" / "velocity").glob("*learnzh-ch14*"))
check("生成了日志目录", bool(run_dirs))
if run_dirs:
    d = run_dirs[-1]
    for p in sorted(d.iterdir()):
        print("   ", p.name)
    print("  model_4.pt = 第 4 次迭代（从 0 数）存下的：actor 与 critic 的全部旋钮（含归一化器统计）、Adam 的两本账；*.onnx = 已烘焙归一化的 actor（第 19 章）。")

done()
