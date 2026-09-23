"""第 14 章 CPU 伴生实验：不训练、不建环境——把一圈训练里的数逐个手算出来，从配置和源码里读出正文引用的常数，画本章全部讲解图。

运行：uv run python docs/learn-zh/labs/ch14_loop_by_hand.py
纯 CPU：构造 velocity 任务的 cfg（做法同第 15、16、18 章的实验），再读几份 rsl_rl、mjlab 源码的文字。
真跑冒烟训练、打印日志的是另一个实验 ch14_smoke_train.py（要 GPU）。本实验用到的那块日志是它 2026-09-22 的一次存档输出，
要用的行原样抄在 ARCHIVE* 里；同一次运行留下的事件文件（tensorboard 格式，每一圈记一笔）里读出的几列，抄在 ARCHIVE_EVENTS 里。
你自己跑过 GPU 实验的话（logs/rsl_rl/velocity/ 下有 *learnzh-ch14* 目录），第 9、10 节还会打开你那次的检查点和事件文件核对；
没有就跳过那几项，其余照常。
小节编号与正文一一对应：实验第 K 节 = 正文 14.K 节（第 11 节对应「映射到项目」）；14.0 节的总览图在第 3 节画。
正文“改一改”要改的三行都带 `# TWEAK-k:` 标记（第 1、5、6 节各一处）。
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

from _common import banner, check, done, lines_in_order, savefig, table
from _draw import (BLUE, CELL, CELL_EDGE, CELL_HOT, FAINT, FS_NOTE, FS_SMALL, FS_STEP, FS_TICK, FS_TITLE, GREEN, INK,
                   MUTED, ORANGE, WHITE_BOX, arrow, cell, data_axes, hand, lesson_figure, lesson_panel, note, panel_note,
                   panel_title, plt)

REPO = Path(__file__).resolve().parents[3]
if not (REPO / "AGENTS.md").is_file():   # 被复制到别处运行（比如按“改一改”改一份副本）：从已安装的项目包找回仓库根目录
    _spec = importlib.util.find_spec("mjlab_microduck")
    if _spec and _spec.origin:
        REPO = Path(_spec.origin).resolve().parents[2]

# GPU 实验 ch14_smoke_train.py 的存档输出（2026-09-22；64 只机器人 × 5 圈，抓的是最后一圈 "Learning iteration 4/5"）。
# 右边是存档里印出来的样子，字符串原样照抄，一个字不改。你自己跑出来的数会不同。
ARCHIVE = {  # 存档第 2 节
    "Total steps": "7680", "Collection time": "0.538s", "Learning time": "0.040s",
    "Mean value loss": "0.0568", "Mean surrogate loss": "-0.0442", "Mean entropy loss": "19.8079",
    "Mean reward": "0.12", "Mean episode length": "35.21", "Mean action std": "1.00",
}
ARCHIVE_REWARD = {  # 存档第 3 节：Episode_Reward/<名字>
    "track_linear_velocity": "0.0224", "track_angular_velocity": "0.0019", "upright": "0.0301", "pose": "0.0081",
    "body_ang_vel": "-0.0232", "angular_momentum": "-0", "dof_pos_limits": "-0.0006", "action_rate_l2": "-0.1028",
    "air_time": "0.0114", "foot_clearance": "-0.0004", "foot_swing_height": "-0.0017", "foot_slip": "-0.0001",
    "self_collisions": "-0.0009", "head_pose_tracking": "0.0583", "body_pose_tracking": "0", "head_pose_bias": "0",
}
ARCHIVE_OTHER = {  # 存档第 4 节
    "Curriculum/action_rate_weight": "-0.1000", "Curriculum/standing_envs": "0.0200",
    "Curriculum/head_pose_range": "0.0700", "Curriculum/body_pose_range": "0.0500", "Curriculum/com_range": "0.0030",
    "Curriculum/head_com_range": "0.0030", "Curriculum/head_pose_bias_weight": "0.0000",
    "Episode_Termination/time_out": "0.0000", "Episode_Termination/fell_over": "2.0833",
    "Episode_Termination/out_of_terrain_bounds": "0.0000", "Episode_Termination/nan_state": "0.0000",
    "Metrics/twist/error_vel_xy": "0.0336", "Metrics/twist/error_vel_yaw": "0.1960",
}
ARCHIVE_FILES = ["2026-09-22_13-21-10_learnzh-ch14.onnx", "events.out.tfevents.1790054472.D0100685.2806623.0", "git",
                 "model_0.pt", "model_4.pt", "params"]  # 存档第 5 节列出的产物
# 存档那次两个产物的字节数（`ls -l` 量出来的）。正文 14.9 节把它们标成"存档那次的值，你的会不同"；
# 第 9 节用它们和"按配置算出来的"4,818,780 / 791,096 对差，这条核对与本机有没有运行目录无关。
ARCHIVE_SIZES = {"model_4.pt": 4_842_943, "onnx": 793_706}
# 同一次运行的事件文件（logs/rsl_rl/velocity/2026-09-22_13-21-10_learnzh-ch14/events.out.tfevents…）：第 0 到第 4 圈，
# 保留 6 位小数。GPU 实验只打印最后一圈；这几列是 14.1、14.8、14.10 节要用的"前几圈"。
ARCHIVE_EVENTS = {
    "Perf/collection_time": [0.918812, 0.781056, 0.539906, 0.528816, 0.537801],
    "Perf/learning_time": [0.088281, 0.043921, 0.043072, 0.042517, 0.0399],
    "Loss/learning_rate": [1e-05, 1e-05, 1e-05, 1.5e-05, 1e-05],
    "Train/mean_episode_length": [17.6, 31.777779, 33.59, 34.869999, 35.209999],
}


def val(s: str) -> float:
    """存档里的字符串 → 数（"0.538s" 去掉单位 s）。"""
    return float(s.rstrip("s"))


def pkg_file(pkg: str, *parts: str):
    """已安装的包里的某个源文件（rsl_rl、mjlab 装在 .venv 里）；找不到就返回 None。"""
    spec = importlib.util.find_spec(pkg)
    if not spec or not spec.origin:
        return None
    path = Path(spec.origin).parent.joinpath(*parts)
    return path if path.is_file() else None


def archive_run_dir():
    """存档那次运行留下的目录（还在本机上的话）：用来核对上面手抄的 ARCHIVE_EVENTS。"""
    d = REPO / "logs" / "rsl_rl" / "velocity" / ARCHIVE_FILES[0].removesuffix(".onnx")
    return d if d.is_dir() else None


def read_events(run_dir, tags):
    """从 tensorboard 事件文件里读出几列标量；读不了就返回 None（不影响其余检查）。"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        ea = EventAccumulator(str(run_dir))
        ea.Reload()
        have = ea.Tags()["scalars"]
        return {t: [e.value for e in ea.Scalars(t)] for t in tags if t in have}
    except Exception as err:
        print(f"  （读事件文件没成功：{type(err).__name__}，跳过）")
        return None


def find_run_dir():
    """你自己跑 GPU 实验留下的运行目录（最新的一个）；没有就返回 None。"""
    runs = sorted((REPO / "logs" / "rsl_rl" / "velocity").glob("*learnzh-ch14*"))
    runs = [d for d in runs if list(d.glob("model_*.pt"))]
    return runs[-1] if runs else None


# ---------------------------------------------------------------------------
banner("1. 四只钟：环境步、迭代、仿真时间、墙上时间")
import mjlab_microduck  # noqa: E402,F401
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (MicroduckRlCfg, NUM_STEPS_PER_ENV,  # noqa: E402
                                                              make_microduck_velocity_env_cfg)

cfg = make_microduck_velocity_env_cfg()
PHYS_DT = cfg.sim.mujoco.timestep                 # 物理步：0.005 s（第 17 章 17.5 节）
DECIMATION = cfg.decimation                       # 一个环境步里算几个物理步：4
ENV_DT = PHYS_DT * DECIMATION                     # 环境步：0.02 s
EPISODE_S = cfg.episode_length_s                  # 一个回合最长 20 s
EPISODE_STEPS = round(EPISODE_S / ENV_DT)         # = 1000 个环境步
STEPS = MicroduckRlCfg.num_steps_per_env          # 每圈每只机器人走 24 步
ALG = MicroduckRlCfg.algorithm
UPDATES = ALG.num_learning_epochs * ALG.num_mini_batches   # 每圈拧 5 × 4 = 20 次
print(f"配置：物理步 {PHYS_DT} s × {DECIMATION} = 环境步 {ENV_DT:.2f} s；回合最长 {EPISODE_S:g} s = {EPISODE_STEPS} 步；"
      f"每圈 {STEPS} 步；每圈拧 {ALG.num_learning_epochs} × {ALG.num_mini_batches} = {UPDATES} 次")
check("配置：物理步 0.005 s × 4 = 环境步 0.02 s；一回合最长 20 s = 1000 个环境步",
      PHYS_DT == 0.005 and DECIMATION == 4 and math.isclose(ENV_DT, 0.02) and EPISODE_S == 20.0 and EPISODE_STEPS == 1000)
check("配置：每圈每只机器人走 24 步（课程表换算用的 NUM_STEPS_PER_ENV 也是 24）；每圈拧 5 × 4 = 20 次",
      STEPS == 24 == NUM_STEPS_PER_ENV and ALG.num_learning_epochs == 5 and ALG.num_mini_batches == 4 and UPDATES == 20)


def clocks(robots: int, steps: int, iters: int) -> dict:
    """一段训练在四只钟上的读数，外加两个计数（记录条数、更新次数）。"""
    env_steps = steps * iters                     # 所有机器人齐步走：这只钟不乘机器人只数
    return {"env_steps": env_steps, "iters": iters, "sim_s": env_steps * ENV_DT, "phys_steps": env_steps * DECIMATION,
            "records": robots * env_steps, "updates": iters * UPDATES}


MINI = clocks(robots=2, steps=3, iters=2)         # 迷你版：2 只机器人，每圈走 3 步，转 2 圈
NUM_ENVS_SMOKE = 64  # TWEAK-1: 4096
SMOKE_ITERS = 5
SMOKE = clocks(NUM_ENVS_SMOKE, STEPS, SMOKE_ITERS)
table(["", "迷你版（2 只 × 3 步 × 2 圈）", f"冒烟测试（{NUM_ENVS_SMOKE} 只 × {STEPS} 步 × {SMOKE_ITERS} 圈）"],
      [["环境步", MINI["env_steps"], SMOKE["env_steps"]],
       ["迭代（圈）", MINI["iters"], SMOKE["iters"]],
       ["仿真时间（秒）", f"{MINI['sim_s']:.2f}", f"{SMOKE['sim_s']:.1f}"],
       ["（更细的钟）物理步", MINI["phys_steps"], SMOKE["phys_steps"]],
       ["（计数）记录条数", MINI["records"], SMOKE["records"]],
       ["（计数）更新次数", MINI["updates"], SMOKE["updates"]]])
print("手算（迷你版）：3 × 2 = 6 个环境步；6 × 0.02 = 0.12 秒；2 只 × 6 = 12 条记录；2 × 20 = 40 次更新")
check("迷你版：6 个环境步、0.12 秒、12 条记录、40 次更新",
      MINI["env_steps"] == 3 * 2 == 6 and round(6 * 0.02, 2) == 0.12 and MINI["records"] == 2 * 6 == 12
      and MINI["updates"] == 2 * 20 == 40)
print(f"手算（冒烟）：5 × 24 = 120 个环境步；120 × 0.02 = 2.4 秒；120 × 4 = 480 个物理步；"
      f"{NUM_ENVS_SMOKE} × 120 = {SMOKE['records']} 条记录；5 × 20 = 100 次更新")
check("冒烟：120 个环境步、2.4 秒仿真时间、480 个物理步、100 次更新（这几只钟和机器人只数无关）",
      SMOKE["env_steps"] == 5 * 24 == 120 and round(120 * 0.02, 1) == 2.4 and math.isclose(SMOKE["sim_s"], 2.4)
      and SMOKE["phys_steps"] == 120 * 4 == 480 and SMOKE["updates"] == 5 * 20 == 100)
check("冒烟：64 × 120 = 7680 条记录——和存档日志的 Total steps 是同一个数",
      SMOKE["records"] == 64 * 120 == 7680 == int(ARCHIVE["Total steps"]))

wall = [c + l for c, l in zip(ARCHIVE_EVENTS["Perf/collection_time"], ARCHIVE_EVENTS["Perf/learning_time"])]
print("墙上时间（存档那次的事件文件，每圈 = 收集 + 学习）：" + "、".join(f"第 {k} 圈 {w:.2f} s" for k, w in enumerate(wall))
      + f"；5 圈合计 {sum(wall):.2f} s（不含开头的启动和编译）")
print(f"存档日志最后一圈：Collection time 0.538 s + Learning time 0.040 s = {0.538 + 0.040:.3f} s")
check("墙上时间：最后一圈 0.538 + 0.040 = 0.578 s，和事件文件第 4 圈一致；第 0 圈 1.01 s、第 1 圈 0.82 s，之后 0.57–0.58 s；5 圈共 3.56 s",
      round(0.538 + 0.040, 3) == 0.578 == round(wall[4], 3) and round(wall[0], 2) == 1.01 and round(wall[1], 2) == 0.82
      and all(round(w, 2) in (0.57, 0.58) for w in wall[2:]) and round(sum(wall), 2) == 3.56
      and val(ARCHIVE["Collection time"]) == 0.538 and val(ARCHIVE["Learning time"]) == 0.040)

_arch_dir = archive_run_dir()
if _arch_dir is None:
    print("  （本机没有存档那次的运行目录，上面这几行秒数只能按手抄的来；跑过 GPU 实验的话第 10 节会读你自己的事件文件）")
else:
    _mine = read_events(_arch_dir, ARCHIVE_EVENTS)
    if _mine:
        check(f"手抄没抄错：{_arch_dir.name} 的事件文件里，这 {len(ARCHIVE_EVENTS)} 列和 ARCHIVE_EVENTS 逐个对得上",
              all(len(_mine.get(tag, [])) == len(vals)
                  and all(round(a, 6) == round(b, 6) for a, b in zip(_mine[tag], vals))
                  for tag, vals in ARCHIVE_EVENTS.items()))

REAL_ENVS, REAL_ITERS = 4096, 1000
real = clocks(REAL_ENVS, STEPS, REAL_ITERS)
robot_s = real["sim_s"]                                   # 每只机器人活过的仿真时间
all_s = REAL_ENVS * robot_s                               # 4096 只加起来
print(f"正式训练 {REAL_ITERS} 圈：{real['env_steps']:,} 个环境步 = {robot_s:.0f} 秒 = {robot_s / 60:.0f} 分钟（每只机器人）；"
      f"4096 只加起来 {all_s:,.0f} 秒 ≈ {all_s / 3600:.0f} 小时 ≈ {all_s / 86400:.1f} 天")
print("墙上时间只能举例：旧教程在一台 RTX 5070 上记过约 2.6 秒一圈（4096 只），1000 圈 = 2600 秒 ≈ 43 分钟")
check("1000 圈：24,000 个环境步 = 480 秒 = 8 分钟；4096 × 480 = 1,966,080 秒 ≈ 546 小时 ≈ 22.8 天",
      real["env_steps"] == 24000 and math.isclose(robot_s, 480) and 4096 * 480 == 1_966_080
      and round(1_966_080 / 3600) == 546 and round(1_966_080 / 86400, 1) == 22.8)
check("旧例子：2.6 × 1000 = 2600 秒 ≈ 43 分钟", round(2.6 * 1000) == 2600 and round(2600 / 60) == 43)
print(f"自测：64 只换成 4096 只，每圈环境步仍是 {STEPS}（课程钟每圈走 24），记录从 {64 * STEPS} 条变成 {4096 * STEPS:,} 条（64 倍）")
check("自测：机器人从 64 只变 4096 只，课程钟每圈照样只走 24；记录 1536 → 98,304，正好 64 倍",
      clocks(4096, STEPS, 1)["env_steps"] == clocks(64, STEPS, 1)["env_steps"] == 24
      and 64 * 24 == 1536 and 4096 * 24 == 98_304 == 64 * 1536)
print(f"回合钟：每只机器人自己的秒表，摔倒就清零。120 步里平均摔了 120 ÷ 35.21 ≈ {120 / 35.21:.1f} 次（35.21 = 存档的 Mean episode length）")
check("回合钟：120 ÷ 35.21 ≈ 3.4 个回合——120 个环境步不是一个 120 步长的回合", round(120 / 35.21, 1) == 3.4)
_runner = pkg_file("rsl_rl", "runners", "on_policy_runner.py")
_train = pkg_file("mjlab", "scripts", "train.py")
if _runner and _train:
    check("开训时回合钟被随机拨到 0–999：learn(init_at_random_ep_len=True) → randint(high = 1000 步)，上限本身取不到",
          lines_in_order(_runner.read_text(encoding="utf-8"), ["torch.randint_like(", "high=int(self.env.max_episode_length)"])
          and "init_at_random_ep_len=True" in _train.read_text(encoding="utf-8") and EPISODE_STEPS == 1000)

# ---------------------------------------------------------------------------
banner("1b. 画图：figures/ch14_four_clocks.png（同一段训练，四只钟）")
X0, W = 0.45, 9.0                                        # 前三把尺子共用的横坐标：环境步 0..120 → x 0.45..9.45
n_steps = SMOKE["env_steps"]


def xs(s):
    return X0 + s * W / n_steps


fig, axes = lesson_figure(4, "冒烟训练的 5 圈：三只钟齐步走，墙上的钟自己走", panel_height=2.75, width=10.6)
ax = axes[0]
lesson_panel(ax, f"① 环境步：每只机器人做了 {n_steps} 次决定（课程表看这只钟）", xmax=10, ymax=4)
ax.plot([xs(0), xs(n_steps)], [2.0, 2.0], color=INK, lw=2)
for s in range(n_steps + 1):
    big = s % STEPS == 0
    ax.plot([xs(s), xs(s)], [2.0, 2.0 + (0.42 if big else 0.16)], color=INK if big else FAINT, lw=1.6 if big else 0.9)
    if big:
        ax.text(xs(s), 1.62, f"{s}", ha="center", va="center", fontsize=FS_TICK, color=INK)
note(ax, X0, 0.72, "一步 = 0.02 秒 = 4 个物理步。64 只一起走，这只钟只走到 120，不是 64 × 120。")
ax = axes[1]
lesson_panel(ax, f"② 迭代：每 {STEPS} 步算一圈，末尾拧 {UPDATES} 次旋钮", xmax=10, ymax=4)
for k in range(SMOKE_ITERS):
    ax.add_patch(plt.Rectangle((xs(k * STEPS), 1.55), xs(STEPS) - xs(0), 0.85, facecolor=CELL, edgecolor=CELL_EDGE, lw=1.2))
    ax.text(xs(k * STEPS + STEPS / 2), 1.975, f"第 {k} 圈", ha="center", va="center", fontsize=FS_SMALL, color=INK)
    ax.plot([xs((k + 1) * STEPS)] * 2, [1.45, 2.5], color=ORANGE, lw=3.2)
ax.text(xs(n_steps), 2.78, f"橙线：拧 {UPDATES} 次，共 {SMOKE['updates']} 次", ha="right", va="center", fontsize=FS_SMALL,
        color=ORANGE)
note(ax, X0, 0.72, "日志块的标题 “Learning iteration 4/5” = 第 4 圈（从 0 数）/ 一共 5 圈。")
ax = axes[2]
lesson_panel(ax, f"③ 仿真时间：{n_steps} × 0.02 = {SMOKE['sim_s']:.1f} 秒", xmax=10, ymax=4)
ax.plot([xs(0), xs(n_steps)], [2.0, 2.0], color=BLUE, lw=2)
for s in range(0, n_steps + 1, STEPS):
    ax.plot([xs(s), xs(s)], [2.0, 2.42], color=BLUE, lw=1.6)
    ax.text(xs(s), 1.62, f"{s * ENV_DT:.2f} s", ha="center", va="center", fontsize=FS_TICK, color=BLUE)
note(ax, X0, 0.72, "和 ① 是同一把尺子，只是每格写成秒。机器人世界里一共只过了 2.4 秒。")
ax = axes[3]
lesson_panel(ax, "④ 墙上时间：同样 24 步一圈，每圈用时不一样", xmax=10, ymax=4)
T_MAX = 4.0                                              # 这把尺子单独定比例：0..4 秒 → x 0.45..9.45


def xt(t):
    return X0 + t * W / T_MAX


t0 = 0.0
for k, w in enumerate(wall):
    ax.add_patch(plt.Rectangle((xt(t0), 1.95), xt(w) - xt(0), 0.62, facecolor=CELL_HOT if k == 0 else CELL,
                               edgecolor=ORANGE if k == 0 else CELL_EDGE, lw=1.2))
    ax.text(xt(t0 + w / 2), 2.26, f"{w:.2f}", ha="center", va="center", fontsize=13, color=INK)
    t0 += w
ax.plot([xt(0), xt(T_MAX)], [1.8, 1.8], color=MUTED, lw=1.5)
for t in range(int(T_MAX) + 1):
    ax.plot([xt(t), xt(t)], [1.8, 1.62], color=MUTED, lw=1.3)
    ax.text(xt(t), 1.38, f"{t} s", ha="center", va="center", fontsize=FS_TICK, color=MUTED)
ax.text(xt(sum(wall)) + 0.12, 2.26, f"5 圈共 {sum(wall):.2f} s", ha="left", va="center", fontsize=FS_SMALL, color=INK)
note(ax, X0, 0.62, "存档那次每圈的秒数。启动、编译不在里面；换一台机器，这一行全都不一样。")
savefig(fig, "ch14_four_clocks")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("2. 采集：用当前的策略原样走 24 步，每一步记一行")
OBS_ACTOR, OBS_CRITIC, N_ACT = 61, 76, 14                # 第 15 章：actor 看 61 项、critic 看 76 项；14 个舵机
ROW = [("actor 看到的观测", OBS_ACTOR), ("critic 看到的观测", OBS_CRITIC), ("抽出的动作", N_ACT),
       ("那口钟的均值 μ", N_ACT), ("那口钟的标准差 σ", N_ACT), ("这个动作的 ln π", 1), ("critic 的 V", 1),
       ("奖励 r", 1), ("结束了吗", 1)]
table(["一行里记的东西", "几个数"], ROW)
per_row = sum(n for _, n in ROW)
print(f"一行共 {' + '.join(str(n) for _, n in ROW)} = {per_row} 个数")
check("一行记 9 样东西，共 61 + 76 + 14 + 14 + 14 + 1 + 1 + 1 + 1 = 183 个数",
      per_row == 61 + 76 + 14 + 14 + 14 + 1 + 1 + 1 + 1 == 183)
rows_mini, rows_smoke, rows_real = 2 * 3, 64 * STEPS, 4096 * STEPS
print(f"行数：迷你版 2 × 3 = {rows_mini}；冒烟 64 × 24 = {rows_smoke}；正式 4096 × 24 = {rows_real:,}")
check("行数：2 × 3 = 6；64 × 24 = 1536；4096 × 24 = 98,304", rows_mini == 6 and rows_smoke == 1536 and rows_real == 98_304)
print(f"形状 vs 乘法：一步里 4096 只的观测是形状 [4096, 61]（4096 行、每行 61 个数，共 {4096 * 61:,} 个数，仍是 4096 条记录）；"
      f"一圈存成 [24, 4096, 61]，共 {24 * 4096 * 61:,} 个数")
check("[4096, 61] 里有 4096 × 61 = 249,856 个数，但只有 4096 条记录；[24, 4096, 61] 共 5,996,544 个数",
      4096 * 61 == 249_856 and 24 * 4096 * 61 == 5_996_544)
check("自测：冒烟一圈 1536 行，actor 观测部分共 1536 × 61 = 93,696 个数", 1536 * 61 == 93_696)
check("61 / 76 / 14 不是手抄的：配置里 actor 这组 8 块、critic 这组 13 块（多 5 块 = 第 15 章说的 15 项特权观测）；"
      "这三个数本身由第 9 节的旋钮个数兜住——61 → 512 → 256 → 128 → 14 算出 197,774，76 → … → 1 算出 203,777",
      len(cfg.observations["actor"].terms) == 8 and len(cfg.observations["critic"].terms) == 13
      and set(cfg.observations) == {"actor", "critic"})

# ---------------------------------------------------------------------------
banner("3. 算优势、拧 20 次：记下的数不动，策略在动")
mb_real, mb_smoke = rows_real // ALG.num_mini_batches, rows_smoke // ALG.num_mini_batches
print(f"每圈 {ALG.num_learning_epochs} 遍 × {ALG.num_mini_batches} 份 = {UPDATES} 次更新；每份 98,304 ÷ 4 = {mb_real:,} 行"
      f"（冒烟 1536 ÷ 4 = {mb_smoke} 行）；每一行在一圈里被用 {ALG.num_learning_epochs} 次")
check("每份 98,304 ÷ 4 = 24,576 行（冒烟 384 行）；每行用 5 次：20 × 24,576 = 98,304 × 5 = 491,520",
      mb_real == 24_576 and mb_smoke == 384 and 20 * 24_576 == 98_304 * 5 == 491_520)
check("自测：份数改成 8，一圈拧 5 × 8 = 40 次，每份 98,304 ÷ 8 = 12,288 行", 5 * 8 == 40 and 98_304 // 8 == 12_288)
share = val(ARCHIVE["Collection time"]) / (val(ARCHIVE["Collection time"]) + val(ARCHIVE["Learning time"]))
print(f"存档最后一圈：收集 0.538 s、学习 0.040 s，收集占 {share:.0%}")
check("存档最后一圈：0.538 ÷ 0.578 ≈ 93% 的墙上时间花在跑仿真收数据上", round(0.538 / 0.578, 2) == 0.93)

# ---------------------------------------------------------------------------
banner("3b. 画图：figures/ch14_overview.png（14.0 节的总览图）")
fig = plt.figure(figsize=(10.6, 12.6))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 10.6)
ax.set_ylim(0, 12.6)
ax.axis("off")
ax.text(0.3, 12.15, "训练的一圈：采 24 步 → 算优势 → 拧 20 次 → 记一笔 → 隔 250 圈存一份", fontsize=FS_TITLE - 1,
        fontweight="bold", color=INK, va="center")
STATIONS = [
    ("① 采集（14.2 节）", f"4096 只 × {STEPS} 步 = {rows_real:,} 行", "每行：看到什么、做了什么、得了几分、critic 估多少、结束了没"),
    ("② 算优势（14.3 节）", f"从第 {STEPS} 步往回扫一遍：每行补上优势 Â 和 returns", "第 12 章的 GAE：这一步比 critic 预想的好多少"),
    ("③ 更新（14.3 节）", f"{ALG.num_learning_epochs} 遍 × {ALG.num_mini_batches} 份 = {UPDATES} 次", "每次：用当前网络重算、算损失、裁剪梯度、Adam 拧一次旋钮"),
    ("④ 记日志（14.4–14.8 节）", "终端打印一块日志", "计数和计时、三个损失、成绩、每一项奖励的分账"),
    ("⑤ 隔 250 圈存一份（14.9 节）", f"第 0、{MicroduckRlCfg.save_interval}、{2 * MicroduckRlCfg.save_interval}……圈和最后一圈：model_N.pt + .onnx",
     "检查点：接着练要用的一切"),
]
BOX_X, BOX_W, BOX_H, TOP, GAP = 0.3, 8.7, 1.72, 11.45, 2.22
for i, (title, numbers, gray) in enumerate(STATIONS):
    y = TOP - BOX_H - i * GAP
    ax.add_patch(plt.Rectangle((BOX_X, y), BOX_W, BOX_H, facecolor=CELL_HOT if i == 2 else CELL,
                               edgecolor=ORANGE if i == 2 else CELL_EDGE, lw=2 if i == 2 else 1.2))
    ax.text(BOX_X + 0.22, y + BOX_H - 0.36, title, fontsize=FS_STEP, fontweight="bold", color=INK, va="center")
    ax.text(BOX_X + 0.22, y + BOX_H / 2 - 0.1, numbers, fontsize=FS_SMALL + 1, color=ORANGE if i == 2 else BLUE, va="center")
    ax.text(BOX_X + 0.22, y + 0.3, gray, fontsize=FS_SMALL - 1, color=MUTED, va="center")
    if i < len(STATIONS) - 1:
        arrow(ax, (BOX_X + BOX_W / 2, y), (BOX_X + BOX_W / 2, y - (GAP - BOX_H)), INK, lw=2.4)
y_top_mid = TOP - BOX_H / 2
y_bot_mid = TOP - BOX_H - 4 * GAP + BOX_H / 2
RX = BOX_X + BOX_W + 0.75
ax.plot([BOX_X + BOX_W, RX, RX], [y_bot_mid, y_bot_mid, y_top_mid], color=GREEN, lw=2.6)
arrow(ax, (RX, y_top_mid), (BOX_X + BOX_W, y_top_mid), GREEN, lw=2.6)
ax.text(RX + 0.22, (y_top_mid + y_bot_mid) / 2, f"扔掉这 {rows_real:,} 行，用拧过的策略再采", rotation=90, fontsize=FS_SMALL, color=GREEN,
        ha="center", va="center")
ax.text(0.3, 0.42, "这一圈叫一次迭代；正式训练转几千圈。", fontsize=FS_NOTE, color=MUTED, va="center")
savefig(fig, "ch14_overview")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("4. 读一块日志：先数数，再看损失，再看成绩")
table(["日志行", "存档里的值"], [[k, v] for k, v in ARCHIVE.items()])
total_steps = int(ARCHIVE["Total steps"])
print(f"Total steps：5 圈 × 64 只 × 24 步 = {5 * 64 * 24}（记录条数）；环境步只有 5 × 24 = 120")
check("Total steps 7680 = 5 × 64 × 24：数的是记录，不是环境步（环境步是 120）", total_steps == 5 * 64 * 24 == 7680 != 120)
H1 = 0.5 * math.log(2 * math.pi * math.e)                 # σ = 1 的一口钟的熵（第 4 章 4.9 节）：1.41894
H14 = 14 * H1
entropy = val(ARCHIVE["Mean entropy loss"])
print(f"熵：14 口 σ = 1 的钟 = 14 × {H1:.5f} = {H14:.3f}；存档 {entropy}，比它小 {H14 - entropy:.4f}")
check("熵：14 × 1.41894 = 19.865（第 4 章 4.9 节）；存档的 19.8079 只比它小约 0.057",
      round(H1, 5) == 1.41894 and round(14 * 1.41894, 3) == 19.865 and round(H14, 3) == 19.865
      and round(19.865 - 19.8079, 3) == 0.057 and round(H14 - entropy, 3) == 0.057)
mean_log_sigma = (entropy - round(H14, 3)) / 14          # 每口钟的熵 = 1.41894 + ln σ：反推 ln σ 的平均
sigma_geo = math.exp(mean_log_sigma)
print(f"反推（折叠）：(19.8079 − 19.865) ÷ 14 = {mean_log_sigma:.4f}；e 的 {mean_log_sigma:.4f} 次方 = {sigma_geo:.4f}"
      f"（浏览器控制台：Math.exp(-0.0041) = {math.exp(-0.0041):.4f}）；日志印两位小数：{ARCHIVE['Mean action std']}")
check("反推：(19.8079 − 19.865) ÷ 14 ≈ −0.0041；Math.exp(−0.0041) = 0.9959 ≈ 0.996，印两位小数就是 1.00",
      round((19.8079 - 19.865) / 14, 4) == -0.0041 and round(mean_log_sigma, 4) == -0.0041
      and round(math.exp(-0.0041), 4) == 0.9959 and f"{sigma_geo:.2f}" == ARCHIVE["Mean action std"]
      and round(0.9959, 3) == 0.996 and round((1 - 0.9959) * 100, 1) == 0.4)
ep_len = val(ARCHIVE["Mean episode length"])
print(f"回合长度：{ep_len} 步 × 0.02 = {ep_len * ENV_DT:.4f} 秒；跑满是 {EPISODE_STEPS} 步")
check("回合长度：35.21 × 0.02 = 0.7042 秒，约 0.7 秒就摔；跑满是 1000 步", round(35.21 * 0.02, 4) == 0.7042
      and round(ep_len * ENV_DT, 4) == 0.7042)
check("自测：4096 只，标题 Learning iteration 9/… 那块是第 10 圈：Total steps = 10 × 4096 × 24 = 983,040；环境步 10 × 24 = 240",
      10 * 4096 * 24 == 983_040 and 10 * STEPS == 240)
check("三个损失的读数：value loss 为正（是平方的平均）；surrogate loss 是个小负数；熵在 19.865 附近",
      val(ARCHIVE["Mean value loss"]) > 0 and -0.1 < val(ARCHIVE["Mean surrogate loss"]) < 0 and abs(entropy - 19.865) < 0.1)
iter_s = round(val(ARCHIVE["Collection time"]) + val(ARCHIVE["Learning time"]), 3)
print(f"Steps per second：存档那一圈收的 64 × 24 = 1536 条记录 ÷ 这一圈的 {iter_s:.3f} 秒 = {1536 / iter_s:.0f} 条/秒"
      f"（它也数记录，不是环境步）")
check("Steps per second：1536 ÷ 0.578 ≈ 2657 条/秒（64 只 × 24 步 = 1536 条，0.538 + 0.040 = 0.578 秒）",
      round(64 * 24 / 0.578) == 2657 and round(1536 / round(0.538 + 0.040, 3)) == 2657)

# ---------------------------------------------------------------------------
banner("4b. 画图：figures/ch14_log_groups.png（读一块日志的动线：四组，每组先盯住一行）")
GROUPS = [  # (组号和组名, [(日志行, 值, 是不是“先看”的那一行)], 看它为了确认什么)
    ("① 计数和计时", [("Total steps", ARCHIVE["Total steps"], True), ("Steps per second", f"≈ {1536 / iter_s:.0f}", False),
                     ("Collection time", ARCHIVE["Collection time"], False), ("Learning time", ARCHIVE["Learning time"], False)],
     f"先看它，确认走到第几圈了。7680 = 5 × 64 × 24，数的是记录条数，不是环境步（环境步只走到 120）。\n"
     f"Steps per second 同理：这一圈的 1536 条 ÷ {iter_s:.3f} 秒 ≈ {1536 / iter_s:.0f} 条/秒。"),
    ("② 三个损失", [("Mean value loss", ARCHIVE["Mean value loss"], False),
                   ("Mean surrogate loss", ARCHIVE["Mean surrogate loss"], False),
                   ("Mean entropy loss", ARCHIVE["Mean entropy loss"], True)],
     "先看它，因为只有熵这一行能直接翻译成“策略现在什么样”：14 口钟还剩多宽（σ = 1 时是 14 × 1.41894 = 19.865）。"
     "另外两行只看“别突然变得很大”。"),
    ("③ 成绩", [("Mean episode length", ARCHIVE["Mean episode length"], True), ("Mean reward", ARCHIVE["Mean reward"], False),
                ("Mean action std", ARCHIVE["Mean action std"], False)],
     "先看它，才读得懂另外两行：活了 35.21 步 × 0.02 = 0.7042 秒就摔。Mean reward 必须配着它读——活得久，分自然多。"),
    ("④ 分项账", [("Episode_Reward/upright", ARCHIVE_REWARD["upright"], False),
                  ("Episode_Reward/action_rate_l2", ARCHIVE_REWARD["action_rate_l2"], True),
                  ("Episode_Termination/fell_over", ARCHIVE_OTHER["Episode_Termination/fell_over"], False),
                  ("Metrics/twist/error_vel_xy", ARCHIVE_OTHER["Metrics/twist/error_vel_xy"], False)],
     "先看的不是某一行，是一整列正负号：16 行 Episode_Reward 里，凡是惩罚项都必须 ≤ 0（14.6 节）。这里 −0.1028，对。"),
]
FIG_H = 14.4
fig = plt.figure(figsize=(10.6, FIG_H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 10.6)
ax.set_ylim(0, FIG_H)
ax.axis("off")
ax.text(0.3, FIG_H - 0.45, "读一块日志的动线：四组，每组先盯住橙色那一行", fontsize=FS_TITLE, fontweight="bold", color=INK,
        va="center")
ax.text(0.3, FIG_H - 0.92, "数取自存档那次冒烟训练的最后一圈（64 只 × 5 圈）。你自己跑出来的值会不同，分组和动线不变。",
        fontsize=FS_NOTE, color=MUTED, va="center")
LINE_H, LX, LW, SPLIT = 0.62, 0.35, 7.1, 4.75      # 一行的高、左边界、框宽、名字与值的分界
y = FIG_H - 1.55
for gname, lines, why in GROUPS:
    top = y
    for row, (name, value, hot) in enumerate(lines):
        yy = top - row * LINE_H - LINE_H
        ax.add_patch(plt.Rectangle((LX, yy), LW, LINE_H - 0.1, facecolor=CELL_HOT if hot else CELL,
                                   edgecolor=ORANGE if hot else CELL_EDGE, lw=2.0 if hot else 1.0))
        ax.text(SPLIT - 0.15, yy + (LINE_H - 0.1) / 2, name + ":", ha="right", va="center", fontsize=FS_SMALL,
                color=ORANGE if hot else INK, fontweight="bold" if hot else "normal")
        ax.text(SPLIT, yy + (LINE_H - 0.1) / 2, value, ha="left", va="center", fontsize=FS_SMALL,
                color=ORANGE if hot else INK)
    bottom = top - len(lines) * LINE_H
    ax.plot([LX + LW + 0.22] * 2, [bottom + 0.06, top - 0.06], color=BLUE, lw=2.4)
    ax.text(LX + LW + 0.42, (top + bottom) / 2, gname, ha="left", va="center", fontsize=FS_STEP - 1,
            fontweight="bold", color=BLUE)
    note(ax, LX + 0.05, bottom - 0.38, why, linespacing=1.5)
    y = bottom - 1.15
ax.text(0.35, y + 0.22, "⑤ 页脚（Iteration time / Time elapsed / ETA）只是墙上时间，本章不再细说。", fontsize=FS_NOTE,
        color=MUTED, va="center")
savefig(fig, "ch14_log_groups")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("5. Episode_Reward 的单位：加权、乘 0.02、整回合加起来、再除以 20 秒")


def episode_reward(kernel: float, weight: float, alive_s: float) -> tuple[float, float, int]:
    """函数值恒为 kernel 的一项，活了 alive_s 秒：返回（每步贡献、回合里的总和、读数）。mjlab reward_manager 的算法。"""
    steps = round(alive_s / ENV_DT)
    per_step = kernel * weight * ENV_DT
    total = sum(per_step for _ in range(steps))           # 回合里一步一步加（Σ = for 循环，第 2 章 2.2 节）
    return per_step, total, steps


KERNEL, WEIGHT = 0.5, 2.0
ALIVE_S = 5  # TWEAK-2: 10
per_step, total_full, n_full = episode_reward(KERNEL, WEIGHT, EPISODE_S)
_, total_short, n_short = episode_reward(KERNEL, WEIGHT, ALIVE_S)
read_full, read_short = total_full / EPISODE_S, total_short / EPISODE_S
table(["活了多久", "步数", "每步 0.5 × 2 × 0.02", "回合里加起来", "÷ 20 秒 = 读数"],
      [[f"{EPISODE_S:g} 秒", n_full, f"{per_step:.2f}", f"{total_full:.4g}", f"{read_full:.4g}"],
       [f"{ALIVE_S:g} 秒", n_short, f"{per_step:.2f}", f"{total_short:.4g}", f"{read_short:.4g}"]])
check("每步 0.5 × 2 × 0.02 = 0.02；活满 20 秒：1000 步加起来 20，读数 20 ÷ 20 = 1",
      math.isclose(per_step, 0.02) and n_full == 1000 and math.isclose(total_full, 20) and math.isclose(read_full, 1.0)
      and math.isclose(0.5 * 2 * 0.02 * 1000 / 20, 1))
check(f"只活 5 秒：250 步加起来 5，读数 5 ÷ 20 = 0.25（当前 ALIVE_S = {ALIVE_S:g} 秒，读数 {read_short:.4g}）",
      n_short == 250 and math.isclose(total_short, 5) and math.isclose(read_short, 0.25)
      and math.isclose(0.5 * 2 * 0.02 * 250 / 20, 0.25))
check("满分（函数值恒为 1）、活满 20 秒时，读数恰好等于权重",
      all(math.isclose(episode_reward(1.0, w, EPISODE_S)[1] / EPISODE_S, w) for w in (2.0, -0.1, 3.0)))
w_track = cfg.rewards["track_linear_velocity"].weight
cap = w_track * ep_len * ENV_DT / EPISODE_S              # 回合只有 35.21 步时，这一项最多能读到多少
got = val(ARCHIVE_REWARD["track_linear_velocity"])
print(f"track_linear_velocity：权重 {w_track:g}（配置）；回合 35.21 步时最多读到 2 × 35.21 × 0.02 ÷ 20 = {cap:.4f}；"
      f"存档 {got}，{got} ÷ {cap:.4f} ≈ {got / round(cap, 4):.2f}")
check("估算：2 × 35.21 × 0.02 ÷ 20 = 0.0704；0.0224 ÷ 0.0704 ≈ 0.32——活着时每步平均只拿到满分的三成左右",
      w_track == 2.0 and round(2 * 35.21 * 0.02 / 20, 4) == 0.0704 and round(cap, 4) == 0.0704
      and round(0.0224 / 0.0704, 2) == 0.32)
check("自测：函数值 0.8、权重 1、活 10 秒：0.8 × 1 × 0.02 × 500 ÷ 20 = 0.4",
      math.isclose(episode_reward(0.8, 1.0, 10)[1] / EPISODE_S, 0.4) and round(0.8 * 1 * 0.02 * 500 / 20, 4) == 0.4)

# ---------------------------------------------------------------------------
banner("5b. 画图：figures/ch14_episode_reward_units.png（同样每步 0.02，活得短读数就小）")
fig, axes = lesson_figure(3, f"每步一样好：活满 20 秒读数 {read_full:g}，只活 {ALIVE_S:g} 秒读数 {read_short:g}",
                          panel_height=2.95, width=10.6)
ax = axes[0]
lesson_panel(ax, "① 每一步：函数值 × 权重 × 0.02 秒", xmax=10, ymax=4)
for j, (text, hot) in enumerate([("函数值 0.5", False), ("× 权重 2", False), ("× 0.02 秒", False),
                                 (f"= {per_step:.2f}", True)]):
    cell(ax, 0.45 + 2.3 * j, 1.55, text, width=2.3, height=0.9, fontsize=FS_STEP - 1,
         facecolor=CELL_HOT if hot else CELL, edgecolor=ORANGE if hot else CELL_EDGE, color=ORANGE if hot else INK)
note(ax, 0.45, 0.75, "一回合里每一步都这样记一笔，只要机器人还没摔。这一笔恰好也是 0.02，和左边那个“0.02 秒”不是一回事。")


def time_bar(ax, alive, label_sum, label_read):
    """0–20 秒的时间尺子：活着的那段涂满（长度和秒数成比例），摔倒之后空着。"""
    x_of = lambda t: 0.45 + t * 9.0 / EPISODE_S  # noqa: E731
    ax.add_patch(plt.Rectangle((x_of(0), 1.95), x_of(EPISODE_S) - x_of(0), 0.72, facecolor="white", edgecolor=CELL_EDGE,
                               lw=1.2, ls="--"))
    ax.add_patch(plt.Rectangle((x_of(0), 1.95), x_of(alive) - x_of(0), 0.72, facecolor=CELL, edgecolor=BLUE, lw=1.6))
    if alive >= EPISODE_S / 2:                                # 条够宽：算式写在条里
        ax.text(x_of(alive / 2), 2.31, label_sum, ha="center", va="center", fontsize=FS_SMALL, color=BLUE)
    else:                                                     # 条太窄：算式挪到条的上方，免得压到虚线框上
        ax.text(x_of(0), 2.88, label_sum, ha="left", va="center", fontsize=FS_SMALL, color=BLUE)
    for t in range(0, int(EPISODE_S) + 1, 5):
        ax.plot([x_of(t), x_of(t)], [1.95, 1.78], color=MUTED, lw=1.3)
        ax.text(x_of(t), 1.52, f"{t} s", ha="center", va="center", fontsize=FS_TICK, color=MUTED)
    ax.text(9.45, 3.35, label_read, ha="right", va="center", fontsize=FS_STEP, color=ORANGE, fontweight="bold")
    if alive < EPISODE_S:
        ax.text(x_of((alive + EPISODE_S) / 2), 2.31, "摔倒了：后面不再记账", ha="center", va="center", fontsize=FS_SMALL,
                color=MUTED)


ax = axes[1]
lesson_panel(ax, f"② 活满 {EPISODE_S:g} 秒", xmax=10, ymax=4)
time_bar(ax, EPISODE_S, f"{n_full} 步 × 每步 {per_step:.2f} = {total_full:g}", f"{total_full:g} ÷ 20 = {read_full:g}")
note(ax, 0.45, 0.72, "日志把回合里的总和再除以 20 秒（配置里的回合时长），满分跑满时读数 = 权重。")
ax = axes[2]
lesson_panel(ax, f"③ 只活 {ALIVE_S:g} 秒", xmax=10, ymax=4)
time_bar(ax, ALIVE_S, f"{n_short} 步 × 每步 {per_step:.2f} = {total_short:g}", f"{total_short:g} ÷ 20 = {read_short:g}")
note(ax, 0.45, 0.72, f"每一步做得一样好，读数差 {read_full / read_short:g} 倍——差在活了多久。")
savefig(fig, "ch14_episode_reward_units")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("6. 符号约定：惩罚项的读数必须 ≤ 0")
terms = {name: t for name, t in cfg.rewards.items() if t is not None}
bias_stages = [s["weight"] for s in cfg.curriculum["head_pose_bias_weight"].params["weight_stages"]]
rows, cost_style, self_neg = [], [], []
for name, t in terms.items():
    mod, fname = t.func.__module__, t.func.__name__
    if mod.startswith("mjlab_microduck") and (fname.endswith("_penalty") or fname.endswith("_l1")):
        kind = "自带负号（≤ 0）→ 正权重"
        self_neg.append(name)
    elif t.weight < 0:
        kind = "代价（≥ 0）→ 负权重"
        cost_style.append(name)
    else:
        kind = "奖励（≥ 0）→ 正权重"
    rows.append([name, fname, "mjlab" if mod.startswith("mjlab.") else "microduck", f"{t.weight:g}", kind])
table(["奖励项", "函数", "谁写的", "起步权重", "写法"], rows)
print(f"head_pose_bias 的课程权重：{' → '.join(f'{w:g}' for w in bias_stages)}")
check("配置里一共 16 项奖励，存档的 Episode_Reward 也是 16 行（正文 14.4、14.6、14.10 说的“16 项”）",
      len(terms) == 16 == len(ARCHIVE_REWARD))
check("配置：8 个代价型惩罚都是 mjlab 的函数、权重都 < 0；16 项里没有别的负权重",
      len(cost_style) == 8 and all(terms[n].weight < 0 and terms[n].func.__module__.startswith("mjlab.") for n in cost_style)
      and sum(t.weight < 0 for t in terms.values()) == 8)
check("配置：自带负号的只有 head_pose_bias（microduck 的 head_pose_bias_penalty），起步权重 0，课程只把它调成 1、2、3（都 ≥ 0）",
      self_neg == ["head_pose_bias"] and terms["head_pose_bias"].weight == 0.0 and bias_stages == [0.0, 1.0, 2.0, 3.0])
check("名字带 _penalty 不等于自带负号：body_ang_vel、angular_momentum 用的是 mjlab 的 *_penalty，算的是平方（≥ 0），配负权重",
      terms["body_ang_vel"].func.__name__ == "body_angular_velocity_penalty" and terms["body_ang_vel"].weight == -0.05
      and terms["angular_momentum"].func.__name__ == "angular_momentum_penalty" and terms["angular_momentum"].weight == -0.02)
print("手算（代价型）：动作变化 (0.1, −0.2) → 0.1² + (−0.2)² = 0.05；× 权重 −0.1 = −0.005；再 × 0.02 = 每步 −0.0001")
cost = 0.1 ** 2 + (-0.2) ** 2
w_rate = terms["action_rate_l2"].weight
check("代价型：0.05 × (−0.1) = −0.005，每步 −0.0001——扣分",
      math.isclose(cost, 0.05) and w_rate == -0.1 and math.isclose(cost * w_rate, -0.005)
      and math.isclose(cost * w_rate * ENV_DT, -0.0001) and round(0.05 * -0.1, 4) == -0.005)
BIAS_ERR = -0.2                                        # head_pose_bias_penalty 的返回值：−|平均偏差|，偏 0.2 rad 时是 −0.2
HEAD_POSE_BIAS_WEIGHT = 3.0  # TWEAK-3: -3.0
contrib = BIAS_ERR * HEAD_POSE_BIAS_WEIGHT
print(f"手算（自带负号）：(−0.2) × {HEAD_POSE_BIAS_WEIGHT:g} = {contrib:+.1f}；每步 {contrib * ENV_DT:+.3f}"
      f"（写反成 −3 的话：(−0.2) × (−3) = +0.6，头越歪越加分）")
check(f"自带负号型：(−0.2) × 权重 {HEAD_POSE_BIAS_WEIGHT:g} = {contrib:+.1f}，应当 ≤ 0（课程的最终权重是 3：−0.6，每步 −0.012）",
      contrib <= 0 and math.isclose(-0.2 * 3, -0.6) and math.isclose(-0.2 * 3 * 0.02, -0.012))
check("写反的样子：(−0.2) × (−3) = +0.6，每步 +0.012——负负得正", math.isclose(-0.2 * -3, 0.6) and math.isclose(0.6 * 0.02, 0.012))
agents_text = (REPO / "AGENTS.md").read_text(encoding="utf-8") if (REPO / "AGENTS.md").is_file() else ""
if agents_text:
    check("AGENTS.md 原话：mjlab 的代价函数 ≥ 0 配负权重；自带负号的 *_penalty、*_l1 配正权重；惩罚项的 Episode_Reward 必须 ≤ 0",
          lines_in_order(agents_text, ["cost functions return ≥ 0 → negative weight.", "(`*_penalty`, `*_l1` returning ≤ 0) → POSITIVE weight.",
                                       "policy will farm it (butt-hopping, crash-sits). **The infallible check: on",
                                       "every run, every `Episode_Reward/<penalty>` in wandb must be ≤ 0.**"]))
import inspect  # noqa: E402
check("权重为 0 的两项都是普通函数（不是类）：mjlab 连调用都不调用它们——冒烟的 5 圈里它们一次都没运行过",
      all(terms[n].weight == 0 and not inspect.isclass(terms[n].func) for n in ("body_pose_tracking", "head_pose_bias")))
penalties = cost_style + self_neg
readings = {n: val(ARCHIVE_REWARD[n]) for n in penalties}
print("存档里 9 个惩罚项的读数：" + "、".join(f"{n} {ARCHIVE_REWARD[n]}" for n in penalties))
check("存档：9 个惩罚项的读数全都 ≤ 0（AGENTS.md 说的那条每次必查）", len(penalties) == 9 and all(v <= 0 for v in readings.values()))
check("存档里的 “-0”：日志印 4 位小数，比 0.00005 还小的负数就印成 −0.0000", ARCHIVE_REWARD["angular_momentum"] == "-0"
      and math.copysign(1, val("-0")) < 0 and f"{-0.00004:.4f}" == "-0.0000")
check("权重为 0 的两项读数恰好是 0：body_pose_tracking、head_pose_bias（没被算，不是做得完美）",
      terms["body_pose_tracking"].weight == 0 == val(ARCHIVE_REWARD["body_pose_tracking"])
      and terms["head_pose_bias"].weight == 0 == val(ARCHIVE_REWARD["head_pose_bias"]))
check("自测：自带负号的函数返回 −0.3、权重写成 −2：(−0.3) × (−2) = +0.6，每步 +0.012", math.isclose(-0.3 * -2, 0.6)
      and round(0.6 * 0.02, 3) == 0.012)

# ---------------------------------------------------------------------------
banner("6b. 画图：figures/ch14_sign_convention.png（函数的正负 × 权重的正负）")
fig = plt.figure(figsize=(10.6, 9.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 10.6)
ax.set_ylim(0, 9.4)
ax.axis("off")
ax.text(0.3, 8.95, "函数的正负 × 权重的正负：惩罚项的乘积必须 ≤ 0", fontsize=FS_TITLE, fontweight="bold", color=INK, va="center")
COL_X, COL_W, ROW_Y, ROW_H = [3.35, 6.85], 3.4, [4.55, 1.45], 2.95
for x, head in zip(COL_X, ["配负权重", "配正权重"]):
    ax.text(x + COL_W / 2, 7.95, head, ha="center", va="center", fontsize=FS_STEP, fontweight="bold", color=INK)
ROW_HEAD = [("函数算“代价”，≥ 0", "（mjlab 的写法）", "例：动作变化 0.05"),
            ("函数自带负号，≤ 0", "（microduck 的写法）", "例：头部偏差 −0.2")]
for y, (a, b, c) in zip(ROW_Y, ROW_HEAD):
    ax.text(0.3, y + ROW_H - 0.55, a, fontsize=FS_SMALL + 1, fontweight="bold", color=INK, va="center")
    ax.text(0.3, y + ROW_H - 1.15, b, fontsize=FS_SMALL, color=MUTED, va="center")
    ax.text(0.3, y + ROW_H - 1.75, c, fontsize=FS_SMALL, color=BLUE, va="center")
CELLS = {  # (行, 列): (算式, 对不对, 大白话, 项目里的例子)
    (0, 0): ("0.05 × (−0.1) = −0.005", True, "扣分：动作越抖扣得越多", "项目：action_rate_l2\n起步权重 −0.1"),
    (0, 1): ("0.05 × 0.1 = +0.005", False, "越抖越加分", ""),
    (1, 0): ("(−0.2) × (−3) = +0.6", False, "负负得正：头越歪越加分", ""),
    (1, 1): ("(−0.2) × 3 = −0.6", True, "扣分：头越歪扣得越多", "项目：head_pose_bias\n权重 0 → 1 → 2 → 3（课程）"),
}
for (r, c), (calc, ok, words, example) in CELLS.items():
    x, y = COL_X[c], ROW_Y[r]
    ax.add_patch(plt.Rectangle((x, y), COL_W, ROW_H, facecolor=CELL if ok else CELL_HOT,
                               edgecolor=GREEN if ok else ORANGE, lw=2.2))
    ax.text(x + COL_W / 2, y + ROW_H - 0.5, calc, ha="center", va="center", fontsize=FS_STEP - 1, color=INK)
    ax.text(x + COL_W / 2, y + ROW_H - 1.22, ("✓ " if ok else "✗ ") + words, ha="center", va="center",
            fontsize=FS_SMALL, color=GREEN if ok else ORANGE, fontweight="bold")
    if example:
        ax.text(x + COL_W / 2, y + 0.75, example, ha="center", va="center", fontsize=FS_SMALL - 2, color=MUTED,
                linespacing=1.4)
ax.text(0.3, 0.8, "日志里的 Episode_Reward 还要再 × 0.02、加满一回合、÷ 20 秒——正负号一路不变，",
        fontsize=FS_NOTE, color=MUTED, va="center")
ax.text(0.3, 0.3, "所以看日志就能查出哪一项配反了：惩罚项的读数必须 ≤ 0。", fontsize=FS_NOTE, color=MUTED, va="center")
savefig(fig, "ch14_sign_convention")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("7. 其余几组：课程到了哪一档、回合为什么结束、跟得准不准")
cur = cfg.curriculum


def stage0(name: str) -> float:
    """课程第 0 档的取值，按项目的课程函数"往日志里报什么"来算（mdp.py 里各课程函数的 return）。"""
    p = cur[name].params
    if "weight_stages" in p:
        return p["weight_stages"][0]["weight"]
    if "standing_stages" in p:
        return p["standing_stages"][0]["rel_standing_envs"]
    if "range_stages" in p and isinstance(p["range_stages"][0].get("ranges"), tuple):
        return max(max(abs(lo), abs(hi)) for lo, hi in p["range_stages"][0]["ranges"])   # 各维上限里最大的那个
    return p["range_stages"][0]["range"]


def first_change(name: str):
    """课程第一次换档是在第几个环境步；只有一档（永远不换）就返回 None。"""
    p = cur[name].params
    stages = p.get("weight_stages") or p.get("standing_stages") or p.get("range_stages")
    return stages[1]["step"] if len(stages) > 1 else None


rows = []
for name in cur:
    logged = val(ARCHIVE_OTHER[f"Curriculum/{name}"])
    step = first_change(name)
    rows.append([name, f"{stage0(name):g}", f"{logged:.4f}", step if step else "不换档",
                 step // NUM_STEPS_PER_ENV if step else "—"])
table(["课程项", "配置里第 0 档", "存档读数", "第一次换档（环境步）", "= 第几圈"], rows)
check("存档的 7 个 Curriculum/* 读数 = 配置里第 0 档的值（5 圈时课程一档都没动）",
      len(rows) == 7 and all(math.isclose(stage0(n), val(ARCHIVE_OTHER[f"Curriculum/{n}"])) for n in cur))
changes = sorted({first_change(n) for n in cur} - {None})
check("最早的换档在第 500 圈 = 500 × 24 = 12,000 个环境步（head_pose_bias 在第 600 圈）",
      changes[0] == 500 * 24 == 12_000 and first_change("head_pose_bias_weight") == 600 * 24)
check("头部命令范围读数 0.07 = 第 0 档四个范围里最大的上限（头部偏航 ±0.07 rad）",
      cur["head_pose_range"].params["range_stages"][0]["ranges"][2] == (-0.07, 0.07) and stage0("head_pose_range") == 0.07)
fell = val(ARCHIVE_OTHER["Episode_Termination/fell_over"])
print(f"fell_over = {fell}。若读成“每只机器人这一圈摔了 {fell} 次”，回合只有 24 ÷ {fell} ≈ {24 / fell:.1f} 步——和 35.21 对不上；"
      f"读成“平均每一步有几只摔倒”：64 只里每步约 64 ÷ 35.21 ≈ {64 / 35.21:.2f} 只走到回合尽头，和 {fell} 差不多")
check("反证：24 ÷ 2.0833 ≈ 11.5 步，不到 35.21 的一半——“每只摔了 2 次”的读法不对",
      round(24 / 2.0833, 1) == 11.5 and round(24 / fell, 1) == 11.5 and 24 / fell < 35.21 / 2)
check("对得上的读法：64 ÷ 35.21 ≈ 1.82 只/步，和日志的 2.08 差不多（相差不到两成；两者统计的不是同一批回合）",
      round(64 / 35.21, 2) == 1.82 and abs(64 / 35.21 / fell - 1) < 0.2)
twist = cfg.commands["twist"]
max_cmd_steps = round(twist.resampling_time_range[1] / ENV_DT)
err_xy = val(ARCHIVE_OTHER["Metrics/twist/error_vel_xy"])
print(f"速度误差：命令最长 {twist.resampling_time_range[1]:g} 秒不换 = {max_cmd_steps} 步；每步的 |命令 − 实际| 除以 {max_cmd_steps} 再整回合加起来。"
      f"{err_xy} × 400 = {err_xy * 400:.2f}，÷ 35.21 ≈ {err_xy * 400 / 35.21:.2f} m/s")
check("Metrics/twist/error_vel_xy：除的是 8 ÷ 0.02 = 400；0.0336 × 400 = 13.44，÷ 35.21 ≈ 0.38 m/s",
      twist.resampling_time_range == (3.0, 8.0) and max_cmd_steps == 400 and round(0.0336 * 400, 2) == 13.44
      and round(13.44 / 35.21, 2) == 0.38 and round(err_xy * 400 / ep_len, 2) == 0.38)
check("自测：fell_over 3.2、回合 20 步：64 ÷ 20 = 3.2 只/步，对得上；20 × 0.02 = 0.4 秒", 64 / 20 == 3.2 and round(20 * 0.02, 2) == 0.4)
check("结束原因：这一圈没有超时、没有出界、没有 NaN", all(val(ARCHIVE_OTHER[f"Episode_Termination/{k}"]) == 0
                                        for k in ("time_out", "out_of_terrain_bounds", "nan_state")))

# ---------------------------------------------------------------------------
banner("8. 看曲线：健康的训练往哪走，四种常见病")
LR0, LR_MIN, LR_MAX = ALG.learning_rate, 1e-5, 1e-2        # 下限、上限写在 rsl_rl 的 ppo.py 里（第 11 节核对）
lr, k = LR0, 0
while lr > LR_MIN:
    lr, k = max(LR_MIN, lr / 1.5), k + 1                 # KL 每次都超过 2 × 0.01：学习率 ÷ 1.5，最低 0.00001
print(f"学习率从 {LR0:g} 起，每次更新都 ÷ 1.5：第 {k} 次更新碰到下限 {LR_MIN:g}"
      f"（1.5 的 11 次方 = {1.5 ** 11:.1f}，0.001 ÷ 86.5 ≈ {LR0 / 1.5 ** 11:.7f}，还没到；12 次方 = {1.5 ** 12:.2f}，"
      f"0.001 ÷ 129.75 ≈ {LR0 / 1.5 ** 12:.7f}，低于下限）")
check("KL 次次超标：从 0.001 起连除 12 次 1.5 就压到下限 0.00001——不到一圈（一圈 20 次）；0.001 ÷ 129.75 ≈ 0.0000077",
      LR0 == 0.001 and ALG.desired_kl == 0.01 and k == 12 and k < UPDATES and round(1.5 ** 11, 1) == 86.5
      and round(1.5 ** 12, 2) == 129.75 and 0.001 / 86.5 > 1e-5 > 0.001 / 129.75 and round(0.001 / 129.75, 7) == 0.0000077)
if agents_text:
    check("AGENTS.md 原话：Measure before theorizing——“失败”先拿检查点量一遍；过去的“失败”有的是看了太早的检查点、有的是判成功的标准有问题",
          lines_in_order(agents_text, ['**Measure before theorizing.** When a run "fails", run a headless eval of the',
                                       'profiles) before changing rewards: past "failures" turned out to be early',
                                       "checkpoints, a success criterion splitting one behavior cluster in half, and"]))
lr_events = ARCHIVE_EVENTS["Loss/learning_rate"]
print(f"存档那次冒烟训练的学习率（事件文件，第 0–4 圈）：{', '.join(f'{v:g}' for v in lr_events)}")
check("存档：第 0 圈结束时学习率已经在下限 0.00001；5 圈里都在 0.00001–0.000015",
      lr_events[0] == 1e-5 and all(1e-5 <= v <= 1.5e-5 for v in lr_events))

# 示意数据：编的曲线，只示意形状（不是真实训练）。固定种子，每次画出来一样。
rng = np.random.default_rng(14)
it = np.arange(0, 3001, 10)
noise = lambda s: rng.normal(0, s, it.size)  # noqa: E731
healthy_len = np.clip(1000 - 965 * np.exp(-it / 650) + noise(18), 0, 1000)
sick_len = np.clip(33 + 5 * np.sin(it / 170) + noise(3), 0, None)
healthy_std = 0.38 + 0.62 * np.exp(-it / 1100) + noise(0.008)
sick_std = 0.03 + 0.97 * np.exp(-it / 90) + noise(0.004)
walk = np.cumsum(noise(0.06))
healthy_lr = np.clip(1.0 * np.exp(0.9 * np.sin(it / 260) * 0.45 + 0.35 * walk / (1 + np.abs(walk))), 0.35, 2.4)
sick_lr = np.where(it < 10, 1.0, 0.01)
healthy_pen = -0.05 - 0.2 * (1 - np.exp(-it / 900)) + noise(0.01)
sick_pen = 0.02 + 0.4 * (1 - np.exp(-it / 700)) + noise(0.012)
check("示意曲线：健康的回合长度走到 900 步以上、σ 慢慢降到 0.4 左右；生病的回合长度一直不到 50 步、σ 几百圈就塌到 0.05 以下",
      healthy_len[-1] > 900 and 0.35 < healthy_std[-1] < 0.45 and sick_len.max() < 50 and sick_std[it >= 400].max() < 0.05)

# ---------------------------------------------------------------------------
banner("8b. 画图：figures/ch14_health_curves.png（示意数据：健康 vs 常见病）")
fig = plt.figure(figsize=(10.6, 17.6))
PANELS = [  # (小标题, 健康, 生病, 健康标签和位置, 生病标签和位置, 纵轴, 纵轴范围, 灰字)
    ("① 回合长度（步）：往 1000 步走，还是一直卡在很短", healthy_len, sick_len, ("健康：越活越久", 1850, 700),
     ("生病：一直几十步就摔", 1850, 130), "步", (0, 1080), "虚线是上限 1000 步（20 秒）。卡在几十步 = 还没学会站稳。"),
    ("② 动作标准差（14 个 σ 的平均）：慢慢变小，还是塌到 0", healthy_std, sick_std, ("健康：慢慢收窄", 1850, 0.6),
     ("生病：过早收敛，塌到 0 附近", 1500, 0.12), "σ", (0, 1.08), "塌到 0 附近 = 不再试别的动作，好走法还没找到就定型了。"),
    ("③ 学习率（纵轴单位 0.001）：在中间晃，还是贴着下限", healthy_lr, sick_lr, ("健康：上下调", 2250, 1.75),
     ("生病：一直贴着下限 0.00001", 1300, 0.25), "× 0.001", (0, 2.7), "KL 次次超标，规则只能一路往下压：连除 12 次 1.5 就到底。"),
    ("④ 某个惩罚项的 Episode_Reward：在 0 以下，还是跑到 0 以上", healthy_pen, sick_pen, ("健康：≤ 0", 1850, -0.37),
     ("生病：> 0，符号配反了", 1650, 0.17), "读数", (-0.45, 0.55), "惩罚项读数为正 = 这一项在给违规加分，策略会去刷它（14.6 节）。"),
]
tops = [0.885, 0.655, 0.425, 0.195]
for (title, good, bad, (g_lbl, gx, gy), (b_lbl, bx, by), ylab, ylim, gray), top in zip(PANELS, tops):
    ax = fig.add_axes([0.11, top - 0.118, 0.84, 0.118])
    data_axes(ax, "迭代（圈）", ylab)
    ax.plot(it, good, color=BLUE, lw=2.4)
    ax.plot(it, bad, color=ORANGE, lw=2.4)
    ax.set_xlim(0, 3000)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000])
    if ylab == "步":
        ax.axhline(1000, color=FAINT, ls="--", lw=1.5)
    if ylab == "读数":
        ax.axhline(0, color=INK, lw=1.2)
    ax.text(gx, gy, g_lbl, color=BLUE, fontsize=FS_SMALL, va="center", bbox=WHITE_BOX)
    ax.text(bx, by, b_lbl, color=ORANGE, fontsize=FS_SMALL, va="center", bbox=WHITE_BOX)
    panel_title(fig, [ax], title)
    panel_note(fig, [ax], gray)
fig.text(0.055, 0.985, "健康的训练往哪走：四条曲线，四种常见病", fontsize=FS_TITLE, fontweight="bold", color=INK, va="top")
fig.text(0.055, 0.962, "示意数据：曲线是编的，只示意形状，不是一次真实训练。蓝 = 健康，橙 = 生病。", fontsize=FS_NOTE, color=MUTED,
         va="top")
savefig(fig, "ch14_health_curves")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("9. 产物：检查点装着接着练所需的一切")


def mlp_params(sizes) -> int:
    """一层接一层的网络：每层 输出 × (输入 + 1) 个旋钮（第 6 章 6.5 节的窍门）。"""
    return sum(o * (i + 1) for i, o in zip(sizes[:-1], sizes[1:]))


hidden = list(MicroduckRlCfg.actor.hidden_dims)
actor_mlp = mlp_params([OBS_ACTOR, *hidden, N_ACT])
actor_all = actor_mlp + N_ACT                           # 再加 14 个 σ（std_type = "scalar"：每个舵机一个）
critic_all = mlp_params([OBS_CRITIC, *list(MicroduckRlCfg.critic.hidden_dims), 1])
trainable = actor_all + critic_all
adam = 2 * trainable                                    # Adam 的两本账：每个旋钮一个 m、一个 v（第 7 章 7.5 节）
ckpt_numbers = trainable + adam
ckpt_bytes, onnx_bytes = ckpt_numbers * 4, actor_mlp * 4   # 每个数 4 字节（float32）
table(["部分", "几个数"], [["actor 的权重和偏置", f"{actor_mlp:,}"], ["actor 的 14 个 σ", N_ACT], ["critic", f"{critic_all:,}"],
                          ["Adam 的 m 和 v", f"{adam:,}"], ["检查点合计（不算归一化器的统计）", f"{ckpt_numbers:,}"]])
print(f"检查点 ≈ {ckpt_numbers:,} × 4 字节 = {ckpt_bytes:,} 字节 ≈ {ckpt_bytes / 1e6:.1f} MB；"
      f"ONNX ≈ {actor_mlp:,} × 4 = {onnx_bytes:,} 字节 ≈ {onnx_bytes / 1e6:.2f} MB；约 {ckpt_bytes / onnx_bytes:.1f} 倍")
check("actor：197,774 + 14 = 197,788；critic 203,777（第 6 章 6.5、6.7 节）；可训练的一共 401,565",
      actor_all - N_ACT == actor_mlp == 197_774 and actor_all == 197_788 and critic_all == 203_777
      and trainable == 197_788 + 203_777 == 401_565 and hidden == [512, 256, 128] and MicroduckRlCfg.actor.distribution_cfg["std_type"] == "scalar")
check("Adam 两本账 2 × 401,565 = 803,130；旋钮加两本账 401,565 + 803,130 = 1,204,695（= 401,565 × 3）；"
      "1,204,695 × 4 = 4,818,780 字节 ≈ 4.8 MB",
      adam == 2 * 401_565 == 803_130 and ckpt_numbers == 401_565 + 803_130 == 401_565 * 3 == 1_204_695
      and ckpt_bytes == 1_204_695 * 4 == 401_565 * 3 * 4 == 4_818_780 and round(4_818_780 / 1e6, 1) == 4.8)
check("ONNX：197,774 × 4 = 791,096 字节 ≈ 0.79 MB；4,818,780 ÷ 791,096 ≈ 6.1 倍",
      onnx_bytes == 197_774 * 4 == 791_096 and round(791_096 / 1e6, 2) == 0.79 and round(4_818_780 / 791_096, 1) == 6.1)
check("字节换 MB：一兆按一百万字节算，4,818,780 ÷ 1,000,000 ≈ 4.8；791,096 ÷ 1,000,000 ≈ 0.79；每个数 4 字节 = 32 位 ÷ 8",
      round(4_818_780 / 1_000_000, 1) == 4.8 and round(791_096 / 1_000_000, 2) == 0.79 and 32 // 8 == 4)
print(f"存档那次的真文件：model_4.pt {ARCHIVE_SIZES['model_4.pt']:,} 字节，比算出来的多 "
      f"{ARCHIVE_SIZES['model_4.pt'] - 4_818_780:,}；.onnx {ARCHIVE_SIZES['onnx']:,} 字节，多 "
      f"{ARCHIVE_SIZES['onnx'] - 791_096:,}（归一化器的统计和文件格式本身；你自己跑出来的会略有不同）")
check("存档的真文件大小（正文 14.9 节引的两个数）：4,842,943 − 4,818,780 = 24,163；793,706 − 791,096 = 2,610",
      ARCHIVE_SIZES["model_4.pt"] - ckpt_bytes == 4_842_943 - 4_818_780 == 24_163
      and ARCHIVE_SIZES["onnx"] - onnx_bytes == 793_706 - 791_096 == 2_610)
save_every = MicroduckRlCfg.save_interval
smoke_saves = sorted({i for i in range(SMOKE_ITERS) if i % save_every == 0} | {SMOKE_ITERS - 1})
check(f"每 {save_every} 圈存一次（圈号能被 250 整除就存，第 0 圈也算），最后一圈再存一次：冒烟只有 model_0 和 model_4",
      save_every == 250 and smoke_saves == [0, 4] and [f for f in ARCHIVE_FILES if f.startswith("model_")] == ["model_0.pt", "model_4.pt"])
SMOKE_COUNTERS = {"iter": SMOKE_ITERS - 1, "common_step_counter": SMOKE["env_steps"], "normalizer_count": SMOKE["records"],
                  "adam_step": SMOKE["updates"]}
print(f"model_4.pt 里应记着四个计数：iter = {SMOKE_COUNTERS['iter']}、环境步 common_step_counter = {SMOKE_COUNTERS['common_step_counter']}、"
      f"归一化器吃进的行数 count = {SMOKE_COUNTERS['normalizer_count']}、Adam 的 step = {SMOKE_COUNTERS['adam_step']}")
check("冒烟结束时的四个计数：圈号 4、环境步 120、归一化器 7680 行、Adam 100 次——正是 14.1 节那张表",
      SMOKE_COUNTERS == {"iter": 4, "common_step_counter": 120, "normalizer_count": 7680, "adam_step": 100})

run_dir = find_run_dir()
if run_dir is None:
    print("  （没找到 GPU 实验留下的 *learnzh-ch14* 运行目录，跳过打开真实检查点这几项；先跑 ch14_smoke_train.py 就会核对）")
else:
    import torch

    print(f"打开你那次的运行目录：{run_dir.name}")
    ck = torch.load(run_dir / "model_4.pt", map_location="cpu", weights_only=False) if (run_dir / "model_4.pt").is_file() else None
    if ck is None:
        print("  （这个目录里没有 model_4.pt——大概不是 5 圈的冒烟训练，跳过）")
    else:
        a, c, opt = ck["actor_state_dict"], ck["critic_state_dict"], ck["optimizer_state_dict"]
        n_actor = sum(v.numel() for k2, v in a.items() if not k2.startswith("obs_normalizer"))
        n_critic = sum(v.numel() for k2, v in c.items() if not k2.startswith("obs_normalizer"))
        n_adam = sum(v.numel() for s in opt["state"].values() for k2, v in s.items() if k2 in ("exp_avg", "exp_avg_sq"))
        real_counters = {"iter": ck["iter"], "common_step_counter": ck["infos"]["env_state"]["common_step_counter"],
                         "normalizer_count": int(a["obs_normalizer.count"]),
                         "adam_step": int(next(iter(opt["state"].values()))["step"])}
        lr_now = opt["param_groups"][0]["lr"]
        size_pt = (run_dir / "model_4.pt").stat().st_size
        onnx_files = list(run_dir.glob("*.onnx"))
        size_onnx = onnx_files[0].stat().st_size if onnx_files else 0
        print(f"  顶层的键：{list(ck.keys())}")
        print(f"  actor {n_actor:,}、critic {n_critic:,}、Adam 的账 {n_adam:,}；四个计数 {real_counters}；Adam 此刻的学习率 {lr_now:.2g}")
        print(f"  文件大小：model_4.pt {size_pt:,} 字节，.onnx {size_onnx:,} 字节")
        check("真实检查点的 5 个键：actor、critic、优化器、圈号、附加信息（课程钟在 infos 里）",
              list(ck.keys()) == ["actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter", "infos"])
        check("真实检查点：actor 197,788、critic 203,777、Adam 803,130——和按配置算的一样",
              n_actor == actor_all and n_critic == critic_all and n_adam == adam)
        check("真实检查点的四个计数 = 4、120、7680、100", real_counters == SMOKE_COUNTERS)
        check("学习率在上下限之间（0.00001 到 0.01）", LR_MIN <= lr_now * (1 + 1e-9) and lr_now <= LR_MAX)
        check(f"文件大小比估算多一点（归一化器的统计和文件格式）：{size_pt:,} − 4,818,780 = {size_pt - ckpt_bytes:,}（两万多）；"
              f"{size_onnx:,} − 791,096 = {size_onnx - onnx_bytes:,}",
              20_000 < size_pt - ckpt_bytes < 30_000 and 0 < size_onnx - onnx_bytes < 10_000)
        if run_dir.name == ARCHIVE_FILES[0].removesuffix(".onnx"):   # 本机上还留着存档那一次：顺手核对手抄的两个字节数
            check("这就是存档那次的运行目录：两个文件大小和正文 14.9 节引的 4,842,943 / 793,706 一字不差",
                  size_pt == ARCHIVE_SIZES["model_4.pt"] and size_onnx == ARCHIVE_SIZES["onnx"])
        check("观测是 61 维：actor 第一层的形状是 [512, 61]；critic 的是 [512, 76]",
              tuple(a["mlp.0.weight"].shape) == (512, 61) and tuple(c["mlp.0.weight"].shape) == (512, 76))

# ---------------------------------------------------------------------------
banner("9b. 画图：figures/ch14_checkpoint.png（检查点和 ONNX 各装了什么）")
fig, axes = lesson_figure(2, f"检查点约是 ONNX 的 {ckpt_bytes / onnx_bytes:.0f} 倍：多装了 critic 和 Adam 的两本账",
                          panel_height=3.3, width=10.6)
scale = 9.0 / ckpt_numbers                              # 两根条共用一把尺子：长度和数的个数成比例


def seg(ax, x, n, color, label, y=1.7, h=0.95, text_color="white"):
    ax.add_patch(plt.Rectangle((x, y), n * scale, h, facecolor=color, edgecolor="white", lw=1.5))
    ax.text(x + n * scale / 2, y + h / 2, label, ha="center", va="center", fontsize=FS_SMALL - 1, color=text_color)
    return x + n * scale


ax = axes[0]
lesson_panel(ax, f"① model_4.pt：{ckpt_numbers:,} 个数 × 4 字节 ≈ {ckpt_bytes / 1e6:.1f} MB", xmax=10, ymax=4)
x = seg(ax, 0.45, actor_all, BLUE, f"actor\n{actor_all:,}")
x = seg(ax, x, critic_all, GREEN, f"critic\n{critic_all:,}")
x = seg(ax, x, trainable, "#8a9aab", f"Adam 的 m\n{trainable:,}")
seg(ax, x, trainable, "#b3c0cc", f"Adam 的 v\n{trainable:,}", text_color=INK)
note(ax, 0.45, 0.85, "接着练要用：两张网、Adam 的账；另有归一化器的统计、圈号和课程钟（数很少，画不出来）。")
ax = axes[1]
lesson_panel(ax, f"② .onnx：{actor_mlp:,} 个数 × 4 字节 ≈ {onnx_bytes / 1e6:.2f} MB", xmax=10, ymax=4)
seg(ax, 0.45, actor_mlp, BLUE, "actor 的\n权重和偏置")
ax.text(0.45 + actor_mlp * scale + 0.25, 2.17, "同一把尺子：条的长度和数的个数成比例", fontsize=FS_SMALL, color=MUTED, va="center")
note(ax, 0.45, 0.85, "上真机只要 actor，再加归一化用的 61 个平均、61 个标准差（第 19 章）。")
savefig(fig, "ch14_checkpoint")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("10. 冒烟测试：5 圈证明“能跑通”，不证明“学得会”")
agents = (REPO / "AGENTS.md").read_text(encoding="utf-8") if (REPO / "AGENTS.md").is_file() else ""
AGENTS_LINES = ["A 5-iteration smoke test at 64 envs catches ~95% of config errors for cents.",
                "Never launch a long run without one.",
                "builds, steps NaN-free, obs is 61D,", "every reward term computes, ONNX exports.",
                "curriculum-heavy recovery need 4000–6000."]
if agents:
    print("AGENTS.md：" + " / ".join(AGENTS_LINES))
    check("AGENTS.md 里原样写着：冒烟测试 64 只 × 5 圈、抓住约 95% 的配置错误、检查清单五条、步态要 4000–6000 圈",
          lines_in_order(agents, AGENTS_LINES) and "uv run train <TASK_ID> --env.scene.num-envs 64 --agent.max_iterations 5" in agents)
else:
    print("  （没找到仓库根目录的 AGENTS.md，跳过）")
print(f"冒烟 {SMOKE_ITERS} 圈 vs 课程第一次换档 {changes[0] // NUM_STEPS_PER_ENV} 圈：{SMOKE_ITERS / (changes[0] // NUM_STEPS_PER_ENV):.0%}；"
      f"vs 步态预算 4000 圈：{SMOKE_ITERS / 4000:.3%}")
check("5 ÷ 500 = 1%：冒烟测试连课程的第一档都没走完；5 ÷ 4000 = 0.125%", math.isclose(SMOKE_ITERS / 500, 0.01)
      and changes[0] // NUM_STEPS_PER_ENV == 500 and 5 / 4000 * 100 == 0.125)
len_events = ARCHIVE_EVENTS["Train/mean_episode_length"]
print("存档那次 Mean episode length（第 0–4 圈）：" + "、".join(f"{v:.2f}" for v in len_events))
check("第 0 圈只有 24 步，能结束的回合最长 24 步：17.6 ≤ 24；之后的“变长”是窗口在填满，不是学会了",
      len_events[0] == 17.6 <= STEPS and all(a < b for a, b in zip(len_events, len_events[1:])))
if run_dir is not None:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        ea = EventAccumulator(str(run_dir))
        ea.Reload()
        mine = {tag: [e.value for e in ea.Scalars(tag)] for tag in ARCHIVE_EVENTS if tag in ea.Tags()["scalars"]}
        rows = [[tag] + [f"{v:.3g}" for v in mine.get(tag, [])] for tag in ARCHIVE_EVENTS]
        table(["你那次的事件文件"] + [f"第 {k} 圈" for k in range(len(rows[0]) - 1)], rows)
        if "Train/mean_episode_length" in mine and mine["Train/mean_episode_length"]:
            check("你那次也一样：第 0 圈的 Mean episode length ≤ 24", mine["Train/mean_episode_length"][0] <= STEPS)
        if "Loss/learning_rate" in mine:
            check("你那次的学习率都在 0.00001 到 0.01 之间", all(LR_MIN * (1 - 1e-6) <= v <= LR_MAX for v in mine["Loss/learning_rate"]))
    except Exception as err:  # 读不了事件文件不影响正文的数
        print(f"  （读事件文件没成功：{type(err).__name__}，跳过）")

# ---------------------------------------------------------------------------
banner("11. 映射到项目：正文引用的常数和源码行还在不在")


RUNNER_LINES = ["for it in range(start_it, total_it):", "start = time.time()", "with torch.inference_mode():",
                'for _ in range(self.cfg["num_steps_per_env"]):', "actions = self.alg.act(obs)",
                "obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))",
                "self.alg.process_env_step(obs, rewards, dones, extras)",
                "self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)",
                "collect_time = stop - start", "self.alg.compute_returns(obs)", "loss_dict = self.alg.update()",
                "learn_time = stop - start", "self.logger.log(",
                'if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:',
                'self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))',
                'self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))']
REWARD_LINES = ["episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])",
                'extras["Episode_Reward/" + key] = (', "episodic_sum_avg / self._env.max_episode_length_s",
                "if term_cfg.weight == 0.0:", "continue",
                "value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * scale",
                "self._reward_buf += value", "self._episode_sums[name] += value"]
OTHER = [  # （包, 路径, 要核对的行, 说明）
    ("rsl_rl", ("utils", "logger.py"), ["self.rewbuffer = deque(maxlen=100)",
                                        'collection_size = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size',
                                        "self.tot_timesteps += collection_size", 'self.writer.add_scalar("Loss/learning_rate", learning_rate, it)',
                                        'self.writer.add_scalar("Policy/mean_std", action_std.mean().item(), it)',
                                        '{"Total steps:":>{pad}} {self.tot_timesteps}', '{f"Mean {key} loss:":>{pad}} {value:.4f}',
                                        '{"Mean reward:":>{pad}} {statistics.mean(self.rewbuffer):.2f}',
                                        '{"Mean action std:":>{pad}} {action_std.mean().item():.2f}'],
     "logger.py：Total steps 每圈加 24 × 机器人只数；Mean reward 是最近 100 个回合；学习率只进 wandb"),
    ("rsl_rl", ("runners", "on_policy_runner.py"), ["saved_dict = self.alg.save()",
                                                     'saved_dict["iter"] = self.current_learning_iteration',
                                                     'saved_dict["infos"] = infos'],
     "on_policy_runner.py：检查点 = PPO 存的三样 + iter + infos，正好 5 样（14.9 节）"),
    ("rsl_rl", ("algorithms", "ppo.py"), ["self.learning_rate = max(1e-5, self.learning_rate / 1.5)",
                                          "self.learning_rate = min(1e-2, self.learning_rate * 1.5)",
                                          "self.storage.clear()",
                                          '"actor_state_dict": self.actor.state_dict(),',
                                          '"critic_state_dict": self.critic.state_dict(),',
                                          '"optimizer_state_dict": self.optimizer.state_dict(),'],
     "ppo.py：学习率夹在 0.00001 和 0.01 之间；update() 末尾扔掉这批数据；save() 存两张网和优化器"),
    ("mjlab", ("managers", "termination_manager.py"), ['extras["Episode_Termination/" + key] = torch.count_nonzero('],
     "termination_manager.py：结束原因记的是“这次重置的机器人里有几只”"),
    ("mjlab", ("tasks", "velocity", "mdp", "velocity_command.py"), ["max_command_time = self.cfg.resampling_time_range[1]",
                                                                     "max_command_step = max_command_time / self._env.step_dt",
                                                                     "/ max_command_step"],
     "velocity_command.py：速度误差每步除以 400 再累加"),
    ("mjlab", ("rl", "runner.py"), ['filename = f"{export_dir.name}.onnx"',
                                    'env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}'],
     "mjlab 的 runner.py：ONNX 文件名 = 运行目录名（每次覆盖同一个）；检查点的 infos 里存着课程钟"),
    ("mjlab", ("tasks", "velocity", "rl", "runner.py"), ["super().save(path, infos)", "self.export_policy_to_onnx(str(policy_dir), filename)"],
     "velocity 的 runner.py：每存一次检查点就导出一次 ONNX"),
    ("mjlab", ("scripts", "train.py"), ["num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True"],
     "train.py：开训时把每只机器人的回合钟随机拨一下"),
]
runner = pkg_file("rsl_rl", "runners", "on_policy_runner.py")
reward_mgr = pkg_file("mjlab", "managers", "reward_manager.py")
if runner and reward_mgr:
    runner_text, reward_text = runner.read_text(encoding="utf-8"), reward_mgr.read_text(encoding="utf-8")
    check(f"rsl_rl/runners/on_policy_runner.py 的 learn()：映射块引用的 {len(RUNNER_LINES)} 行原样存在、顺序一致",
          lines_in_order(runner_text[runner_text.find("    def learn("):], RUNNER_LINES))
    at = reward_text.find("  def reset(")
    check(f"mjlab/managers/reward_manager.py：reset() 在前、compute() 在后，映射块引用的 {len(REWARD_LINES)} 行原样存在、顺序一致",
          at >= 0 and lines_in_order(reward_text[at:], REWARD_LINES) and reward_text.find("  def compute(") > at)
    for pkg, parts, lines, what in OTHER:
        path = pkg_file(pkg, *parts)
        check(what, path is not None and lines_in_order(path.read_text(encoding="utf-8"), lines))
else:
    print("  （当前 Python 环境里没有 rsl_rl / mjlab，跳过源码核对；用 uv run 运行就会核对）")
cfg_path = REPO / "src" / "mjlab_microduck" / "tasks" / "microduck_velocity_env_cfg.py"
mdp_path = REPO / "src" / "mjlab_microduck" / "tasks" / "mdp.py"
if cfg_path.is_file() and mdp_path.is_file():
    cfg_text = cfg_path.read_text(encoding="utf-8")
    at = cfg_text.find("MicroduckRlCfg = RslRlOnPolicyRunnerCfg(")
    CFG_LINES = ["num_learning_epochs=5,", "num_mini_batches=4,", 'experiment_name="velocity",', "save_interval=250,",
                 "num_steps_per_env=24,", "max_iterations=50_000,"]
    check("microduck_velocity_env_cfg.py 的 MicroduckRlCfg：5 遍、4 份、实验名 velocity、每 250 圈存一次、每圈 24 步、上限 50,000 圈",
          at >= 0 and lines_in_order(cfg_text[at:], CFG_LINES))
    mdp_text = mdp_path.read_text(encoding="utf-8")
    rewards_py = pkg_file("mjlab", "envs", "mdp", "rewards.py")
    check("两种惩罚写法对着源码：mjlab 的 action_rate_l2 返回平方和（≥ 0）；microduck 的 head_pose_bias_penalty 返回 −|平均偏差|（≤ 0）",
          "out = -env._head_bias_ema.abs().mean(dim=-1)" in mdp_text
          and (rewards_py is None or "torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1"
               in rewards_py.read_text(encoding="utf-8")))
else:
    print("  （没找到项目的源码目录，跳过配置行的核对）")

done()
