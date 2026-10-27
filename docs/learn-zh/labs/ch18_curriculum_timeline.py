"""第 18 章实验：从真实的 env cfg 里读出 7 项课程，算清每一项在第几圈换档，
并复核正文里的每一个数（尺子、阶梯、活的管理器、反事实账表）。

运行：uv run python docs/learn-zh/labs/ch18_curriculum_timeline.py
纯 CPU：只构造配置对象，不建仿真、不训练、不需要 GPU（头一次 import 要十几秒）。
小节编号与正文一一对应：实验第 K 节 = 正文 18.K 节（第 10 节对应「映射到项目」）。
正文“改一改”要改的三行都带 `# TWEAK-k:` 标记（第 2、6、8 节各一处）。
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import re
from collections import Counter
from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _common import banner, check, done, lines_in_order, num, savefig, table
from _draw import (BLUE, CELL, CELL_EDGE, CELL_HOT, FAINT, FS_SMALL, FS_STEP, FS_TICK, FS_TITLE, GREEN, INK,
                   MUTED, ORANGE, WHITE_BOX, arrow, cell, data_axes, hand, lesson_figure, lesson_panel, note,
                   panel_note, panel_title, plt)

# ---------------------------------------------------------------------------
banner("1. 为什么先易后难（两笔账），以及走路任务的 7 项课程")
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (  # noqa: E402
    NUM_STEPS_PER_ENV,
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

cfg = make_microduck_velocity_env_cfg()
VEL_SRC_PATH = Path(inspect.getsourcefile(make_microduck_velocity_env_cfg))   # 配置文件本身（第 2、10 节读它的原文）
DT = round(cfg.decimation * cfg.sim.mujoco.timestep, 6)   # 一个环境步 = decimation × 物理步（都从 cfg 读）
EPISODE_S = cfg.episode_length_s                          # 配置里的回合时长
EPISODE_STEPS = round(EPISODE_S / DT)                     # 活满一整回合是多少个环境步


def stages_of(name: str) -> list[dict]:
    """一项课程的阶梯清单。params 里那个名字以 stages 结尾的键就是它。"""
    for key, value in cfg.curriculum[name].params.items():
        if key.endswith("stages"):
            return value
    raise KeyError(name)


def value_of(stage: dict):
    """一档里除了 step 以外的那个值（weight / rel_standing_envs / range / ranges）。"""
    rest = [v for k, v in stage.items() if k != "step"]
    assert len(rest) == 1, stage
    return rest[0]


def max_abs(ranges) -> float:
    """一组 (下限, 上限) 里最大的那个绝对值——课程函数报给 wandb 的就是它。"""
    return max(max(abs(lo), abs(hi)) for lo, hi in ranges)


def sgn(v, places: int = 4) -> str:
    """正文和图里的写法：负号用 −，不用 ASCII 的 -。"""
    return num(v, places).replace("-", "−")


def wstr(v) -> str:
    """权重固定印一位小数：−0.1、−1.0、0.0、3.0。"""
    return f"{v:.1f}".replace("-", "−")


MINUS_TICKS = plt.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−"))


def readable(v) -> str:
    """把一档的取值写成正文里的样子。"""
    if isinstance(v, (tuple, list)):
        return f"最大 ±{num(max_abs(v))} rad"
    return sgn(v)


NAMES = ["action_rate_weight", "head_pose_bias_weight", "standing_envs",
         "com_range", "head_com_range", "head_pose_range", "body_pose_range"]
WHAT = {
    "action_rate_weight": "动作平滑惩罚的权重",
    "head_pose_bias_weight": "头部下垂惩罚的权重",
    "standing_envs": "抽中“站着别动”的比例",
    "com_range": "躯干质心随机偏移的范围（m）",
    "head_com_range": "头部质心随机偏移的范围（m）",
    "head_pose_range": "4 个头部命令的抽签范围",
    "body_pose_range": "6 个身体命令的抽签范围",
}
rows = []
for name in NAMES:
    st = stages_of(name)
    later = [s["step"] for s in st if s["step"] > 0]
    fmt = wstr if name.endswith("_weight") else readable
    rows.append([name, WHAT[name], fmt(value_of(st[0])), fmt(value_of(st[-1])),
                 len(st), (min(later) // NUM_STEPS_PER_ENV) if later else "不换档"])
table(["课程项", "改什么", "起步", "最后一档", "共几档", "第一次换档（圈）"], rows)

check("走路任务一共 7 项课程", len(cfg.curriculum) == 7)
check("7 项的名字就是上表这 7 个", set(cfg.curriculum) == set(NAMES))
check("模板自带的速度命令课程 command_vel 被删掉了（速度命令范围从头到尾不变）",
      "command_vel" not in cfg.curriculum)
check("平地任务里没有地形课程 terrain_levels", "terrain_levels" not in cfg.curriculum)
twist = cfg.commands["twist"]
print(f"  速度命令的范围（固定，不在课程里）：前后 {twist.ranges.lin_vel_x}、左右 {twist.ranges.lin_vel_y}、"
      f"转 {twist.ranges.ang_vel_z}")
check("速度命令范围从头到尾就是 ±0.4 / ±0.3 / ±1.0",
      twist.ranges.lin_vel_x == (-0.4, 0.4) and twist.ranges.lin_vel_y == (-0.3, 0.3)
      and twist.ranges.ang_vel_z == (-1.0, 1.0))
check(f"配置里一共 {len(cfg.rewards)} 项奖励、{len(cfg.events)} 个事件、{len(cfg.commands)} 组命令（正文写的是 16 / 13 / 3）",
      len(cfg.rewards) == 16 and len(cfg.events) == 13 and len(cfg.commands) == 3)
cur_rewards = {cfg.curriculum[n].params["reward_name"] for n in NAMES if "reward_name" in cfg.curriculum[n].params}
cur_events = {cfg.curriculum[n].params["event_name"] for n in NAMES if "event_name" in cfg.curriculum[n].params}
check(f"课程只碰 {len(cur_rewards)} 项奖励的权重、{len(cur_events)} 个事件的范围：另外 16 − 2 = 14 项奖励、13 − 2 = 11 个事件从头到尾不变",
      len(cur_rewards) == 2 and len(cur_events) == 2 and len(cfg.rewards) - len(cur_rewards) == 14
      and len(cfg.events) - len(cur_events) == 11)
check("每一项的阶梯都按环境步从小到大写（源码取“最后一个满足的档”，顺序写反就全错）",
      all([s["step"] for s in stages_of(n)] == sorted(s["step"] for s in stages_of(n)) for n in NAMES))
check("读表：第一次换档（圈）= 第二档的门槛 ÷ 24，12,000 ÷ 24 = 500；质心 0.003 m × 1000 = 3 mm",
      stages_of("action_rate_weight")[1]["step"] == 12000 and 12000 / 24 == 500 and round(0.003 * 1000, 6) == 3)

print("\n18.1 第一笔账：2026-08 那次事故（每步的“函数值 × 权重”；实测数抄自配置注释，第 10 节核对注释还在）")
AIR_WALK, AIR_STAND, TAX_WALK = 1.01, 0.02, 0.77
air_max = cfg.rewards["air_time"].weight * 2              # 两只脚各 1 分
print(f"  一直走：{AIR_WALK} + (−{TAX_WALK}) = {AIR_WALK - TAX_WALK:.2f}；站着不动：{AIR_STAND} + 0 = {AIR_STAND}；"
      f"税占迈步收入 {TAX_WALK} ÷ {AIR_WALK} = {TAX_WALK / AIR_WALK:.2f}；迈步满分 {num(air_max)}")
check("迈步 air_time 每步最多 3 × 2 = 6 分（权重从 cfg 读）", air_max == 6.0 and 3 * 2 == 6)
check("迈步算分的腾空窗口是 0.125–0.3 秒（cfg）", cfg.rewards["air_time"].params["threshold_min"] == 0.125
      and cfg.rewards["air_time"].params["threshold_max"] == 0.3)
check("账表：1.01 + (−0.77) = 0.24；0.02 + 0 = 0.02；走路这两项多出 0.24 − 0.02 = 0.22",
      round(AIR_WALK - TAX_WALK, 2) == 0.24 and round(1.01 + -0.77, 2) == 0.24 and round(0.24 - 0.02, 2) == 0.22)
check("税吃掉迈步收入的 0.77 ÷ 1.01 ≈ 0.76（76%）", round(TAX_WALK / AIR_WALK, 2) == 0.76)

print("\n18.1 第二笔账：开训时光噪声就让动作平滑这一项每步多大（±σ 的“硬币”噪声，相邻两步四种组合）")
SIGMA0 = MicroduckRlCfg.actor.distribution_cfg["init_std"]   # 策略抽动作时那口钟的初始宽度 σ（第 4 章 4.7 节）
N_JOINTS = 14                                                 # 14 个舵机，每步的动作是 14 个数
coin = [(last, this, this - last, (this - last) ** 2) for last in (SIGMA0, -SIGMA0) for this in (SIGMA0, -SIGMA0)]
def signed(v) -> str:
    """带正负号：+1、−2、0。"""
    return "0" if v == 0 else f"{v:+g}".replace("-", "−")


table(["上一步的噪声", "这一步的噪声", "这一步 − 上一步", "差的平方"], [[signed(a), signed(b), signed(d), f"{q:g}"] for a, b, d, q in coin])
per_joint = sum(q for *_, q in coin) / 4
noise_cost = N_JOINTS * per_joint
check("σ 从训练配置里读出来是 1.0（开训时的值）", SIGMA0 == 1.0)
check("±1 噪声四格：差 0、−2、+2、0，平方 0、4、4、0，平均 (0 + 4 + 4 + 0) ÷ 4 = 2；(−1) − (+1) = −2，(−2) × (−2) = 4",
      [d for _, _, d, _ in coin] == [0, -2, 2, 0] and [q for *_, q in coin] == [0, 4, 4, 0] and per_joint == 2
      and (-1) - (+1) == -2 and (-2) * (-2) == 4 and (0 + 4 + 4 + 0) / 4 == 2)
check("14 个关节加起来：14 × 2 = 28", noise_cost == 28 and 14 * 2 == 28)
pairs = np.random.default_rng(18).standard_normal((100_000, 2))
gauss_mean = float(np.mean((pairs[:, 1] - pairs[:, 0]) ** 2))
print(f"  换成钟形噪声：10 万对相邻两步，差的平方平均 = {gauss_mean:.3f}（理论值 2 = 1 + 1）")
check(f"钟形噪声 10 万对：差的平方平均印成 {gauss_mean:.3f}（正文写的是 1.995）", round(gauss_mean, 3) == 1.995)
check("钟形噪声：和理论值 2 相差不到 0.03（换种子也成立）", abs(gauss_mean - 2) < 0.03)

TASK_KMAX = {"track_linear_velocity": 1, "track_angular_velocity": 1, "upright": 1, "pose": 1, "air_time": 2,
             "head_pose_tracking": 1}                        # 六项任务奖励各自核的最大值（air_time 是两只脚，所以是 2）
best = sum(cfg.rewards[n].weight * k for n, k in TASK_KMAX.items())
w0, w_last = value_of(stages_of("action_rate_weight")[0]), value_of(stages_of("action_rate_weight")[-1])
table(["动作平滑的权重", "光噪声就要扣（每步）", "六项奖励全拿满（每步）", "拿满了还剩"],
      [[wstr(w), f"{noise_cost:g} × {abs(w):.1f} = {noise_cost * abs(w):g}", f"{best:g}",
        f"{best:g} − {noise_cost * abs(w):g} = {sgn(best - noise_cost * abs(w), 1)}"] for w in (w0, w_last)])
check("六项任务奖励的满分（权重从 cfg 读）：2 + 2 + 2 + 1 + 6 + 2 = 15", best == 15 and 2 + 2 + 2 + 1 + 6 + 2 == 15)
check("−0.1：28 × 0.1 = 2.8，15 − 2.8 = 12.2；−1.0：28 × 1.0 = 28，15 − 28 = −13",
      w0 == -0.1 and w_last == -1.0 and round(noise_cost * abs(w0), 1) == 2.8 and round(15 - 2.8, 1) == 12.2
      and noise_cost * abs(w_last) == 28 and 15 - 28 == -13)
check("2.8 还不到 15 的五分之一：2.8 ÷ 15 ≈ 0.19", round(2.8 / 15, 2) == 0.19 and 2.8 / 15 < 0.2)

ARCH_READ, ARCH_LEN = -0.1028, 35.21                      # 第 14 章存档：action_rate_l2 的读数、回合平均步数
arch_raw = ARCH_READ * EPISODE_S / (w0 * DT * ARCH_LEN)
print(f"  进阶：存档倒推 {sgn(ARCH_READ)} × {num(EPISODE_S)} = {sgn(ARCH_READ * EPISODE_S, 3)}；"
      f"(−0.1) × {num(DT, 2)} × {ARCH_LEN} = {sgn(w0 * DT * ARCH_LEN, 5)}；相除 ≈ {arch_raw:.1f}")
check("进阶（字面值）：−0.1028 × 20 = −2.056；(−0.1) × 0.02 × 35.21 = −0.07042；−2.056 ÷ (−0.07042) ≈ 29.2",
      round(-0.1028 * 20, 3) == -2.056 and round(-0.1 * 0.02 * 35.21, 5) == -0.07042
      and round(-2.056 / -0.07042, 1) == 29.2 and round(arch_raw, 1) == 29.2)
check("进阶：倒推出的 29.2 比纯噪声的 28 多一点（网络算的均值本身也在变）", 28 < arch_raw < 31)
half = [(this - last) ** 2 for last in (0.5, -0.5) for this in (0.5, -0.5)]
check("自测：±0.5 噪声，平方 0、1、1、0，平均 0.5；14 × 0.5 = 7；按 −1.0 扣 7、按 −0.1 扣 0.7；28 → 7 是四分之一（0.5 × 0.5 = 0.25）",
      half == [0, 1, 1, 0] and sum(half) / 4 == 0.5 and 14 * 0.5 == 7 and round(7 * 0.1, 1) == 0.7 and 28 / 7 == 4
      and 0.5 * 0.5 == 0.25 and math.isclose(2 * 0.5**2, 0.5))
check("头部命令的最终范围 ±1.40 rad：1.40 × 57.3 ≈ 80°", round(1.40 * 57.3) == 80)

# ---------------------------------------------------------------------------
banner("2. 尺子与阶梯函数：迭代 × 24 = 环境步；两档之间一动不动")
# 一次迭代里每只机器人走几步。“改一改”第 1 条改这一行。
STEPS_PER_ITER = NUM_STEPS_PER_ENV  # TWEAK-1: 48

print(f"训练配置里的 MicroduckRlCfg.num_steps_per_env = {MicroduckRlCfg.num_steps_per_env}（一圈每只机器人真正走几步）")
print(f"配置文件开头的 NUM_STEPS_PER_ENV = {NUM_STEPS_PER_ENV}（两个权重课程的门槛用它乘）")
print(f"本节用的尺子 STEPS_PER_ITER = {STEPS_PER_ITER}")
check("一圈几步（训练配置的 num_steps_per_env）和 NUM_STEPS_PER_ENV 眼下都是 24",
      MicroduckRlCfg.num_steps_per_env == 24 == NUM_STEPS_PER_ENV)
rows = [[it, it * STEPS_PER_ITER, f"{it} × {STEPS_PER_ITER} = {it * STEPS_PER_ITER}"]
        for it in (500, 600, 750, 1000, 1500, 2000)]
table(["圈（迭代）", "环境步", "手算"], rows)
check(f"课程表上的“第 500 圈”= {500 * STEPS_PER_ITER} 个环境步（worked_examples.py 里冻结的那条手算）",
      500 * STEPS_PER_ITER == 12000)
check(f"第 600 圈 = {600 * STEPS_PER_ITER:,} 个环境步（正文写的是 14,400）", 600 * STEPS_PER_ITER == 14400)
check(f"最后一档的第 2000 圈 = {2000 * STEPS_PER_ITER:,} 个环境步（正文写的是 48,000）", 2000 * STEPS_PER_ITER == 48000)
check("反过来除：门槛 18,000 步 ÷ 24 = 第 750 圈", 18000 / 24 == 750)
vel_src = VEL_SRC_PATH.read_text(encoding="utf-8")
n_named = len(re.findall(r'"step":\s*\d+\s*\*\s*NUM_STEPS_PER_ENV', vel_src))
n_literal = len(re.findall(r'"step":\s*\d+\s*\*\s*24\b', vel_src))
print(f"配置文件里门槛的两种写法：“圈数 * NUM_STEPS_PER_ENV” {n_named} 处（两个权重课程），“圈数 * 24” 写死的 {n_literal} 处（另外四项）")
check("两种写法：两个权重课程 5 + 3 = 8 处用 NUM_STEPS_PER_ENV，另外四项 5 + 4 + 3 + 2 = 14 处写死 24；一圈几步也写死成 num_steps_per_env=24",
      n_named == 8 == 5 + 3 and n_literal == 14 == 5 + 4 + 3 + 2 and "num_steps_per_env=24," in vel_src)
check("⚠️ 只把一圈改成 48 步：门槛还是 12,000 步，12,000 ÷ 48 = 第 250 圈", 12000 / 48 == 250)


def stage_value(stages: list[dict], step: int, strict: bool = True):
    """照源码的写法查一档：从第 0 档开始，凡是“已经过了”的档都覆盖一次，最后一个满足的生效。

    strict=True 用 >（reward_weight / com_range_curriculum / standing_envs_curriculum）；
    strict=False 用 >=（pose_command_range_curriculum）。
    """
    current = value_of(stages[0])
    for s in stages:
        if (step > s["step"]) if strict else (step >= s["step"]):
            current = value_of(s)
    return current


act_stages = stages_of("action_rate_weight")
rows = [[it, it * STEPS_PER_ITER, wstr(stage_value(act_stages, it * STEPS_PER_ITER))]
        for it in (0, 300, 499, 500, 501, 700, 750, 1500, 2000)]
table(["圈", "环境步", "action_rate_l2 的权重"], rows)
check("第 0 圈到第 499 圈整整 500 圈，权重一直是 −0.1（阶梯，不是斜坡）",
      all(stage_value(act_stages, it * STEPS_PER_ITER) == -0.1 for it in range(0, 500)))
check(f"恰好走到第 {500 * STEPS_PER_ITER:,} 步时还是 −0.1（源码写的是 >，不是 >=）",
      stage_value(act_stages, 500 * STEPS_PER_ITER) == -0.1)
check(f"再走一步（第 {500 * STEPS_PER_ITER + 1:,} 步）就换成 −0.2", stage_value(act_stages, 500 * STEPS_PER_ITER + 1) == -0.2)
RAMP_625 = -0.2 + (-0.4 + 0.2) * (625 - 500) / (750 - 500)   # 500→750 圈按直线插值的话
print("  “斜坡”版本（代码里没有这回事）：如果 500→750 圈之间按直线插值，第 625 圈会是",
      wstr(RAMP_625), "——实际是", wstr(stage_value(act_stages, 625 * STEPS_PER_ITER)))
check("斜线上第 625 圈会读成 −0.3（正文和图里打叉的那个数）", round(RAMP_625, 2) == -0.3)
check("字面值：625 是正中间 (500 + 750) ÷ 2 = 625，斜线上是两头的平均 (−0.2 + (−0.4)) ÷ 2 = −0.3",
      (500 + 750) / 2 == 625 and round((-0.2 + -0.4) / 2, 2) == -0.3)

q625 = 625 * STEPS_PER_ITER
trace, w_now = [], value_of(act_stages[0])
for k, s in enumerate(act_stages):                         # 照 18.2 节那段 JS：从第 0 档起逐档比 >
    passed = q625 > s["step"]
    if passed:
        w_now = value_of(s)
    trace.append([f"第 {k} 档", s["step"], "是" if passed else "否", wstr(w_now)])
print(f"\n逐档追踪：第 625 圈开始时是第 {q625} 步")
table(["档", "门槛（环境步）", f"{q625} > 门槛？", "查完这一档，w 是"], trace)
rev = list(reversed(act_stages))                           # 把清单倒过来写，同样从上往下查
w_rev = value_of(rev[0])
for s in rev:
    if q625 > s["step"]:
        w_rev = value_of(s)
print(f"  清单倒过来写再查一遍：w = {wstr(w_rev)}")
check("第 625 圈开始时是 625 × 24 = 15,000 步；逐档查：前两档“是”、后四档“否”，最后 w = −0.2",
      q625 == 15000 and [t[2] for t in trace] == ["是", "是", "否", "否", "否", "否"] and w_now == -0.2)
check("清单倒过来写：最后一个“过了”的变成第 0 档，查出 −0.1，错了一档", w_rev == -0.1)
check("18.2 节自测：第 1300 圈查出来是 −0.8（第 1250 档生效、第 1500 档还没到）",
      stage_value(act_stages, 1300 * STEPS_PER_ITER) == -0.8)
check(f"18.2 节自测：1300 × {STEPS_PER_ITER} = {1300 * STEPS_PER_ITER:,} 个环境步（正文写的是 31,200）",
      1300 * STEPS_PER_ITER == 31200)

# ---------------------------------------------------------------------------
banner("2b. 画图：figures/ch18_step_not_ramp.png（阶梯不是斜坡）")


def step_xy(pairs, x_end):
    """把 [(圈, 值), ...] 展开成阶梯的折线：每一档先横着走，再竖着跳。"""
    xs, ys = [], []
    for i, (x, y) in enumerate(pairs):
        nxt = pairs[i + 1][0] if i + 1 < len(pairs) else x_end
        xs += [x, nxt]
        ys += [y, y]
    return xs, ys


act_pairs = [(s["step"] // STEPS_PER_ITER, value_of(s)) for s in act_stages]
IT_END = 1800
fig = plt.figure(figsize=(9.6, 11.0))
gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.58, left=0.13, right=0.965, top=0.80, bottom=0.07)
fig.suptitle("阶梯不是斜坡：两档之间，权重一动不动", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.982)

ax = fig.add_subplot(gs[0, 0])
data_axes(ax, "圈（迭代）", "action_rate_l2 的权重")
xs, ys = step_xy(act_pairs, IT_END)
ax.plot(xs, ys, color=BLUE, lw=3.2, solid_joinstyle="miter")
for x, y in act_pairs:
    ax.plot([x], [y], "o", color=ORANGE, ms=9, zorder=6)
    ax.annotate(wstr(y), (x, y), textcoords="offset points", xytext=(8, 9),
                fontsize=FS_TICK, color=ORANGE, bbox=WHITE_BOX)
ax.set_xlim(-40, IT_END)
ax.set_ylim(-1.18, 0.12)
ax.set_xticks([0, 500, 750, 1000, 1250, 1500])
ax.set_yticks([0, -0.2, -0.4, -0.6, -0.8, -1.0])
ax.yaxis.set_major_formatter(MINUS_TICKS)
top = ax.twiny()
top.set_xlim(*ax.get_xlim())
top.set_xticks([0, 500, 1000, 1500])
top.set_xticklabels(["0", "12,000", "24,000", "36,000"])
top.tick_params(labelsize=FS_TICK, colors=MUTED, length=0, pad=6)
top.set_xlabel("同一把尺子换成环境步（圈 × 24）", fontsize=FS_SMALL, color=INK, labelpad=10)
for side in ("left", "right", "bottom"):
    top.spines[side].set_visible(False)
top.spines["top"].set_color(CELL_EDGE)
hand(ax, 560, -0.93, f"500 × {STEPS_PER_ITER} = {500 * STEPS_PER_ITER:,} 个环境步", color=GREEN, fontsize=FS_SMALL)
hand(ax, 560, -1.08, "配置里写的就是这个数", color=GREEN, fontsize=FS_SMALL)
panel_title(fig, [ax, top], "① 真实的样子：走 500 圈都是 −0.1，到点一下子跳到 −0.2")
panel_note(fig, [ax], "横线 = 这一档管用的那段圈数；竖线 = 换档的那一瞬间。橙点标的是每一档的值。")

ax2 = fig.add_subplot(gs[1, 0])
data_axes(ax2, "圈（迭代）", "action_rate_l2 的权重")
ax2.plot([p[0] for p in act_pairs] + [IT_END], [p[1] for p in act_pairs] + [act_pairs[-1][1]],
         color=FAINT, lw=3.0, ls="--")
ax2.plot([p[0] for p in act_pairs], [p[1] for p in act_pairs], "o", color=FAINT, ms=8)
ax2.set_xlim(-40, IT_END)
ax2.set_ylim(-1.18, 0.12)
ax2.set_xticks([0, 500, 750, 1000, 1250, 1500])
ax2.set_yticks([0, -0.2, -0.4, -0.6, -0.8, -1.0])
ax2.yaxis.set_major_formatter(MINUS_TICKS)
ax2.plot([625], [RAMP_625], "x", color=ORANGE, ms=15, mew=3.2, zorder=7)
ax2.annotate(f"斜线上读到 {wstr(RAMP_625)}", (625, RAMP_625), textcoords="offset points", xytext=(12, -24),
             fontsize=FS_TICK, color=ORANGE, bbox=WHITE_BOX)
ax2.text(780, -0.14, "✗ 代码里没有这回事", color=ORANGE, fontsize=FS_STEP, fontweight="bold", bbox=WHITE_BOX)
ax2.text(780, -0.31, f"第 625 圈实际是 {wstr(stage_value(act_stages, 625 * STEPS_PER_ITER))}，不是 {wstr(RAMP_625)}",
         color=ORANGE, fontsize=FS_SMALL, bbox=WHITE_BOX)
panel_title(fig, [ax2], "② 把这些点连成斜线，就读错了")
panel_note(fig, [ax2], "画图的人常把几个档位连成线。代码里没有插值：权重只在换档那一步变，其余时间一动不动。")
savefig(fig, "ch18_step_not_ramp")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("3. 两个权重课程：一个越来越负，一个越来越正")
for name in ("action_rate_weight", "head_pose_bias_weight"):
    st = stages_of(name)
    print(f"\n  {name}（改的是奖励项 {cfg.curriculum[name].params['reward_name']} 的权重）")
    table(["环境步", "= 第几圈", "权重"],
          [[s["step"], s["step"] // STEPS_PER_ITER, wstr(value_of(s))] for s in st])

act_w = [value_of(s) for s in act_stages]
bias_stages = stages_of("head_pose_bias_weight")
bias_w = [value_of(s) for s in bias_stages]
check("action_rate_l2：6 档，从 −0.1 一路到 −1.0", len(act_w) == 6 and act_w[0] == -0.1 and act_w[-1] == -1.0)
check("action_rate_l2 的每一档权重都 ≤ 0（mjlab 自带的代价函数返回 ≥ 0，配负权重）", all(w <= 0 for w in act_w))
check("head_pose_bias：4 档，从 0 到 3.0", len(bias_w) == 4 and bias_w[0] == 0.0 and bias_w[-1] == 3.0)
check("head_pose_bias 的每一档权重都 ≥ 0（这个函数自己返回 ≤ 0，配正权重）", all(w >= 0 for w in bias_w))
check("head_pose_bias 在第 600 圈之前权重是 0",
      stage_value(bias_stages, 599 * STEPS_PER_ITER) == 0.0 and bias_stages[1]["step"] // STEPS_PER_ITER == 600)

# 权重 3.0 下，头部四个关节各偏 15° 和 2° 的每步代价（配置注释里记的就是这两个数）
print()
rows = []
for deg in (15.0, 2.0):
    rad = math.radians(deg)
    rows.append([f"{deg:.0f}°", f"{rad:.4f}", f"{rad * 3.0:.4f}", f"{deg:.0f}° = {rad:.4f} rad；{rad:.4f} × 3 = {rad * 3.0:.2f}"])
table(["头一直偏多少", "换成弧度", "× 权重 3.0 = 每步扣多少分", "手算"], rows)
check("权重 3.0 下，四个关节各偏 15°，每步扣 0.79 分", round(math.radians(15) * 3.0, 2) == 0.79)
check("同样的权重下，偏 2° 每步只扣 0.10 分", round(math.radians(2) * 3.0, 2) == 0.10)
check("偏 15° 要交的分是偏 2° 的 7.5 倍（15 ÷ 2）",
      round(math.radians(15) / math.radians(2), 2) == 7.5)
check("字面值重算：0.2618 × 3 = 0.79，0.0349 × 3 = 0.10",
      round(0.2618 * 3, 2) == 0.79 and round(0.0349 * 3, 2) == 0.10)
check("迷你版：两个关节变了 0.1 和 −0.2，0.1² + (−0.2)² = 0.01 + 0.04 = 0.05",
      round(0.1**2 + (-0.2) ** 2, 6) == 0.05 and round(0.01 + 0.04, 6) == 0.05)
check("读表：18,000 ÷ 24 = 750，14,400 ÷ 24 = 600；先翻倍到 −0.2、之后每档再负 0.2；最后一档是起步的 (−1.0) ÷ (−0.1) = 10 倍",
      18000 / 24 == 750 and 14400 / 24 == 600 and act_w[1] == 2 * act_w[0]
      and all(math.isclose(b - a, -0.2) for a, b in zip(act_w[1:], act_w[2:])) and round(act_w[-1] / act_w[0], 6) == 10)
TAU = cfg.rewards["head_pose_bias"].params["tau_s"]
alpha = DT / TAU
check(f"1 秒平均（tau_s = {num(TAU)} 秒）：每步新的一步占 0.02 ÷ 1 = 2%，旧的平均留 98%；+0.2、−0.2 来回摆，平均是 0",
      TAU == 1.0 and round(alpha, 6) == 0.02 and round(1 - alpha, 6) == 0.98 and (0.2 - 0.2 + 0.2 - 0.2) / 4 == 0)
check("函数自带负号：(−0.2618) × 3 = −0.79，(−0.0349) × 3 = −0.10；另一种写法 0.05 × (−0.1) = −0.005",
      round(-0.2618 * 3, 2) == -0.79 and round(-0.0349 * 3, 2) == -0.10 and round(0.05 * -0.1, 6) == -0.005
      and round(-math.radians(15) * 3.0, 2) == -0.79)
check("7.5 倍：不舍入的 0.7854 ÷ 0.1047 → 7.5；印出来的 0.79 ÷ 0.10 = 7.9（两个数各自舍入过）；平方的话 7.5 × 7.5 = 56.25",
      round(math.radians(15) * 3, 4) == 0.7854 and round(math.radians(2) * 3, 4) == 0.1047
      and round(0.7854 / 0.1047, 1) == 7.5 and round(0.79 / 0.10, 1) == 7.9 and 7.5 * 7.5 == 56.25)

# ---------------------------------------------------------------------------
banner("4. 五个范围课程：考题范围从多小放宽到多大")
st = stages_of("standing_envs")
print("\n  standing_envs（抽中“站着别动”的比例；4096 只机器人里平均有几只）")
table(["环境步", "= 第几圈", "比例", "4096 只里约几只"],
      [[s["step"], s["step"] // STEPS_PER_ITER, num(value_of(s)), round(value_of(s) * 4096)] for s in st])
check("站立比例从 2% 涨到 25%", value_of(st[0]) == 0.02 and value_of(st[-1]) == 0.25)
check("2% 的 4096 只 ≈ 82 只，25% 的 4096 只 = 1024 只",
      round(0.02 * 4096) == 82 and 0.25 * 4096 == 1024)
check("读表：4096 × 0.02 = 81.92 ≈ 82；4096 × 0.05 = 204.8 ≈ 205；4096 × 0.25 = 1024",
      round(4096 * 0.02, 2) == 81.92 and round(4096 * 0.05, 1) == 204.8 and round(4096 * 0.05) == 205 and 4096 * 0.25 == 1024)
TURN = twist.rel_turn_in_place_envs
print(f"  速度命令每 {twist.resampling_time_range[0]:g}–{twist.resampling_time_range[1]:g} 秒重抽一次；"
      f"抽中站着的，还有 {TURN:.0%} 的机会被原地转的桶改走")
check(f"速度命令每 3–8 秒重抽一次（cfg：{twist.resampling_time_range}）", twist.resampling_time_range == (3.0, 8.0))
eff0, eff_last = value_of(st[0]) * (1 - TURN), value_of(st[-1]) * (1 - TURN)
check("被原地转的桶拿走 15% 以后：0.02 × 0.85 = 0.017，4096 × 0.017 ≈ 70；0.25 × 0.85 = 0.2125，4096 × 0.2125 ≈ 870",
      TURN == 0.15 and round(1 - TURN, 2) == 0.85 and round(eff0, 4) == 0.017 and round(4096 * 0.017) == 70
      and round(eff_last, 4) == 0.2125 and round(4096 * 0.2125) == 870)
rng_cmd = np.random.default_rng(4)
u_stand, u_turn = rng_cmd.random(1_000_000), rng_cmd.random(1_000_000)
sim_frac = float(np.mean((u_stand <= 0.25) & ~(u_turn < TURN)))   # 照源码的先后：先按比例抽中站着，再被转圈的桶拿走一部分
check(f"照源码的先后抽 100 万次（比例 0.25）：最后还站着的占 {sim_frac:.4f}，和 0.25 × 0.85 = 0.2125 相差不到 0.002",
      abs(sim_frac - 0.2125) < 0.002)

for name in ("com_range", "head_com_range"):
    st = stages_of(name)
    print(f"\n  {name}（每回合抽一次的质心偏移范围）")
    table(["环境步", "= 第几圈", "范围（m）", "换成毫米"],
          [[s["step"], s["step"] // STEPS_PER_ITER, num(value_of(s), 4), f"±{num(value_of(s) * 1000, 1)} mm"] for s in st])
com_mm = [value_of(s) * 1000 for s in stages_of("com_range")]
head_com_mm = [value_of(s) * 1000 for s in stages_of("head_com_range")]
check("躯干质心：±3 → ±5 → ±10 → ±15 mm，封顶 15 mm", [round(v) for v in com_mm] == [3, 5, 10, 15])
check("头部质心：±3 → ±5 → ±10 mm，封顶 10 mm", [round(v) for v in head_com_mm] == [3, 5, 10])

head_stages = stages_of("head_pose_range")
final = value_of(head_stages[-1])
print("\n  head_pose_range（4 个头部命令的抽签范围，弧度；颈俯仰和头俯仰每一档都相同）")
rows = []
for s in head_stages:
    r = value_of(s)
    rows.append([s["step"] // STEPS_PER_ITER, f"±{num(r[0][1])}", f"±{num(r[2][1])}", f"±{num(r[3][1])}",
                 f"{round(100 * r[2][1] / final[2][1])}%"])
table(["圈", "颈俯仰 = 头俯仰", "头偏航", "头横滚", "头偏航占最终的"], rows)
check("最后一档就是第 15 章 15.5 节命令表里的上限 ±1.10 / ±1.10 / ±1.40 / ±0.31 rad（第 8 章 8.2 节也提过）",
      [max(abs(lo), abs(hi)) for lo, hi in final] == [1.10, 1.10, 1.40, 0.31])
check("最后一档从第 2000 圈起", head_stages[-1]["step"] // STEPS_PER_ITER == 2000)
check("头偏航这一列正好是最终范围的 5%、15%、35%、65%、100%",
      [round(100 * value_of(s)[2][1] / 1.40) for s in head_stages] == [5, 15, 35, 65, 100])
check("手算：1.40 × 0.05 = 0.07（起步那一档）", round(1.40 * 0.05, 2) == 0.07)
check("手算：1.40 × 0.35 = 0.49（第 1000 圈那一档）", round(1.40 * 0.35, 2) == 0.49)
check("起步的 ±0.07 rad ≈ ±4°，最后一档的 ±1.40 rad ≈ ±80°（1 rad ≈ 57.3°）",
      round(0.07 * 57.3) == 4 and round(1.40 * 57.3, 1) == 80.2)
check("18.4 节自测：第 1200 圈头偏航 ±0.49 rad ≈ ±28°，躯干质心 ±10 mm",
      round(max_abs(stage_value(head_stages, 1200 * STEPS_PER_ITER, strict=False)), 4) == 0.49
      and round(0.49 * 57.3) == 28
      and round(stage_value(stages_of("com_range"), 1200 * STEPS_PER_ITER) * 1000) == 10)
check("每一档颈俯仰和头俯仰都一样", all(value_of(s)[0] == value_of(s)[1] for s in head_stages))
check("旧课程的三档上限 0.015 → 0.02 → 0.03 m，就是 15 → 20 → 30 mm", [round(v * 1000) for v in (0.015, 0.02, 0.03)] == [15, 20, 30])
check("头部表：0.21 ÷ 1.40 = 0.15；颈俯仰 0.17 ÷ 1.10 ≈ 0.155；起步 ±0.07 rad：0.07 × 57.3 ≈ 4.0°",
      round(0.21 / 1.40, 2) == 0.15 and round(0.17 / 1.10, 3) == 0.155 and round(0.07 * 57.3, 1) == 4.0)
check("身体命令：0.005 m = 5 mm；0.05 rad：0.05 × 57.3 ≈ 2.9°；死权重的迷你版 −6 × 0 = 0",
      round(0.005 * 1000, 6) == 5 and round(0.05 * 57.3, 1) == 2.9 and -6 * 0 == 0)
body = stages_of("body_pose_range")
print("\n  body_pose_range（6 个身体命令的抽签范围）：只有一档，永远不换 →", value_of(body[0]))
switch_iters = sorted({s["step"] // STEPS_PER_ITER for n in NAMES for s in stages_of(n) if s["step"] > 0})
print("\n  7 项课程的换档圈号合起来：", switch_iters)
check("全部换档点只有 7 个圈号：500、600、750、1000、1250、1500、2000",
      switch_iters == [500, 600, 750, 1000, 1250, 1500, 2000])
check("身体命令只有一档：±0.005 m（x/y/z）和 ±0.05 rad（横滚/俯仰/偏航）",
      len(body) == 1 and [max(abs(lo), abs(hi)) for lo, hi in value_of(body[0])] == [0.005, 0.005, 0.005, 0.05, 0.05, 0.05])

# ---------------------------------------------------------------------------
banner("4b. 画图：figures/ch18_curriculum_timeline.png（7 项课程的时间线）")
IT_MAX = 2300
fig = plt.figure(figsize=(10.0, 15.6))
gs = fig.add_gridspec(4, 1, hspace=0.78, left=0.125, right=0.965, top=0.89, bottom=0.045)
SWITCH_ITERS = sorted({s["step"] // STEPS_PER_ITER for n in NAMES for s in stages_of(n) if s["step"] > 0})
fig.suptitle(f"难度只在 {len(SWITCH_ITERS)} 个圈号上换档，两档之间一动不动",
             fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.977)
TICKS = [0, 500, 750, 1000, 1500, 2000]


def timeline_axes(row, ylabel, marks):
    """一块面板。marks 是这一项（或这两项）自己的换档圈号，画成竖的虚线。"""
    ax = fig.add_subplot(gs[row, 0])
    data_axes(ax, "圈（迭代）" if row == 3 else "", ylabel)
    ax.set_xlim(-50, IT_MAX)
    ax.set_xticks(TICKS)
    if row < 3:
        ax.set_xticklabels([str(t) for t in TICKS], fontsize=FS_TICK)
    for m in marks:
        ax.axvline(m, color=FAINT, lw=0.9, ls=":", zorder=0)
    return ax


def marks_of(*names):
    return sorted({s["step"] // STEPS_PER_ITER for n in names for s in stages_of(n) if s["step"] > 0})


def draw_steps(ax, pairs, color, label, ls="-", lw=3.0):
    xs, ys = step_xy(pairs, IT_MAX)
    ax.plot(xs, ys, color=color, lw=lw, label=label, ls=ls)
    for x, y in pairs:
        ax.plot([x], [y], "o", color=color, ms=7, zorder=6)


ax = timeline_axes(0, "权重", marks_of("action_rate_weight", "head_pose_bias_weight"))
draw_steps(ax, act_pairs, BLUE, "动作平滑 action_rate_l2")
bias_pairs = [(s["step"] // STEPS_PER_ITER, value_of(s)) for s in bias_stages]
draw_steps(ax, bias_pairs, GREEN, "头部下垂 head_pose_bias")
ax.axhline(0, color=MUTED, lw=1.4)
ax.set_ylim(-1.45, 3.6)
ax.set_yticks([-1, 0, 1, 2, 3])
ax.yaxis.set_major_formatter(MINUS_TICKS)
ax.legend(fontsize=FS_TICK, loc="upper left", frameon=False)
for m in (600, 1250):   # 横轴上没单独标这两个刻度，在图里标出来
    ax.text(m, -1.38, str(m), fontsize=FS_TICK, color=MUTED, ha="center")
panel_title(fig, [ax], "① 两个权重课程：一条往下、一条往上，都是“惩罚变重”")
panel_note(fig, [ax], "蓝线的函数返回 ≥ 0，配负权重；绿线的函数自己返回 ≤ 0，配正权重。两条都在离开 0 那条横线。\n"
                       "这一块有两个横轴上没标的换档点：600（绿线）和 1250（蓝线）。")

ax = timeline_axes(1, "比例", marks_of("standing_envs"))
stand_pairs = [(s["step"] // STEPS_PER_ITER, value_of(s)) for s in stages_of("standing_envs")]
draw_steps(ax, stand_pairs, ORANGE, "")
ax.set_ylim(0, 0.30)
ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
ax.text(40, 0.122, f"2%：4096 × 0.02 ≈ {round(0.02 * 4096)} 只", fontsize=FS_TICK, color=ORANGE)
ax.text(1000, 0.268, f"25%：4096 × 0.25 = {round(0.25 * 4096)} 只", fontsize=FS_TICK, color=ORANGE)
panel_title(fig, [ax], "② 抽中“站着别动”的比例：2% → 25%")
panel_note(fig, [ax], "站着别动是真机上最常见的命令，可它得等会走路了再大量练。\n"
                       "抽中的里面，还有 15% 会被原地转圈的桶改走（18.4 节）。")

ax = timeline_axes(2, "±毫米", marks_of("com_range", "head_com_range"))
draw_steps(ax, [(s["step"] // STEPS_PER_ITER, value_of(s) * 1000) for s in stages_of("com_range")], BLUE, "躯干质心", lw=4.8)
draw_steps(ax, [(s["step"] // STEPS_PER_ITER, value_of(s) * 1000) for s in stages_of("head_com_range")], GREEN, "头部质心", ls="--")
ax.set_ylim(0, 18)
ax.set_yticks([0, 3, 5, 10, 15])
ax.legend(fontsize=FS_TICK, loc="upper left", frameon=False)
split_iter = stages_of("com_range")[-1]["step"] // STEPS_PER_ITER
ax.text(1120, 9.3, f"两条线在第 {split_iter} 圈之前完全重合\n（绿虚线底下压着蓝线）",
        fontsize=FS_TICK, color=MUTED, va="top")
panel_title(fig, [ax], "③ 每回合重抽的质心偏移范围：±3 mm → ±15 mm，到此封顶")
panel_note(fig, [ax], "再往上放就超出脚掌能托住的范围（脚跟只在踝后 20 mm），身体本身就站不住了。")

ax = timeline_axes(3, "±弧度", marks_of("head_pose_range"))
draw_steps(ax, [(s["step"] // STEPS_PER_ITER, value_of(s)[2][1]) for s in head_stages], ORANGE, "头部命令（头偏航）")
draw_steps(ax, [(0, max_abs(value_of(body[0])))], MUTED, "身体命令（最大的一项）")
ax.set_ylim(0, 1.75)
ax.set_yticks([0.07, 0.49, 0.91, 1.40])
ax.text(1180, 0.16, f"身体命令 ±{num(max_abs(value_of(body[0])))}：从头到尾不动", fontsize=FS_TICK, color=MUTED)
for s in head_stages:
    x = s["step"] // STEPS_PER_ITER
    y = value_of(s)[2][1]
    ax.annotate(f"{round(100 * y / 1.40)}%", (x, y), textcoords="offset points", xytext=(8, 6),
                fontsize=FS_TICK, color=ORANGE)
ax.legend(fontsize=FS_TICK, loc="upper left", frameon=False)
panel_title(fig, [ax], "④ 头部命令的抽签范围：最终范围的 5% → 100%；身体命令一直不动")
panel_note(fig, [ax], "起步不是 0：那 4 项输入连着的权重得从第一步起就有梯度（第 15 章 15.7 节的死权重）。")
savefig(fig, "ch18_curriculum_timeline")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("5. 课程要改哪一份配置：管理器开课时复印过一份")


@dataclass
class TermCfg:
    """一个奖励项的配置，这里只留权重这一个字段。"""
    weight: float


wall = {"action_rate_l2": TermCfg(weight=-0.1)}        # env.cfg：你写的那份
manager = deepcopy(wall)                                # 管理器 __init__ 里的 deepcopy(cfg)
print("复印之后，两份的权重：env.cfg =", wstr(wall["action_rate_l2"].weight),
      "，管理器手里 =", wstr(manager["action_rate_l2"].weight))
check("复印出来的内容一样", wall["action_rate_l2"].weight == manager["action_rate_l2"].weight)
check("但不是同一个对象（deepcopy 连里面的对象也复印了一份）",
      wall["action_rate_l2"] is not manager["action_rate_l2"])

same = wall                                             # 不复印：同一个对象多起一个名字（命令管理器的 self.cfg = cfg）
wall["action_rate_l2"].weight = -0.2                    # ✗ 改 env.cfg
print("改了 env.cfg 之后：env.cfg =", wstr(wall["action_rate_l2"].weight),
      "，管理器手里 =", wstr(manager["action_rate_l2"].weight), "← 训练用的是这一个",
      "；不复印、只起别名的那份 =", wstr(same["action_rate_l2"].weight))
check("写 env.cfg 是静默的无操作：管理器手里那份还是 −0.1", manager["action_rate_l2"].weight == -0.1)
check("不复印、只多起一个名字（引用）：两个名字是同一个对象，改一边另一边跟着变成 −0.2",
      same is wall and same["action_rate_l2"].weight == -0.2)

manager["action_rate_l2"].weight = -0.2                 # ✓ 改管理器手里那份（get_term_cfg）
print("改了管理器手里那份之后：管理器手里 =", wstr(manager["action_rate_l2"].weight))
check("通过 get_term_cfg 改，才真的改到了训练用的那份", manager["action_rate_l2"].weight == -0.2)

nested = {"rewards": {"action_rate_l2": TermCfg(weight=-0.1)}}   # 一层套一层，像 env.cfg
shallow = copy(nested)                                            # 浅复印：只复印最外层（JS 的 { ...cfg }）
deep = deepcopy(nested)                                           # 深复印：每一层都复印
nested["rewards"]["action_rate_l2"].weight = -0.2                 # 改原件里面那一层
print("进阶：改原件里面那一层之后，浅复印读到", wstr(shallow["rewards"]["action_rate_l2"].weight),
      "，深复印读到", wstr(deep["rewards"]["action_rate_l2"].weight))
check("进阶：浅复印只复印最外层，里面那层还是同一个对象，跟着变成 −0.2；深复印还是 −0.1",
      shallow is not nested and shallow["rewards"] is nested["rewards"]
      and shallow["rewards"]["action_rate_l2"].weight == -0.2 and deep["rewards"]["action_rate_l2"].weight == -0.1)

# ---------------------------------------------------------------------------
banner("5b. 画图：figures/ch18_live_cfg.png（改哪一份）")
fig, axes = lesson_figure(3, "课程要改的是管理器手里那份，不是 env.cfg", panel_height=3.3, width=9.6)


def two_boxes(ax, left_val, right_val, hot_left=False, hot_right=False):
    cell(ax, 0.35, 1.45, f"env.cfg 里写的\naction_rate_l2：{left_val}", width=3.5, height=1.45,
         facecolor=CELL_HOT if hot_left else CELL, edgecolor=ORANGE if hot_left else CELL_EDGE,
         fontsize=FS_SMALL, color=ORANGE if hot_left else INK)
    cell(ax, 5.9, 1.45, f"管理器手里那份\naction_rate_l2：{right_val}", width=3.6, height=1.45,
         facecolor=CELL_HOT if hot_right else CELL, edgecolor=ORANGE if hot_right else CELL_EDGE,
         fontsize=FS_SMALL, color=ORANGE if hot_right else INK)


lesson_panel(axes[0], "① 建管理器时，先把配置复印一份")
two_boxes(axes[0], "−0.1", "−0.1")
arrow(axes[0], (4.0, 2.18), (5.8, 2.18), MUTED, lw=2.6)
hand(axes[0], 4.20, 2.66, "deepcopy", color=MUTED, fontsize=FS_SMALL)
note(axes[0], 0.35, 0.75, "训练每一步读的都是右边这份。左边那份复印完就没人看了。")

lesson_panel(axes[1], "② 改左边：什么也没发生")
two_boxes(axes[1], "−0.2", "−0.1", hot_left=True)
hand(axes[1], 0.25, 0.95, '✗ env.cfg.rewards["action_rate_l2"].weight = −0.2', color=ORANGE, fontsize=FS_TICK)
note(axes[1], 0.35, 0.45, "不报错、不警告：奖励照旧按 −0.1 结账。")

lesson_panel(axes[2], "③ 改右边：这才是课程的写法")
two_boxes(axes[2], "−0.1", "−0.2", hot_right=True)
hand(axes[2], 0.25, 0.95, '✓ env.reward_manager.get_term_cfg("action_rate_l2").weight = −0.2', color=GREEN, fontsize=FS_TICK)
note(axes[2], 0.35, 0.45, "这一格是从 ① 的状态重来（左边那份没动过）：改了右边，下一步结账就按 −0.2 算。")
savefig(fig, "ch18_live_cfg")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("6. 换档到底发生在哪一步：> 和 >= 差一步")
# 想查第几圈。“改一改”第 3 条改这一行。
QUERY_ITER = 500  # TWEAK-3: 501


def counter_span(it: int) -> tuple[int, int]:
    """第 it 圈（从 0 数）里，common_step_counter 从哪一步走到哪一步。"""
    return it * STEPS_PER_ITER + 1, (it + 1) * STEPS_PER_ITER


rows = []
for it in (QUERY_ITER - 1, QUERY_ITER, QUERY_ITER + 1):
    first, last = counter_span(it)
    rows.append([it, f"{first}–{last}",
                 wstr(stage_value(act_stages, first)), wstr(stage_value(act_stages, last)),
                 sgn(max_abs(stage_value(head_stages, first, strict=False))),
                 sgn(max_abs(stage_value(head_stages, last, strict=False)))])
table(["圈", "这一圈的环境步", "权重（首步）", "权重（末步）", "头部范围（首步）", "头部范围（末步）"], rows)

prev_first, prev_last = counter_span(QUERY_ITER - 1)
this_first, this_last = counter_span(QUERY_ITER)
check(f"第 {QUERY_ITER - 1} 圈的最后一步是第 {prev_last} 个环境步，权重课程用 >，这一步还是 −0.1",
      stage_value(act_stages, prev_last) == -0.1)
check(f"“第 {QUERY_ITER} 圈起加难”名副其实：这一圈的首末两步（第 {this_first}、{this_last} 个环境步）查出来都是 −0.2",
      stage_value(act_stages, this_first) == -0.2 and stage_value(act_stages, this_last) == -0.2)
check(f"头部命令范围用 >=，比权重早一步：第 {QUERY_ITER - 1} 圈的最后一步（第 {prev_last} 步）就已经放宽到 ±0.21 rad",
      round(max_abs(stage_value(head_stages, prev_last, strict=False)), 4) == 0.21)
check(f"而同一步上权重还没换（第 {prev_last} 步的权重应当还是 −0.1，查出来是 {wstr(stage_value(act_stages, prev_last))}）",
      stage_value(act_stages, prev_last) != -0.2)
print(f"  一圈 {STEPS_PER_ITER} 步，两种写法差的就是这 1 步：{round(100 / STEPS_PER_ITER, 1)}% 圈。")
print("  还有一处 1 步的零头：mjlab 先加计数器、再结账、最后才在重置流程里调用课程，"
      "所以换档那一步的奖励仍按旧权重结账。")
check("第 k 圈 = 第 k × 24 + 1 到 (k + 1) × 24 步：500 × 24 + 1 = 12,001，501 × 24 = 12,024；499 × 24 + 1 = 11,977",
      500 * 24 + 1 == 12001 and 501 * 24 == 12024 and 499 * 24 + 1 == 11977)
check("> 和 >=：12,000 > 12,000 不成立，12,001 > 12,000 成立，12,000 >= 12,000 成立",
      (not 12000 > 12000) and 12001 > 12000 and 12000 >= 12000)
funcs = Counter(cfg.curriculum[n].func.__name__ for n in NAMES)
print("  7 项课程由 4 个课程函数驱动：", dict(sorted(funcs.items())))
check("4 个课程函数分 7 项：reward_weight 2、com_range_curriculum 2、pose_command_range_curriculum 2、standing_envs_curriculum 1（2 + 2 + 2 + 1 = 7）",
      funcs == {"reward_weight": 2, "com_range_curriculum": 2, "pose_command_range_curriculum": 2,
                "standing_envs_curriculum": 1} and 2 + 2 + 2 + 1 == 7)
check(f"一只都不摔、回合最长 {EPISODE_STEPS} 步：4096 只错开到点，平均每步 4096 ÷ 1000 ≈ 4 只重置；只有一只时最长等 {num(EPISODE_S)} 秒",
      EPISODE_STEPS == 1000 and round(4096 / 1000) == 4 and EPISODE_S == 20)
check("自测：只改一圈 48 步 → 12,000 ÷ 48 = 250；连 NUM_STEPS_PER_ENV 也改 → 500 × 48 = 24,000，24,000 ÷ 48 = 500",
      12000 / 48 == 250 and 500 * 48 == 24000 and 24000 / 48 == 500)

# ---------------------------------------------------------------------------
banner("7. wandb 上的 Curriculum/*：课程函数返回什么，那条曲线就是什么")


def reading(name: str, step: int) -> float:
    """复现每个课程函数 return 出去的那个数（rsl_rl 把它记成 Curriculum/<名字>）。"""
    strict = name not in ("head_pose_range", "body_pose_range")   # 这两项用 >=
    v = stage_value(stages_of(name), step, strict=strict)
    return max_abs(v) if isinstance(v, (tuple, list)) else v


rows = []
for name in NAMES:
    rows.append([f"Curriculum/{name}", sgn(reading(name, 0)), sgn(reading(name, 2000 * STEPS_PER_ITER + 1))])
table(["wandb 上的曲线", "第 0 圈的读数", "第 2000 圈以后的读数"], rows)
check("第 0 圈的 7 个读数 = 第 14 章 14.7 节那张表里的“存档读数”",
      [round(reading(n, 0), 4) for n in NAMES] == [-0.1, 0.0, 0.02, 0.003, 0.003, 0.07, 0.05])
check("到顶之后的 7 个读数", [round(reading(n, 2000 * STEPS_PER_ITER + 1), 4) for n in NAMES]
      == [-1.0, 3.0, 0.25, 0.015, 0.01, 1.4, 0.05])
check("Curriculum/head_pose_range 报的是 4 个范围里最大的那个上限：起步 0.07 = 头偏航的 ±0.07",
      round(reading("head_pose_range", 0), 4) == 0.07 and max_abs(value_of(head_stages[0])) == 0.07)
check("Curriculum/body_pose_range 从头到尾都是 0.05（6 项里最大的那个）",
      reading("body_pose_range", 0) == 0.05 and reading("body_pose_range", 10**6) == 0.05)
check("读数的单位：质心范围是米，0.015 × 1000 = 15 mm", round(0.015 * 1000, 6) == 15)
first_head, second_head = max_abs(value_of(head_stages[0])), max_abs(value_of(head_stages[1]))
mixed = (23 * first_head + second_head) / 24
print(f"进阶：第 499 圈前 23 步记 {num(first_head)}、最后 1 步记 {num(second_head)}，"
      f"这一圈平均 ({num(23 * first_head)} + {num(second_head)}) ÷ 24 = {mixed:.4f}")
check("进阶：(23 × 0.07 + 0.21) ÷ 24 = 1.82 ÷ 24 ≈ 0.0758；23 × 0.07 = 1.61",
      round(mixed, 4) == 0.0758 and round(23 * 0.07, 2) == 1.61 and round(1.61 + 0.21, 2) == 1.82 and round(1.82 / 24, 4) == 0.0758)
check(f"进阶：活满 {num(EPISODE_S)} 秒的回合是 {EPISODE_STEPS} 步，1000 ÷ 24 ≈ 41.7，约 42 圈",
      round(EPISODE_STEPS / 24) == 42 and round(1000 / 24, 1) == 41.7)

# ---------------------------------------------------------------------------
banner("8. 曲线掉了：先算一张反事实账表")
# 换档后“动作其实更平滑了”这一栏假设的原始代价。“改一改”第 2 条改这一行。
RAW_SMOOTHER = 0.03  # TWEAK-2: 0.02
# DT、EPISODE_S、EPISODE_STEPS 在第 1 节已经从 cfg 读出来了
check(f"从 cfg 读出来：一个环境步 {num(DT, 3)} 秒（decimation {cfg.decimation} × 物理步 {num(cfg.sim.mujoco.timestep, 4)} 秒）",
      DT == 0.02)
check(f"从 cfg 读出来：一个回合最长 {num(EPISODE_S, 1)} 秒 = {EPISODE_STEPS} 个环境步（正文写的是 20 秒、1000 步）",
      EPISODE_S == 20.0 and EPISODE_STEPS == 1000)


def episode_reward(raw: float, weight: float) -> float:
    """第 14 章 14.5 节的算法：每步 f × 权重 × 0.02，整回合加起来，再除以 20 秒。"""
    return raw * weight * DT * EPISODE_STEPS / EPISODE_S


cases = [("换档前（第 499 圈）", 0.05, -0.1),
         ("换档后，动作一模一样", 0.05, -0.2),
         ("换档后，动作其实更平滑", RAW_SMOOTHER, -0.2)]
table(["情况", "每步的动作差平方和", "权重", "每步记的一笔", "日志读数"],
      [[name, num(raw, 4), wstr(w), sgn(raw * w * DT, 5), sgn(episode_reward(raw, w), 5)] for name, raw, w in cases])
print("  手算：0.05 × (−0.1) × 0.02 = −0.0001；−0.0001 × 1000 步 = −0.1；−0.1 ÷ 20 秒 = −0.005")
print("  活满 20 秒时，读数正好等于“每步的函数值 × 权重”：0.05 × (−0.1) = −0.005")
check("换档前读 −0.005", round(episode_reward(0.05, -0.1), 5) == -0.005)
check("换档后、动作一模一样，读 −0.01：只因为计分口径变了", round(episode_reward(0.05, -0.2), 5) == -0.01)
check("字面值重算：0.05 × (−0.1) = −0.005，0.05 × (−0.2) = −0.01",
      round(0.05 * -0.1, 5) == -0.005 and round(0.05 * -0.2, 5) == -0.01)
check(f"动作差平方和降到 {num(RAW_SMOOTHER, 4)}，读数是 {sgn(episode_reward(RAW_SMOOTHER, -0.2), 5)}",
      round(episode_reward(RAW_SMOOTHER, -0.2), 5) == -0.006)
check(f"它比换档前的 −0.005 还低 —— 曲线掉了，动作却更平滑（{num(RAW_SMOOTHER, 4)} < 0.05）",
      episode_reward(RAW_SMOOTHER, -0.2) < episode_reward(0.05, -0.1) and RAW_SMOOTHER < 0.05)
raw_same = 0.05 * -0.1 / -0.2
print(f"  要让读数原地不动（还是 −0.005），动作差平方和得降到 {num(raw_same, 4)}：正好一半。")
check("读数不变的条件：权重翻倍，原始代价减半（0.05 → 0.025）", round(raw_same, 4) == 0.025)
check("18.8 节自测：换档后读 −0.0075，还原成原始值 −0.0075 ÷ (−0.2) = 0.0375",
      round(-0.0075 / -0.2, 4) == 0.0375)
check("自测：扣的分多了一半，0.0075 ÷ 0.005 = 1.5 倍", round(0.0075 / 0.005, 6) == 1.5)
check("约掉：0.02 × 1000 ÷ 20 = 20 ÷ 20 = 1（一步的秒数、一回合的步数、回合秒数都从 cfg 读）",
      math.isclose(DT * EPISODE_STEPS / EPISODE_S, 1.0) and round(0.02 * 1000, 6) == 20 and 20 / 20 == 1)
check(f"降了几成：(0.05 − {num(RAW_SMOOTHER, 4)}) ÷ 0.05 = {num((0.05 - RAW_SMOOTHER) / 0.05, 4)}（正文写的是 0.4）",
      round((0.05 - RAW_SMOOTHER) / 0.05, 4) == 0.4)
check("读数不变的条件（字面值）：f × (−0.2) = −0.005，两边同时除以 −0.2，f = −0.005 ÷ (−0.2) = 0.025",
      round(-0.005 / -0.2, 6) == 0.025 and round(0.025 * -0.2, 6) == -0.005)
check("还原：−0.01 ÷ (−0.2) = 0.05，−0.005 ÷ (−0.1) = 0.05——动作没变", round(-0.01 / -0.2, 6) == 0.05 and round(-0.005 / -0.1, 6) == 0.05)


def episode_reward_steps(raw: float, weight: float, steps: int) -> float:
    """只活了 steps 步的回合：每步 f × 权重 × 0.02，加 steps 步，再除以 20 秒（第 14 章 14.5 节）。"""
    return raw * weight * DT * steps / EPISODE_S


short = episode_reward_steps(0.05, -0.2, round(10 / DT))
check(f"反例：动作不变、只活 10 秒（{round(10 / DT)} 步）：0.05 × (−0.2) × 0.02 × 500 ÷ 20 = {sgn(short, 5)}，看着和换档前一样",
      round(10 / DT) == 500 and round(short, 6) == -0.005 and round(0.05 * -0.2 * 0.02 * 500 / 20, 6) == -0.005)

# ---------------------------------------------------------------------------
banner("8b. 画图：figures/ch18_weight_step_drop.png 和 18.0 节的总览图 ch18_overview.png")
fig = plt.figure(figsize=(9.8, 12.6))
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.72, 1.0], hspace=0.58, left=0.155, right=0.96, top=0.875, bottom=0.05)
fig.suptitle("动作一点没变，这条奖励曲线照样翻倍变负", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.978)
iters = np.arange(480, 521)

ax = fig.add_subplot(gs[0, 0])
data_axes(ax, "", "每步的动作差平方和")
ax.plot(iters, np.full_like(iters, 0.05, dtype=float), color=BLUE, lw=3.2)
ax.plot(iters[iters >= 500], np.full((iters >= 500).sum(), RAW_SMOOTHER), color=ORANGE, lw=3.0, ls="--")
ax.axvline(500, color=FAINT, lw=1.4, ls=":")
ax.set_xlim(480, 520)
ax.set_ylim(0, 0.065)
ax.set_xticks([480, 490, 500, 510, 520])
ax.set_yticks([0, 0.025, 0.05])
ax.text(502, 0.0535, "策略的动作没变：0.05", color=BLUE, fontsize=FS_TICK, bbox=WHITE_BOX)
ax.text(502, RAW_SMOOTHER - 0.009, f"另一种情况：真的更平滑了 {num(RAW_SMOOTHER, 4)}", color=ORANGE, fontsize=FS_TICK, bbox=WHITE_BOX)
panel_title(fig, [ax], "① 机器人干了什么：第 500 圈前后一模一样")
panel_note(fig, [ax], "这是奖励函数算出来的原始数，还没乘权重。虚线那条是“万一动作真的变好了”的假设。")

ax = fig.add_subplot(gs[1, 0])
_pos = ax.get_position()
ax.set_position([0.055, _pos.y0, 0.905, _pos.height])   # 和上下两块面板的小标题对齐
lesson_panel(ax, "② 同一段动作，换一个权重结账", xmax=10, ymax=4)
hand(ax, 0.25, 2.70, f"换档前：0.05 × (−0.1) = {sgn(episode_reward(0.05, -0.1), 5)}", color=BLUE, fontsize=FS_SMALL)
hand(ax, 0.25, 1.85, f"换档后：0.05 × (−0.2) = {sgn(episode_reward(0.05, -0.2), 5)}", color=ORANGE, fontsize=FS_SMALL)
hand(ax, 0.25, 1.00, f"更平滑：{num(RAW_SMOOTHER, 4)} × (−0.2) = {sgn(episode_reward(RAW_SMOOTHER, -0.2), 5)}",
     color=ORANGE, fontsize=FS_SMALL)
note(ax, 0.25, 0.20, f"活满 {num(EPISODE_S, 1)} 秒时，读数 = 每步的函数值 × 权重"
                     f"（每步乘 {num(DT, 2)} 秒、加 {EPISODE_STEPS} 步、再除以 {num(EPISODE_S, 1)} 秒）。")

ax = fig.add_subplot(gs[2, 0])
data_axes(ax, "圈（迭代）", "这一项的日志读数")
before = episode_reward(0.05, -0.1)
after = episode_reward(0.05, -0.2)
smoother = episode_reward(RAW_SMOOTHER, -0.2)
ax.plot(iters, np.where(iters < 500, before, after), color=BLUE, lw=3.2, drawstyle="steps-post")
ax.plot(iters[iters >= 500], np.full((iters >= 500).sum(), smoother), color=ORANGE, lw=3.0, ls="--")
ax.axvline(500, color=FAINT, lw=1.4, ls=":")
ax.set_xlim(480, 520)
ax.set_ylim(min(after, smoother) * 1.35, 0.001)
ax.set_xticks([480, 490, 500, 510, 520])
ax.set_yticks(sorted({0, round(before, 5), round(after, 5), round(smoother, 5)}))
ax.yaxis.set_major_formatter(MINUS_TICKS)
ax.annotate(sgn(before, 5), (492, before), textcoords="offset points", xytext=(0, 8), fontsize=FS_TICK, color=BLUE, ha="center")
ax.annotate(sgn(after, 5), (512, after), textcoords="offset points", xytext=(0, -20), fontsize=FS_TICK, color=BLUE, ha="center")
ax.annotate(sgn(smoother, 5), (512, smoother), textcoords="offset points", xytext=(0, 8), fontsize=FS_TICK, color=ORANGE, ha="center")
panel_title(fig, [ax], "③ wandb 的 Episode_Reward/action_rate_l2 上看到的：一个台阶")
panel_note(fig, [ax], "蓝线是“动作没变”的情况；橙虚线是“动作真的更平滑了”的情况——它也比换档前低。\n光看这条曲线掉了，分不出是哪一种。")
savefig(fig, "ch18_weight_step_drop")
plt.close(fig)

fig, axes = lesson_figure(4, "课程：一张按圈数换档的时间表，和它在曲线上的样子", panel_height=3.15, width=9.6)
lesson_panel(axes[0], "① 7 样东西随训练进度变难")
hand(axes[0], 0.3, 2.75, "动作平滑惩罚的权重   −0.1 → −1.0", fontsize=FS_SMALL)
hand(axes[0], 0.3, 2.05, "头部命令的抽签范围   ±0.07 → ±1.40 rad", fontsize=FS_SMALL)
hand(axes[0], 0.3, 1.35, "躯干质心的随机范围   ±3 → ±15 mm", fontsize=FS_SMALL)
note(axes[0], 0.3, 0.6, f"还有 4 项，本章逐项讲。其余 {len(cfg.rewards)} 项奖励、{len(cfg.events)} 个事件的设定从头到尾不变。")

lesson_panel(axes[1], "② 到点换档，尺子是环境步")
hand(axes[1], 0.3, 2.6, f"第 500 圈 = 500 × {STEPS_PER_ITER} = {500 * STEPS_PER_ITER:,} 个环境步", color=GREEN)
hand(axes[1], 0.3, 1.6, "两档之间一动不动（阶梯，不是斜坡）", color=GREEN)
note(axes[1], 0.3, 0.7, "这只钟跟机器人有几只、电脑多快都无关。")

lesson_panel(axes[2], "③ 换档 = 改管理器手里那份配置")
hand(axes[2], 0.3, 2.6, "env.reward_manager.get_term_cfg(…).weight = −0.2", color=BLUE, fontsize=FS_SMALL)
hand(axes[2], 0.3, 1.6, "改 env.cfg.rewards 不报错，也不生效", color=ORANGE, fontsize=FS_SMALL)
note(axes[2], 0.3, 0.7, "管理器在开课时复印过一份，之后只认自己那份。")

lesson_panel(axes[3], "④ 于是 wandb 的曲线上出现一个台阶")
hand(axes[3], 0.3, 2.6, f"这一项的读数：{sgn(before, 5)} → {sgn(after, 5)}", color=ORANGE)
hand(axes[3], 0.3, 1.6, "可机器人的动作一点没变", color=ORANGE)
axes[3].plot([7.2, 8.4, 8.4, 9.6], [2.55, 2.55, 1.55, 1.55], color=ORANGE, lw=3.2)
axes[3].text(7.2, 3.05, "曲线长这样", color=MUTED, fontsize=FS_SMALL)
note(axes[3], 0.3, 0.7, "看到台阶先查课程表：是考题变了，还是策略变了？")
savefig(fig, "ch18_overview")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("9. 课程跑完，训练才走了三分之一；换个“容易”法：逆向课程")
last_iter = max(s["step"] for n in NAMES for s in stages_of(n)) // STEPS_PER_ITER
sim_seconds = last_iter * STEPS_PER_ITER * DT
table(["项目", "数"], [
    ["最后一档在第几圈", last_iter],
    ["折成环境步", last_iter * STEPS_PER_ITER],
    ["每只机器人的仿真时间（秒）", sim_seconds],
    ["换成分钟", round(sim_seconds / 60, 1)],
    ["一次步态训练的常见预算（圈）", "4000–6000"],
])
check("最后一档在第 2000 圈", last_iter == 2000)
check(f"{last_iter} × {STEPS_PER_ITER} × {num(DT, 2)} = {num(sim_seconds, 1)} 秒 = {round(sim_seconds / 60, 1)} 分钟"
      f"（正文写的是 960 秒 = 16 分钟）", round(sim_seconds, 6) == 960 and round(sim_seconds / 60, 1) == 16.0)
check("按 4000 圈的预算算，课程只占前一半；按 6000 圈算，只占前三分之一",
      round(last_iter / 4000, 2) == 0.5 and round(last_iter / 6000, 4) == round(1 / 3, 4))
check("字面值：2000 × 24 = 48,000；48,000 × 0.02 = 960；960 ÷ 60 = 16",
      2000 * 24 == 48000 and round(48000 * 0.02, 6) == 960 and 960 / 60 == 16)

lx, ly, az = twist.ranges.lin_vel_x, twist.ranges.lin_vel_y, twist.ranges.ang_vel_z
p_x = 0.1 / (lx[1] - lx[0])                       # 前后速度落在 ±0.05 以内：这一段宽 0.1
p_y = 0.1 / (ly[1] - ly[0])                       # 左右速度落在 ±0.05 以内
p_w = 2 * (az[1] - 0.4) / (az[1] - az[0])         # 转速超过 0.4（往哪边都算）：两头各一段
p_spin = p_x * p_y * p_w
print(f"原地转身（几乎不平移、转得快）的机会：{p_x:.3f} × {p_y:.3f} × {p_w:.1f} = {p_spin:.4f}")
check("原地转身：0.1 ÷ 0.8 = 0.125，0.1 ÷ 0.6 ≈ 0.167，1.2 ÷ 2 = 0.6；0.125 × 0.167 × 0.6 ≈ 0.0125",
      round(p_x, 3) == 0.125 and round(p_y, 3) == 0.167 and round(p_w, 6) == 0.6 and round(p_spin, 4) == 0.0125
      and round(0.125 * 0.167 * 0.6, 4) == 0.0125 and round(1.2 / 2, 6) == 0.6)
check("原地转的桶：转速在 0.4 到 1.0 rad/s 之间（0.4 × 最大转速 1.0）", 0.4 * max(abs(az[0]), abs(az[1])) == 0.4)

from mjlab_microduck.tasks.microduck_standup_env_cfg import make_microduck_standup_env_cfg  # noqa: E402

standup = make_microduck_standup_env_cfg()
ground = standup.curriculum["ground_state_mix"]
rows = []
for s in ground.params["param_stages"]:
    p = value_of(s)
    rows.append([s["step"] // STEPS_PER_ITER, num(p["standing_prob"]), num(p["sitting_prob"]),
                 num(p["face_down_prob"]), num(p["face_up_prob"])])
table(["圈", "已经站着", "坐着", "趴着", "仰面躺着"], rows)
first, last = value_of(ground.params["param_stages"][0]), value_of(ground.params["param_stages"][-1])
check("起身任务的课程项比走路多一倍还不止", len(standup.curriculum) == 14)
check("逆向课程：起步 80% 的回合从“已经站着”或“坐着”开始，最难的仰面一档都没有",
      first["standing_prob"] + first["sitting_prob"] == 0.8 and first["face_up_prob"] == 0.0)
check("到最后一档，仰面躺着涨到 35%，已经站着降到 15%",
      last["face_up_prob"] == 0.35 and last["standing_prob"] == 0.15)
check("原地转的环境固定占 15%（显式的命令桶，不是课程）", twist.rel_turn_in_place_envs == 0.15)
check("每一档四种起点的比例加起来都是 1",
      all(abs(sum(value_of(s).values()) - 1.0) < 1e-9 for s in ground.params["param_stages"]))
check("字面值：第二行 0.25 + 0.3 + 0.35 + 0.1 = 1；起步 0.4 + 0.4 = 0.8",
      round(0.25 + 0.3 + 0.35 + 0.1, 6) == 1 and round(0.4 + 0.4, 6) == 0.8)

# ---------------------------------------------------------------------------
banner("10. 映射到项目：正文引用的常数和源码行还在不在")
REPO = Path(__file__).resolve().parents[3]
cfg_path = REPO / "src" / "mjlab_microduck" / "tasks" / "microduck_velocity_env_cfg.py"
mdp_path = REPO / "src" / "mjlab_microduck" / "tasks" / "mdp.py"
VEL_LINES = [
    "independent uniform sampling makes spin-on-the-spot ~2% of data → untrained)",
    "# Commands — deepcopy to avoid shared-state corruption from other env cfgs",
    "# post-iter-1000 reward/episode-length decline. ang ±1.0 is the big",
    "# fine_std=0.1 and the policy stopped walking entirely by iter 300 (air_time",
    "# 1.01 → 0.02, peak foot height 15 mm → 2 mm, entropy collapsed 10.9 → 1.9).",
    "# An instantaneous tight tolerance taxes walking 0.77/step — 76% of the whole",
    "# air_time reward — and is UNESCAPABLE, since a 280 g head (38% of robot",
    "# mass) must oscillate while stepping. Standing still scored higher, so it",
    "# At the optimum this costs a walking policy nothing.",
    'cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(',
    "func=microduck_mdp.reward_weight,",
    '"reward_name": "action_rate_l2",',
    '"weight_stages": [',
    '{"step": 0, "weight": -0.1},',
    '{"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.2},',
    '{"step": 1500 * NUM_STEPS_PER_ENV, "weight": -1.0},',
    '{"step": 500 * 24,    "rel_standing_envs": 0.05},',
    '{"step": 500 * 24,  "ranges": ((-0.17, 0.17),  (-0.17, 0.17),  (-0.21, 0.21),  (-0.047, 0.047))},',
    "# Capped at ±15 mm (2026-07 audit): the previous ramp to ±30 mm",
    "# exceeded the foot support polygon (heel is only 20 mm behind",
    "# ramp increases: 0.015 → 0.02 → 0.03 as policies got worse.",
    'del cfg.curriculum["command_vel"]',
    "# a gait exists. At weight 3.0 a 15° residual bias costs 0.79/step and a",
    "# 2° bias costs 0.10/step.",
    '{"step": 600 * NUM_STEPS_PER_ENV, "weight": 1.0},',
    '{"step": 1500 * NUM_STEPS_PER_ENV, "weight": 3.0},',
    "num_steps_per_env=24,",
]
MDP_LINES = [
    "def reward_weight(",
    "term_cfg = env.reward_manager.get_term_cfg(reward_name)",
    "for stage in weight_stages:",
    'if env.common_step_counter > stage["step"]:',
    'term_cfg.weight = stage["weight"]',
    "return torch.tensor([term_cfg.weight])",
    "# NOTE: must update the live EventManager term_cfg, not env.cfg.events —",
    "# EventManager.__init__ does deepcopy(cfg), so mutating env.cfg.events is a no-op.",
    "event_cfg = env.event_manager.get_term_cfg(event_name)",
    "turn_ids = env_ids[r.uniform_(0.0, 1.0) < p]",
    "mag = torch.empty(len(turn_ids), device=self.device).uniform_(0.4 * maxr, maxr)",
    "self.is_standing_env[turn_ids] = False",
    "measured at ~0.77/step against an air_time reward of",
    "~1.01/step, which is exactly what made velocity run 2026-08-20 abandon",
    "def pose_command_range_curriculum(",
    'if env.common_step_counter >= stage["step"]:',
]
standup_path = REPO / "src" / "mjlab_microduck" / "tasks" / "microduck_standup_env_cfg.py"
STANDUP_LINES = [
    'raw = os.environ.get("STANDUP_PLAY_FACE_UP")',
    "# of a flat 25/25/25/25 from step 0. With the flat split the policy optimized",
    '# trained — front only partially rose and face-up (back) froze into "do',
]
if cfg_path.is_file() and mdp_path.is_file() and standup_path.is_file():
    print("microduck_velocity_env_cfg.py：7 项课程的定义（阶梯写在 params 里）；mdp.py：reward_weight 等 4 个课程函数、原地转的桶；"
          "microduck_standup_env_cfg.py：逆向课程的来历")
    check(f"cfg 里引用的 {len(VEL_LINES)} 行都原样存在，顺序一致", lines_in_order(cfg_path.read_text(encoding="utf-8"), VEL_LINES))
    check(f"mdp.py 里引用的 {len(MDP_LINES)} 行都原样存在，顺序一致", lines_in_order(mdp_path.read_text(encoding="utf-8"), MDP_LINES))
    check(f"起身任务配置里引用的 {len(STANDUP_LINES)} 行都原样存在，顺序一致",
          lines_in_order(standup_path.read_text(encoding="utf-8"), STANDUP_LINES))
else:
    print("  （没找到项目源码，跳过这一项）")

spec = importlib.util.find_spec("mjlab")
mjlab_dir = Path(spec.origin).parent if spec and spec.origin else None
MJLAB_FILES = {
    "managers/event_manager.py": ["self.cfg = deepcopy(cfg)"],
    "managers/reward_manager.py": ["self.cfg = deepcopy(cfg)"],
    "managers/curriculum_manager.py": ['extras[f"Curriculum/{term_name}"] = term_state',
                                       "state = term_cfg.func(self._env, env_ids, **term_cfg.params)"],
    "envs/manager_based_rl_env.py": ["self.common_step_counter += 1",
                                     "self.reward_buf = self.reward_manager.compute(dt=self.step_dt)",
                                     "self.curriculum_manager.compute(env_ids=env_ids)"],
    "rl/runner.py": ['env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}',
                     'self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]'],
    "rl/vecenv_wrapper.py": ["self.env.reset()"],
    "scripts/play.py": ["env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)",
                        "env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)",
                        "runner.load("],
    "tasks/velocity/mdp/velocity_command.py": ["self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs"],
}
if mjlab_dir and mjlab_dir.is_dir():
    cmd_src = (mjlab_dir / "managers/command_manager.py").read_text(encoding="utf-8")
    check("mjlab/managers/command_manager.py：只有 self.cfg = cfg，整个文件里没有 deepcopy（18.5 节那个例外）",
          "self.cfg = cfg" in cmd_src and "deepcopy" not in cmd_src)
    for rel, wanted in MJLAB_FILES.items():
        path = mjlab_dir / rel
        ok = path.is_file() and lines_in_order(path.read_text(encoding="utf-8"), wanted)
        check(f"mjlab/{rel}：{'、'.join(wanted)}", ok)
else:
    print("  （当前 Python 环境里没有 mjlab，跳过源码核对；用 uv run 运行就会核对）")

done()
