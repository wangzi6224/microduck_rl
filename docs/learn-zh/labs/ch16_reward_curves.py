"""第 16 章实验：16 项奖励的权重与参数、六种核的形状、头部偏置的 1 秒平均、满分算术与势能式塑形。

运行：uv run python docs/learn-zh/labs/ch16_reward_curves.py            # 全部 CPU：读 cfg、手算、画五张图
      uv run python docs/learn-zh/labs/ch16_reward_curves.py --env      # 🖥️ 可选：再建 4 个环境跑 50 步验证符号（需要 GPU）

第 1–11 节只读配置和源码的文字，不建环境、不训练，纯 CPU。第 12 节要 GPU，不加 --env 就跳过。
小节编号与正文一一对应：实验第 K 节 = 正文 16.K 节（第 11 节对应「映射到项目」，第 12 节是可选的 GPU 复核）。
正文“改一改”要改的三行都带 `# TWEAK-k:` 标记（第 1、8、9 节各一处）。
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

from _common import banner, check, done, lines_in_order, num, savefig, table
from _draw import (BLUE, CELL, CELL_EDGE, FAINT, FS_NOTE, FS_SMALL, FS_STEP, FS_TICK, FS_TITLE, GREEN, GRID, INK,
                   MUTED, ORANGE, WHITE_BOX, arrow, cell, data_axes, hand, lesson_cells, lesson_figure, lesson_panel,
                   note, panel_note, panel_title, plt)

def mn(x, places: int = 4) -> str:
    """图里用的数字：负号写成 −（正文和图保持一致）。"""
    return num(x, places).replace("-", "−")


DT = 0.02          # 一个环境步 0.02 秒（第 14 章 14.1 节的第二只钟）
KERNEL_MAX_AIR = 2 # air_time 的核最大值：两只脚各 1 分
STEPS_20S = 1000   # 跑满 20 秒 = 1000 步

# ---------------------------------------------------------------------------
banner("1. 一项奖励的完整结账：0.1 m/s 的速度差，值多少分")
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (  # noqa: E402
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)

cfg = make_microduck_velocity_env_cfg()
R = cfg.rewards

# 速度核的宽容度。配置里写的是 math.sqrt(0.1)，存进 std 的就是 σ；这里用 σ² 记，和第 1 章 1.10 节一致。
STD_SQ_LIN = 0.1  # TWEAK-1: 0.5
check(f"σ² = {STD_SQ_LIN} 与 cfg 里 track_linear_velocity 的 std 对得上",
      math.isclose(R["track_linear_velocity"].params["std"] ** 2, STD_SQ_LIN))
check("track_linear_velocity 的权重是 2.0", R["track_linear_velocity"].weight == 2.0)

cmd_v = np.array([0.3, 0.0, 0.0])        # 命令：往前 0.3 m/s，不横移，竖直速度的目标恒为 0
act_v = np.array([0.2, 0.1, 0.05])       # 实际：慢了、飘了、还在往上弹
diff = cmd_v - act_v
err_sq = float(np.sum(diff**2))
kernel = math.exp(-err_sq / STD_SQ_LIN)
weighted = kernel * R["track_linear_velocity"].weight
per_step = weighted * DT
table(["逐项作差", "差的平方"], [[num(d, 4), num(d * d, 6)] for d in diff])
print(f"  ① 平方误差和 E = {num(err_sq, 6)}      （已经是平方和，不能再平方一次）")
print(f"  ② 钟形打分 f = exp(−{num(err_sq, 4)} / {STD_SQ_LIN}) = {num(kernel, 6)}")
print(f"  ③ 乘权重 w = {R['track_linear_velocity'].weight}：{num(weighted, 6)}")
print(f"  ④ 再乘一步的 {DT} 秒：这一项给这一步的分是 {num(per_step, 6)}")
check("平方误差和 (0.3−0.2)² + (0−0.1)² + 0.05² = 0.0225", round(err_sq, 6) == 0.0225)
check("exp(−0.0225 / 0.1) = exp(−0.225) = 0.798516", round(kernel, 6) == 0.798516)
check("单步贡献 0.798516 × 2 × 0.02 = 0.031941", round(per_step, 6) == 0.031941)
check("字面值重算：round(0.798516 × 2 × 0.02, 6) 还是 0.031941", round(0.798516 * 2 * 0.02, 6) == 0.031941)
check("错误写法：把 0.0225 再平方一次，分数会变成 0.99495（几乎满分，等于没罚）",
      round(math.exp(-err_sq**2 / STD_SQ_LIN), 5) == 0.99495)

# 为什么平方再相加：不平方直接加，正负会抵消；平方和开根号才是"一共差了多少"
raw_sum = float(np.sum(diff))
err_len = math.sqrt(err_sq)
sigma_lin = math.sqrt(STD_SQ_LIN)
print(f"  不平方直接加：{' + '.join('(' + mn(d, 2) + ')' for d in diff)} = {mn(raw_sum, 4)}（正负抵消，还成了负数）；"
      f"平方和开根号 = 误差箭头的长度 {num(err_len, 4)} m/s，是 σ = {num(sigma_lin, 3)} 的 {num(err_len / sigma_lin, 2)} 倍")
check("不平方直接加：0.1 + (−0.1) + (−0.05) = −0.05（往前慢的和往左飘的抵消了）", round(raw_sum, 6) == -0.05)
check("误差箭头的长度 √0.0225 = 0.15 m/s（0.15 × 0.15 = 0.0225）",
      round(err_len, 6) == 0.15 and round(0.15 * 0.15, 6) == 0.0225)
check("钟形把'差多远'换成 0 到 1 的分：e⁰ = 1，e^−1 = 0.367879，e^−10 = 0.000045（第 1 章 1.8 节的表）",
      math.exp(0) == 1.0 and round(math.exp(-1), 6) == 0.367879 and round(math.exp(-10), 6) == 0.000045)
check("σ = √0.1 ≈ 0.316；误差 0.15 ÷ 0.316 ≈ 0.47，不到半个 σ（字面值重算也是 0.47）",
      round(sigma_lin, 3) == 0.316 and round(err_len / sigma_lin, 2) == 0.47 and round(0.15 / 0.316, 2) == 0.47)

# 同一步里的第二项：动作比上一步变了多少（第 14 章 14.6 节手算过同一个 0.05）
act_delta = np.zeros(14)
act_delta[3], act_delta[9] = 0.1, -0.2
action_cost = float(np.sum(act_delta**2))
early = action_cost * -0.1 * DT
late = action_cost * -1.0 * DT
print(f"  同一步的第二项：动作变化代价 0.1² + (−0.2)² = {num(action_cost, 4)}；"
      f"权重 −0.1 时 {mn(early, 6)}，课程收紧到 −1.0 后 {mn(late, 6)}")
check("动作变化代价 0.1² + (−0.2)² = 0.05", round(action_cost, 6) == 0.05)
check("权重 −0.1：0.05 × (−0.1) × 0.02 = −0.0001", round(early, 6) == -0.0001)
check("权重 −1.0：同样的动作，扣分变十倍 = −0.001", round(late, 6) == -0.001)
check("这两项合计 0.031941 − 0.0001 = 0.031841", round(per_step + early, 6) == 0.031841)
w_lin = round(weighted, 6)   # 1.597032：正文印的那个数
check("分配律：先各自乘 0.02 再加，和先加 1.597032 − 0.005 = 1.592032 再乘 0.02，都是 0.031841",
      round(round(0.02 * w_lin, 6) + 0.02 * -0.005, 6) == 0.031841
      and round(w_lin - 0.005, 6) == 1.592032 and round(0.02 * 1.592032, 6) == 0.031841)
hz_rows = []
for hz in (50, 100):
    hz_rows.append([f"{hz} Hz", hz, round(hz * w_lin * (1 / hz), 6), round(hz * w_lin, 6)])
table(["控制频率", "一秒几步", "乘了 Δt：这一秒挣多少", "不乘 Δt：这一秒挣多少"], hz_rows, floatfmt=".7g")
check("乘了 Δt：50 × 1.597032 × 0.02 和 100 × 1.597032 × 0.01 都是 1.597032（换频率，一秒挣的一样）",
      hz_rows[0][2] == 1.597032 and hz_rows[1][2] == 1.597032)
check("不乘 Δt：50 × 1.597032 = 79.8516，100 × 1.597032 = 159.7032，翻了一倍",
      hz_rows[0][3] == 79.8516 and hz_rows[1][3] == 159.7032)
check("自测：三个差都是 0 时拿满 1 × 2 × 0.02 = 0.04；平方误差和 0.1 时 exp(−1) × 2 × 0.02 = 0.014715",
      round(1.0 * 2 * DT, 6) == 0.04 and round(math.exp(-1) * 2 * DT, 6) == 0.014715)
check("自测：平方误差和 0.1 = σ² 时，误差长度 √0.1 ≈ 0.316，正好一个 σ", round(math.sqrt(0.1), 3) == 0.316)

# ---------------------------------------------------------------------------
banner("1b. 画图：figures/ch16_one_term.png（一项奖励从误差到分数的四步）")
fig, axes = lesson_figure(4, "一项奖励的完整结账：0.1 m/s 的速度差 → 0.031941 分", panel_height=3.3)
fig.subplots_adjust(top=0.945, bottom=0.035, hspace=0.40)

ax = axes[0]
# 三行格子铺开占满整块面板（原来挤在上半边，下半边一大片空白）
lesson_panel(ax, "① 命令减实际，逐项作差（单位 m/s）")
for label, values, bottom, hi in (("命令", [0.3, 0, 0], 2.50, ()),
                                  ("实际", [0.2, 0.1, 0.05], 1.68, ()),
                                  ("差", [round(float(d), 2) for d in diff], 0.60,
                                   ((0, 0), (0, 1), (0, 2)))):
    ax.text(2.0, bottom + 0.36, label, fontsize=FS_SMALL, color=MUTED, ha="right", va="center")
    lesson_cells(ax, [values], left=2.2, bottom=bottom, width=1.3, height=0.72, highlights=hi)
ax.text(1.85, 2.04, "−", fontsize=FS_STEP, color=INK, ha="center", va="center")
ax.plot([2.1, 6.2], [1.56, 1.56], color=INK, lw=1.6)
ax.text(6.6, 2.04, "前后 / 左右 / 上下", fontsize=FS_SMALL, color=MUTED, va="center")
ax.text(6.6, 0.96, "竖直方向没有命令，目标就是 0", fontsize=FS_SMALL, color=MUTED, va="center")
note(ax, 0.2, 0.20, "命令只给两个数，实际速度有三个：第三个的目标恒为 0，所以一起作差。")

ax = axes[1]
lesson_panel(ax, "② 各自平方，加起来")
hand(ax, 0.3, 2.45, f"E = {mn(diff[0], 1)}² + ({mn(diff[1], 1)})² + ({mn(diff[2], 2)})²")
hand(ax, 0.3, 1.45, f"  = {mn(diff[0] ** 2, 2)} + {mn(diff[1] ** 2, 2)} + {mn(diff[2] ** 2, 4)}"
                    f" = {mn(err_sq, 4)}", color=ORANGE)
note(ax, 0.3, 0.5, "E 已经是“平方误差和”。下一步直接拿它去除，不要再平方一次。")

ax = axes[2]
# 横轴是“误差这根箭头有多长”，长度不会是负的 → 只画右半口钟（第 1 章 1.10 节那张图的下半张也是这么画的）
ax.set_xlim(-0.015, 0.56)
ax.set_ylim(-0.08, 1.16)
data_axes(ax, "速度误差的长度（m/s）", "这一项的分")
grid_e = np.linspace(0.0, 0.56, 400)
ax.plot(grid_e, np.exp(-grid_e**2 / STD_SQ_LIN), color=BLUE, lw=3)
ax.set_xticks([0, 0.15, 0.316, 0.5])
ax.set_xticklabels(["0", "0.15", "0.316 = σ", "0.5"])
ax.set_yticks([0, 0.367879, kernel, 1.0])
ax.set_yticklabels(["0", "0.37", num(kernel, 6), "1"])
ax.axhline(kernel, color=FAINT, ls="--", lw=1.4)
ax.plot([math.sqrt(err_sq)], [kernel], "o", color=ORANGE, ms=13, zorder=6)
ax.annotate(f"误差长度 {num(math.sqrt(err_sq), 3)} m/s\n打分 {num(kernel, 6)}",
            xy=(math.sqrt(err_sq), kernel), xytext=(0.255, 0.60), fontsize=FS_SMALL, color=ORANGE,
            bbox=WHITE_BOX, arrowprops=dict(arrowstyle="-|>", lw=2.2, color=ORANGE, shrinkA=0, shrinkB=6))
ax.text(0.30, 0.94, f"exp(−E / {STD_SQ_LIN})", fontsize=FS_SMALL, color=BLUE, bbox=WHITE_BOX)
panel_title(fig, [ax], "③ 把 E 代进第 1 章的钟形打分（横轴从 0 起：长度不会是负的）")
panel_note(fig, [ax], "钟顶在误差 0 处、高度 1。σ² = 0.1 是这一项的宽容度：偏到 σ = 0.316 m/s 就掉到 0.37 分。")

ax = axes[3]
lesson_panel(ax, "④ 乘权重，再乘一步的 0.02 秒")
hand(ax, 0.3, 2.45, f"{num(kernel, 6)} × 2 = {num(weighted, 6)}")
hand(ax, 0.3, 1.45, f"{num(weighted, 6)} × 0.02 = {num(per_step, 6)}", color=ORANGE)
note(ax, 0.3, 0.5, "这一项给这一步贡献 0.031941 分。另外 15 项各自算一遍，加起来才是这一步的总分。")
savefig(fig, "ch16_one_term")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("1c. 画图：figures/ch16_overview.png（16.0 节的总览图）")
fig, axes = lesson_figure(3, "一步的分数：16 张单子各自打分，各自加权，最后加成一个数", panel_height=3.35)

ax = axes[0]
lesson_panel(ax, "① 16 张单子，分成三堆")
for x0, n, label, color in ((0.35, 6, "6 张奖状\n做对了给分", GREEN),
                            (3.75, 8, "8 张罚单\n做错了扣分", ORANGE),
                            (7.75, 2, "2 张停用\n权重 0", MUTED)):
    for k in range(n):
        ax.add_patch(plt.Rectangle((x0 + k * 0.38, 1.75), 0.3, 0.72, facecolor=CELL,
                                   edgecolor=color, lw=2.0))
    ax.text(x0 + n * 0.19 - 0.04, 1.15, label, fontsize=FS_SMALL, color=color, ha="center", va="center")
note(ax, 0.0, 0.3, "分堆只看权重的正负号：正的是任务，负的是代价，0 的是占着位子等课程打开。")

ax = axes[1]
lesson_panel(ax, "② 每张单子先各自打一个分，再乘自己的权重")
rows = [("速度跟得准吗", num(kernel, 6), "× 2", mn(weighted, 6), GREEN),
        ("动作抖不抖", num(action_cost, 2), "× (−0.1)", mn(action_cost * -0.1, 4), ORANGE),
        ("…… 另外 14 张", "…", "…", "…", MUTED)]
for k, (name, f_v, w_v, prod, color) in enumerate(rows):
    y = 2.45 - k * 0.72
    ax.text(0.3, y, name, fontsize=FS_SMALL, color=INK, va="center")
    ax.text(3.6, y, f"打分 {f_v}", fontsize=FS_SMALL, color=color, va="center")
    ax.text(6.3, y, w_v, fontsize=FS_SMALL, color=color, va="center")
    ax.text(8.0, y, f"→ {prod}", fontsize=FS_SMALL, color=color, va="center")
note(ax, 0.3, 0.42, "函数的正负号 × 权重的正负号，必须是负的，这一项才真的在扣分（16.2 节）。")

ax = axes[2]
lesson_panel(ax, "③ 16 个加权值相加，再乘一步的 0.02 秒")
hand(ax, 0.3, 2.4, "这一步的分 = 0.02 × ( 1.597032 + (−0.005) + …… 共 16 项 )")
note(ax, 0.3, 1.5, "这一个数就是第 9 章 9.2 节那一圈里环境交回来的“奖励”。")
note(ax, 0.3, 0.9, "wandb 上每一项还单独记一笔账，整回合加起来再除以 20 秒（第 14 章 14.5 节）。")
savefig(fig, "ch16_overview")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("2. 符号约定：从 cfg 读出 16 项，分出两种写法")
SELF_NEGATING = {"head_pose_bias"}   # 函数自带负号、配正权重的那一种
rows = []
for name, term in R.items():
    fn = term.func.__name__ if hasattr(term.func, "__name__") else type(term.func).__name__
    if term.weight == 0:
        kind = "停用"
    elif name in SELF_NEGATING:
        kind = "自负号惩罚"
    elif term.weight < 0:
        kind = "代价"
    else:
        kind = "任务"
    rows.append([name, fn, term.weight, kind])
table(["奖励项", "函数", "权重", "这是哪一种"], rows, floatfmt=".4g")
pos = [n for n, t in R.items() if t.weight > 0]
neg = [n for n, t in R.items() if t.weight < 0]
zero = [n for n, t in R.items() if t.weight == 0]
print(f"  正权重 {len(pos)} 项、负权重 {len(neg)} 项、权重 0 的 {len(zero)} 项")
check("一共 16 项", len(R) == 16)
check("6 项正权重、8 项负权重、2 项权重 0", (len(pos), len(neg), len(zero)) == (6, 8, 2))
check("自负号的 head_pose_bias 配的是正权重（课程打开后是 3.0，现在是 0）",
      R["head_pose_bias"].weight >= 0)
check("名字带 _penalty 不等于自带负号：body_ang_vel 用的是 mjlab 的 body_angular_velocity_penalty，配负权重",
      R["body_ang_vel"].func.__name__ == "body_angular_velocity_penalty" and R["body_ang_vel"].weight < 0)
check("自测：函数返回 −0.2、权重写成 −3，乘出来是 +0.6（负负得正，惩罚变奖励）",
      round(-0.2 * -3, 6) == 0.6)
check("代价那一例：0.01 + 0.04 = 0.05，× (−0.1) = −0.005（16.0 节图里那个数），再 × 0.02 = −0.0001",
      round(0.01 + 0.04, 6) == 0.05 and round(0.05 * -0.1, 6) == -0.005 and round(-0.005 * DT, 6) == -0.0001)
check("自负号那一例：(−0.2) × 3 = −0.6，再 × 0.02 = −0.012；0.2 rad ≈ 11.5°",
      round(-0.2 * 3, 6) == -0.6 and round(-0.6 * DT, 6) == -0.012 and round(math.degrees(0.2), 1) == 11.5)
grid = [["代价：函数 ≥ 0", 0.05 * -0.1, 0.05 * 0.1], ["自负号惩罚：函数 ≤ 0", -0.2 * -3, -0.2 * 3]]
table(["函数的写法", "配负权重", "配正权重"], grid, floatfmt=".4g")
check("四格：−0.005 扣分（对）、+0.005（配错）、+0.6（配错）、−0.6 扣分（对）——只有异号的两格乘出负数",
      [round(v, 6) for v in (grid[0][1], grid[0][2], grid[1][1], grid[1][2])] == [-0.005, 0.005, 0.6, -0.6]
      and [v < 0 for v in (grid[0][1], grid[0][2], grid[1][1], grid[1][2])] == [True, False, False, True])
check("配错时头偏 0.4：(−0.4) × (−3) = +1.2，比偏 0.2 的 +0.6 还多——头越歪分越高", round(-0.4 * -3, 6) == 1.2)

# ---------------------------------------------------------------------------
banner("3. 六项任务奖励：参数，和每一项的一次手算")
TASK_TERMS = ["track_linear_velocity", "track_angular_velocity", "upright", "pose", "air_time",
              "head_pose_tracking"]
rows = []
for name in TASK_TERMS:
    p = R[name].params
    if "std" in p:
        key = f"σ = {num(p['std'], 6)}（σ² = {num(p['std'] ** 2, 4)}）"
    elif name == "air_time":
        key = f"腾空窗口 ({p['threshold_min']}, {p['threshold_max']}) 秒"
    else:
        key = "每个关节一个 σ，站着和走着两套"
    rows.append([name, R[name].weight, key])
table(["任务项", "权重", "宽容度 / 窗口"], rows, floatfmt=".4g")
check("三个钟的 σ² 是 0.1（线速度）、0.5（角速度）、0.05（站直）——第 1 章 1.10 节用的就是这三个",
      round(R["track_linear_velocity"].params["std"] ** 2, 4) == 0.1
      and round(R["track_angular_velocity"].params["std"] ** 2, 4) == 0.5
      and round(R["upright"].params["std"] ** 2, 4) == 0.05)
up_sig2 = R["upright"].params["std"] ** 2
gx30 = math.sin(math.radians(30))          # 第 2 章 2.4 节：前倾 30° 时，前后那一项是 0.50
upright30 = math.exp(-(gx30 ** 2) / up_sig2)
print(f"  站直那一项：前倾 30°，g_x = {num(gx30, 2)}，{num(gx30 ** 2, 2)} ÷ {num(up_sig2, 2)} = {num(gx30 ** 2 / up_sig2, 4)}，"
      f"打分 exp(−5) = {num(upright30, 6)}")
check("前倾 30°：0.5² = 0.25，0.25 ÷ 0.05 = 5，e^−5 = 0.006738（几乎 0 分）",
      round(gx30, 6) == 0.5 and round(gx30 ** 2 / up_sig2, 6) == 5 and round(upright30, 6) == 0.006738)
check("头部跟踪的 σ 是 0.5（这里配置里直接写 σ，不是 σ²）", R["head_pose_tracking"].params["std"] == 0.5)
check("腾空窗口是 (0.125, 0.3) 秒，权重 3.0",
      (R["air_time"].params["threshold_min"], R["air_time"].params["threshold_max"],
       R["air_time"].weight) == (0.125, 0.300, 3.0))
check("air_time 只在命令速度大于 0.01 时才算分（站着不迈步不算错）",
      R["air_time"].params["command_threshold"] == 0.01)

# pose 到底管哪几个关节，以及“站着窄、走着宽”这个两档说法的前提
pose_p = R["pose"].params
print(f"  pose 挑关节的正则：{pose_p['asset_cfg'].joint_names[0]}"
      f"（14 个舵机去掉 4 个颈 / 头 = 10 个腿关节）；每条腿 {len(pose_p['std_walking'])} 种："
      f"{'、'.join(k.strip('.*') for k in pose_p['std_walking'])}")
check("pose 只挑腿关节：正则把 passive_ / neck / head 都排除了",
      pose_p["asset_cfg"].joint_names == (r"^(?!passive_|.*neck.*|.*head.*).*",))
check("腿关节 5 种 × 2 条腿 = 10 个（手算里除的就是这个 10）",
      len(pose_p["std_walking"]) == 5 and len(pose_p["std_standing"]) == 5)
check("正文只说“站着 / 走着”两档，前提是跑着那一档的 σ 和走着完全一样",
      pose_p["std_running"] == pose_p["std_walking"])
check("“命令速度几乎为 0 才算站着”：切换门槛是 0.01", pose_p["walking_threshold"] == 0.01)

head_range = {st["step"] // NUM_STEPS_PER_ENV: st["ranges"]
              for st in cfg.curriculum["head_pose_range"].params["range_stages"]}
print(f"  head_pose 命令范围是课程一级级放宽的：第 0 次迭代 ±{head_range[0][0][1]} rad 起步，"
      f"第 2000 次迭代放到 ±{head_range[2000][0][1]}（头的左右转 ±{head_range[2000][2][1]}）")
check("头部命令一开始只有 ±0.05 rad 上下", head_range[0][0] == (-0.05, 0.05))
check("课程终点在第 2000 次迭代：两个俯仰 ±1.1 rad，左右转 ±1.4 rad",
      head_range[2000][0] == (-1.1, 1.1) and head_range[2000][2] == (-1.4, 1.4))

# pose：同样偏 0.2 rad 的膝盖，站着和走着扣的分不一样
std_walk_knee = 0.4
std_stand_knee = 0.15
pose_walk = math.exp(-((0.2 / std_walk_knee) ** 2) / 10)
pose_stand = math.exp(-((0.2 / std_stand_knee) ** 2) / 10)
print(f"  pose：膝盖偏 0.2 rad，其余 9 个腿关节不动。走着 σ = {std_walk_knee}：exp(−0.25 / 10) = {num(pose_walk, 6)}；"
      f"站着 σ = {std_stand_knee}：exp(−1.777778 / 10) = {num(pose_stand, 6)}")
check("pose 走路时膝盖的 σ 是 0.4、站着是 0.15（从 cfg 读）",
      R["pose"].params["std_walking"][r".*knee.*"] == 0.4
      and R["pose"].params["std_standing"][r".*knee.*"] == 0.15)
check("走着：(0.2/0.4)² = 0.25，除以 10 个关节，exp(−0.025) = 0.97531", round(pose_walk, 5) == 0.97531)
check("站着：(0.2/0.15)² = 1.777778，除以 10，exp(−0.177778) = 0.837128", round(pose_stand, 6) == 0.837128)
check("字面值重算：round(exp(−0.025), 5) 就是 0.97531", round(math.exp(-0.025), 5) == 0.97531)
check("偏了几把尺子：走着 0.2 ÷ 0.4 = 0.5，平方 0.25；站着 0.2 ÷ 0.15 = 1.3333，平方 1.777778；取平均 ÷ 10",
      round(0.2 / std_walk_knee, 6) == 0.5 and round(0.5 ** 2, 6) == 0.25
      and round(0.2 / std_stand_knee, 4) == 1.3333 and round((0.2 / std_stand_knee) ** 2, 6) == 1.777778
      and round(0.25 / 10, 6) == 0.025 and round(1.777778 / 10, 6) == 0.177778)

# head_pose_tracking：4 个钟取平均（部分分），不是相乘（全有或全无）
head_errs = np.array([0.2, 0.0, 0.0, 0.0])
per_joint = np.exp(-((head_errs / 0.5) ** 2))
head_mean = float(per_joint.mean())
head_prod = float(np.prod(per_joint))
far_one = math.exp(-((1.0 / 0.5) ** 2))
print(f"  head_pose_tracking：4 个关节误差 (0.2, 0, 0, 0) rad → 每个关节 {num(per_joint[0], 6)}、1、1、1；"
      f"取平均 {num(head_mean, 6)}，若改成相乘只剩 {num(head_prod, 6)}")
check("每个关节 exp(−(0.2/0.5)²) = exp(−0.16) = 0.852144", round(float(per_joint[0]), 6) == 0.852144)
check("4 个取平均 (0.852144 + 3) / 4 = 0.963036", round(head_mean, 6) == 0.963036)
check("字面值重算：round((0.852144 + 3) / 4, 6) 还是 0.963036", round((0.852144 + 3) / 4, 6) == 0.963036)
check("命令拉到 ±1.0 rad 时单关节还剩 exp(−4) = 0.018316（很小，但不是 0）", round(far_one, 6) == 0.018316)
check("偏了的那个关节：0.2 ÷ 0.5 = 0.4，0.4² = 0.16；四个加起来 0.852144 + 3 = 3.852144",
      round(0.2 / 0.5, 6) == 0.4 and round(0.4 ** 2, 6) == 0.16 and round(0.852144 + 3, 6) == 3.852144)
check("0.5 rad ≈ 29°，1.1 rad ≈ 63°（1 rad ≈ 57.3°）", round(math.degrees(0.5)) == 29 and round(math.degrees(1.1)) == 63)
near_one = math.exp(-((0.9 / 0.5) ** 2))
print(f"  从 1.0 rad 跟到 0.9 rad：单关节 {num(far_one, 6)} → {num(near_one, 6)}，翻了 {num(near_one / far_one, 2)} 倍")
check("从 1.0 rad 跟到 0.9 rad：(0.9 ÷ 0.5)² = 3.24，e^−3.24 = 0.039164，比 0.018316 翻了一倍多",
      round((0.9 / 0.5) ** 2, 6) == 3.24 and round(near_one, 6) == 0.039164 and near_one / far_one > 2)
check("自测：4 个关节都偏 0.5 rad → 取平均 0.367879，四个相乘 0.018316",
      round(math.exp(-1), 6) == 0.367879 and round(math.exp(-1) ** 4, 6) == 0.018316)
check("自测：四个 e^−1 相乘就是 e^−4（指数相加），和拉到 1.0 rad 时是同一个数",
      round(math.exp(-1) ** 4, 6) == round(math.exp(-4), 6) == round(far_one, 6))

# ---------------------------------------------------------------------------
banner("4. 八项代价：权重，和每一项的一次手算")
COST_TERMS = ["body_ang_vel", "angular_momentum", "dof_pos_limits", "action_rate_l2",
              "foot_clearance", "foot_swing_height", "foot_slip", "self_collisions"]
table(["代价项", "权重", "函数"],
      [[n, R[n].weight, R[n].func.__name__ if hasattr(R[n].func, "__name__") else type(R[n].func).__name__]
       for n in COST_TERMS], floatfmt=".4g")
check("八项代价的权重依次是 −0.05、−0.02、−1.0、−0.1、−2.0、−0.25、−0.1、−1.0",
      [R[n].weight for n in COST_TERMS] == [-0.05, -0.02, -1.0, -0.1, -2.0, -0.25, -0.1, -1.0])
check("foot_clearance 和 foot_swing_height 的目标脚高都是 0.02 m",
      R["foot_clearance"].params["target_height"] == 0.02
      and R["foot_swing_height"].params["target_height"] == 0.02)

body_ang = 1.0**2 + 0.5**2
limit_over = round(0.92 - 0.9, 6)
clearance_fast = abs(0.005 - 0.02) * 0.3
clearance_slow = abs(0.005 - 0.02) * 0.05
swing_low = (0.010 / 0.02 - 1) ** 2
swing_high = (0.030 / 0.02 - 1) ** 2
table(["代价项", "一次手算", "代价值", "× 权重 × 0.02 秒"],
      [["body_ang_vel", "躯干每秒翻 1.0、侧滚 0.5 rad：1² + 0.5²", body_ang, body_ang * R["body_ang_vel"].weight * DT],
       ["dof_pos_limits", "软限位 0.9，关节顶到 0.92：0.92 − 0.9", limit_over, limit_over * R["dof_pos_limits"].weight * DT],
       ["foot_clearance", "脚高 5 mm、快速划过 0.3 m/s：0.015 × 0.3", clearance_fast, clearance_fast * R["foot_clearance"].weight * DT],
       ["foot_clearance", "同样脚高，慢慢蹭 0.05 m/s：0.015 × 0.05", clearance_slow, clearance_slow * R["foot_clearance"].weight * DT],
       ["foot_swing_height", "落地时峰值只有 10 mm：(0.01/0.02 − 1)²", swing_low, swing_low * R["foot_swing_height"].weight * DT],
       ["foot_swing_height", "落地时峰值到了 30 mm：(0.03/0.02 − 1)²", swing_high, swing_high * R["foot_swing_height"].weight * DT]],
      floatfmt=".6f")
check("body_ang_vel：1² + 0.5² = 1.25，× (−0.05) × 0.02 = −0.00125", round(body_ang * -0.05 * DT, 6) == -0.00125)
check("dof_pos_limits：顶出软限位 0.02 rad，代价就是 0.02（铰链：限位内一律 0）", round(limit_over, 6) == 0.02)
check("foot_clearance：快速划过 0.015 × 0.3 = 0.0045，慢慢蹭 0.015 × 0.05 = 0.00075，贵 6 倍",
      round(clearance_fast, 6) == 0.0045 and round(clearance_slow, 6) == 0.00075
      and round(clearance_fast / clearance_slow, 6) == 6)
check("foot_swing_height：抬 10 mm 和抬 30 mm 罚得一样多，都是 0.25",
      round(swing_low, 6) == 0.25 and round(swing_high, 6) == 0.25)
check("走路任务没有力矩 / 关节速度 / 关节加速度惩罚",
      not {"joint_torques_l2", "joint_vel_l2", "joint_acc_l2"} & {
          (t.func.__name__ if hasattr(t.func, "__name__") else "") for t in R.values()})

_robot = cfg.scene.entities["robot"]
print(f"  软限位 = 量程的中间 {_robot.articulation.soft_joint_pos_limit_factor:.0%}（两端各留 5%）："
      f"soft_joint_pos_limit_factor = {_robot.articulation.soft_joint_pos_limit_factor}")
check("软限位是量程的中间 90%：soft_joint_pos_limit_factor = 0.9",
      _robot.articulation.soft_joint_pos_limit_factor == 0.9)
# 正文的构造例子：一个量程从 −1 到 1 rad 的关节。mjlab 的算法是"中点 ± 半个量程 × 0.9"（第 11 节核对这段源码）
_f = _robot.articulation.soft_joint_pos_limit_factor
_lo, _hi = -1.0, 1.0
_soft = ((_lo + _hi) / 2 - 0.5 * (_hi - _lo) * _f, (_lo + _hi) / 2 + 0.5 * (_hi - _lo) * _f)
print(f"  量程 [−1, 1] rad 的关节，软限位是 [{mn(_soft[0], 3)}, {mn(_soft[1], 3)}] rad")
check("量程 −1 到 1 rad 的关节，软限位是 −0.9 到 0.9（手算里的 0.9 就是它）",
      (round(_soft[0], 6), round(_soft[1], 6)) == (-0.9, 0.9))
check("第一行：1² + 0.5² = 1 + 0.25 = 1.25，× (−0.05) = −0.0625，再 × 0.02 = −0.00125",
      round(1 + 0.25, 6) == 1.25 and round(1.25 * -0.05, 6) == -0.0625 and round(-0.0625 * DT, 6) == -0.00125)
check("脚高 5 mm = 0.005 m，目标 2 cm = 0.02 m，差 0.015 m；0.015 × 0.3 = 0.0045，0.015 × 0.05 = 0.00075，0.3 ÷ 0.05 = 6",
      round(0.02 - 0.005, 6) == 0.015 and round(0.015 * 0.3, 6) == 0.0045 and round(0.015 * 0.05, 6) == 0.00075
      and round(0.3 / 0.05, 6) == 6)
check("落地 10 mm：10 ÷ 20 − 1 = −0.5（差五成）；30 mm：30 ÷ 20 − 1 = 0.5（多五成）；平方都是 0.25",
      round(10 / 20 - 1, 6) == -0.5 and round(30 / 20 - 1, 6) == 0.5 and round((-0.5) ** 2, 6) == round(0.5 ** 2, 6) == 0.25)
check("动作 0.10 → 0.20：目标角跳 0.1 rad ≈ 5.73°（0.1 × 57.3）",
      round(math.degrees(0.1), 2) == 5.73 and round(0.1 * 57.3, 2) == 5.73)
check("10 N 大约是 1 公斤的东西压在手上的力（10 ÷ 9.8 ≈ 1.02 公斤）", round(10 / 9.8, 2) == 1.02)
check("自测：权重 −1.0 时 0.05 × (−1.0) = −0.05，再 × 0.02 = −0.001",
      round(0.05 * -1.0, 6) == -0.05 and round(-0.05 * DT, 6) == -0.001)

import mjlab_microduck.tasks.mdp as microduck_mdp  # noqa: E402
EXTRA_SMOOTHERS = ["joint_torques_l2", "joint_torque_rate_l2", "joint_accelerations_l2",
                   "joint_vel_l2_when_standing"]
print(f"  仓库自己的 mdp.py 里另写的四个平滑项：{'、'.join(EXTRA_SMOOTHERS)}——走路任务一项都没用")
check("mdp.py 里另写了四个（力矩、力矩变化率、关节加速度、站着时的关节速度）",
      all(hasattr(microduck_mdp, n) for n in EXTRA_SMOOTHERS))
check("这四个走路任务也一项都没用",
      not set(EXTRA_SMOOTHERS) & {(t.func.__name__ if hasattr(t.func, "__name__") else "")
                                  for t in R.values()})

# ---------------------------------------------------------------------------
banner("5. 两个权重为 0 的项：位子占着，等课程打开")
print(f"  body_pose_tracking：权重 {R['body_pose_tracking'].weight}，"
      f"站姿高度 {R['body_pose_tracking'].params['nominal_height']} m，6 个轴各一个钟取平均")
print(f"  head_pose_bias：权重 {R['head_pose_bias'].weight}，平均时长 tau_s = {R['head_pose_bias'].params['tau_s']} 秒")
check("body_pose_tracking 的权重是 0（观测里那 6 个槽的对应物，第 15 章 15.7 节）",
      R["body_pose_tracking"].weight == 0.0)
check("head_pose_bias 的初始权重是 0", R["head_pose_bias"].weight == 0.0)

stages = {c.params["reward_name"]: c.params["weight_stages"]
          for c in cfg.curriculum.values() if c.func.__name__ == "reward_weight"}
table(["课程", "第几次迭代", "权重变成"],
      [[name, st["step"] // NUM_STEPS_PER_ENV, st["weight"]]
       for name in ("head_pose_bias", "action_rate_l2") for st in stages[name]], floatfmt=".4g")
check("一次迭代 = 24 个环境步", NUM_STEPS_PER_ENV == 24)
check("head_pose_bias 的课程：第 600 次迭代打开到 1.0，第 1500 次到 3.0",
      [(s["step"] // NUM_STEPS_PER_ENV, s["weight"]) for s in stages["head_pose_bias"]]
      == [(0, 0.0), (600, 1.0), (1000, 2.0), (1500, 3.0)])
check("action_rate_l2 的课程：−0.1 起步，第 1500 次迭代到 −1.0",
      stages["action_rate_l2"][0]["weight"] == -0.1 and stages["action_rate_l2"][-1]["weight"] == -1.0
      and stages["action_rate_l2"][-1]["step"] // NUM_STEPS_PER_ENV == 1500)
check("action_rate_l2 的整条阶梯：500 次 −0.2、750 次 −0.4、1000 次 −0.6、1250 次 −0.8、1500 次 −1.0",
      [(s["step"] // NUM_STEPS_PER_ENV, s["weight"]) for s in stages["action_rate_l2"]]
      == [(0, -0.1), (500, -0.2), (750, -0.4), (1000, -0.6), (1250, -0.8), (1500, -1.0)])

# ---------------------------------------------------------------------------
banner("6. 血泪教训里的数：钻空子、头奖、折中姿势、相对大小、罚不掉的税")
# 第 1 条：只盯一件事的项会被钻空子——原地踏步照样在腾空窗口里拿 air_time，是速度那一项把它拉回来
std_sq_from_cfg = R["track_linear_velocity"].params["std"] ** 2
stand_still = math.exp(-(0.3**2) / std_sq_from_cfg)
air_lo, air_hi = R["air_time"].params["threshold_min"], R["air_time"].params["threshold_max"]
print(f"  第 1 条：原地踏步、每只脚离地 0.2 秒——{air_lo} < 0.2 < {air_hi}，在腾空窗口里，air_time 照样给分；"
      f"但命令 0.3 m/s 却原地不动时，速度那一项只剩 exp(−0.09 / {num(std_sq_from_cfg, 1)}) = {num(stand_still, 4)}")
check("原地踏步每只脚离地 0.2 秒：0.125 < 0.2 < 0.3，落在腾空窗口里（air_time 只看计时）", air_lo < 0.2 < air_hi)
check("原地不动、命令 0.3 m/s：E = 0.09，exp(−0.9) = 0.4066（第 1 章 1.10 节那张表里就是这个数）",
      round(stand_still, 4) == 0.4066)
check("0.09 ÷ 0.1 = 0.9；比满分少 1 − 0.4066 = 0.5934，乘权重 2 是每步少 1.1868（未乘 0.02）",
      round(0.3**2 / std_sq_from_cfg, 6) == 0.9 and round(1 - 0.4066, 4) == 0.5934 and round(0.5934 * 2, 4) == 1.1868)
# 第 2 条：头奖的账——早到赚的是每一步的钱，限速罚款是一笔有限的钱
jackpot = 50 * 0.02
speed_fine = 50 * 0.001
print(f"  第 2 条：早到 1 秒（50 步）白赚 50 × 0.02 = {num(jackpot, 2)} 分；"
      f"冲刺那 50 步的限速罚款一共才 50 × 0.001 = {num(speed_fine, 2)} 分，差 {round(jackpot / speed_fine)} 倍")
check("头奖：早到 50 步多赚 1 分，限速罚款只有 0.05 分，相差 20 倍",
      round(jackpot, 6) == 1.0 and round(speed_fine, 6) == 0.05 and round(jackpot / speed_fine, 6) == 20)
check("1 秒 = 1 ÷ 0.02 = 50 步；回合还剩 500 步就是 500 × 0.02 = 10 分", round(1 / DT) == 50 and round(500 * 0.02, 6) == 10)
# 第 4 条：同样幅度，分几步慢慢做，平方和小得多
one_shot = 0.4**2
four_steps = 4 * 0.1**2
print(f"  第 4 条：动作要变 0.4，一步变完代价 {num(one_shot, 4)}；分 4 步每步 0.1，代价 {num(four_steps, 4)}")
check("一步变完 0.4² = 0.16；分 4 步 4 × 0.1² = 0.04，便宜 4 倍",
      round(one_shot, 6) == 0.16 and round(four_steps, 6) == 0.04 and round(one_shot / four_steps, 6) == 4)
add_all = (0.8 * 4) / 4
mul_all = 0.8**4
add_one_zero = (0.8 * 3 + 0.0) / 4
mul_one_zero = 0.8**3 * 0.0
table(["四项各自的分", "加起来取平均", "四项相乘"],
      [["0.8、0.8、0.8、0.8", add_all, mul_all],
       ["0.8、0.8、0.8、0", add_one_zero, mul_one_zero]], floatfmt=".6g")
check("加法：每项 0.8 的折中姿势能拿 0.8；一项归零还能拿 0.6",
      round(add_all, 6) == 0.8 and round(add_one_zero, 6) == 0.6)
check("乘法：0.8⁴ = 0.4096；一项归零，总分直接是 0", round(mul_all, 6) == 0.4096 and mul_one_zero == 0.0)
check("0.8⁴ 分两步算：0.8 × 0.8 = 0.64，0.64 × 0.64 = 0.4096", round(0.8 * 0.8, 6) == 0.64 and round(0.64 * 0.64, 6) == 0.4096)
tiny_all = 0.01**4
tiny_one_up = 0.01**3 * 0.02
print(f"  每项只有 0.01 分：四个相乘 {tiny_all:.0e}；一项进步到 0.02，乘积 {tiny_one_up:.0e}，只多 {tiny_one_up - tiny_all:.0e}")
check("0.01⁴ = 0.00000001；一项进步到 0.02，乘积 0.00000002，只多 0.00000001",
      round(tiny_all, 12) == 0.00000001 and round(tiny_one_up, 12) == 0.00000002
      and round(tiny_one_up - tiny_all, 12) == 0.00000001)

small_stack, big_stack = 0.3, 1.2
share_small = 0.005 / small_stack
share_big = 0.005 / big_stack
print(f"  同一笔 0.005 的扣分：任务项每步共 {small_stack} 分时占 1/60 ≈ {num(share_small, 6)}；"
      f"共 {big_stack} 分时占 1/240 ≈ {num(share_big, 6)}")
check("同一个权重，在 4 倍大的正分堆里只有 1/4 的分量（0.3 × 4 = 1.2）",
      round(share_small / share_big, 6) == 4 and round(small_stack * 4, 6) == big_stack)
check("1/60 ≈ 0.016667，1/240 ≈ 0.004167", round(share_small, 6) == 0.016667 and round(share_big, 6) == 0.004167)
print("  罚不掉的税：走路时头必然摆（第 1 章 1.14 节说的“改得掉吗”），cfg 注释里记着 fine_std = 0.1 那次"
      "每步要收 0.77 分，而当时 air_time 每步实际只拿到 1.01 分（这一项的上限是 6，见第 9 节）")
check("0.77 / 1.01 ≈ 0.76：一项罚款吃掉了迈步奖励的四分之三", round(0.77 / 1.01, 2) == 0.76)
# 这几个数读者本机复现不了，只能当史料：把它们的出处（cfg 注释）原样锁住
from mjlab_microduck.tasks import microduck_velocity_env_cfg as _vcfg  # noqa: E402
HISTORY_LINES = [
    "# Head droop fix (2026-08-20). The head walks pitched ~15° down (measured:",
    "# run ww1g2198 head_pose_tracking 1.544/2.0 → 14.6° mean joint error).",
    "# fine_std=0.1 and the policy stopped walking entirely by iter 300 (air_time",
    "# 1.01 → 0.02, peak foot height 15 mm → 2 mm, entropy collapsed 10.9 → 1.9).",
    "# An instantaneous tight tolerance taxes walking 0.77/step — 76% of the whole",
    "# air_time reward — and is UNESCAPABLE, since a 280 g head (38% of robot",
]
check("16.8 节那几个史料（15°、iter 300、1.01 → 0.02、15 mm → 2 mm、0.77/step、280 g = 38%）"
      "还原样记在 microduck_velocity_env_cfg.py 的注释里",
      lines_in_order(Path(_vcfg.__file__).read_text(encoding="utf-8"), HISTORY_LINES))

# ---------------------------------------------------------------------------
banner("7. 六种核：同样的误差，各卖多少钱")
errs = np.array([0.1, 0.2, 0.4])
table(["误差", "平方 err²", "绝对值 |err|", "钟形 exp(−err²/0.1)"],
      [[num(e, 2), e * e, abs(e), math.exp(-e * e / 0.1)] for e in errs], floatfmt=".6g")
check("平方：误差 0.1 → 0.2，代价 0.01 → 0.04，翻 4 倍", round(0.2**2 / 0.1**2, 6) == 4)
check("绝对值：同样的变化只翻 2 倍", round(0.2 / 0.1, 6) == 2)
check("钟形（σ² = 0.1）：0.1 得 0.904837，0.2 得 0.670320——和第 1 章 1.10 节那张表同一组数",
      round(math.exp(-0.01 / 0.1), 6) == 0.904837 and round(math.exp(-0.04 / 0.1), 6) == 0.670320)
check("钟形是给分：误差从 0.1 到 0.2，分数少了 0.904837 − 0.670320 = 0.234517", round(0.904837 - 0.670320, 6) == 0.234517)
check("误差 2：平方 4、绝对值 2——超过 1 以后平方反超", 2**2 == 4 and abs(-2) == 2 and 2**2 > abs(2))

# 六种核，代进同样两个数 0.1、0.2（后三种各自说明这个数代表什么）
SOFT_Q = 0.9                      # 16.4 节构造的那个关节：软限位 ±0.9 rad
FOOT_V = 0.3                      # 16.4 节的脚速 0.3 m/s
XS = (0.1, 0.2)
KERNELS = {
    "平方 err²": [x * x for x in XS],
    "绝对值 |err|": [abs(x) for x in XS],
    "钟形 exp(−err²/0.1)": [math.exp(-x * x / 0.1) for x in XS],
    f"方盒子（x = 离地秒数，窗口 {air_lo}–{air_hi}）": [1.0 if air_lo < x < air_hi else 0.0 for x in XS],
    "铰链 max(0, |x| − 0.9)（x = 关节角）": [max(0.0, abs(x) - SOFT_Q) for x in XS],
    "距离 × 速度 |x| × 0.3（x = 离目标高度）": [abs(x) * FOOT_V for x in XS],
}
table(["核", "代进 0.1", "代进 0.2"], [[k, v[0], v[1]] for k, v in KERNELS.items()], floatfmt=".6g")
check("代进 0.2，六种核开出六个价：0.04、0.2、0.670320、1、0、0.06",
      [round(v[1], 6) for v in KERNELS.values()] == [0.04, 0.2, 0.670320, 1.0, 0.0, 0.06])
check("代进 0.1：0.01、0.1、0.904837、0（还不到 0.125 秒）、0（在软限位以内）、0.03",
      [round(v[0], 6) for v in KERNELS.values()] == [0.01, 0.1, 0.904837, 0.0, 0.0, 0.03])
check("铰链要到 0.92 才收 0.92 − 0.9 = 0.02；距离 × 速度在脚速 0 时一律 0",
      round(max(0.0, 0.92 - SOFT_Q), 6) == 0.02 and all(abs(x) * 0.0 == 0.0 for x in XS))

combos = [("平方误差和直接取指数（= 每项的钟相乘）", math.exp(-(0 + 1))),
          ("先把平方误差平均，再取指数", math.exp(-(0 + 1) / 2)),
          ("每项先取指数，再平均（项目里头部跟踪用的）", (math.exp(0) + math.exp(-1)) / 2)]
table(["两项误差是 0 和 σ 时，.mean() 放哪一行", "总分"], [[k, v] for k, v in combos], floatfmt=".6g")
check("三种写法给出三个不同的数：0.367879、0.606531、0.683940",
      [round(v, 6) for _, v in combos] == [0.367879, 0.606531, 0.683940])
check("第一种 = 每项的钟相乘：e^−0 × e^−1 = e^−(0+1)（数相乘，指数相加）",
      math.isclose(math.exp(-0) * math.exp(-1), math.exp(-(0 + 1))))
check("第二种 = 几何平均：√(1 × 0.367879) ≈ 0.6065，和 0.606531 是同一个数",
      round(math.sqrt(1 * 0.367879), 4) == 0.6065 and round(combos[1][1], 4) == 0.6065)
check("自测：代价 0.1 时误差翻倍 → 平方核变 0.4，绝对值核变 0.2",
      round(0.1 * 2**2, 6) == 0.4 and round(0.1 * 2, 6) == 0.2)

# ---------------------------------------------------------------------------
banner("7b. 画图：figures/ch16_reward_kernels.png（六种核的形状）")
fig = plt.figure(figsize=(14.5, 13.6))
fig.suptitle("六种核：同一个“差得多远”，卖出六种不同的价钱", fontsize=FS_TITLE, fontweight="bold", color=INK,
             y=0.975)
gs = fig.add_gridspec(3, 2, hspace=0.52, wspace=0.22, left=0.07, right=0.98, top=0.915, bottom=0.05)


def _sub(r, c, title, xlabel, ylabel):
    ax = fig.add_subplot(gs[r, c])
    data_axes(ax, xlabel, ylabel)
    ax.set_title(title, fontsize=FS_STEP, color=INK, pad=12)
    return ax


ax = _sub(0, 0, "① 钟形：σ 越大越宽容（任务奖励）", "误差", "分数（越高越好）")
e = np.linspace(0, 1.0, 400)
for s2, lab, col in ((0.05, "σ² = 0.05 站直", ORANGE), (0.1, "σ² = 0.1 线速度", BLUE), (0.5, "σ² = 0.5 角速度", GREEN)):
    ax.plot(e, np.exp(-e**2 / s2), color=col, lw=3, label=lab)
ax.plot([0.2], [math.exp(-0.04 / 0.1)], "o", color=BLUE, ms=11)
ax.annotate(f"误差 0.2 → {math.exp(-0.04 / 0.1):.6f}", xy=(0.2, math.exp(-0.04 / 0.1)), xytext=(0.36, 0.52),
            fontsize=FS_TICK, color=BLUE, bbox=WHITE_BOX,
            arrowprops=dict(arrowstyle="-|>", lw=2.0, color=BLUE, shrinkA=0, shrinkB=6))
ax.legend(fontsize=FS_TICK, frameon=False)
ax.set_ylim(-0.04, 1.08)

ax = _sub(0, 1, "② 平方 vs 绝对值：远处谁涨得快（代价）", "误差", "代价（越低越好）")
x = np.linspace(-0.45, 0.45, 400)
ax.plot(x, x**2, color=BLUE, lw=3, label="平方 err²（动作抖、身体晃）")
ax.plot(x, np.abs(x), color=ORANGE, lw=3, label="绝对值 |err|（头部偏置）")
for xx in (0.1, 0.2):
    ax.plot([xx], [xx**2], "o", color=BLUE, ms=10)
    ax.plot([xx], [abs(xx)], "o", color=ORANGE, ms=10)
ax.text(0.225, 0.055, "0.01 → 0.04（翻 4 倍）", fontsize=FS_TICK, color=BLUE, bbox=WHITE_BOX)
ax.text(0.225, 0.245, "0.1 → 0.2（翻 2 倍）", fontsize=FS_TICK, color=ORANGE, bbox=WHITE_BOX)
ax.legend(fontsize=FS_TICK, frameon=False, loc="upper center")
ax.set_ylim(-0.03, 0.52)

ax = _sub(1, 0, "③ 盒式指示器：窗口内 1 分，窗口外 0 分", "这只脚腾空了多久（秒）", "这只脚的分")
t = np.linspace(0, 0.5, 600)
ax.plot(t, ((t > 0.125) & (t < 0.300)).astype(float), color=BLUE, lw=3)
ax.set_xticks([0, 0.125, 0.3, 0.5])
ax.set_yticks([0, 1])
ax.set_ylim(-0.08, 1.2)
ax.text(0.2125, 1.06, "腾空窗口", fontsize=FS_TICK, color=ORANGE, ha="center")
arrow(ax, (0.125, 1.02), (0.300, 1.02), ORANGE, lw=2.2, style="<|-|>")
ax.text(0.33, 0.12, "太久 = 在跳", fontsize=FS_TICK, color=MUTED)
ax.text(0.012, 0.12, "太短 = 拖着脚", fontsize=FS_TICK, color=MUTED)

ax = _sub(1, 1, "④ 铰链：软限位里一律 0，出去才线性涨", "关节角（软限位 ±0.9 rad）", "代价")
q = np.linspace(-1.1, 1.1, 600)
ax.plot(q, np.clip(-0.9 - q, 0, None) + np.clip(q - 0.9, 0, None), color=BLUE, lw=3)
ax.axvspan(-0.9, 0.9, color=GRID, alpha=0.55, lw=0)
ax.set_xticks([-0.9, 0, 0.9])
ax.plot([0.92], [0.02], "o", color=ORANGE, ms=11)
ax.annotate("顶到 0.92：代价 0.02", xy=(0.92, 0.02), xytext=(0.12, 0.12), fontsize=FS_TICK, color=ORANGE,
            arrowprops=dict(arrowstyle="-|>", lw=2.0, color=ORANGE, shrinkA=0, shrinkB=6))
ax.text(-0.86, 0.17, "灰色区间内一律 0 分\n（贴着边走也不扣）", fontsize=FS_TICK, color=MUTED)
ax.set_ylim(-0.015, 0.24)

ax = _sub(2, 0, "⑤ 距离 × 脚速：拖得快才贵", "脚离地多高（米）", "这只脚的代价")
h = np.linspace(0, 0.05, 400)
ax.plot(h, np.abs(h - 0.02) * 0.30, color=BLUE, lw=3, label="脚速 0.30 m/s")
ax.plot(h, np.abs(h - 0.02) * 0.05, color=GREEN, lw=3, label="脚速 0.05 m/s")
ax.plot([0.005], [clearance_fast], "o", color=BLUE, ms=11)
ax.plot([0.005], [clearance_slow], "o", color=GREEN, ms=11)
ax.text(0.0078, clearance_fast + 0.0004, f"{num(clearance_fast, 4)}", fontsize=FS_TICK, color=BLUE, bbox=WHITE_BOX)
ax.text(0.0078, clearance_slow - 0.0006, f"{num(clearance_slow, 5)}", fontsize=FS_TICK, color=GREEN, bbox=WHITE_BOX)
ax.set_xticks([0, 0.02, 0.04])
ax.legend(fontsize=FS_TICK, frameon=False)

ax = _sub(2, 1, "⑥ 落地结账：比目标高和比目标低，一样罚", "落地前这一步抬到的最高点（毫米）", "落地那一步的代价")
pk = np.linspace(0, 0.04, 400)
ax.plot(pk * 1000, (pk / 0.02 - 1) ** 2, color=BLUE, lw=3)
for mm, val in ((10, swing_low), (30, swing_high)):
    ax.plot([mm], [val], "o", color=ORANGE, ms=11)
    ax.text(mm, val + 0.045, f"{mm} mm → {num(val, 2)}", fontsize=FS_TICK, color=ORANGE, ha="center")
ax.set_xticks([0, 10, 20, 30, 40])
ax.set_ylim(-0.04, 1.15)
ax.text(20, 0.10, "目标 20 mm 处代价为 0", fontsize=FS_TICK, color=MUTED, ha="center")
savefig(fig, "ch16_reward_kernels")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("8. 只收直流分量：头部偏置的 1 秒平均")
TAU_S = 1.0  # TWEAK-2: 0.2
check(f"tau_s = {TAU_S} 秒与 cfg 里 head_pose_bias 的参数对得上", R["head_pose_bias"].params["tau_s"] == TAU_S)
alpha = min(1.0, DT / TAU_S)
print(f"  α = 一步 {DT} 秒 ÷ {TAU_S} 秒 = {num(alpha, 4)}；旧值留 {num(1 - alpha, 4)}（第 7 章 7.4 节的滑动平均）")

# 事故：σ = 0.5 的宽钟对 10° 的下垂几乎不收钱；σ = 0.1 的窄钟几乎全扣
check("15° ≈ 0.26 rad（15 ÷ 57.3）", round(math.radians(15), 2) == 0.26 and round(15 / 57.3, 2) == 0.26)
sag10 = 1 - (math.exp(-((math.radians(10) / 0.5) ** 2)) + 3) / 4
print(f"  为什么不直接把 head_pose_tracking 调严：一个关节下垂 10°（{math.radians(10):.4f} rad），"
      f"4 个取平均后一共只掉 {num(sag10, 4)} 分")
check("下垂 10° 只掉 0.0287 分——配置注释里说的“几乎免费”", round(sag10, 4) == 0.0287)
check("手算链：10 ÷ 57.3 = 0.1745 rad；(0.1745 ÷ 0.5)² = 0.349² = 0.1218；e^−0.1218 = 0.8853；1 − 0.8853 = 0.1147；÷ 4 = 0.0287",
      round(math.radians(10), 4) == 0.1745 and round(10 / 57.3, 4) == 0.1745 and round(0.1745 / 0.5, 4) == 0.349
      and round(0.349**2, 4) == 0.1218 and round(math.exp(-0.1218), 4) == 0.8853
      and round(1 - 0.8853, 4) == 0.1147 and round(0.1147 / 4, 4) == 0.0287)
narrow10 = math.exp(-((0.1745 / 0.1) ** 2))
print(f"  叠一口 σ = 0.1 的窄钟：同样下垂 10°，(0.1745 ÷ 0.1)² = {num((0.1745 / 0.1) ** 2, 3)}，"
      f"这口钟只给 {num(narrow10, 4)} 分")
check("窄钟：(0.1745 ÷ 0.1)² = 3.045，e^−3.045 = 0.0476（几乎一分不给）",
      round((0.1745 / 0.1) ** 2, 3) == 3.045 and round(math.exp(-3.045), 4) == 0.0476 and round(narrow10, 4) == 0.0476)

# 办法一：先平均再取绝对值
swing = np.array([0.2, -0.2, 0.2, -0.2])
print(f"  走路时头来回摆 [{', '.join(num(v, 1) for v in swing)}]：先平均再取绝对值 = {num(abs(swing.mean()), 4)}；"
      f"先取绝对值再平均 = {num(np.abs(swing).mean(), 4)}")
check("来回摆：先平均再取绝对值是 0（罚不到），先取绝对值再平均是 0.2（罚得到）",
      round(abs(float(swing.mean())), 6) == 0.0 and round(float(np.abs(swing).mean()), 6) == 0.2)
mixed = np.array([-0.3, 0.1, -0.3, 0.1])          # 一直低 0.1，再叠上 ±0.2 的摆动
m_mean, m_abs = float(mixed.mean()), float(np.abs(mixed).mean())
m_sq, m_var = float((mixed**2).mean()), float(mixed.var())   # np.var 除以个数 n，和第 4 章 4.4 节手算一样
print(f"  又摆又偏 [{', '.join(mn(v, 1) for v in mixed)}]：先平均再取绝对值 = {num(abs(m_mean), 4)}；先取绝对值再平均 = {num(m_abs, 4)}；"
      f"平方的平均 {num(m_sq, 4)} = 平均的平方 {num(m_mean**2, 4)} + 方差 {num(m_var, 4)}")
check("又摆又偏：平均 −0.4 ÷ 4 = −0.1，绝对值 0.1 正是'一直低着'的那部分；先取绝对值再平均是 0.2",
      round(m_mean, 6) == -0.1 and round(-0.4 / 4, 6) == -0.1 and round(m_abs, 6) == 0.2)
check("第 4 章 4.4 节：平方的平均 0.05 = 平均的平方 0.01 + 方差 0.04；摆动占 0.04 ÷ 0.05 = 八成",
      round(m_sq, 6) == 0.05 and round(m_mean**2, 6) == 0.01 and round(m_var, 6) == 0.04
      and round(0.01 + 0.04, 6) == 0.05 and round(0.04 / 0.05, 6) == 0.8)

# 办法二：滑动平均。α = 第 7 章 7.4 节的 1 − β
beta = 1 - alpha
memory_steps = 1 / (1 - beta)
recent_share = 1 - beta ** round(memory_steps)
print(f"  β = 1 − α = {num(beta, 4)}：记得最近 1 ÷ (1 − β) = {num(memory_steps, 2)} 步 = {num(memory_steps * DT, 2)} 秒，"
      f"这几步合起来占 {num(recent_share, 5)}")
check("1 − α = 0.98 就是第 7 章 7.4 节的 β；记性 1 ÷ 0.02 = 50 步，50 × 0.02 = 1 秒",
      round(beta, 6) == 0.98 and round(memory_steps, 6) == 50 and round(memory_steps * DT, 6) == 1.0)
check("最近 50 步占 1 − 0.98⁵⁰ = 1 − 0.36417 = 0.63583，约三分之二（7.4 节的 0.651、0.632 同一个规律）",
      round(recent_share, 5) == 0.63583 and round(1 - 0.36417, 5) == 0.63583
      and round(1 - 0.9**10, 3) == 0.651 and round(1 - 0.999**1000, 3) == 0.632)
check("α = 一步的秒数 ÷ 想记住的秒数：记 0.2 秒就是 0.02 ÷ 0.2 = 0.1", round(DT / 0.2, 6) == 0.1)

bias = 0.2
ema, first_two = 0.0, []
for k in range(50):
    ema = (1 - alpha) * ema + alpha * bias
    if k < 2:
        first_two.append(ema)
closed = bias * (1 - (1 - alpha) ** 50)
print(f"  头一直低 {bias} rad：第 1 步 EMA = {num(first_two[0], 5)}，第 2 步 = {num(first_two[1], 5)}，"
      f"第 50 步 = {num(ema, 6)}")
check("第 1 步 0.98 × 0 + 0.02 × 0.2 = 0.004", round(first_two[0], 6) == 0.004)
check("第 2 步 0.98 × 0.004 + 0.02 × 0.2 = 0.00392 + 0.004 = 0.00792",
      round(first_two[1], 6) == 0.00792 and round(0.98 * 0.004, 6) == 0.00392)
check("第 50 步 = 0.2 × (1 − 0.98⁵⁰) = 0.127166（递推 = 份额算法 = 几何级数）",
      math.isclose(ema, closed, rel_tol=1e-12) and round(ema, 6) == 0.127166)
check("0.98⁵⁰ = 0.36417：起步时那个假的 0，50 步后还占三成多的份额", round(0.98**50, 5) == 0.36417)
check("字面值重算：0.2 × (1 − 0.36417) = 0.2 × 0.63583 = 0.127166",
      round(0.2 * (1 - 0.36417), 6) == 0.127166 and round(0.2 * 0.63583, 6) == 0.127166)
check("拿式子核对前两步：0.2 × (1 − 0.98) = 0.004；0.98² = 0.9604，0.2 × 0.0396 = 0.00792",
      round(0.2 * (1 - 0.98), 6) == 0.004 and round(0.98**2, 4) == 0.9604
      and round(1 - 0.9604, 4) == 0.0396 and round(0.2 * 0.0396, 6) == 0.00792)
check("两秒以前的旧账还占 0.98¹⁰⁰ ≈ 0.13（所以回合一重置就要清零）", round(0.98**100, 2) == 0.13)
check("进阶折叠：n = 3 时 S = 1 + 0.98 + 0.9604 = 2.9404；0.98³ = 0.941192；(1 − 0.941192) ÷ 0.02 = 2.9404",
      round(1 + 0.98 + 0.9604, 4) == 2.9404 and round(0.98**3, 6) == 0.941192
      and round(1 - 0.941192, 6) == 0.058808 and round(0.058808 / 0.02, 4) == 2.9404 and round(0.004 / 0.02, 6) == 0.2)

HALF = 10          # 半个步态周期 = 10 步 = 0.2 秒（腾空窗口 0.125–0.3 秒量级）
ema_swing, peak = 0.0, 0.0
for k in range(1000):
    ema_swing = (1 - alpha) * ema_swing + alpha * (bias * (1 if (k // HALF) % 2 == 0 else -1))
    if k > 500:
        peak = max(peak, abs(ema_swing))
q_half = (1 - alpha) ** HALF
peak_closed = bias * (1 - q_half) / (1 + q_half)       # 进阶折叠：a = 0.81707 × (−a) + 0.18293 × 0.2
print(f"  换成 ±{bias} rad、0.4 秒一个来回的摆动：EMA 稳定下来只在 ±{num(peak, 4)} 之间晃，"
      f"约为持续偏置 {num(ema, 6)} 的六分之一；式子给 {num(peak_closed, 4)}")
check("来回摆动的 1 秒平均稳定在 ±0.0201，只有持续偏置 0.127166 的六分之一", round(peak, 4) == 0.0201)
check(f"进阶折叠的式子 a = 0.2 × (1 − q) ÷ (1 + q)，q = (1 − α)^10，给出 {num(peak_closed, 4)}，和逐步模拟一致",
      math.isclose(peak, peak_closed, rel_tol=1e-3))
check("字面值：0.98¹⁰ = 0.81707，1 − 0.81707 = 0.18293，0.18293 × 0.2 = 0.036586，1 + 0.81707 = 1.81707，0.036586 ÷ 1.81707 = 0.0201",
      round(0.98**10, 5) == 0.81707 and round(1 - 0.81707, 5) == 0.18293 and round(0.18293 * 0.2, 6) == 0.036586
      and round(1 + 0.81707, 5) == 1.81707 and round(0.036586 / 1.81707, 4) == 0.0201)
bias_pay = -0.127166 * 3.0 * DT
print(f"  课程把权重涨到 3.0 之后，持续低头 0.2 rad 每步扣 {num(bias_pay, 6)} 分")
check("(−0.127166) × 3 × 0.02 = −0.00763", round(bias_pay, 5) == -0.00763)
check("(−0.127166) × 3 = −0.381498；账面爬满到 0.2 以后是 (−0.2) × 3 × 0.02 = −0.012",
      round(-0.127166 * 3, 6) == -0.381498 and round(-0.2 * 3 * DT, 6) == -0.012)
check("摆动最多付 0.0201 × 3 × 0.02 = 0.001206，约是 0.00763 的六分之一",
      round(0.0201 * 3 * DT, 6) == 0.001206 and round(0.00763 / 0.001206) == 6)
check("自测：α 换成 0.1（平均 0.2 秒）时第 1 步是 0.02，是 α = 0.02 时 0.004 的 5 倍",
      round(0.9 * 0 + 0.1 * bias, 6) == 0.02 and round(0.02 / 0.004, 6) == 5)
peak_01 = bias * (1 - 0.9**HALF) / (1 + 0.9**HALF)
ema01, peak01_sim = 0.0, 0.0
for k in range(1000):
    ema01 = 0.9 * ema01 + 0.1 * (bias * (1 if (k // HALF) % 2 == 0 else -1))
    if k > 500:
        peak01_sim = max(peak01_sim, abs(ema01))
print(f"  自测：α = 0.1 时同样的摆动，账面在 ±{num(peak_01, 4)} 之间晃（逐步模拟 ±{num(peak01_sim, 4)}）")
check("自测：α = 0.1 时摆动的账在 ±0.0966 之间晃，差不多是 ±0.0201 的 5 倍",
      round(peak_01, 4) == 0.0966 and math.isclose(peak_01, peak01_sim, rel_tol=1e-3) and round(0.0966 / 0.0201) == 5)

# ---------------------------------------------------------------------------
banner("8b. 画图：figures/ch16_head_bias_ema.png（摆动被平均掉，偏置留下来）")
fig, axes = lesson_figure(2, "1 秒平均只收持续的偏差，收不到来回的摆动", panel_height=4.6)
fig.subplots_adjust(top=0.885, bottom=0.085, hspace=0.62)
n_show = 150
tt = (np.arange(n_show) + 1) * DT
osc = bias * np.where((np.arange(n_show) // HALF) % 2 == 0, 1.0, -1.0)
ema_osc, ema_dc, cur_a, cur_b = [], [], 0.0, 0.0
# 上半张画的是“已经走了一会儿”：先让摆动的 EMA 空跑 600 步（= 30 个来回）进入稳态，
# 否则画面开头那段从 0 起步的暂态会冲出 ±peak 的窄带，和图注说的“始终夹在里面”打架。
WARMUP = 600
for k in range(WARMUP):
    cur_a = (1 - alpha) * cur_a + alpha * (bias * (1 if (k // HALF) % 2 == 0 else -1))
for k in range(n_show):
    cur_a = (1 - alpha) * cur_a + alpha * osc[k]
    cur_b = (1 - alpha) * cur_b + alpha * bias
    ema_osc.append(cur_a)
    ema_dc.append(cur_b)
check(f"上半张图里画的 3 秒，1 秒平均全都落在 ±{num(peak, 4)} 之内（和图注说的一致）",
      max(abs(v) for v in ema_osc) <= peak + 1e-9)

for ax, series, ema_series, title, tail in (
        (axes[0], osc, ema_osc, "① 走路：头随步伐来回摆 ±0.2 rad，0.4 秒一个来回（这本账已经稳住了）",
         f"摆动是正负交替的，一平均就抵消：橙线只在 ±{num(peak, 4)} 之间晃，这一项几乎不扣分。"),
        (axes[1], np.full(n_show, bias), ema_dc, "② 低头：头一直比命令低 0.2 rad", "偏置是一直同号的，平均下来原样留着，这一项照扣不误。")):
    ax.clear()
    data_axes(ax, "时间（秒）", "头部跟踪误差（rad）")
    ax.axhline(0, color=CELL_EDGE, lw=1.2)
    ax.plot(tt, series, color=FAINT, lw=2.6, label="每一步的误差")
    ax.plot(tt, ema_series, color=ORANGE, lw=3.4, label="1 秒平均（EMA）")
    ax.set_ylim(-0.42, 0.30)
    ax.set_xlim(0, tt[-1])
    ax.set_yticks([-0.2, 0, 0.2])
    ax.legend(fontsize=FS_TICK, frameon=False, loc="lower center", ncols=2)
    panel_title(fig, [ax], title)
    panel_note(fig, [ax], tail)
axes[1].plot([tt[49]], [ema_dc[49]], "o", color=ORANGE, ms=12, zorder=7)
axes[1].annotate(f"满 1 秒（50 步）时爬到 {num(ema_dc[49], 6)}", xy=(tt[49], ema_dc[49]), xytext=(1.25, 0.035),
                 fontsize=FS_SMALL, color=ORANGE,
                 arrowprops=dict(arrowstyle="-|>", lw=2.2, color=ORANGE, shrinkA=0, shrinkB=7))
axes[1].plot(tt[:2], ema_dc[:2], "o", color=BLUE, ms=8, zorder=7)
axes[1].text(0.06, -0.075, "前两步只有 0.004、0.00792", fontsize=FS_TICK, color=BLUE)
axes[0].axhspan(-peak, peak, color=ORANGE, alpha=0.12, lw=0)
axes[0].text(2.98, 0.075, f"橙线始终夹在 ±{num(peak, 4)} 之间", fontsize=FS_TICK, color=ORANGE, ha="right")
savefig(fig, "ch16_head_bias_ema")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("9. 满分算术：六项任务奖励的上界加起来是多少")
W_AIR = 3.0  # TWEAK-3: 1.0
check(f"air_time 的权重 {W_AIR} 与 cfg 对得上", R["air_time"].weight == W_AIR)
KERNEL_MAX = {"track_linear_velocity": 1.0, "track_angular_velocity": 1.0, "upright": 1.0,
              "pose": 1.0, "air_time": 2.0, "head_pose_tracking": 1.0}
rows, total = [], 0.0
for name in TASK_TERMS:
    w = W_AIR if name == "air_time" else R[name].weight
    mx = KERNEL_MAX[name]
    per = w * mx * DT
    rows.append([name, w, mx, per, per * STEPS_20S])
    total += per * STEPS_20S
table(["任务项", "权重", "核最大值", "每步最多", "跑满 20 秒最多"], rows, floatfmt=".6g")
print(f"  六项的代数上界相加：{num(total, 1)} 分 / 回合（同一条轨迹并不能同时拿满）")
check("一回合 20 秒 = 1000 步：cfg 的 episode_length_s 是 20.0，一步 0.02 秒",
      cfg.episode_length_s == 20.0 and round(cfg.episode_length_s / DT) == STEPS_20S)
check("air_time 的核最大值是 2：两只脚各 1 分", KERNEL_MAX["air_time"] == 2.0)
check("每步上界 2+2+2+1+6+2 = 15 分，乘 0.02 是 0.3 分", round(total / STEPS_20S, 6) == 0.3)
check("一回合 20 秒的上界是 300 分", round(total, 6) == 300.0)
check("air_time 一项就占 120 / 300 = 40%", round(W_AIR * 2 * DT * STEPS_20S / total, 6) == 0.4)
check("跑满 20 秒且每步满分时，Episode_Reward 的读数 = 权重 × 核最大值（第 14 章 14.5 节）",
      round(2 * 1 * DT * STEPS_20S / 20, 6) == 2.0 and round(W_AIR * 2 * DT * STEPS_20S / 20, 6) == 6.0)
alt = sum((1.0 if n == "air_time" else R[n].weight) * KERNEL_MAX[n] * DT * STEPS_20S for n in TASK_TERMS)
check("自测：air_time 的权重改成 1，六项合计 220 分，迈步只占 18%",
      round(alt, 6) == 220.0 and round(1.0 * 2 * DT * STEPS_20S / alt, 2) == 0.18)
max_per_step = [round(r[3], 6) for r in rows]
print(f"  一项一项：每步最多 {' + '.join(num(p, 2) for p in max_per_step)} = {num(sum(max_per_step), 2)} 分；"
      f"权重 × 核最大值加起来 {num(sum(r[1] * r[2] for r in rows), 1)}，再乘 0.02 × 1000 = {num(DT * STEPS_20S, 1)}")
check("每步最多依次是 0.04、0.04、0.04、0.02、0.12、0.04，加起来 0.3；0.3 × 1000 = 300",
      max_per_step == [0.04, 0.04, 0.04, 0.02, 0.12, 0.04] and round(sum(max_per_step), 6) == 0.3 and round(0.3 * 1000, 6) == 300)
check("另一条路：权重 × 核最大值加起来 15，0.02 × 1000 = 20（正好一回合 20 秒），15 × 20 = 300",
      round(sum(r[1] * r[2] for r in rows), 6) == 15 and round(DT * STEPS_20S, 6) == 20
      and round(DT * STEPS_20S, 6) == cfg.episode_length_s and 15 * 20 == 300)
check("六项的回合上界 40 + 40 + 40 + 20 + 120 + 40 = 300；读数 40 ÷ 20 = 2、120 ÷ 20 = 6",
      [round(r[4], 6) for r in rows] == [40, 40, 40, 20, 120, 40] and 40 / 20 == 2 and 120 / 20 == 6)
check("速度跟踪占 40 ÷ 300 ≈ 13%；同样涨 0.1，pose（读数最多 1）是满分的 1/10，air_time（最多 6）只是 1/60",
      round(40 / total, 2) == 0.13 and round(0.1 / 1, 6) == 0.1 and round(0.1 / (W_AIR * 2), 6) == round(1 / 60, 6))

# ---------------------------------------------------------------------------
banner("9b. 画图：figures/ch16_score_budget.png（300 分是怎么分的）")
fig, axes = lesson_figure(1, "满分算术：六项任务奖励的上界加起来，一回合 300 分", panel_height=5.6, width=9.6)
ax = axes[0]
ax.clear()
labels = ["迈步 air_time", "速度跟踪（前后左右）", "转向跟踪", "站直", "头部跟踪", "腿别乱动 pose"]
order = ["air_time", "track_linear_velocity", "track_angular_velocity", "upright", "head_pose_tracking", "pose"]
vals = [(W_AIR if n == "air_time" else R[n].weight) * KERNEL_MAX[n] * DT * STEPS_20S for n in order]
data_axes(ax, "跑满 20 秒、每步都满分时，这一项最多能拿多少分", "")
ax.barh(range(len(vals))[::-1], vals, color=[ORANGE] + [BLUE] * 5, height=0.62)
ax.set_yticks(range(len(vals))[::-1])
ax.set_yticklabels(labels, fontsize=FS_SMALL)
ax.set_xlim(0, 136)
ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
for k, v in enumerate(vals):
    ax.text(v + 3, len(vals) - 1 - k, f"{v:g} 分", fontsize=FS_TICK, color=INK, va="center")
ax.grid(axis="y", visible=False)
panel_title(fig, [ax], f"① 条的长度和分数成比例，六条加起来 {total:g} 分")
panel_note(fig, [ax],
           "迈步一项就占 120 分，因为两只脚各能拿 1 分，再乘权重 3。\n"
           "这是代数上界：站着不动时迈步得 0 分，走起来时站直和 pose 都拿不满——同一条轨迹拿不全。",
           pad=0.022)
savefig(fig, "ch16_score_budget")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("10. 只付进步的钱：望远镜消去，和裁成 0 之后的刷分")
phi = {60: math.cos(math.radians(60)), 30: math.cos(math.radians(30)), 0: 1.0}
print(f"  进度分 Φ = cos(倾角)：60° → {num(phi[60], 3)}，30° → {num(phi[30], 3)}，0° → {num(phi[0], 3)}")
up1 = round(phi[30], 3) - round(phi[60], 3)
up2 = round(phi[0], 3) - round(phi[30], 3)
check("Φ(60°) = 0.5，Φ(30°) = 0.866，Φ(0°) = 1",
      round(phi[60], 3) == 0.5 and round(phi[30], 3) == 0.866 and phi[0] == 1.0)
check("一路站起来：0.366 + 0.134 = 0.5 = 终点 Φ − 起点 Φ（中间项相消）",
      round(up1, 3) == 0.366 and round(up2, 3) == 0.134 and round(up1 + up2, 3) == 0.5)
check("来回摆 60° ↔ 30°，简单差分一涨一落加起来是 0", round(up1 + (-up1), 6) == 0.0)
check("去掉括号：(0.866 − 0.5) + (1 − 0.866) = −0.5 + 0.866 − 0.866 + 1 = −0.5 + 1 = 0.5",
      round(-0.5 + 0.866 - 0.866 + 1, 6) == 0.5 and round(-0.5 + 1, 6) == 0.5)
# 绕远路：60° → 30° → 60° → 30° → 0°，一行一行摆开，看错开一行的 Φ 一正一负地消掉
path = [60, 30, 60, 30, 0]
after = [round(phi[b], 3) for b in path[1:]]
before = [round(phi[a], 3) for a in path[:-1]]
f_steps = [round(x - y, 3) for x, y in zip(after, before)]
table(["这一步", "+ 这一步之后的 Φ", "− 这一步之前的 Φ", "这一笔 F"],
      [[f"{a}° → {b}°", x, -y, f] for (a, b), x, y, f in zip(zip(path[:-1], path[1:]), after, before, f_steps)]
      + [["竖着加", round(sum(after), 3), -round(sum(before), 3), round(sum(f_steps), 3)]], floatfmt=".4g")
check("绕远路四笔：+0.366、−0.366、+0.366、+0.134；三列竖着加 3.232、−2.732、0.5——还是终点 1 减起点 0.5",
      f_steps == [0.366, -0.366, 0.366, 0.134] and round(sum(after), 3) == 3.232 and round(sum(before), 3) == 2.732
      and round(sum(f_steps), 3) == 0.5 and round(3.232 - 2.732, 3) == 0.5)
check("每一行'之后的 Φ'正是下一行'之前的 Φ'（错开一行，一正一负）", after[:-1] == before[1:])
check("站着不动：0.866 − 0.866 = 0，在 30° 站 50 步还是 0；要是每步直接付 Φ，50 × 0.866 = 43.3",
      round(phi[30] - phi[30], 6) == 0.0 and round(50 * 0.866, 6) == 43.3)
farm = 10 * up1
print(f"  把下降的差分裁成 0，只留上涨的：摆 10 个来回白拿 {num(farm, 3)} 分，比站起来的 0.5 分还多")
check("裁成 0 之后，10 个来回能刷 3.66 分", round(farm, 3) == 3.66)
check("3.66 ÷ 0.5 = 7.32：是老实站起来的 7 倍还多", round(3.66 / 0.5, 2) == 7.32)
cos70 = math.cos(math.radians(70))
check("重置那一跳：摔倒判据 70° 时 Φ = cos 70° = 0.342，新回合站直 Φ = 1，跳了 1 − 0.342 = 0.658",
      round(cos70, 3) == 0.342 and round(1 - 0.342, 3) == 0.658)
up_full = phi[0] - round(phi[60], 3)
check("自测：60°→0°→60°→0°，不裁的差分合计 0.5；把下降裁成 0 就变成 1.0",
      round(up_full + (-up_full) + up_full, 3) == 0.5 and round(up_full + 0 + up_full, 3) == 1.0)

GAMMA = 0.99
f1 = round(GAMMA * round(phi[30], 3) - round(phi[60], 3), 5)
f2 = round(GAMMA * phi[0] - round(phi[30], 3), 5)
total_disc = round(f1 + GAMMA * f2, 4)
closed_disc = round(GAMMA**2 * phi[0] - round(phi[60], 3), 4)
table(["折扣版 F = γΦ(s′) − Φ(s)", "这一步的值", "乘上 γᵗ 之后"],
      [["第 1 步：0.99 × 0.866 − 0.5", f1, f1],
       ["第 2 步：0.99 × 1 − 0.866", f2, round(GAMMA * f2, 5)]], floatfmt=".6g")
print(f"  两步加起来 {num(total_disc, 4)}，正好等于 γ²Φ(终) − Φ(起) = {num(closed_disc, 4)}")
check("0.99 × 0.866 − 0.5 = 0.35734", f1 == 0.35734)
check("0.99 × 1 − 0.866 = 0.124", f2 == 0.124)
check("0.35734 + 0.99 × 0.124 = 0.4801 = 0.9801 − 0.5（带折扣也相消）",
      total_disc == 0.4801 and closed_disc == 0.4801)
check("字面值：0.99 × 0.124 = 0.12276，0.35734 + 0.12276 = 0.4801",
      round(0.99 * 0.124, 5) == 0.12276 and round(0.35734 + 0.12276, 4) == 0.4801)
check("不改写法、只给第二笔打折：0.99 × 0.866 = 0.85734，和 0.866 差 0.00866，抵不干净（合计 0.49866）",
      round(0.99 * 0.866, 5) == 0.85734 and round(0.866 - 0.85734, 5) == 0.00866
      and round((0.866 - 0.5) + 0.99 * (1 - 0.866), 5) == 0.49866)
check("改写法后摆开：0.99 × 0.866 − 0.5 + 0.9801 × 1 − 0.99 × 0.866 = 0.4801（0.99 × 0.866 一正一负）",
      round(0.99 * 0.866 - 0.5 + 0.9801 * 1 - 0.99 * 0.866, 4) == 0.4801 and round(0.99 * 0.99, 4) == 0.9801)
check("带折扣时原地不动：0.99 × 0.866 − 0.866 = −0.00866，略小于 0", round(GAMMA * 0.866 - 0.866, 5) == -0.00866)
# 终止时结清：终止之后的 Φ 记 0（第 9 章 9.6 节：终止之后一律按 0 算）
settle_a = (1.0 - 0.5) + (0.0 - 1.0)          # 从 60° 站直，再一路摔倒、终止
settle_b = 0.0 - 0.5                          # 从 60° 直接摔倒、终止
check("结清以后：先站直再摔倒是 0.5 + (0 − 1) = −0.5，直接摔倒是 0 − 0.5 = −0.5——总额都是 −Φ(起点)",
      round(settle_a, 6) == -0.5 and round(settle_b, 6) == -0.5)
mdp_text = Path(microduck_mdp.__file__).read_text(encoding="utf-8")
check("项目的 upright_progress（mdp.py）在重置时把'上一步的 Φ'换成新起点，再算差分",
      lines_in_order(mdp_text, ["def upright_progress(", "fresh = env.episode_length_buf <= 1",
                                "env._upright_potential_prev[fresh] = cos_tilt[fresh]",
                                "delta = cos_tilt - env._upright_potential_prev"]))

# ---------------------------------------------------------------------------
banner("11. 映射到项目：正文引用的常数和源码行还在不在")
REPO = Path(__file__).resolve().parents[3]
if not (REPO / "AGENTS.md").is_file():   # 被复制到别处运行（比如按"改一改"改一份副本）：从已安装的项目包找回仓库根目录
    _spec = importlib.util.find_spec("mjlab_microduck")
    if _spec and _spec.origin:
        REPO = Path(_spec.origin).resolve().parents[2]
CFG_LINES = [
    'cfg.rewards["upright"].weight = 2.0',
    'cfg.rewards["upright"].params["std"] = math.sqrt(0.05)',
    'cfg.rewards["air_time"].weight = 3.0',
    'cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.1)',
    'cfg.rewards["action_rate_l2"].weight = -0.1',
    "func=microduck_mdp.head_pose_tracking,",
    "weight=2.0,",
    "func=microduck_mdp.head_pose_bias_penalty,",
    'params={"command_name": "head_pose", "tau_s": 1.0},',
]
MDP_LINES = [
    "def head_pose_bias_penalty(",
    "fresh = env.episode_length_buf <= 1",
    "env._head_bias_ema[fresh] = 0.0",
    "alpha = min(1.0, float(env.step_dt) / max(tau_s, 1e-6))",
    "env._head_bias_ema = (1.0 - alpha) * env._head_bias_ema + alpha * err",
    "out = -env._head_bias_ema.abs().mean(dim=-1)",
]
HOMES = {}
for _n, _t in R.items():
    HOMES.setdefault(getattr(_t.func, "__module__", "?"), []).append(_n)
table(["这一项的函数住在哪个文件", "有几项"],
      [[m.replace(".", "/") + ".py", len(v)] for m, v in sorted(HOMES.items(), key=lambda kv: -len(kv[1]))])
check("16 项分在三个文件里：mjlab 走路模板 11 项、mjlab 通用 2 项、这个仓库自己写的 3 项",
      [len(HOMES.get(m, [])) for m in ("mjlab.tasks.velocity.mdp.rewards", "mjlab.envs.mdp.rewards",
                                       "mjlab_microduck.tasks.mdp")] == [11, 2, 3])

cfg_src = (REPO / "src" / "mjlab_microduck" / "tasks" / "microduck_velocity_env_cfg.py").read_text(encoding="utf-8")
mdp_src = (REPO / "src" / "mjlab_microduck" / "tasks" / "mdp.py").read_text(encoding="utf-8")
check("microduck_velocity_env_cfg.py 里的权重和 std 还是正文引用的那几行", lines_in_order(cfg_src, CFG_LINES))
check("mdp.py 的 head_pose_bias_penalty 还是正文引用的五行（回合开头清零两行 + 记账出分三行）",
      lines_in_order(mdp_src, MDP_LINES))

try:
    import mjlab

    pkg = Path(mjlab.__file__).parent
    rm_src = (pkg / "managers" / "reward_manager.py").read_text(encoding="utf-8")
    rw_src = (pkg / "tasks" / "velocity" / "mdp" / "rewards.py").read_text(encoding="utf-8")
    base_src = (pkg / "envs" / "mdp" / "rewards.py").read_text(encoding="utf-8")
    check("reward_manager.py 的 compute() 还是「函数值 × 权重 × dt」",
          lines_in_order(rm_src, ["if term_cfg.weight == 0.0:", "continue",
                                  "value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * scale"]))
    check("velocity 的核还在 mjlab/tasks/velocity/mdp/rewards.py 里",
          lines_in_order(rw_src, ["def track_linear_velocity(", "return torch.exp(-lin_vel_error / std**2)",
                                  "def feet_air_time(", "def feet_clearance(", "def feet_slip("]))
    check("action_rate_l2 和 joint_pos_limits 还在 mjlab/envs/mdp/rewards.py 里",
          lines_in_order(base_src, ["def action_rate_l2(", "def joint_pos_limits("]))
    # 16.3 节迈步那一项的计时规矩：每一步看"这一次已经离地多久"，命令 = 前后左右的大小 + 转向的大小
    cs_src = (pkg / "sensor" / "contact_sensor.py").read_text(encoding="utf-8")
    ent_src = (pkg / "entity" / "entity.py").read_text(encoding="utf-8")
    check("feet_air_time 每一步看 current_air_time 在不在窗口里；命令速度 = 前后左右的大小 + 转向的大小",
          lines_in_order(rw_src, ["def feet_air_time(", "current_air_time = sensor_data.current_air_time",
                                  "in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)",
                                  "total_command = linear_norm + angular_norm"]))
    check("离地计时：没着地就一直往上加，一着地就归零（contact_sensor.py）",
          lines_in_order(cs_src, ["state.current_air_time[:] = torch.where(", "~is_contact,",
                                  "state.current_air_time + elapsed_time,", "torch.zeros_like(state.current_air_time),"]))
    check("软限位 = 量程中点 ± 半个量程 × 0.9（entity.py）",
          lines_in_order(ent_src, ["joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor,",
                                   "joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor,"]))
except (ImportError, FileNotFoundError):
    print("  找不到 mjlab 包，跳过这几条源码核对（装好依赖后重跑即可）。")

# ---------------------------------------------------------------------------
if "--env" in sys.argv:
    banner("12. 🖥️ 建 4 个环境跑 50 步，看每一项的符号（AGENTS.md 的不变量：惩罚项 ≤ 0）")
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
        env.step(torch.randn(4, 14, device="cuda:0") * 0.05)
        rm = env.reward_manager
        for i, name in enumerate(rm.active_terms):
            sums[name] = sums.get(name, 0.0) + rm._step_reward[:, i].mean().item()
    table(["奖励项", "50 步累计（× 权重，未乘 dt）", "权重"],
          [[n, v, R[n].weight] for n, v in sums.items()], floatfmt=".4g")
    check("负权重项全部 ≤ 0", not [n for n, v in sums.items() if R[n].weight < 0 and v > 1e-9])
    check("自带负号的 head_pose_bias ≤ 0", not [n for n, v in sums.items() if n == "head_pose_bias" and v > 1e-9])
    env.close()
else:
    banner("12. 🖥️ 可选：建 4 个环境跑 50 步验证符号")
    print("  这一节要 GPU。加 --env 参数才会跑；第 1–11 节的结论不依赖它。")

done()
