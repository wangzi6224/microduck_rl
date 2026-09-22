"""第 15 章 CPU 伴生实验：不建环境，只读 env cfg 和机器人模型，逐块核对 61 维观测，并画本章全部讲解图。

运行：uv run python docs/learn-zh/labs/ch15_obs_layout.py
纯 CPU：构造 velocity 任务的 cfg（做法同第 16、18 章的实验），用 MuJoCo 读机器人模型里的关节名和脚的名字，
torch 只在 CPU 上算几个小梯度。真建环境、打印一帧真实观测的是另一个实验 ch15_print_obs.py（要 GPU）；
本实验引用的"一帧"就是它的一次真实输出，原样抄在 FRAME 里。
小节编号与正文一一对应：实验第 K 节 = 正文 15.K 节（第 10 节对应「映射到项目」）。
正文“改一改”要改的三行都带 `# TWEAK-k:` 标记（第 1、4、7 节各一处）。
"""

import math
import re
from pathlib import Path

import numpy as np
import torch

from _common import banner, check, done, lines_in_order, savefig, table
from _draw import (BLUE, CELL, CELL_EDGE, CELL_HOT, FAINT, FS_NOTE, FS_SMALL, FS_STEP, FS_TICK, FS_TITLE, GREEN, INK,
                   MUTED, ORANGE, WHITE_BOX, arrow, cell, data_axes, hand, lesson_cells, lesson_figure, lesson_panel,
                   note, plt)

# 本章讲解图里 8 块（和 critic 多出的 5 块）的配色：自己测的偏蓝、命令偏绿、特权观测偏橙；相邻两块深浅交替
FILL = {"base_ang_vel": "#cfe0f1", "projected_gravity": "#9fc0e2", "joint_pos": "#cfe0f1", "joint_vel": "#9fc0e2",
        "actions": "#cfe0f1", "command": "#cfe9dd", "head_command": "#9fd0b8", "body_command": "#cfe9dd",
        "base_lin_vel": "#f6c79c", "foot_height": "#fbe3cc", "foot_air_time": "#f6c79c", "foot_contact": "#fbe3cc",
        "foot_contact_forces": "#f6c79c"}
ZH = {"base_ang_vel": "陀螺仪", "projected_gravity": "重力方向", "joint_pos": "关节角", "joint_vel": "关节速度",
      "actions": "上一步动作", "command": "速度", "head_command": "头", "body_command": "身体",
      "base_lin_vel": "真实线速度", "foot_height": "脚高", "foot_air_time": "腾空多久", "foot_contact": "着地没",
      "foot_contact_forces": "接触力"}
DEG = 180 / math.pi                                       # 1 rad 是多少度（附录 D.2）

# ---------------------------------------------------------------------------
banner("1. 读表的三把钥匙：[a, b) 数项数、弧度换角度、一步 = 0.02 秒")
import mujoco  # noqa: E402

import mjlab_microduck  # noqa: E402
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg  # noqa: E402
from mjlab_microduck.robot.microduck_constants import HOME_FRAME, get_walk_spec  # noqa: E402  （先 import 任务再 import 常数，避免循环 import）

cfg = make_microduck_velocity_env_cfg()
model = get_walk_spec().compile()                        # 走路用的机器人模型（和 velocity 任务同一个 XML）
hinges = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
feet_sensor = next(s for s in cfg.scene.sensors if s.name == "feet_ground_contact")
height_sensor = next(s for s in cfg.scene.sensors if s.name == "foot_height_scan")
feet = [model.geom(i).name for i in range(model.ngeom) if re.fullmatch(feet_sensor.primary.pattern, model.geom(i).name)]
TWIST_FIELDS = ("lin_vel_x", "lin_vel_y", "ang_vel_z")   # 速度命令进观测的三项：前后、左右、转向（heading 不进观测）
VEC3 = {"base_ang_vel_imu_misaligned", "projected_gravity_imu_misaligned", "builtin_sensor", "projected_gravity",
        "base_lin_vel"}                                   # 三维空间里的一根箭头：x、y、z 三项


def term_dim(term) -> int:
    """不建环境，按"这一块算的是什么"推出它有几项。认不出来的块记 0，后面"加起来 = 61"那条会报 ✗。"""
    f = term.func.__name__
    if f in VEC3:
        return 3
    if f in ("joint_pos_rel", "joint_vel_rel"):          # 名字不以 passive_ 开头的关节（15.6 节）
        return sum(bool(re.fullmatch(term.params["asset_cfg"].joint_names[0], n)) for n in hinges)
    if f == "last_action":
        return model.nu                                  # 每个舵机一项
    if f == "generated_commands":
        name = term.params["command_name"]
        return len(TWIST_FIELDS) if name == "twist" else len(cfg.commands[name].ranges)
    if f == "foot_height_safe":
        return len(height_sensor.frame)                  # 每只脚一根测距射线
    if f in ("foot_air_time_safe", "foot_contact"):
        return len(feet)
    if f == "foot_contact_forces_safe":
        return 3 * len(feet)                             # 每只脚一个三维的力
    return 0


def layout(group: str, drop=None) -> list[tuple[str, int, int, int]]:
    rows, start = [], 0
    for name, term in cfg.observations[group].terms.items():
        if name == drop:
            continue
        d = term_dim(term)
        rows.append((name, d, start, start + d))         # 这一块占 [start, start + d)
        start += d
    return rows


def owner(rows, i: int) -> str:
    return next((n for n, _, a, b in rows if a <= i < b), "（越界：没有这一项）")


# GPU 实验第 2、4 节真建环境读出来的表（名字, 项数），原样抄在这里对照
GPU_ACTOR = [("base_ang_vel", 3), ("projected_gravity", 3), ("joint_pos", 14), ("joint_vel", 14), ("actions", 14),
             ("command", 3), ("head_command", 4), ("body_command", 6)]
GPU_CRITIC = [("base_lin_vel", 3), ("base_ang_vel", 3), ("projected_gravity", 3), ("joint_pos", 14), ("joint_vel", 14),
              ("actions", 14), ("command", 3), ("foot_height", 2), ("foot_air_time", 2), ("foot_contact", 2),
              ("foot_contact_forces", 6), ("head_command", 4), ("body_command", 6)]

DROP_BLOCK = None  # TWEAK-1: "joint_vel"
full = layout("actor")                                   # 训练用的布局
actor = layout("actor", DROP_BLOCK)                      # “改一改”第 1 条：删掉一块之后的布局
table(["观测块", "项数", "区间 [a, b)", "最后一项的编号 b − 1"], [[n, d, f"[{a}, {b})", b - 1] for n, d, a, b in actor])
bounds = [0] + [b for _, _, _, b in actor]
total = bounds[-1]
print("边界（每块的起点，最后一个是总项数）：", ", ".join(map(str, bounds)))
check("从 cfg 和机器人模型推出的 8 块（名字、顺序、项数）= GPU 实验第 2 节真建环境读出的表",
      [(n, d) for n, d, _, _ in actor] == GPU_ACTOR)
check("8 块首尾相接：每块的起点 = 上一块的终点", all(actor[k][2] == actor[k - 1][3] for k in range(1, len(actor))))
check(f"加起来 3 + 3 + 14 + 14 + 14 + 3 + 4 + 6 = 61（当前布局是 {total} 项）",
      total == 61 and 3 + 3 + 14 + 14 + 14 + 3 + 4 + 6 == 61)
check("边界依次是 0, 3, 6, 20, 34, 48, 51, 55, 61", bounds == [0, 3, 6, 20, 34, 48, 51, 55, 61])
check("[6, 20) 有 20 − 6 = 14 项，从编号 6 数到编号 19", dict((n, (a, b)) for n, _, a, b in actor).get("joint_pos") == (6, 20)
      and 20 - 6 == 14 and len(range(6, 20)) == 14 and max(range(6, 20)) == 19)
check("前 5 块 48 项（0 到 47），后 3 块 13 项（48 到 60）", bounds[5] == 48 and total - 48 == 13)
check("编号 0 = 第 1 项 = 身体绕前后轴（x 轴）转多快；编号 6 = 第 7 项 = 左髋偏航角（第 2 章 2.1 节的承诺）",
      owner(full, 0) == "base_ang_vel" and owner(full, 6) == "joint_pos" and hinges[6 - 6] == "left_hip_yaw")
if DROP_BLOCK is None:
    print("DROP_BLOCK = None：布局没动。“改一改”第 1 条会删掉一块，看看后面的编号都串到哪里去。")
else:
    print(f"删掉 {DROP_BLOCK} 之后，同一个编号在训练时和现在各是哪一块：")
    table(["编号", "训练时（61 项的布局）", "删块之后"], [[i, owner(full, i), owner(actor, i)] for i in (0, 6, 20, 34, 48, 55, 60)])

print(f"\n1 rad = 180 / π = {DEG:.4f}°；0.1 rad = {0.1 * DEG:.2f}°；π rad = {math.pi * DEG:g}°")
check("1 rad ≈ 57.3°，0.1 rad ≈ 5.73°，π rad = 180°", round(DEG, 1) == 57.3 and round(0.1 * DEG, 2) == 5.73
      and math.isclose(math.pi * DEG, 180))
print(f"0.5 rad/s 转 0.02 s：0.5 × 0.02 = {0.5 * 0.02:g} rad")
check("0.5 rad/s × 0.02 s = 0.01 rad ≈ 0.57°（rad/s 乘上时间才是转角）", math.isclose(0.5 * 0.02, 0.01)
      and round(0.01 * DEG, 2) == 0.57)
check("⚠️ 的对照：关节角 0.5 rad ≈ 29°；自测：0.3 rad ≈ 0.3 × 57.3 = 17.19°", round(0.5 * DEG) == 29
      and round(0.3 * 57.3, 2) == 17.19 and [51, 51 + 4, 51 + 4 - 1] == [51, 55, 54])
dt = cfg.decimation * cfg.sim.mujoco.timestep
print(f"一步 = decimation × 物理步 = {cfg.decimation} × {cfg.sim.mujoco.timestep} s = {dt:g} s，每秒 {1 / dt:g} 步")
check("一步 = 4 × 0.005 s = 0.02 s（50 Hz）；延迟 1 步 = 20 ms", cfg.decimation == 4 and cfg.sim.mujoco.timestep == 0.005
      and math.isclose(dt, 0.02) and round(1 / dt) == 50 and round(dt * 1000) == 20)


def block_strip(ax, rows, x0, y0, unit, h, labels=True):
    """按项数成比例画一条：每项一格（细白线隔开，数得清），块与块之间一条深色分界。"""
    for name, d, a, b in rows:
        for i in range(a, b):
            ax.add_patch(plt.Rectangle((x0 + i * unit, y0), unit, h, facecolor=FILL[name], edgecolor="white", lw=0.7))
        ax.add_patch(plt.Rectangle((x0 + a * unit, y0), d * unit, h, facecolor="none", edgecolor=INK, lw=1.3))
        if labels:
            ax.text(x0 + (a + b) / 2 * unit, y0 + h / 2, str(d), ha="center", va="center", fontsize=15, color=INK)


# ---------------------------------------------------------------------------
banner("1b. 画图：figures/ch15_overview.png（本章地图：61 = 8 块首尾相接）")
if DROP_BLOCK is not None:
    print("  删过块了：这张地图是照着 61 项的布局画的，跳过。看上面那张对照表就行。")
else:
    fig, axes = lesson_figure(3, "61 项观测 = 8 块首尾相接：前 48 项自己测，后 13 项听命令", panel_height=3.3)
    ax = axes[0]
    lesson_panel(ax, "① 8 块一块接一块（格子宽度按项数画，一项一格）")
    unit, x0 = 9.4 / 61, 0.3
    block_strip(ax, full, x0, 1.35, unit, 0.62)
    lift = {"projected_gravity": 0.5, "head_command": 0.5}          # 挨得太近的两块，名字抬高一层
    for name, d, a, b in full:
        xm = x0 + (a + b) / 2 * unit
        y = 2.2 + lift.get(name, 0)
        if lift.get(name):
            ax.plot([xm, xm], [2.0, y - 0.15], color=MUTED, lw=1)
        ax.text(xm, y, ZH[name], ha="center", va="center", fontsize=15, color=INK)
    for k in bounds:
        ax.text(x0 + k * unit, 1.08, str(k), ha="center", va="center", fontsize=13, color=MUTED)
    for (a, b, text, color) in ((0, 48, "自己身上测到的：48 项", BLUE), (48, 61, "人给的命令：13 项", GREEN)):
        xa, xb = x0 + a * unit + 0.03, x0 + b * unit - 0.03
        ax.plot([xa, xa, xb, xb], [0.78, 0.62, 0.62, 0.78], color=color, lw=2)
        ax.text((xa + xb) / 2, 0.28, text, ha="center", va="center", fontsize=FS_SMALL, color=color)
    ax = axes[1]
    lesson_panel(ax, "② critic 那张清单再多 15 项：只有仿真器知道的")
    cell(ax, 0.3, 1.5, "actor 的 61 项", width=2.6, height=0.9, fontsize=FS_SMALL)
    ax.text(3.2, 1.95, "+", fontsize=FS_STEP, ha="center", va="center", color=INK)
    cell(ax, 3.6, 1.5, "15 项仿真才知道的", width=3.3, height=0.9, facecolor="#fbe3cc", edgecolor=ORANGE, fontsize=FS_SMALL)
    ax.text(7.3, 1.95, "=", fontsize=FS_STEP, ha="center", va="center", color=INK)
    cell(ax, 7.7, 1.5, "critic 的 76 项", width=2.2, height=0.9, fontsize=FS_SMALL)
    note(ax, 0.3, 0.75, "critic 只在训练时用，不上真机，所以可以多看（15.8 节）。")
    ax = axes[2]
    lesson_panel(ax, "③ 每一项都走同一条生产线，最后拼进自己的位置")
    steps = ["算出来", "加噪声", "裁剪", "缩放", "延迟", "拼接"]
    w, gap = 1.3, 0.36
    for k, s in enumerate(steps):
        x = 0.15 + k * (w + gap)
        cell(ax, x, 1.55, s, width=w, height=0.85, facecolor=CELL if s not in ("裁剪", "缩放") else "white",
             color=INK if s not in ("裁剪", "缩放") else MUTED, fontsize=FS_SMALL)
        if k:
            arrow(ax, (x - gap + 0.03, 1.975), (x - 0.03, 1.975), MUTED, lw=1.6)
    note(ax, 0.3, 0.75, "灰色两道工序这个任务没用上；噪声和延迟只加给 actor（15.4、15.9 节）。")
    savefig(fig, "ch15_overview")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("1c. 画图：figures/ch15_half_open.png（[a, b)：项数 = b − a，下一块从 b 接着数）")
if DROP_BLOCK is not None:
    print("  删过块了：这张图照着 61 项的布局画，跳过。")
else:
    fig, axes = lesson_figure(3, "[a, b)：从编号 a 数到 b 之前，项数 = b − a", panel_height=3.4)
    ax = axes[0]
    lesson_panel(ax, "① [0, 3) 和 [3, 6)：编号 3 归后一块，不归前一块")
    for i in range(9):
        fc = FILL["base_ang_vel"] if i < 3 else FILL["projected_gravity"] if i < 6 else "#eef1f4"
        cell(ax, 0.5 + i * 1.0, 1.3, str(i), width=1.0, height=0.8, facecolor=fc)
    for (a, b, text, name, color) in ((0, 3, "[0, 3)：3 − 0 = 3 项", "陀螺仪", BLUE), (3, 6, "[3, 6)：6 − 3 = 3 项", "重力方向", GREEN)):
        xa, xb = 0.5 + a + 0.05, 0.5 + b - 0.05
        ax.plot([xa, xa, xb, xb], [2.2, 2.35, 2.35, 2.2], color=color, lw=2)
        ax.text((xa + xb) / 2, 2.72, text, fontsize=FS_SMALL, color=color, ha="center", va="center")
        ax.text((xa + xb) / 2, 0.95, name, fontsize=16, color=color, ha="center", va="center")
    ax.text(8.0, 0.95, "6 往后是下一块", fontsize=16, color=MUTED, ha="center", va="center")
    note(ax, 0.5, 0.3, "左端 [ 含，右端 ) 不含：编号 3 只算一次，算在后一块。")
    ax = axes[1]
    lesson_panel(ax, "② 长的一块也一样：[6, 20) 有 20 − 6 = 14 项")
    cw = 0.62
    for k, i in enumerate(range(6, 20)):
        hot = i in (6, 19)
        cell(ax, 0.6 + k * cw, 1.6, str(i), width=cw, height=0.72, facecolor=CELL_HOT if hot else FILL["joint_pos"],
             edgecolor=ORANGE if hot else CELL_EDGE, fontsize=15)
    hand(ax, 0.6, 1.05, "编号 6 = 第 7 项 = 左髋偏航角", color=ORANGE, fontsize=FS_SMALL)
    hand(ax, 5.55, 1.05, "最后一项：20 − 1 = 19", color=ORANGE, fontsize=FS_SMALL)
    note(ax, 0.6, 0.4, "编号从 0 数，所以编号 6 是人话里的“第 7 项”。")
    ax = axes[2]
    lesson_panel(ax, "③ 每块的起点 = 前面各块项数加起来")
    step_x = 9.2 / 8
    for k, v in enumerate(bounds):
        x = 0.4 + k * step_x
        ax.text(x, 1.7, str(v), fontsize=FS_STEP, ha="center", va="center", color=INK, fontweight="bold")
        if k:
            arrow(ax, (x - step_x + 0.28, 1.7), (x - 0.28, 1.7), MUTED, lw=1.6)
            ax.text(x - step_x / 2, 2.25, f"+{bounds[k] - bounds[k - 1]}", fontsize=15, ha="center", color=BLUE)
    hand(ax, 0.4, 0.95, "3 + 3 + 14 + 14 + 14 + 3 + 4 + 6 = 61", fontsize=FS_SMALL)
    note(ax, 0.4, 0.35, "最后一个边界 61 就是总项数。")
    savefig(fig, "ch15_half_open")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("2. 身体的姿态：陀螺仪 3 项 + 重力方向 3 项（GPU 实验第 3 节的那一帧）")
# GPU 实验第 3 节：走 30 步随机小动作后，第 0 号机器人的一帧（原样抄下，精度 3 位小数）
FRAME = {
    "base_ang_vel": [-0.032, 0.677, -0.163],
    "projected_gravity": [0.243, 0.012, -0.971],
    "joint_pos": [0.004, 0.012, -0.043, 0.162, -0.086, -0.026, 0.035, -0.02, 0.022, -0.004, -0.001, 0.045, -0.123, 0.075],
    "joint_vel": [-0.171, 0.101, -0.01, 0.069, -0.714, -0.351, 0.084, -0.217, -0.014, -0.273, -0.267, 0.667, 0.152, -0.01],
    "actions": [0.108, 0.067, -0.022, -0.031, -0.007, -0.007, 0.094, -0.014, 0.007, -0.006, -0.047, -0.04, -0.037, -0.005],
    "command": [0.231, 0.267, 0.613],
    "head_command": [0.046, -0.042, -0.038, -0.005],
    "body_command": [-0.004, 0.005, -0.0, -0.029, -0.039, 0.0],
}
check("这一帧每块的项数和布局一致，一共 61 个数", all(len(FRAME[n]) == d for n, d, _, _ in full)
      and sum(map(len, FRAME.values())) == 61)
gx, gy, gz = FRAME["projected_gravity"]
glen = math.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
tilt = math.degrees(math.acos(-gz))
print(f"投影重力 ({gx}, {gy}, {gz})：长度 √(0.243² + 0.012² + 0.971²) = {glen:.3f}；倾角 arccos(0.971) = {tilt:.1f}°")
check("投影重力的长度 = 1.001（带着 ±0.01 的噪声，不再恰好是 1）", round(glen, 3) == 1.001
      and round(math.sqrt(0.243 ** 2 + 0.012 ** 2 + 0.971 ** 2), 3) == 1.001)
check("倾角 arccos(0.971) = 13.8°，约 14°；第 1 项为正、第 2 项几乎为 0：主要往前倾", round(tilt, 1) == 13.8
      and gx > 0 and abs(gy) < 0.02)
wy = FRAME["base_ang_vel"][1]
print(f"陀螺仪第 2 项 {wy} rad/s = {wy * DEG:.1f}°/s；一步转 {wy} × 0.02 = {wy * 0.02:.5f} rad = {wy * 0.02 * DEG:.2f}°")
check("0.677 rad/s ≈ 每秒 38.8°；一步 0.677 × 0.02 = 0.01354 rad ≈ 0.78°（按印出来的 0.01354 × 57.3 重算也是 0.78）",
      round(wy * DEG, 1) == 38.8 and round(0.677 * 0.02, 5) == 0.01354 and round(0.677 * 0.02 * DEG, 2) == 0.78
      and round(0.01354 * 57.3, 2) == 0.78)
th = 0.1                                                  # 绕 y 轴“正着转”0.1 rad（右手定则：大拇指指向 y 轴正方向 = 左边）
Ry = np.array([[math.cos(th), 0.0, math.sin(th)], [0.0, 1.0, 0.0], [-math.sin(th), 0.0, math.cos(th)]])
nose = Ry @ np.array([1.0, 0.0, 0.0])                     # 机头（x 轴，朝前）转到了哪
top = Ry @ np.array([0.0, 0.0, 1.0])                      # 头顶（z 轴，朝上）转到了哪
g_after = Ry.T @ np.array([0.0, 0.0, -1.0])               # 转完以后，从身体看“下”在哪（投影重力，2.4 节）
print(f"绕 y 轴正着转 0.1 rad：头顶 z 轴 → ({top[0]:.3f}, {top[1] + 0.0:.0f}, {top[2]:.3f})，"
      f"机头 x 轴 → ({nose[0]:.3f}, {nose[1] + 0.0:.0f}, {nose[2]:.3f})；投影重力 → ({g_after[0]:.3f}, {g_after[1] + 0.0:.0f}, {g_after[2]:.3f})")
check("右手定则：绕 y 轴正着转 = 头顶往前、机头往下（往前低头），投影重力第 1 项随之变正（2.4 节：往前倾 g_x 为正）；"
      "这一帧陀螺仪第 2 项 +0.677 > 0：正往前低头", top[0] > 0 and nose[2] < 0 and g_after[0] > 0 and wy > 0)
fall = cfg.terminations["fell_over"].params["limit_angle"]
print(f"fell_over 阈值 {math.degrees(fall):g}°：cos 70° = {math.cos(fall):.3f}，g_z 高过 −{math.cos(fall):.3f} 就算摔倒")
check("摔倒阈值 70°，对应 g_z = −cos 70° = −0.342", round(math.degrees(fall), 6) == 70 and round(math.cos(fall), 3) == 0.342)
play_cfg = make_microduck_velocity_env_cfg(play=True)
push = play_cfg.events["push_robot"].interval_range_s
print(f"演示模式（play=True）每 {push[0]}–{push[1]} 秒推一把")
check("演示模式每 0.5–1 秒推一把（GPU 实验用的就是它）", tuple(push) == (0.5, 1.0))
check("自测：重力方向 (0, −0.5, −0.866) → arccos(0.866) = 30°；g_y 为负 → 往右歪", round(math.degrees(math.acos(0.866))) == 30)
imu = [cfg.observations["actor"].terms[n].params.get("max_angle_deg") for n in ("base_ang_vel", "projected_gravity")]
print(f"IMU 安装误差：陀螺仪、重力方向各最多 {imu[0]}°、{imu[1]}°；站得笔直也会读到 g_z = −cos 6° = {-math.cos(math.radians(6)):.4f}，"
      f"水平分量最多 sin 6° = {math.sin(math.radians(6)):.3f}")
check("cfg 注释：真机板子约 5° 的固定前后偏差在运行时校准（imu-pitch-offset），不靠这项随机化",
      "systematic ~5° pitch offset is corrected at the source in the runtime (imu-pitch-offset)"
      in (Path(mjlab_microduck.__file__).resolve().parent / "tasks" / "microduck_velocity_env_cfg.py").read_text(encoding="utf-8"))
check("IMU 安装误差最多 6°（两块共用同一个旋转）；装歪 6° 时水平分量最多 sin 6° = 0.105", imu == [6.0, 6.0]
      and cfg.observations["actor"].terms["base_ang_vel"].func.__name__ == "base_ang_vel_imu_misaligned"
      and round(math.sin(math.radians(6)), 3) == 0.105 and round(math.cos(math.radians(6)), 4) == 0.9945)

# ---------------------------------------------------------------------------
banner("3. 14 个关节：关节角（减 HOME）、关节速度、上一步动作")
home = {}
for n in hinges:
    home[n] = next(v for k, v in HOME_FRAME.joint_pos.items() if re.fullmatch(k, n))
rows = []
for j, n in enumerate(hinges):
    rows.append([j, n, f"{home[n]:.4f}", 6 + j, 20 + j, 34 + j])
table(["关节序号", "关节名", "HOME (rad)", "关节角的编号", "关节速度的编号", "动作的编号"], rows)
check("14 个舵机关节：0–4 左腿、5–8 颈和头、9–13 右腿（和动作一一对应）", len(hinges) == 14 and model.nu == 14
      and hinges[:5] == ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle"]
      and hinges[5:9] == ["neck_pitch", "head_pitch", "head_yaw", "head_roll"] and hinges[9].startswith("right_hip_yaw")
      and [model.actuator(i).name for i in range(model.nu)] == hinges)
groups = [(0, 5), (5, 9), (9, 14)]                       # 左腿、颈和头、右腿（关节序号的 [a, b)）
check("分组编号：关节角 6–10 / 11–14 / 15–19，关节速度 20–24 / 25–28 / 29–33，动作 34–38 / 39–42 / 43–47；左膝是 9、23、37",
      [(base + a, base + b - 1) for base in (6, 20, 34) for a, b in groups]
      == [(6, 10), (11, 14), (15, 19), (20, 24), (25, 28), (29, 33), (34, 38), (39, 42), (43, 47)]
      and [base + hinges.index("left_knee") for base in (6, 20, 34)] == [9, 23, 37])
print(f"颈俯仰的 HOME = {home['neck_pitch']} rad = {home['neck_pitch'] * DEG:.1f}°；教学例子取整成 0.35")
check("自测：0.30 − 0.35 = −0.05；离目标还差 0.10 − (−0.05) = 0.15 rad", round(0.30 - 0.35, 2) == -0.05
      and round(0.10 - (-0.05), 2) == 0.15)
check("颈俯仰 HOME = 0.3491 rad = 20.0°", home["neck_pitch"] == 0.3491 and round(0.3491 * DEG, 1) == 20.0)
print(f"读数 0.40、HOME 0.35：观测 = 0.40 − 0.35 = {0.40 - 0.35:.2f} rad = {0.05 * DEG:.4f}°；"
      f"按 57.3 算是 0.05 × 57.3 = {0.05 * 57.3:.3f}°，都约 2.9°")
check("0.40 − 0.35 = 0.05 rad，约 2.9°（比站姿多转的角度，不是关节的绝对角）", round(0.40 - 0.35, 2) == 0.05
      and round(0.05 * DEG, 1) == 2.9 and round(0.05 * 57.3, 1) == 2.9)
check("上一步动作 +0.10、现在 +0.05：还差 0.10 − 0.05 = 0.05 rad 没追上", round(0.10 - 0.05, 2) == 0.05)
jp, jv = FRAME["joint_pos"], FRAME["joint_vel"]
k_pos = int(np.argmax(np.abs(jp)))
k_vel = int(np.argmax(np.abs(jv)))
print(f"这一帧离 HOME 最远：{hinges[k_pos]}（编号 {6 + k_pos}）{jp[k_pos]} rad = {jp[k_pos] * DEG:.1f}°；"
      f"转得最快：{hinges[k_vel]}（编号 {20 + k_vel}）{jv[k_vel]} rad/s")
check("离站姿最远的是左膝（编号 9）：0.162 rad ≈ 9.3°；转得最快的是左踝（编号 24）：−0.714 rad/s",
      hinges[k_pos] == "left_knee" and 6 + k_pos == 9 and round(0.162 * DEG, 1) == 9.3
      and hinges[k_vel] == "left_ankle" and 20 + k_vel == 24 and jv[k_vel] == -0.714)
print(f"左髋偏航的关节角 {jp[0]} 和重力第 3 项 {gz}：这一帧相差 {abs(gz) / jp[0]:.0f} 倍；"
      f"按第 8 章 8.2 节各列平常的大小，关节速度 ±8 是身体位置命令 ±0.005 的 {8 / 0.005:.0f} 倍")
check("0.971 ÷ 0.004 ≈ 243（这一帧碰巧）；第 8 章 8.2 节：8 ÷ 0.005 = 1600（各列平常的大小，归一化的真正理由）",
      round(0.971 / 0.004) == 243 and round(8 / 0.005) == 1600)
check("随机小动作全在 ±0.11 以内，最大的是左髋偏航 0.108", max(map(abs, FRAME["actions"])) == 0.108 == FRAME["actions"][0])
act = cfg.actions["joint_pos"]
print(f"动作的换算：scale = {act.scale}，use_default_offset = {act.use_default_offset}（目标角 = HOME + 动作）")
check("动作 scale = 1、以 HOME 为零点：动作 0.108 = 目标比站姿多 0.108 rad", act.scale == 1.0 and act.use_default_offset)
bias = cfg.events["encoder_bias"].params["bias_range"]
print(f"编码器偏置：每只机器人每个关节固定一个 U{bias} rad = ±{bias[1] * DEG:.2f}°")
check("编码器偏置 ±0.015 rad ≈ ±0.86°：actor 的关节角带偏置，critic 的不带", bias == (-0.015, 0.015)
      and round(0.015 * DEG, 2) == 0.86 and cfg.observations["actor"].terms["joint_pos"].params["biased"] is True
      and cfg.observations["critic"].terms["joint_pos"].params["biased"] is False)
check("actor 里没有身体线速度 base_lin_vel（critic 里有），两组都没有地形扫描 height_scan",
      "base_lin_vel" not in cfg.observations["actor"].terms and "base_lin_vel" in cfg.observations["critic"].terms
      and all("height_scan" not in cfg.observations[g].terms for g in ("actor", "critic")))

# ---------------------------------------------------------------------------
banner("4. 噪声和延迟：每一项都“差一点、晚一点”")
actor_grp = cfg.observations["actor"]
rows = []
for name, term in actor_grp.terms.items():
    nz = term.noise if actor_grp.enable_corruption else None
    lag = f"{term.delay_min_lag}–{term.delay_max_lag}" if term.delay_max_lag else "无"
    period = f"{term.delay_update_period}" if term.delay_max_lag and term.delay_update_period else "—"
    rows.append([name, f"±{nz.n_max:g}" if nz is not None else "无", lag, period])
table(["观测块", "均匀噪声", "延迟（步）", "每几步重抽一次延迟"], rows)
T = actor_grp.terms
noise = {n: (T[n].noise.n_min, T[n].noise.n_max) for n in T if T[n].noise is not None}
check("噪声：陀螺仪 ±0.03、重力 ±0.01、关节角 ±0.001、关节速度 ±0.25，其余 4 块没有", actor_grp.enable_corruption
      and noise == {"base_ang_vel": (-0.03, 0.03), "projected_gravity": (-0.01, 0.01), "joint_pos": (-0.001, 0.001),
                    "joint_vel": (-0.25, 0.25)} and all(T[n].noise.operation == "add" for n in noise))
lags = {n: (T[n].delay_min_lag, T[n].delay_max_lag) for n in T}
check("延迟：陀螺仪、重力 0–1 步，关节速度固定 1 步，其余 5 块没有", lags["base_ang_vel"] == (0, 1)
      and lags["projected_gravity"] == (0, 1) and lags["joint_vel"] == (1, 1)
      and all(lags[n] == (0, 0) for n in ("joint_pos", "actions", "command", "head_command", "body_command")))
per = T["base_ang_vel"].delay_update_period
print(f"陀螺仪和重力的延迟每 {per} 步重抽一次：{per} × 0.02 = {per * 0.02:.2f} s")
check("0 步还是 1 步，每 64 步（1.28 s）重抽一次", per == 64 == T["projected_gravity"].delay_update_period
      and round(64 * 0.02, 2) == 1.28)
print(f"关节速度噪声 ±0.25 占第 8 章 8.2 节典型幅度 ±8 rad/s 的 {0.25 / 8:.1%}；关节角噪声 ±0.001 rad = ±{0.001 * DEG:.4f}°")
check("0.25 ÷ 8 ≈ 3%；0.001 rad ≈ 0.057°", round(0.25 / 8 * 100) == 3 and round(0.001 * DEG, 3) == 0.057)
print("真实速度 0.40 rad/s 的关节，actor 读到的在 0.40 − 0.25 = 0.15 到 0.40 + 0.25 = 0.65 之间")
check("0.40 ± 0.25：0.15 到 0.65", math.isclose(0.40 - 0.25, 0.15) and math.isclose(0.40 + 0.25, 0.65))

check("自测：上一步 0.20 加 ±0.25 的噪声 → −0.05 到 0.45", round(0.20 - 0.25, 2) == -0.05 and round(0.20 + 0.25, 2) == 0.45)
TRUE = [0.10, 0.30, 0.50, 0.70, 0.90]                    # 某个关节连续 5 步的真实速度（教学构造的数）
LAG_STEPS = 1  # TWEAK-3: 3


def delayed_by_queue(values, lag):
    """JS 那段 push / 往回数 lag 个的 Python 版：不够 lag 个时，拿最早的那个。"""
    history, out = [], []
    for v in values:
        history.append(v)
        out.append(history[max(0, len(history) - 1 - lag)])
    return out


from mjlab.utils.buffers.delay_buffer import DelayBuffer  # noqa: E402

buf = DelayBuffer(min_lag=LAG_STEPS, max_lag=LAG_STEPS, batch_size=1, device="cpu")
seen_mjlab = []
for v in TRUE:
    buf.append(torch.tensor([[v]]))
    seen_mjlab.append(round(float(buf.compute()[0, 0]), 2))
seen_queue = delayed_by_queue(TRUE, LAG_STEPS)
table(["第几步", "真实值", f"晚 {LAG_STEPS} 步交出去的（队列）", "mjlab 的 DelayBuffer"],
      [[t, f"{v:.2f}", f"{q:.2f}", f"{m:.2f}"] for t, (v, q, m) in enumerate(zip(TRUE, seen_queue, seen_mjlab))])
check(f"手写的队列和 mjlab 的 DelayBuffer 交出同一串数（晚 {LAG_STEPS} 步 = {LAG_STEPS * 20} ms）", seen_queue == seen_mjlab)
check("晚 1 步：真实 0.10, 0.30, 0.50, 0.70, 0.90 → 交出去 0.10, 0.10, 0.30, 0.50, 0.70",
      seen_queue == [0.10, 0.10, 0.30, 0.50, 0.70])

# ---------------------------------------------------------------------------
banner("5. 命令块：速度 3 项 + 头 4 项 + 身体 6 项")
tw = cfg.commands["twist"]
print("速度命令范围：", {f: getattr(tw.ranges, f) for f in TWIST_FIELDS}, " 每", tw.resampling_time_range, "秒重抽一次")
check("速度命令：前后 ±0.4 m/s、左右 ±0.3 m/s、转向 ±1.0 rad/s；每 3–8 s 重抽", tw.ranges.lin_vel_x == (-0.4, 0.4)
      and tw.ranges.lin_vel_y == (-0.3, 0.3) and tw.ranges.ang_vel_z == (-1.0, 1.0) and tw.resampling_time_range == (3.0, 8.0))
stand = cfg.curriculum["standing_envs"].params["standing_stages"]
print(f"拿到恰好全零速度命令（站着别动）的机器人比例：起步 {tw.rel_standing_envs:.0%}，课程最后 {stand[-1]['rel_standing_envs']:.0%}")
check("全零速度命令的比例：起步 2%，课程涨到 25%", tw.rel_standing_envs == 0.02 == stand[0]["rel_standing_envs"]
      and stand[-1]["rel_standing_envs"] == 0.25)
hp = cfg.curriculum["head_pose_range"].params["range_stages"]
head_first = [r[1] for r in hp[0]["ranges"]]
head_last = [r[1] for r in hp[-1]["ranges"]]
table(["头部关节", "起步 ±(rad)", "最后 ±(rad)", "最后 ±(度)"],
      [[n, f"{a:g}", f"{b:.2f}", f"{b * DEG:.0f}"] for n, a, b in zip(hinges[5:9], head_first, head_last)])
check("头部命令起步 ±0.05 / ±0.05 / ±0.07 / ±0.015 rad，和 cfg.commands 的初值一致",
      head_first == [0.05, 0.05, 0.07, 0.015] == [r[1] for r in cfg.commands["head_pose"].ranges])
check("头部命令最后 ±1.10 / ±1.10 / ±1.40 / ±0.31 rad（约 ±63° / ±63° / ±80° / ±18°），在第 2000 次迭代（步 = 2000 × 24）放到最宽",
      head_last == [1.10, 1.10, 1.40, 0.31] and [round(b * DEG) for b in head_last] == [63, 63, 80, 18]
      and hp[-1]["step"] == 2000 * 24 and len(hp) == 5)
bp = cfg.commands["body_pose"]
print("身体命令范围：", bp.ranges, " 课程只有", len(cfg.curriculum["body_pose_range"].params["range_stages"]), "档（一直不变）")
check("身体命令：位置 ±0.005 m（5 mm）、角度 ±0.05 rad（约 2.9°），整个训练不变",
      [r[1] for r in bp.ranges] == [0.005] * 3 + [0.05] * 3 and len(cfg.curriculum["body_pose_range"].params["range_stages"]) == 1
      and round(0.05 * DEG, 1) == 2.9)
check("头、身体命令每 2–5 s 重抽", cfg.commands["head_pose"].resampling_time_range == (2.0, 5.0) == bp.resampling_time_range)
vx, vy, wz = FRAME["command"]
print(f"这一帧的速度命令：前进 {vx} m/s、左移 {vy} m/s、左转 {wz} rad/s = {wz * DEG:.1f}°/s")
check("命令 (0.231, 0.267, 0.613)：都在范围里；0.613 rad/s ≈ 每秒 35°", abs(vx) <= 0.4 and abs(vy) <= 0.3 and abs(wz) <= 1.0
      and round(wz * DEG) == 35)
check("这一帧的头部命令在起步范围里（课程刚开始），身体命令在 ±0.005 m / ±0.05 rad 里",
      all(abs(v) <= r for v, r in zip(FRAME["head_command"], head_first))
      and all(abs(v) <= r[1] for v, r in zip(FRAME["body_command"], bp.ranges)))

# ---------------------------------------------------------------------------
banner("5b. 画图：figures/ch15_one_frame.png（一帧 61 个数逐块翻成人话）")
fig, ax = plt.subplots(figsize=(10.5, 14.2))
fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)
fig.suptitle(f"一帧 61 个数翻成人话：前倾约 {tilt:.0f}°，命令要它前进、左移、左转", fontsize=FS_TITLE,
             fontweight="bold", color=INK)
lesson_panel(ax, xmax=10.5, ymax=18.4)
LONG = {**ZH, "command": "速度命令", "head_command": "头部命令", "body_command": "身体命令"}
ax.text(0.1, 17.9, "① 前 48 项：机器人自己身上测到的", fontsize=FS_STEP, fontweight="bold", color=INK, va="center")
ax.text(0.1, 7.6, "② 后 13 项：人给的命令", fontsize=FS_STEP, fontweight="bold", color=INK, va="center")
lines = {
    "base_ang_vel": ("(−0.032, 0.677, −0.163) rad/s", f"绕左右轴转得最快，为正：往前低头，每秒 {wy * DEG:.1f}°"),
    "projected_gravity": ("(0.243, 0.012, −0.971)", f"arccos(0.971) ≈ {tilt:.1f}°；第 1 项为正、第 2 项≈0：往前倾"),
    "joint_pos": ("14 个数，离 0 最远的是编号 9：0.162 rad", "左膝离站姿最远：0.162 rad ≈ 9.3°（相对 HOME）"),
    "joint_vel": ("14 个数，离 0 最远的是编号 24：−0.714 rad/s", "左踝转得最快（而且是晚 1 步的读数）"),
    "actions": ("14 个数，都在 ±0.11 以内", "随机小动作：上一步让目标离站姿最多 0.108 rad"),
    "command": ("(0.231, 0.267, 0.613)", f"前进 0.231 m/s、左移 0.267 m/s、左转 ≈ 每秒 {wz * DEG:.0f}°"),
    "head_command": ("(0.046, −0.042, −0.038, −0.005) rad", "4 个头部关节的目标偏移，都在课程起步的范围里"),
    "body_command": ("(−0.004, 0.005, −0.000, −0.029, −0.039, 0.000)", "位置 ≤ 5 mm、角度 ≤ 0.05 rad：占位的小命令"),
}
for k, (name, d, a, b) in enumerate(full):
    y = 16.55 - k * 1.9 - (0.75 if a >= 48 else 0)
    cell(ax, 0.1, y - 0.3, f"{LONG[name]}  [{a}, {b})", width=3.0, height=1.05, facecolor=FILL[name], fontsize=FS_SMALL)
    raw, plain = lines[name]
    ax.text(3.35, y + 0.42, raw, fontsize=16, color=INK, va="center")
    ax.text(3.35, y - 0.12, plain, fontsize=16, color=BLUE if a < 48 else GREEN, va="center")
note(ax, 0.1, 0.55, "这一帧来自随机小动作（没有训练好的策略在开），命令只是被抽出来，并没有被执行。", fontsize=16)
savefig(fig, "ch15_one_frame")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("6. 热切换：同一张 61 项的清单，几个策略轮流读")
cmd = [0.2, 0.0, 0.5]
swapped = [cmd[2], cmd[1], cmd[0]]
print(f"约定 [vx, vy, ωz] = {cmd}；把第 1、3 项对调 → {swapped}：网络读成前进 {swapped[0]} m/s、转 {swapped[2]} rad/s")
check("对调之后形状照样是 3 项，意思全变：前进 0.5 m/s 还超出了训练范围 ±0.4", len(swapped) == 3 and swapped[0] == 0.5
      and swapped[0] > tw.ranges.lin_vel_x[1])
check("部署演练脚本的拼法：3 + 3 + 14 + 14 + 14 + 13 = 61（开 --new-cmd-obs）；旧布局命令只有 3 项：51",
      3 + 3 + 14 + 14 + 14 + 13 == 61 and 3 + 3 + 14 + 14 + 14 + 3 == 51)
hc = FRAME["head_command"]
qlen = math.sqrt(sum(v * v for v in hc))
print(f"头部命令 {hc} 的长度 √(0.046² + 0.042² + 0.038² + 0.005²) = {qlen:.3f}（表示旋转的四元数长度必须是 1）")
check("头部命令这 4 个数的长度 = 0.073，不是 1：它不是四元数", round(qlen, 3) == 0.073
      and round(math.sqrt(0.046 ** 2 + 0.042 ** 2 + 0.038 ** 2 + 0.005 ** 2), 3) == 0.073)

# ---------------------------------------------------------------------------
banner("7. 零填充与死权重：恒为 0 的输入，那一列权重永远拿不到梯度")
# 迷你版 = 第 7 章 7.2 节的例一（x = 2、y = 5、w = 1、b = 0），再接一个恒为 0 的输入 x₂，权重 w₂ = 0.5
w = torch.tensor([1.0, 0.5], requires_grad=True)
b0 = torch.tensor(0.0, requires_grad=True)
x = torch.tensor([2.0, 0.0])
pred = (w * x).sum() + b0
loss = (pred - 5.0) ** 2
loss.backward()
g1, g2 = w.grad.tolist()
print(f"预测 1 × 2 + 0.5 × 0 + 0 = {pred.item():g}；误差 {pred.item() - 5:g}；往回乘到预测处 2 × (−3) = −6")
print(f"对 w₁：−6 × 2 = {g1 + 0.0:g}；对 w₂：−6 × 0 = {g2 + 0.0:g}；对 b：{b0.grad.item():g}")
check("autograd：对 w₁ 的梯度 −12，对 w₂ 的梯度 0（和第 7 章 7.2 节的 −12、−6 同一个例子）", g1 == -12 and g2 == 0
      and b0.grad.item() == -6)
w_after = [1.0 - 0.1 * g1, 0.5 - 0.1 * g2]
print(f"学习率 0.1，更新一次：w₁ = 1 − 0.1 × (−12) = {w_after[0]:g}，w₂ = 0.5 − 0.1 × 0 = {w_after[1]:g}")
check("更新一次：w₁ 从 1 变成 2.2，w₂ 停在 0.5", math.isclose(w_after[0], 2.2) and w_after[1] == 0.5)
check("删掉身体命令 6 项：61 − 6 = 55，和别的策略的 61 项对不上", 61 - 6 == 55 and dict((n, d) for n, d, *_ in full)["body_command"] == 6)
w2 = 0.5
for _ in range(1000):                                    # 再更新 1000 次：x₂ 一直是 0，w₂ 一直拿到 0
    w2 -= 0.1 * (-6.0 * 0.0)
check("更新 1000 次之后 w₂ 仍是出厂的 0.5；部署时 x₂ 第一次变成 0.3，它贡献 0.5 × 0.3 = 0.15——没学过的数",
      w2 == 0.5 and math.isclose(0.5 * 0.3, 0.15))

up = torch.tensor([-6.0, 2.0, 1.0, -3.0])                # 往回乘到 4 个神经元的数（教学构造）
xin = torch.tensor([2.0, 1.0, 0.0])                      # 3 个输入，第 3 个恒为 0
W = torch.zeros(4, 3, requires_grad=True)
(W @ xin * up).sum().backward()                          # 构造成：往回乘到第 k 行的数恰好是 up[k]
grid = W.grad
print("一层 4 个神经元 × 3 个输入的梯度表（每格 = 这一行往回乘到的数 × 这一列的输入）：")
table(["", "输入 2", "输入 1", "输入 0"], [[f"行 {k + 1}（{up[k]:g}）", *[f"{v + 0.0:g}" for v in grid[k].tolist()]] for k in range(4)])
check("梯度表 = 往回乘的数 × 输入：第 3 列（输入恒为 0）4 格全是 0", torch.equal(grid, torch.outer(up, xin))
      and torch.all(grid[:, 2] == 0) and grid[0].tolist() == [-12.0, -6.0, 0.0])

from rsl_rl.modules import EmpiricalNormalization  # noqa: E402

SMALL_RANGE = 0.005  # TWEAK-2: 0.0005
gen = torch.Generator().manual_seed(15)
norm = EmpiricalNormalization(2)                          # 项目用的归一化器（ε = 0.01，第 8 章）
norm.train()
for _ in range(50):                                      # 50 批 × 4096 行：第 0 列恒为 0，第 1 列在 ±SMALL_RANGE 里均匀抽
    batch = torch.zeros(4096, 2)
    batch[:, 1] = (torch.rand(4096, generator=gen) * 2 - 1) * SMALL_RANGE
    norm.update(batch)
edge = norm(torch.tensor([[0.0, SMALL_RANGE]]))[0]
sigma_theory = 2 * SMALL_RANGE / math.sqrt(12)           # 第 4 章 4.4 节：均匀分布的方差 = 宽度² ÷ 12
edge_theory = SMALL_RANGE / (sigma_theory + norm.eps)
print(f"恒为 0 的一列：(0 − 0) ÷ (0 + ε) = {edge[0].item():g}（ε = {norm.eps:g}）")
print(f"±{SMALL_RANGE:g} 的一列：宽度 {2 * SMALL_RANGE:g}，σ = 宽度 ÷ √12 = {sigma_theory:.5f}；"
      f"{SMALL_RANGE:g} ÷ ({sigma_theory:.5f} + ε)"
      f" = {edge_theory:.3f}（归一化器实测 {edge[1].item():.3f}）；梯度 −6 × {edge_theory:.3f} = {-6 * edge_theory:.2f}")
check("恒为 0 的一列，归一化之后还是恰好 0", edge[0].item() == 0.0 and norm.eps == 0.01)
check(f"归一化器实测的 ±{SMALL_RANGE:g} 边缘 ≈ 理论值 {edge_theory:.3f}（差不到 1%）", abs(edge[1].item() - edge_theory) < 0.01 * edge_theory)
check("±0.005 → 宽度 0.01，σ = 0.01 ÷ 3.464 = 0.00289 → 0.005 ÷ (0.00289 + ε) = 0.388（ε = 0.01）；梯度 −6 × 0.388 = −2.33",
      round(sigma_theory, 5) == 0.00289 and round(math.sqrt(12), 3) == 3.464 and round(0.01 / 3.464, 5) == 0.00289
      and round(edge_theory, 3) == 0.388 and round(0.005 / (0.00289 + 0.01), 3) == 0.388
      and round(-6 * 0.388, 2) == -2.33)

# ---------------------------------------------------------------------------
banner("7b. 画图：figures/ch15_dead_weight.png（恒为 0 的一列拿不到梯度；小随机数就能）")
if SMALL_RANGE != 0.005:
    print("  SMALL_RANGE 改过了：这张图照着正文的 ±0.005 画，跳过。看上面打印的数就行。")
else:
    fig, axes = lesson_figure(3, "输入恒为 0，那一列权重的梯度就恒为 0：出厂的随机值一直留着", panel_height=3.6, width=9.6)
    ax = axes[0]
    lesson_panel(ax, "① 迷你版：第 7 章的例一，多接一个恒为 0 的输入")
    hand(ax, 0.2, 2.75, "预测 = 1 × 2 + 0.5 × 0 + 0 = 2，误差 = 2 − 5 = −3", color=INK, fontsize=FS_SMALL)
    hand(ax, 0.2, 2.1, "往回乘到预测处：2 × (−3) = −6", color=INK, fontsize=FS_SMALL)
    hand(ax, 0.2, 1.4, "对 w₁：−6 × 2 = −12", fontsize=FS_SMALL)
    hand(ax, 5.2, 1.4, "对 w₂：−6 × 0 = 0", color=ORANGE, fontsize=FS_SMALL)
    note(ax, 0.2, 0.6, "学习率 0.1，更新一次：w₁ = 1 + 1.2 = 2.2；w₂ = 0.5 − 0 = 0.5，一动不动。")
    ax = axes[1]
    lesson_panel(ax, "② 一层 4 个神经元：每格梯度 = 这一行往回乘到的数 × 这一列的输入")
    lesson_cells(ax, [[2, 1, 0]], left=3.3, bottom=2.55, width=1.3, height=0.62, facecolor="#eef1f4",
                 highlights={(0, 2)}, fontsize=17)
    ax.text(3.2, 2.86, "输入", fontsize=16, color=MUTED, ha="right", va="center")
    lesson_cells(ax, [[v] for v in up.tolist()], left=1.6, bottom=0.05, width=1.2, height=0.6, facecolor="#eef1f4", fontsize=17)
    ax.text(1.5, 2.2, "往回乘到\n各行的数", fontsize=15, color=MUTED, ha="right", va="center")
    lesson_cells(ax, [[int(v) for v in r] for r in grid.tolist()], left=3.3, bottom=0.05, width=1.3, height=0.6,
                 highlights={(k, 2) for k in range(4)}, fontsize=17)
    note(ax, 7.4, 1.6, "第 3 列全是 0：\n这 4 个权重\n永远不会动。", fontsize=16, linespacing=1.3)
    ax = axes[2]
    lesson_panel(ax, "③ 换成很小的随机数：归一化把 ±0.005 放大到 ±0.39")
    hand(ax, 0.2, 2.75, "恒为 0 的一列：(0 − 0) ÷ (0 + 0.01) = 0，归一化救不了", color=ORANGE, fontsize=FS_SMALL)
    hand(ax, 0.2, 2.05, "±0.005 的一列：宽度 0.01，σ = 0.01 ÷ √12 = 0.00289", fontsize=FS_SMALL)
    hand(ax, 0.2, 1.4, "边缘 0.005 ÷ (0.00289 + ε) = 0.388，ε = 0.01", fontsize=FS_SMALL)
    hand(ax, 0.2, 0.75, "这一列的梯度：−6 × 0.388 = −2.33，不再是 0", color=GREEN, fontsize=FS_SMALL)
    savefig(fig, "ch15_dead_weight")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("8. critic 多看的 15 项：特权观测")
critic = layout("critic")
actor_names = {n for n, *_ in full}
table(["critic 观测块", "项数", "区间 [a, b)", ""], [[n, d, f"[{a}, {b})", "" if n in actor_names else "★ 仅 critic"]
                                                  for n, d, a, b in critic])
extra = [(n, d) for n, d, _, _ in critic if n not in actor_names]
print("多出来的：", " + ".join(f"{n} {d}" for n, d in extra), f"= {sum(d for _, d in extra)}")
check("从 cfg 推出的 critic 13 块 = GPU 实验第 4 节读出的表", [(n, d) for n, d, _, _ in critic] == GPU_CRITIC)
check("多出 3 + 2 + 2 + 2 + 6 = 15 项；61 + 15 = 76", [d for _, d in extra] == [3, 2, 2, 2, 6]
      and critic[-1][3] == 76 == 61 + 15)
import mjlab.tasks.velocity.velocity_env_cfg as tmpl_mod  # noqa: E402

tmpl_text = Path(tmpl_mod.__file__).read_text(encoding="utf-8")
check("mjlab 模板：critic 的清单 = actor 的清单（打头是线速度）+ 地形扫描 + 脚的 4 块——所以 critic 里线速度打头、脚夹在中间",
      lines_in_order(tmpl_text, ["actor_terms = {", '"base_lin_vel": ObservationTermCfg(', "critic_terms = {", "**actor_terms,",
                                 '"height_scan": ObservationTermCfg(', '"foot_height": ObservationTermCfg(',
                                 '"foot_air_time": ObservationTermCfg(', '"foot_contact": ObservationTermCfg(',
                                 '"foot_contact_forces": ObservationTermCfg(']))
crit_grp = cfg.observations["critic"]
check("critic 这一组 enable_corruption = False（不加噪声），也没有任何延迟", crit_grp.enable_corruption is False
      and all(t.delay_max_lag == 0 for t in crit_grp.terms.values()))
check("critic 的陀螺仪读仿真里的真值（没有装歪的旋转），关节角不带编码器偏置",
      crit_grp.terms["base_ang_vel"].func.__name__ == "builtin_sensor"
      and crit_grp.terms["projected_gravity"].func.__name__ == "projected_gravity")
leftover = sorted(n for n, t in crit_grp.terms.items() if t.noise is not None)
print("⚠ critic 的 cfg 里其实还挂着噪声设置（从模板继承来的）：", leftover, "——建环境时 mjlab 会因为 enable_corruption=False 把它们清掉")
cd = dict((n, (a, b)) for n, _, a, b in critic)
print(f"同名的块在两张清单里编号不同：关节速度 actor [20, 34)，critic [{cd['joint_vel'][0]}, {cd['joint_vel'][1]})")
check("critic 的关节速度从 3 + 3 + 3 + 14 = 23 开始（actor 是 20）", cd["joint_vel"] == (23, 37) and 3 + 3 + 3 + 14 == 23)
check("自测：critic 的身体命令 [70, 76)，头部命令 [66, 70)", cd["body_command"] == (76 - 6, 76) and cd["head_command"] == (66, 70))
F = torch.tensor([4.0, 40.0, -4.0])
comp = torch.sign(F) * torch.log1p(torch.abs(F))        # mjlab 的 foot_contact_forces 就是这一行
print(f"接触力压缩 sign(F) × ln(1 + |F|)：4 N → {comp[0]:.3f}，40 N → {comp[1]:.3f}，−4 N → {comp[2]:.3f}；"
      f"力大 10 倍，数只大 {comp[1] / comp[0]:.1f} 倍")
check("ln(1 + 4) = 1.609，ln(1 + 40) = 3.714，−4 → −1.609；3.714 ÷ 1.609 ≈ 2.3；F = 0 记 ln 1 = 0", [round(v, 3) for v in comp.tolist()]
      == [1.609, 3.714, -1.609] and round(3.714 / 1.609, 1) == 2.3 and math.log1p(0.0) == 0.0)
mass = float(sum(model.body_mass))
print(f"机器人模型总质量 {mass:.3f} kg；两只脚站着，每只脚托 {mass:.3f} × 9.8 ÷ 2 = {mass * 9.8 / 2:.2f} N，和 4 N 差不多")
check("模型总质量 0.737 kg；0.737 × 9.8 ÷ 2 = 3.61 N", round(mass, 3) == 0.737 and round(0.737 * 9.8 / 2, 2) == 3.61)

# ---------------------------------------------------------------------------
banner("8b. 画图：figures/ch15_actor_critic.png（actor 61 项 vs critic 76 项）")
fig = plt.figure(figsize=(10.2, 13.4))
fig.suptitle("critic 的清单 = actor 的 61 项 + 15 项仿真才知道的", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
ax = fig.add_axes([0.02, 0.585, 0.96, 0.36])
lesson_panel(ax, "① 同一把尺画两条：橙色 5 块只有 critic 有，插在不同的位置", xmax=10.4, ymax=5.6)
unit, x0 = 9.8 / 76, 0.3
ax.text(x0, 4.55, "actor 61 项", fontsize=FS_SMALL, color=INK, va="center")
block_strip(ax, full, x0, 3.6, unit, 0.55, labels=False)
for k in (20, 34):
    ax.text(x0 + k * unit, 4.37, str(k), fontsize=13, color=MUTED, ha="center", va="center")
ax.text(x0, 2.95, "critic 76 项", fontsize=FS_SMALL, color=INK, va="center")
block_strip(ax, critic, x0, 2.0, unit, 0.55, labels=False)
for k in (23, 37, 54, 66, 76):
    ax.text(x0 + k * unit, 2.77, str(k), fontsize=13, color=MUTED, ha="center", va="center")
for (a, b, text) in ((0, 3, "真实线速度 3"), (54, 66, "脚的 4 块：2 + 2 + 2 + 6 = 12")):
    xa, xb = x0 + a * unit + 0.02, x0 + b * unit - 0.02
    ax.plot([xa, xa, xb, xb], [1.88, 1.74, 1.74, 1.88], color=ORANGE, lw=2)
    ax.text((xa + xb) / 2 if a else xa, 1.4, text, fontsize=15, color=ORANGE, ha="center" if a else "left", va="center")
hand(ax, x0, 0.78, "3 + 12 = 15 项，61 + 15 = 76", fontsize=FS_SMALL)
ax.text(x0, 0.15, "关节速度：actor 从编号 20 起，critic 从 23 起——同名的块，编号不一样。", fontsize=16, color=MUTED, va="center")
ax = fig.add_axes([0.02, 0.29, 0.96, 0.28])
lesson_panel(ax, "② 同名的块，critic 拿的是干净值", xmax=10.4, ymax=4.6)
rows = [["", "actor", "critic"], ["噪声", "±0.03 等", "无"], ["延迟", "0–1 步 / 1 步", "无"], ["IMU 装歪", "最多 6°", "无"],
        ["编码器偏置", "±0.015 rad", "无"]]
lesson_cells(ax, rows, left=0.6, bottom=0.2, width=2.9, height=0.72, fontsize=17,
             highlights={(k, 2) for k in range(1, 5)})
ax = fig.add_axes([0.10, 0.045, 0.52, 0.19])
Fs = np.linspace(0, 45, 200)
data_axes(ax, "一只脚的接触力 F（N）", "记下的数")
ax.plot(Fs, np.log1p(Fs), color=ORANGE, lw=3)
for f_val, y_val in ((4.0, float(comp[0])), (40.0, float(comp[1]))):
    ax.plot([f_val], [y_val], "o", color=INK, ms=8, zorder=5)
    ax.text(f_val + 1.2 if f_val < 10 else f_val - 1.2, y_val - (0.5 if f_val < 10 else 0.85), f"{f_val:g} N → {y_val:.3f}", fontsize=15, color=INK,
            ha="left" if f_val < 10 else "right")
ax.set_xlim(0, 46)
ax.set_ylim(0, 4.3)
ax.set_xticks([0, 4, 10, 20, 30, 40])
fig.text(0.02, 0.262, "③ 接触力先压缩：记 ln(1 + F)，力大 10 倍，数只大 2.3 倍", fontsize=FS_STEP, fontweight="bold", color=INK)
fig.text(0.66, 0.13, f"4 N 左右：站着时\n一只脚托的重量\n（{mass:.3f} kg × 9.8 ÷ 2\n = {mass * 9.8 / 2:.2f} N）", fontsize=16,
         color=MUTED, va="center", linespacing=1.35)
savefig(fig, "ch15_actor_critic")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("9. 一项观测的生产线：算出来 → 加噪声 → 裁剪 → 缩放 → 延迟 → 拼接")
import mjlab.managers.observation_manager as om_mod  # noqa: E402

om_text = Path(om_mod.__file__).read_text(encoding="utf-8")
check("mjlab 写明的工序：compute → noise → clip → scale → delay → history",
      "Processing pipeline: compute → noise → clip → scale → delay → history." in om_text)
check("actor 的 8 块都没设裁剪、缩放、历史：这 3 道工序本任务没用上",
      all(t.clip is None and t.scale is None and t.history_length == 0 for t in actor_grp.terms.values()))
knee = hinges.index("left_knee")
i_actor = dict((n, a) for n, _, a, _ in full)["joint_vel"] + knee
i_critic = cd["joint_vel"][0] + knee
print(f"左膝速度：真实 0.40 rad/s；这一步抽到噪声 +0.12（教学构造）→ 0.40 + 0.12 = {0.40 + 0.12:.2f}；"
      f"排队晚 1 步；拼进 actor 的编号 {i_actor}（critic 里是编号 {i_critic}，拿到的是 0.40）")
check("0.40 + 0.12 = 0.52（+0.12 在 ±0.25 以内）；左膝速度在 actor 的编号 20 + 3 = 23，critic 的 23 + 3 = 26",
      math.isclose(0.40 + 0.12, 0.52) and abs(0.12) <= 0.25 and i_actor == 23 and i_critic == 26)

# ---------------------------------------------------------------------------
banner("9b. 画图：figures/ch15_pipeline.png（一项观测的生产线）")
fig, axes = lesson_figure(3, "一项观测的生产线：加噪声、晚一步，再拼进自己的位置", panel_height=3.3, width=10.2)


def pipe_row(ax, items, y, w=1.52, gap=0.22):
    for k, (top, bottom, active) in enumerate(items):
        x = 0.1 + k * (w + gap)
        cell(ax, x, y, "", width=w, height=1.25, facecolor=CELL if active else "white",
             edgecolor=CELL_EDGE if active else FAINT)
        ax.text(x + w / 2, y + 0.88, top, ha="center", va="center", fontsize=16, color=INK if active else FAINT)
        ax.text(x + w / 2, y + 0.36, bottom, ha="center", va="center", fontsize=15, color=BLUE if active else FAINT)
        if k:
            arrow(ax, (x - gap + 0.02, y + 0.62), (x - 0.02, y + 0.62), MUTED, lw=1.5)


ax = axes[0]
lesson_panel(ax, f"① actor 的左膝速度（编号 {i_actor}）走一遍", xmax=10.4)
ax.text(0.1 + 1.52 + 0.22 + 0.76, 2.85, "抽到 +0.12", fontsize=15, color=ORANGE, ha="center", va="center")
pipe_row(ax, [("算出来", "0.40", True), ("加噪声", "0.52", True), ("裁剪", "不用", False),
              ("缩放", "不用", False), ("延迟 1 步", "排队", True), ("拼接", f"放进 {i_actor}", True)], 1.35)
note(ax, 0.1, 0.62, "0.52 先进队列，下一步才交给网络；这一步交出去的是上一步的数。", fontsize=16)
ax = axes[1]
lesson_panel(ax, f"② critic 的同一项（编号 {i_critic}）：不加噪声、不排队", xmax=10.4)
pipe_row(ax, [("算出来", "0.40", True), ("加噪声", "关掉", False), ("裁剪", "不用", False),
              ("缩放", "不用", False), ("延迟", "没有", False), ("拼接", f"放进 {i_critic}", True)], 1.35)
note(ax, 0.1, 0.62, "critic 拿到的就是这一步的真值 0.40。", fontsize=16)
ax = axes[2]
lesson_panel(ax, "③ 拼好的 61 项，再交给归一化器和网络", xmax=10.4)
for k, (t, hot) in enumerate([("61 项原始读数", False), ("归一化器", True), ("网络", True), ("14 个动作", False)]):
    x = 0.1 + k * 2.6
    cell(ax, x, 1.45, t, width=2.15, height=0.95, facecolor=CELL_HOT if hot else CELL, fontsize=FS_SMALL)
    if k:
        arrow(ax, (x - 0.43, 1.925), (x - 0.02, 1.925), MUTED, lw=1.5)
ax.add_patch(plt.Rectangle((2.62, 1.3), 4.8, 1.25, facecolor="none", edgecolor=BLUE, lw=2, ls="--"))
note(ax, 0.1, 0.62, "蓝虚线框 = 导出的 ONNX 文件（第 8 章 8.6 节）：真机只喂原始读数。", fontsize=16)
savefig(fig, "ch15_pipeline")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("10. 映射到项目：正文引用的配置行和源码行还在不在")
PKG = Path(mjlab_microduck.__file__).resolve().parent
REPO = PKG.parents[1]
cfg_text = (PKG / "tasks" / "microduck_velocity_env_cfg.py").read_text(encoding="utf-8")
CFG_LINES = [
    'del cfg.observations["actor"].terms["base_lin_vel"]',
    'del cfg.observations["actor"].terms["height_scan"]',
    'del cfg.observations["critic"].terms["height_scan"]',
    'cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(',
    'cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1',
    'cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)',
    'cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)',
    "av.func = microduck_mdp.base_ang_vel_imu_misaligned",
    'cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1',
    'cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1',
    'cfg.observations["actor"].terms["joint_pos"].params["biased"] = True',
    'cfg.observations["critic"].terms["joint_pos"].params["biased"] = False',
    'for group in ("actor", "critic"):',
    'cfg.observations[group].terms["head_command"] = ObservationTermCfg(',
    'cfg.observations[group].terms["body_command"] = ObservationTermCfg(',
]
at = cfg_text.find("def make_microduck_velocity_env_cfg(")
check(f"microduck_velocity_env_cfg.py 的 make_microduck_velocity_env_cfg()：映射块的 {len(CFG_LINES)} 行原样存在、顺序一致",
      at >= 0 and lines_in_order(cfg_text[at:], CFG_LINES))
infer = REPO / "scripts" / "infer_policy.py"
if infer.is_file():
    it = infer.read_text(encoding="utf-8")
    body = it[it.find("    def get_observations(self):"):it.find("    def trigger_ground_pick(self):")]
    INFER_LINES = ["self.command = np.zeros(13 if self.new_cmd_obs else 3, dtype=np.float32)", "Total: 51D",
                   "obs.append(self.command)", "return np.concatenate(obs).astype(np.float32)",
                   "expected_obs_size = 3 + 3 + policy.n_joints + policy.n_joints + policy.n_joints + cmd_dim",
                   "if test_obs.size != expected_obs_size:", 'WARNING: Observation size mismatch!']
    check("infer_policy.py：命令 13 项（开 --new-cmd-obs）/ 3 项；get_observations() 的说明还写着 51D、里面没有 assert；"
          "启动时按 3+3+14+14+14+13 核对长度，对不上只打印 WARNING", lines_in_order(it, INFER_LINES) and "Total: 51D" in body
          and "assert" not in body and "assert test_obs" not in it)
    order = re.findall(r"obs\.append\(self\.(\w+)", body)
    print("get_observations() 依次拼接：", order, "（get_raw_accelerometer 是不用投影重力时的备选，二选一）")
    check("部署演练脚本的拼接顺序 = 训练的 8 块顺序（3 块命令在这里并成一块 13 项）",
          [o for o in order if o != "get_raw_accelerometer"] == ["get_base_ang_vel", "get_projected_gravity",
                                                                  "get_joint_pos_relative", "get_joint_vel",
                                                                  "last_action", "command"])
    check("sitstand 策略把“坐 / 站”写在速度命令的第 1 项", "cmd[0] = 1.0 if self.sit_mode else 0.0" in it)
else:
    print("  （没找到 scripts/infer_policy.py，跳过这几项）")
import mjlab.rl.exporter_utils as eu  # noqa: E402

check("导出 ONNX 时 metadata 里写进 actor 的块名顺序 observation_names",
      '"observation_names": env.observation_manager.active_terms["actor"],' in Path(eu.__file__).read_text(encoding="utf-8"))
check("mjlab：enable_corruption=False 的组，建环境时把噪声设置清掉",
      lines_in_order(om_text, ["if not group_cfg.enable_corruption:", "term_cfg.noise = None"]))
agents = REPO / "AGENTS.md"
if agents.is_file():
    check("AGENTS.md 原句：A command input that is never non-zero has dead weights forever.",
          "A command input that is never non-zero has dead weights forever." in agents.read_text(encoding="utf-8"))
else:
    print("  （没找到 AGENTS.md，跳过这一项）")

done()
