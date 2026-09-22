"""第 17 章实验：动作 → 目标角；舵机里的两站（固件位置环 → 电机），再加摩擦与电池；延迟；阶跃响应；域随机化。

运行：uv run python docs/learn-zh/labs/ch17_bam_motor.py
纯 CPU：构造 velocity 任务的 cfg（做法同第 15、16、18 章的实验），从 bam 包里读 XL330 的拟合参数，
直接调用 bam 自己的 compute_control / compute_torque 核对手算。不建环境、不训练。
第 6 节的阶跃响应是简化的单关节（没有重力、没有地面；积分和“推不动就停住”是这里自己写的，不是 MuJoCo），
只用来看电压、延迟、摩擦各自的效果。
小节编号与正文一一对应：实验第 K 节 = 正文 17.K 节（第 9 节对应「映射到项目」）。
正文“改一改”要改的三行都带 `# TWEAK-k:` 标记（第 2、3、6 节各一处）。
"""

import dataclasses
import importlib.util
import inspect
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from _common import banner, check, done, lines_in_order, savefig, table
from _draw import (BLUE, CELL, CELL_EDGE, CELL_HOT, FAINT, FS_SMALL, FS_STEP, FS_TITLE, GREEN, INK,
                   MUTED, ORANGE, WHITE_BOX, arrow, cell, data_axes, hand, lesson_cells, lesson_figure, lesson_panel,
                   note, panel_note, panel_title, plt)

DEG = 180 / math.pi                                       # 1 rad 是多少度（附录 D.2）
PURPLE = "#6d4c9f"                                        # 本章第 4 种颜色：只在阶跃响应图里给“摩擦”那条线用

# ---------------------------------------------------------------------------
banner("1. 动作 → 目标角：站姿 HOME + 动作 × scale − 编码器偏置")
import mujoco  # noqa: E402

import mjlab_microduck  # noqa: E402,F401
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg  # noqa: E402
from mjlab_microduck.robot import microduck_constants as C  # noqa: E402  （先 import 任务再 import 常数，避免循环 import）

cfg = make_microduck_velocity_env_cfg()
act_cfg = cfg.actions["joint_pos"]
model = C.get_walk_spec().compile()                      # 走路用的机器人模型：读 14 个关节的名字和顺序
hinges = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
home = {n: next(v for k, v in C.HOME_FRAME.joint_pos.items() if re.fullmatch(k, n)) for n in hinges}
table(["关节", "HOME (rad)", "HOME (°)"], [[n, f"{home[n]:+.4f}", f"{home[n] * DEG:+.2f}"] for n in hinges])
check("14 个关节；左右髋俯仰 ∓26.24°、左右踝 ±25.95°、颈俯仰和头俯仰各 20.00°、髋横滚 ∓5.00°、膝 ∓0.28°",
      len(hinges) == 14 and round(home["left_hip_pitch"] * DEG, 2) == -26.24 and round(home["right_hip_pitch"] * DEG, 2) == 26.24
      and round(home["left_ankle"] * DEG, 2) == 25.95 and round(home["right_ankle"] * DEG, 2) == -25.95
      and round(home["neck_pitch"] * DEG, 2) == 20.00 and round(home["head_pitch"] * DEG, 2) == 20.00
      and round(home["left_hip_roll"] * DEG, 2) == -5.00 and round(home["left_knee"] * DEG, 2) == -0.28)
print(f"动作的换算：scale = {act_cfg.scale}，以默认站姿为零点 use_default_offset = {act_cfg.use_default_offset}，裁剪 clip = {act_cfg.clip}")
check("scale = 1.0、零点是 HOME、策略输出不裁剪（clip = None）",
      act_cfg.scale == 1.0 and act_cfg.use_default_offset and act_cfg.clip is None)
check("动作 1 个单位 = 1 rad ≈ 57.3°；每步变 0.1，目标角就跳约 5.7°", round(1.0 * act_cfg.scale * DEG, 1) == 57.3 and round(0.1 * DEG, 1) == 5.7)
check("关节限位惩罚 dof_pos_limits 是奖励里的一项（权重 −1.0），不是硬性的安全限制", cfg.rewards["dof_pos_limits"].weight == -1.0)

HOME_J, A_J, BIAS_J = 0.35, 0.10, 0.01                   # 教学构造的数：HOME 取整自颈俯仰 0.3491
target = HOME_J + A_J * act_cfg.scale - BIAS_J
reading = target + BIAS_J                                 # 真机：编码器读数 = 真实角 + 偏置
print(f"手算：{HOME_J} + {A_J} × {act_cfg.scale:g} − {BIAS_J} = {target:.2f} rad = {target * DEG:.1f}°")
print(f"真机：固件把读数推到 HOME + 动作 = {HOME_J + A_J:.2f}；读数比真实大 {BIAS_J}，真实角 = {HOME_J + A_J - BIAS_J:.2f}")
check("目标角 = 0.35 + 0.10 − 0.01 = 0.44 rad ≈ 25.2°", round(target, 2) == 0.44 and round(0.44 * DEG, 1) == 25.2)
check("字面值重算：0.35 + 0.1 − 0.01 = 0.44", round(0.35 + 0.1 - 0.01, 2) == 0.44)
check("真机读数推到 0.45、真实角 0.44；仿真直接给 0.44——两边真实角一样", round(reading, 2) == 0.45 and round(reading - BIAS_J, 2) == round(target, 2))
check("自测：HOME −0.46、动作 +0.06、偏置 −0.01 → −0.46 + 0.06 − (−0.01) = −0.39", round(-0.46 + 0.06 - (-0.01), 2) == -0.39)

# ---------------------------------------------------------------------------
banner("1b. 画图：figures/ch17_action_target.png（数轴上加站姿、加动作、减偏置）")
fig = plt.figure(figsize=(9.6, 9.0))
fig.suptitle("目标角 = 站姿 + 动作 − 偏置：策略只说“比站姿多转多少”", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
ax = fig.add_axes([0.08, 0.63, 0.86, 0.22])
data_axes(ax, "关节角（rad）", "", grid=False)
ax.set_xlim(-0.02, 0.50)
ax.set_ylim(-0.5, 1.6)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.44, 0.5])
ax.set_xticklabels(["0", "0.1", "0.2", "0.3", "0.35", "", "0.44", "0.5"])
ax.axhline(0, color=CELL_EDGE, lw=1.5)
ax.plot([0.0], [0], "o", color=MUTED, ms=9)
ax.text(0.0, 0.25, "关节角 0", color=MUTED, fontsize=FS_SMALL, ha="left")
ax.plot([HOME_J], [0], "o", color=BLUE, ms=12)
ax.text(HOME_J, -0.42, "站姿 HOME", color=BLUE, fontsize=FS_SMALL, ha="center")
arrow(ax, (0.0, 0.55), (HOME_J, 0.55), BLUE, lw=3)
ax.text(HOME_J / 2, 0.72, f"站姿 {HOME_J}", color=BLUE, fontsize=FS_SMALL, ha="center")
arrow(ax, (HOME_J, 1.05), (HOME_J + A_J, 1.05), GREEN, lw=4)
ax.text(HOME_J + A_J / 2, 1.24, f"+ 动作 {A_J:.2f}", color=GREEN, fontsize=FS_SMALL, ha="center")
arrow(ax, (HOME_J + A_J, 0.55), (target, 0.55), ORANGE, lw=4)
ax.text(HOME_J + A_J + 0.004, 0.72, f"− 偏置 {BIAS_J}", color=ORANGE, fontsize=FS_SMALL, ha="left")
ax.plot([target], [0], "o", color=ORANGE, ms=13, zorder=6)
ax.text(target, -0.42, "目标角", color=ORANGE, fontsize=FS_SMALL, ha="center")
panel_title(fig, [ax], f"① 手算：{HOME_J} + {A_J:.2f} − {BIAS_J} = {target:.2f} rad（约 {target * DEG:.1f}°）")
panel_note(fig, [ax], "三根箭头按同一把尺子画：动作 0.10 的箭头正好是偏置 0.01 的 10 倍长。")

ax = fig.add_axes([0.03, 0.05, 0.94, 0.36])
lesson_panel(ax, "② 为什么要减偏置：真机和仿真，最后停在同一个真实角度", xmax=10, ymax=5.2)
cell(ax, 0.2, 3.35, "真机", width=1.3, height=0.8, facecolor=CELL_HOT, edgecolor=ORANGE)
hand(ax, 1.75, 3.95, "编码器读数 = 真实角 + 0.01（这把尺子偏大）", color=INK, fontsize=FS_SMALL)
hand(ax, 1.75, 3.25, "固件把【读数】推到 0.35 + 0.10 = 0.45 → 真实角 0.44")
cell(ax, 0.2, 1.55, "仿真", width=1.3, height=0.8, facecolor=CELL)
hand(ax, 1.75, 2.15, "仿真里的舵机看的是真实角，没有歪尺子", color=INK, fontsize=FS_SMALL)
hand(ax, 1.75, 1.45, "所以直接给它 0.45 − 0.01 = 0.44 → 真实角 0.44", color=GREEN)
note(ax, 0.2, 0.45, "策略看到的关节角带着同一个偏置（第 15 章 15.3 节）：它说话用的一直是这把歪尺子。")
savefig(fig, "ch17_action_target")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("2. 固件位置环：误差 → 占空比 → 电压")
from bam.dynamixel import actuator as bam_dxl  # noqa: E402
from bam.model import load_model  # noqa: E402

bam_model = load_model(C.actuators._resolved_json_path)  # 项目用的那份：xl330 / m6
xl = bam_model.actuator
KE = (bam_dxl.XL330_ENCODER_COUNTS_PER_REV / (2 * math.pi)) / (bam_dxl.XL330_KP_DIVISOR * bam_dxl.XL330_PWM_LIMIT)
print(f"XL330：一圈 {bam_dxl.XL330_ENCODER_COUNTS_PER_REV} 格，固件除数 {bam_dxl.XL330_KP_DIVISOR}，满占空比 {bam_dxl.XL330_PWM_LIMIT}")
print(f"换算系数 k_e = (4096 / 2π) / (256 × 885) = {KE:.7f}；bam 里的 error_gain = {xl.error_gain:.7f}")
check("k_e = 0.0028774（计算器：4096 ÷ 6.2832 ÷ 256 ÷ 885 = 0.0028774）",
      math.isclose(KE, xl.error_gain) and round(KE, 7) == 0.0028774 and round(4096 / 6.2832 / 256 / 885, 7) == 0.0028774)
print(f"一格 = 360° / 4096 = {360 / 4096:.4f}°；1 rad = 4096 / 2π = {4096 / (2 * math.pi):.1f} 格")
check("1 rad ≈ 651.9 格编码器读数", round(4096 / (2 * math.pi), 1) == 651.9)

KP_FW = 200.0  # TWEAK-1: 400.0
print(f"固件 P 增益：项目 kp_fw = {C.actuators.kp_fw:g}；bam 给 XL330 的默认值 = {xl.kp:g}")
check("kp_fw = 200（bam 里 XL330 的默认值是 400）", C.actuators.kp_fw == 200.0 and xl.kp == 400)
VIN = 7.5                                                 # 名义电压（bam 里 XL330 的默认值），手算都用它
check("bam 里 XL330 的名义电压 7.5 V", xl.vin == VIN)
ERR = 0.44 - 0.40                                         # 17.1 的目标角 0.44，此刻关节在 0.40
duty_hand = ERR * KP_FW * 0.00288                         # 正文手算用三位有效数字的 0.00288
v_hand = VIN * duty_hand
print(f"手算：误差 {ERR:.2f} rad；占空比 = {ERR:.2f} × {KP_FW:g} × 0.00288 = {duty_hand:.5f}；电压 = {VIN} × {duty_hand:.5f} = {v_hand:.4f} V")
check("占空比 = 0.04 × 200 × 0.00288 = 0.02304（2.304%）", round(duty_hand, 5) == 0.02304)
check("电压 = 7.5 × 0.02304 = 0.1728 V（第 14–18 章手算复核里的同一个数）", round(v_hand, 4) == 0.1728)
check("字面值重算：0.04 × 200 × 0.00288 × 7.5 = 0.1728", round(0.04 * 200 * 0.00288 * 7.5, 4) == 0.1728)
xl.kp, xl.vin = KP_FW, VIN
v_bam = float(xl.compute_control(0.44, 0.40, 0.0, 0.005))
print(f"bam 自己的 compute_control（k_e 不取整）：{v_bam:.5f} V")
check("bam 算出的电压和手算只差 k_e 取整带来的零头（< 0.1%）", abs(v_bam - v_hand) / v_hand < 1e-3)


def firmware_voltage(err, vin=VIN, kp=None):
    """只看占空比上限的固件位置环（正文 17.2 的公式）：电压 = 电池电压 × clip(误差 × kp × k_e, −1, 1)。"""
    kp = KP_FW if kp is None else kp
    return vin * float(np.clip(err * kp * KE, -1.0, 1.0))


rows = []
for deg in (1, 5, 20):
    e = math.radians(deg)
    rows.append([deg, f"{e * KP_FW * KE:.4f}", f"{firmware_voltage(e):.4f}", f"{float(xl.compute_control(e, 0.0, 0.0, 0.005)):.4f}"])
table(["误差 (°)", "占空比", "电压 (V)", "舵机模型算的 (V)"], rows)
check("误差 1°、5°、20° → 电压 0.0753、0.3766、1.5066 V；舵机模型的代码算的一样（这么小的误差碰不到任何上限）",
      [r[2] for r in rows] == ["0.0753", "0.3766", "1.5066"] and all(r[2] == r[3] for r in rows))
FULL_DEG = 1 / (KP_FW * KE) * DEG
print(f"占空比到 1（电池电压全给上）要差 1 / (kp × k_e) = {1 / (KP_FW * KE):.4f} rad = {FULL_DEG:.1f}°")
check("误差 1 / (200 × 0.0028774) = 1.738 rad ≈ 99.6° 占空比才到 1", round(FULL_DEG, 1) == 99.6 and round(1 / (200 * 0.0028774), 3) == 1.738)
check("进阶：0.04 × 651.9 ≈ 26.08 格；26.08 × 200 ÷ 256 = 20.375；20.375 ÷ 885 ≈ 0.02302",
      round(0.04 * 651.9, 2) == 26.08 and 26.08 * 200 / 256 == 20.375 and round(20.375 / 885, 5) == 0.02302)
check("图 ①：开 3 成平均 7.5 × 0.3 = 2.25 V，开 6 成平均 4.5 V", round(7.5 * 0.3, 2) == 2.25 and round(7.5 * 0.6, 2) == 4.5)
check("自测（17.2）：目标 0.44、此刻 0.54 → 误差 −0.1 rad；电池 8.0 V → 占空比 −0.0576、电压 −0.4608 V", round(0.44 - 0.54, 2) == -0.1 and
      round(-0.1 * 200 * 0.00288, 4) == -0.0576 and round(-0.1 * 200 * 0.00288 * 8.0, 4) == -0.4608)

# ---------------------------------------------------------------------------
banner("2b. 画图：figures/ch17_duty_voltage.png（占空比 = 通电时间占几成；误差越大电压越高）")
fig = plt.figure(figsize=(9.6, 11.6))
fig.suptitle("固件位置环：差得越多，通电的时间越长", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
ax = fig.add_axes([0.12, 0.60, 0.66, 0.25])
data_axes(ax, "时间（示意，不按真实比例）", "电压（V）", grid=False)
t_pwm = np.linspace(0, 5, 2001)
for duty, color, y0 in ((0.3, BLUE, 0.0), (0.6, GREEN, 11.0)):
    wave = np.where((t_pwm % 1.0) < duty, VIN, 0.0)
    ax.plot(t_pwm, wave + y0, color=color, lw=2.4)
    ax.plot([0, 5], [VIN * duty + y0] * 2, color=ORANGE, lw=2.6, ls="--")
    ax.text(5.12, y0 + VIN - 0.6, f"占空比 {duty:g}", color=color, fontsize=FS_SMALL, va="center", fontweight="bold")
    ax.text(5.12, VIN * duty + y0 - 0.9, f"平均 {VIN:g} × {duty:g}\n= {VIN * duty:g} V", color=ORANGE, fontsize=FS_SMALL, va="center")
ax.set_xlim(0, 5)
ax.set_ylim(-0.6, 19.2)
ax.set_yticks([0, VIN, 11, 11 + VIN])
ax.set_yticklabels(["0", "7.5", "0", "7.5"])
ax.set_xticks([])
panel_title(fig, [ax], "① 芯片只会“开 / 关”：占空比 = 每一小段里开着的时间占几成")
panel_note(fig, [ax], "开关极快，电机跟不上，只感到平均电压 = 电池电压 × 占空比。")

ax = fig.add_axes([0.12, 0.12, 0.66, 0.30])
data_axes(ax, "误差：目标角 − 当前角（°）", "电压（V）")
e_deg = np.linspace(0, 120, 481)
v_line = [firmware_voltage(math.radians(d)) for d in e_deg]
ax.plot(e_deg, v_line, color=BLUE, lw=3)
ax.plot([FULL_DEG], [VIN], "o", color=BLUE, ms=9)
ax.annotate(f"差 {FULL_DEG:.1f}° 占空比到 1，电压封顶 {VIN:g} V", (FULL_DEG, VIN), (8, 7.6), fontsize=FS_SMALL,
            color=BLUE, va="center", arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.2))
for d, lab_x, lab_y in ((5, 14, 0.35), (20, 30, 1.2)):
    v = firmware_voltage(math.radians(d))
    ax.plot([d], [v], "o", color=ORANGE, ms=9, zorder=6)
    ax.annotate(f"{d}° → {v:.2f} V", (d, v), (lab_x, lab_y), color=ORANGE, fontsize=FS_SMALL, va="center",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.0))
ax.set_xlim(0, 120)
ax.set_ylim(0, 8.4)
ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
panel_title(fig, [ax], f"② 占空比 = 误差 × {KP_FW:g} × 0.00288，夹在 ±1 以内（电池 {VIN:g} V）")
panel_note(fig, [ax], f"正文手算的 2.3° 只给 {v_hand:.4f} V，在这张图里几乎贴着横轴。\n舵机不转的时候，电压其实到不了 {VIN:g} V：还有一道电流上限（17.3 节）。")
savefig(fig, "ch17_duty_voltage")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("3. 电机：电压 → 电流 → 力矩；转得越快，反电动势越大")
KT, R = bam_model.kt.value, bam_model.R.value
m6 = json.loads(Path(C.actuators._resolved_json_path).read_text(encoding="utf-8"))
print(f"bam 对 XL330 拟合的 m6 参数：k_t = {KT:.6f} N·m/A，R = {R:.6f} Ω（正文取三位：0.366、2.81）")
check("k_t ≈ 0.366 N·m/A、R ≈ 2.81 Ω（xl330/m6.json）", round(KT, 3) == 0.366 and round(R, 2) == 2.81 and m6["model"] == "m6")
kt, r = 0.366, 2.81                                       # 手算用的三位数
i_rest = 0.1728 / r
tau_rest = kt * round(i_rest, 6)
print(f"静止：电流 = 0.1728 ÷ 2.81 = {i_rest:.6f} A；力矩 = 0.366 × {i_rest:.6f} = {tau_rest:.6f} N·m")
check("静止时：电流 0.061495 A，力矩 0.022507 N·m", round(i_rest, 6) == 0.061495 and round(tau_rest, 6) == 0.022507)
emf = kt * 0.2
net = 0.1728 - emf
i_move = net / r
tau_move = kt * round(i_move, 6)
print(f"以 0.2 rad/s 转着：反电动势 = 0.366 × 0.2 = {emf:.4f} V；净电压 = 0.1728 − {emf:.4f} = {net:.4f} V；"
      f"电流 = {net:.4f} ÷ 2.81 = {i_move:.6f} A；力矩 = 0.366 × {i_move:.6f} = {tau_move:.6f} N·m")
check("反电动势 0.0732 V、净电压 0.0996 V、电流 0.035445 A、力矩 0.012973 N·m",
      round(emf, 4) == 0.0732 and round(net, 4) == 0.0996 and round(i_move, 6) == 0.035445 and round(tau_move, 6) == 0.012973)
two_terms = 0.366 * 0.1728 / 2.81 - 0.366 ** 2 * 0.2 / 2.81
print(f"拆开写：0.366 × 0.1728 / 2.81 − 0.366² × 0.2 / 2.81 = {0.366 * 0.1728 / 2.81:.6f} − {0.366 ** 2 * 0.2 / 2.81:.6f} = {two_terms:.6f}")
check("转着比不转少了四成多：1 − 0.012973 ÷ 0.022507 ≈ 0.42", round(1 - 0.012973 / 0.022507, 2) == 0.42)
check("两项相减：0.022507 − 0.009534 = 0.012973（手算复核里的 0.012972811）",
      round(0.366 ** 2 * 0.2 / 2.81, 6) == 0.009534 and round(two_terms, 9) == 0.012972811 and round(0.022507 - 0.009534, 6) == 0.012973)
check("字面值重算：0.1728 − 0.0732 = 0.0996；0.366 × 0.035445 = 0.012973", round(0.1728 - 0.0732, 4) == 0.0996
      and round(0.366 * 0.035445, 6) == 0.012973)

MAX_CURRENT = 1.75  # TWEAK-2: 3.5
xl.max_current = MAX_CURRENT
print(f"bam 里 XL330 的固件电流上限 max_current = {bam_model.actuator.max_current} A（项目没有改它）")
check("固件电流上限 1.75 A 是 bam 里 XL330 的默认值", load_model(C.actuators._resolved_json_path).actuator.max_current == 1.75)
const_text = Path(C.__file__).read_text(encoding="utf-8")
bam_src = Path(importlib.util.find_spec("bam").origin).parent
mjlab_bam = (bam_src / "mjlab.py").read_text(encoding="utf-8")
check("项目没改电流上限：执行器配置里没有 max_current 这一项，常数文件里那一行是注释掉的",
      "max_current" not in {f.name for f in dataclasses.fields(C.actuators)} and "    # max_current=1.75," in const_text)
check("bam/mjlab.py 的 compute() 直接调用 compute_control（电流上限就在那里），自己不碰 max_current",
      "control = act.compute_control(" in mjlab_bam and "max_current" not in mjlab_bam)
stall_formula = kt * VIN / r
no_load = VIN / kt
cap = kt * MAX_CURRENT
v_cap = MAX_CURRENT * r
knee = max(0.0, (VIN - round(v_cap, 2)) / kt)
print(f"满电压 7.5 V：公式的堵转力矩 0.366 × 7.5 ÷ 2.81 = {stall_formula:.3f} N·m；空载最高转速 7.5 ÷ 0.366 = {no_load:.2f} rad/s")
print(f"电流上限：力矩最多 0.366 × {MAX_CURRENT:g} = {cap:.4f} N·m；静止时要推出 {MAX_CURRENT:g} A 得给 {MAX_CURRENT:g} × 2.81 = {v_cap:.4f} V；"
      + (f"转速超过 (7.5 − {v_cap:.2f}) ÷ 0.366 = {knee:.2f} rad/s 后，满电压也推不出 {MAX_CURRENT:g} A" if v_cap < VIN
         else f"比电池的 {VIN:g} V 还高：这个上限碰不到，力矩就是公式值"))
check("公式的堵转力矩 0.977 N·m、空载最高转速 20.49 rad/s；转速每快 1 rad/s 少 0.0477 N·m", round(stall_formula, 3) == 0.977
      and round(no_load, 2) == 20.49 and round(0.366 ** 2 / 2.81, 4) == 0.0477 and round(KT ** 2 / R, 4) == 0.0477)
check("电流上限 1.75 A → 力矩封顶 0.6405 N·m；静止时电压最多约 4.92 V；拐点约 7.05 rad/s",
      round(cap, 4) == 0.6405 and round(v_cap, 2) == 4.92 and round(knee, 2) == 7.05)
v_big = float(xl.compute_control(math.radians(120), 0.0, 0.0, 0.005))
tau_big = float(xl.compute_torque(v_big, True, 0.0, 0.0))
print(f"bam：误差 120°、静止 → 电压 {v_big:.4f} V（电池 {VIN:g} V），力矩 {tau_big:.4f} N·m（公式值 {KT * VIN / R:.4f}）")
check(f"bam 的 compute_control 把静止时的电压限在 {MAX_CURRENT:g} × R = {MAX_CURRENT * R:.4f} V，力矩 = k_t × {MAX_CURRENT:g} = {KT * MAX_CURRENT:.4f}",
      math.isclose(v_big, min(VIN, MAX_CURRENT * R), rel_tol=1e-6) and math.isclose(tau_big, min(KT * VIN / R, KT * MAX_CURRENT), rel_tol=1e-6))
check("按 bam 的精确参数：力矩封顶 0.6405 N·m（堵转力矩约 0.64，不是 0.98）", round(KT * 1.75, 4) == 0.6405)
v5 = float(xl.compute_control(math.radians(170), 0.0, 5.0, 0.005))
print(f"5 rad/s、全力推：反电动势 0.366 × 5 = 1.83 V；要推出 {MAX_CURRENT:g} A 得给 1.83 + {v_cap:.2f} = {1.83 + round(v_cap, 2):.2f} V"
      f"（电池 {VIN:g} V）；bam 算出的电压 {v5:.4f} V")
check("5 rad/s：反电动势 1.83 V，固件给 1.83 + 4.92 = 6.75 V（没到 7.5 V），净剩 4.92 V → 仍是 1.75 A、0.64 N·m",
      round(0.366 * 5, 2) == 1.83 and round(1.83 + 4.92, 2) == 6.75 and round(v5, 2) == 6.75 and v5 < VIN
      and round(float(xl.compute_torque(v5, True, 0.0, 5.0)), 4) == 0.6405)


def bam_max_torque(speed, vin=VIN):
    """用 bam 自己的两个函数：误差很大（全力推）、正以 speed 转着时，电机能出多少力矩。"""
    xl.vin = vin
    v = float(xl.compute_control(math.radians(170), 0.0, speed, 0.005))
    xl.vin = VIN
    return float(xl.compute_torque(v, True, 0.0, speed))


rows = [[f"{s:g}", f"{KT * VIN / R - KT ** 2 * s / R:.4f}", f"{bam_max_torque(s):.4f}"] for s in (0, 5, 10, 15, round(VIN / KT, 2))]
table(["转速 (rad/s)", "只按公式 (N·m)", "舵机模型：加上电流上限 (N·m)"], rows)
check("转速 0、5 时 bam 封顶 0.6405；10、15 时两列相同（0.4999、0.2617）；20.49 时降到 0",
      rows[0][2] == rows[1][2] == "0.6405" and rows[2][1] == rows[2][2] == "0.4999" and rows[3][2] == "0.2617" and rows[4][2] == "0.0000")
check("自测：满电压、10 rad/s → 净电压 3.84 V、电流 1.3665 A（没到上限）、力矩约 0.500 N·m",
      round(7.5 - 0.366 * 10, 2) == 3.84 and round(3.84 / 2.81, 4) == 1.3665 and round(0.366 * 1.3665, 3) == 0.500)

# ---------------------------------------------------------------------------
banner("3b. 画图：figures/ch17_torque_speed.png（反电动势吃掉一截电压；力矩–转速线和电流上限）")
fig = plt.figure(figsize=(9.6, 12.4))
fig.suptitle("电机：转得越快，推得越弱；电流上限再削平顶部", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
ax = fig.add_axes([0.04, 0.60, 0.92, 0.29])
lesson_panel(ax, "", xmax=10, ymax=5)
SC = 30.0                                                 # 1 V 画成 30 格：三根条按同一比例
for y, val, color, label in ((3.9, 0.1728, BLUE, "外加电压"), (2.7, emf, ORANGE, "反电动势"), (1.5, net, GREEN, "净电压")):
    x0 = 2.3 if label != "反电动势" else 2.3 + SC * net
    ax.add_patch(plt.Rectangle((x0, y - 0.32), SC * val, 0.64, facecolor=color, alpha=0.85, edgecolor="none"))
    ax.text(0.1, y, label, fontsize=FS_STEP, color=color, va="center")
    ax.text(x0 + SC * val + 0.15, y, f"{val:.4f} V", fontsize=FS_STEP, color=color, va="center")
ax.plot([2.3 + SC * net] * 2, [0.9, 4.4], color=MUTED, lw=1.2, ls=":")
hand(ax, 0.1, 0.45, f"电流 = 0.0996 ÷ 2.81 = 0.035445 A；力矩 = 0.366 × 0.035445 = {tau_move:.6f} N·m", fontsize=FS_SMALL)
panel_title(fig, [ax], "① 以 0.2 rad/s 转着：反电动势 0.366 × 0.2 = 0.0732 V，先从电压里扣掉")
panel_note(fig, [ax], "三根条按同一比例画：橙条正好是“外加”比“净”长出的那一截。")

ax = fig.add_axes([0.12, 0.10, 0.80, 0.33])
data_axes(ax, "关节转速（rad/s）", "电机力矩（N·m）")
spd = np.linspace(0, VIN / KT, 400)
formula = KT * VIN / R - KT ** 2 * spd / R
real = np.minimum(formula, KT * MAX_CURRENT)
ax.fill_between(spd, 0, real, color=CELL, zorder=1)
ax.plot(spd, formula, color=FAINT, lw=2.4, ls="--", zorder=2)
ax.plot(spd, real, color=BLUE, lw=3.2, zorder=3)
ax.plot([0], [KT * VIN / R], "o", color=MUTED, ms=8)
capped = KT * MAX_CURRENT < KT * VIN / R                  # 电流上限碰得到吗？（“改一改”第 2 条把它调到碰不到）
ax.text(0.5, KT * VIN / R + 0.035, f"只按公式：静止时约 {KT * VIN / R:.2f}" + ("（真舵机到不了）" if capped else ""),
        color=MUTED, fontsize=FS_SMALL)
if capped:
    ax.text(0.5, KT * MAX_CURRENT - 0.105, f"电流上限 {KT * MAX_CURRENT:.2f} N·m", color=BLUE, fontsize=FS_SMALL)
    k_real = (VIN - MAX_CURRENT * R) / KT
    ax.plot([k_real], [KT * MAX_CURRENT], "o", color=ORANGE, ms=9, zorder=5)
    ax.annotate(f"拐点 {k_real:.2f} rad/s", (k_real, KT * MAX_CURRENT), (9.2, 0.78), color=ORANGE, fontsize=FS_SMALL,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.2))
ax.plot([VIN / KT], [0], "o", color=GREEN, ms=9, zorder=5)
ax.annotate(f"空载最高转速\n{VIN / KT:.2f} rad/s", (VIN / KT, 0), (15.2, 0.36), color=GREEN, fontsize=FS_SMALL, va="bottom",
            arrowprops=dict(arrowstyle="-", color=GREEN, lw=1.2))
ax.text(4.2, 0.2, "能用的范围", color=MUTED, fontsize=FS_SMALL)
ax.set_xlim(0, 22)
ax.set_ylim(0, 1.1)
ax.set_xticks([0, 5, 10, 15, 20])
panel_title(fig, [ax], "② 电池 7.5 V、全力推：每个转速下最多能出多少力矩")
panel_note(fig, [ax], "灰虚线：力矩 = 0.366 × 7.5 ÷ 2.81 − 0.366² × 转速 ÷ 2.81，斜着往下走的就是反电动势。\n"
                      + ("蓝线：项目的舵机模型，先被电流上限削平，再沿着灰线往下。" if capped else
                         f"蓝线：项目的舵机模型。电流上限 {MAX_CURRENT:g} A 碰不到（电池最多推出 {VIN:g} ÷ 2.81 ≈ {VIN / 2.81:.2f} A），和灰线重合。"))
savefig(fig, "ch17_torque_speed")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("3c. 画图：figures/ch17_overview.png（17.0 节的总览图）")
fig, axes = lesson_figure(3, "一个数 → 一个角 → 一股力；一台机器人 → 4096 台", panel_height=3.3, width=9.8)
ax = axes[0]
lesson_panel(ax, "① 动作 → 目标角：加上站姿（17.1 节）", xmax=10, ymax=4)
lesson_cells(ax, [["动作 0.10", "+ 站姿 0.35", "− 偏置 0.01", "= 目标角 0.44"]], left=0.1, bottom=1.55, width=2.4, height=0.9,
             highlights=[(0, 3)], fontsize=FS_SMALL + 1)
note(ax, 0.1, 0.8, "策略吐出的数是“比站姿多转多少”，单位 rad。")
ax = axes[1]
lesson_panel(ax, "② 目标角 → 电压 → 力矩：舵机里的两站（17.2–17.6 节）", xmax=10, ymax=4)
boxes = [("0.44 − 0.40 = 0.04 rad", 0.1, 3.3), (f"{v_hand:.4f} V", 4.25, 2.4), (f"{kt * v_hand / r:.4f} N·m", 7.55, 2.4)]
for text, x, w in boxes:
    cell(ax, x, 1.75, text, width=w, height=0.85, fontsize=FS_SMALL + 1)
for x0, x1, label in ((3.4, 4.25, "固件位置环"), (6.65, 7.55, "电机")):
    arrow(ax, (x0, 2.17), (x1, 2.17), ORANGE, lw=2.5)
    ax.text((x0 + x1) / 2, 2.75, label, color=ORANGE, fontsize=FS_SMALL, ha="center")
note(ax, 0.1, 1.05, "再过摩擦、电池两关；目标角还要晚 15–30 ms 才送到舵机。")
ax = axes[2]
lesson_panel(ax, "③ 一台 → 4096 台：每只机器人抽一套自己的身体（17.7–17.8 节）", xmax=10, ymax=4)
rng_demo = np.random.default_rng(17)                      # 和第 7 节同一个种子、同样的抽法：这 5 只就是 ch17_dr_robots 图 ① 的 1–5 号
rng_demo.uniform(*C.actuators.vin_range, 4096)            # 第 7 节先给 4096 只各抽一块（画直方图），再抽这 5 只
vins = rng_demo.uniform(*C.actuators.vin_range, 5)
for k, v in enumerate(vins):
    cell(ax, 0.1 + 1.72 * k, 1.75, f"{v:.2f} V", width=1.52, height=0.85, fontsize=FS_SMALL)
ax.text(8.85, 2.17, "… 共 4096 只", color=INK, fontsize=FS_SMALL, va="center")
note(ax, 0.1, 1.05, "例如电池电压：每只开机时在 6.5–8.2 V 里抽一个（这 5 只就是 17.7 节图里的 1–5 号）。")
savefig(fig, "ch17_overview")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("4. 执行器与 BAM：再加上摩擦和电池")
print(f"项目的执行器：{type(C.actuators).__name__}（bam 的执行器 + 每只机器人一个摩擦倍率），模型 {C.actuators.motor_name}/{C.actuators.model}")
check("执行器是 FrictionDRBamActuatorCfg，模型 xl330 / m6", type(C.actuators).__name__ == "FrictionDRBamActuatorCfg"
      and C.actuators.motor_name == "xl330" and C.actuators.model == "m6")
FB, FS_, LFM, VISC = m6["friction_base"], m6["friction_stribeck"], m6["load_friction_motor"], m6["friction_viscous"]
DTHETA, ALPHA_S, LFMS = m6["dtheta_stribeck"], m6["alpha"], m6["load_friction_motor_stribeck"]
print(f"m6 摩擦：库仑 {FB:.5f}、低速额外 {FS_:.5f} N·m；负载系数 {LFM:.4f}；粘滞 {VISC:.5f} N·m/(rad/s)")
check("m6 的四个摩擦常数：固定 0.00477、低速额外 0.00468 N·m，负载系数 0.267，粘滞 0.00536 N·m/(rad/s)",
      round(FB, 5) == 0.00477 and round(FS_, 5) == 0.00468 and round(LFM, 3) == 0.267 and round(VISC, 5) == 0.00536)
check("BAM 有 m1 到 m6 六种摩擦模型（bam 包里 xl330 的六份参数），项目用 m6",
      all((Path(C.actuators._resolved_json_path).parent / f"m{k}.json").is_file() for k in range(1, 7)))


def stribeck(speed):
    """低速额外那一份的开关：很慢时≈1，快了≈0（bam 的 exp(−(|v|/dθ)^α)）。"""
    return math.exp(-(abs(speed) / DTHETA) ** ALPHA_S)


def friction_budget(tau_motor, speed, scale=1.0):
    """bam m6 的摩擦预算（外部力矩取 0：单关节、没有重力；二次项在外部力矩为 0 时也是 0）。"""
    s = stribeck(speed)
    return scale * (FB + s * FS_ + abs(tau_motor) * LFM + s * abs(tau_motor) * LFMS)


print("低速额外那一份的开关：" + "，".join(f"{v:g} rad/s → {stribeck(v):.3f}" for v in (1, 2, 3.5)))
check("低速额外那一份：2 rad/s 时还有 96%，3.5 rad/s 时几乎为 0", round(stribeck(2), 2) == 0.96 and stribeck(3.5) < 0.01)
fb_hand = 0.00477 + 0.00468 + 0.267 * 0.1
print(f"手算：电机出 0.1 N·m、几乎不转：0.00477 + 0.00468 + 0.267 × 0.1 = {fb_hand:.5f} ≈ {fb_hand:.3f} N·m；bam 的式子：{friction_budget(0.1, 0.0):.6f}")
check("摩擦预算 ≈ 0.00945 + 0.0267 = 0.03615 ≈ 0.036 N·m（bam 的式子 0.0361）；电机的 0.1 被吃掉三成多", round(0.03615 / 0.1, 2) == 0.36
      and round(0.00477 + 0.00468, 5) == 0.00945
      and round(fb_hand, 5) == 0.03615 and round(friction_budget(0.1, 0.0), 3) == 0.036)
FRIC_RANGE = cfg.events["randomize_joint_friction"].params["scale_range"]
print(f"摩擦倍率 friction_scale 每回合在 {FRIC_RANGE} 里抽：预算 × 0.9 … × 1.1（粘滞那一份不乘）")
check("friction_scale 每回合抽 U(0.9, 1.1)", FRIC_RANGE == (0.9, 1.1) and cfg.events["randomize_joint_friction"].mode == "reset")
src_fdr = inspect.getsource(importlib.import_module("mjlab_microduck.actuator.friction_dr_bam"))
check("倍率乘在 bam 的摩擦预算上：base * fs", "return base if fs is None else base * fs" in src_fdr)
v_low, v_high = 0.02304 * 6.5, 0.02304 * 8.2
print(f"同样差 0.04 rad：电池 6.5 V → {v_low:.5f} V；8.2 V → {v_high:.6f} V；8.2 ÷ 6.5 = {8.2 / 6.5:.2f}")
check("同一个误差：6.5 V 给 0.14976 V，8.2 V 给 0.188928 V，差 1.26 倍", round(v_low, 5) == 0.14976
      and round(v_high, 6) == 0.188928 and round(8.2 / 6.5, 2) == 1.26)
print(f"电池：vin_range = {C.actuators.vin_range}，压降增益 = {C.actuators.vin_drop_gain_range} V/(N·m)，地板 vin_min = {C.actuators.vin_min} V")
check("电池 U(6.5, 8.2)、压降增益 U(0, 0.2)、最低 6.0 V", C.actuators.vin_range == (6.5, 8.2)
      and C.actuators.vin_drop_gain_range == (0.0, 0.2) and C.actuators.vin_min == 6.0)


def sagged(vin, gain, load):
    """有效电压 = max(电池电压 − 压降增益 × 所有舵机力矩大小之和, 地板)。"""
    return max(vin - gain * load, C.actuators.vin_min)


print(f"压降：7.0 − 0.15 × 2.0 = {sagged(7.0, 0.15, 2.0):.1f} V；6.5 − 0.15 × 4.0 = {6.5 - 0.15 * 4.0:.1f} → 地板 {sagged(6.5, 0.15, 4.0):.1f} V")
check("压降手算：7.0 − 0.15 × 2.0 = 6.7 V；6.5 − 0.15 × 4.0 = 5.9 V 低于地板，按 6.0 V 算", round(sagged(7.0, 0.15, 2.0), 1) == 6.7
      and round(6.5 - 0.15 * 4.0, 1) == 5.9 and sagged(6.5, 0.15, 4.0) == 6.0)
check("自测（17.4）：差 0.05 rad → 占空比 0.0288；6.5 V 给 0.1872 V，8.2 V 给 0.23616 V",
      round(0.05 * 200 * 0.00288, 4) == 0.0288 and round(0.0288 * 6.5, 4) == 0.1872 and round(0.0288 * 8.2, 5) == 0.23616)
check("电池电压和压降增益开机抽一次、reset 不重抽；压降用的是上一个物理步的力矩",
      "vin_tensor and vin_drop_gain are startup-randomized: do NOT re-sample on reset." in mjlab_bam
      and "load = self._prev_motor_torque.abs().sum(dim=-1, keepdim=True)" in mjlab_bam)
print(f"执行器每个物理步算一次：物理步 {cfg.sim.mujoco.timestep} s，一个环境步 {cfg.decimation} 个物理步")
check("物理步 0.005 s，decimation = 4", cfg.sim.mujoco.timestep == 0.005 and cfg.decimation == 4)

# ---------------------------------------------------------------------------
banner("5. 延迟：目标角晚 3–6 个物理步才到")
A = C.actuators
print(f"执行器延迟：delay_min_lag = {A.delay_min_lag}，delay_max_lag = {A.delay_max_lag}，"
      f"每隔 {A.delay_update_period} 步重抽（0 = 每步），保持原值的概率 {A.delay_hold_prob}")
check("执行器延迟 3–6 个物理步，每个物理步重抽一次", (A.delay_min_lag, A.delay_max_lag) == (3, 6)
      and A.delay_update_period == 0 and A.delay_hold_prob == 0.0)
PHYS = cfg.sim.mujoco.timestep
ENV_DT = PHYS * cfg.decimation
print(f"3 × {PHYS} = {3 * PHYS:.3f} s = {3 * PHYS * 1000:g} ms；6 × {PHYS} = {6 * PHYS:.3f} s = {6 * PHYS * 1000:g} ms")
check("3 个物理步 = 15 ms，6 个物理步 = 30 ms", round(3 * 0.005 * 1000) == 15 and round(6 * 0.005 * 1000) == 30
      and math.isclose(3 * PHYS * 1000, 15) and math.isclose(6 * PHYS * 1000, 30))
check("要是误当成环境步：3 × 0.02 = 60 ms，6 × 0.02 = 120 ms——差了 4 倍",
      round(3 * 0.02 * 1000) == 60 and round(6 * 0.02 * 1000) == 120 and round(ENV_DT / PHYS) == 4)
jv = cfg.observations["actor"].terms["joint_vel"]
print(f"对比第 15 章：关节速度的观测固定晚 {jv.delay_min_lag} 个环境步 = {jv.delay_min_lag * ENV_DT * 1000:g} ms")
check("观测延迟按环境步数：关节速度晚 1 步 = 20 ms", jv.delay_min_lag == jv.delay_max_lag == 1 and math.isclose(ENV_DT * 1000, 20))
from mjlab.utils.buffers.delay_buffer import DelayBuffer  # noqa: E402

WRITES = [1, 1, 1, 1, 1]
for lag in (3, 0):
    buf = DelayBuffer(min_lag=lag, max_lag=lag, batch_size=1, device="cpu")
    buf.append(torch.tensor([[0.0]]))                    # 旧目标 0 已经在队里
    buf.compute()
    got = []
    for w in WRITES:
        buf.append(torch.tensor([[float(w)]]))
        got.append(int(buf.compute()[0, 0]))
    print(f"lag = {lag}：写入 {WRITES} → 到达 {got}")
    check(f"lag = {lag}：到达的是 {'0 0 0 1 1' if lag == 3 else '1 1 1 1 1'}", got == ([0, 0, 0, 1, 1] if lag == 3 else [1] * 5))
lags = torch.randint(A.delay_min_lag, A.delay_max_lag + 1, (100_000,), generator=torch.Generator().manual_seed(0))
share = [float((lags == k).float().mean()) for k in range(3, 7)]
print("每步重抽 10 万次，3/4/5/6 各占：" + "、".join(f"{s:.3f}" for s in share) + f"；平均 {float(lags.float().mean()):.3f} 步")
check("自测（17.5）：执行器 2–8 步 = 10–40 ms；观测的 8 步 = 160 ms", 2 * 5 == 10 and 8 * 5 == 40 and 8 * 20 == 160)
check("3、4、5、6 四个数一样可能（各约 25%），平均约 4.5 步 ≈ 22.5 ms",
      all(abs(s - 0.25) < 0.01 for s in share) and abs(float(lags.float().mean()) - 4.5) < 0.02 and 4.5 * 5 == 22.5)

# ---------------------------------------------------------------------------
banner("5b. 画图：figures/ch17_delay_clocks.png（两只钟：物理步 5 ms，环境步 20 ms）")
fig = plt.figure(figsize=(9.8, 8.6))
fig.suptitle("两只钟：执行器延迟数物理步，观测延迟数环境步", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
ax = fig.add_axes([0.10, 0.52, 0.86, 0.30])
data_axes(ax, "时间（ms）", "", grid=False)
ax.set_xlim(-2, 62)
ax.set_ylim(-0.4, 3.2)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xticks(range(0, 61, 10))
for t in range(0, 61, 5):
    ax.plot([t, t], [0.3, 0.75], color=GREEN, lw=2.2 if t % 20 == 0 else 1.4)
for t in range(0, 61, 20):
    ax.plot([t, t], [1.9, 2.55], color=BLUE, lw=3)
ax.plot([0, 60], [0.3, 0.3], color=GREEN, lw=1.4)
ax.plot([0, 60], [1.9, 1.9], color=BLUE, lw=1.4)
ax.text(0, 2.75, "环境步：每 20 ms 一格（策略做一次决定）", color=BLUE, fontsize=FS_SMALL)
ax.text(0, 0.95, "物理步：每 5 ms 一格（执行器算一次）", color=GREEN, fontsize=FS_SMALL)
panel_title(fig, [ax], "① 同一段 60 ms：环境步 3 格，物理步 12 格")
panel_note(fig, [ax], "一个环境步 = 4 个物理步（第 15 章 15.1 节）。")

ax = fig.add_axes([0.10, 0.08, 0.86, 0.25])
data_axes(ax, "从策略写下新目标角算起的时间（ms）", "", grid=False)
ax.set_xlim(-2, 62)
ax.set_ylim(-0.3, 2.6)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xticks([0, 15, 20, 30, 40, 60])
ax.axvspan(3 * PHYS * 1000, 6 * PHYS * 1000, ymin=0.52, ymax=0.9, color=CELL_HOT, zorder=0)
ax.text(22.5, 1.95, "目标角到达舵机：15–30 ms", color=ORANGE, fontsize=FS_SMALL, ha="center", bbox=WHITE_BOX)
for k in range(3, 7):
    ax.plot([k * PHYS * 1000], [1.55], "o", color=ORANGE, ms=10)
    ax.text(k * PHYS * 1000, 1.18, f"{k}", color=ORANGE, fontsize=FS_SMALL, ha="center")
ax.plot([0], [1.55], "D", color=INK, ms=9)
ax.text(0.8, 1.05, "写下\n新目标角", color=INK, fontsize=FS_SMALL - 1, va="center")
ax.plot([ENV_DT * 1000], [0.35], "s", color=BLUE, ms=11)
ax.text(ENV_DT * 1000 + 1.5, 0.35, "关节速度观测：晚 1 个环境步 = 20 ms", color=BLUE, fontsize=FS_SMALL, va="center")
panel_title(fig, [ax], "② 延迟 3–6 个物理步 = 15–30 ms，不是 60–120 ms")
panel_note(fig, [ax], "橙点下的数是晚了几个物理步：每个物理步从 3、4、5、6 里重抽一个。")
savefig(fig, "ch17_delay_clocks")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("6. 连起来：一个 20° 的目标，关节怎么追（简化的单关节）")
INERTIA = xl.get_extra_inertia() + 0.0005                 # 转子折算惯量 + 一点负载（教学取值）
STEPS, TARGET = 120, math.radians(20)                     # 0.6 s；目标从 0 跳到 20°
DELAY_STEPS = 6  # TWEAK-3: 24
print(f"转子折算惯量 {xl.get_extra_inertia():.6f} + 负载 0.0005 = {INERTIA:.6f} kg·m²；每步 {PHYS} s，共 {STEPS} 步")


def step_response(vin=VIN, delay=0, friction=False):
    """单关节追 20°：电压和力矩都用 bam 自己的 compute_control / compute_torque；friction=True 时加上 m6 摩擦（推不动就停着）。

    简化之处：积分和“推不动就停着”是这里自己写的，不是 MuJoCo 的约束求解；摩擦预算用这一步的电机力矩（bam 的 mjlab 版用上一步的）。
    """
    xl.vin = vin
    q, v, out = 0.0, 0.0, []
    queue = [0.0] * delay                                  # 第 15 章 15.4 节的队列：先排着 delay 个旧目标 0
    for _ in range(STEPS):
        queue.append(TARGET)
        cmd = queue.pop(0)                                 # delay = 0 时当步就到
        tau = float(xl.compute_torque(float(xl.compute_control(cmd, q, v, PHYS)), True, q, v))
        if friction:
            budget = friction_budget(tau, v)
            if v == 0.0 and abs(tau) <= budget:
                out.append(q * DEG)                        # 推力没超过摩擦预算：停着不动
                continue
            sign = math.copysign(1.0, v if v != 0.0 else tau)
            v_new = v + (tau - sign * budget - VISC * v) / INERTIA * PHYS
            v = 0.0 if (v != 0.0 and v_new * v < 0 and abs(tau) <= budget) else v_new
        else:
            v += tau / INERTIA * PHYS
        q += v * PHYS
        out.append(q * DEG)
    xl.vin = VIN
    return np.array(out)


t_ms = (np.arange(STEPS) + 1) * PHYS                      # 每一步积分完之后的时刻
CASES = [("电池 7.5 V", dict()), ("电池 6.5 V", dict(vin=6.5)), (f"7.5 V + 延迟 {DELAY_STEPS} 步", dict(delay=DELAY_STEPS)),
         ("7.5 V + m6 摩擦", dict(friction=True))]
curves = {name: step_response(**kw) for name, kw in CASES}


def first_time(curve, level):
    hit = np.nonzero(curve >= level)[0]
    return float(t_ms[hit[0]]) if len(hit) else float("nan")


rows = []
for name, _ in CASES:
    c = curves[name]
    rows.append([name, f"{c[3]:.2f}", f"{c[19]:.1f}", f"{first_time(c, 18.0):.3f}", f"{c.max():.1f}", f"{c[-1]:.1f}"])
table(["情形", "0.02 s (°)", "0.1 s (°)", "到 18° (s)", "最高 (°)", "0.6 s (°)"], rows)
ideal, low, lagged, fric = (curves[n] for n, _ in CASES)
check("一个环境步（0.02 s）之后，理想情形只追了 1.09°", round(ideal[3], 2) == 1.09)
check("理想情形：0.1 s 时 12.0°，0.160 s 到 18°，最高冲到 21.1°，0.6 s 时 20.0°", round(ideal[19], 1) == 12.0
      and round(first_time(ideal, 18.0), 3) == 0.16 and round(ideal.max(), 1) == 21.1 and round(ideal[-1], 1) == 20.0)
check("电池 6.5 V：更软——0.1 s 时 10.7°，0.185 s 才到 18°，最高 20.7°", round(low[19], 1) == 10.7
      and round(first_time(low, 18.0), 3) == 0.185 and round(low.max(), 1) == 20.7 and first_time(low, 18.0) > first_time(ideal, 18.0))
check(f"延迟 {DELAY_STEPS} 步：整条曲线原样往后挪 {DELAY_STEPS} 步（{DELAY_STEPS * PHYS * 1000:g} ms），形状不变",
      np.all(lagged[:DELAY_STEPS] == 0) and np.allclose(lagged[DELAY_STEPS:], ideal[:STEPS - DELAY_STEPS]))
check("延迟 6 步 = 30 ms：到 18° 从 0.160 s 变成 0.190 s", round(first_time(lagged, 18.0), 3) == 0.19)
check("6.5 V 比 7.5 V 晚 0.185 − 0.160 = 0.025 s 到 18°", round(first_time(low, 18.0) - first_time(ideal, 18.0), 3) == 0.025)
check("进阶手算：0.00945 ÷ 0.733 ≈ 0.0129；200 × 0.00288 × 7.5 × 0.366 ÷ 2.81 ≈ 0.563；0.0129 ÷ 0.563 ≈ 0.0229 rad ≈ 1.31°",
      round(0.00945 / 0.733, 4) == 0.0129 and round(200 * 0.00288 * 7.5, 2) == 4.32 and round(4.32 * 0.366 / 2.81, 3) == 0.563
      and round(200 * 0.00288 * 7.5 * 0.366 / 2.81, 3) == 0.563
      and round(0.0129 / 0.563, 4) == 0.0229 and round(0.0229 * DEG, 2) == 1.31 and round(1 - LFM, 3) == 0.733)
dead = friction_budget(0, 0) / (KP_FW * KE * VIN * KT / R * (1 - LFM - LFMS)) * DEG
print(f"摩擦：静止时推力 ≤ 预算的误差范围 ≈ {dead:.2f}°（小于它就推不动）；最后停在 {fric[-1]:.1f}°，差 {20 - fric[-1]:.1f}°")
check("m6 摩擦：更慢（0.1 s 时 9.3°，0.220 s 才到 18°），停在 19.3°——差 0.7°，在推不动的范围（约 1.31°）以内",
      round(fric[19], 1) == 9.3 and round(first_time(fric, 18.0), 3) == 0.22 and round(fric[-1], 1) == 19.3
      and 0 < 20 - fric[-1] <= round(dead, 2) and round(dead, 2) == 1.31)
check("自测：电池 8.2 V 更硬——到 18° 更早、冲得更高", first_time(step_response(vin=8.2), 18.0) <= first_time(ideal, 18.0)
      and step_response(vin=8.2).max() > ideal.max())

# ---------------------------------------------------------------------------
banner("6b. 画图：figures/ch17_bam_step.png（同一个 20° 目标，四种情形）")
fig = plt.figure(figsize=(9.8, 12.6))
fig.suptitle("同一个 20° 目标（简化的单关节）：电压、延迟、摩擦各改一样", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
STYLE = {CASES[0][0]: (BLUE, "-"), CASES[1][0]: (GREEN, "-"), CASES[2][0]: (ORANGE, "--"), CASES[3][0]: (PURPLE, "-")}
ax = fig.add_axes([0.12, 0.56, 0.84, 0.31])
data_axes(ax, "时间（s）", "关节角（°）")
for name, _ in CASES:
    color, ls = STYLE[name]
    ax.plot(t_ms, curves[name], color=color, ls=ls, lw=2.8, label=name)
ax.axhline(20, color=MUTED, lw=1.2, ls=":")
ax.text(0.605, 20, "目标 20°", color=MUTED, fontsize=FS_SMALL, va="center")
ax.set_xlim(0, 0.6)
ax.set_ylim(0, 23.5)
ax.legend(fontsize=FS_SMALL - 2, loc="lower right", frameon=False)
panel_title(fig, [ax], "① 0 到 0.6 s：四条线最后都停在 20° 附近")
panel_note(fig, [ax], f"紫线（摩擦）最慢，而且停在 {fric[-1]:.1f}°：差的那一点，推力不够抵过摩擦。")

ax = fig.add_axes([0.12, 0.10, 0.84, 0.30])
data_axes(ax, "时间（s）", "关节角（°）")
for name, _ in CASES:
    color, ls = STYLE[name]
    ax.plot(t_ms, curves[name], color=color, ls=ls, lw=2.8, marker="o", ms=4)
ax.axvline(ENV_DT, color=MUTED, lw=1.2, ls=":")
ax.text(ENV_DT + 0.002, 16.5, "一个环境步\n0.02 s", color=MUTED, fontsize=FS_SMALL)
ax.annotate(f"{ideal[3]:.2f}°", (ENV_DT, ideal[3]), (0.004, 5.5), color=BLUE, fontsize=FS_SMALL,
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.2))
t_lvl = float(np.interp(8.0, ideal, t_ms))              # 蓝线到 8° 的时刻；橙线晚 DELAY_STEPS 步到同一高度
if t_lvl + DELAY_STEPS * PHYS <= 0.12:
    for tt in (t_lvl, t_lvl + DELAY_STEPS * PHYS):
        ax.plot([tt, tt], [8.0, 16.0], color=ORANGE, lw=1.2, ls=":")
        ax.plot([tt], [8.0], "o", color=ORANGE, ms=7, zorder=6)
    ax.annotate("", (t_lvl + DELAY_STEPS * PHYS, 16.0), (t_lvl, 16.0), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2))
    ax.text(t_lvl + DELAY_STEPS * PHYS / 2, 16.7, f"同到 8°：晚 {DELAY_STEPS * PHYS * 1000:g} ms", color=ORANGE, fontsize=FS_SMALL,
            ha="center")
ax.set_xlim(0, 0.12)
ax.set_ylim(0, 22)
panel_title(fig, [ax], "② 放大前 0.12 s：每个点是一个物理步（5 ms）")
panel_note(fig, [ax], "橙虚线是蓝线原样往右挪 30 ms：延迟只让一切晚到，不改舵机自己追的样子。")
savefig(fig, "ch17_bam_step")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("7. 域随机化：从配置里读出每一项的范围和抽样时机")
EV = cfg.events
MODE_ZH = {"startup": "开机一次", "reset": "每回合", "interval": "每隔几秒"}
for name in ("foot_friction", "encoder_bias", "randomize_mass_inertia", "randomize_com", "randomize_head_com",
             "randomize_joint_friction", "randomize_armature", "push_robot", "base_com"):
    e = EV[name]
    p = {k: v for k, v in e.params.items() if k != "asset_cfg"}
    print(f"  {name:26s} {MODE_ZH[e.mode]:6s} {getattr(e.func, '__name__', e.func)}  {p}"
          + (f"  每 {e.interval_range_s} s" if e.mode == "interval" else ""))
alpha = EV["randomize_mass_inertia"].params["alpha_range"]
mass_lo, mass_hi = math.exp(2 * alpha[0]), math.exp(2 * alpha[1])
imu_deg = cfg.observations["actor"].terms["base_ang_vel"].params["max_angle_deg"]
checks = [
    ("电池 6.5–8.2 V、压降增益 0–0.2：开机一次（bam 执行器）", C.actuators.vin_range == (6.5, 8.2)),
    ("脚底摩擦 0.7–1.3：开机一次，两只脚抽同一个数", EV["foot_friction"].mode == "startup"
     and EV["foot_friction"].params["ranges"] == (0.7, 1.3) and EV["foot_friction"].params["shared_random"]),
    ("编码器偏置 ±0.015 rad（±0.86°）：开机一次", EV["encoder_bias"].mode == "startup"
     and EV["encoder_bias"].params["bias_range"] == (-0.015, 0.015) and round(0.015 * DEG, 2) == 0.86),
    ("躯干质量和转动惯量一起乘 0.95–1.05：开机一次", EV["randomize_mass_inertia"].mode == "startup"
     and round(mass_lo, 6) == 0.95 and round(mass_hi, 6) == 1.05),
    ("IMU 装歪：随机一根轴、0–6°，整个训练不变", imu_deg == 6.0),
    ("躯干质心 ±3 mm（课程放宽）、头部质心 ±3 mm：每回合", EV["randomize_com"].mode == EV["randomize_head_com"].mode == "reset"
     and EV["randomize_com"].params["ranges"] == (-0.003, 0.003) and EV["randomize_head_com"].params["ranges"] == (-0.003, 0.003)),
    ("摩擦倍率 0.9–1.1、转子惯量 × 0.9–1.1：每回合", EV["randomize_joint_friction"].mode == EV["randomize_armature"].mode == "reset"
     and EV["randomize_armature"].params["ranges"] == (0.9, 1.1)),
    ("推一把：每 3–6 s，前后、左右各 ±0.3 m/s", EV["push_robot"].mode == "interval" and EV["push_robot"].interval_range_s == (3.0, 6.0)
     and EV["push_robot"].params["velocity_range"] == {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}),
]
for label, ok in checks:
    check(label, ok)
import mjlab.envs.mdp.dr.joint as dr_joint  # noqa: E402
from mjlab_microduck.tasks import mdp as microduck_mdp  # noqa: E402

check("编码器偏置是每只机器人、每个关节各抽一个（mjlab 的 encoder_bias）",
      lines_in_order(inspect.getsource(dr_joint.encoder_bias), ["num_joints = len(joint_ids_tensor)", "bias_samples = sample_uniform("]))
check("IMU 装歪：每只机器人第一次读 IMU 时抽一次，之后缓存、整个训练不变",
      "constant per env for the whole run" in inspect.getsource(microduck_mdp._imu_misalignment_quat))
print(f"模板自带的 base_com：body_names = {EV['base_com'].params['asset_cfg'].body_names}（空的：Microduck 没填，不作用于任何部件）")
check("base_com 的部件名是空的", EV["base_com"].params["asset_cfg"].body_names == ())
check("第 4 章的期望：电池 U(6.5, 8.2) 的期望 (6.5 + 8.2)/2 = 7.35 V", (6.5 + 8.2) / 2 == 7.35)
rng = np.random.default_rng(17)
fleet_v = rng.uniform(*C.actuators.vin_range, 4096)
print(f"教学模拟：4096 只各抽一块电池，平均 {fleet_v.mean():.3f} V，最低 {fleet_v.min():.3f}，最高 {fleet_v.max():.3f}")
check("4096 只的平均离 7.35 不到 0.02 V（第 4 章 4.5 节：抽得越多，平均越贴近期望）", abs(fleet_v.mean() - 7.35) < 0.02)
check("这次模拟：平均 7.34 V；每 0.1 V 一格，平均每格 4096 ÷ 17 ≈ 241 只", round(fleet_v.mean(), 2) == 7.34 and round(4096 / 17) == 241)
check("每回合开局还随机摆放位置和朝向（reset_base：前后左右 ±0.5 m、朝向 ±3.14 rad）", EV["reset_base"].mode == "reset"
      and EV["reset_base"].params["pose_range"]["x"] == (-0.5, 0.5) and EV["reset_base"].params["pose_range"]["yaw"] == (-3.14, 3.14))
ROW_NAMES = ["电池电压 (V)", "脚底摩擦", "摩擦倍率", "躯干质心 (mm)"]
startup = np.column_stack([rng.uniform(6.5, 8.2, 5), rng.uniform(0.7, 1.3, 5)])
episodes = np.column_stack([rng.uniform(0.9, 1.1, 3), rng.uniform(-3, 3, 3)])
print("5 只机器人开机时抽的前两行：", np.round(startup, 2).tolist())
print("1 号机器人 3 个回合里每回合抽的后两行：", np.round(episodes, 2).tolist())
robots5 = np.column_stack([startup, rng.uniform(0.9, 1.1, 5), rng.uniform(-3, 3, 5)])
robots5[0, 2:] = episodes[0]                              # 1 号开机那一回合，就是图 ② 的“回合 1”：两处必须是同一组数
check("1 号机器人：电池 7.66 V；摩擦倍率三个回合依次 1.09、0.96、1.06（17.7 节自测）；图 ① 的 1 号那一列 = 图 ② 的回合 1",
      round(startup[0, 0], 2) == 7.66 and [round(float(x), 2) for x in episodes[:, 0]] == [1.09, 0.96, 1.06]
      and np.array_equal(robots5[0, 2:], episodes[0]))
check("总览图 ③ 的 5 个电池电压就是这 5 只：" + "、".join(f"{v:.2f}" for v in startup[:, 0]) + " V", np.allclose(vins, startup[:, 0]))

# ---------------------------------------------------------------------------
banner("7b. 画图：figures/ch17_dr_robots.png（开机抽的不变，每回合抽的会变）")
fig = plt.figure(figsize=(9.8, 14.6))
fig.suptitle("域随机化：4096 只机器人，每只一套自己的身体", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.99)


def fmt_dr(value, row):
    return f"{value:.2f}" if row < 3 else f"{value:+.1f}".replace("-", "−")


def dr_table(ax, title, headers, rows, width, highlights=()):
    """左边 4 行名字（蓝 = 开机抽一次，绿 = 每回合重抽），右边一张数表；行名和格子逐行对齐。"""
    lesson_panel(ax, title, xmax=10, ymax=5.0)
    h, bottom = 0.72, 0.15
    for i, rn in enumerate(ROW_NAMES):
        ax.text(0.05, bottom + (3 - i) * h + h / 2, rn, color=BLUE if i < 2 else GREEN, fontsize=FS_SMALL, va="center")
    lesson_cells(ax, [headers], left=3.0, bottom=bottom + 4 * h + 0.12, width=width, height=0.6, facecolor="white",
                 fontsize=FS_SMALL)
    lesson_cells(ax, rows, left=3.0, bottom=bottom, width=width, height=h, fontsize=FS_SMALL, highlights=highlights)


ax = fig.add_axes([0.02, 0.68, 0.96, 0.25])
dr_table(ax, "① 开机（也就是第 1 回合）：5 只机器人各抽一套（只列 4 项）", [f"{k + 1} 号" for k in range(5)],
         [[fmt_dr(robots5[k, i], i) for k in range(5)] for i in range(4)], width=1.35)
ax = fig.add_axes([0.02, 0.39, 0.96, 0.25])
ep_rows = [[fmt_dr(robots5[0, i], i)] * 3 if i < 2 else [fmt_dr(episodes[k, i - 2], i) for k in range(3)] for i in range(4)]
dr_table(ax, "② 同一只（1 号）的 3 个回合：橙格每回合都换", [f"回合 {k + 1}" for k in range(3)], ep_rows, width=1.8,
         highlights=[(i, k) for i in (2, 3) for k in range(3)])
note(ax, 0.05, -0.35, "蓝字的两项开机抽一次，整个训练不变；绿字的两项每回合开局重抽。")
ax = fig.add_axes([0.12, 0.07, 0.84, 0.21])
data_axes(ax, "电池电压（V）", "只数")
ax.hist(fleet_v, bins=17, range=(6.5, 8.2), color=CELL, edgecolor=CELL_EDGE)
ax.axvline(fleet_v.mean(), color=ORANGE, lw=2.6)
ax.text(fleet_v.mean() + 0.03, 280, f"平均 {fleet_v.mean():.2f} V ≈ 期望 7.35", color=ORANGE, fontsize=FS_SMALL, bbox=WHITE_BOX)
ax.set_xlim(6.4, 8.3)
ax.set_ylim(0, 300)
panel_title(fig, [ax], "③ 4096 只的电池电压：每 0.1 V 一格，数一数各有几只")
panel_note(fig, [ax], "每格约 4096 ÷ 17 ≈ 241 只：均匀分布，哪儿都一样可能（第 4 章 4.2 节）。")
savefig(fig, "ch17_dr_robots")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("8. 目标是对各种身体取平均；随机化要符合物理、不能累积")
mix = 0.9 * 10 + 0.1 * (-20)
print(f"九成的身体回报 10、一成回报 −20：平均 0.9 × 10 + 0.1 × (−20) = 9 − 2 = {mix:g}")
check("混合回报 = 0.9 × 10 + 0.1 × (−20) = 7（手算复核里的同一个数）", mix == 7)
check("自测：0.5 × 12 + 0.5 × 2 = 7——平均一样，最差的那种身体一个是 2、一个是 −20", 0.5 * 12 + 0.5 * 2 == 7)
stages = cfg.curriculum["com_range"].params["range_stages"]
print("躯干质心课程：" + "；".join(f"第 {s['step'] // 24} 次迭代起 ±{s['range'] * 1000:g} mm" for s in stages))
check("躯干质心课程：±3 → ±5 → ±10 → ±15 mm，在第 0、500、1000、1500 次迭代换挡",
      [(s["step"] // 24, round(s["range"] * 1000)) for s in stages] == [(0, 3), (500, 5), (1000, 10), (1500, 15)]
      and all(s["step"] % 24 == 0 for s in stages))
head_stages = cfg.curriculum["head_com_range"].params["range_stages"]
check("头部质心课程：±3 → ±5 → ±10 mm", [round(s["range"] * 1000) for s in head_stages] == [3, 5, 10])
cfg_text = (Path(C.__file__).resolve().parents[1] / "tasks" / "microduck_velocity_env_cfg.py").read_text(encoding="utf-8")
check("配置注释记下了这段经过：曾经放宽到 ±30 mm，可脚跟只比踝关节靠后 20 mm，所以上限定在 ±15 mm",
      "heel is only 20 mm behind" in cfg_text and "the previous ramp to ±30 mm" in cfg_text and "Capped at ±15 mm" in cfg_text)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg  # noqa: E402
check("一次迭代 = 每只机器人走 24 个环境步（num_steps_per_env = 24），所以课程的 step ÷ 24 = 迭代数",
      MicroduckRlCfg.num_steps_per_env == 24)
DEFAULT_MM, OFFSET_MM = 10, 1
right = [DEFAULT_MM + OFFSET_MM for _ in range(3)]
wrong = [DEFAULT_MM + OFFSET_MM * (k + 1) for k in range(3)]
print(f"默认 10 mm、每回合抽到 +1 mm：正确（先恢复默认再加）{right}；错误（在上一回合上接着加）{wrong}")
check("不累积：11、11、11；累积：11、12、13", right == [11, 11, 11] and wrong == [11, 12, 13])
mdp_src = inspect.getsource(importlib.import_module("mjlab_microduck.tasks.mdp"))
fn = mdp_src[mdp_src.find("def randomize_bam_friction("):mdp_src.find("def randomize_mass_and_inertia(")]
check("randomize_bam_friction 先恢复默认（reset_friction_scale）再设新值（set_friction_scale）",
      lines_in_order(fn, ["actuator.reset_friction_scale(env_ids)", "actuator.set_friction_scale(env_ids, samples)"]))
check("关节摩擦的随机化走 BAM 的 friction_scale（randomize_bam_friction），不是 dof_frictionloss",
      EV["randomize_joint_friction"].func.__name__ == "randomize_bam_friction")
check("bam 每个物理步都把自己算的摩擦预算写进 dof_frictionloss（所以别处改它会被覆盖）",
      "fl_field[:, self._dof_ids] = frictionloss" in mjlab_bam and "self._write_frictions(frictionloss, friction_viscous)" in mjlab_bam)
check("expand_bam_friction_fields 是开机事件；不注册的话，环境多于 1 个时 bam 直接报错（RuntimeError）",
      EV["expand_bam_friction_fields"].mode == "startup" and 'raise RuntimeError(\n                    "BamActuator writes per-environment dof_frictionloss/"' in mjlab_bam)

# ---------------------------------------------------------------------------
banner("9. 映射到项目：正文引用的源码行还在不在")
MJ = Path(importlib.util.find_spec("mjlab").origin).parent
ACTIONS = ["self._processed_actions = self._raw_actions * self._scale + self._offset",
           "self._offset = self._entity.data.default_joint_pos[:, self._target_ids].clone()",
           "encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]",
           "target = self._processed_actions - encoder_bias",
           "self._entity.set_joint_position_target(target, joint_ids=self._target_ids)"]
BAM_LINES = ["duty_cycle = (q_target - q) * self.kp * self.error_gain",
             "back_emf = self.model.kt.value * dq",
             "duty_span = self.model.R.value * self.max_current / self.vin",
             "duty_center = back_emf / self.vin",
             "duty_cycle = self.backend.clamp(",
             "duty_cycle, duty_center - duty_span, duty_center + duty_span",
             "duty_cycle = self.backend.clamp(duty_cycle, -self.max_pwm, self.max_pwm)",
             "return self.vin * duty_cycle",
             "torque = self.model.kt.value * volts / self.model.R.value",
             "torque -= (self.model.kt.value**2) * dq / self.model.R.value"]
CONST = ['_BAM_ACTUATOR_KWARGS = dict(', 'motor_name="xl330",', 'model="m6",', 'kp_fw=200.0,', "vin_range=(6.5, 8.2),",
         "vin_drop_gain_range=(0.0, 0.2),", "vin_min=6.0,", "# max_current=1.75,", "delay_min_lag=3,", "delay_max_lag=6,",
         "actuators = FrictionDRBamActuatorCfg(**_BAM_ACTUATOR_KWARGS)"]
ENTITY = ["def _apply_actuator_controls(self) -> None:", "command = act.get_command(self._data)",
          "command = act.apply_delay(command)", "self._data.write_ctrl(act.compute(command), act.ctrl_ids)"]
STEP = ["for _ in range(self.cfg.decimation):", "self.action_manager.apply_action()", "self.scene.write_data_to_sim()",
        "self.sim.step()"]
XL = ["XL330_ENCODER_COUNTS_PER_REV = 4096", "XL330_KP_DIVISOR = 256", "XL330_PWM_LIMIT = 885", "vin=7.5,", "kp=400,",
      "max_current=1.75,"]
CFG = ["JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)", "ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)",
       "VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)", "VELOCITY_PUSH_RANGE = (-0.3, 0.3)", "IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0",
       "ENCODER_BIAS_RANGE = (-0.015, 0.015)", "joint_pos_action.scale = 1.0", 'cfg.events["expand_bam_friction_fields"] = EventTermCfg(',
       'cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)', "func=microduck_mdp.randomize_bam_friction,"]
for label, path, wanted in (
        ("mjlab/envs/mdp/actions/actions.py（JointPositionAction）", MJ / "envs" / "mdp" / "actions" / "actions.py", ACTIONS),
        ("bam/actuator.py（compute_control / compute_torque）", bam_src / "actuator.py", BAM_LINES),
        ("microduck_constants.py（_BAM_ACTUATOR_KWARGS）", Path(C.__file__), CONST),
        ("mjlab/entity/entity.py（_apply_actuator_controls）", MJ / "entity" / "entity.py", ENTITY),
        ("mjlab/envs/manager_based_rl_env.py（step 的 decimation 循环）", MJ / "envs" / "manager_based_rl_env.py", STEP),
        ("bam/dynamixel/actuator.py（XL330Actuator）", bam_src / "dynamixel" / "actuator.py", XL),
        ("microduck_velocity_env_cfg.py（域随机化的范围和事件）", Path(C.__file__).resolve().parents[1] / "tasks" / "microduck_velocity_env_cfg.py", CFG)):
    if path.is_file():
        check(f"{label}：{len(wanted)} 行原样存在、顺序一致", lines_in_order(path.read_text(encoding="utf-8"), wanted))
    else:
        print(f"  （没找到 {label}，跳过）")

done()
