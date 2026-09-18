"""第 17 章实验：动作 → 目标角；BAM 舵机模型（固件 P 环 → 电压 → 直流电机力矩）；域随机化 = 对参数分布取期望。
纯 numpy 复现电机方程，画一次阶跃响应。

运行：uv run python docs/learn-zh/labs/ch17_bam_motor.py
纯 CPU。
"""

import math

import numpy as np

from _common import banner, check, done, savefig, table, use_headless_matplotlib

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
banner("1. 动作 → 目标关节角：target = HOME + a × scale − encoder_bias（scale = 1.0）")
HOME = np.array([0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0, 0.0, 0.0873, 0.4579, 0.0049, -0.4530])
names = ["L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle", "neck_pitch", "head_pitch", "head_yaw", "head_roll",
         "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle"]
a = rng.normal(0, 0.3, 14)              # 假装是策略输出的 14 个数（弧度）
target = HOME + a * 1.0
table(["idx", "关节", "HOME (rad)", "动作 a", "目标角 (rad)", "目标角 (°)"], [[i, names[i], HOME[i], a[i], target[i], math.degrees(target[i])] for i in range(14)])
print("scale=1.0：策略输出 1 个单位 = 1 弧度 ≈ 57°。所以策略必须学会输出很小的数。")

# ---------------------------------------------------------------------------
banner("2. XL330 舵机的固件位置环：目标角 − 实际角 → 占空比 → 电压")
import json, pathlib, bam  # noqa: E402
m6 = json.loads((pathlib.Path(bam.__file__).parent / "params" / "xl330" / "m6.json").read_text())
KT, R, ARMATURE = m6["kt"], m6["R"], m6["armature"]  # BAM 对真实 XL330 拟合出来的 M6 参数
VIN, KP = 7.5, 200.0                                 # 名义电压 7.5 V；microduck kp_fw=200
print(f"BAM xl330/m6 拟合参数：kt = {KT:.4f} N·m/A，R = {R:.3f} Ω，armature = {ARMATURE:.5f} kg·m²")
ERROR_GAIN = (4096 / (2 * math.pi)) / (256 * 885)    # 编码器计数/弧度 ÷ (KP 除数 × PWM 上限)
print(f"error_gain = (4096/2π)/(256×885) = {ERROR_GAIN:.5f}")


def firmware_voltage(q_target, q, vin=VIN, kp=KP):
    duty = (q_target - q) * kp * ERROR_GAIN          # 占空比 ∝ 位置误差
    duty = np.clip(duty, -1.0, 1.0)                  # 物理 PWM 上限
    return vin * duty


rows = []
for err_deg in (0.5, 1, 2, 5, 10, 20, 60, 120):
    err = math.radians(err_deg)
    rows.append([err_deg, (err * KP * ERROR_GAIN), firmware_voltage(err, 0.0)])
table(["位置误差 (°)", "占空比（裁剪前）", "电压 V"], rows)
print(f"误差约 {math.degrees(1.0/(KP*ERROR_GAIN)):.1f}° 时占空比到 1.0 → 电压饱和在 7.5 V，再大的误差也推不动更快。")
check("误差 120° 时电压饱和 7.5 V", abs(firmware_voltage(math.radians(120), 0.0) - 7.5) < 1e-9)
print("正常走路时关节误差只有几度，电压只用到零点几伏——这只是电压指令的例子；实际响应还受反电动势、摩擦与负载影响。")

# ---------------------------------------------------------------------------
banner("3. 直流电机：τ = kt·V/R − kt²·q̇/R（第二项是反电动势：转得越快，推力越小）")


def motor_torque(V, dq):
    return KT * V / R - KT**2 * dq / R


rows = []
for dq in (0.0, 2.0, 5.0, 10.0, 7.5 / KT):
    rows.append([dq, motor_torque(7.5, dq)])
table(["转速 q̇ (rad/s)", "满电压 7.5 V 时的力矩 (N·m)"], rows)
print(f"堵转力矩 kt·V/R = {KT*7.5/R:.3f} N·m；空载最高转速 V/kt = {7.5/KT:.3f} rad/s（力矩降到 0）。")
check("空载转速处力矩 ≈ 0", abs(motor_torque(7.5, 7.5 / KT)) < 1e-9)

# ---------------------------------------------------------------------------
banner("4. 阶跃响应：让一个关节从 0 追到 20°，仿真 200 Hz（和项目一样 dt=0.005）")
I = ARMATURE + 0.0005         # 转子折算惯量 armature + 一点负载
DT = 0.005


def simulate(vin, delay_steps=0, friction=0.0, steps=120):
    q, dq, log = 0.0, 0.0, []
    q_target = math.radians(20)
    cmd_buf = [0.0] * delay_steps               # 命令延迟：BAM 的 delay_min_lag..delay_max_lag
    for k in range(steps):
        cmd_buf.append(q_target); cmd = cmd_buf.pop(0)  # lag=0 当步到达，lag=N 等 N 步
        V = firmware_voltage(cmd, q, vin=vin)
        tau = motor_torque(V, dq)
        tau -= friction * np.sign(dq)                  # 库仑摩擦（BAM 用更细的 Stribeck 模型，这里简化）
        dq += tau / I * DT
        q += dq * DT
        log.append(math.degrees(q))
    return np.array(log)


# 检查延迟单位和边界，防止把 lag=0 也额外延迟一帧。
check("零延迟首步已运动", simulate(7.5, delay_steps=0, steps=1)[0] > 0)
lagged = simulate(7.5, delay_steps=6, steps=7)
check("6 步延迟前 6 帧静止，第 7 帧开始运动", np.all(lagged[:6] == 0) and lagged[6] > 0)
check("6 个物理步是 30 ms", abs(6 * DT - 0.030) < 1e-12)

plt = use_headless_matplotlib()
fig, ax = plt.subplots(figsize=(6.5, 3.6))
t = (np.arange(120) + 1) * DT  # 记录每次积分结束后的状态
for label, kw in [("vin=7.5 V", dict(vin=7.5)), ("vin=6.5 V（电池快没电）", dict(vin=6.5)),
                  ("vin=7.5 V + 延迟 6 步(30 ms)", dict(vin=7.5, delay_steps=6)), ("vin=7.5 V + 摩擦 0.05 N·m", dict(vin=7.5, friction=0.05))]:
    ax.plot(t, simulate(**kw), label=label)
ax.axhline(20, color="k", lw=0.5, ls="--")
ax.set_xlabel("时间 s"); ax.set_ylabel("关节角 °"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_title("同一个 20° 阶跃指令，不同电压/延迟/摩擦下的响应")
savefig(fig, "ch17_bam_step")
resp = simulate(7.5)
print(f"7.5 V：0.1 s 时到 {resp[19]:.1f}°，0.6 s 时到 {resp[-1]:.1f}°")

# ---------------------------------------------------------------------------
banner("5. 域随机化 = 每只机器人抽一组参数：训练目标变成“对参数分布的期望”")
rows = []
for name, lo, hi in [("电池电压 vin (V)", 6.5, 8.2), ("压降增益 (V/N·m)", 0.0, 0.2), ("摩擦倍率 friction_scale", 0.9, 1.1),
                     ("脚底摩擦系数", 0.7, 1.3), ("躯干质量倍率", 0.95, 1.05), ("编码器偏置 (rad)", -0.015, 0.015),
                     ("躯干质心偏移 (m)", -0.003, 0.003), ("IMU 安装误差角 (°)", 0.0, 6.0)]:
    rows.append([name, f"U({lo}, {hi})", (lo + hi) / 2, rng.uniform(lo, hi)])
table(["随机化的量", "分布", "期望", "这只机器人抽到的"], rows)
print("4096 只机器人各抽一组。策略必须在设定的参数分布上平均表现好，不保证每种条件都成功——这就是 sim2real 的数学含义：")
print("  max_θ E_{参数~分布}[ E[G | 参数] ]，而不是只在一台“标准机器人”上最优。")

done()
