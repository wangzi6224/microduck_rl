"""第 1 章实验：变化率、导数、链式法则，以及项目里的高斯奖励核。

运行：uv run python docs/learn-zh/labs/ch01_derivative.py
纯 CPU，只用 numpy + matplotlib。
"""

import math

import numpy as np

from _common import banner, check, done, savefig, table, use_headless_matplotlib

# ---------------------------------------------------------------------------
banner("1. 平均变化率 → 导数：f(x) = x², 在 x = 3 处")


def f(x):
    return x**2


rows = []
for h in [1.0, 0.1, 0.01, 0.001, 1e-4, 1e-5]:
    avg_rate = (f(3 + h) - f(3)) / h  # 平均变化率 = 割线斜率
    rows.append([h, f(3 + h), avg_rate, avg_rate - 6.0])
table(["h", "f(3+h)", "(f(3+h)-f(3))/h", "与 6 的差"], rows)
print("解析导数 f'(x) = 2x  →  f'(3) = 6")
check("h 越小，平均变化率越接近 6", abs(rows[-1][2] - 6.0) < 1e-3)

# ---------------------------------------------------------------------------
banner("2. 链式法则：y = f(g(x))，g(x) = 2x + 1，f(u) = u²")


def g(x):
    return 2 * x + 1


def y(x):
    return f(g(x))


x0 = 1.5
h = 1e-6
numeric = (y(x0 + h) - y(x0)) / h
# 链式法则：dy/dx = f'(g(x)) · g'(x) = 2·g(x) · 2
analytic = 2 * g(x0) * 2
table(["x0", "数值导数", "链式法则", "差"], [[x0, numeric, analytic, numeric - analytic]])
check("链式法则 = 数值导数", abs(numeric - analytic) < 1e-4)

# ---------------------------------------------------------------------------
banner("3. e^x 的导数是它自己；e^{-x} 单调递减到 0")
rows = []
for x in [-1.0, 0.0, 1.0, 2.0]:
    d_num = (math.exp(x + h) - math.exp(x)) / h
    rows.append([x, math.exp(x), d_num, math.exp(-x)])
table(["x", "e^x", "(e^x)' 数值", "e^{-x}"], rows)
check("(e^x)' ≈ e^x", all(abs(r[1] - r[2]) < 1e-4 for r in rows))

# ---------------------------------------------------------------------------
banner("4. 项目里的奖励核：reward = exp(-err² / std²)")
print("velocity cfg:343  std = sqrt(0.1)  → std² = 0.1  (线速度跟踪)")
print("velocity cfg:345  std = sqrt(0.5)  → std² = 0.5  (角速度跟踪)")
rows = []
for err in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
    r_lin = math.exp(-(err**2) / 0.1)
    r_ang = math.exp(-(err**2) / 0.5)
    rows.append([err, err**2, r_lin, r_ang])
table(["误差 err", "err²", "exp(-err²/0.1)", "exp(-err²/0.5)"], rows)
check("误差为 0 时奖励恰好是 1", rows[0][2] == 1.0)
check("误差越大奖励越小", all(rows[i][2] > rows[i + 1][2] for i in range(len(rows) - 1)))
# 半衰点：err² = std² · ln2
print(f"奖励掉到一半的误差：lin {math.sqrt(0.1*math.log(2)):.3f} m/s, ang {math.sqrt(0.5*math.log(2)):.3f} rad/s")

# 这条核的导数：d/d(err) exp(-err²/s²) = -(2·err/s²)·exp(-err²/s²)
# 误差为 0 时导数为 0（顶点是平的），很小的非零误差处导数一般仍非零；这也不是 PPO 直接使用的参数梯度。
rows = []
for err in [0.0, 0.05, 0.1, 0.3, 0.6]:
    s2 = 0.1
    grad = -(2 * err / s2) * math.exp(-(err**2) / s2)
    rows.append([err, grad])
table(["误差 err", "d reward / d err (std²=0.1)"], rows)

plt = use_headless_matplotlib()
fig, ax = plt.subplots(figsize=(6, 3.6))
e = np.linspace(0, 1.2, 300)
for s2, label in [(0.05, "std²=0.05 (upright)"), (0.1, "std²=0.1 (lin vel)"), (0.5, "std²=0.5 (ang vel)")]:
    ax.plot(e, np.exp(-(e**2) / s2), label=label)
ax.set_xlabel("误差 err")
ax.set_ylabel("exp(-err² / std²)")
ax.set_title("同一个奖励核，std 决定“多宽容”")
ax.grid(alpha=0.3)
ax.legend()
savefig(fig, "ch01_gaussian_kernel")

# ---------------------------------------------------------------------------
banner("5. 画图：割线一步步变成切线")
fig, ax = plt.subplots(figsize=(6, 4))
xs = np.linspace(1.5, 5.0, 200)
ax.plot(xs, xs**2, color="black", label="f(x) = x²")
for hh, c in [(2.0, "tab:red"), (1.0, "tab:orange"), (0.5, "tab:green")]:
    slope = (f(3 + hh) - f(3)) / hh
    ax.plot([3, 3 + hh], [f(3), f(3 + hh)], "o", color=c)
    ax.plot(xs, f(3) + slope * (xs - 3), "--", color=c, label=f"h={hh}: 斜率 {slope:.1f}")
ax.plot(xs, f(3) + 6 * (xs - 3), color="tab:blue", lw=2, label="h→0: 切线, 斜率 6")
ax.plot([3], [9], "o", color="tab:blue")
ax.set_xlim(1.5, 5); ax.set_ylim(0, 26)
ax.set_xlabel("x"); ax.set_ylabel("f(x)")
ax.set_title("从 x=3 出发走 h 步：h 越小，割线越贴近切线")
ax.grid(alpha=0.3); ax.legend(fontsize=8)
savefig(fig, "ch01_secant_to_tangent")

# ---------------------------------------------------------------------------
banner("6. 相同误差改善、不同加分；导数是局部变化率，不是 PPO 参数梯度")
def kernel(err, sigma):
    return math.exp(-(err / sigma) ** 2)

sigma = math.sqrt(0.1)
rows = []
for old, new in [(0.05, 0.0), (0.25, 0.20), (1.0, 0.95)]:
    before, after = kernel(old, sigma), kernel(new, sigma)
    rows.append([old, new, before, after, after - before])
table(["改善前误差", "改善后误差", "原奖励", "新奖励", "增加奖励"], rows)
check("相同改善量，中间区域比顶部和尾部加分多", rows[1][-1] > rows[0][-1] > rows[2][-1])
peak = sigma / math.sqrt(2)
print(f"最敏感位置 sigma/sqrt(2) = {peak:.6f}，这里奖励 = {kernel(peak, sigma):.6f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
errors = np.linspace(0, 1.5, 1200)
for width in [0.1, math.sqrt(0.1), 1.0]:
    values = np.exp(-(errors / width) ** 2)
    sensitivity = 2 * errors / width**2 * values
    label = f"σ={width:.3f}"
    axes[0].plot(errors, values, label=label)
    line, = axes[1].plot(errors, sensitivity, label=label)
    peak = width / math.sqrt(2)
    axes[1].plot(peak, math.sqrt(2) / width * math.exp(-0.5), "o", color=line.get_color())
for ax in axes:
    ax.set_xlabel("误差大小（同一物理量、同一单位）")
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel("奖励 R")
axes[0].set_title("分数高，不等于对改善最敏感")
axes[1].set_ylabel("敏感程度 |dR / d误差|")
axes[1].set_title("圆点：σ/√2；不是 PPO 的参数梯度")
fig.tight_layout()
savefig(fig, "ch01_reward_sensitivity")

done()
