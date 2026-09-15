"""第 3 章实验：偏导数、梯度、梯度下降拟合一条直线、学习率与梯度裁剪。

运行：uv run python docs/learn-zh/labs/ch03_gradient_descent.py
纯 CPU，numpy + matplotlib。
"""

import numpy as np

from _common import banner, check, done, savefig, table, use_headless_matplotlib

np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------------------------------------
banner("1. 偏导数：f(x, y) = x² + 3y²，在 (1, 2) 处")


def f(x, y):
    return x**2 + 3 * y**2


h = 1e-6
x0, y0 = 1.0, 2.0
df_dx_num = (f(x0 + h, y0) - f(x0, y0)) / h   # 只动 x，y 当常数
df_dy_num = (f(x0, y0 + h) - f(x0, y0)) / h   # 只动 y，x 当常数
table(["", "数值", "解析"], [["∂f/∂x", df_dx_num, 2 * x0], ["∂f/∂y", df_dy_num, 6 * y0]])
grad = np.array([2 * x0, 6 * y0])
print("梯度 ∇f(1,2) =", grad, "  —— 把两个偏导数排成一个向量")
check("偏导数数值 ≈ 解析", abs(df_dx_num - 2) < 1e-4 and abs(df_dy_num - 12) < 1e-4)

# 梯度是“上升最快”的方向：往各个方向走同样一小步，看谁涨得最多
banner("2. 梯度方向 = 上升最快的方向（走一步 0.01，比较涨幅）")
step = 0.01
rows = []
best = None
for deg in range(0, 360, 30):
    th = np.radians(deg)
    d = np.array([np.cos(th), np.sin(th)])
    rise = f(x0 + step * d[0], y0 + step * d[1]) - f(x0, y0)
    rows.append([deg, rise])
    if best is None or rise > best[1]:
        best = (deg, rise)
table(["方向角°", "f 的涨幅"], rows)
grad_deg = np.degrees(np.arctan2(grad[1], grad[0]))
print(f"涨得最多的方向 ≈ {best[0]}°，梯度方向 = {grad_deg:.1f}°")
check("最陡方向与梯度方向相差 < 30°", abs(best[0] - grad_deg) < 30)

# ---------------------------------------------------------------------------
banner("3. 梯度下降拟合直线 y = w·x + b（真值 w=2, b=1）")
rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, size=50)
Y = 2.0 * X + 1.0 + rng.normal(0, 0.1, size=50)


def loss(w, b):
    pred = w * X + b
    return np.mean((pred - Y) ** 2)          # 均方误差 MSE


def grad_loss(w, b):
    pred = w * X + b
    err = pred - Y
    dw = np.mean(2 * err * X)                # ∂L/∂w
    db = np.mean(2 * err)                    # ∂L/∂b
    return dw, db


w, b = 0.0, 0.0
alpha = 0.1                                  # 学习率
history = []
rows = []
for it in range(201):
    L = loss(w, b)
    history.append(L)
    if it in (0, 1, 2, 5, 10, 20, 50, 100, 200):
        rows.append([it, w, b, L])
    dw, db = grad_loss(w, b)
    w -= alpha * dw                           # θ ← θ − α ∇L
    b -= alpha * db
table(["迭代", "w", "b", "损失 L"], rows)
check("学到 w ≈ 2", abs(w - 2.0) < 0.05)
check("学到 b ≈ 1", abs(b - 1.0) < 0.05)
check("损失单调下降", all(history[i] >= history[i + 1] - 1e-12 for i in range(len(history) - 1)))

# ---------------------------------------------------------------------------
banner("4. 学习率太大会发散")
rows = []
for alpha in [0.01, 0.1, 0.5, 1.0, 1.6]:
    w, b = 0.0, 0.0
    for _ in range(50):
        dw, db = grad_loss(w, b)
        w -= alpha * dw
        b -= alpha * db
    rows.append([alpha, loss(w, b) if np.isfinite(loss(w, b)) else float("inf")])
table(["学习率 α", "50 步后的损失"], rows)
print("项目里 PPO 的初始学习率 learning_rate=1e-3，且由 KL 自适应调整（第 13 章）。")

# ---------------------------------------------------------------------------
banner("5. 梯度裁剪：max_grad_norm = 1.0 是什么意思")
g = np.array([3.0, 4.0])                    # ‖g‖ = 5
max_norm = 1.0
scale = min(1.0, max_norm / np.linalg.norm(g))
g_clipped = g * scale
print("原梯度 g =", g, " ‖g‖ =", np.linalg.norm(g))
print("裁剪后   =", g_clipped, " ‖g‖ =", np.linalg.norm(g_clipped))
print("方向不变，只把长度压到 1.0 —— 这就是 ppo.py 里 clip_grad_norm_(…, 1.0) 做的事。")
check("裁剪后范数 = 1", abs(np.linalg.norm(g_clipped) - 1.0) < 1e-12)
check("裁剪不改方向", np.allclose(g_clipped / np.linalg.norm(g_clipped), g / np.linalg.norm(g)))

plt = use_headless_matplotlib()
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(history)
ax.set_yscale("log")
ax.set_xlabel("迭代")
ax.set_ylabel("损失 L (对数刻度)")
ax.set_title("梯度下降：损失随迭代下降 (α=0.1)")
ax.grid(alpha=0.3)
savefig(fig, "ch03_loss_curve")

banner("6. 画图：碗状地形、等高线和梯度箭头")
fig, ax = plt.subplots(figsize=(5.5, 5))
gx, gy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-2, 2, 200))
cs = ax.contour(gx, gy, gx**2 + 3 * gy**2, levels=[0.5, 2, 4, 8, 13, 20], cmap="Greys")
ax.clabel(cs, fontsize=7)
for (px, py) in [(1, 2), (-2, 0.5), (0.5, -1.5), (2, -1)]:
    gxv, gyv = 2 * px, 6 * py
    ax.arrow(px, py, 0.08 * gxv, 0.08 * gyv, head_width=0.12, color="tab:red")
    ax.plot(px, py, "ko", ms=3)
ax.plot(0, 0, "b*", ms=12, label="碗底 (0,0)")
ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("f = x² + 3y²：灰线是等高线，红箭头是梯度（指向上坡）")
ax.legend(loc="lower right"); ax.grid(alpha=0.2)
savefig(fig, "ch03_gradient_arrows")

done()
