"""第 5 章实验：用 torch 把第 3 章的"找直线"再做一遍，看清"学习"的四个零件：
数据、模型、损失、优化。并第一次用 loss.backward() 让 torch 自动算梯度。

运行：uv run python docs/learn-zh/labs/ch05_linear_regression.py
纯 CPU。
"""

import numpy as np
import torch

from _common import banner, check, done, table

torch.manual_seed(0)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
banner("1. 零件一：数据。50 个点，大致落在 y = 2x + 1 附近")
X_np = rng.uniform(-1, 1, size=50)
Y_np = 2.0 * X_np + 1.0 + rng.normal(0, 0.1, size=50)
X = torch.tensor(X_np, dtype=torch.float32).unsqueeze(1)   # 形状 (50, 1)
Y = torch.tensor(Y_np, dtype=torch.float32).unsqueeze(1)
table(["x", "y"], [[float(X[i, 0]), float(Y[i, 0])] for i in range(5)])
print("…… 共 50 行。X 形状", tuple(X.shape), "Y 形状", tuple(Y.shape))

# ---------------------------------------------------------------------------
banner("2. 零件二：模型。ŷ = w·x + b，两个旋钮，初值 0")
w = torch.zeros(1, requires_grad=True)   # requires_grad=True：告诉 torch “我要它的导数”
b = torch.zeros(1, requires_grad=True)


def model(x):
    return w * x + b


print("初始 w =", w.item(), " b =", b.item())
print("初始预测 ŷ(0.5) =", model(torch.tensor([0.5])).item(), "（真值应该接近 2）")

# ---------------------------------------------------------------------------
banner("3. 零件三：损失。均方误差 MSE = mean((ŷ − y)²)")


def mse(pred, target):
    return torch.mean((pred - target) ** 2)


L0 = mse(model(X), Y)
print("初始损失 L =", L0.item())

# ---------------------------------------------------------------------------
banner("4. 零件四：优化。loss.backward() 自动算梯度，再 θ ← θ − α∇L")
alpha = 0.1
rows = []
for it in range(201):
    pred = model(X)
    L = mse(pred, Y)
    if it in (0, 1, 2, 5, 10, 20, 50, 100, 200):
        rows.append([it, w.item(), b.item(), L.item()])
    # --- 第 3 章手写的两行 dw/db，这里让 torch 算 ---
    L.backward()                       # 沿计算图反向传播（第 7 章），把 ∂L/∂w、∂L/∂b 写进 w.grad、b.grad
    with torch.no_grad():              # 更新参数时暂停“记录导数”
        w -= alpha * w.grad
        b -= alpha * b.grad
    w.grad.zero_()                     # 清空，否则下一次 backward 会累加
    b.grad.zero_()
table(["迭代", "w", "b", "损失 L"], rows)
check("学到 w ≈ 2", abs(w.item() - 2.0) < 0.05)
check("学到 b ≈ 1", abs(b.item() - 1.0) < 0.05)

# 和第 3 章 numpy 手写梯度的结果比一比（同一份数据、同一个 α）
w_np, b_np = 0.0, 0.0
for _ in range(201):
    err = w_np * X_np + b_np - Y_np
    w_np -= alpha * np.mean(2 * err * X_np)
    b_np -= alpha * np.mean(2 * err)
table(["", "w", "b"], [["torch autograd", w.item(), b.item()], ["numpy 手写梯度(第3章)", w_np, b_np]])
check("torch 自动梯度 = 手写梯度", abs(w.item() - w_np) < 1e-3 and abs(b.item() - b_np) < 1e-3)

# ---------------------------------------------------------------------------
banner("5. 泛化：模型在“没见过的点”上表现如何")
X_new = torch.linspace(-1, 1, 5).unsqueeze(1)
Y_true = 2.0 * X_new + 1.0
with torch.no_grad():
    Y_pred = model(X_new)
table(["新的 x", "真值 2x+1", "模型预测"], [[X_new[i].item(), Y_true[i].item(), Y_pred[i].item()] for i in range(5)])
check("新点上误差 < 0.05", (Y_pred - Y_true).abs().max().item() < 0.05)
print("模型只见过 50 个带噪声的点，却在任何 x 上都能预测——这就是“学到了规律”而不是“背下了数据”。")

done()
