"""第 7 章实验：反向传播 = 链式法则；SGD / 动量 / Adam 三种优化器；mini-batch 训练一个小网络。

运行：uv run python docs/learn-zh/labs/ch07_backprop.py
纯 CPU。
"""

import math

import numpy as np
import torch

from _common import banner, check, done, savefig, table, use_headless_matplotlib

torch.manual_seed(0)
np.set_printoptions(precision=5, suppress=True)

# ---------------------------------------------------------------------------
banner("1. 手算一个两层小网络的梯度，与 autograd 对比")
# 网络：x → [w1] → z → ELU → h → [w2] → y；损失 L = (y − t)²
x, t = 1.5, 2.0
w1 = torch.tensor(0.8, requires_grad=True)
w2 = torch.tensor(-0.5, requires_grad=True)

z = w1 * x
h = torch.nn.functional.elu(z)
y = w2 * h
L = (y - t) ** 2
L.backward()

# 链式法则手算（从外往里一层层乘）
dL_dy = 2 * (y.item() - t)                 # 外层：(y−t)² 对 y
dy_dw2 = h.item()                          # y = w2·h 对 w2
dy_dh = w2.item()                          # y = w2·h 对 h
dh_dz = 1.0 if z.item() > 0 else math.exp(z.item())   # ELU 的导数
dz_dw1 = x                                 # z = w1·x 对 w1
grad_w2 = dL_dy * dy_dw2
grad_w1 = dL_dy * dy_dh * dh_dz * dz_dw1
table(["", "手算链式法则", "autograd"], [["∂L/∂w2", grad_w2, w2.grad.item()], ["∂L/∂w1", grad_w1, w1.grad.item()]])
check("手算 = autograd", abs(grad_w2 - w2.grad.item()) < 1e-5 and abs(grad_w1 - w1.grad.item()) < 1e-5)
print("∂L/∂w1 = (∂L/∂y)(∂y/∂h)(∂h/∂z)(∂z/∂w1) —— 四个“汇率”相乘，从最外层乘到最里层。")

# ---------------------------------------------------------------------------
banner("2. 一个“扁碗” f = x² + 100y²：y 方向比 x 方向陡 100 倍，从 (3, 2) 出发走 100 步")


def f(p):
    return p[0] ** 2 + 100 * p[1] ** 2


def run(opt_name, lr, steps=100):
    p = torch.tensor([3.0, 2.0], requires_grad=True)
    if opt_name == "sgd":
        opt = torch.optim.SGD([p], lr=lr)
    elif opt_name == "momentum":
        opt = torch.optim.SGD([p], lr=lr, momentum=0.9)
    else:
        opt = torch.optim.Adam([p], lr=lr)
    path = [p.detach().clone().numpy()]
    for _ in range(steps):
        opt.zero_grad()
        f(p).backward()
        opt.step()
        path.append(p.detach().clone().numpy())
    return np.array(path)


paths = {"SGD lr=0.1": run("sgd", 0.1), "SGD lr=0.005": run("sgd", 0.005), "Adam lr=0.3": run("adam", 0.3)}
rows = []
for k, v in paths.items():
    fv = [f(torch.tensor(v[i])).item() for i in (10, 30, 60, 100)]
    rows.append([k] + [x if np.isfinite(x) and x < 1e12 else float("inf") for x in fv])
table(["优化器", "10 步后 f", "30 步后 f", "60 步后 f", "100 步后 f"], rows)
check("SGD lr=0.1 在陡方向发散", rows[0][4] == float("inf"))
check("SGD lr=0.005 100 步还没到底", rows[1][4] > 0.5)
check("Adam lr=0.3 100 步到底", rows[2][4] < 0.01)
print("同一个学习率：SGD 在陡的 y 方向飞出去；调小到 0.005 又在平的 x 方向爬不动。Adam 给每个方向各自定步长，两头都顾到。")

plt = use_headless_matplotlib()
fig, ax = plt.subplots(figsize=(6, 5))
gx, gy = np.meshgrid(np.linspace(-3.5, 3.5, 200), np.linspace(-2.5, 2.5, 200))
ax.contour(gx, gy, gx**2 + 100 * gy**2, levels=[1, 5, 20, 60, 150, 300], cmap="Greys")
for (k, v), c in zip(paths.items(), ["tab:red", "tab:green", "tab:blue"]):
    vv = v[np.all(np.abs(v) < 4, axis=1)]          # 发散的路线只画还在图里的部分
    ax.plot(vv[:, 0], vv[:, 1], "o-", ms=3, lw=1, color=c, label=k)
ax.plot(0, 0, "k*", ms=12)
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-2.5, 2.5); ax.legend(); ax.grid(alpha=0.2)
ax.set_title("扁碗 x²+100y²：SGD 要么飞要么慢，Adam 两头兼顾")
savefig(fig, "ch07_optimizers")

# ---------------------------------------------------------------------------
banner("3. Adam 的一步：手算 vs torch.optim.Adam")
p = torch.tensor([3.0, 2.0], requires_grad=True)
opt = torch.optim.Adam([p], lr=0.3)          # 默认 β1=0.9, β2=0.999, ε=1e-8
opt.zero_grad(); f(p).backward()
g = p.grad.clone()
opt.step()
# 手算第 1 步（t=1）
beta1, beta2, eps, lr = 0.9, 0.999, 1e-8, 0.3
m = (1 - beta1) * g                            # 一阶动量（梯度的滑动平均）
v = (1 - beta2) * g * g                        # 二阶动量（梯度平方的滑动平均）
m_hat = m / (1 - beta1**1)                     # 偏差修正
v_hat = v / (1 - beta2**1)
p_manual = torch.tensor([3.0, 2.0]) - lr * m_hat / (torch.sqrt(v_hat) + eps)
table(["", "x", "y"], [["梯度 g", g[0].item(), g[1].item()], ["手算 Adam 一步", p_manual[0].item(), p_manual[1].item()], ["torch Adam 一步", p[0].item(), p[1].item()]])
check("手算 Adam = torch", torch.allclose(p_manual, p.detach(), atol=1e-4))
print("注意：梯度 g 在 y 方向是 x 方向的 67 倍，Adam 第一步却在两个方向各走 0.3——它把每个方向的步长自动归一了。")

# ---------------------------------------------------------------------------
banner("4. mini-batch 训练一个小 MLP 拟合 sin(x)")
X = torch.linspace(-math.pi, math.pi, 400).unsqueeze(1)
Y = torch.sin(X)
net = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.ELU(), torch.nn.Linear(32, 32), torch.nn.ELU(), torch.nn.Linear(32, 1))
opt = torch.optim.Adam(net.parameters(), lr=1e-2)
batch_size, epochs = 64, 60
rows = []
n_updates = 0
for epoch in range(epochs):
    perm = torch.randperm(len(X))                     # 打乱
    for i in range(0, len(X), batch_size):            # 每次只用 64 个点算梯度
        idx = perm[i:i + batch_size]
        opt.zero_grad()
        loss = torch.mean((net(X[idx]) - Y[idx]) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)   # 第 3 章的梯度裁剪
        opt.step()
        n_updates += 1
    if epoch in (0, 1, 2, 5, 10, 20, 40, 59):
        with torch.no_grad():
            rows.append([epoch, torch.mean((net(X) - Y) ** 2).item()])
table(["epoch", "全部 400 点的损失"], rows)
print(f"400 个点 / 64 每批 = 每 epoch {math.ceil(400/64)} 次更新，{epochs} 个 epoch 共 {n_updates} 次更新。")
check("最终损失 < 0.01", rows[-1][1] < 0.01)

fig, ax = plt.subplots(figsize=(6, 3.4))
with torch.no_grad():
    ax.plot(X.squeeze(), Y.squeeze(), label="真值 sin(x)")
    ax.plot(X.squeeze(), net(X).squeeze(), "--", label="MLP 拟合")
ax.legend(); ax.grid(alpha=0.3); ax.set_title("一个 1→32→32→1 的 MLP 学会了 sin")
savefig(fig, "ch07_sin_fit")

done()
