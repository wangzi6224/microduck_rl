"""第 6 章实验：神经元、激活函数、多层感知机（MLP）。
用 numpy 手写项目 actor 的四层前向计算，与 rsl_rl 的 MLP 类逐位对比，并数出参数个数。

运行：uv run python docs/learn-zh/labs/ch06_mlp.py
纯 CPU。
"""

import numpy as np
import torch

from _common import banner, check, done, table

torch.manual_seed(0)
np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------------------------------------
banner("1. 一个神经元 = 点积 + 偏置 + 激活函数")
x = np.array([1.0, 2.0, 3.0])          # 输入 3 项
w = np.array([0.5, -1.0, 0.2])         # 这个神经元的“价格表”
b = 0.1                                # 偏置
z = w @ x + b                          # 加权和（第 2 章的点积）
print("z = w·x + b =", z)


def relu(z):
    return np.maximum(0.0, z)


def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(z) - 1.0))


rows = []
for zz in [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]:
    rows.append([zz, float(relu(zz)), float(elu(zz))])
table(["z", "ReLU(z)", "ELU(z)"], rows)
print("ReLU：负数一律砍成 0。ELU：负数平滑地压到 −1 以上，正数原样通过。项目用 ELU。")
check("ELU 在 z>0 处等于 z", elu(2.0) == 2.0)
check("ELU 在 z→−∞ 处趋近 −1", abs(elu(-10.0) - (-1.0)) < 1e-4)
check("ELU 与 torch.nn.ELU 一致", np.allclose(elu(np.array([-2.0, -0.3, 0.7])), torch.nn.ELU()(torch.tensor([-2.0, -0.3, 0.7])).numpy(), atol=1e-6))

# ---------------------------------------------------------------------------
banner("2. 为什么中间要夹非线性：两层纯线性 = 一层线性")
rng = np.random.default_rng(0)
W1 = rng.normal(size=(4, 3))
W2 = rng.normal(size=(2, 4))
x = rng.normal(size=3)
two_layers = W2 @ (W1 @ x)
one_layer = (W2 @ W1) @ x
print("W2 @ (W1 @ x) =", two_layers)
print("(W2 @ W1) @ x =", one_layer, "  ← 同一个 2×3 的矩阵就能替代")
check("没有激活函数时叠层是徒劳", np.allclose(two_layers, one_layer))
with_act = W2 @ elu(W1 @ x)
print("W2 @ ELU(W1 @ x) =", with_act, "  ← 夹了 ELU 就不能再合并了")

# ---------------------------------------------------------------------------
banner("3. 项目 actor 的四层：61 → 512 → 256 → 128 → 14（ELU）")
from rsl_rl.modules import MLP  # noqa: E402

net = MLP(input_dim=61, output_dim=14, hidden_dims=(512, 256, 128), activation="elu")
print(net)

rows = []
total = 0
for name, p in net.named_parameters():
    rows.append([name, tuple(p.shape), p.numel()])
    total += p.numel()
table(["参数名", "形状", "个数"], rows)
print("网络参数总数 =", total)
check("网络参数 197,774 个", total == 197774)
print("再加上第 4 章的 14 个 σ，actor 一共", total + 14, "个旋钮。")

# ---------------------------------------------------------------------------
banner("4. numpy 手写四层前向，与 rsl_rl 的 MLP 逐位对比")
obs = torch.randn(61)
with torch.no_grad():
    out_torch = net(obs).numpy()

# 把 torch 的权重拷出来，用 numpy 重算
Ws = [net[i].weight.detach().numpy() for i in (0, 2, 4, 6)]
bs = [net[i].bias.detach().numpy() for i in (0, 2, 4, 6)]
h = obs.numpy()
h = elu(Ws[0] @ h + bs[0])      # 61 → 512，ELU
h = elu(Ws[1] @ h + bs[1])      # 512 → 256，ELU
h = elu(Ws[2] @ h + bs[2])      # 256 → 128，ELU
out_np = Ws[3] @ h + bs[3]      # 128 → 14，最后一层不加激活（输出可正可负）
print("torch 前 5 个输出:", out_torch[:5])
print("numpy 前 5 个输出:", out_np[:5])
check("手写前向 = rsl_rl MLP", np.allclose(out_torch, out_np, atol=1e-5))

# ---------------------------------------------------------------------------
banner("5. 同一个网络，不同观测 → 不同输出（它是一个 R^61 → R^14 的函数）")
with torch.no_grad():
    for k in range(3):
        o = torch.randn(61)
        print(f"  观测 {k}: 输出前 3 项 = {net(o).numpy()[:3]}")

done()
