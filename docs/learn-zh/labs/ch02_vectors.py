"""第 2 章实验：向量、点积、范数、矩阵乘法，以及“一层神经网络就是一次矩阵乘”。

运行：uv run python docs/learn-zh/labs/ch02_vectors.py
纯 CPU，numpy + torch。
"""

import math

import numpy as np
import torch

from _common import banner, check, done, table

np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------------------------------------
banner("1. 向量就是定长数组：加法、数乘、点积、长度")
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, -1.0, 0.5])
print("a =", a)
print("b =", b)
print("a + b       =", a + b)
print("2 * a       =", 2 * a)
dot_manual = sum(ai * bi for ai, bi in zip(a, b))
print("a·b (手算)  =", dot_manual, "  = 1·4 + 2·(-1) + 3·0.5")
print("a·b (numpy) =", a @ b)
norm_manual = math.sqrt(sum(ai * ai for ai in a))
print("‖a‖ (手算)  =", norm_manual, "  = sqrt(1² + 2² + 3²)")
print("‖a‖ (numpy) =", np.linalg.norm(a))
check("点积手算 = numpy", abs(dot_manual - a @ b) < 1e-12)
check("范数手算 = numpy", abs(norm_manual - np.linalg.norm(a)) < 1e-12)
unit = a / np.linalg.norm(a)
print("a 的单位向量 =", unit, "  长度 =", np.linalg.norm(unit))
cos_theta = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"a、b 夹角的余弦 = {cos_theta:.4f}  → 夹角 {math.degrees(math.acos(cos_theta)):.1f}°")

# ---------------------------------------------------------------------------
banner("2. 重力向量在“身体坐标系”里的投影（projected_gravity 的直觉）")
print("世界坐标系里重力永远是 g = (0, 0, -1)（单位向量）。")
print("机器人向前倾 θ 度，从机器人自己看，重力就“往前偏”了。")
rows = []
for deg in [0, 10, 30, 70]:
    th = math.radians(deg)
    # 绕 y 轴旋转 θ 后的重力在身体系里的 (x, z) 分量（二维旋转矩阵）
    gx = -math.sin(th)
    gz = -math.cos(th)
    rows.append([deg, gx, 0.0, gz, gx**2])
table(["前倾角°", "g_x(身体系)", "g_y", "g_z", "g_x²+g_y²"], rows)
print("最后一列正是 upright 奖励里的 xy_squared = sin²(tilt)；fell_over 终止阈值是 70°。")

# ---------------------------------------------------------------------------
banner("3. 矩阵 × 向量 = 每一行做一次点积")
W = np.array([[1.0, 0.0, 2.0],
              [0.5, -1.0, 1.0]])
x = np.array([1.0, 2.0, 3.0])
row0 = W[0] @ x
row1 = W[1] @ x
print("W =\n", W)
print("x =", x)
print("第 0 行·x =", row0, "  第 1 行·x =", row1)
print("W @ x     =", W @ x)
check("矩阵乘 = 逐行点积", np.allclose(W @ x, [row0, row1]))
print("形状规则：(2×3) @ (3,) → (2,)   ——  输入 3 个数，输出 2 个数")

# ---------------------------------------------------------------------------
banner("4. 项目 actor 的第一层：61 维观测 → 512 个数")
torch.manual_seed(0)
layer = torch.nn.Linear(61, 512)          # 和 rsl_rl 里 MLP 的第一层完全一样
obs = torch.randn(61)                     # 假装这是一帧 61 维观测
out_torch = layer(obs)
W_np = layer.weight.detach().numpy()      # 形状 (512, 61)
b_np = layer.bias.detach().numpy()        # 形状 (512,)
out_np = W_np @ obs.numpy() + b_np        # y = W x + b
print("weight 形状 =", tuple(W_np.shape), " bias 形状 =", tuple(b_np.shape))
print("参数个数 = 512×61 + 512 =", 512 * 61 + 512)
print("前 5 个输出 (torch) =", out_torch[:5].detach().numpy())
print("前 5 个输出 (numpy) =", out_np[:5])
check("torch.nn.Linear == W@x + b", np.allclose(out_torch.detach().numpy(), out_np, atol=1e-5))

n_params = sum(p.numel() for p in layer.parameters())
check("参数个数 = 31744", n_params == 31744)

# ---------------------------------------------------------------------------
banner("5. 转置：把行变成列")
print("W.T =\n", W.T, "\n形状", W.shape, "→", W.T.shape)

done()
