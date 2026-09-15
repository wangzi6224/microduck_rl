"""第 8 章实验：为什么要归一化输入；在线均值/方差（EmpiricalNormalization）手写对比；训练时更新、推理时冻结。

运行：uv run python docs/learn-zh/labs/ch08_normalization.py
纯 CPU。
"""

import numpy as np
import torch

from _common import banner, check, done, table

torch.manual_seed(0)
rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------------------------------------
banner("1. 第 3 章的伏笔：x 的范围放大 10 倍，同一个学习率就发散")
Y_fn = lambda X: 2.0 * X + 1.0 + rng.normal(0, 0.1, size=len(X))


def gd(X, Y, alpha=0.1, steps=100):
    w, b = 0.0, 0.0
    for _ in range(steps):
        err = w * X + b - Y
        w -= alpha * np.mean(2 * err * X)
        b -= alpha * np.mean(2 * err)
    L = np.mean((w * X + b - Y) ** 2)
    return w, b, (L if np.isfinite(L) else float("inf"))


X_small = rng.uniform(-1, 1, 50); Y_small = Y_fn(X_small)
X_big = rng.uniform(-10, 10, 50); Y_big = Y_fn(X_big)
rows = [["x ∈ (−1, 1)", *gd(X_small, Y_small)], ["x ∈ (−10, 10)", *gd(X_big, Y_big)]]
# 把大范围的 x 归一化：减均值、除标准差
mu, sd = X_big.mean(), X_big.std()
X_norm = (X_big - mu) / sd
w_n, b_n, L_n = gd(X_norm, Y_big)
rows.append(["x ∈ (−10,10) 但归一化后", w_n, b_n, L_n])
table(["数据", "w", "b", "100 步后损失"], rows)
check("大范围输入发散（损失爆炸）", rows[1][3] > 1e6)
check("归一化后收敛", rows[2][3] < 0.05)
print("归一化后的 w 不再是 2，因为它现在是“y 对 (x−μ)/σ 的斜率”= 2σ ≈", 2 * sd)

# ---------------------------------------------------------------------------
banner("2. 观测的各项量纲差别有多大（模拟一帧 61 维观测）")
obs_ranges = [
    ("base_ang_vel (rad/s)", 3, 3.0),
    ("projected_gravity (无量纲)", 3, 1.0),
    ("joint_pos−HOME (rad)", 14, 0.5),
    ("joint_vel (rad/s)", 14, 8.0),
    ("last_action (rad)", 14, 0.5),
    ("command (混合单位)", 13, 0.4),
]
rows = [[n, d, f"±{s}"] for n, d, s in obs_ranges]
table(["观测块", "维数", "典型幅度"], rows)
print("关节速度的数值可以是投影重力的 8 倍以上。不归一化，第一层的权重就得为不同列准备完全不同的尺度。")

# ---------------------------------------------------------------------------
banner("3. 在线均值/方差：手写 vs rsl_rl 的 EmpiricalNormalization")
from rsl_rl.modules import EmpiricalNormalization  # noqa: E402

D = 4
norm = EmpiricalNormalization(D)
norm.train()

# 手写版：和 normalization.py:50-66 一模一样的公式
count = 0
mean = torch.zeros(1, D)
var = torch.ones(1, D)
all_x = []
for step in range(5):
    x = torch.randn(64, D) * torch.tensor([1.0, 8.0, 0.5, 3.0]) + torch.tensor([0.0, 2.0, -1.0, 5.0])
    all_x.append(x)
    norm.update(x)                                  # rsl_rl
    n_x = x.shape[0]                                # 手写
    count += n_x
    rate = n_x / count
    var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
    mean_x = torch.mean(x, dim=0, keepdim=True)
    delta = mean_x - mean
    mean = mean + rate * delta
    var = var + rate * (var_x - var + delta * (mean_x - mean))
X_all = torch.cat(all_x)
rows = []
for j in range(D):
    rows.append([j, norm.mean[j].item(), mean[0, j].item(), X_all[:, j].mean().item(), norm.std[j].item(), X_all[:, j].std(unbiased=False).item()])
table(["维", "rsl_rl 均值", "手写均值", "真实均值", "rsl_rl std", "真实 std"], rows)
check("手写在线均值 = rsl_rl", torch.allclose(norm.mean, mean.squeeze(0), atol=1e-5))
check("在线 std ≈ 全量 std", torch.allclose(norm.std, X_all.std(dim=0, unbiased=False), atol=1e-4))
print("每来一批数据只更新一次 mean/var，不用把历史数据存下来——这就是“在线”（running）统计。")

# ---------------------------------------------------------------------------
banner("4. forward = (x − mean) / (std + eps)；eval 模式下 update 是空操作")
x = torch.randn(3, D) * 5
y = norm(x)
y_manual = (x - norm.mean) / (norm.std + norm.eps)
check("forward 公式", torch.allclose(y, y_manual, atol=1e-6))
print("eps =", norm.eps, "（防止某一维 std=0 时除以 0）")

m_before = norm.mean.clone()
norm.eval()
norm.update(torch.randn(64, D) * 100)          # 喂离谱的数据
check("eval 模式下统计量不变", torch.allclose(norm.mean, m_before))
print("训练时 norm.train() 每批更新；回放/导出时 norm.eval() 冻结——ONNX 里烘进去的就是冻结那一刻的 mean/std。")

# ---------------------------------------------------------------------------
banner("5. “烘进模型”是什么意思：f(normalize(x)) 合成一个函数")
net = torch.nn.Sequential(torch.nn.Linear(D, 8), torch.nn.ELU(), torch.nn.Linear(8, 2))
baked = torch.nn.Sequential(norm, net)          # 归一化器 + 网络 = 一个新模块
with torch.no_grad():
    x = torch.randn(1, D) * 5
    a = net(norm(x))
    b = baked(x)
check("baked(x) == net(norm(x))", torch.allclose(a, b))
print("导出 ONNX 时导的是 baked，所以真机只需要喂原始观测，不用自己再减均值除标准差。")

done()
