"""第 19 章实验：打开一个导出的 policy.onnx，看它的图结构（归一化 + 4 层 MLP），
用 numpy 从 model_N.pt 的权重手写同样的计算，和 onnxruntime 逐位对比；验证输出是均值、不带噪声。

运行：uv run python docs/learn-zh/labs/ch19_onnx_vs_torch.py [路径/到/run_dir]
默认找 logs/rsl_rl/velocity/ 下最新的 learnzh-* 冒烟训练目录（第 14 章实验会生成）。纯 CPU。
# LAB_REQUIRES: run_dir
"""

import glob
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from _common import banner, check, done, table

np.set_printoptions(precision=4, suppress=True)
REPO = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
banner("1. 找到一次训练的产物：model_N.pt 和 .onnx")
if len(sys.argv) > 1:
    run_dir = Path(sys.argv[1])
else:
    cands = [Path(d) for d in sorted(glob.glob(str(REPO / "logs" / "rsl_rl" / "velocity" / "*learnzh-*")))
             if list(Path(d).glob("*.onnx")) and list(Path(d).glob("model_*.pt"))]
    if not cands:
        print("没有找到同时含 .onnx 和 model_*.pt 的 learnzh-* 运行目录，先跑 ch14_smoke_train.py（需要 GPU），或传入一个 run_dir。")
        sys.exit(1)
    run_dir = cands[-1]
onnx_path = sorted(run_dir.glob("*.onnx"))[-1]
ckpt_path = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[-1]
print("run_dir :", run_dir.relative_to(REPO))
print("onnx    :", onnx_path.name)
print("ckpt    :", ckpt_path.name)

# ---------------------------------------------------------------------------
banner("2. ONNX 图：输入/输出名和形状、算子序列")
m = onnx.load(str(onnx_path))
inp = m.graph.input[0]; out = m.graph.output[0]
print("输入 :", inp.name, [d.dim_value for d in inp.type.tensor_type.shape.dim])
print("输出 :", out.name, [d.dim_value for d in out.type.tensor_type.shape.dim])
ops = [n.op_type for n in m.graph.node]
print("算子 :", " → ".join(ops))
print("      Sub, Div = 归一化 (x − μ)/(σ + ε)；Gemm = 矩阵乘 + 偏置；Elu = 门。4 个 Gemm、3 个 Elu = 第 6 章的四层。")
check("输入是 obs [1, 61]", inp.name == "obs" and [d.dim_value for d in inp.type.tensor_type.shape.dim] == [1, 61])
check("输出是 actions [1, 14]", out.name == "actions" and [d.dim_value for d in out.type.tensor_type.shape.dim] == [1, 14])
check("图里有归一化算子", ops[:2] == ["Sub", "Div"])
meta = {p.key: p.value for p in m.metadata_props}
print("metadata.observation_names =", meta.get("observation_names"))
print("metadata.action_scale      =", meta.get("action_scale"))

# ---------------------------------------------------------------------------
banner("3. 从 model_N.pt 取出 actor 的全部旋钮和归一化统计")
ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
sd = ck["actor_state_dict"]
rows = [[k, tuple(v.shape)] for k, v in sd.items()]
table(["actor_state_dict 里的键", "形状"], rows)
mean = sd["obs_normalizer._mean"].numpy()[0]
std = sd["obs_normalizer._std"].numpy()[0]
W = [sd[f"mlp.{i}.weight"].numpy() for i in (0, 2, 4, 6)]
b = [sd[f"mlp.{i}.bias"].numpy() for i in (0, 2, 4, 6)]
sigma = sd["distribution.std_param"].numpy()
print("归一化 μ 的前 6 项:", mean[:6], " σ 的前 6 项:", std[:6])
print("动作噪声 std_param 前 4 项:", sigma[:4], "（训练用；ONNX 里没有它）")
check("ONNX 的 initializer 里没有 std_param", not any("std_param" in t.name for t in m.graph.initializer))


# ---------------------------------------------------------------------------
banner("4. numpy 手写：归一化 → 四层 MLP，和 onnxruntime 逐位对比")
def elu(z):
    return np.where(z > 0, z, np.exp(z) - 1.0)


def policy_numpy(obs):
    x = (obs - mean) / (std + 0.01)          # 第 8 章：eps = 0.01
    h = elu(W[0] @ x + b[0])
    h = elu(W[1] @ h + b[1])
    h = elu(W[2] @ h + b[2])
    return W[3] @ h + b[3]                   # 输出 = 均值 μ，不加噪声


sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
rng = np.random.default_rng(0)
max_diff = 0.0
for k in range(5):
    obs = (rng.normal(size=61) * 0.5).astype(np.float32)
    out_np = policy_numpy(obs)
    out_ort = sess.run(["actions"], {"obs": obs[None, :]})[0][0]
    max_diff = max(max_diff, float(np.max(np.abs(out_np - out_ort))))
    if k == 0:
        table(["", "前 5 个动作"], [["numpy 手写", str(out_np[:5])], ["onnxruntime", str(out_ort[:5])]])
print(f"5 帧随机观测的最大差异 = {max_diff:.2e}")
check("numpy 手写 = onnxruntime（差异 < 1e-4）", max_diff < 1e-4)

# ---------------------------------------------------------------------------
banner("5. 确定性：同一帧观测跑两次，输出完全一样（部署不采样）")
obs = (rng.normal(size=61) * 0.5).astype(np.float32)
o1 = sess.run(["actions"], {"obs": obs[None, :]})[0]
o2 = sess.run(["actions"], {"obs": obs[None, :]})[0]
check("两次输出完全相同", np.array_equal(o1, o2))
print("训练时动作 = μ + σ·ε（第 4 章），部署时只用 μ。σ 留在 model_N.pt 里，导出时丢掉。")

# ---------------------------------------------------------------------------
banner("6. 忘记烘焙归一化会怎样：直接把原始观测喂给“没有 Sub/Div 的网络”")
def policy_no_norm(obs):
    h = elu(W[0] @ obs + b[0]); h = elu(W[1] @ h + b[1]); h = elu(W[2] @ h + b[2])
    return W[3] @ h + b[3]


obs = (rng.normal(size=61) * 0.5).astype(np.float32)
obs[20:34] *= 8.0                                  # 关节速度那一块幅度大（第 8 章 8.2）
a_ok = policy_numpy(obs); a_bad = policy_no_norm(obs)
table(["", "前 5 个动作"], [["烘焙了归一化", str(a_ok[:5])], ["没烘焙", str(a_bad[:5])]])
print(f"差异 = {np.max(np.abs(a_ok - a_bad)):.3f} rad ≈ {np.degrees(np.max(np.abs(a_ok - a_bad))):.0f}°——同一帧观测，动作面目全非。")

done()
