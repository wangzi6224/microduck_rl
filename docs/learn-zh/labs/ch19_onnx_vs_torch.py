"""第 19 章实验：从检查点到 ONNX，再到真机每 0.02 秒的那一圈。

先用一个 3 进 2 出的迷你网络把整件事做一遍（自己搭、自己导出、自己手算，全部能对上），
再打开项目真导出的 policy.onnx：九个算子、numpy 手写 = onnxruntime、部署不采样、
漏烘焙会怎样、拼一帧 61 维跑一次、最后核对 .pt 和 .onnx 是不是同一个检查点。

运行：uv run python docs/learn-zh/labs/ch19_onnx_vs_torch.py
指定运行目录：LEARNZH_RUN_DIR=logs/rsl_rl/velocity/<某个训练目录> uv run python docs/learn-zh/labs/ch19_onnx_vs_torch.py
不给就找 logs/rsl_rl/velocity/ 下最新的、同时含 .onnx 和 model_*.pt 的 learnzh-* 目录（第 14 章实验生成）。
纯 CPU，不训练、不建仿真环境。第 2、6、8、9 节的数全是手算得出来的，没有运行目录也照样跑。
小节编号与正文一一对应：实验第 K 节 = 正文 19.K 节（第 11 节对应「映射到项目」）。
正文“改一改”要改的三行都带 `# TWEAK-k:` 标记（第 2 节两处、第 4 节一处）。
# LAB_REQUIRES: run_dir
"""

import contextlib
import io
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnx import numpy_helper

from _common import banner, check, done, lines_in_order, num, savefig, table
from _draw import (BLUE, CELL, CELL_EDGE, CELL_HOT, FAINT, FS_SMALL, FS_TICK, GREEN, INK, MUTED, ORANGE,
                   arrow, cell, hand, lesson_cells, lesson_figure, lesson_panel, note, plt)

np.set_printoptions(precision=4, suppress=True)
def _find_repo() -> Path:
    """仓库根目录。脚本被复制到别处跑时（“改一改”的副本），按当前目录再找一遍。"""
    for base in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for cand in (base, *base.parents):
            if (cand / "src" / "mjlab_microduck").is_dir():
                return cand
    return Path(__file__).resolve().parents[3]


REPO = _find_repo()
TMP = Path(tempfile.mkdtemp(prefix="learnzh-ch19-"))


def minus(s: str) -> str:
    """把 ASCII 的减号换成正文里用的 −，打印出来的表才和正文一个样。"""
    return s.replace("-", "−")


def vec(a, places: int = 4) -> str:
    """把一个小数组打印成正文里的写法 (1, 2, 0.5)，去掉尾零。"""
    return minus("(" + ", ".join(num(float(x), places) for x in np.ravel(a)) + ")")


def draw_chain(ax, boxes, y, x0=0.35, w=1.45, gap=0.42, height=0.78, hot=(), fontsize=FS_SMALL):
    """画一排带箭头的算子方框，返回每个方框的中心横坐标。"""
    centers = []
    for i, text in enumerate(boxes):
        x = x0 + i * (w + gap)
        cell(ax, x, y, text, width=w, height=height,
             facecolor=CELL_HOT if i in hot else CELL,
             edgecolor=ORANGE if i in hot else CELL_EDGE, fontsize=fontsize)
        centers.append(x + w / 2)
        if i:
            arrow(ax, (x - gap, y + height / 2), (x, y + height / 2), MUTED, lw=2)
    return centers


REAL_OPS = ["Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"]


# ---------------------------------------------------------------------------
banner("1. 两个文件：一次训练留下的 model_N.pt 和 .onnx")

run_dir = None
env_dir = os.environ.get("LEARNZH_RUN_DIR")
if env_dir:
    cand = Path(env_dir)
    run_dir = cand if cand.is_absolute() else REPO / cand
else:
    usable = [d for d in sorted((REPO / "logs" / "rsl_rl" / "velocity").glob("*learnzh-*"))
              if list(d.glob("*.onnx")) and list(d.glob("model_*.pt"))]
    run_dir = usable[-1] if usable else None

HAVE_RUN = run_dir is not None and run_dir.is_dir() and list(run_dir.glob("*.onnx")) and list(run_dir.glob("model_*.pt"))
if not HAVE_RUN:
    print("没有找到同时含 .onnx 和 model_*.pt 的运行目录。")
    print("第 1、3、4、5、7、10 节会跳过；第 2、6、8、9 节的手算不需要运行目录，照常跑。")
    print("要跑全：先做第 14 章的冒烟训练（需要 GPU），或者设 LEARNZH_RUN_DIR=<某个训练目录>。")
    onnx_path = ckpt_paths = None
else:
    onnx_path = sorted(run_dir.glob("*.onnx"))[-1]
    ckpt_paths = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    ckpt_path = ckpt_paths[-1]
    print("运行目录 :", run_dir.relative_to(REPO))
    rows = [[p.name, p.stat().st_size] for p in ckpt_paths] + [[onnx_path.name, onnx_path.stat().st_size]]
    table(["文件", "字节数"], rows)
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    print("检查点里的五样东西 :", ", ".join(ck.keys()))
    sd = ck["actor_state_dict"]
    table(["actor_state_dict 里的键", "形状"], [[k, tuple(v.shape)] for k, v in sd.items()])
    mean = sd["obs_normalizer._mean"].numpy()[0]
    std = sd["obs_normalizer._std"].numpy()[0]
    W = [sd[f"mlp.{i}.weight"].numpy() for i in (0, 2, 4, 6)]
    B = [sd[f"mlp.{i}.bias"].numpy() for i in (0, 2, 4, 6)]
    sigma = sd["distribution.std_param"].numpy()
    n_mlp = sum(v.size for k, v in ((k, v.numpy()) for k, v in sd.items()) if k.startswith("mlp."))
    print(f"四层网络的权重和偏置一共 {n_mlp:,} 个数（第 6 章 6.5 节数过）；14 个动作 σ 另算。")
    check("检查点里有 5 样东西：actor、critic、优化器、圈号、infos（第 14 章 14.9 节）",
          set(ck.keys()) == {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter", "infos"})
    check("actor 的四层网络是 197,774 个权重和偏置", n_mlp == 197774)
    size_ratio = ckpt_path.stat().st_size / onnx_path.stat().st_size
    print(f"检查点 ÷ ONNX = {size_ratio:.2f}（正文说的“差不多 6 倍”，第 14 章 14.9 节量过）")
    check("检查点比 ONNX 大（里面还装着 critic 和 Adam 的两本账）",
          ckpt_path.stat().st_size > onnx_path.stat().st_size)
    check(f"检查点差不多是 ONNX 的 6 倍（这次是 {size_ratio:.2f} 倍）", round(size_ratio) == 6)

# ---------------------------------------------------------------------------
banner("1b. 画图：figures/ch19_overview.png（本章地图：两个文件、一串算子、一圈 0.02 秒）")
fig, axes = lesson_figure(3, "训练留下两个文件：一个留着接着练，一个交给机器人", panel_height=3.1, width=11.5)

ax = axes[0]
lesson_panel(ax, "① 一次训练结束，运行目录里躺着这两样", xmax=11.5)
ax.add_patch(plt.Rectangle((0.3, 1.05), 5.0, 2.1, facecolor=CELL, edgecolor=CELL_EDGE, lw=1.8))
ax.text(2.8, 2.85, "model_N.pt  检查点", fontsize=FS_SMALL, ha="center", color=INK)
ax.text(2.8, 1.95, "actor ＋ critic ＋ Adam 的两本账\n＋ 圈号 ＋ 步数（第 14 章 14.9 节）",
        fontsize=15, ha="center", va="center", color=MUTED, linespacing=1.6)
ax.text(2.8, 1.25, "→ 接着练、在仿真里回放", fontsize=15, ha="center", color=BLUE)
ax.add_patch(plt.Rectangle((6.2, 1.05), 5.0, 2.1, facecolor=CELL_HOT, edgecolor=ORANGE, lw=1.8))
ax.text(8.7, 2.85, "xxx.onnx", fontsize=FS_SMALL, ha="center", color=INK)
ax.text(8.7, 1.95, "只有 actor 的那一串计算步骤\n（归一化已经烘在里面）",
        fontsize=15, ha="center", va="center", color=MUTED, linespacing=1.6)
ax.text(8.7, 1.25, "→ 交给机器人", fontsize=15, ha="center", color=ORANGE)
note(ax, 0.3, 0.45, "左边那个大约是右边的 6 倍（第 14 章 14.9 节量过）：多出来的全是训练才要的东西。")

ax = axes[1]
lesson_panel(ax, "② 打开 .onnx：里面是一串“怎么算”，不是一段录好的动作", xmax=11.5)
ax.text(0.3, 2.35, "61 个数 →", fontsize=FS_SMALL, va="center", color=INK)
draw_chain(ax, REAL_OPS, y=1.95, x0=1.68, w=0.83, gap=0.145, height=0.78, fontsize=15)
ax.text(10.45, 2.35, "→ 14 个数", fontsize=FS_SMALL, va="center", color=INK)
note(ax, 0.3, 1.25, "九个算子，名字 19.2 节逐个讲。喂不同的观测，它就算出不同的动作。")

ax = axes[2]
lesson_panel(ax, "③ 机器人上，这个文件每 0.02 秒被用一次", xmax=11.5)
loop = ["读传感器\n拼成 61 个数", "交给 .onnx\n拿回 14 个数", "HOME + 动作\n= 目标角", "物理走 4 小步\n再回到第一步"]
for i, text in enumerate(loop):
    x = 0.35 + i * 2.85
    ax.add_patch(plt.Rectangle((x, 1.35), 2.4, 1.3, facecolor=CELL_HOT if i == 1 else CELL,
                               edgecolor=ORANGE if i == 1 else CELL_EDGE, lw=1.6))
    ax.text(x + 1.2, 2.0, text, fontsize=15, ha="center", va="center", color=INK, linespacing=1.6)
    if i:
        arrow(ax, (x - 0.4, 2.0), (x, 2.0), MUTED, lw=2)
note(ax, 0.35, 0.7, "一秒钟转 50 圈。19.7 节把这一圈拆开看。")
savefig(fig, "ch19_overview")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("2. 迷你版：3 个数进、2 个数出，自己搭一个、自己导出、自己手算")

# 迷你版的全部数字都是教学构造的，选成整数是为了能手算。
MINI_MEAN = np.array([10.0, 0.0, 2.0])
MINI_STD = np.array([1.99, 0.99, 3.99])          # 加上 ε = 0.01 之后，分母正好是 (2, 1, 4)
MINI_EPS = 0.01                                  # 第 8 章 8.3 节分母上那个防止除以 0 的小数
MINI_W1 = np.array([[1.0, 0.0, 2.0], [-1.0, 1.0, 0.0]])
MINI_B1 = np.array([0.5, -3.0])  # TWEAK-2: np.array([0.5, 3.0])
MINI_W2 = np.array([[2.0, 1.0], [0.0, -1.0]])
MINI_B2 = np.array([0.0, 1.0])
MINI_OBS = np.array([12.0, 2.0, 4.0])  # TWEAK-1: np.array([12.0, 2.0, 12.0])


def elu(z):
    """第 6 章 6.4 节那道门：正数原样过，负数压成 e^z − 1（永远不低于 −1）。"""
    return np.where(z >= 0, z, np.exp(z) - 1.0)


def mini_forward(obs, normalize=True):
    """迷你版的五步：减 μ、除以 σ+ε、第一层、ELU、第二层。normalize=False 就是“忘了烘焙”。"""
    x = (obs - MINI_MEAN) / (MINI_STD + MINI_EPS) if normalize else obs
    h = elu(MINI_W1 @ x + MINI_B1)
    return MINI_W2 @ h + MINI_B2, x, h


class MiniPolicy(nn.Module):
    """和 mini_forward 一模一样的 torch 版；导出 ONNX 用它。"""

    def __init__(self):
        super().__init__()
        self.register_buffer("_mean", torch.tensor(MINI_MEAN, dtype=torch.float32).unsqueeze(0))
        self.register_buffer("_std", torch.tensor(MINI_STD, dtype=torch.float32).unsqueeze(0))
        self.eps = MINI_EPS
        self.fc1 = nn.Linear(3, 2)
        self.fc2 = nn.Linear(2, 2)
        self.gate = nn.ELU()
        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor(MINI_W1, dtype=torch.float32))
            self.fc1.bias.copy_(torch.tensor(MINI_B1, dtype=torch.float32))
            self.fc2.weight.copy_(torch.tensor(MINI_W2, dtype=torch.float32))
            self.fc2.bias.copy_(torch.tensor(MINI_B2, dtype=torch.float32))

    def forward(self, x):
        x = (x - self._mean) / (self._std + self.eps)     # 归一化器：烘在同一个模块里（第 8 章 8.6 节）
        return self.fc2(self.gate(self.fc1(x)))


mini_onnx = TMP / "mini.onnx"
with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
    warnings.simplefilter("ignore")                      # 导出器会打印一串进度行，这里不占版面
    torch.onnx.export(MiniPolicy().eval(), (torch.zeros(1, 3),), str(mini_onnx),
                      export_params=True, opset_version=18,
                      input_names=["obs"], output_names=["actions"])

mini_graph = onnx.load(str(mini_onnx))
mini_ops = [n.op_type for n in mini_graph.graph.node]
mini_init = {t.name: numpy_helper.to_array(t) for t in mini_graph.graph.initializer}
print("迷你 ONNX 的算子 :", " → ".join(mini_ops))
print("它存下来的常数   :", ", ".join(f"{k} {list(v.shape)}" for k, v in mini_init.items()))
div_const = [v for k, v in mini_init.items() if v.shape == (1, 3) and not np.array_equal(v[0], MINI_MEAN)][0][0]
print("除法用的那个常数 :", vec(div_const), "= σ + ε，导出时就加好了，图里只有一次除法")

out_np, x_norm, h_mid = mini_forward(MINI_OBS)
mini_sess = ort.InferenceSession(str(mini_onnx), providers=["CPUExecutionProvider"])
out_ort = mini_sess.run(["actions"], {"obs": MINI_OBS[None, :].astype(np.float32)})[0][0]
z_mid = MINI_W1 @ x_norm + MINI_B1
table(["这一步", "算出来是"],
      [["读数 obs", vec(MINI_OBS)],
       ["减 μ 再除以 σ+ε", vec(x_norm)],
       ["第一层（价格表 + 运费）", vec(z_mid)],
       ["过 ELU 门", vec(h_mid)],
       ["第二层 = 输出", vec(out_np)],
       ["onnxruntime 跑同一个文件", vec(out_ort)]])
def say_elu(z, h):
    """把“过门”这一步的手算原样打印出来；正数和负数走的是门的两条分支。"""
    if z < 0:
        # e^z 小到 4 位小数全是 0 时（比如 z = −13），多印两位，别让手算式变成 “0 − 1”
        k = 4 if np.exp(z) >= 5e-5 else 6
        return minus(f"e^({num(z)}) = {np.exp(z):.{k}f}，所以 ELU({num(z)}) = {num(h, k)}")
    return minus(f"{num(z)} 不是负数，门原样放行：ELU({num(z)}) = {num(h)}")


print("手算：" + say_elu(float(z_mid[1]), float(h_mid[1])))
print(minus(f"      2 × {num(h_mid[0])} + 1 × ({num(h_mid[1])}) + 0 = {num(out_np[0])}；"
            f"0 × {num(h_mid[0])} + (−1) × ({num(h_mid[1])}) + 1 = {num(out_np[1])}"))
check("迷你 ONNX 的算子就是 Sub → Div → Gemm → Elu → Gemm",
      mini_ops == ["Sub", "Div", "Gemm", "Elu", "Gemm"])
check("除数是一个算好的常数 (2, 1, 4)，等于 σ + ε", np.allclose(div_const, MINI_STD + MINI_EPS))
check("归一化后是 (1, 2, 0.5)", np.allclose(x_norm, [1.0, 2.0, 0.5]))
check("第一层的两个数是 (2.5, −2)", np.allclose(z_mid, [2.5, -2.0]))
check("ELU(2.5) = 2.5 原样；ELU(−2) = −0.8647", np.allclose(np.round(h_mid, 4), [2.5, -0.8647]))
check("输出是 (4.1353, 1.8647)", np.allclose(np.round(out_np, 4), [4.1353, 1.8647]))
check("按印出来的数重算：2 × 2.5 + 1 × (−0.8647) = 4.1353", round(2 * 2.5 + 1 * (-0.8647), 4) == 4.1353)
check("按印出来的数重算：0 × 2.5 − 1 × (−0.8647) + 1 = 1.8647", round(0.8647 + 1, 4) == 1.8647)
check("手算 = onnxruntime（差异 < 1e-6）", float(np.max(np.abs(out_np - out_ort))) < 1e-6)
quiz_obs = np.array([8.0, 1.0, 6.0])                 # 19.2 节自测那一帧
quiz_out, quiz_x, quiz_h = mini_forward(quiz_obs)
print(minus(f"自测那一帧 {vec(quiz_obs)}：归一化 {vec(quiz_x)} → 第一层 {vec(MINI_W1 @ quiz_x + MINI_B1)}"
            f" → 过门 {vec(quiz_h)} → 输出 {vec(quiz_out)}"))
check("自测：(8, 1, 6) 归一化后是 (−1, 1, 1)，输出是 (2.3679, 1.6321)",
      np.allclose(quiz_x, [-1.0, 1.0, 1.0]) and np.allclose(np.round(quiz_out, 4), [2.3679, 1.6321]))
check("按印出来的数重算：3 − 0.6321 = 2.3679", round(3 - 0.6321, 4) == 2.3679)

# ---------------------------------------------------------------------------
banner("2b. 画图：figures/ch19_mini_graph.png（迷你版的五个算子，每一步的数都写在图里）")


fig, axes = lesson_figure(4, "一个 3 进 2 出的迷你策略：ONNX 里就是这五步", panel_height=2.9, width=11.0)

ax = axes[0]
lesson_panel(ax, "① 三个读数，先减平常值 μ、再除以起伏 σ + ε", xmax=11)
lesson_cells(ax, [[12, 2, 4]], left=0.35, bottom=2.0, width=0.95, height=0.72)
hand(ax, 3.45, 2.36, "−  μ (10, 0, 2)  ÷  (σ+ε) (2, 1, 4)", fontsize=FS_SMALL)
arrow(ax, (7.8, 2.36), (8.25, 2.36), MUTED, lw=2)
lesson_cells(ax, [[1, 2, 0.5]], left=8.3, bottom=2.0, width=0.85, height=0.72, highlights=[(0, 0), (0, 1), (0, 2)])
hand(ax, 0.35, 1.15, "(12 − 10) ÷ 2 = 1     (2 − 0) ÷ 1 = 2     (4 − 2) ÷ 4 = 0.5")
note(ax, 0.35, 0.45, "σ + ε 在导出时就算好了，ONNX 里存的是一个常数 (2, 1, 4)。")

ax = axes[1]
lesson_panel(ax, "② 第一层：两家店各拿一张价格表，对同一张清单结账，再加运费", xmax=11)
hand(ax, 0.35, 2.45, "店 1： 1×1 + 0×2 + 2×0.5 + 0.5 = 2.5")
hand(ax, 0.35, 1.65, "店 2： (−1)×1 + 1×2 + 0×0.5 − 3 = −2", color=GREEN)
arrow(ax, (6.9, 2.05), (7.6, 2.05), MUTED, lw=2)
lesson_cells(ax, [[2.5], [-2]], left=7.7, bottom=1.35, width=1.15, height=0.72)
note(ax, 0.35, 0.55, "这就是第 2 章 2.6 节的价格表 + 运费，只不过这里只有两家店。")

ax = axes[2]
lesson_panel(ax, "③ ELU 门：正数原样过，负数压成 e^z − 1", xmax=11)
hand(ax, 0.35, 2.45, "ELU(2.5) = 2.5   （正数，门不管它）")
hand(ax, 0.35, 1.65, minus(f"ELU(−2) = e^(−2) − 1 = {np.exp(-2.0):.4f} − 1 = {num(h_mid[1])}"), color=GREEN)
arrow(ax, (7.5, 2.05), (8.2, 2.05), MUTED, lw=2)
lesson_cells(ax, [[2.5], [round(float(h_mid[1]), 4)]], left=8.3, bottom=1.35, width=1.35, height=0.72,
             highlights=[(1, 0)])
note(ax, 0.35, 0.55, "门只对负数动手：−2 被压到 −0.8647，再负也不会低于 −1（第 6 章 6.4 节）。")

ax = axes[3]
lesson_panel(ax, "④ 第二层给出 2 个动作；这五步在 ONNX 里就是五个算子", xmax=11)
hand(ax, 0.35, 2.85, minus(f"2 × 2.5 + 1 × ({num(h_mid[1])}) + 0 = {num(out_np[0])}"))
hand(ax, 0.35, 2.20, minus(f"0 × 2.5 + (−1) × ({num(h_mid[1])}) + 1 = {num(out_np[1])}"), color=GREEN)
draw_chain(ax, ["Sub", "Div", "Gemm", "Elu", "Gemm"], y=0.95, x0=0.35, w=1.5, gap=0.5, height=0.8, hot=(3,))
hand(ax, 10.2, 1.35, f"→ {vec(out_np)}", color=ORANGE, fontsize=FS_SMALL, ha="left")
note(ax, 0.35, 0.35, "Sub、Div 是①，Gemm 是②和④（矩阵乘 + 偏置），Elu 是③。")
savefig(fig, "ch19_mini_graph")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("3. 打开真的那一个：输入输出、九个算子、存下来的常数、metadata")
if not HAVE_RUN:
    print("  （没有运行目录，跳过。）")
    meta = {}
    ops = []
else:
    graph = onnx.load(str(onnx_path))
    inp, outp = graph.graph.input[0], graph.graph.output[0]
    in_shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    out_shape = [d.dim_value for d in outp.type.tensor_type.shape.dim]
    ops = [n.op_type for n in graph.graph.node]
    print("输入 :", inp.name, in_shape)
    print("输出 :", outp.name, out_shape)
    print("算子 :", " → ".join(ops))
    init = {t.name: numpy_helper.to_array(t) for t in graph.graph.initializer}
    table(["ONNX 里存下来的常数", "形状", "有几个数"],
          [[k, list(v.shape), v.size] for k, v in init.items()])
    n_const = sum(v.size for v in init.values())
    print(f"常数一共 {n_const:,} 个数 = 四层的 197,774 + 烘进来的 61 + 61。")
    meta = {p.key: p.value for p in graph.metadata_props}
    obs_blocks = meta["observation_names"].split(",")
    print("metadata 一共", len(meta), "项；其中：")
    print("  observation_names =", " | ".join(obs_blocks))
    print("  joint_names       =", meta["joint_names"])
    print("  default_joint_pos =", meta["default_joint_pos"])
    print("  action_scale      =", meta["action_scale"])
    check("输入叫 obs，形状 [1, 61]", inp.name == "obs" and in_shape == [1, 61])
    check("输出叫 actions，形状 [1, 14]", outp.name == "actions" and out_shape == [1, 14])
    check("九个算子：Sub, Div, 4 个 Gemm, 3 个 Elu",
          ops == ["Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"])
    check("和迷你版是同一串，只是 Gemm/Elu 各多了几个",
          [o for o in ops if o in ("Sub", "Div")] == ["Sub", "Div"] and ops.count("Gemm") == 4 and ops.count("Elu") == 3)
    check("除数那个常数 = 检查点里的 σ + 0.01（导出时就加好了）",
          any(np.allclose(v[0], std + 0.01) for v in init.values() if v.shape == (1, 61)))
    check("ONNX 里存了 10 个常数、一共 197,896 个数", len(init) == 10 and n_const == 197896)
    check("比四层网络的 197,774 多出 122 个，就是烘进来的 61 个 μ 和 61 个 σ+0.01",
          n_const - 197774 == 122)
    check("metadata 恰好 8 项，正文点名的就是这 8 个",
          len(meta) == 8 and set(meta) == {"run_path", "joint_names", "joint_stiffness", "joint_damping",
                                           "default_joint_pos", "command_names", "observation_names",
                                           "action_scale"})
    check("BAM 之下 joint_stiffness 读出来全是 1、joint_damping 全是 0（第 17 章 17.4 节）",
          set(meta["joint_stiffness"].split(",")) == {"1.000"} and set(meta["joint_damping"].split(",")) == {"0.000"})
    check("observation_names 是 8 块（第 15 章 15.1 节），最后三块是三块命令（15.5 节）",
          len(obs_blocks) == 8 and obs_blocks[-3:] == ["command", "head_command", "body_command"])
    check("joint_names 和 default_joint_pos 都是 14 项（顺序见第 15 章 15.3 节，HOME 见第 17 章 17.1 节）",
          len(meta["joint_names"].split(",")) == 14 and len(meta["default_joint_pos"].split(",")) == 14)
    check("action_scale = 1.0（第 17 章 17.1 节）", float(meta["action_scale"]) == 1.0)

# ---------------------------------------------------------------------------
banner("3b. 画图：figures/ch19_nine_ops.png（九个算子，和每一步数据有多宽）")
WIDTHS = [("输入 obs", 61), ("第 1 层", 512), ("第 2 层", 256), ("第 3 层", 128), ("输出 actions", 14)]
fig, axes = lesson_figure(2, "真的那个 ONNX：九个算子，两头是 61 和 14", panel_height=3.6, width=12.0)

ax = axes[0]
lesson_panel(ax, "① 九个算子排成一条流水线", xmax=12)
draw_chain(ax, REAL_OPS, y=2.35, x0=0.3, w=1.05, gap=0.22, height=0.8)
hand(ax, 0.3, 1.85, "Sub, Div = 归一化（第 8 章）", color=BLUE, fontsize=FS_SMALL)
hand(ax, 4.4, 1.85, "Gemm = 价格表 + 运费（第 2 章 2.6 节）", color=GREEN, fontsize=FS_SMALL)
hand(ax, 0.3, 1.25, "Elu = 那道门（第 6 章 6.4 节）", color=ORANGE, fontsize=FS_SMALL)
hand(ax, 4.4, 1.25, "2 个归一化 + 4 个 Gemm + 3 个 Elu = 9 个算子", fontsize=FS_SMALL)
note(ax, 0.3, 0.55, "最左边进去的是 61 个数，最右边出来的是 14 个数，中间一步不多、一步不少。")

ax = axes[1]
lesson_panel(ax, "② 每一步手里有多少个数（柱子高度成比例）", xmax=12)
top = 1.62
for i, (label, n) in enumerate(WIDTHS):
    x = 1.2 + i * 2.1
    h = top * n / 512
    ax.add_patch(plt.Rectangle((x, 1.25), 1.0, h, facecolor=CELL_HOT if i in (0, 4) else CELL,
                               edgecolor=ORANGE if i in (0, 4) else CELL_EDGE, lw=1.6))
    ax.text(x + 0.5, 1.25 + h + 0.17, str(n), fontsize=FS_SMALL, ha="center", color=INK)
    ax.text(x + 0.5, 0.92, label, fontsize=FS_SMALL, ha="center", color=MUTED)
ax.plot([1.0, 11.4], [1.25, 1.25], color=FAINT, lw=1.4)
note(ax, 0.3, 0.58, f"柱子严格按个数成比例：512 ÷ 14 ≈ {512 / 14:.0f}，所以最右边那根薄成一条线——不是画漏了。")
note(ax, 0.3, 0.18, "61 项观测先被摊开成 512 个数，再一层层收回到 14 个动作。")
savefig(fig, "ch19_nine_ops")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("4. numpy 手写九步 vs onnxruntime：喂同样的观测，比最大差异")
EPS_HAND = 0.01  # TWEAK-3: 0.0
FRAMES = 5
if not HAVE_RUN:
    print("  （没有运行目录，跳过。）")
else:
    def policy_numpy(obs):
        """第 8 章的归一化 + 第 6 章的四层，一共九步；输出就是 14 个动作的均值 μ。"""
        x = (obs - mean) / (std + EPS_HAND)
        h = elu(W[0] @ x + B[0])
        h = elu(W[1] @ h + B[1])
        h = elu(W[2] @ h + B[2])
        return W[3] @ h + B[3]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    max_diff = 0.0
    for k in range(FRAMES):
        obs = (rng.normal(size=61) * 0.5).astype(np.float32)
        a_np, a_ort = policy_numpy(obs), sess.run(["actions"], {"obs": obs[None, :]})[0][0]
        max_diff = max(max_diff, float(np.max(np.abs(a_np - a_ort))))
        if k == 0:
            table(["", "前 5 个动作（rad）"],
                  [["numpy 手写", str(np.round(a_np[:5], 4))], ["onnxruntime", str(np.round(a_ort[:5], 4))]])
    print(f"{FRAMES} 帧随机观测的最大差异 = {max_diff:.2e}  （你的数字会不同；量级应该在 1e-6 上下）")
    check("numpy 手写 = onnxruntime（最大差异 < 1e-4）", max_diff < 1e-4)

# ---------------------------------------------------------------------------
banner("5. 部署不采样：同一帧喂两次，输出一模一样")
if not HAVE_RUN:
    print("  （没有运行目录，跳过。）")
else:
    frame = (np.random.default_rng(7).normal(size=61) * 0.5).astype(np.float32)
    a1 = sess.run(["actions"], {"obs": frame[None, :]})[0]
    a2 = sess.run(["actions"], {"obs": frame[None, :]})[0]
    print("两次输出的最大差异 =", float(np.max(np.abs(a1 - a2))))
    print(f"检查点里的 14 个动作 σ，前 4 个：{np.round(sigma[:4], 4)}（训练时 μ + σ×ε 要用它，第 4 章 4.7 节）")
    names = [t.name for t in onnx.load(str(onnx_path)).graph.initializer]
    print("ONNX 存下来的常数名字里有没有 std_param :", any("std_param" in n for n in names))
    check("同一帧两次，输出逐位相同", np.array_equal(a1, a2))
    check("ONNX 里根本没有 std_param：导出时就丢掉了", not any("std_param" in n for n in names))

# ---------------------------------------------------------------------------
banner("6. 忘了烘焙会怎样：跳过 Sub/Div，直接把原始读数喂给网络")
bad_np, _, bad_h = mini_forward(MINI_OBS, normalize=False)
bad_z = MINI_W1 @ MINI_OBS + MINI_B1
table(["迷你版这一步", "走归一化", "跳过归一化"],
      [["进第一层的三个数", vec(x_norm), vec(MINI_OBS)],
       ["第一层算出来", vec(z_mid), vec(bad_z)],
       ["过 ELU 门", vec(h_mid), vec(bad_h)],
       ["输出（rad）", vec(out_np), vec(bad_np)]])
gap = float(bad_np[0] - out_np[0])
print(minus(f"第一个动作：{num(out_np[0])} rad → {num(bad_np[0])} rad，差 {num(gap)} rad ≈ {np.degrees(gap):.0f}°。"))
print("手算：" + say_elu(float(bad_z[1]), float(bad_h[1]))
      + minus(f"；所以 2 × {num(bad_h[0])} + ({num(bad_h[1], 6)}) = {num(bad_np[0], 6)}"))
check("跳过归一化：第一层变成 (20.5, −13)", np.allclose(bad_z, [20.5, -13.0]))
check("ELU(−13) 已经贴着下限 −1（四舍五入到 4 位就是 −1）", round(float(bad_h[1]), 4) == -1.0)
check("输出从 (4.1353, 1.8647) 变成 (40, 2)", np.allclose(np.round(bad_np, 4), [40.0, 2.0]))
check("按印出来的数重算：40 − 4.1353 = 35.8647", round(40.0 - 4.1353, 4) == 35.8647)
check("按印出来的数重算：(−1) × (−0.999998) + 1 = 1.999998（表里四舍五入成 2）",
      round(0.999998 + 1.0, 6) == 1.999998 and round(1.999998, 4) == 2.0)
check("按印出来的数重算：35.8647 × 57.3 ≈ 2055°，2055 ÷ 360 ≈ 5.7 圈",
      round(35.8647 * 57.3) == 2055 and round(2055 / 360, 1) == 5.7)
if HAVE_RUN:
    frame = (np.random.default_rng(11).normal(size=61) * 0.5).astype(np.float32)

    def policy_no_norm(obs):
        h = elu(W[0] @ obs + B[0])
        h = elu(W[1] @ h + B[1])
        h = elu(W[2] @ h + B[2])
        return W[3] @ h + B[3]

    real_gap = float(np.max(np.abs(policy_numpy(frame) - policy_no_norm(frame))))
    print(f"真模型上同一帧：最大差 {real_gap:.3f} rad ≈ {np.degrees(real_gap):.0f}°（你的数字会不同，量级是 1 rad，不是 1e-6）")
    check("真模型上漏归一化的差，比浮点误差大 4 个数量级以上", real_gap > 1e4 * max_diff)

# ---------------------------------------------------------------------------
banner("6b. 画图：figures/ch19_bake_or_not.png（同一帧读数，走不走那两步，差 35.86 rad）")
fig, axes = lesson_figure(3, "漏掉归一化：同一帧读数，动作从 4.14 rad 变成 40 rad", panel_height=3.0, width=11.0)

ax = axes[0]
lesson_panel(ax, "① 烘焙了：读数先减 μ、除以 σ+ε，再进网络", xmax=11)
lesson_cells(ax, [[12, 2, 4]], left=0.35, bottom=2.05, width=0.9, height=0.72)
arrow(ax, (3.2, 2.41), (3.9, 2.41), MUTED, lw=2)
cell(ax, 3.95, 2.05, "Sub, Div", width=1.9, height=0.72, fontsize=FS_SMALL)
arrow(ax, (5.95, 2.41), (6.6, 2.41), MUTED, lw=2)
lesson_cells(ax, [[1, 2, 0.5]], left=6.65, bottom=2.05, width=0.85, height=0.72)
hand(ax, 0.35, 1.25, minus(f"第一层 (2.5, −2) → ELU (2.5, {num(h_mid[1])}) → 输出 {vec(out_np)}"))
note(ax, 0.35, 0.5, "这条路和训练时网络看到的完全一样（第 8 章 8.6 节）。")

ax = axes[1]
lesson_panel(ax, "② 漏了：原始读数直接进网络", xmax=11)
lesson_cells(ax, [[12, 2, 4]], left=0.35, bottom=2.05, width=0.9, height=0.72, highlights=[(0, 0), (0, 1), (0, 2)])
arrow(ax, (3.2, 2.41), (6.6, 2.41), ORANGE, lw=2.4)
hand(ax, 3.4, 2.75, "Sub, Div 两步没有了", color=ORANGE, fontsize=FS_SMALL)
lesson_cells(ax, [[12, 2, 4]], left=6.65, bottom=2.05, width=0.9, height=0.72, highlights=[(0, 0), (0, 1), (0, 2)])
hand(ax, 0.35, 1.25, f"第一层 (20.5, −13) → ELU (20.5, −1) → 输出 {vec(bad_np)}", color=ORANGE)
note(ax, 0.35, 0.5, "网络一句错也不报，照样给出 14 个数——只是全错了。")

ax = axes[2]
lesson_panel(ax, "③ 把这两个动作画在同一把尺子上", xmax=11)
scale = 8.4 / 40.0
for y, val, color, label in [(2.25, float(out_np[0]), BLUE, "烘焙了"), (1.25, float(bad_np[0]), ORANGE, "漏了")]:
    ax.add_patch(plt.Rectangle((2.2, y), max(val * scale, 0.02), 0.55, facecolor=color, edgecolor="none", alpha=0.85))
    ax.text(2.1, y + 0.28, label, fontsize=FS_SMALL, ha="right", va="center", color=color)
    if 2.3 + val * scale + 1.7 < 11:
        ax.text(2.3 + val * scale, y + 0.28, f"{num(val)} rad", fontsize=FS_SMALL, va="center", color=color)
    else:
        ax.text(2.1 + val * scale, y + 0.28, f"{num(val)} rad", fontsize=FS_SMALL, va="center", ha="right",
                color="white")
for tick in (0, 10, 20, 30, 40):
    ax.plot([2.2 + tick * scale] * 2, [0.95, 1.05], color=MUTED, lw=1.2)
    ax.text(2.2 + tick * scale, 0.72, str(tick), fontsize=FS_TICK, ha="center", color=MUTED)
ax.text(1.95, 0.72, "动作（rad）", fontsize=FS_TICK, ha="right", va="center", color=MUTED)
note(ax, 0.3, 0.25, "1 rad ≈ 57.3°。40 rad 这个目标角，舵机只会一头撞到限位上。")
savefig(fig, "ch19_bake_or_not")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("7. 真机每 0.02 秒的那一圈：拼一帧 61 维，跑一次，算出 14 个目标角")
if not HAVE_RUN:
    print("  （没有运行目录，跳过。）")
else:
    home = np.array([float(v) for v in meta["default_joint_pos"].split(",")])
    joint_names = meta["joint_names"].split(",")
    action_scale = float(meta["action_scale"])
    obs = np.zeros(61, dtype=np.float32)          # 站着不动、没有速度、命令全零的一帧
    obs[48:51] = [0.2, 0.0, 0.5]                  # 命令块的前 3 项：前进 0.2 m/s、左转 0.5 rad/s（第 15 章 15.5 节）
    action = sess.run(["actions"], {"obs": obs[None, :]})[0][0]
    target = home + action * action_scale         # 第 17 章 17.1 节：目标角 = HOME + 动作 × scale
    table(["关节", "HOME（rad）", "策略给的动作（rad）", "目标角（rad）"],
          [[joint_names[i], round(float(home[i]), 4), round(float(action[i]), 4), round(float(target[i]), 4)]
           for i in (0, 3, 5, 13)], floatfmt=".4g")
    print("（这几个数依赖检查点，你的会不同。冒烟训练只练了 5 圈，动作基本是随机的。）")
    print("一个环境步 0.02 s = 4 个物理步 × 0.005 s（第 14 章 14.1 节）；50 Hz。")
    check("目标角 = HOME + 动作 × 1.0，逐项对得上", np.allclose(target, home + action))
    check("14 个关节名、14 个 HOME、14 个动作，三者一样长",
          len(joint_names) == len(home) == len(action) == 14)

# ---------------------------------------------------------------------------
banner("7b. 画图：figures/ch19_deploy_loop.png 和 figures/ch19_hotswap.png")
fig, axes = lesson_figure(2, "真机上每 0.02 秒转一圈：五件事，其中一件是跑 ONNX", panel_height=3.3, width=12.0)

ax = axes[0]
lesson_panel(ax, "① 一圈五步", xmax=12)
steps = ["① 读传感器\n拼成 61 个数", "② 交给 ONNX\n拿回 14 个数", "③ HOME + 动作\n= 14 个目标角",
         "④ 物理走 4 小步\n舵机去追目标角", "⑤ 把这 14 个数\n记成“上一步动作”"]
xs = []
for i, text in enumerate(steps):
    x = 0.2 + i * 2.35
    ax.add_patch(plt.Rectangle((x, 1.95), 2.2, 1.2, facecolor=CELL_HOT if i == 1 else CELL,
                               edgecolor=ORANGE if i == 1 else CELL_EDGE, lw=1.6))
    ax.text(x + 1.1, 2.55, text, fontsize=15, ha="center", va="center", color=INK, linespacing=1.6)
    xs.append(x + 1.1)
    if i:
        arrow(ax, (x - 0.15, 2.55), (x, 2.55), MUTED, lw=2)
ax.plot([xs[-1], xs[-1], xs[0]], [1.95, 1.45, 1.45], color=FAINT, lw=2)
arrow(ax, (xs[0], 1.45), (xs[0], 1.95), FAINT, lw=2)
hand(ax, 6.0, 1.05, "⑤ 之后回到 ①", color=MUTED, fontsize=FS_SMALL, ha="center")
note(ax, 0.2, 0.45, "第 15 章的 61 项清单在①拼出来，第 17 章的目标角在③算出来。")

ax = axes[1]
lesson_panel(ax, "② 这一圈 0.02 秒里，物理走了 4 小步", xmax=12)
bar_left, bar_w = 1.2, 9.6
for i in range(4):
    x = bar_left + i * bar_w / 4
    ax.add_patch(plt.Rectangle((x, 1.70), bar_w / 4, 0.72, facecolor=CELL, edgecolor=CELL_EDGE, lw=1.4))
    ax.text(x + bar_w / 8, 2.06, "0.005 s", fontsize=FS_SMALL, ha="center", va="center", color=MUTED)
ax.text(bar_left - 0.1, 2.82, "策略在这里算一次动作，这一圈就不再算了", fontsize=FS_SMALL, color=ORANGE)
arrow(ax, (bar_left, 2.66), (bar_left, 2.46), ORANGE, lw=2)
ax.text(bar_left + bar_w / 2, 1.30, "一个环境步 = 0.02 s = 50 Hz", fontsize=FS_SMALL, ha="center", color=INK)
note(ax, 0.2, 0.55, "策略每 0.02 s 才说一次话；这 0.02 s 里目标角不变，舵机一直在追它（第 17 章）。")
savefig(fig, "ch19_deploy_loop")
plt.close(fig)

fig, axes = lesson_figure(2, "几个 ONNX 轮流上场：接口一样，命令格子的意思不一样", panel_height=3.5, width=11.5)
POLICIES = [("走路 walk.onnx", BLUE), ("坐下／起立 sitstand.onnx", GREEN), ("前滚翻 roulade.onnx", ORANGE)]
ax = axes[0]
lesson_panel(ax, "① 三个文件，同一个插口：61 个数进，14 个数出", xmax=11.5, ymax=4.4)
for i, (name, color) in enumerate(POLICIES):
    y = 3.15 - i * 0.78
    ax.add_patch(plt.Rectangle((0.3, y - 0.3), 3.9, 0.62, facecolor=CELL, edgecolor=color, lw=1.8))
    ax.text(2.25, y, name, fontsize=FS_SMALL, ha="center", va="center", color=INK)
    arrow(ax, (4.3, y), (5.5, 2.37), color, lw=1.8)
ax.add_patch(plt.Rectangle((5.55, 1.82), 2.6, 1.1, facecolor=CELL_HOT, edgecolor=ORANGE, lw=1.8))
ax.text(6.85, 2.37, "运行时\n一次只装一个", fontsize=FS_SMALL, ha="center", va="center", color=INK, linespacing=1.5)
arrow(ax, (8.25, 2.37), (8.9, 2.37), MUTED, lw=2)
ax.text(9.0, 2.37, "14 个动作", fontsize=FS_SMALL, va="center", color=INK)
hand(ax, 0.3, 0.95, "三个文件都是 61 个数进、14 个数出，所以能互换。", fontsize=FS_SMALL)
note(ax, 0.3, 0.35, "按一下按钮换一个文件，机器人不用停下来（第 15 章 15.6 节）。")

ax = axes[1]
lesson_panel(ax, "② 13 格命令块：谁用哪几格，训练时就定死了", xmax=11.5, ymax=4.4)
labels = ["vx", "vy", "ωz"] + [f"头{i}" for i in range(1, 5)] + [f"身{i}" for i in range(1, 7)]
used = {0: [0, 1, 2], 1: [0], 2: []}
for i, (name, color) in enumerate(POLICIES):
    y = 2.8 - i * 0.7
    ax.text(0.3, y + 0.22, name.split()[0], fontsize=FS_SMALL, color=color, va="center")
    for j in range(13):
        x = 2.6 + j * 0.62
        on = j in used[i]
        ax.add_patch(plt.Rectangle((x, y), 0.56, 0.44, facecolor=CELL_HOT if on else CELL,
                                   edgecolor=color if on else CELL_EDGE, lw=1.6 if on else 1.0))
        if i == 0:
            ax.text(x + 0.28, y + 0.50, labels[j], fontsize=13, ha="center", color=MUTED)
ax.add_patch(plt.Rectangle((2.50, 1.30), 0.76, 1.96, facecolor="none", edgecolor=ORANGE, lw=1.8, ls="--"))
hand(ax, 0.3, 1.00, "虚线框里是同一格：走路读作 vx（前进多快），坐下／起立读作姿态标志（1 坐下 / 0 站起）。",
     fontsize=FS_SMALL, color=ORANGE)
note(ax, 0.3, 0.45, "走路用前 3 格，坐下／起立只用第 1 格，前滚翻 13 格全零；没上色的是零填充（第 15 章 15.7 节）。")
savefig(fig, "ch19_hotswap")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("8. 存下来的是“怎样计算”，不是一段录好的动作：换一帧输入，输出就变")
other_obs = MINI_MEAN.copy()                       # 每一项都正好等于 μ：归一化后全是 0
out_other, x_other, h_other = mini_forward(other_obs)
table(["喂进去的三个读数", "归一化后", "第一层", "过门后", "输出"],
      [[vec(MINI_OBS), vec(x_norm), vec(z_mid), vec(h_mid), vec(out_np)],
       [vec(other_obs), vec(x_other), vec(MINI_W1 @ x_other + MINI_B1), vec(h_other), vec(out_other)]])
print("手算：" + say_elu(float((MINI_W1 @ x_other + MINI_B1)[1]), float(h_other[1]))
      + minus(f"；2 × {num(h_other[0])} + ({num(h_other[1])}) = {num(out_other[0])}"))
check("读数正好等于 μ 时，归一化后是 (0, 0, 0)", np.allclose(x_other, [0.0, 0.0, 0.0]))
check("第二帧的输出是 (0.0498, 1.9502)", np.allclose(np.round(out_other, 4), [0.0498, 1.9502]))
check("按印出来的数重算：1 + (−0.9502) = 0.0498", round(1 - 0.9502, 4) == 0.0498)
check("同一个文件、两帧不同的读数，输出不同", not np.allclose(out_np, out_other))

# ---------------------------------------------------------------------------
banner("9. 三种差异，量级差得很远：浮点误差 / 漏归一化 / 把格子填错")
SWAP = (1, 2)  # 把读数的第 2、3 项对调，模拟“两头的约定不一致”
swapped = MINI_OBS.copy()
swapped[[SWAP[0], SWAP[1]]] = swapped[[SWAP[1], SWAP[0]]]
out_swap, x_swap, h_swap = mini_forward(swapped)
mini_float_gap = float(np.max(np.abs(out_np - out_ort)))
rows = [["浮点误差（手写 vs onnxruntime）", f"{mini_float_gap:.0e}", "算的是同一件事，末位不同"],
        ["把读数的第 2、3 项对调", num(abs(float(out_swap[0] - out_np[0]))), "形状还是 3 项，意思全变了"],
        ["漏掉归一化", num(abs(gap)), "网络看到的数完全不是训练时的量级"]]
table(["哪种问题", "第 1 个动作差多少（rad）", "怎么回事"], rows)
print(f"对调之后：归一化 {vec(x_swap)} → 第一层 {vec(MINI_W1 @ x_swap + MINI_B1)} → 输出 {vec(out_swap)}")
check("对调后归一化是 (1, 4, 0)", np.allclose(x_swap, [1.0, 4.0, 0.0]))
check("对调后输出是 (3, 1)", np.allclose(np.round(out_swap, 4), [3.0, 1.0]))
check("按印出来的数重算：4.1353 − 3 = 1.1353", round(4.1353 - 3.0, 4) == 1.1353)
check("三者量级排序：浮点误差 < 填错格子 < 漏归一化",
      mini_float_gap < abs(float(out_swap[0] - out_np[0])) < abs(gap))
check(f"迷你版的浮点误差落在 1e-7 量级（表里印成 {mini_float_gap:.0e}）", 1e-9 < mini_float_gap < 1e-6)
swap_ort = mini_sess.run(["actions"], {"obs": swapped[None, :].astype(np.float32)})[0][0]
swap_float_gap = float(np.max(np.abs(out_swap - swap_ort)))
print(f"把填错的那一帧也喂给 onnxruntime：手写 {vec(out_swap)}，onnxruntime {vec(swap_ort)}，"
      f"差 {swap_float_gap:.0e}——“数值一致”这一层照样满分（19.10 节）。")
check("对调那一帧：手写和 onnxruntime 差 0，数值一致这一层查不出接口错了", swap_float_gap == 0.0)

# ---------------------------------------------------------------------------
banner("10. 留证据：手里这个 .onnx，到底是哪个检查点导出来的？")
if not HAVE_RUN:
    print("  （没有运行目录，跳过。）")
else:
    onnx_w = {k: v for k, v in init.items() if k == "mlp.0.weight"}["mlp.0.weight"]
    rows, matched = [], []
    for p in ckpt_paths:
        w0 = torch.load(str(p), map_location="cpu", weights_only=False)["actor_state_dict"]["mlp.0.weight"].numpy()
        same = bool(np.array_equal(w0, onnx_w))
        rows.append([p.name, f"{float(np.max(np.abs(w0 - onnx_w))):.3e}", "配对" if same else "不是这个"])
        if same:
            matched.append(p.name)
    table(["检查点", "和 ONNX 第一层权重的最大差", "结论"], rows)
    print("配对的是：", ", ".join(matched) if matched else "一个都不是（这个 ONNX 不是这里导出来的）")
    check("恰好有一个检查点和这个 ONNX 逐位配对", len(matched) == 1)
    check("配对的是编号最大的那个检查点（每存一次检查点就导一次 ONNX，第 14 章 14.9 节）",
          matched == [ckpt_paths[-1].name])
print("这个实验证明了什么、没证明什么：")
table(["这个实验做到的", "它不能替代的"],
      [["手写九步 = ONNX（数值一致）", "传感器坐标、单位、命令含义两边一致"],
       ["部署路径是确定的（不采样）", "仿真里真的能站住、能走"],
       ["ONNX 和某个检查点逐位配对", "真机上的物理迁移"]])

# ---------------------------------------------------------------------------
banner("11. 映射到项目：正文引用的源码行还在不在")
EXPORT_LINES = [
    "runner.export_policy_to_onnx(path, filename)",
    "metadata = get_base_metadata(runner.env.unwrapped, run_path=cfg.checkpoint_file)",
    "attach_metadata_to_onnx(onnx_path, metadata)",
]
INFER_LINES = [
    "action = self.ort_session.run([self.output_name], {self.input_name: obs_batch})[0]",
    "self.last_action = action.copy()",
    "target_positions = self.default_pose + action * self.action_scale",
    "for _ in range(decimation):",
    "mujoco.mj_step(model, data)",
]
MANIFEST_LINES = ["SCHEMA_VERSION = 2", "MODEL_API = 1", "OBS_LEN = 61", "ACTION_LEN = 14",
                  '"control_hz": 50', 'POLICY_FILE = "policy.onnx"']


def read_src(rel: str) -> str:
    path = REPO / rel
    if not path.is_file():
        print(f"  找不到 {rel}（不在仓库里跑？），下面几条跳过。")
        return ""
    return path.read_text(encoding="utf-8")


export_src = read_src("src/mjlab_microduck/export.py")
infer_src = read_src("scripts/infer_policy.py")
manifest_src = read_src("src/mjlab_microduck/publish/manifest.py")
check("src/mjlab_microduck/export.py 的 run_export()：三行导出步骤还在，顺序一致",
      not export_src or lines_in_order(export_src, EXPORT_LINES))
check("src/mjlab_microduck/export.py 的说明里还写着“归一化自动烘进导出的图”",
      not export_src or "emits `actor(normalizer(obs))`" in export_src)
check("scripts/infer_policy.py 的 infer() / apply_action() / 主循环五行还在，顺序一致",
      not infer_src or lines_in_order(infer_src, INFER_LINES))
check("publish/manifest.py 里的 schema 2、61、14 还是这些数", not manifest_src or lines_in_order(manifest_src, MANIFEST_LINES))
try:
    import rsl_rl
    mlp_src = (Path(rsl_rl.__file__).parent / "models" / "mlp_model.py").read_text(encoding="utf-8")
    ppo_lines = (Path(rsl_rl.__file__).parent / "algorithms" / "ppo.py").read_text(encoding="utf-8").splitlines()
    check("rsl_rl 的 _OnnxMLPModel.forward 还是归一化 → 四层 → 取均值这三行",
          lines_in_order(mlp_src, ["x = self.obs_normalizer(x)", "out = self.mlp(x)",
                                   "return self.deterministic_output(out)"]))
    check(f"rsl_rl/algorithms/ppo.py 一共 {len(ppo_lines)} 行（正文说的是 545 行）", len(ppo_lines) == 545)
except ImportError:
    print("  没装 rsl_rl，跳过这两条。")

done()
