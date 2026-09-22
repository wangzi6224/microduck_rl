"""第 13 章实验：PPO。数据过期、比率（重要性采样）、裁剪目标的四格与平坦区、三项损失（含价值裁剪）、
KL 散度与自适应学习率、一次迭代的记录数与更新次数——全部用小数字手算，再用 numpy / torch 核对。

运行：uv run python docs/learn-zh/labs/ch13_ppo_clip.py
纯 CPU，numpy + torch + matplotlib。第 7 节读三份源码的文字，并把 rsl_rl 的 update() 里那几行原样拿来跑本章的小例子；
不加载机器人、不训练。
小节编号与正文一一对应：实验第 K 节 = 正文 13.K 节（第 7 节对应「映射到项目」）。
正文“改一改”要改的三行都带 `# TWEAK-k:` 标记（第 1、3、5 节各一处）。
"""

import importlib.metadata
import importlib.util
import math
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.distributions import Normal, kl_divergence

from _common import banner, check, done, lines_in_order, num, savefig, table
from _draw import (BLUE, CELL, CELL_EDGE, CELL_HOT, FAINT, FS_NOTE, FS_SMALL, FS_STEP, FS_TICK, GREEN, INK, MUTED,
                   ORANGE, WHITE_BOX, arrow, cell, data_axes, hand, lesson_figure, lesson_panel, note, panel_note,
                   panel_title, plt)

np.set_printoptions(precision=4, suppress=True)
torch.set_default_dtype(torch.float64)   # 用双精度，打印出来的末几位和手算一致


def m(x, fmt=".5f"):
    """印进图里的数：负号用正式的“−”，和正文一致。"""
    return format(x, fmt).replace("-", "−")


def bell(x, mu, sigma):
    """第 4 章 4.6 节的钟形：高度（密度）。"""
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))


# ---------------------------------------------------------------------------
banner("1. 数据过期：旧策略玩的 10 局，只说得出旧策略的平均分")
SCORE_BLUE, SCORE_GREEN = 3.0, 1.0      # 第 11 章 11.1 节的游戏机：蓝键 3 分，绿键 1 分
P_OLD = 0.5                             # 采数据时的策略：一半按蓝键
P_NEW = 0.7  # TWEAK-1: 0.3


def J(p):
    """平均分 J(p) = 3p + 1 × (1 − p) = 1 + 2p（第 11 章 11.1 节）。"""
    return SCORE_BLUE * p + SCORE_GREEN * (1 - p)


records = ["蓝"] * 5 + ["绿"] * 5       # 教学构造：旧策略玩的 10 局，蓝、绿各 5 局
scores = np.array([SCORE_BLUE if k == "蓝" else SCORE_GREEN for k in records])
avg_records = float(scores.mean())
print("10 局记录：", " ".join(f"{k}{s:g}" for k, s in zip(records, scores)))
table(["", "平均分"], [["10 局记录直接平均", num(avg_records)], ["旧策略 J(0.5)", num(J(P_OLD))],
                      [f"新策略 J({P_NEW:g})", num(J(P_NEW))]])
print(f"手算：(5 × 3 + 5 × 1) ÷ 10 = {num(avg_records)}；J({P_NEW:g}) = 1 + 2 × {P_NEW:g} = {num(J(P_NEW))}")
check("10 局直接平均 = 2 = 旧策略的 J(0.5)", avg_records == 2.0 and J(P_OLD) == 2.0)
check("更新一次后 p = 0.7，新策略的平均分 J(0.7) = 2.4（第 11 章 11.1 节）", P_NEW == 0.7 and math.isclose(J(P_NEW), 2.4))

rng = np.random.default_rng(0)
N_PLAYS = 100_000
blue = rng.random(N_PLAYS) < P_OLD                      # 旧策略再多玩 10 万局
avg_many = float(np.where(blue, SCORE_BLUE, SCORE_GREEN).mean())
print(f"旧策略玩 {N_PLAYS:,} 局，直接平均 = {avg_many:.4f}")
check("10 万局旧记录直接平均 2.0020（固定种子）", round(avg_many, 4) == 2.002)
check(f"多抽也没用：10 万局旧记录的平均仍在 J(0.5) = 2 附近，离新策略的 {J(P_NEW):g} 差得远",
      abs(avg_many - J(P_OLD)) < 0.02 and abs(avg_many - J(P_NEW)) > 0.3)
check("自测：p 更新到 0.9，10 局直接平均还是 2，J(0.9) = 2.8，差 0.8；p = 0.7 时只差 0.4",
      avg_records == 2.0 and math.isclose(J(0.9), 2.8) and math.isclose(J(0.9) - 2, 0.8) and math.isclose(J(0.7) - 2, 0.4))

# ---------------------------------------------------------------------------
banner("2. 比率：每局乘“新概率 ÷ 旧概率”；连续动作就是两口钟的高度之比")
w_blue, w_green = P_NEW / P_OLD, (1 - P_NEW) / (1 - P_OLD)
weights = np.where(np.array(records) == "蓝", w_blue, w_green)
avg_weighted = float((weights * scores).mean())
table(["键", "旧概率", "新概率", "比率 = 新 ÷ 旧", "局数"],
      [["蓝", num(P_OLD), num(P_NEW), num(w_blue), 5], ["绿", num(1 - P_OLD), num(1 - P_NEW), num(w_green), 5]])
print(f"加权平均 = (5 × {num(w_blue)} × 3 + 5 × {num(w_green)} × 1) ÷ 10 = {num(avg_weighted)}")
check("问卷的分量：(1/4) ÷ (1/2) = 0.5 份，(3/4) ÷ (1/2) = 1.5 份", (1 / 4) / (1 / 2) == 0.5 and (3 / 4) / (1 / 2) == 1.5)
check("比率：蓝 0.7 ÷ 0.5 = 1.4，绿 0.3 ÷ 0.5 = 0.6", math.isclose(w_blue, 1.4) and math.isclose(w_green, 0.6))
check("加权后 (5 × 1.4 × 3 + 5 × 0.6 × 1) ÷ 10 = 2.4 = J(0.7)（字面值重算）",
      math.isclose(avg_weighted, 2.4) and math.isclose((5 * 1.4 * 3 + 5 * 0.6 * 1) / 10, 2.4))
w_many = np.where(blue, w_blue, w_green)
avg_many_w = float((w_many * np.where(blue, SCORE_BLUE, SCORE_GREEN)).mean())
print(f"10 万局旧记录加权平均 = {avg_many_w:.4f}（新策略的真值 {J(P_NEW):g}）")
check("10 万局旧记录加权平均 2.4035（固定种子）", round(avg_many_w, 4) == 2.4035)
check(f"理论：加权以后，10 万局旧记录的平均落在新策略的 J = {J(P_NEW):g} 附近", abs(avg_many_w - J(P_NEW)) < 0.03)
check("自测：p = 0.9 时比率 1.8、0.2，(5 × 1.8 × 3 + 5 × 0.2 × 1) ÷ 10 = (27 + 1) ÷ 10 = 2.8 = J(0.9)",
      math.isclose(0.9 / 0.5, 1.8) and math.isclose(0.1 / 0.5, 0.2) and math.isclose((5 * 1.8 * 3 + 5 * 0.2 * 1) / 10, 2.8)
      and math.isclose(J(0.9), 2.8))

# 进阶折叠：梯度也一样（第 11 章 11.2 节：每局记“ln 概率的导数 × 分数”）
g_blue, g_green = SCORE_BLUE / P_NEW, -SCORE_GREEN / (1 - P_NEW)   # 按新策略 p 算：蓝 3/p，绿 −1/(1−p)
g_stale = float(np.where(np.array(records) == "蓝", g_blue, g_green).mean())
g_fixed = float((weights * np.where(np.array(records) == "蓝", g_blue, g_green)).mean())
print(f"梯度（p = {P_NEW:g}）：每局记 蓝 {g_blue:.4f}、绿 {g_green:.4f}；旧记录直接平均 {g_stale:.4f}；"
      f"乘比率后 {w_blue * g_blue:.4f}、{w_green * g_green:.4f}，平均 {g_fixed:.4f}；真值 2")
check("进阶折叠：新策略自己玩，0.7 × 4.2857 + 0.3 × (−3.3333) = 3 − 1 = 2（字面值重算）",
      round(0.7 * 4.2857, 4) == 3.0 and round(0.3 * 3.3333, 4) == 1.0 and math.isclose(0.7 * g_blue + 0.3 * g_green, 2.0))
check("进阶折叠：旧记录直接平均 (4.2857 − 3.3333) ÷ 2 = 0.4762，不是 2，小了 4 倍多（字面值重算）",
      round(g_stale, 4) == 0.4762 and round((4.2857 - 3.3333) / 2, 4) == 0.4762 and 4 < 2 / 0.4762 < 5)
check("进阶折叠：乘比率后 1.4 × 4.2857 = 6.0000、0.6 × (−3.3333) = −2.0000，平均回到 2（第 11 章的 6 和 −2）",
      round(w_blue * g_blue, 4) == 6.0 and round(w_green * g_green, 4) == -2.0 and math.isclose(g_fixed, 2.0)
      and round(1.4 * 4.2857, 4) == 6.0 and round(0.6 * -3.3333, 4) == -2.0)

# 连续动作：两口 σ = 1 的钟，旧中心 0、新中心 0.2；这一步做过的动作 a = 0.5
MU_OLD, MU_NEW, SIG, A_REC = 0.0, 0.2, 1.0, 0.5
h_old, h_new = bell(A_REC, MU_OLD, SIG), bell(A_REC, MU_NEW, SIG)
ln_old, ln_new = math.log(h_old), math.log(h_new)
rho = h_new / h_old
lp_old_t = Normal(MU_OLD, SIG).log_prob(torch.tensor(A_REC)).item()
lp_new_t = Normal(MU_NEW, SIG).log_prob(torch.tensor(A_REC)).item()
table(["", "旧钟（中心 0）", "新钟（中心 0.2）"],
      [["动作 0.5 处的高度", f"{h_old:.5f}", f"{h_new:.5f}"], ["ln 高度", f"{ln_old:.5f}", f"{ln_new:.5f}"],
       ["torch 的 log_prob", f"{lp_old_t:.5f}", f"{lp_new_t:.5f}"]])
print(f"直接除：{h_new:.5f} ÷ {h_old:.5f} = {rho:.4f}")
print(f"取 ln：{ln_new:.5f} − ({ln_old:.5f}) = {ln_new - ln_old:.5f}；exp({ln_new - ln_old:.2f}) = {math.exp(ln_new - ln_old):.4f}")
check("两个高度 0.35207、0.38139；比率 0.38139 ÷ 0.35207 = 1.0833（字面值重算）",
      round(h_old, 5) == 0.35207 and round(h_new, 5) == 0.38139 and round(rho, 4) == 1.0833
      and round(0.38139 / 0.35207, 4) == 1.0833)
check("ln 之差 = (0.5² − 0.3²) ÷ 2 = (0.25 − 0.09) ÷ 2 = 0.08：钟高公式里的 ln(σ√(2π)) 相减时抵消了",
      math.isclose(ln_new - ln_old, 0.08) and math.isclose((0.25 - 0.09) / 2, 0.08))
check("两条路同一个数：exp(0.08) = 1.0833 = 高度之比；torch 的 log_prob 也给出同样的差",
      round(math.exp(0.08), 4) == 1.0833 and math.isclose(math.exp(lp_new_t - lp_old_t), rho))

LN_TABLE = [(-1.0, -1.0), (-1.0, -0.8), (-1.0, -1.4)]          # 旧稿的三行：没变、更爱、更不爱
rows = [[f"{o:g}", f"{n:g}", f"{n - o:+.1f}", f"{math.exp(n - o):.4f}"] for o, n in LN_TABLE]
table(["ln π_old", "ln π_θ", "相减", "比率 = exp(相减)"], rows)
check("自测：−0.8 − (−1) = 0.2，exp(0.2) = 1.2214；−1.4 − (−1) = −0.4，exp(−0.4) = 0.6703；没变的是 1",
      round(math.exp(0.2), 4) == 1.2214 and round(math.exp(-0.4), 4) == 0.6703 and math.exp(0.0) == 1.0)

net = torch.nn.Linear(3, 14)                                    # 第一次更新之前：同一张网络、同一批观测
obs = torch.randn(5, 3, generator=torch.Generator().manual_seed(1))
act = torch.randn(5, 14, generator=torch.Generator().manual_seed(2))
lp_a = Normal(net(obs), 1.0).log_prob(act).sum(dim=-1)
lp_b = Normal(net(obs), 1.0).log_prob(act).sum(dim=-1)
check("第一次更新之前，新旧是同一张网络：14 个 ln 密度相加后相减恰好是 0，比率全是 1",
      torch.equal(torch.exp(lp_b - lp_a), torch.ones(5)))
check("14 个关节：比率 = 14 个关节各自比率的乘积；每个 1.02，合起来 1.02^14 = 1.3195，已经超过 1.2",
      round(1.02 ** 14, 4) == 1.3195 and math.isclose(math.exp(14 * math.log(1.02)), 1.02 ** 14))

# ---------------------------------------------------------------------------
banner("2b. 画图：figures/ch13_stale_data.png（直接平均是 2，乘上比率才是新策略的平均分）")
fig, axes = lesson_figure(3, f"过期的记录直接平均是 {num(avg_records)}，乘上新旧概率之比才是 {num(avg_weighted)}",
                          panel_height=3.15, width=10.6)
X0, CW, CH = 1.35, 0.88, 0.75                                    # 10 个格子：起点、宽、高


def ten_cells(ax, texts, bottom, fontsize=FS_STEP - 1):
    for i, (k, t) in enumerate(zip(records, texts)):
        face = "#e3edf8" if k == "蓝" else "#e2f1ea"
        cell(ax, X0 + i * CW, bottom, t, width=CW, height=CH, facecolor=face,
             color=BLUE if k == "蓝" else GREEN, fontsize=fontsize)


ax = axes[0]
lesson_panel(ax, f"① 旧策略 p = {num(P_OLD)} 玩了 10 局：蓝、绿各 5 局", xmax=10.6, ymax=4.0)
ax.text(0.15, 2.28, "得分", fontsize=FS_SMALL, color=INK, va="center")
ten_cells(ax, [f"{s:g}" for s in scores], bottom=1.9)
ax.text(X0 + 2.5 * CW, 2.95, "蓝键", fontsize=FS_SMALL, color=BLUE, ha="center", va="center")
ax.text(X0 + 7.5 * CW, 2.95, "绿键", fontsize=FS_SMALL, color=GREEN, ha="center", va="center")
hand(ax, 1.35, 1.25, f"直接平均 = (5 × 3 + 5 × 1) ÷ 10 = {num(avg_records)}")
note(ax, 0.15, 0.45, f"10 局是编的，为了好算。平均 {num(avg_records)} 正是旧策略的平均分 J(0.5)。")

ax = axes[1]
lesson_panel(ax, f"② 更新一次：p 变成 {num(P_NEW)}（第 11 章 11.1 节）", xmax=10.6, ymax=4.0)
hand(ax, 0.4, 2.55, f"新策略的平均分  J({num(P_NEW)}) = 1 + 2 × {num(P_NEW)} = {num(J(P_NEW))}", color=GREEN)
hand(ax, 0.4, 1.6, f"那 10 局直接平均  还是 {num(avg_records)}：过期了", color=ORANGE)
note(ax, 0.4, 0.6, f"记录里蓝、绿各占一半；新策略按蓝键的机会却是 {P_NEW * 100:.0f}%。")

ax = axes[2]
lesson_panel(ax, "③ 每局乘上“新概率 ÷ 旧概率”", xmax=10.6, ymax=4.0)
ax.text(0.15, 2.48, "乘", fontsize=FS_SMALL, color=INK, va="center")
ten_cells(ax, [f"×{num(w)}" for w in weights], bottom=2.1, fontsize=FS_SMALL - 1)
hand(ax, 1.35, 1.52, f"蓝：{num(P_NEW)} ÷ {num(P_OLD)} = {num(w_blue)}      绿：{num(1 - P_NEW)} ÷ {num(1 - P_OLD)} = {num(w_green)}",
     fontsize=FS_SMALL + 1)
hand(ax, 1.35, 0.85, f"(5 × {num(w_blue)} × 3 + 5 × {num(w_green)} × 1) ÷ 10 = {num(avg_weighted)}", color=GREEN)
note(ax, 0.15, 0.2, "新策略更常按的键，每局多算一点；更少按的，少算一点。")
savefig(fig, "ch13_stale_data")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("2c. 画图：figures/ch13_ratio_density.png（同一个动作，在两口钟下的高度之比）")
fig = plt.figure(figsize=(10, 10.6))
fig.suptitle("比率：同一个动作，在新旧两口钟下的高度之比", fontsize=23, fontweight="bold", color=INK, y=0.985)
ax = fig.add_axes([0.11, 0.43, 0.84, 0.43])
data_axes(ax, "动作 a（弧度）", "钟的高度（密度）")
xs = np.linspace(-3, 3.2, 400)
ax.plot(xs, bell(xs, MU_OLD, SIG), color=BLUE, lw=3)
ax.plot(xs, bell(xs, MU_NEW, SIG), color=ORANGE, lw=3)
ax.set_xlim(-3, 3.2)
ax.set_ylim(0, 0.46)
ax.set_xticks([-3, -2, -1, 0, 0.5, 1, 2, 3])
ax.set_xticklabels(["−3", "−2", "−1", "0", "0.5", "1", "2", "3"])
ax.axvline(A_REC, color=INK, lw=1.4, ls=":")
ax.plot([A_REC, A_REC], [h_old, h_new], color=INK, lw=0)
ax.scatter([A_REC], [h_old], s=90, color=BLUE, zorder=6)
ax.scatter([A_REC], [h_new], s=90, color=ORANGE, zorder=6)
ax.text(A_REC - 0.04, h_old - 0.075, f"旧钟 {h_old:.5f}", color=BLUE, fontsize=FS_SMALL, ha="right", va="center")
ax.text(A_REC + 0.12, h_new + 0.01, f"新钟 {h_new:.5f}", color=ORANGE, fontsize=FS_SMALL, va="bottom", bbox=WHITE_BOX)
ax.text(-2.9, 0.43, "蓝：旧钟，中心 0（采数据时）", color=BLUE, fontsize=FS_SMALL, va="top")
ax.text(-2.9, 0.395, "橙：新钟，中心 0.2（更新后）", color=ORANGE, fontsize=FS_SMALL, va="top")
ax.text(A_REC + 0.05, 0.02, "这一步做过的动作", color=INK, fontsize=FS_SMALL - 2, ha="left")
panel_title(fig, [ax], "① 同一个动作 a = 0.5，在两口钟下各有一个高度")
panel_note(fig, [ax], "两口钟一样宽（σ = 1），新钟往右挪了 0.2：动作 0.5 离新中心更近，所以更高。")
ax2 = fig.add_axes([0.03, 0.0, 0.94, 0.27])
lesson_panel(ax2, "② 比率 = 两个高度相除；代码先取 ln 再相减，是同一个数", xmax=10, ymax=4.0)
hand(ax2, 0.3, 2.85, f"直接除：{h_new:.5f} ÷ {h_old:.5f} = {rho:.4f}")
hand(ax2, 0.3, 2.0, f"取 ln：{m(ln_new)} − ({m(ln_old)}) = {ln_new - ln_old:.2f}", color=GREEN)
hand(ax2, 0.3, 1.2, f"再 exp：exp({ln_new - ln_old:.2f}) = {math.exp(ln_new - ln_old):.4f}", color=GREEN)
note(ax2, 0.3, 0.35, "相减时，钟高公式里的 ln(σ√(2π)) 两边一样大，抵消了（这里的 π 是 3.14159…）。")
savefig(fig, "ch13_ratio_density")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("3. 裁剪：好处到 1.2（或 0.8）封顶，坏处照算")
CLIP = 0.2  # TWEAK-2: 0.1
LO, HI = 1 - CLIP, 1 + CLIP

# 为什么要刹车：一条好记录（动作 0.5，Â = +2），把钟挪到 0.5 再收窄到 σ = 0.1
h_narrow = bell(A_REC, A_REC, 0.1)
rho_narrow = h_narrow / h_old
print(f"新钟中心 0.5、σ = 0.1：动作 0.5 处高 {h_narrow:.4f}；比率 = {h_narrow:.4f} ÷ {h_old:.5f} = {rho_narrow:.2f}；"
      f"目标 ρ × Â = {rho_narrow * 2:.2f}（原来 2）")
check("没有刹车：3.989 ÷ 0.35207 = 11.33，ρ × Â = 22.66（字面值重算）",
      round(h_narrow, 3) == 3.989 and round(rho_narrow, 2) == 11.33 and round(3.989 / 0.35207, 2) == 11.33
      and round(11.33 * 2, 2) == 22.66)


def clipped_objective(ratio, adv, lo=None, hi=None):
    """一条记录的裁剪目标：min(ρ × Â, clip(ρ, lo, hi) × Â)。"""
    lo = LO if lo is None else lo
    hi = HI if hi is None else hi
    return torch.min(ratio * adv, torch.clamp(ratio, lo, hi) * adv)


def slope(r, a):
    """这一条的目标对 ρ 的导数：torch 自动求导。"""
    rt = torch.tensor(float(r), requires_grad=True)
    clipped_objective(rt, torch.tensor(float(a))).backward()
    return rt.grad.item()


FOUR = [(1.5, 2.0), (0.5, 2.0), (0.5, -2.0), (1.5, -2.0)]         # 冻结的四格（worked_examples 第 13 章）
FOUR_EXPECT = {(1.5, 2.0): 2.4, (0.5, 2.0): 1.0, (0.5, -2.0): -1.6, (1.5, -2.0): -3.0}
rows = []
four_val, four_slope = {}, {}
for r, a in FOUR:
    un = r * a
    cl = min(HI, max(LO, r)) * a
    v = clipped_objective(torch.tensor(r), torch.tensor(a)).item()
    h = 1e-6
    fd = (clipped_objective(torch.tensor(r + h), torch.tensor(a)).item()
          - clipped_objective(torch.tensor(r - h), torch.tensor(a)).item()) / (2 * h)
    four_val[(r, a)], four_slope[(r, a)] = v, slope(r, a)
    flat = (a > 0 and r > HI) or (a < 0 and r < LO)
    rows.append([f"({r:g}, {a:+g})", num(un), num(cl), num(v), num(fd, 3), "平坦区" if flat else "斜线"])
table(["(ρ, Â)", "不裁剪 ρÂ", f"裁剪后 clip(ρ, {num(LO)}, {num(HI)})Â", "两者取小", "缩 h 量的斜率", "在哪"], rows)
check(f"四格（裁剪范围 {num(LO)}–{num(HI)}）：min(3, 2.4) = 2.4，min(1, 1.6) = 1，min(−1, −1.6) = −1.6，min(−3, −2.4) = −3",
      all(math.isclose(four_val[k], FOUR_EXPECT[k]) for k in FOUR))
check("斜率：平坦区（Â > 0 且 ρ > 1.2，或 Â < 0 且 ρ < 0.8）是 0，其余等于 Â",
      all(math.isclose(four_slope[(r, a)], 0.0 if ((a > 0 and r > 1.2) or (a < 0 and r < 0.8)) else a) for r, a in FOUR)
      and CLIP == 0.2)
check("ρ 从 1.5 再涨到 1.6，(Â = 2) 的目标还是 2.4：这一条不再奖励继续变",
      math.isclose(clipped_objective(torch.tensor(1.6), torch.tensor(2.0)).item(), 2.4))
v_self = clipped_objective(torch.tensor(1.4), torch.tensor(-1.0)).item()
check("自测：Â = −1、ρ = 1.4 → min(−1.4, −1.2) = −1.4，斜率 −1，不在平坦区",
      math.isclose(v_self, -1.4) and math.isclose(slope(1.4, -1.0), -1.0))

SHAPE_R = [0.5, 0.8, 1.0, 1.2, 1.5]
table(["ρ"] + [num(r) for r in SHAPE_R],
      [["Â = +2 时的目标"] + [num(clipped_objective(torch.tensor(r), torch.tensor(2.0)).item()) for r in SHAPE_R],
       ["Â = −2 时的目标"] + [num(clipped_objective(torch.tensor(r), torch.tensor(-2.0)).item()) for r in SHAPE_R]])

# 一大批记录里有多少落在平坦区（合成数据：新旧 ln π 之差的标准差 0.15，Â 正负各半）
N_FLAT = 10_000
LOG_SPREAD = 0.15
rng_f = np.random.default_rng(0)
log_ratio = rng_f.normal(0.0, LOG_SPREAD, N_FLAT)
adv_f = rng_f.normal(0.0, 1.0, N_FLAT)
ratio_f = np.exp(log_ratio)
flat_mask = ((ratio_f > HI) & (adv_f > 0)) | ((ratio_f < LO) & (adv_f < 0))
frac = float(flat_mask.mean())
theory = 0.5 * (1 - 0.5 * (1 + math.erf(math.log(HI) / LOG_SPREAD / math.sqrt(2)))) + \
    0.5 * 0.5 * (1 + math.erf(math.log(LO) / LOG_SPREAD / math.sqrt(2)))
print(f"{N_FLAT:,} 条记录，ln 之差的标准差 {LOG_SPREAD}：比率 {ratio_f.min():.3f} ~ {ratio_f.max():.3f}，"
      f"落在平坦区的 {frac * 100:.1f}%（理论 {theory * 100:.1f}%）")
check("9.0% 的记录落在平坦区（固定种子）", round(frac * 100, 1) == 9.0)
check(f"理论：落在平坦区的比例 ≈ {theory * 100:.1f}%（用高斯的面积算，容差 1 个百分点）", abs(frac - theory) < 0.01)

# 进阶折叠：平坦区的这一条不推了，不等于它的比率不再变——两条记录共用一个旋钮 μ
A1, A2 = 2.0, 0.3                                               # 两个动作；优势都是 +1；σ = 1 不动；μ 从 0 出发
mu = torch.tensor(0.0, requires_grad=True)
opt = torch.optim.SGD([mu], lr=0.5)
for _ in range(400):
    r1 = torch.exp(A1 * mu - mu ** 2 / 2)                        # ln 高度之差 = aμ − μ²/2（σ = 1、旧中心 0）
    r2 = torch.exp(A2 * mu - mu ** 2 / 2)
    obj = (clipped_objective(r1, torch.tensor(1.0), 0.8, 1.2) + clipped_objective(r2, torch.tensor(1.0), 0.8, 1.2)) / 2
    opt.zero_grad()
    (-obj).backward()
    opt.step()
mu_end = mu.item()
r1_end = math.exp(A1 * mu_end - mu_end ** 2 / 2)
mu_enter = 2 - math.sqrt(4 - 2 * math.log(1.2))                 # 2μ − μ²/2 = ln 1.2 的较小根
print(f"记录 1 在 μ = {mu_enter:.3f} 时比率到 1.2、进平坦区；记录 2 继续推，μ 停在 {mu_end:.3f}；"
      f"这时记录 1 的真实比率 = exp(2 × 0.3 − 0.3²/2) = exp(0.555) = {r1_end:.3f}")
check("进阶折叠：记录 2 的比率最多 exp(0.3² ÷ 2) = exp(0.045) = 1.046，到不了 1.2", round(math.exp(0.3 ** 2 / 2), 3) == 1.046)
check("进阶折叠：μ 停在 0.3；记录 1 的比率 exp(0.555) = 1.742，远远超过 1.2（字面值重算：(4 − 2.89) ÷ 2 = 0.555）",
      abs(mu_end - 0.3) < 1e-4 and round(math.exp(0.555), 3) == 1.742 and math.isclose((4 - 2.89) / 2, 0.555)
      and math.isclose(1.7 ** 2, 2.89) and round(mu_enter, 3) == 0.093)

# ---------------------------------------------------------------------------
banner("3b. 画图：figures/ch13_ppo_clip.png（四格与平坦区）")
fig = plt.figure(figsize=(10, 13.4))
ax_top = fig.add_axes([0.13, 0.575, 0.82, 0.30])
ax_bot = fig.add_axes([0.13, 0.105, 0.82, 0.30])
rr = np.linspace(0.3, 1.7, 400)
for ax, a in ((ax_top, 2.0), (ax_bot, -2.0)):
    data_axes(ax, "比率 ρ（新 ÷ 旧）", "这一条记录的目标")
    obj_line = np.minimum(rr * a, np.clip(rr, LO, HI) * a)
    ax.axvspan(LO, HI, color=CELL, zorder=0)
    ax.text((LO + HI) / 2, (3.45 if a > 0 else -0.55), f"{num(LO)} ~ {num(HI)}", color=BLUE, fontsize=FS_SMALL - 1,
            ha="center", va="center")
    ax.plot(rr, rr * a, color=FAINT, lw=2.4, ls="--")
    ax.plot(rr, obj_line, color=BLUE, lw=3.4)
    flat = (rr >= HI) if a > 0 else (rr <= LO)
    ax.plot(rr[flat], obj_line[flat], color=ORANGE, lw=7, solid_capstyle="butt", zorder=4)
    ax.set_xlim(0.3, 1.7)
    ax.set_xticks([0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
    ax.set_xticklabels([f"{t:g}" for t in [0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6]])
    if a > 0:
        ax.set_ylim(0.4, 3.6)
        ax.set_yticks([1, LO * 2, 2, HI * 2, 3])
        ax.set_yticklabels([m(t, "g") for t in (1, LO * 2, 2, HI * 2, 3)])
        ax.scatter([1.5], [3.0], s=80, color=FAINT, zorder=5)
        ax.scatter([1.5], [four_val[(1.5, 2.0)]], s=110, color=ORANGE, zorder=6)
        arrow(ax, (1.5, 2.95), (1.5, four_val[(1.5, 2.0)] + 0.07), ORANGE, lw=2.2)
        ax.text(1.53, 3.0, "不裁剪：3", color=MUTED, fontsize=FS_SMALL - 1, va="center")
        ax.text(1.46, four_val[(1.5, 2.0)] - 0.22, f"(1.5, 2) → min(3, {num(HI * 2)}) = {num(four_val[(1.5, 2.0)])}",
                color=ORANGE, fontsize=FS_SMALL - 1, ha="right", va="center", bbox=WHITE_BOX)
        ax.scatter([0.5], [four_val[(0.5, 2.0)]], s=110, color=BLUE, zorder=6)
        ax.text(0.55, four_val[(0.5, 2.0)] - 0.2, f"(0.5, 2) → min(1, {num(LO * 2)}) = {num(four_val[(0.5, 2.0)])}",
                color=BLUE, fontsize=FS_SMALL - 1, va="center", bbox=WHITE_BOX)
        ax.text(1.635, HI * 2 - 0.34, "平坦区\n斜率 0", color=ORANGE, fontsize=FS_SMALL - 1, ha="center", va="center")
        ax.text(0.33, 3.35, "虚线：不裁剪 ρ × Â", color=MUTED, fontsize=FS_SMALL - 1, va="center")
    else:
        ax.set_ylim(-3.6, -0.4)
        ax.set_yticks([-3, -HI * 2, -2, -LO * 2, -1])
        ax.set_yticklabels([m(t, "g") for t in (-3, -HI * 2, -2, -LO * 2, -1)])
        ax.scatter([0.5], [-1.0], s=80, color=FAINT, zorder=5)
        ax.scatter([0.5], [four_val[(0.5, -2.0)]], s=110, color=ORANGE, zorder=6)
        arrow(ax, (0.5, -1.05), (0.5, four_val[(0.5, -2.0)] + 0.07), ORANGE, lw=2.2)
        ax.text(0.53, -1.0, "不裁剪：−1", color=MUTED, fontsize=FS_SMALL - 1, va="center")
        ax.text(0.33, four_val[(0.5, -2.0)] - 0.6, f"(0.5, −2) → min(−1, −{num(LO * 2)}) = −{num(-four_val[(0.5, -2.0)])}",
                color=ORANGE, fontsize=FS_SMALL - 1, va="center", bbox=WHITE_BOX)
        ax.scatter([1.5], [four_val[(1.5, -2.0)]], s=110, color=BLUE, zorder=6)
        ax.text(1.45, four_val[(1.5, -2.0)] + 0.05, f"(1.5, −2) → min(−3, −{num(HI * 2)}) = −{num(-four_val[(1.5, -2.0)])}",
                color=BLUE, fontsize=FS_SMALL - 1, ha="right", va="center", bbox=WHITE_BOX)
        ax.text(0.37, -LO * 2 + 0.3, "平坦区\n斜率 0", color=ORANGE, fontsize=FS_SMALL - 1, ha="center", va="center")
fig.suptitle("PPO 的裁剪：好处到边界就封顶，坏处照单全收", fontsize=23, fontweight="bold", color=INK, y=0.985)
panel_title(fig, [ax_top], f"① Â = +2（好动作）：ρ 涨过 {num(HI)} 就不再加分；跌下去照样扣")
panel_note(fig, [ax_top], "橙色平台：这一条再往这边走也不加分，斜率是 0，不再推它。\n"
                          "蓝色斜线：没过界照常加分；往坏的方向走一分不少地扣，斜率照样把它拉回来。")
panel_title(fig, [ax_bot], f"② Â = −2（坏动作）：ρ 跌过 {num(LO)} 就不再加分；涨上去照样扣")
panel_note(fig, [ax_bot], "坏动作的概率已经压低了不少，再压也不加分；它的概率要是反而涨了，照扣。")
savefig(fig, "ch13_ppo_clip")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("4. 完整的损失：−裁剪目标 + 1.0 × 价值损失 − 0.01 × 熵")
ratio4 = torch.tensor([r for r, _ in FOUR])
adv4 = torch.tensor([a for _, a in FOUR])
objective_mean = clipped_objective(ratio4, adv4).mean().item()
surrogate = -adv4 * ratio4                                            # rsl_rl 的写法：先取负号
surrogate_clipped = -adv4 * torch.clamp(ratio4, LO, HI)
surrogate_loss4 = torch.max(surrogate, surrogate_clipped).mean().item()
table(["", "值"], [["教科书：裁剪目标的平均（要变大）", num(objective_mean)],
                  ["rsl_rl：surrogate_loss（要变小）", num(surrogate_loss4)]])
print("手算：(2.4 + 1 − 1.6 − 3) ÷ 4 = −0.3；取负号 → 0.3")
check("四格当一小批：裁剪目标的平均 = (2.4 + 1 − 1.6 − 3) ÷ 4 = −0.3，surrogate_loss = 0.3",
      math.isclose(objective_mean, -0.3) and math.isclose(surrogate_loss4, 0.3) and CLIP == 0.2)
check("rsl_rl 的 max(−…, −…) = −（教科书的 min(…, …)），换什么裁剪范围都成立", math.isclose(surrogate_loss4, -objective_mean))

# 价值裁剪：一条记录，采数据时 V = 0.2，标签（returns）0.5；更新后 V 可能在三个位置
V_OLD, RET = 0.2, 0.5
V_CASES = [0.55, 0.35, -0.1]


def value_loss_parts(v_new, v_old=V_OLD, ret=RET, clip=0.2):
    v = torch.tensor(v_new, requires_grad=True)
    v_clip = v_old + (v - v_old).clamp(-clip, clip)
    unclipped = (v - ret) ** 2
    clipped = (v_clip - ret) ** 2
    loss = torch.max(unclipped, clipped)
    loss.backward()
    return unclipped.item(), v_clip.item(), clipped.item(), loss.item(), v.grad.item()


rows, vparts = [], {}
for v_new in V_CASES:
    un, vc, cl, lo_, g = value_loss_parts(v_new)
    vparts[v_new] = (un, vc, cl, lo_, g)
    rows.append([num(v_new), num(un), num(vc), num(cl), num(lo_), num(g)])
table(["更新后的 V", "不裁剪 (V − 0.5)²", "裁剪后的 V", "(它 − 0.5)²", "取大的", "对 V 的斜率"], rows)
check("价值裁剪：V = 0.55 → max(0.0025, 0.01) = 0.01，用的是裁剪版，斜率 0（这条不再推 critic）",
      math.isclose(vparts[0.55][0], 0.0025) and math.isclose(vparts[0.55][1], 0.4) and math.isclose(vparts[0.55][3], 0.01)
      and vparts[0.55][4] == 0.0)
check("V = 0.35（没挪够 0.2）→ 两个都是 0.0225，斜率 2 × (0.35 − 0.5) = −0.3，照常推",
      math.isclose(vparts[0.35][0], 0.0225) and math.isclose(vparts[0.35][3], 0.0225) and math.isclose(vparts[0.35][4], -0.3))
check("V = −0.1（往反方向跑）→ max(0.36, 0.25) = 0.36，用不裁剪版，斜率 −1.2，照常拉回",
      math.isclose(vparts[-0.1][0], 0.36) and math.isclose(vparts[-0.1][2], 0.25) and math.isclose(vparts[-0.1][3], 0.36)
      and math.isclose(vparts[-0.1][4], -1.2))

ent_one = Normal(0.0, 1.0).entropy().item()
ent14 = Normal(torch.zeros(14), torch.ones(14)).entropy().sum().item()
print(f"熵：一口 σ = 1 的钟 {ent_one:.5f}，14 口相加 {ent14:.3f}；乘 0.01 = {0.01 * ent14:.3f}（第 11 章 11.6 节）")
check("熵：14 × 1.41894 = 19.865，乘 0.01 是 0.199", round(ent14, 3) == 19.865 and round(0.01 * ent14, 3) == 0.199)

VAL_COEF, ENT_COEF = 1.0, 0.01
loss_one = -four_val[(1.5, 2.0)] + VAL_COEF * vparts[0.55][3] - ENT_COEF * round(ent14, 3)
print(f"一条记录的总账：−{num(four_val[(1.5, 2.0)])} + 1.0 × {num(vparts[0.55][3])} − 0.01 × 19.865 = {loss_one:.5f}")
check("一条记录的总账：−2.4 + 0.01 − 0.19865 = −2.58865（字面值重算）",
      math.isclose(loss_one, -2.58865) and math.isclose(-2.4 + 0.01 - 0.19865, -2.58865))

# 优势归一化之后：比率全是 1 时，surrogate_loss 的值是 0，梯度却不是 0
raw = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
adv_n = (raw - raw.mean()) / (raw.std() + 1e-8)                       # 第 12 章 12.7 节
rho1 = torch.ones(8, requires_grad=True)
sl = torch.max(-adv_n * rho1, -adv_n * torch.clamp(rho1, LO, HI)).mean()
sl.backward()
print(f"8 条归一化后的 Â：{adv_n.numpy().round(4)}；比率全是 1 → surrogate_loss = {abs(sl.item()):.6f}；"
      f"对第 0 条比率的斜率 = {rho1.grad[0].item():.4f}")
check("比率全是 1：surrogate_loss 是 0（归一化后 Â 的平均是 0），可每一条的斜率 −Â ÷ 8 都不是 0",
      abs(sl.item()) < 1e-12 and torch.allclose(rho1.grad, -adv_n / 8) and rho1.grad.abs().min().item() > 0.01)
rho_moved = torch.exp(0.05 * adv_n)                                  # 教学构造：好动作概率涨一点、坏动作降一点
sl_moved = torch.max(-adv_n * rho_moved, -adv_n * torch.clamp(rho_moved, LO, HI)).mean().item()
print(f"比率往好的方向挪一点（ρ = exp(0.05 × Â)）：surrogate_loss = {sl_moved:.3f}")
check("策略往好的方向挪了一点，surrogate_loss 就成了小负数 −0.044", round(sl_moved, 3) == -0.044)

# ---------------------------------------------------------------------------
banner("5. KL 散度：两口钟差多远；超过 0.02 学习率 ÷ 1.5，不到 0.005 × 1.5")
SHIFT = 0.2  # TWEAK-3: 0.4


def kl_same_sigma(shift, sigma=1.0):
    """σ 相同的两口钟：KL = (1 + (挪动/σ)²) ÷ 2 − 0.5 = (挪动/σ)² ÷ 2。"""
    return (1 + (shift / sigma) ** 2) / 2 - 0.5


kl_a = kl_same_sigma(SHIFT)
kl_half = kl_same_sigma(SHIFT / 2)
kl_torch = kl_divergence(Normal(0.0, 1.0), Normal(SHIFT, 1.0)).item()
table(["中心挪多远（σ = 1）", "手算 KL", "torch"],
      [[num(SHIFT / 2), f"{kl_half:.4g}", f"{kl_divergence(Normal(0.0, 1.0), Normal(SHIFT / 2, 1.0)).item():.4g}"],
       [num(SHIFT), f"{kl_a:.4g}", f"{kl_torch:.4g}"]])
check("KL：σ = 1、中心挪 0.2 → (1 + 0.2²) ÷ 2 − 0.5 = 0.52 − 0.5 = 0.02（字面值重算）",
      math.isclose(kl_a, 0.02) and math.isclose((1 + 0.2 ** 2) / 2 - 0.5, 0.02) and math.isclose(kl_torch, 0.02))
check("挪 0.1 → 0.005", math.isclose(kl_half, 0.005) and SHIFT == 0.2)
check(f"理论：挪动翻倍，KL 翻四倍（现在 {kl_a:.4g} ÷ {kl_half:.4g} = {kl_a / kl_half:.4g}）", math.isclose(kl_a / kl_half, 4.0))
d_target = math.sqrt(2 * 0.01)
d_14 = math.sqrt(2 * 0.01 / 14)
print(f"KL = 0.01 对应挪 {d_target:.4f} 个 σ；14 个关节一起挪、合计 0.01：每个挪 {d_14:.4f} 个 σ")
check("目标 0.01：一个关节挪 0.1414 σ（0.1414² ÷ 2 ≈ 0.01）；14 个关节合计 0.01，每个只挪 0.0378 σ（14 × 0.0378² ÷ 2 ≈ 0.01）",
      round(d_target, 4) == 0.1414 and round(d_14, 4) == 0.0378 and round(0.1414 ** 2 / 2, 4) == 0.01
      and round(14 * 0.0378 ** 2 / 2, 4) == 0.01)
kl14 = kl_divergence(Normal(torch.zeros(14), torch.ones(14)), Normal(torch.full((14,), 0.2), torch.ones(14))).sum().item()
check("14 个关节的中心都挪 0.2 σ：14 × 0.02 = 0.28", math.isclose(kl14, 0.28))

mu1, s1, mu2, s2 = 0.0, 1.0, 0.3, 0.8                            # 进阶折叠：宽度也变
kl_on = math.log(s2 / s1) + (s1 ** 2 + (mu1 - mu2) ** 2) / (2 * s2 ** 2) - 0.5
kl_no = math.log(s1 / s2) + (s2 ** 2 + (mu2 - mu1) ** 2) / (2 * s1 ** 2) - 0.5
kl_on_t = kl_divergence(Normal(mu1, s1), Normal(mu2, s2)).item()
table(["", "手算公式", "torch"], [["KL(旧‖新)", f"{kl_on:.4f}", f"{kl_on_t:.4f}"],
                                 ["KL(新‖旧)", f"{kl_no:.4f}", f"{kl_divergence(Normal(mu2, s2), Normal(mu1, s1)).item():.4f}"]])
check("进阶折叠：旧钟 (0, 1)、新钟 (0.3, 0.8)：KL(旧‖新) = ln 0.8 + 1.09 ÷ 1.28 − 0.5 = 0.1284，和 torch 一致",
      round(kl_on, 4) == 0.1284 and math.isclose(kl_on, kl_on_t) and round(math.log(0.8) + 1.09 / 1.28 - 0.5, 4) == 0.1284)
check("进阶折叠：反过来 KL(新‖旧) = 0.0881 ≠ 0.1284：不对称，所以叫散度", round(kl_no, 4) == 0.0881)
check("同一口钟和自己比：KL 恰好是 0（旋钮和输入归一化器都没动时就是这样）",
      kl_divergence(Normal(torch.tensor([0.3]), torch.tensor([0.8])), Normal(torch.tensor([0.3]), torch.tensor([0.8]))).item() == 0.0)

DESIRED_KL = 0.01


def adapt(lr, kl):
    """rsl_rl 的规则：KL 超过目标的 2 倍 → ÷ 1.5（最低 1e-5）；不到一半且大于 0 → × 1.5（最高 1e-2）；否则不动。"""
    if kl > DESIRED_KL * 2.0:
        return max(1e-5, lr / 1.5)
    if kl < DESIRED_KL / 2.0 and kl > 0.0:
        return min(1e-2, lr * 1.5)
    return lr


KL_SEQ = [0.03, 0.025, 0.012, 0.004, 0.003, 0.02]              # 旧稿的六次
lr, lr_seq = 1e-3, []
for kl in KL_SEQ:
    lr = adapt(lr, kl)
    lr_seq.append(lr)
table(["这次量到的 KL", "调整后的学习率", "以 0.001 为单位"], [[f"{k:g}", f"{v:.9f}", f"{v / 1e-3:.4f}"] for k, v in zip(KL_SEQ, lr_seq)])
check("0.03 > 0.02：0.001 ÷ 1.5 = 0.00066667；0.025：再 ÷ 1.5，= 0.001 ÷ 2.25 = 0.00044444（字面值重算）",
      math.isclose(lr_seq[0], 0.001 / 1.5) and round(lr_seq[0], 8) == 0.00066667
      and math.isclose(lr_seq[1], 0.001 / 2.25) and round(lr_seq[1], 8) == 0.00044444)
check("0.012 在 0.005 ~ 0.02 之间：不动；0.004、0.003 < 0.005：各 × 1.5，回到 0.001",
      lr_seq[2] == lr_seq[1] and math.isclose(lr_seq[3], 0.001 / 1.5) and math.isclose(lr_seq[4], 0.001))
check("0.02 恰好是目标的 2 倍：规则写的是“大于”，不动", lr_seq[5] == lr_seq[4] and adapt(1e-3, 0.02) == 1e-3)
check("KL = 0（第一个 mini-batch）：规则要求“大于 0”才放大，不动", adapt(1e-3, 0.0) == 1e-3)
lr_low = 1e-3
for _ in range(20):
    lr_low = adapt(lr_low, 0.05)
lr_high = 1e-3
for _ in range(20):
    lr_high = adapt(lr_high, 0.001)
check("上下限：连着 20 次太大，学习率停在 0.00001；连着 20 次太小，停在 0.01", lr_low == 1e-5 and lr_high == 1e-2)
lr_t = [adapt(1e-3, 0.021)]
lr_t.append(adapt(lr_t[-1], 0.02))
lr_t.append(adapt(lr_t[-1], 0.004))
check("自测：0.021 → 0.001 × 2/3 = 0.00066667；0.02 → 不动；0.004 → × 1.5，回到 0.001",
      round(lr_t[0], 8) == 0.00066667 and lr_t[1] == lr_t[0] and math.isclose(lr_t[2], 0.001))

# 进阶折叠：第一次更新时 KL 真的是 0 吗？用 rsl_rl 自己的网络类（带输入归一化）演示
try:
    from tensordict import TensorDict
    from rsl_rl.models.mlp_model import MLPModel
except ImportError:
    MLPModel = None
if MLPModel is None:
    print("  （当前 Python 环境里没有 rsl_rl，跳过这个演示；用 uv run 运行就有）")
else:
    def first_update(n_seen):
        """归一化器先见过 n_seen 批观测；采数据后它又更新一次；旋钮不动，重算 KL 和比率。"""
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(0)
        batch_obs = TensorDict({"actor": torch.randn(512, 3, generator=g)}, batch_size=[512])
        model = MLPModel(batch_obs, {"actor": ["actor"]}, "actor", 2, hidden_dims=(8,), obs_normalization=True,
                         distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"})
        model.train()
        for _ in range(n_seen):
            model.update_normalization(TensorDict({"actor": torch.randn(512, 3, generator=g)}, batch_size=[512]))
        with torch.no_grad():
            acts = model(batch_obs, stochastic_output=True)                  # act()：抽动作，存 ln π 和钟的参数
            lp_old_m = model.get_output_log_prob(acts).clone()
            params_old = tuple(p.clone() for p in model.output_distribution_params)
            shifted = torch.randn(512, 3, generator=g) + 0.3                # 采数据时，归一化器继续跟着新观测挪
            model.update_normalization(TensorDict({"actor": shifted}, batch_size=[512]))
            model(batch_obs, stochastic_output=True)                        # update()：旋钮没动，重算
            kl_m = model.get_kl_divergence(params_old, model.output_distribution_params).mean().item()
            r_m = torch.exp(model.get_output_log_prob(acts) - lp_old_m)
        return kl_m, r_m.min().item(), r_m.max().item()

    torch.set_default_dtype(torch.float32)                                  # 和训练时一样：32 位小数
    kl_early, rmin_e, rmax_e = first_update(5)
    kl_late, rmin_l, rmax_l = first_update(2000)
    kl_tiny32 = kl_divergence(Normal(torch.tensor(0.0), torch.tensor(1.0)), Normal(torch.tensor(0.0002), torch.tensor(1.0))).item()
    torch.set_default_dtype(torch.float64)
    kl_tiny64 = kl_divergence(Normal(0.0, 1.0), Normal(0.0002, 1.0)).item()
    table(["归一化器见过几批", "第一次更新的 KL", "比率的范围"],
          [["5（训练早期）", f"{kl_early:.2g}", f"{rmin_e:.4f} ~ {rmax_e:.4f}"],
           ["2000（训练很久）", f"{kl_late:g}", f"{rmin_l:.5f} ~ {rmax_l:.5f}"]])
    print(f"中心只差 0.0002：64 位小数算 KL = {kl_tiny64:.8f}（就是 0.0002² ÷ 2），32 位小数算出来 = {kl_tiny32:g}")
    check("进阶折叠的演示数：早期 KL 0.0003、比率 0.9240 ~ 1.1055；很久以后比率 0.99974 ~ 1.00033",
          round(kl_early, 4) == 0.0003 and round(rmin_e, 4) == 0.924 and round(rmax_e, 4) == 1.1055
          and round(rmin_l, 5) == 0.99974 and round(rmax_l, 5) == 1.00033)
    check("进阶折叠：训练早期，旋钮没动，KL 也不是 0，而是比 0.005 小的正数 → 学习率先 × 1.5",
          0 < kl_early < 0.005 and math.isclose(adapt(1e-3, kl_early), 1.5e-3))
    check("进阶折叠：训练很久，比率仍不是恰好 1，可 32 位小数算出的 KL 恰好是 0 → 学习率不动",
          kl_late == 0.0 and (rmin_l < 1.0 or rmax_l > 1.0) and adapt(1e-3, kl_late) == 1e-3)
    check("进阶折叠：中心差 0.0002 的两口钟，KL 该是 0.0002² ÷ 2 = 0.00000002（公式里先算 1 + 0.0002² = 1 + 0.00000004），32 位小数算出来是 0",
          math.isclose(kl_tiny64, 2e-8, rel_tol=1e-6) and math.isclose(0.0002 ** 2, 4e-8) and kl_tiny32 == 0.0
          and float(torch.tensor(1.0, dtype=torch.float32) + torch.tensor(4e-8, dtype=torch.float32)) == 1.0)

# ---------------------------------------------------------------------------
banner("5b. 画图：figures/ch13_kl_shift.png（挪一倍远，KL 变四倍；三段规则）")
fig = plt.figure(figsize=(10, 13.2))
ax_top = fig.add_axes([0.12, 0.60, 0.83, 0.27])
ax_bot = fig.add_axes([0.12, 0.11, 0.83, 0.33])
data_axes(ax_top, "动作（弧度）", "钟的高度（密度）")
xs = np.linspace(-3, 3, 400)
ax_top.plot(xs, bell(xs, 0.0, 1.0), color=BLUE, lw=3)
ax_top.plot(xs, bell(xs, SHIFT / 2, 1.0), color=GREEN, lw=2.6, ls="--")
ax_top.plot(xs, bell(xs, SHIFT, 1.0), color=ORANGE, lw=3)
ax_top.set_xlim(-3, 3)
ax_top.set_ylim(0, 0.5)
ax_top.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax_top.set_xticklabels(["−3", "−2", "−1", "0", "1", "2", "3"])
ax_top.text(-2.9, 0.47, "蓝：旧钟，中心 0", color=BLUE, fontsize=FS_SMALL, va="top")
ax_top.text(-2.9, 0.425, f"绿虚线：中心挪 {num(SHIFT / 2)}，KL = {kl_half:.3g}", color=GREEN, fontsize=FS_SMALL, va="top")
ax_top.text(-2.9, 0.38, f"橙：中心挪 {num(SHIFT)}，KL = {kl_a:.3g}", color=ORANGE, fontsize=FS_SMALL, va="top")
panel_title(fig, [ax_top], "① 三口一样宽的钟（σ = 1），中心各挪一点")
panel_note(fig, [ax_top], "三口钟几乎叠在一起：KL 量的就是这种肉眼难分的差别。")
data_axes(ax_bot, "中心挪了多远（以 σ 为单位）", "KL")
d_max = max(0.3, SHIFT * 1.15)
ds = np.linspace(0, d_max, 300)
kl_top = kl_same_sigma(d_max) * 1.08
ax_bot.axhspan(0, DESIRED_KL / 2, color="#e2f1ea", zorder=0)
ax_bot.axhspan(DESIRED_KL * 2, kl_top, color=CELL_HOT, zorder=0)
ax_bot.plot(ds, kl_same_sigma(ds), color=INK, lw=3)
for d, c in ((SHIFT / 2, GREEN), (d_target, BLUE), (SHIFT, ORANGE)):
    k = kl_same_sigma(d)
    ax_bot.plot([d, d], [0, k], color=c, lw=1.3, ls=":")
    ax_bot.plot([0, d], [k, k], color=c, lw=1.3, ls=":")
    ax_bot.scatter([d], [k], s=90, color=c, zorder=6)
ax_bot.text(SHIFT / 2 - 0.005, kl_half + 0.0008, f"({num(SHIFT / 2)}, {kl_half:.3g})", color=GREEN, fontsize=FS_SMALL - 1,
            ha="right", va="bottom", bbox=WHITE_BOX)
ax_bot.text(d_target - 0.005, 0.01 + 0.0008, f"目标：({d_target:.4f}, 0.01)", color=BLUE, fontsize=FS_SMALL - 1,
            ha="right", va="bottom", bbox=WHITE_BOX)
ax_bot.text(SHIFT - 0.006, kl_a + 0.0015, f"({num(SHIFT)}, {kl_a:.3g})", color=ORANGE, fontsize=FS_SMALL - 1,
            ha="right", va="bottom", bbox=WHITE_BOX)
ax_bot.text(d_max * 0.985, DESIRED_KL / 4, "不到 0.005：学习率 × 1.5", color=GREEN, fontsize=FS_SMALL - 1, ha="right", va="center")
ax_bot.text(0.006, (DESIRED_KL * 2 + kl_top) / 2, "超过 0.02：学习率 ÷ 1.5", color=ORANGE, fontsize=FS_SMALL - 1, va="center")
ax_bot.text(d_max * 0.985, 0.0125, "0.005 ~ 0.02：不动", color=MUTED, fontsize=FS_SMALL - 1, ha="right", va="center")
ax_bot.set_xlim(0, d_max)
ax_bot.set_ylim(0, kl_top)
ax_bot.set_xticks([0, SHIFT / 2, d_target, SHIFT] + ([0.3] if d_max >= 0.3 and SHIFT < 0.29 else []))
ax_bot.set_xticklabels(["0", num(SHIFT / 2), f"{d_target:.2f}", num(SHIFT)] + (["0.3"] if d_max >= 0.3 and SHIFT < 0.29 else []))
ax_bot.set_yticks([0, 0.005, 0.01, 0.02] + ([0.04] if kl_top > 0.04 else []))
ax_bot.set_yticklabels(["0", "0.005", "0.01", "0.02"] + (["0.04"] if kl_top > 0.04 else []))
panel_title(fig, [ax_bot], "② KL = 挪动² ÷ 2：挪一倍远，KL 变四倍")
panel_note(fig, [ax_bot], "每次更新前量一次（14 个关节的 KL 相加，再对整批取平均），落在哪一段就照那一段调学习率。")
fig.suptitle("KL：中心挪远一倍，KL 变四倍；超过 0.02 就把学习率调小", fontsize=23, fontweight="bold", color=INK, y=0.985)
if np.isfinite(kl_a):
    savefig(fig, "ch13_kl_shift")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("6. 数一数：一次迭代 98,304 条记录，4 份 × 5 遍 = 20 次更新")
N_ENVS, N_STEPS, N_MB, N_EPOCHS = 4096, 24, 4, 5
batch = N_ENVS * N_STEPS
mb = batch // N_MB
n_updates = N_MB * N_EPOCHS
table(["", "数"], [["4096 只 × 24 步", f"{batch:,}"], ["切 4 份，每份", f"{mb:,}"], ["过 5 遍，更新次数", n_updates]])
check("4096 × 24 = 98,304；÷ 4 = 24,576；4 × 5 = 20（第 5 章 📍、第 7 章 7.6 节）",
      batch == 98_304 and mb == 24_576 and n_updates == 20)
# 每次更新都用当前网络，对旧动作 0.5 重新求高度（13.2 节那两口钟）
check("重算的是“新钟在旧动作 0.5 处的高度”：比率还是 13.2 节的 1.0833", round(bell(A_REC, MU_NEW, SIG) / bell(A_REC, MU_OLD, SIG), 4) == 1.0833)
check("自测：切 8 份、过 10 遍：98,304 ÷ 8 = 12,288 条一份，8 × 10 = 80 次更新", batch // 8 == 12_288 and 8 * 10 == 80)

# ---------------------------------------------------------------------------
banner("6b. 画图：figures/ch13_overview.png（13.0 节的总览）")
check("总览图 ③：比率 1.0833 在范围里，目标 1.0833 × 2 = 2.1666（字面值重算）",
      round(rho * 2, 4) == 2.1666 and round(1.0833 * 2, 4) == 2.1666)
fig, axes = lesson_figure(4, "PPO：旧数据接着用，但每一次更新都看住“变了多少”", panel_height=3.15, width=10.8)
ax = axes[0]
lesson_panel(ax, "① 旧策略采数据：每条记录存下动作、当时的 ln π、优势", xmax=10.8, ymax=4.0)
for i, (lab, val) in enumerate([("动作", "0.5"), ("当时的 ln π", m(ln_old)), ("优势 Â", "+2")]):
    ax.text(0.5 + i * 3.4, 2.75, lab, fontsize=FS_SMALL, color=MUTED, va="center")
    cell(ax, 0.5 + i * 3.4, 1.55, val, width=2.6, height=0.8)
note(ax, 0.3, 0.55, "这些数只当数据存下来；之后 20 次更新都拿它们来比。")
ax = axes[1]
lesson_panel(ax, "② 更新时，用现在的网络对同一个动作重算 ln π", xmax=10.8, ymax=4.0)
hand(ax, 0.3, 2.45, f"现在的 ln π = {m(ln_new)}")
hand(ax, 0.3, 1.5, f"比率 = exp({m(ln_new)} − ({m(ln_old)})) = exp(0.08) = {rho:.4f}")
note(ax, 0.3, 0.55, "比率 > 1：现在的网络比采数据时更爱这个动作。")
ax = axes[2]
lesson_panel(ax, "③ 裁剪：比率走出 0.8 ~ 1.2，有利的那一边不再加分", xmax=10.8, ymax=4.0)
hand(ax, 0.3, 2.45, f"比率 {rho:.4f} 在范围里：目标 = {rho:.4f} × 2 = {rho * 2:.4f}")
hand(ax, 0.3, 1.5, "比率要是走到 1.5：min(1.5 × 2, 1.2 × 2) = min(3, 2.4) = 2.4", color=ORANGE)
note(ax, 0.3, 0.55, "好处到 1.2 封顶，坏处照算。")
ax = axes[3]
lesson_panel(ax, "④ 三项加成损失，KL 看住整体，同一批数据更新 20 次", xmax=10.8, ymax=4.7)
hand(ax, 0.3, 3.0, "损失 = −裁剪目标 + 1.0 × 价值损失 − 0.01 × 熵", fontsize=FS_SMALL + 1)
hand(ax, 0.3, 2.2, "每次更新前量 KL：超过 0.02，学习率 ÷ 1.5；不到 0.005，× 1.5", color=GREEN, fontsize=FS_SMALL + 1)
hand(ax, 0.3, 1.4, f"{batch:,} 条切 4 份，过 5 遍 = {n_updates} 次更新；然后扔掉，重新采", color=ORANGE, fontsize=FS_SMALL + 1)
note(ax, 0.3, 0.55, "扔掉的是记录；学到的东西留在旋钮里。")
savefig(fig, "ch13_overview")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("7. 映射到项目：正文引用的配置和源码行还在不在；拿 update() 的原行算本章的小例子")
REPO = Path(__file__).resolve().parents[3]
cfg_path = REPO / "src" / "mjlab_microduck" / "tasks" / "microduck_velocity_env_cfg.py"
CFG_LINES = ["value_loss_coef=1.0,", "use_clipped_value_loss=True,", "clip_param=0.2,", "entropy_coef=0.01,",
             "num_learning_epochs=5,", "num_mini_batches=4,", "learning_rate=1.0e-3,", 'schedule="adaptive",',
             "desired_kl=0.01,", "max_grad_norm=1.0,"]
if cfg_path.is_file():
    cfg_text = cfg_path.read_text(encoding="utf-8")
    at = cfg_text.find("MicroduckRlCfg = RslRlOnPolicyRunnerCfg(")
    print("microduck_velocity_env_cfg.py 的 MicroduckRlCfg：", "  ".join(CFG_LINES))
    check("项目配置：价值系数 1.0、价值裁剪开、裁剪 0.2、熵系数 0.01、5 遍、4 份、学习率 0.001、自适应、目标 KL 0.01",
          at >= 0 and lines_in_order(cfg_text[at:], CFG_LINES))
    check("velocity 任务没开对称（ENABLE_SYMMETRY = False）：update() 里对称那一段不执行", "\nENABLE_SYMMETRY = False\n" in cfg_text)
else:
    print("  （没找到项目的 env cfg，跳过这一项）")

PPO_LINES = [
    "generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)",
    "for batch in generator:",
    "actions_log_prob = self.actor.get_output_log_prob(batch.actions)",
    "values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])",
    "entropy = self.actor.output_entropy[:original_batch_size]",
    'if self.desired_kl is not None and self.schedule == "adaptive":',
    "with torch.inference_mode():",
    "kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)",
    "kl_mean = torch.mean(kl)",
    "if self.gpu_global_rank == 0:",
    "if kl_mean > self.desired_kl * 2.0:",
    "self.learning_rate = max(1e-5, self.learning_rate / 1.5)",
    "elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:",
    "self.learning_rate = min(1e-2, self.learning_rate * 1.5)",
    "for param_group in self.optimizer.param_groups:",
    'param_group["lr"] = self.learning_rate',
    "ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))",
    "surrogate = -torch.squeeze(batch.advantages) * ratio",
    "surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(",
    "ratio, 1.0 - self.clip_param, 1.0 + self.clip_param",
    "surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()",
    "if self.use_clipped_value_loss:",
    "value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)",
    "value_losses = (values - batch.returns).pow(2)",
    "value_losses_clipped = (value_clipped - batch.returns).pow(2)",
    "value_loss = torch.max(value_losses, value_losses_clipped).mean()",
    "loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()",
    "self.optimizer.zero_grad()",
    "loss.backward()",
    "nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)",
    "nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)",
    "self.optimizer.step()",
]
rsl_spec = importlib.util.find_spec("rsl_rl")
rsl_dir = Path(rsl_spec.origin).parent if rsl_spec and rsl_spec.origin else None
ppo_path = rsl_dir / "algorithms" / "ppo.py" if rsl_dir else None
if ppo_path and ppo_path.is_file():
    ppo_text = ppo_path.read_text(encoding="utf-8")
    upd = ppo_text.find("    def update(self)")
    print(f"rsl_rl 版本：{importlib.metadata.version('rsl-rl-lib')}")
    check("装的是 rsl_rl 5.0.1", importlib.metadata.version("rsl-rl-lib") == "5.0.1")
    check(f"正文映射块引用的 {len(PPO_LINES)} 行都在 update() 里原样存在，且顺序一致",
          upd >= 0 and lines_in_order(ppo_text[upd:], PPO_LINES))
    check("rsl_rl 的默认值也是 5 遍、4 份、裁剪 0.2", "num_learning_epochs: int = 5," in ppo_text
          and "num_mini_batches: int = 4," in ppo_text and "clip_param: float = 0.2," in ppo_text)

    # 把 update() 里“Surrogate loss”到“Symmetry loss”之间的原行拿出来，喂本章的小例子
    start = ppo_text.find("            # Surrogate loss", upd)
    stop = ppo_text.find("            # Symmetry loss", start)
    real_block = textwrap.dedent(ppo_text[start:stop])
    this = SimpleNamespace(clip_param=0.2, use_clipped_value_loss=True, value_loss_coef=1.0, entropy_coef=0.01)

    def run_real(ratios, advs, v_old, v_new, ret, ent):
        old_lp = torch.full((len(ratios), 1), ln_old)
        ns = {"torch": torch, "self": this,
              "batch": SimpleNamespace(old_actions_log_prob=old_lp, advantages=torch.tensor(advs).reshape(-1, 1),
                                       values=torch.tensor(v_old).reshape(-1, 1), returns=torch.tensor(ret).reshape(-1, 1)),
              "actions_log_prob": old_lp.squeeze(1) + torch.log(torch.tensor(ratios)),
              "values": torch.tensor(v_new).reshape(-1, 1), "entropy": torch.tensor(ent)}
        exec(real_block, ns)
        return ns

    ns4 = run_real([1.5, 0.5, 0.5, 1.5], [2.0, 2.0, -2.0, -2.0], [0.2] * 4, [0.2] * 4, [0.5] * 4, [19.865] * 4)
    ns3 = run_real([1.0] * 3, [0.0] * 3, [0.2] * 3, V_CASES, [0.5] * 3, [19.865] * 3)
    ns1 = run_real([1.5], [2.0], [0.2], [0.55], [0.5], [19.865])
    print(f"源码原行算四格：surrogate_loss = {ns4['surrogate_loss'].item():.4f}；"
          f"算三种 V：{[round(x, 4) for x in torch.max(ns3['value_losses'], ns3['value_losses_clipped']).squeeze(1).tolist()]}；"
          f"算一条记录的总账：loss = {ns1['loss'].item():.5f}")
    check("源码原行：四格的 surrogate_loss = 0.3，和 13.4 节手算一样", math.isclose(ns4["surrogate_loss"].item(), 0.3))
    check("源码原行：三种 V 的价值损失 0.01、0.0225、0.36，和 13.4 节手算一样",
          torch.allclose(torch.max(ns3["value_losses"], ns3["value_losses_clipped"]).squeeze(1), torch.tensor([0.01, 0.0225, 0.36])))
    check("源码原行：一条记录的总账 −2.58865", math.isclose(ns1["loss"].item(), -2.58865))

    k0 = ppo_text.find("if kl_mean > self.desired_kl * 2.0:", upd)
    kl_block = textwrap.dedent("\n".join(ppo_text[ppo_text.rfind("\n", 0, k0) + 1:].splitlines()[:4]))
    lr_state = SimpleNamespace(desired_kl=0.01, learning_rate=1e-3)
    lr_real = []
    for kl in KL_SEQ:
        exec(kl_block, {"self": lr_state, "kl_mean": kl})
        lr_real.append(lr_state.learning_rate)
    check("源码原行：六次 KL 调出来的学习率，和 13.5 节的表一样", lr_real == lr_seq)

    dist_path = rsl_dir / "modules" / "distribution.py"
    dist_text = dist_path.read_text(encoding="utf-8") if dist_path.is_file() else ""
    check("rsl_rl/modules/distribution.py：KL(旧‖新)，14 个关节相加",
          "return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)" in dist_text)
    stor_path = rsl_dir / "storage" / "rollout_storage.py"
    stor_text = stor_path.read_text(encoding="utf-8") if stor_path.is_file() else ""
    check("rsl_rl/storage/rollout_storage.py：每份 = 总条数 // 份数（98,304 // 4 = 24,576）",
          "mini_batch_size = batch_size // num_mini_batches" in stor_text)
    log_path = rsl_dir / "utils" / "logger.py"
    log_text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    check("rsl_rl/utils/logger.py：训练日志印“Mean surrogate loss:”，wandb 里叫 Loss/surrogate",
          'f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\\n"""' in log_text and 'f"Loss/{key}"' in log_text
          and '"surrogate": mean_surrogate_loss,' in ppo_text)
else:
    print("  （当前 Python 环境里没有 rsl_rl，跳过源码核对；用 uv run 运行就会核对）")

done()
