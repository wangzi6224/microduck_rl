"""第 13 章实验：PPO。数据过期、比率（重要性采样）、裁剪目标的四格与平坦区、三项损失（含价值裁剪）、
KL 散度与自适应学习率、一次迭代的记录数与更新次数——全部用小数字手算，再用 numpy / torch 核对。

运行：uv run python docs/learn-zh/labs/ch13_ppo_clip.py
纯 CPU，numpy + torch + matplotlib。第 5、6 节借 rsl_rl 自己的网络类和发牌函数（mini_batch_generator）做演示；
第 7 节读项目配置和 rsl_rl 的几份源码，并把 update() 里那几行原样拿来跑本章的小例子；不加载机器人、不训练。
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
from _draw import (BLUE, CELL, CELL_HOT, FAINT, FS_SMALL, FS_STEP, FS_TITLE, GREEN, INK, MUTED, ORANGE,
                   WHITE_BOX, arrow, cell, data_axes, hand, lesson_figure, lesson_panel, note, panel_note,
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
check("J(p) 的来历：3 × p + 1 × (1 − p) = 3p + 1 − p = 1 + 2p（几个 p 都核对）；斜率 2、学习率 0.1：p ← 0.5 + 0.1 × 2 = 0.7",
      all(math.isclose(3 * p + 1 * (1 - p), 1 + 2 * p) and math.isclose(3 * p + 1 - p, 1 + 2 * p) and math.isclose(J(p), 1 + 2 * p)
          for p in (0.0, 0.3, 0.5, 0.7, 1.0))
      and math.isclose((J(0.51) - J(0.49)) / 0.02, 2.0) and math.isclose(0.5 + 0.1 * 2, 0.7))

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
check("问卷：除以 1/2 就是乘 2（0.25 × 2 = 0.5）；100 份里各 50 份：50 × 0.5 = 25 份，50 × 1.5 = 75 份，合 100 份，地铁族占 25 ÷ 100 = 1/4",
      0.25 * 2 == 0.5 == (1 / 4) / (1 / 2) and 50 * 0.5 == 25 and 50 * 1.5 == 75 and 25 + 75 == 100 and 25 / 100 == 1 / 4)
check("比率：蓝 0.7 ÷ 0.5 = 1.4，绿 0.3 ÷ 0.5 = 0.6", math.isclose(w_blue, 1.4) and math.isclose(w_green, 0.6))
check("加权后 (5 × 1.4 × 3 + 5 × 0.6 × 1) ÷ 10 = 2.4 = J(0.7)（字面值重算）",
      math.isclose(avg_weighted, 2.4) and math.isclose((5 * 1.4 * 3 + 5 * 0.6 * 1) / 10, 2.4))
check("分子两块：5 × 1.4 × 3 = 7 × 3 = 21，5 × 0.6 × 1 = 3 × 1 = 3，(21 + 3) ÷ 10 = 2.4（字面值重算）",
      math.isclose(5 * 1.4 * 3, 21) and math.isclose(5 * 1.4, 7) and math.isclose(5 * 0.6 * 1, 3) and math.isclose(5 * 0.6, 3)
      and (21 + 3) / 10 == 2.4)

# 为什么乘上分量就对：三步（13.2 节）
units_blue, units_green = 5 * w_blue, 5 * w_green                 # 第一步：数份数
print(f"份数：蓝 5 × {num(w_blue)} = {num(units_blue)} 份，绿 5 × {num(w_green)} = {num(units_green)} 份，"
      f"合 {num(units_blue + units_green)} 份；蓝占 {num(units_blue / 10)}，绿占 {num(units_green / 10)}")
check("第一步：7 份 + 3 份 = 10 份；(7 × 3 + 3 × 1) ÷ 10 = 0.7 × 3 + 0.3 × 1 = 2.1 + 0.3 = 2.4（字面值重算）",
      math.isclose(units_blue, 7) and math.isclose(units_green, 3) and math.isclose(units_blue + units_green, 10)
      and (7 * 3 + 3 * 1) / 10 == 2.4 and math.isclose(0.7 * 3 + 0.3 * 1, 2.4) and math.isclose(2.1 + 0.3, 2.4))
check(f"理论：乘过分量，份数总和还是 10，蓝局占的比例变成新概率 {P_NEW:g}（换成别的 p 也一样）",
      math.isclose(units_blue + units_green, 10) and math.isclose(units_blue / 10, P_NEW))
table(["键", "在旧记录里占多少（旧概率）", "分量（新 ÷ 旧）", "占多少 × 分量", "新概率"],
      [["蓝", num(P_OLD), num(w_blue), num(P_OLD * w_blue), num(P_NEW)],
       ["绿", num(1 - P_OLD), num(w_green), num((1 - P_OLD) * w_green), num(1 - P_NEW)]])
check(f"第二步（理论）：旧概率 × (新 ÷ 旧) = 新概率——蓝 {num(P_OLD)} × {num(w_blue)} = {num(P_NEW)}，绿 {num(1 - P_OLD)} × {num(w_green)} = {num(1 - P_NEW)}",
      math.isclose(P_OLD * w_blue, P_NEW) and math.isclose((1 - P_OLD) * w_green, 1 - P_NEW))
check("第二步的字面值：0.5 × (0.7 ÷ 0.5) = 0.7，0.5 × (0.3 ÷ 0.5) = 0.3；0.5 × 1.4 = 0.7，0.5 × 0.6 = 0.3；10 ÷ 0.5 × 0.5 = 10",
      math.isclose(0.5 * (0.7 / 0.5), 0.7) and math.isclose(0.5 * (0.3 / 0.5), 0.3) and math.isclose(0.5 * 1.4, 0.7)
      and math.isclose(0.5 * 0.6, 0.3) and 10 / 0.5 * 0.5 == 10)
w_many = np.where(blue, w_blue, w_green)
avg_many_w = float((w_many * np.where(blue, SCORE_BLUE, SCORE_GREEN)).mean())
print(f"10 万局旧记录加权平均 = {avg_many_w:.4f}（新策略的真值 {J(P_NEW):g}）；这 10 万局里蓝局占 {float(blue.mean()):.4f}，不恰好是 0.5")
check("10 万局旧记录加权平均 2.4035（固定种子）", round(avg_many_w, 4) == 2.4035)
check(f"理论：加权以后，10 万局旧记录的平均落在新策略的 J = {J(P_NEW):g} 附近", abs(avg_many_w - J(P_NEW)) < 0.03)
check("第三步：随机的记录里蓝局不恰好占一半，所以只是约等于；2.4035 − 2.4 = 0.0035（字面值重算）",
      float(blue.mean()) != 0.5 and round(2.4035 - 2.4, 4) == 0.0035)

# 写成式子：四步，每一步游戏机代入都是 2.4（13.2 节）
F_OF = {"蓝": SCORE_BLUE, "绿": SCORE_GREEN}
PI_NEW, PI_OLD = {"蓝": P_NEW, "绿": 1 - P_NEW}, {"蓝": P_OLD, "绿": 1 - P_OLD}
step1 = sum(PI_NEW[k] * F_OF[k] for k in F_OF)                                  # ① 定义：按新概率加权
step2 = sum(PI_OLD[k] * (PI_NEW[k] / PI_OLD[k]) * F_OF[k] for k in F_OF)       # ② 塞进一个“旧 ÷ 旧”
step3 = sum(PI_OLD[k] * ((PI_NEW[k] / PI_OLD[k]) * F_OF[k]) for k in F_OF)     # ③ 把 ρ × f 括成一块
step4 = float(np.mean([(PI_NEW[k] / PI_OLD[k]) * F_OF[k] for k in records]))  # ④ 按旧策略抽的记录，每条 ρ × f 再平均
print(f"四步：① {step1:.4g}  ② {step2:.4g}  ③ {step3:.4g}  ④ {step4:.4g}")
check(f"理论：四步每一步都等于新策略的平均分 {J(P_NEW):g}",
      all(math.isclose(s, J(P_NEW)) for s in (step1, step2, step3, step4)))
check("四步的字面值：0.7 × 3 + 0.3 × 1 = 0.5 × 1.4 × 3 + 0.5 × 0.6 × 1 = 0.5 × 4.2 + 0.5 × 0.6 = 2.1 + 0.3 = 2.4；"
      "1.4 × 3 = 4.2，0.6 × 1 = 0.6，(4.2 + 0.6) ÷ 2 = 2.4；2 × 3 × 4 先乘哪两个都是 24",
      math.isclose(0.7 * 3 + 0.3 * 1, 2.4) and math.isclose(0.5 * 1.4 * 3 + 0.5 * 0.6 * 1, 2.4)
      and math.isclose(0.5 * 4.2 + 0.5 * 0.6, 2.4) and math.isclose(1.4 * 3, 4.2) and 0.6 * 1 == 0.6
      and math.isclose((4.2 + 0.6) / 2, 2.4) and (2 * 3) * 4 == 2 * (3 * 4) == 24)
rng_is = np.random.default_rng(3)                                  # 不只游戏机：随便编 5 种动作的新旧概率和 f
p_o5 = rng_is.random(5); p_o5 /= p_o5.sum()
p_n5 = rng_is.random(5); p_n5 /= p_n5.sum()
f5 = rng_is.normal(size=5)
check("一般成立：随便编 5 种动作的新旧概率和 f，Σ π_新 × f = Σ π_旧 × (π_新 ÷ π_旧) × f",
      math.isclose(float((p_n5 * f5).sum()), float((p_o5 * (p_n5 / p_o5) * f5).sum())))
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
      [["动作 0.5 处的高度", f"{h_old:.5f}", f"{h_new:.5f}"], ["ln 高度", f"{ln_old:.4f}", f"{ln_new:.4f}"],
       ["torch 的 log_prob", f"{lp_old_t:.4f}", f"{lp_new_t:.4f}"]])
print(f"直接除：{h_new:.5f} ÷ {h_old:.5f} = {rho:.4f}")
print(f"取 ln：{ln_new:.4f} − ({ln_old:.4f}) = {ln_new - ln_old:.4f}；exp({ln_new - ln_old:.2f}) = {math.exp(ln_new - ln_old):.4f}")
check("字面值重算：exp(−0.125) ÷ √(2π) = 0.35207（正文让读者在控制台敲的那一行）",
      round(math.exp(-0.125) / math.sqrt(2 * math.pi), 5) == 0.35207)
check("字面值重算：(1 + 0.1²) ÷ 2 = 0.505，减 0.5 得 0.005",
      round((1 + 0.1 ** 2) / 2, 3) == 0.505 and round(0.505 - 0.5, 3) == 0.005)
check("两个高度 0.35207、0.38139；比率 0.38139 ÷ 0.35207 = 1.0833（字面值重算）",
      round(h_old, 5) == 0.35207 and round(h_new, 5) == 0.38139 and round(rho, 4) == 1.0833
      and round(0.38139 / 0.35207, 4) == 1.0833)
check("两个 ln 高度 −1.0439、−0.9639（正文和 13.0 的总览图印的就是这两个数；字面值重算：ln 0.35207、ln 0.38139）",
      round(ln_old, 4) == -1.0439 and round(ln_new, 4) == -0.9639
      and round(math.log(0.35207), 4) == -1.0439 and round(math.log(0.38139), 4) == -0.9639)
check("ln 之差 = (0.5² − 0.3²) ÷ 2 = (0.25 − 0.09) ÷ 2 = 0.08：钟高公式里的 ln(σ√(2π)) 相减时抵消了",
      math.isclose(ln_new - ln_old, 0.08) and math.isclose((0.25 - 0.09) / 2, 0.08)
      and round(-0.9639 - (-1.0439), 4) == 0.08)
check("两条路同一个数：exp(0.08) = 1.0833 = 高度之比；torch 的 log_prob 也给出同样的差",
      round(math.exp(0.08), 4) == 1.0833 and math.isclose(math.exp(lp_new_t - lp_old_t), rho))
print(f"一步一步：√(2π) = {math.sqrt(2 * math.pi):.4f}；旧钟 exp(−0.125) = {math.exp(-0.125):.4f}，÷ 2.5066 = {math.exp(-0.125) / 2.5066:.5f}；"
      f"新钟 exp(−0.045) = {math.exp(-0.045):.4f}，÷ 2.5066 = {math.exp(-0.045) / 2.5066:.5f}")
check("钟的高度一步一步（字面值重算）：√(2π) → 2.5066；0.25 ÷ 2 = 0.125，exp(−0.125) → 0.8825，0.8825 ÷ 2.5066 → 0.35207；"
      "0.09 ÷ 2 = 0.045，exp(−0.045) → 0.9560，0.9560 ÷ 2.5066 → 0.38139",
      round(math.sqrt(2 * math.pi), 4) == 2.5066 and 0.25 / 2 == 0.125 and round(math.exp(-0.125), 4) == 0.8825
      and round(0.8825 / 2.5066, 5) == 0.35207 and math.isclose(0.09 / 2, 0.045) and round(math.exp(-0.045), 4) == 0.956
      and round(0.9560 / 2.5066, 5) == 0.38139 and math.isclose((0.5 - 0.2) ** 2, 0.09))
xs_strip = A_REC - 0.005 + (np.arange(100) + 0.5) * 0.0001          # 动作 0.5 附近宽 0.01 的一小段，再切 100 条细条
p_strip_new = float(np.sum(bell(xs_strip, MU_NEW, SIG)) * 0.0001)
p_strip_old = float(np.sum(bell(xs_strip, MU_OLD, SIG)) * 0.0001)
print(f"宽 0.01 的一小段：新钟里的概率 {p_strip_new:.7f}，旧钟里 {p_strip_old:.7f}，相除 {p_strip_new / p_strip_old:.4f}")
check("宽度约掉：0.38139 × 0.01 = 0.0038139，0.35207 × 0.01 = 0.0035207，0.0038139 ÷ 0.0035207 → 1.0833（字面值重算）",
      math.isclose(0.38139 * 0.01, 0.0038139) and math.isclose(0.35207 * 0.01, 0.0035207) and round(0.0038139 / 0.0035207, 4) == 1.0833)
check("理论：把那一小段真的切成细条加面积，两个概率相除也是 1.0833（和高度之比相差不到 0.0001）",
      abs(p_strip_new / p_strip_old - rho) < 1e-4 and round(p_strip_new / p_strip_old, 4) == 1.0833)
check("减去负数就是加：−0.9639 − (−1.0439) = −0.9639 + 1.0439 = 0.08；数相除、指数相减：exp(−0.045) ÷ exp(−0.125) = exp(−0.045 − (−0.125)) = exp(0.08)",
      round(-0.9639 - (-1.0439), 4) == round(-0.9639 + 1.0439, 4) == 0.08
      and math.isclose(math.exp(-0.045) / math.exp(-0.125), math.exp(0.08)) and math.isclose(-0.045 - (-0.125), 0.08))


def fast_ln_ratio(a, mu_old, mu_new, sigma):
    """13.2 节的快算法：两口钟一样宽时，ln 比率 = [(a − 旧中心)² − (a − 新中心)²] ÷ (2σ²)。"""
    return ((a - mu_old) ** 2 - (a - mu_new) ** 2) / (2 * sigma ** 2)


check("理论：快算法和“两个 ln 高度直接相减”处处相同（换几个动作、中心、σ 都核对）",
      all(math.isclose(fast_ln_ratio(a, mo, mn, s), math.log(bell(a, mn, s)) - math.log(bell(a, mo, s)), abs_tol=1e-12)
          for a in (-1.3, 0.0, 0.5, 2.0) for mo, mn in ((0.0, 0.2), (0.4, -0.1)) for s in (0.5, 1.0, 1.7))
      and math.isclose(fast_ln_ratio(0.5, 0.0, 0.2, 1.0), 0.08))
check("配对相减：(5 + 7) − (2 + 3) = (5 − 2) + (7 − 3) = 7；数相乘、指数相加：exp(0.0953 + (−0.1054)) = exp(0.0953) × exp(−0.1054)",
      (5 + 7) - (2 + 3) == (5 - 2) + (7 - 3) == 7 and math.isclose(math.exp(0.0953 + -0.1054), math.exp(0.0953) * math.exp(-0.1054)))
check("exp 在 0 处的斜率是 1：exp(0.001) → 1.0010005，ln π 多 0.001，比率也几乎正好多 0.001",
      round(math.exp(0.001), 7) == 1.0010005 and abs((math.exp(0.001) - 1) / 0.001 - 1) < 0.001)

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
adv5 = torch.tensor([1.0, -0.5, 2.0, 0.3, -1.2])                     # 编的 5 个优势
lp_now = Normal(net(obs), 1.0).log_prob(act).sum(dim=-1)
g_ratio = torch.autograd.grad((-adv5 * torch.exp(lp_now - lp_now.detach())).mean(), net.weight)[0]
lp_now = Normal(net(obs), 1.0).log_prob(act).sum(dim=-1)
g_reinforce = torch.autograd.grad((-adv5 * lp_now).mean(), net.weight)[0]
check("比率恰好是 1 时，“−Â × 比率”对旋钮的梯度和 REINFORCE 的“−Â × ln π”一模一样（exp 在 0 处斜率是 1）",
      torch.allclose(g_ratio, g_reinforce))
print(f"2 个关节的迷你版：ln 1.1 = {math.log(1.1):.4f}，ln 0.9 = {math.log(0.9):.4f}；相加 {math.log(1.1) + math.log(0.9):.4f}，"
      f"exp 回去 {math.exp(math.log(1.1) + math.log(0.9)):.2f} = 1.1 × 0.9")
check("迷你版：ln 1.1 = 0.0953、ln 0.9 = −0.1054，相加 −0.0101，exp(−0.0101) = 0.99 = 1.1 × 0.9（字面值重算）",
      round(math.log(1.1), 4) == 0.0953 and round(math.log(0.9), 4) == -0.1054
      and round(0.0953 + (-0.1054), 4) == -0.0101 and round(math.exp(-0.0101), 2) == 0.99
      and round(1.1 * 0.9, 2) == 0.99)
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
hand(ax2, 0.3, 2.0, f"取 ln：{m(ln_new, '.4f')} − ({m(ln_old, '.4f')}) = {ln_new - ln_old:.2f}", color=GREEN)
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
h_peak = bell(A_REC, A_REC, 1.0)                                  # 第一步：中心挪到 0.5，σ 还是 1
print(f"分两步：中心挪到 0.5 → 高 {h_peak:.4f}，比率 {h_peak / h_old:.3f}；再收窄到 σ = 0.1 → 高 {h_narrow:.3f}，比率 {rho_narrow:.2f}")
check("没有刹车的两步（字面值重算）：1 ÷ 2.5066 → 0.3989，0.3989 ÷ 0.35207 → 1.133；1 ÷ (0.1 × 2.5066) → 3.989；没更新时 1 × 2 = 2，22.66 ÷ 2 = 11.33",
      round(h_peak, 4) == 0.3989 and round(1 / 2.5066, 4) == 0.3989 and round(0.3989 / 0.35207, 3) == 1.133
      and round(h_peak / h_old, 3) == 1.133 and round(1 / (0.1 * 2.5066), 3) == 3.989 and 1 * 2 == 2 and round(22.66 / 2, 2) == 11.33)
check("σ = 0.1 的钟，95% 的动作落在 0.5 ± 2 × 0.1，即 0.3 到 0.7（第 4 章 4.6 节的 ±2σ）",
      round(math.erf(2 / math.sqrt(2)), 2) == 0.95 and math.isclose(A_REC - 2 * 0.1, 0.3) and math.isclose(A_REC + 2 * 0.1, 0.7))
check("限位的来历：0.8 = 1 − 0.2，1.2 = 1 + 0.2（clip_param = 0.2）", math.isclose(LO, 1 - 0.2) and math.isclose(HI, 1 + 0.2))


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
check("1.6 × 2 = 3.2，比 2.4 大，取不到它；min(3, 2.4) 就是 JS 的 Math.min(3, 2.4) = 2.4",
      math.isclose(1.6 * 2, 3.2) and 3.2 > 2.4 and min(3, 2.4) == 2.4)
nudge1 = clipped_objective(torch.tensor(1.51), torch.tensor(2.0)).item()
nudge2 = clipped_objective(torch.tensor(0.51), torch.tensor(2.0)).item()
print(f"ρ 挪 0.01：第 1 格 1.5 → 1.51，分 {num(nudge1)}（没变）；第 2 格 0.5 → 0.51，分 1 → {num(nudge2)}，多了 {num(nudge2 - 1)}")
check("挪 0.01 量斜率：第 1 格 min(3.02, 2.4) = 2.4，分没变，斜率 0；第 2 格 min(1.02, 1.6) = 1.02，多 0.02，0.02 ÷ 0.01 = 2",
      math.isclose(nudge1, 2.4) and math.isclose(1.51 * 2, 3.02) and math.isclose(nudge2, 1.02) and math.isclose(0.51 * 2, 1.02)
      and math.isclose((nudge2 - 1) / 0.01, 2.0) and math.isclose(0.02 / 0.01, 2))


def value_and_slope(fn, r, a):
    """某种做法下，一条记录的分和它对 ρ 的斜率（torch 自动求导）。"""
    rt = torch.tensor(float(r), requires_grad=True)
    v = fn(rt, torch.tensor(float(a)))
    v.backward()
    return v.item(), rt.grad.item()


THREE_WAYS = [("不裁剪：ρ × Â", lambda r, a: r * a),
              ("只用裁剪版", lambda r, a: torch.clamp(r, LO, HI) * a),
              ("两个里取小的（PPO）", clipped_objective)]
three = {(name, k): value_and_slope(fn, *k) for name, fn in THREE_WAYS for k in FOUR}
table(["格 (ρ, Â)"] + [name for name, _ in THREE_WAYS],
      [[f"{i + 1} ({k[0]:g}, {k[1]:+g})"] + [f"{num(three[(name, k)][0])}（斜率 {num(three[(name, k)][1])}）" for name, _ in THREE_WAYS]
       for i, k in enumerate(FOUR)])
check("三列对照的值：不裁剪 3、1、−1、−3；只用裁剪版 2.4、1.6、−1.6、−2.4；取小的 2.4、1、−1.6、−3",
      [round(three[("不裁剪：ρ × Â", k)][0], 4) for k in FOUR] == [3.0, 1.0, -1.0, -3.0]
      and [round(three[("只用裁剪版", k)][0], 4) for k in FOUR] == [2.4, 1.6, -1.6, -2.4]
      and [round(three[("两个里取小的（PPO）", k)][0], 4) for k in FOUR] == [2.4, 1.0, -1.6, -3.0])
check("三列对照的斜率：不裁剪 2、2、−2、−2；只用裁剪版四格全是 0（第 2、4 格也不纠正了）；取小的 0、2、0、−2",
      [three[("不裁剪：ρ × Â", k)][1] for k in FOUR] == [2.0, 2.0, -2.0, -2.0]
      and [three[("只用裁剪版", k)][1] for k in FOUR] == [0.0, 0.0, 0.0, 0.0]
      and [three[("两个里取小的（PPO）", k)][1] for k in FOUR] == [0.0, 2.0, 0.0, -2.0] and CLIP == 0.2)
check("min 选中谁：走过了界时裁剪版更小（2.4 < 3，−1.6 < −1），往不利方向走时不裁剪的更小（1 < 1.6，−3 < −2.4）",
      2.4 < 3 and -1.6 < -1 and 1 < 1.6 and -3 < -2.4)
check("JS 的 clip 怎么走：max(1.5, 0.8) = 1.5，min(1.5, 1.2) = 1.2；max(0.5, 0.8) = 0.8，min(0.8, 1.2) = 0.8",
      max(1.5, 0.8) == 1.5 and min(1.5, 1.2) == 1.2 and max(0.5, 0.8) == 0.8 and min(0.8, 1.2) == 0.8)
check("L^CLIP 由里往外，代入第 1 格：1.5 × 2 = 3；clip(1.5, 0.8, 1.2) = 1.2，1.2 × 2 = 2.4；min(3, 2.4) = 2.4",
      1.5 * 2 == 3 and min(max(1.5, 0.8), 1.2) == 1.2 and 1.2 * 2 == 2.4 and min(3, 2.4) == 2.4)
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
ln_hi, ln_lo = math.log(HI), math.log(LO)
tail_hi = 0.5 * (1 - math.erf(ln_hi / LOG_SPREAD / math.sqrt(2)))   # ln 之差那口钟，在 ln 1.2 以右的面积
tail_lo = 0.5 * (1 + math.erf(ln_lo / LOG_SPREAD / math.sqrt(2)))   # 在 ln 0.8 以左的面积
print(f"拆开：ln {num(HI)} = {ln_hi:.3f}，比率超过它的占 {tail_hi * 100:.1f}%，其中 Â 为正的一半 {tail_hi * 50:.1f}%；"
      f"ln {num(LO)} = {ln_lo:.3f}，低于它的占 {tail_lo * 100:.1f}%，其中 Â 为负的一半 {tail_lo * 50:.1f}%；合 {(tail_hi + tail_lo) * 50:.1f}%")
print(f"  这 {N_FLAT:,} 条假记录里实数：超过 {num(HI)} 的 {float((ratio_f > HI).mean()) * 100:.1f}%，低于 {num(LO)} 的 {float((ratio_f < LO).mean()) * 100:.1f}%")
check("拆开 9.0%（字面值重算）：ln 1.2 → 0.182，ln 0.8 → −0.223；两边的面积 11.2%、6.8%，各只算一半：5.6% + 3.4% = 9.0%",
      round(ln_hi, 3) == 0.182 and round(ln_lo, 3) == -0.223 and round(tail_hi * 100, 1) == 11.2 and round(tail_lo * 100, 1) == 6.8
      and round(tail_hi * 50, 1) == 5.6 and round(tail_lo * 50, 1) == 3.4 and round(5.6 + 3.4, 1) == 9.0
      and round(11.2 / 2, 1) == 5.6 and round(6.8 / 2, 1) == 3.4)
check(f"理论：平坦区比例 = 上边出界的一半 + 下边出界的一半，正是上面用面积算的 {theory * 100:.1f}%；假记录里两边出界的比例也和面积对得上（容差 1 个百分点）",
      math.isclose((tail_hi + tail_lo) / 2, theory) and abs(float((ratio_f > HI).mean()) - tail_hi) < 0.01
      and abs(float((ratio_f < LO).mean()) - tail_lo) < 0.01)

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
check("让 −L 变小就是让 L 变大：L 得 2.4 和 1，取负号 −2.4 和 −1，小的 −2.4 对应大的 2.4",
      -2.4 < -1 and 2.4 > 1 and min(-2.4, -1) == -max(2.4, 1))
check("四格平均从左往右（字面值重算）：2.4 + 1 = 3.4，3.4 − 1.6 = 1.8，1.8 − 3 = −1.2，−1.2 ÷ 4 = −0.3",
      math.isclose(2.4 + 1, 3.4) and math.isclose(3.4 - 1.6, 1.8) and math.isclose(1.8 - 3, -1.2) and math.isclose(-1.2 / 4, -0.3))
rng_mm = np.random.default_rng(5)
check("−min(u, v) = max(−u, −v)：第 1 格 −min(3, 2.4) = −2.4 = max(−3, −2.4)；第 4 格 −min(−3, −2.4) = 3 = max(3, 2.4)；再随便抽 1000 对数核对",
      -min(3, 2.4) == -2.4 == max(-3, -2.4) and -min(-3, -2.4) == 3 == max(3, 2.4)
      and all(-min(u, v) == max(-u, -v) for u, v in rng_mm.normal(size=(1000, 2)).tolist()))

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
# 平坦区只有 0.4 ~ 0.6 这一段：(V − 0.5)² 重新超过 0.01 之后，取大的又取回不裁剪版
rows = [[num(v), num(value_loss_parts(v)[3]), num(value_loss_parts(v)[4])] for v in (0.45, 0.55, 0.65, 0.7)]
table(["更新后的 V", "取大的", "对 V 的斜率"], rows)
print("平坦区只有 0.4 ~ 0.6 这一段：两端 (0.4 − 0.5)² = (0.6 − 0.5)² = 0.01，和封顶值一样大；再往右不裁剪的那个又更大了")
check("平坦区是 0.4 ~ 0.6：中间（0.45、0.55）斜率恰好 0；出了 0.6（0.65、0.7）又回到斜线，斜率 +0.3、+0.4",
      value_loss_parts(0.45)[4] == 0.0 and value_loss_parts(0.55)[4] == 0.0
      and math.isclose(value_loss_parts(0.65)[3], 0.0225) and math.isclose(value_loss_parts(0.65)[4], 0.3)
      and math.isclose(value_loss_parts(0.7)[4], 0.4) and math.isclose((0.4 - 0.5) ** 2, 0.01)
      and math.isclose((0.6 - 0.5) ** 2, 0.01))
check("自测：V = 0.7 → 不裁剪 (0.7 − 0.5)² = 0.04，裁剪版仍是 (0.4 − 0.5)² = 0.01，取大的 0.04，用的是不裁剪版（字面值重算）",
      math.isclose(value_loss_parts(0.7)[0], 0.04) and math.isclose(value_loss_parts(0.7)[2], 0.01)
      and math.isclose(value_loss_parts(0.7)[3], 0.04) and math.isclose((0.7 - 0.5) ** 2, 0.04))
check("式子代入（字面值重算）：V = 0.55，挪动 0.55 − 0.2 = 0.35，夹成 0.2，V夹 = 0.2 + 0.2 = 0.4；V = −0.1，挪动 −0.3，夹成 −0.2，V夹 = 0；"
      "max(0.0025, 0.01) = 0.01",
      math.isclose(0.55 - 0.2, 0.35) and min(max(0.35, -0.2), 0.2) == 0.2 and math.isclose(0.2 + 0.2, 0.4)
      and math.isclose(-0.1 - 0.2, -0.3) and min(max(-0.3, -0.2), 0.2) == -0.2 and math.isclose(0.2 + -0.2, 0.0)
      and math.isclose(vparts[-0.1][1], 0.0) and max(0.0025, 0.01) == 0.01 and math.isclose((0.55 - 0.5) ** 2, 0.0025))
check("斜率手算：(V − 0.5)² 的斜率 2 × (V − 0.5)；V = 0.35 → 2 × (−0.15) = −0.3，V = −0.1 → 2 × (−0.6) = −1.2（和自动求导一致）",
      math.isclose(2 * (0.35 - 0.5), -0.3) and math.isclose(2 * (-0.1 - 0.5), -1.2)
      and math.isclose(vparts[0.35][4], 2 * (0.35 - 0.5)) and math.isclose(vparts[-0.1][4], 2 * (-0.1 - 0.5)))

ent_one = Normal(0.0, 1.0).entropy().item()
ent14 = Normal(torch.zeros(14), torch.ones(14)).entropy().sum().item()
print(f"熵：一口 σ = 1 的钟 {ent_one:.5f}，14 口相加 {ent14:.3f}；乘 0.01 = {0.01 * ent14:.5f}（第 11 章 11.6 节）")
check("熵：14 × 1.41894 = 19.865（字面值重算），乘 0.01 是 0.19865",
      round(ent14, 3) == 19.865 and round(14 * 1.41894, 3) == 19.865 and round(0.01 * ent14, 5) == 0.19865)

VAL_COEF, ENT_COEF = 1.0, 0.01
loss_one = -four_val[(1.5, 2.0)] + VAL_COEF * vparts[0.55][3] - ENT_COEF * round(ent14, 3)
print(f"一条记录的总账：−{num(four_val[(1.5, 2.0)])} + 1.0 × {num(vparts[0.55][3])} − 0.01 × 19.865 = {loss_one:.5f}")
check("一条记录的总账：−2.4 + 0.01 − 0.19865 = −2.58865（字面值重算）",
      math.isclose(loss_one, -2.58865) and math.isclose(-2.4 + 0.01 - 0.19865, -2.58865))
check("从左往右（字面值重算）：−2.4 + 0.01 = −2.39，−2.39 − 0.19865 = −2.58865；熵的分量 0.01 是价值损失 1.0 的百分之一",
      math.isclose(-2.4 + 0.01, -2.39) and math.isclose(-2.39 - 0.19865, -2.58865) and math.isclose(ENT_COEF / VAL_COEF, 0.01))
# 三项加成一个数，梯度各走各的：一个 actor 旋钮（钟的中心 μ），一个 critic 旋钮（V）
mu_a = torch.tensor(0.0, requires_grad=True)
v_c = torch.tensor(0.35, requires_grad=True)                         # 选一个照常推的位置，梯度不是 0
log_std = torch.tensor(0.0)
lp_cur = Normal(mu_a, log_std.exp()).log_prob(torch.tensor(A_REC))
ratio_c = torch.exp(lp_cur - math.log(h_old))
term1 = -clipped_objective(ratio_c, torch.tensor(2.0))
term2 = torch.max((v_c - RET) ** 2, (V_OLD + (v_c - V_OLD).clamp(-0.2, 0.2) - RET) ** 2)
term3 = -ENT_COEF * Normal(mu_a, log_std.exp()).entropy()
g_total = torch.autograd.grad(term1 + VAL_COEF * term2 + term3, (mu_a, v_c), retain_graph=True)
g_actor_only = torch.autograd.grad(term1 + term3, mu_a, retain_graph=True)[0]
g_critic_only = torch.autograd.grad(term2, v_c)[0]
check("加法分开求导：总损失对 actor 旋钮的梯度 = 第一、三项的梯度；对 critic 旋钮的梯度 = 第二项的梯度",
      math.isclose(g_total[0].item(), g_actor_only.item()) and math.isclose(g_total[1].item(), VAL_COEF * g_critic_only.item()))

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
banner("4b. 画图：figures/ch13_value_clip.png（价值裁剪：夹的是 V 挪了多远，两个里取大的）")
V_LO, V_HI = V_OLD - 0.2, V_OLD + 0.2                                 # V 允许算进去的范围：旧 V ± 0.2


def v_curves(xs):
    """三条线：不裁剪、裁剪版、两个里取大的（真正进损失的那条）。"""
    un = (xs - RET) ** 2
    cl = (np.clip(xs, V_LO, V_HI) - RET) ** 2
    return un, cl, np.maximum(un, cl)


fig = plt.figure(figsize=(10, 15.0))
ax1 = fig.add_axes([0.14, 0.720, 0.81, 0.185])
ax2 = fig.add_axes([0.14, 0.440, 0.81, 0.185])
ax_cmp = fig.add_axes([0.03, 0.035, 0.94, 0.295])
fig.suptitle("价值裁剪：夹的是 V 挪了多远，两个里取大的", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.968)

xs1 = np.linspace(-0.2, 0.8, 500)
un1, cl1, mx1 = v_curves(xs1)
data_axes(ax1, "更新后的 V（critic 现在给这条记录打的分）", "这一条记录的价值损失")
ax1.axvspan(V_LO, V_HI, color=CELL, zorder=0)
ax1.text((V_LO + V_HI) / 2, 0.395, f"{num(V_LO)} ~ {num(V_HI)}（旧 V {num(V_OLD)} ± 0.2）", color=BLUE,
         fontsize=FS_SMALL - 2, ha="center", va="center")
ax1.plot(xs1, un1, color=FAINT, lw=2.4, ls="--")
ax1.plot(xs1, cl1, color=ORANGE, lw=2.4, ls=":")
ax1.plot(xs1, mx1, color=BLUE, lw=3.4)
ax1.axvline(RET, color=GREEN, lw=1.4, ls=":")
ax1.text(RET + 0.015, 0.335, "标签 0.5", color=GREEN, fontsize=FS_SMALL - 1, va="center")
ax1.scatter([-0.1], [vparts[-0.1][3]], s=110, color=BLUE, zorder=6)
ax1.scatter([-0.1], [vparts[-0.1][2]], s=80, color=ORANGE, zorder=5)
ax1.text(-0.065, 0.362, f"V = −0.1：取大的 = {num(vparts[-0.1][3])}", color=BLUE, fontsize=FS_SMALL - 1,
         ha="left", va="center", bbox=WHITE_BOX)
ax1.text(-0.065, 0.245, f"裁剪版只有 {num(vparts[-0.1][2])}，没取它", color=ORANGE, fontsize=FS_SMALL - 2,
         ha="left", va="center", bbox=WHITE_BOX)
ax1.text(0.215, 0.215, "灰虚线：不裁剪 (V − 0.5)²（多半被蓝线压着）", color=MUTED, fontsize=FS_SMALL - 2, va="center")
ax1.text(0.215, 0.172, "橙点线：裁剪版（V 先夹进 0 ~ 0.4 再算）", color=ORANGE, fontsize=FS_SMALL - 2, va="center")
ax1.text(0.215, 0.129, "蓝实线：两个里取大的 = 真正进损失的", color=BLUE, fontsize=FS_SMALL - 2, va="center")
ax1.set_xlim(-0.2, 0.8)
ax1.set_ylim(0, 0.42)
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8])
ax1.set_xticklabels([m(t, "g") for t in (-0.1, 0, 0.2, 0.4, 0.6, 0.8)])
ax1.set_yticks([0, 0.1, 0.25, 0.36])
ax1.set_yticklabels(["0", "0.1", "0.25", "0.36"])
panel_title(fig, [ax1], "① 全景：采数据时 V = 0.2，标签 0.5，更新后 V 挪到哪算哪")
panel_note(fig, [ax1], "V 往反方向跑到 −0.1：不裁剪的 0.36 比裁剪版的 0.25 大，取大的取到它，照常把 critic 拉回来。")

xs2 = np.linspace(0.28, 0.8, 500)
un2, cl2, mx2 = v_curves(xs2)
data_axes(ax2, "更新后的 V", "价值损失（放大看）")
ax2.axvspan(0.28, V_HI, color=CELL, zorder=0)
ax2.plot(xs2, un2, color=FAINT, lw=2.4, ls="--")
ax2.plot(xs2, cl2, color=ORANGE, lw=2.4, ls=":")
ax2.plot(xs2, mx2, color=BLUE, lw=3.4)
flat = (xs2 >= V_HI) & (xs2 <= 0.6)
ax2.plot(xs2[flat], mx2[flat], color=ORANGE, lw=7, solid_capstyle="butt", zorder=4)
for v_new, col in ((0.35, BLUE), (0.55, ORANGE), (0.65, BLUE)):
    ax2.scatter([v_new], [value_loss_parts(v_new)[3]], s=110, color=col, zorder=6)
ax2.scatter([0.55], [vparts[0.55][0]], s=80, color=FAINT, zorder=5)
arrow(ax2, (0.55, vparts[0.55][0]), (0.55, vparts[0.55][3] - 0.0006), ORANGE, lw=2.0)
ax2.text(0.362, 0.0315, f"V = 0.35：两个都是 {num(vparts[0.35][3])}", color=BLUE, fontsize=FS_SMALL - 1,
         ha="left", va="center", bbox=WHITE_BOX)
ax2.text(0.50, 0.0182, f"V = 0.55：max({num(vparts[0.55][0])}, {num(vparts[0.55][3])}) = {num(vparts[0.55][3])}",
         color=ORANGE, fontsize=FS_SMALL - 1, ha="center", va="center", bbox=WHITE_BOX)
ax2.text(0.662, 0.0305, f"V = 0.65：又是 {num(value_loss_parts(0.65)[3])}，斜率回来了",
         color=BLUE, fontsize=FS_SMALL - 1, ha="center", va="center", bbox=WHITE_BOX)
ax2.text(0.462, 0.0058, "平坦区：斜率 0", color=ORANGE, fontsize=FS_SMALL - 1, ha="center", va="center")
ax2.set_xlim(0.28, 0.8)
ax2.set_ylim(0, 0.042)
ax2.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
ax2.set_xticklabels([f"{t:g}" for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)])
ax2.set_yticks([0, 0.01, 0.0225, 0.04])
ax2.set_yticklabels(["0", "0.01", "0.0225", "0.04"])
panel_title(fig, [ax2], "② 放大右半边：平坦区只有 0.4 ~ 0.6 这一段")
panel_note(fig, [ax2], "橙色平台的高度不随 V 变，斜率 0：这条记录不再推 critic。\n"
                       "可 V 一过 0.6，灰虚线又比 0.01 高了，取大的重新取到它——平坦区是有头的。")

lesson_panel(ax_cmp, "③ 两道裁剪，夹的东西和取的方向都不同", xmax=10.4, ymax=6.2)
ax_cmp.text(3.55, 5.15, "13.3 比率裁剪（actor）", color=BLUE, fontsize=FS_SMALL, ha="center", va="center")
ax_cmp.text(7.55, 5.15, "13.4 价值裁剪（critic）", color=ORANGE, fontsize=FS_SMALL, ha="center", va="center")
CMP = [("夹的是", "比率 ρ 本身", "V 离“旧 V”有多远"),
       ("夹成", f"0.8 ~ 1.2（1 ± {num(CLIP)}）", f"{num(V_LO)} ~ {num(V_HI)}（旧 V {num(V_OLD)} ± 0.2）"),
       ("两个里取", "小的（目标要变大）", "大的（损失要变小）"),
       ("都是这一句", "挪过头的好处不算", "挪过头的好处不算")]
for j, (lab, a_, b_) in enumerate(CMP):
    y = 4.25 - j * 0.88
    ax_cmp.text(0.15, y, lab, color=MUTED, fontsize=FS_SMALL, va="center")
    cell(ax_cmp, 1.75, y - 0.30, a_, width=3.6, height=0.60, fontsize=FS_SMALL - 1, color=BLUE)
    cell(ax_cmp, 5.75, y - 0.30, b_, width=3.6, height=0.60, fontsize=FS_SMALL - 1, color=ORANGE,
         facecolor=CELL_HOT)
note(ax_cmp, 0.15, 0.45, "两边都是“往坏处想”：往有利的方向走过了头就不再给分，往不利的方向走照罚。\n"
                         "两处读的还是同一个设置 clip_param = 0.2，可单位一个是倍数、一个是价值本身。")
savefig(fig, "ch13_value_clip")
plt.close(fig)

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
# KL = 按旧钟取平均的“ln 旧 − ln 新”，也就是 13.2 节那个 ln 比率取相反数。先一对一对手算
pair_rows = []
for a_p in (0.5, 1.0):
    up, dn = fast_ln_ratio(a_p, 0.0, SHIFT, 1.0), fast_ln_ratio(-a_p, 0.0, SHIFT, 1.0)
    pair_rows.append([f"±{a_p:g}", f"{up:.4g}", f"{dn:.4g}", f"{(-up - dn) / 2:.4g}"])
table(["一对动作", "a 的 ln 比率", "−a 的 ln 比率", "取相反数再平均"], pair_rows)
check("一对一对手算（字面值重算）：a = 0.5 → (0.25 − 0.09) ÷ 2 = 0.08，a = −0.5 → (0.25 − 0.49) ÷ 2 = −0.12，(−0.08 + 0.12) ÷ 2 = 0.02；"
      "a = ±1 → (1 − 0.64) ÷ 2 = 0.18、(1 − 1.44) ÷ 2 = −0.22，(−0.18 + 0.22) ÷ 2 = 0.02",
      math.isclose(fast_ln_ratio(0.5, 0.0, 0.2, 1.0), 0.08) and math.isclose(fast_ln_ratio(-0.5, 0.0, 0.2, 1.0), -0.12)
      and math.isclose((0.25 - 0.49) / 2, -0.12) and math.isclose((-0.08 + 0.12) / 2, 0.02)
      and math.isclose(fast_ln_ratio(1.0, 0.0, 0.2, 1.0), 0.18) and math.isclose(fast_ln_ratio(-1.0, 0.0, 0.2, 1.0), -0.22)
      and math.isclose((1 - 0.64) / 2, 0.18) and math.isclose((1 - 1.44) / 2, -0.22) and math.isclose((-0.18 + 0.22) / 2, 0.02)
      and math.isclose(0.8 ** 2, 0.64) and math.isclose(1.2 ** 2, 1.44) and math.isclose(0.7 ** 2, 0.49))
check(f"理论：任何一对 ±a，ln 比率取相反数再平均都是 挪动² ÷ 2 = {SHIFT ** 2 / 2:.4g}（挪动 {num(SHIFT)}）",
      all(math.isclose((-fast_ln_ratio(a_p, 0.0, SHIFT, 1.0) - fast_ln_ratio(-a_p, 0.0, SHIFT, 1.0)) / 2, SHIFT ** 2 / 2)
          for a_p in (0.1, 0.5, 1.0, 2.0, 3.7)))
check("展开（字面值重算）：(a − 0.2)² = a² − 0.4a + 0.04，[a² − (a² − 0.4a + 0.04)] ÷ 2 = 0.2a − 0.02；a = 0.5 → 0.2 × 0.5 − 0.02 = 0.1 − 0.02 = 0.08",
      all(math.isclose((a_p - 0.2) ** 2, a_p * a_p - 0.4 * a_p + 0.04, abs_tol=1e-12)
          and math.isclose((a_p * a_p - (a_p * a_p - 0.4 * a_p + 0.04)) / 2, 0.2 * a_p - 0.02, abs_tol=1e-12)
          for a_p in (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0))
      and math.isclose(0.2 * 0.5, 0.1) and math.isclose(0.1 - 0.02, 0.08))
check(f"理论：一般的挪动 d，ln 比率 = d × a − d² ÷ 2（d = {num(SHIFT)}，几个 a 都核对）；旧钟下 a 的平均是 0，平均下来只剩 −d² ÷ 2",
      all(math.isclose(fast_ln_ratio(a_p, 0.0, SHIFT, 1.0), SHIFT * a_p - SHIFT ** 2 / 2, abs_tol=1e-12) for a_p in (-2.0, -0.5, 0.3, 1.0)))
check("d² ÷ 2（字面值重算）：0.2² ÷ 2 = 0.04 ÷ 2 = 0.02；0.1² ÷ 2 = 0.01 ÷ 2 = 0.005；0.04 正好是 0.01 的 4 倍",
      math.isclose(0.2 ** 2, 0.04) and math.isclose(0.04 / 2, 0.02) and math.isclose(0.1 ** 2, 0.01) and math.isclose(0.01 / 2, 0.005)
      and math.isclose(0.04 / 0.01, 4))
a_many = np.random.default_rng(0).normal(0.0, 1.0, 200_000)      # 从旧钟（中心 0、σ = 1）里抽 20 万个动作
ln_ratio_many = (a_many ** 2 - (a_many - SHIFT) ** 2) / 2        # 每个动作的 ln 比率（13.2 节的快算法）
ln_ratio_mean = float(ln_ratio_many.mean())
print(f"从旧钟抽 20 万个动作：ln 比率的平均 = {ln_ratio_mean:.4f}；取相反数 = {-ln_ratio_mean:.4f}，正是 KL {kl_a:.4g}")
check(f"ln 比率在旧钟下的平均 {ln_ratio_mean:.4f}，取相反数正是 KL {kl_a:.4g}（正文的数：−0.0200，固定种子）",
      round(ln_ratio_mean, 4) == -0.02 and SHIFT == 0.2 and abs(ln_ratio_mean + kl_a) < 0.002)
check("动作 0.5 那一条的 ln 比率是 0.08，只是一条，不是平均（13.2 节）",
      math.isclose(ln_new - ln_old, 0.08))
kl_narrow = kl_divergence(Normal(0.0, 0.5), Normal(0.2, 0.5)).item()
check("钟窄一半（σ = 0.5）、同样挪 0.2：(0.2 ÷ 0.5)² ÷ 2 = 0.4² ÷ 2 = 0.08，是 σ = 1 时 0.02 的 4 倍；torch 也是 0.08",
      math.isclose(kl_same_sigma(0.2, 0.5), 0.08) and math.isclose(kl_narrow, 0.08) and math.isclose(0.2 / 0.5, 0.4)
      and math.isclose(0.4 ** 2 / 2, 0.08) and math.isclose(0.08 / 0.02, 4))
d_target = math.sqrt(2 * 0.01)
d_14 = math.sqrt(2 * 0.01 / 14)
print(f"KL = 0.01 对应挪 {d_target:.4f} 个 σ；14 个关节一起挪、合计 0.01：每个挪 {d_14:.4f} 个 σ")
check("目标 0.01：一个关节挪 0.1414 σ（0.1414² ÷ 2 ≈ 0.01）；14 个关节合计 0.01，每个只挪 0.0378 σ（14 × 0.0378² ÷ 2 ≈ 0.01）",
      round(d_target, 4) == 0.1414 and round(d_14, 4) == 0.0378 and round(0.1414 ** 2 / 2, 4) == 0.01
      and round(14 * 0.0378 ** 2 / 2, 4) == 0.01)
check("倒过来解（字面值重算）：0.01 × 2 = 0.02，√0.02 → 0.1414；0.02 ÷ 14 → 0.00143，√0.00143 → 0.0378；≈ 只因为四舍五入：0.1414² ÷ 2 = 0.0099970",
      math.isclose(0.01 * 2, 0.02) and round(math.sqrt(0.02), 4) == 0.1414 and round(0.02 / 14, 5) == 0.00143
      and round(math.sqrt(0.00143), 4) == 0.0378 and round(0.1414 ** 2 / 2, 7) == 0.0099970)
kl14 = kl_divergence(Normal(torch.zeros(14), torch.ones(14)), Normal(torch.full((14,), 0.2), torch.ones(14))).sum().item()
check("14 个关节的中心都挪 0.2 σ：14 × 0.02 = 0.28", math.isclose(kl14, 0.28))
shifts14 = np.linspace(0.02, 0.3, 14)                              # 14 个关节各挪各的
a14 = np.random.default_rng(4).normal(0.0, 1.0, (200_000, 14))      # 从旧的 14 口钟里抽 20 万个 14 维动作
ln_gap14 = ((a14 ** 2 - (a14 - shifts14) ** 2) / 2).sum(axis=1)     # 每条动作：14 个关节的 ln 比率相加
check(f"理论：14 维动作的 KL（按旧钟抽、把 −ln 比率的 14 项和取平均）= 14 个关节各自的 KL 相加 = {float(np.sum(shifts14 ** 2 / 2)):.4f}（抽样容差 0.003）",
      abs(-float(ln_gap14.mean()) - float(np.sum(shifts14 ** 2 / 2))) < 0.003)

mu1, s1, mu2, s2 = 0.0, 1.0, 0.3, 0.8                            # 宽度也变：一般公式的三块（13.5 节）
kl_on = math.log(s2 / s1) + (s1 ** 2 + (mu1 - mu2) ** 2) / (2 * s2 ** 2) - 0.5
kl_no = math.log(s1 / s2) + (s2 ** 2 + (mu2 - mu1) ** 2) / (2 * s1 ** 2) - 0.5
kl_on_t = kl_divergence(Normal(mu1, s1), Normal(mu2, s2)).item()
print(f"三块：ln(0.8 ÷ 1) = {math.log(s2 / s1):.5f}；(1 + 0.3²) ÷ (2 × 0.8²) = {(s1 ** 2 + (mu1 - mu2) ** 2) / (2 * s2 ** 2):.5f}；−0.5；合 {kl_on:.5f}")
table(["", "手算公式", "torch"], [["KL(旧‖新)", f"{kl_on:.4f}", f"{kl_on_t:.4f}"],
                                 ["KL(新‖旧)", f"{kl_no:.4f}", f"{kl_divergence(Normal(mu2, s2), Normal(mu1, s1)).item():.4f}"]])
check("旧钟 (0, 1)、新钟 (0.3, 0.8)：KL(旧‖新) = ln 0.8 + 1.09 ÷ 1.28 − 0.5 = 0.1284，和 torch 一致",
      round(kl_on, 4) == 0.1284 and math.isclose(kl_on, kl_on_t) and round(math.log(0.8) + 1.09 / 1.28 - 0.5, 4) == 0.1284)
check("三块逐块（字面值重算）：ln 0.8 → −0.22314；1 + 0.3² = 1.09，2 × 0.8² = 1.28，1.09 ÷ 1.28 → 0.85156；−0.22314 + 0.85156 − 0.5 = 0.12842 → 0.1284",
      round(math.log(0.8), 5) == -0.22314 and math.isclose(1 + 0.3 ** 2, 1.09) and math.isclose(2 * 0.8 ** 2, 1.28)
      and round(1.09 / 1.28, 5) == 0.85156 and round(-0.22314 + 0.85156 - 0.5, 5) == 0.12842 and round(0.12842, 4) == 0.1284)


def kl_general(mu_o, s_o, mu_n, s_n):
    """一般公式：KL(旧‖新) = ln(σn/σo) + (σo² + (μo − μn)²) ÷ (2σn²) − ½。"""
    return math.log(s_n / s_o) + (s_o ** 2 + (mu_o - mu_n) ** 2) / (2 * s_n ** 2) - 0.5


check("理论：宽度相同时一般公式退回 d² ÷ 2——(1 + d²) ÷ 2 − 0.5 = 0.5 + d² ÷ 2 − 0.5；各种宽度、中心下也和 torch 一致",
      all(math.isclose(kl_general(0.0, 1.0, d, 1.0), d * d / 2, abs_tol=1e-15) and math.isclose((1 + d * d) / 2 - 0.5, d * d / 2, abs_tol=1e-15)
          for d in (0.0, 0.1, 0.2, 0.7))
      and all(math.isclose(kl_general(mo, so, mn, sn), kl_divergence(Normal(mo, so), Normal(mn, sn)).item())
              for mo, so, mn, sn in ((0.0, 1.0, 0.3, 0.8), (0.5, 0.3, -0.2, 1.4), (1.0, 2.0, 1.0, 0.5))))
ln_old_many = -(a_many ** 2) / 2 - 0.5 * math.log(2 * math.pi)                                  # 旧钟 (0, 1) 的 ln 高度
ln_new_many = -((a_many - mu2) ** 2) / (2 * s2 ** 2) - math.log(s2) - 0.5 * math.log(2 * math.pi)  # 新钟 (0.3, 0.8)
kl_sampled = float((ln_old_many - ln_new_many).mean())
sq_mean = float(((a_many - mu2) ** 2).mean())
print(f"进阶折叠：同一批 20 万个旧动作，“ln 旧 − ln 新”直接平均 = {kl_sampled:.4f}（公式 {kl_on:.4f}）；(a − 0.3)² 的平均 = {sq_mean:.4f}（公式 1.09）")
check("进阶折叠：抽样直接平均 → 0.129，和公式的 0.1284 只差运气；(a − 0.3)² 的平均 → 1.092，公式 1 + 0.3² = 1.09（固定种子）",
      round(kl_sampled, 3) == 0.129 and abs(kl_sampled - kl_on) < 0.005 and round(sq_mean, 3) == 1.092 and abs(sq_mean - 1.09) < 0.01)
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
KL_WHERE = ["大于 0.02", "大于 0.02", "中间（0.005 ~ 0.02）", "小于 0.005", "小于 0.005", "恰好等于 0.02"]
KL_HOW = ["0.001 ÷ 1.5", "再 ÷ 1.5 = 0.001 ÷ 2.25", "不动", "× 1.5", "再 × 1.5", "不动（规则写的是“大于”）"]
lr, lr_seq = 1e-3, []
for kl in KL_SEQ:
    lr = adapt(lr, kl)
    lr_seq.append(lr)
table(["这次量到的 KL", "落在哪一段", "怎么算", "学习率"],
      [[f"{k:g}", w, h, f"{v:.8f}".rstrip("0")] for k, w, h, v in zip(KL_SEQ, KL_WHERE, KL_HOW, lr_seq)])
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
check("自测：0.021 → 0.001 ÷ 1.5 = 0.00066667；0.02 → 不动；0.004 → × 1.5，回到 0.001",
      round(lr_t[0], 8) == 0.00066667 and lr_t[1] == lr_t[0] and math.isclose(lr_t[2], 0.001))
check("记号：1.0e-3 = 0.001，1e-5 = 0.00001，1e-2 = 0.01；上下限差 0.01 ÷ 0.00001 = 1000 倍；连除两次 1.5 就是除以 1.5 × 1.5 = 2.25",
      1.0e-3 == 0.001 and 1e-5 == 0.00001 and 1e-2 == 0.01 and round(0.01 / 0.00001) == 1000 and 1.5 * 1.5 == 2.25
      and math.isclose(0.001 / 1.5 / 1.5, 0.001 / 2.25))


def kl_after_adam(lr, steps=4):
    """一个关节的钟，从中心 0 出发用 Adam 走几步（梯度方向一直不变），量走完时和起点那口钟的 KL。"""
    mu_k = torch.tensor(0.0, requires_grad=True)
    opt_k = torch.optim.Adam([mu_k], lr=lr)
    for _ in range(steps):
        opt_k.zero_grad()
        ((mu_k - 3.0) ** 2).backward()
        opt_k.step()
    return kl_same_sigma(mu_k.item())


kl_lr, kl_lr_small = kl_after_adam(0.05), kl_after_adam(0.05 / 1.5)
print(f"Adam 走 4 步：学习率 0.05 → KL {kl_lr:.5f}；学习率 ÷ 1.5 → KL {kl_lr_small:.5f}；两者之比 {kl_lr / kl_lr_small:.3f}")
check("为什么调学习率管用（字面值重算）：KL 跟着挪动的平方走，学习率 ÷ 1.5 → KL 约 ÷ 2.25；0.03 ÷ 2.25 → 0.0133，落回 0.005 ~ 0.02",
      math.isclose(1.5 ** 2, 2.25) and round(0.03 / 2.25, 4) == 0.0133 and 0.005 < 0.03 / 2.25 < 0.02)
check("理论：用 Adam 真走 4 步，同样几步攒下的 KL，学习率 ÷ 1.5 之后约是原来的 1/2.25（比值差不到 0.05）",
      abs(kl_lr / kl_lr_small - 2.25) < 0.05)

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
    eps32 = torch.finfo(torch.float32).eps
    print(f"float32 能分辨的最小相对差别 = {eps32:.3g}（大约 7 位有效数字）；1 + 0.00000004 存下来 = "
          f"{float(torch.tensor(1.0, dtype=torch.float32) + torch.tensor(4e-8, dtype=torch.float32)):.10f}")
    check("进阶折叠：float32 只记得约 7 位有效数字（能分辨的最小相对差别在 0.0000001 到 0.0000002 之间），亿分之几加到 1 上就没了",
          1e-7 < eps32 < 2e-7 and 4e-8 < eps32 / 2)

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
ax_bot.set_xticklabels(["0", num(SHIFT / 2), f"{d_target:.4f}", num(SHIFT)] + (["0.3"] if d_max >= 0.3 and SHIFT < 0.29 else []))
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
check("迷你版：2 只 × 3 步 = 6 条；6 ÷ 2 = 3 条一份；2 份 × 2 遍 = 4 次更新，每条用 2 次",
      2 * 3 == 6 and 6 // 2 == 3 and 2 * 2 == 4)
check("手乘（字面值重算）：4096 × 20 = 81,920，4096 × 4 = 16,384，81,920 + 16,384 = 98,304；对半再对半：98,304 ÷ 2 = 49,152，49,152 ÷ 2 = 24,576",
      4096 * 20 == 81_920 and 4096 * 4 == 16_384 and 81_920 + 16_384 == 98_304 == 4096 * (20 + 4)
      and 98_304 // 2 == 49_152 and 49_152 // 2 == 24_576)
try:
    from tensordict import TensorDict as _TD
    from rsl_rl.storage.rollout_storage import RolloutStorage
except ImportError:
    RolloutStorage = None
if RolloutStorage is None:
    print("  （当前 Python 环境里没有 rsl_rl，跳过“真发一次牌”；用 uv run 运行就有）")
else:
    def deal(n_envs, n_steps, n_mb, n_epochs):
        """用 rsl_rl 自己的 mini_batch_generator 发一次牌：每条记录的编号存在 values 里，数每份有谁、每条被用了几次。"""
        torch.manual_seed(0)
        st = RolloutStorage("rl", n_envs, n_steps, _TD({"actor": torch.zeros(n_envs, 1)}, batch_size=[n_envs]), [1])
        st.values = torch.arange(n_envs * n_steps, dtype=torch.float64).reshape(n_steps, n_envs, 1)
        st.distribution_params = (torch.zeros(n_steps, n_envs, 1), torch.ones(n_steps, n_envs, 1))
        ids = [b.values.view(-1).long() for b in st.mini_batch_generator(n_mb, n_epochs)]
        uses = torch.bincount(torch.cat(ids), minlength=n_envs * n_steps)
        same_split = all(torch.equal(ids[e * n_mb + i], ids[i]) for e in range(n_epochs) for i in range(n_mb))
        return len(ids), [len(x) for x in ids], uses, same_split

    n_mini, sizes_mini, uses_mini, same_mini = deal(2, 3, 2, 2)
    n_real, sizes_real, uses_real, same_real = deal(N_ENVS, N_STEPS, N_MB, N_EPOCHS)
    print(f"rsl_rl 真发一次牌：迷你版 {n_mini} 份、每份 {sizes_mini[0]} 条、每条用 {int(uses_mini.min())}–{int(uses_mini.max())} 次；"
          f"项目规模 {n_real} 份、每份 {sizes_real[0]:,} 条、每条用 {int(uses_real.min())}–{int(uses_real.max())} 次")
    check("rsl_rl 的 mini_batch_generator 真发一次：迷你版 4 份 × 3 条、每条正好 2 次；项目规模 20 份 × 24,576 条、每条正好 5 次",
          n_mini == 4 and set(sizes_mini) == {3} and bool((uses_mini == 2).all())
          and n_real == 20 and set(sizes_real) == {24_576} and bool((uses_real == 5).all()))
    check("开头只洗一次牌：5 遍里，第 k 份每一遍都是同一批记录（迷你版和项目规模都是）", same_mini and same_real)
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
for i, (lab, val) in enumerate([("动作", "0.5"), ("当时的 ln π", m(ln_old, ".4f")), ("优势 Â", "+2")]):
    ax.text(0.5 + i * 3.4, 2.75, lab, fontsize=FS_SMALL, color=MUTED, va="center")
    cell(ax, 0.5 + i * 3.4, 1.55, val, width=2.6, height=0.8)
note(ax, 0.3, 0.55, "这些数只当数据存下来；之后 20 次更新都拿它们来比。")
ax = axes[1]
lesson_panel(ax, "② 更新时，用现在的网络对同一个动作重算 ln π", xmax=10.8, ymax=4.0)
hand(ax, 0.3, 2.45, f"现在的 ln π = {m(ln_new, '.4f')}")
hand(ax, 0.3, 1.5, f"比率 = exp({m(ln_new, '.4f')} − ({m(ln_old, '.4f')})) = exp(0.08) = {rho:.4f}")
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

    check("rsl_rl/algorithms/ppo.py 的 process_env_step()：采数据时每走一个环境步，就更新一次输入归一化器（13.5 节折叠块）",
          "def process_env_step(" in ppo_text and "self.actor.update_normalization(obs)" in ppo_text)
    dist_path = rsl_dir / "modules" / "distribution.py"
    dist_text = dist_path.read_text(encoding="utf-8") if dist_path.is_file() else ""
    check("rsl_rl/modules/distribution.py：KL(旧‖新)，14 个关节相加",
          "return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)" in dist_text)
    stor_path = rsl_dir / "storage" / "rollout_storage.py"
    stor_text = stor_path.read_text(encoding="utf-8") if stor_path.is_file() else ""
    check("rsl_rl/storage/rollout_storage.py：每份 = 总条数 // 份数（98,304 // 4 = 24,576）",
          "mini_batch_size = batch_size // num_mini_batches" in stor_text)
    gen_at = stor_text.find("    def mini_batch_generator(")
    check("rsl_rl/storage/rollout_storage.py 的 mini_batch_generator()：先洗一次牌（randperm），再进“5 遍 × 4 份”的两层循环（13.6 节）",
          gen_at >= 0 and lines_in_order(stor_text[gen_at:], [
              "indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)",
              "for epoch in range(num_epochs):",
              "for i in range(num_mini_batches):",
              "batch_idx = indices[start:stop]"]))
    log_path = rsl_dir / "utils" / "logger.py"
    log_text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    check("rsl_rl/utils/logger.py：训练日志印“Mean surrogate loss:”，wandb 里叫 Loss/surrogate",
          'f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\\n"""' in log_text and 'f"Loss/{key}"' in log_text
          and '"surrogate": mean_surrogate_loss,' in ppo_text)
else:
    print("  （当前 Python 环境里没有 rsl_rl，跳过源码核对；用 uv run 运行就会核对）")

done()
