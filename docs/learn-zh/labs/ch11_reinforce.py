"""第 11 章实验：策略梯度。两个键的游戏机、对数导数技巧、钟的中心怎么挪、基线、REINFORCE、熵奖励。

运行：uv run python docs/learn-zh/labs/ch11_reinforce.py
纯 CPU，numpy + torch + matplotlib，半分钟上下。第 7 节只读四份源码的文字，不加载机器人、不训练。
小节编号与正文一一对应：实验第 K 节 = 正文 11.K 节（第 7 节对应「映射到项目」）。
正文“改一改”要改的三行都带 `# TWEAK-k:` 标记（第 1、4、6 节各一处）。

第 5、6 节的“推小车”：观测 s ∈ (−1, 1) 是小车离目标的位置，动作 a 是把小车推多远，推完小车到 s + a，
奖励 r = −(s + a)²，最优动作是 a = −s。一局只有一步（推一次就结束）。
第 6 节还有一台“两座山的老虎机”：没有观测，动作 a 的分数是两座钟形的山——小山在 a = 0 值 1 分，大山在 a = 1.5 值 2 分。
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial.hermite_e import hermegauss

from _common import banner, check, done, lines_in_order, savefig, table
from _draw import (BLUE, CELL_EDGE, FAINT, FS_NOTE, FS_SMALL, FS_STEP, FS_TICK, FS_TITLE, GREEN, INK, MUTED, ORANGE,
                   WHITE_BOX, cell, data_axes, hand, lesson_figure, lesson_panel, note, panel_note, panel_title, plt)

np.set_printoptions(precision=4, suppress=True)


def fmt(v, digits=None) -> str:
    """正文写法：去掉多余的零，负号用 −。"""
    s = f"{v:g}" if digits is None else f"{v:.{digits}f}"
    return s.replace("-", "−")


def f(v, digits) -> str:
    """固定小数位，负号用 −；−0.00 写成 0.00。"""
    s = f"{v:.{digits}f}"
    return (s[1:] if float(s) == 0 and s.startswith("-") else s).replace("-", "−")


# ---------------------------------------------------------------------------
banner("1. 策略梯度：两个键的游戏机，平均分对 p 的斜率")
# 教学构造：蓝键 3 分、绿键 1 分，一按就结束。策略只有一个旋钮 p = 按蓝键的概率。
R_BLUE = 3.0  # TWEAK-1: 5.0
R_GREEN = 1.0
P0 = 0.5


def J(p):
    """平均分：按概率加权（第 4 章 4.3 节）。"""
    return R_BLUE * p + R_GREEN * (1 - p)


H = 0.01
slope_fd = (J(P0 + H) - J(P0 - H)) / (2 * H)          # 第 1 章 1.4 节“缩 h”：左右各挪 0.01
slope = R_BLUE - R_GREEN                             # J = 1 + 2p 是一条直线，斜率处处是 2
ALPHA = 0.1
p1 = P0 + ALPHA * slope                              # 梯度上升：加号
p2 = p1 + ALPHA * slope                              # 自测：再走一步
p3 = p2 + ALPHA * slope                              # 再走一步就越过 1 了
table(["p", "J(p) = 3p + 1 × (1 − p)"], [[fmt(p), fmt(round(J(p), 10))] for p in (0.49, P0, 0.51, p1, p2)])
print(f"缩 h：({fmt(round(J(P0 + H), 10))} − {fmt(round(J(P0 - H), 10))}) / 0.02 = {fmt(round(slope_fd, 10))}")
print(f"梯度上升一步（α = {ALPHA:g}）：p = {P0:g} + {ALPHA:g} × {fmt(slope)} = {fmt(round(p1, 10))}，J = {fmt(round(J(p1), 10))}")
check("J(0.5) = 3 × 0.5 + 1 × 0.5 = 2", math.isclose(J(P0), 2.0))
check("缩 h 量出的斜率 = 2，就是 J = 1 + 2p 的斜率", math.isclose(slope_fd, 2.0) and math.isclose(slope, 2.0))
check("字面值：J(0.51) = 2.02，J(0.49) = 1.98，(2.02 − 1.98) / 0.02 = 2",
      math.isclose(J(0.51), 2.02) and math.isclose(J(0.49), 1.98) and math.isclose((2.02 - 1.98) / 0.02, 2.0))
check("梯度上升一步：p 0.5 → 0.7，J 2 → 2.4", math.isclose(p1, 0.7) and math.isclose(J(p1), 2.4))
check("自测：再走一步 p = 0.7 + 0.1 × 2 = 0.9，J = 2.8；下一步 1.1 越过了 1",
      math.isclose(p2, 0.9) and math.isclose(J(p2), 2.8) and math.isclose(p3, 1.1) and p3 > 1)

# ---------------------------------------------------------------------------
banner("2. 对数导数技巧：抽到谁就推谁，推多少看分数")


def blue_share(p, b=0.0):
    """抽到蓝键的样本贡献：ln p 对 p 的导数 1/p，乘（分数 − 基线）。"""
    return (1 / p) * (R_BLUE - b)


def green_share(p, b=0.0):
    """抽到绿键的样本贡献：ln(1 − p) 对 p 的导数 −1/(1 − p)，乘（分数 − 基线）。"""
    return (-1 / (1 - p)) * (R_GREEN - b)


def average_share(p, b=0.0):
    return p * blue_share(p, b) + (1 - p) * green_share(p, b)


rows = []
for p in (P0, 0.8, 0.2):
    rows.append([fmt(p), fmt(round(blue_share(p), 10)), fmt(round(green_share(p), 10)), fmt(round(average_share(p), 10))])
table(["p", "抽到蓝键记", "抽到绿键记", "按概率平均"], rows)
no_divide = {p: p * R_BLUE * 1 + (1 - p) * R_GREEN * (-1) for p in (P0, 0.8)}
print(f"不除以概率（每次原样记 +3 或 −1）：p = 0.5 平均 {fmt(round(no_divide[P0], 10))}，p = 0.8 平均 {fmt(round(no_divide[0.8], 10))}"
      "——都不是斜率 2")
check("p = 0.5：抽到蓝键 (1/0.5) × 3 = 6，抽到绿键 (−1/0.5) × 1 = −2，平均 2 = 斜率",
      math.isclose(blue_share(P0), 6) and math.isclose(green_share(P0), -2) and math.isclose(average_share(P0), slope))
check("p = 0.8：3.75 和 −5，平均 0.8 × 3.75 + 0.2 × (−5) = 3 − 1 = 2",
      math.isclose(blue_share(0.8), 3.75) and math.isclose(green_share(0.8), -5)
      and math.isclose(0.8 * 3.75 + 0.2 * -5, 2) and math.isclose(average_share(0.8), 2))
check("自测 p = 0.2：15 和 −1.25，平均 0.2 × 15 + 0.8 × (−1.25) = 3 − 1 = 2",
      math.isclose(blue_share(0.2), 15) and math.isclose(green_share(0.2), -1.25)
      and math.isclose(0.2 * 15 + 0.8 * -1.25, 2) and math.isclose(average_share(0.2), 2))
check("两块面积与 p 无关：蓝 p × 3/p = 3，绿 (1 − p) × (−1/(1 − p)) = −1",
      all(math.isclose(p * blue_share(p), 3) and math.isclose((1 - p) * green_share(p), -1) for p in (0.2, 0.5, 0.8)))
check("不除以概率就不对：p = 0.5 平均 1，p = 0.8 平均 2.2",
      math.isclose(no_divide[P0], 1.0) and math.isclose(no_divide[0.8], 2.2))
raw_area = {p: (p * R_BLUE * 1, (1 - p) * R_GREEN * -1) for p in (P0, 0.8)}
print(f"不除以概率时的面积：p = 0.5 → {fmt(round(raw_area[P0][0], 10))} 和 {fmt(round(raw_area[P0][1], 10))}；"
      f"p = 0.8 → {fmt(round(raw_area[0.8][0], 10))} 和 {fmt(round(raw_area[0.8][1], 10))}（跟着 p 变）")
check("第二步：两笔各算一次 3 + (−1) = 2 = 斜率；p 加 0.01，蓝键那项多 3 × 0.01 = 0.03，绿键那项少 0.01，合起来 0.02",
      R_BLUE * 1 + R_GREEN * -1 == 2 == slope and math.isclose(3 * 0.01, 0.03) and math.isclose(0.03 - 0.01, 0.02)
      and math.isclose(J(P0 + 0.01) - J(P0), 0.02))
check("图 ①：不除以概率时面积跟着 p 变——p = 0.5 是 1.5 和 −0.5，p = 0.8 是 2.4 和 −0.2",
      all(math.isclose(x, y) for x, y in zip(raw_area[P0] + raw_area[0.8], (1.5, -0.5, 2.4, -0.2))))

# ln 的导数：缩 h 量一次，torch 自动求导再算一次
ln_fd = (math.log(0.501) - math.log(0.5)) / 0.001
p_t = torch.tensor(P0, requires_grad=True)
torch.log(p_t).backward()
grad_ln_blue = p_t.grad.item()
p_t.grad = None
torch.log(1 - p_t).backward()
grad_ln_green = p_t.grad.item()
print(f"缩 h：(ln 0.501 − ln 0.5) / 0.001 = {ln_fd:.3f}，靠近 1/0.5 = 2")
print(f"torch：ln p 对 p 的导数 {grad_ln_blue:g}，ln(1 − p) 对 p 的导数 {grad_ln_green:g}")
check("(ln 0.501 − ln 0.5) / 0.001 = 1.998，靠近 1/0.5 = 2", round(ln_fd, 3) == 1.998)
check("torch：d ln p / dp = 1/p = 2，d ln(1 − p) / dp = −1/(1 − p) = −2",
      math.isclose(grad_ln_blue, 1 / P0) and math.isclose(grad_ln_green, -1 / (1 - P0)))

# 只凭“玩”的记录：按策略抽 10 万次，每次记那一笔，取平均
N_PLAYS = 100_000
rng2 = np.random.default_rng(2)                      # 每一节用自己的随机数（种子 = 节号），互不牵连
blue_drawn = rng2.random(N_PLAYS) < P0
plays = np.where(blue_drawn, blue_share(P0), green_share(P0))
play_mean = float(plays.mean())
print(f"按策略抽 {N_PLAYS:,} 次：蓝键 {int(blue_drawn.sum()):,} 次，每次记的数取平均 = {play_mean:.3f}")
check("抽 10 万次取平均 = 1.990（这一组随机数）", round(play_mean, 3) == 1.990)
check(f"和斜率 {fmt(slope)} 之差小于 4 × 单次标准差 / √n（{4 * 4 / math.sqrt(N_PLAYS):.3f}）",
      abs(play_mean - slope) < 4 * 4 / math.sqrt(N_PLAYS))

# ---------------------------------------------------------------------------
banner("2b. 画图：figures/ch11_two_buttons.png（原样记 vs 除以概率：宽 = 抽到的比例，高 = 每次记的数，面积 = 在平均里占的份）")
fig = plt.figure(figsize=(10.4, 13.6))
fig.suptitle("除以概率：面积不再随抽到的频率变，合起来正是斜率", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
ymax = max(7.0, blue_share(P0) * 1.18)
row_specs = (("raw", lambda p: R_BLUE * 1.0, lambda p: R_GREEN * -1.0, 0.615),   # 上一行：原样记“导数 × 分数”
             ("div", blue_share, green_share, 0.170))                            # 下一行：再除以这个键的概率
row_axes, row_avg = {}, {}
for kind, rec_blue, rec_green, bottom in row_specs:
    row_axes[kind], row_avg[kind] = [], []
    for col, p in enumerate((P0, 0.8)):
        ax = fig.add_axes([0.115 + col * 0.47, bottom, 0.39, 0.235])
        data_axes(ax, "抽到的比例（概率）", "每次记的数" if col == 0 else "")
        hb, hg = rec_blue(p), rec_green(p)
        avg = p * hb + (1 - p) * hg
        ax.add_patch(plt.Rectangle((0, 0), p, hb, facecolor=BLUE, alpha=0.85, edgecolor="none", zorder=3))
        ax.add_patch(plt.Rectangle((p, 0), 1 - p, hg, facecolor=GREEN, alpha=0.85, edgecolor="none", zorder=3))
        ax.axhline(0, color=CELL_EDGE, lw=1.4, zorder=2)
        ax.plot([0, 1], [avg, avg], color=ORANGE, ls="--", lw=2.6, zorder=5)
        y_blue = (avg + hb) / 2 if hb - avg >= avg else avg / 2          # 面积字放在虚线上下更宽的那一边
        ax.text(p / 2, y_blue, f"面积 {fmt(round(p * hb, 10))}", color="white", fontsize=FS_SMALL, ha="center", va="center", zorder=6)
        green_label = f"面积 {fmt(round((1 - p) * hg, 10))}"
        if 1 - p >= 0.45:
            ax.text(p + (1 - p) / 2, hg - 0.35, green_label, color=GREEN, fontsize=FS_SMALL, ha="center", va="top", zorder=6)
        else:
            ax.text(p - 0.03, min(hg * 0.55, -1.0), green_label, color=GREEN, fontsize=FS_SMALL, ha="right", va="center",
                    bbox=WHITE_BOX, zorder=6)
        ax.text(0.5, -0.26, f"平均 {fmt(p)} × {fmt(round(hb, 10))} + {fmt(round(1 - p, 10))} × ({fmt(round(hg, 10))}) = "
                f"{fmt(round(avg, 10))}", transform=ax.transAxes, color=ORANGE, fontsize=FS_SMALL, ha="center", va="top")
        ax.set_title(f"p = {fmt(p)}", loc="left", fontsize=FS_SMALL, color=INK, pad=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(-6.4, ymax)
        ax.set_xticks(sorted({0, 0.5, p, 1}))
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−")))
        row_axes[kind].append(ax)
        row_avg[kind].append(avg)
panel_title(fig, row_axes["raw"], f"① 原样记 {fmt(R_BLUE)} 或 −{fmt(R_GREEN)}：面积跟着抽到的频率变")
panel_note(fig, row_axes["raw"], "橙色虚线是平均：两块面积相加，摊在总宽 1 上。蓝块变宽，面积就变大——\n"
           f"平均 {fmt(round(row_avg['raw'][0], 10))} 和 {fmt(round(row_avg['raw'][1], 10))}，都不是斜率 {fmt(slope)}。")
panel_title(fig, row_axes["div"], f"② 除以概率：面积锁在 {fmt(R_BLUE)} 和 −{fmt(R_GREEN)}，平均总是斜率 {fmt(slope)}")
panel_note(fig, row_axes["div"], "宽了就矮、窄了就深：抽得多就少记，抽得少就多记，面积不变。")
savefig(fig, "ch11_two_buttons")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("3. 连续动作：钟的中心往好动作那边挪")
MU, SIGMA, A_SAMPLE, SCORE = 0.0, 0.5, 0.2, 1.5    # 教学构造：中心 0、宽 0.5，抽到 0.2，这次得 1.5 分


def first_term(mu, a=A_SAMPLE, sigma=SIGMA):
    """ln 密度里唯一含 μ 的一项：−(a − μ)² / (2σ²)。"""
    return -(a - mu) ** 2 / (2 * sigma ** 2)


t0, t1, t2 = first_term(0.0), first_term(0.01), first_term(0.001)
fd1, fd2 = (t1 - t0) / 0.01, (t2 - t0) / 0.001
rule = (A_SAMPLE - MU) / SIGMA ** 2
push = rule * SCORE
LR_MU = 0.01
mu_up, mu_down = MU + LR_MU * push, MU + LR_MU * rule * (-SCORE)
table(["μ", "−(0.2 − μ)² / (2 × 0.25)", "比 μ = 0 时多了", "÷ 挪动的量"],
      [["0", fmt(round(t0, 6)), "", ""], ["0.01", fmt(round(t1, 6)), fmt(round(t1 - t0, 6)), fmt(round(fd1, 6))],
       ["0.001", fmt(round(t2, 6)), fmt(round(t2 - t0, 6)), fmt(round(fd2, 6))]])
print(f"规则：(a − μ)/σ² = (0.2 − 0)/0.5² = {fmt(rule)}；乘分数 1.5：{fmt(rule)} × 1.5 = {fmt(round(push, 10))}")
print(f"μ 走一步（倍数 0.01）：0 + 0.01 × {fmt(round(push, 10))} = {fmt(round(mu_up, 10))}；分数若是 −1.5：{fmt(round(mu_down, 10))}")
mu_t = torch.tensor(MU, requires_grad=True)
torch.distributions.Normal(mu_t, SIGMA).log_prob(torch.tensor(A_SAMPLE)).backward()
print(f"torch：Normal(μ, 0.5).log_prob(0.2) 对 μ 的导数 = {mu_t.grad.item():.4f}")
check("缩 h：μ = 0 → −0.08，μ = 0.01 → −0.0722（0.78），μ = 0.001 → −0.079202（0.798）",
      math.isclose(t0, -0.08) and math.isclose(t1, -0.0722) and math.isclose(t2, -0.079202)
      and round(fd1, 6) == 0.78 and round(fd2, 6) == 0.798)
check("字面值：0.04/0.5 = 0.08，0.0361/0.5 = 0.0722，0.039601/0.5 = 0.079202，0.0078/0.01 = 0.78，0.000798/0.001 = 0.798",
      math.isclose(0.04 / 0.5, 0.08) and math.isclose(0.0361 / 0.5, 0.0722) and math.isclose(0.039601 / 0.5, 0.079202)
      and math.isclose(0.0078 / 0.01, 0.78) and math.isclose(0.000798 / 0.001, 0.798))
check("规则 (0.2 − 0)/0.5² = 0.8，和 torch 的自动求导一致", math.isclose(rule, 0.8) and math.isclose(mu_t.grad.item(), 0.8, rel_tol=1e-6))
check("单样本贡献 (0.2 − 0)/0.5² × 1.5 = 1.2；μ → 0.012；分数 −1.5 时 μ → −0.012",
      math.isclose(push, 1.2) and math.isclose(mu_up, 0.012) and math.isclose(mu_down, -0.012))

# 一个能对答案的例子：中心 0.3、宽 0.5，分数 r = −(a − 1)²（a = 1 最好）
MU_EX, SIG_EX = 0.3, 0.5


def r_ex(a):
    return -((a - 1.0) ** 2)


def J_ex(mu):
    """平均分：平方的平均 = 平均的平方 + 方差（第 4 章 4.4 节倒过来用）。"""
    return -((mu - 1.0) ** 2 + SIG_EX ** 2)


slope_ex_fd = (J_ex(MU_EX + 0.01) - J_ex(MU_EX)) / 0.01
slope_ex = -2 * (MU_EX - 1.0)
print(f"J(0.3) = −(0.7² + 0.5²) = {fmt(round(J_ex(MU_EX), 10))}；J(0.31) = {fmt(round(J_ex(MU_EX + 0.01), 10))}；"
      f"缩 h 斜率 {slope_ex_fd:.2f}；规则 −2(μ − 1) = {fmt(round(slope_ex, 10))}")
nodes, weights = hermegauss(60)                     # 标准钟下的期望，用 60 个点精确积分（独立于抽样）
weights = weights / math.sqrt(2 * math.pi)


def expect(f):
    return float(np.sum(weights * f(nodes)))


g_plain = lambda e: r_ex(MU_EX + SIG_EX * e) * e / SIG_EX          # (a − μ)/σ² = ε/σ
exact_mean = expect(g_plain)
exact_std_plain = math.sqrt(expect(lambda e: g_plain(e) ** 2) - exact_mean ** 2)
rng3 = np.random.default_rng(3)
rows, trick_est = [], {}
for n in (100, 10_000, 1_000_000):
    a = MU_EX + SIG_EX * rng3.standard_normal(n)
    trick_est[n] = float((r_ex(a) * (a - MU_EX) / SIG_EX ** 2).mean())
    rows.append([f"{n:,}", f"{trick_est[n]:.3f}"])
table(["抽几个样本", "样本贡献的平均"], rows)
print(f"精确积分：样本贡献的平均 = {exact_mean:.4f}，单个样本贡献的标准差 = {exact_std_plain:.4f}")
check("J(0.3) = −0.74，J(0.31) = −0.7261，缩 h 斜率 1.39；规则 −2 × (0.3 − 1) = 1.4",
      math.isclose(J_ex(0.3), -0.74) and math.isclose(J_ex(0.31), -0.7261) and round(slope_ex_fd, 2) == 1.39
      and math.isclose(slope_ex, 1.4))
check("字面值：0.49 + 0.25 = 0.74，0.4761 + 0.25 = 0.7261，(−0.7261 + 0.74)/0.01 = 1.39",
      math.isclose(0.49 + 0.25, 0.74) and math.isclose(0.4761 + 0.25, 0.7261) and math.isclose((-0.7261 + 0.74) / 0.01, 1.39))
check("精确积分：样本贡献的平均正好是 1.4（对数导数技巧给的就是斜率）", math.isclose(exact_mean, 1.4, rel_tol=1e-9))
check("J = −((μ − 1)² + σ²) 和逐点积分出来的平均分一致（平方的平均 = 平均的平方 + 方差）；a − 1 的平均 0.3 − 1 = −0.7；"
      "ln 密度里 2π 的 π 是圆周率 3.14159",
      math.isclose(expect(lambda e: r_ex(MU_EX + SIG_EX * e)), J_ex(MU_EX), rel_tol=1e-9) and math.isclose(MU_EX - 1, -0.7)
      and round(math.pi, 5) == 3.14159)
check("抽 100、1 万、100 万个：1.815、1.405、1.398（这一组随机数）",
      [round(trick_est[n], 3) for n in (100, 10_000, 1_000_000)] == [1.815, 1.405, 1.398])
check("三个估计离 1.4 都不到 4 × 单次标准差 / √n：样本越多越准（第 4 章 4.5 节）",
      all(abs(trick_est[n] - slope_ex) < 4 * exact_std_plain / math.sqrt(n) for n in trick_est))

# 自测：台阶奖励——a 超过 0.5 得 1 分，否则 0 分。它对 a 的导数几乎处处是 0
step_r = lambda a: (a > 0.5).astype(float)
share_07 = (0.7 - MU) / SIGMA ** 2 * 1.0
share_03 = (0.3 - MU) / SIGMA ** 2 * 0.0
step_exact = math.exp(-0.5) / (SIGMA * math.sqrt(2 * math.pi))                  # 台阶处 a = 0.5 的密度：φ(1)/σ
a = MU + SIGMA * rng3.standard_normal(1_000_000)
step_est = float((step_r(a) * (a - MU) / SIGMA ** 2).mean())
print(f"台阶奖励：抽到 0.7 记 {fmt(round(share_07, 10))}，抽到 0.3 记 {fmt(share_03)}；"
      f"100 万个样本平均 {step_est:.3f}，公式 exp(−0.5)/(0.5 × 2.5066) = {step_exact:.4f}")
check("自测：台阶奖励下，抽到 0.7 记 (0.7/0.25) × 1 = 2.8；抽到 0.3，0.3/0.25 = 1.2，乘分数 0 记 0",
      math.isclose(share_07, 2.8) and share_03 == 0 and math.isclose((0.3 - MU) / SIGMA ** 2, 1.2))
check("图 ①：σ = 0.5 时直线斜率 1/0.25 = 4；钟窄一半（σ = 0.25），同一个 0.2 推到 0.2/0.0625 = 3.2，是 0.8 的 4 倍",
      math.isclose(1 / SIGMA ** 2, 4) and math.isclose(A_SAMPLE / (SIGMA / 2) ** 2, 3.2) and math.isclose(3.2 / 0.8, 4))
check("台阶奖励的策略梯度不是 0：平均 0.48（= 钟在 0.5 处的高度）", round(step_exact, 2) == 0.48 and abs(step_est - step_exact) < 0.01)

# 把钟一步步挪过去：每次更新抽 64 个，按对数导数技巧估斜率，按学习率 0.02 挪一次中心（σ 不动）
N_PUSH, ALPHA_PUSH, STEPS_PUSH = 64, 0.02, 60
mu_path = [MU_EX]
for _ in range(STEPS_PUSH):
    a = mu_path[-1] + SIG_EX * rng3.standard_normal(N_PUSH)
    mu_path.append(mu_path[-1] + ALPHA_PUSH * float((r_ex(a) * (a - mu_path[-1]) / SIG_EX ** 2).mean()))
SHOW_STEPS = (0, 10, 25, 60)
print("中心一步步挪：" + "，".join(f"更新 {k} 次 μ = {mu_path[k]:.2f}" for k in SHOW_STEPS))
check("更新 60 次之后，中心到了 a = 1 附近（误差小于 0.1）", abs(mu_path[-1] - 1.0) < 0.1)
check("图里标的四个中心：0.30、0.55、0.76、0.95（这一组随机数）", [round(mu_path[k], 2) for k in SHOW_STEPS] == [0.30, 0.55, 0.76, 0.95])

# ---------------------------------------------------------------------------
banner("3b. 画图：figures/ch11_gauss_push.png（推力是一条直线；钟一步步挪向分数最高处）")
fig = plt.figure(figsize=(9.6, 12.8))
fig.suptitle("连续动作：抽到哪边就往哪边推，按分数加权，钟的中心挪向好动作", fontsize=FS_TITLE - 1, fontweight="bold", color=INK, y=0.985)
ax = fig.add_axes([0.15, 0.585, 0.80, 0.29])
data_axes(ax, "抽到的动作 a", "ln π(a) 对 μ 的导数")
aa = np.linspace(-1.2, 1.2, 200)
ax.plot(aa, (aa - MU) / SIGMA ** 2, color=BLUE, lw=3, zorder=4)
ax.plot(aa, (aa - MU) / (SIGMA / 2) ** 2, color=GREEN, lw=2.4, ls="--", zorder=4)
ax.axvline(MU, color=FAINT, lw=1.6, ls=":", zorder=2)
ax.axhline(0, color=CELL_EDGE, lw=1.2, zorder=2)
ax.plot([A_SAMPLE], [rule], "o", color=ORANGE, ms=11, zorder=6)
ax.plot([A_SAMPLE], [A_SAMPLE / (SIGMA / 2) ** 2], "o", color=GREEN, ms=9, zorder=6)
ax.text(A_SAMPLE + 0.09, rule - 0.05, f"a = 0.2：(0.2 − 0)/0.5² = {fmt(rule)}", color=ORANGE, fontsize=FS_SMALL, va="center",
        bbox=WHITE_BOX, zorder=7)
ax.text(A_SAMPLE - 0.08, A_SAMPLE / (SIGMA / 2) ** 2, f"钟窄一半（σ = 0.25）：0.2/0.25² = {fmt(A_SAMPLE / (SIGMA / 2) ** 2)}",
        color=GREEN, fontsize=FS_SMALL, ha="right", va="center", bbox=WHITE_BOX, zorder=7)
ax.text(MU + 0.06, -1.5, "中心 μ = 0：抽到正中，不推", color=MUTED, fontsize=FS_SMALL, ha="left", va="center", bbox=WHITE_BOX, zorder=7)
ax.text(-1.15, -2.5, "σ = 0.5", color=BLUE, fontsize=FS_SMALL, va="center", bbox=WHITE_BOX)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-5.5, 5.5)
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}".replace("-", "−")))
panel_title(fig, [ax], "① 每个样本把中心往自己那边推：(a − μ)/σ²")
panel_note(fig, [ax], "抽在中心右边，数是正的（把 μ 往右推）；在左边是负的。离中心越远、钟越窄，推得越狠。")

ax = fig.add_axes([0.15, 0.115, 0.80, 0.29])
data_axes(ax, "动作 a", "密度")
aa = np.linspace(-1.5, 2.6, 400)
bell = lambda x, m: np.exp(-(x - m) ** 2 / (2 * SIG_EX ** 2)) / (SIG_EX * math.sqrt(2 * math.pi))
for i, (k, color, lw) in enumerate(zip(SHOW_STEPS, (FAINT, "#8fb3d9", BLUE, ORANGE), (2.4, 2.4, 2.6, 3.2))):
    m = mu_path[k]
    ax.plot(aa, bell(aa, m), color=color, lw=lw, zorder=4)
    ax.text(-1.45, 1.10 - 0.085 * i, f"更新 {k} 次：μ = {m:.2f}", color={FAINT: MUTED, "#8fb3d9": "#6f98c4"}.get(color, color), fontsize=FS_SMALL,
            va="center", zorder=6)
ax.annotate("", xy=(mu_path[-1], 0.88), xytext=(mu_path[0], 0.88), zorder=6,
            arrowprops=dict(arrowstyle="-|>", lw=2.4, color=ORANGE, shrinkA=0, shrinkB=0))
ax.axvline(1.0, color=GREEN, lw=2, ls="--", zorder=3)
ax.text(1.06, 1.10, "a = 1：分数 −(a − 1)² 最高", color=GREEN, fontsize=FS_SMALL, va="center", bbox=WHITE_BOX, zorder=7)
ax.set_xlim(-1.5, 2.6)
ax.set_ylim(0, 1.18)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}".replace("-", "−")))
panel_title(fig, [ax], "② 每次更新抽 64 个、按分数加权推一把：中心从 0.3 往 1 挪（σ 不动）")
panel_note(fig, [ax], "四口钟一样宽，只是中心不同。橙色箭头是 60 次更新一共挪的距离：\n离 1 远时斜率大、挪得快，越靠近 1 挪得越慢。")
savefig(fig, "ch11_gauss_push")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("4. 基线：只看比平均好多少")
B = 2.0
blue_b, green_b = blue_share(P0, B), green_share(P0, B)
mult_mean = {p: p * (1 / p) + (1 - p) * (-1 / (1 - p)) for p in (P0, 0.8)}
std_plain = math.sqrt(P0 * (blue_share(P0) - average_share(P0)) ** 2 + (1 - P0) * (green_share(P0) - average_share(P0)) ** 2)
std_base = math.sqrt(P0 * (blue_b - average_share(P0, B)) ** 2 + (1 - P0) * (green_b - average_share(P0, B)) ** 2)
V_game = P0 * R_BLUE + (1 - P0) * R_GREEN
adv_blue, adv_green = R_BLUE - V_game, R_GREEN - V_game
table(["", "抽到蓝键记", "抽到绿键记", "平均", "离平均多远"],
      [["不减基线", fmt(round(blue_share(P0), 10)), fmt(round(green_share(P0), 10)), fmt(round(average_share(P0), 10)), fmt(round(std_plain, 10))],
       [f"减基线 b = {fmt(B)}", fmt(round(blue_b, 10)), fmt(round(green_b, 10)), fmt(round(average_share(P0, B), 10)), fmt(round(std_base, 10))]])
print(f"乘数 d ln π/dp 自己的平均：p = 0.5 时 0.5 × 2 + 0.5 × (−2) = {fmt(round(mult_mean[P0], 10))}；"
      f"p = 0.8 时 0.8 × 1.25 + 0.2 × (−5) = {fmt(round(mult_mean[0.8], 10))}")
print(f"平均分 V = 0.5 × 3 + 0.5 × 1 = {fmt(V_game)}；优势：蓝 {fmt(adv_blue)}，绿 {fmt(adv_green)}，"
      f"按策略加权平均 {fmt(round(P0 * adv_blue + (1 - P0) * adv_green, 10))}")
check("减基线 2：蓝 (1/0.5) × (3 − 2) = 2，绿 (−1/0.5) × (1 − 2) = 2，每次都是 2",
      math.isclose(blue_b, 2) and math.isclose(green_b, 2))
check("平均不变（还是 2）；离平均的距离从 4 变成 0", math.isclose(average_share(P0, B), average_share(P0))
      and math.isclose(std_plain, 4) and math.isclose(std_base, 0, abs_tol=1e-12))
check("乘数 d ln π/dp 的平均是 0：p = 0.5 和 p = 0.8 都是",
      all(math.isclose(v, 0, abs_tol=1e-12) for v in mult_mean.values()) and math.isclose(0.8 * 1.25 + 0.2 * -5, 0, abs_tol=1e-12))
check("基线 2 正是平均分 V；优势 +1、−1，按策略加权平均为 0",
      math.isclose(V_game, 2) and math.isclose(adv_blue, 1) and math.isclose(adv_green, -1)
      and math.isclose(P0 * adv_blue + (1 - P0) * adv_green, 0, abs_tol=1e-12))
check("一次 64 个样本：不减基线的估计一般离 2 差 4/√64 = 0.5；减了基线差 0", math.isclose(std_plain / 8, 0.5))
# p = 0.5 的游戏机：两种样本贡献都离平均 2|2 − b|——基线离平均分多远，抖动就有多大
spread = lambda b: 2 * abs(2 - b)
print("离平均多远 = 2 × |2 − b|：" + "，".join(f"b = {fmt(b)} → {fmt(round(spread(b), 10))}" for b in (0.0, 1.0, B, 3.0, 100.0)))
check("抖动的规律 2|2 − b|：b = 0 → 4，b = 1 → 2，b = 2 → 0，b = 3 → 2，b = 100 → 196（和逐个算出来的一致）",
      all(math.isclose(spread(b), math.sqrt(P0 * (blue_share(P0, b) - 2) ** 2 + (1 - P0) * (green_share(P0, b) - 2) ** 2))
          for b in (0.0, 1.0, B, 3.0, 100.0))
      and [spread(b) for b in (0.0, 1.0, B, 3.0, 100.0)] == [4, 2, 0, 2, 196])
check("自测（11.5 节）：两个键的分数都加 100、不减基线——蓝键 (1/0.5) × 103 = 206，绿键 (−1/0.5) × 101 = −202，"
      "平均 0.5 × 206 + 0.5 × (−202) = 2 还是斜率；离平均 204 = 2 × |102 − 0|，平均分 102",
      math.isclose((1 / P0) * 103, 206) and math.isclose((-1 / (1 - P0)) * 101, -202)
      and math.isclose(P0 * 206 + (1 - P0) * -202, slope) and math.isclose(206 - 2, 204) and math.isclose(-202 - 2, -204)
      and math.isclose(P0 * 103 + (1 - P0) * 101, 102) and math.isclose(2 * abs(102 - 0), 204))
check("字面值：b = 100 时蓝键记 2 × (3 − 100) = −194、绿键记 (−2) × (1 − 100) = 198，都离平均 2 有 196",
      math.isclose(blue_share(P0, 100), -194) and math.isclose(green_share(P0, 100), 198)
      and math.isclose(-194 - 2, -196) and math.isclose(198 - 2, 196))
check("自测 p = 0.8、b = 2：1.25 和 5，平均 0.8 × 1.25 + 0.2 × 5 = 2",
      math.isclose(blue_share(0.8, 2), 1.25) and math.isclose(green_share(0.8, 2), 5)
      and math.isclose(0.8 * 1.25 + 0.2 * 5, 2) and math.isclose(average_share(0.8, 2), 2))
check("自测 b = 100：−121.25 和 495，平均 0.8 × (−121.25) + 0.2 × 495 = −97 + 99 = 2",
      math.isclose(blue_share(0.8, 100), -121.25) and math.isclose(green_share(0.8, 100), 495)
      and math.isclose(0.8 * -121.25, -97) and math.isclose(0.2 * 495, 99) and math.isclose(average_share(0.8, 100), 2))

# 连续动作：11.3 节那个例子（中心 0.3、宽 0.5、分数 −(a − 1)²），基线取平均分 V = J(0.3) = −0.74
N_SAMPLES = 64  # TWEAK-2: 256
ROUNDS = 2000
V_EX = J_ex(MU_EX)
g_base = lambda e: (r_ex(MU_EX + SIG_EX * e) - V_EX) * e / SIG_EX
exact_mean_base = expect(g_base)
exact_std_base = math.sqrt(expect(lambda e: g_base(e) ** 2) - exact_mean_base ** 2)
rng4 = np.random.default_rng(4)
a = MU_EX + SIG_EX * rng4.standard_normal((ROUNDS, N_SAMPLES))
score = (a - MU_EX) / SIG_EX ** 2
est_plain = (r_ex(a) * score).mean(axis=1)
est_base = ((r_ex(a) - V_EX) * score).mean(axis=1)
m_plain, m_base = float(est_plain.mean()), float(est_base.mean())
s_plain, s_base = float(est_plain.std()), float(est_base.std())
th_plain, th_base = exact_std_plain / math.sqrt(N_SAMPLES), exact_std_base / math.sqrt(N_SAMPLES)
table([f"每次 {N_SAMPLES} 个样本，估 {ROUNDS} 次", "估计的平均", "估计的标准差", f"理论：单个的标准差 / √{N_SAMPLES}"],
      [["不减基线", f"{m_plain:.3f}", f"{s_plain:.3f}", f"{exact_std_plain:.3f} / {math.sqrt(N_SAMPLES):g} = {th_plain:.3f}"],
       [f"减基线 V = {fmt(round(V_EX, 10))}", f"{m_base:.3f}", f"{s_base:.3f}", f"{exact_std_base:.3f} / {math.sqrt(N_SAMPLES):g} = {th_base:.3f}"]])
sample_ratio = (exact_std_plain / exact_std_base) ** 2
print(f"不减基线想和减了一样准：样本要多 ({exact_std_plain:.3f} / {exact_std_base:.3f})² = {sample_ratio:.1f} 倍")
check("正文引用：两种估计的平均 1.389 和 1.395（这一组随机数）", round(m_plain, 3) == 1.389 and round(m_base, 3) == 1.395)
check(f"两种估计的平均都在 1.4 附近（离 1.4 不到 4 × 标准差 / √{ROUNDS}）",
      abs(m_plain - 1.4) < 4 * th_plain / math.sqrt(ROUNDS) and abs(m_base - 1.4) < 4 * th_base / math.sqrt(ROUNDS))
check("理论：单个样本贡献的标准差 3.402 → 2.534；平均都还是 1.4",
      round(exact_std_plain, 3) == 3.402 and round(exact_std_base, 3) == 2.534 and math.isclose(exact_mean_base, 1.4, rel_tol=1e-9))
check(f"实测的标准差和理论值（{th_plain:.3f}、{th_base:.3f}）相差不到 10%",
      abs(s_plain / th_plain - 1) < 0.1 and abs(s_base / th_base - 1) < 0.1)
check("正文引用：64 个样本时理论 0.425 → 0.317，实测 0.427 → 0.317（图上 0.43 → 0.32）；样本要多 1.8 倍",
      round(th_plain, 3) == 0.425 and round(th_base, 3) == 0.317 and round(s_plain, 3) == 0.427 and round(s_base, 3) == 0.317
      and round(s_plain, 2) == 0.43 and round(s_base, 2) == 0.32 and round(sample_ratio, 1) == 1.8)
check("字面值：3.402 / 8 = 0.425，2.534 / 8 = 0.317，(3.402 / 2.534)² = 1.8",
      round(3.402 / 8, 3) == 0.425 and round(2.534 / 8, 3) == 0.317 and round((3.402 / 2.534) ** 2, 1) == 1.8)

# 进阶：拿这一批自己的平均当基线（第 5 节就是这么做的），平均会缩成 (N − 1)/N 倍
ROUNDS_BIG = 20_000
a = MU_EX + SIG_EX * rng4.standard_normal((ROUNDS_BIG, N_SAMPLES))
r_big = r_ex(a)
est_batch = ((r_big - r_big.mean(axis=1, keepdims=True)) * (a - MU_EX) / SIG_EX ** 2).mean(axis=1)
shrink = (N_SAMPLES - 1) / N_SAMPLES
print(f"批平均当基线：{ROUNDS_BIG:,} 次估计的平均 = {est_batch.mean():.4f}；1.4 × {N_SAMPLES - 1}/{N_SAMPLES} = {1.4 * shrink:.4f}")
check(f"批平均当基线：平均靠近 1.4 × (N − 1)/N = {1.4 * shrink:.4f}，而不是 1.4",
      abs(est_batch.mean() - 1.4 * shrink) < 0.006 and (N_SAMPLES > 200 or abs(est_batch.mean() - 1.4) > 0.012))
check("正文引用：N = 64 时 63/64 = 0.984；这一组随机数的 2 万次平均是 1.3745",
      round(63 / 64, 3) == 0.984 and round(float(est_batch.mean()), 4) == 1.3745 and round(1.4 * 63 / 64, 4) == 1.3781)

# ---------------------------------------------------------------------------
banner("4b. 画图：figures/ch11_baseline_variance.png（减了基线，样本贡献挤到平均附近）")
fig = plt.figure(figsize=(9.6, 13.2))
fig.suptitle("基线：平均不变，每一次的贡献挤到平均附近", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
ax_top = fig.add_axes([0.15, 0.745, 0.80, 0.115])
ax_bot = fig.add_axes([0.15, 0.575, 0.80, 0.115])
for axk, shares, label in ((ax_top, ((green_share(P0), 1 - P0, GREEN), (blue_share(P0), P0, BLUE)), "不减基线"),
                           (ax_bot, ((green_b, 1 - P0, GREEN), (blue_b, P0, BLUE)), f"减基线 b = {fmt(B)}")):
    data_axes(axk, "", "概率")
    bottom = {}
    for x, prob, color in shares:
        y0 = bottom.get(round(x, 9), 0.0)
        axk.bar([x], [prob], bottom=[y0], width=0.5, color=color, alpha=0.85, zorder=3)
        bottom[round(x, 9)] = y0 + prob
    axk.axvline(average_share(P0), color=ORANGE, ls="--", lw=2.4, zorder=4)
    axk.set_xlim(-3.5, max(7.5, blue_share(P0) + 1.2))
    axk.set_ylim(0, 1.15)
    axk.set_yticks([0, 0.5, 1])
    axk.text(-3.35, 0.93, label, fontsize=FS_SMALL, color=INK, va="center", bbox=WHITE_BOX, zorder=6)
    axk.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−")))
ax_top.text(green_share(P0), 0.56, f"绿键 {fmt(round(green_share(P0), 10))}", ha="center", va="bottom", fontsize=FS_SMALL, color=GREEN)
ax_top.text(blue_share(P0), 0.56, f"蓝键 {fmt(round(blue_share(P0), 10))}", ha="center", va="bottom", fontsize=FS_SMALL, color=BLUE)
ax_top.text(average_share(P0) + 0.12, 0.93, f"平均 {fmt(round(average_share(P0), 10))}", color=ORANGE, fontsize=FS_SMALL, va="center")
ax_bot.text(blue_b + 0.35, 0.5, "蓝、绿都记 2" if math.isclose(blue_b, green_b) else f"蓝 {fmt(round(blue_b, 10))}，绿 {fmt(round(green_b, 10))}",
            fontsize=FS_SMALL, color=INK, va="center")
ax_bot.set_xlabel("每次记的数（样本贡献）", fontsize=FS_SMALL, color=INK)
panel_title(fig, [ax_top, ax_bot], "① 游戏机（p = 0.5）：从“6 或 −2”变成“每次都是 2”")
panel_note(fig, [ax_top, ax_bot], "柱子的高度是这个数出现的概率。橙色虚线是平均：两行都在 2。")

ax = fig.add_axes([0.15, 0.115, 0.80, 0.30])
data_axes(ax, f"一次估计（{N_SAMPLES} 个样本的平均）", f"次数（共 {ROUNDS} 次）")
bins = np.arange(-0.2, 3.05, 0.1)
ax.hist(est_plain, bins=bins, color=FAINT, alpha=0.75, zorder=3)
ax.hist(est_base, bins=bins, histtype="step", color=ORANGE, lw=2.6, zorder=4)
ax.axvline(1.4, color=INK, ls="--", lw=2, zorder=5)
ax.text(1.43, ax.get_ylim()[1] * 0.97, "真值 1.4", color=INK, fontsize=FS_SMALL, va="top", bbox=WHITE_BOX, zorder=6)
ax.text(0.02, 0.62, f"灰色：不减基线\n标准差 {s_plain:.2f}", transform=ax.transAxes, color=MUTED, fontsize=FS_SMALL, va="center",
        linespacing=1.3)
ax.text(0.98, 0.62, f"橙线：减基线 V = −0.74\n标准差 {s_base:.2f}", transform=ax.transAxes, color=ORANGE, fontsize=FS_SMALL,
        va="center", ha="right", linespacing=1.3)
ax.set_xlim(-0.2, 3.0)
panel_title(fig, [ax], f"② 11.3 节的钟（中心 0.3）：每次抽 {N_SAMPLES} 个估一次，估 {ROUNDS} 次")
panel_note(fig, [ax], "两堆的平均都是 1.4（右边拖着尾巴，最高处略偏左）；\n减了基线的那堆窄一截，但窄不成一根线。")
savefig(fig, "ch11_baseline_variance")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("4c. 画图：figures/ch11_overview.png（11.0 节的总览）")
fig, axes = lesson_figure(4, "抽到谁，就按分数把谁往上推一把", panel_height=3.0, width=10.2)
ax = axes[0]
lesson_panel(ax, "① 平均分 J：蓝键 3 分、绿键 1 分，p = 按蓝键的概率", xmax=10.2, ymax=4.0)
cell(ax, 0.3, 1.85, f"蓝键 {fmt(R_BLUE)} 分", width=2.0, height=0.75, edgecolor=BLUE, color=BLUE)
cell(ax, 2.6, 1.85, f"绿键 {fmt(R_GREEN)} 分", width=2.0, height=0.75, edgecolor=GREEN, color=GREEN)
hand(ax, 5.0, 2.55, f"J = {fmt(R_BLUE)} × 0.5 + {fmt(R_GREEN)} × 0.5 = {fmt(round(J(P0), 10))}")
hand(ax, 5.0, 1.75, f"p 加 0.01，J 加 {fmt(round(0.01 * slope, 10))}：斜率 {fmt(slope)}")
note(ax, 0.3, 0.9, "想让 J 变大，就让 p 顺着斜率往上走（梯度上升）。")
ax = axes[1]
lesson_panel(ax, "② 抽到谁，就推谁：（ln 概率对 p 的导数）× 分数", xmax=10.2, ymax=4.0)
hand(ax, 0.3, 2.6, f"抽到蓝键：(1/0.5) × {fmt(R_BLUE)} = {fmt(round(blue_share(P0), 10))}")
hand(ax, 0.3, 1.9, f"抽到绿键：(−1/0.5) × {fmt(R_GREEN)} = {fmt(round(green_share(P0), 10))}", color=GREEN)
hand(ax, 0.3, 1.2, f"平均：0.5 × {fmt(round(blue_share(P0), 10))} + 0.5 × ({fmt(round(green_share(P0), 10))}) = "
     f"{fmt(round(average_share(P0), 10))} = 斜率", color=ORANGE)
note(ax, 0.3, 0.45, "分数只被当成数乘上去，从头到尾没被求导。")
ax = axes[2]
lesson_panel(ax, f"③ 减去基线 {fmt(B)}（平均分）：只按“比平均好多少”推", xmax=10.2, ymax=4.0)
hand(ax, 0.3, 2.45, f"蓝键：2 × ({fmt(R_BLUE)} − {fmt(B)}) = {fmt(round(blue_b, 10))}")
hand(ax, 0.3, 1.6, f"绿键：(−2) × ({fmt(R_GREEN)} − {fmt(B)}) = {fmt(round(green_b, 10))}", color=GREEN)
if math.isclose(blue_b, green_b):
    hand(ax, 6.2, 2.0, "每次都是 2", color=ORANGE)
note(ax, 0.3, 0.7, "平均还是 2，每一次却不再忽大忽小。")
ax = axes[3]
lesson_panel(ax, "④ 连续动作：钟的中心往好动作那边挪", xmax=10.2, ymax=4.0)
xs = np.linspace(-1.3, 1.3, 200)
bx = lambda v: 0.5 + (v + 1.3) / 2.6 * 3.4                 # 动作 −1.3 … 1.3 按比例摆到 0.5 … 3.9
by = lambda d: 0.55 + d / 0.8 * 2.1                        # 密度按比例画高
ax.plot(bx(xs), by(np.exp(-xs ** 2 / (2 * SIGMA ** 2)) / (SIGMA * math.sqrt(2 * math.pi))), color=BLUE, lw=2.6)
ax.plot([bx(0), bx(0)], [0.55, by(1 / (SIGMA * math.sqrt(2 * math.pi)))], color=FAINT, lw=1.6, ls=":")
ax.plot([bx(A_SAMPLE)], [0.55], "o", color=ORANGE, ms=10, clip_on=False)
ax.text(bx(0), 0.4, "μ = 0", fontsize=FS_SMALL - 2, color=MUTED, ha="center", va="top")
ax.text(bx(A_SAMPLE) + 0.08, 0.72, "a = 0.2", fontsize=FS_SMALL - 2, color=ORANGE, ha="left", va="bottom")
hand(ax, 4.25, 2.55, f"(0.2 − 0)/0.5² × 1.5 = {fmt(rule)} × 1.5 = {fmt(round(push, 10))}")
hand(ax, 4.25, 1.75, f"μ：0 + 0.01 × {fmt(round(push, 10))} = {fmt(round(mu_up, 10))}", color=ORANGE)
note(ax, 4.25, 0.95, "这次得 1.5 分：中心往 0.2 那边挪。")
savefig(fig, "ch11_overview")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("5. REINFORCE：推小车——抽一批、算权重、走一步，反复做")
TEST_S = torch.tensor([[-0.8], [0.0], [0.6]])
LOG_ROUNDS = (0, 50, 200, 500, 1500)
SNAP_ROUNDS = (0, 50, 200, 1500)
S_GRID = torch.linspace(-1, 1, 101).unsqueeze(1)


def reinforce(c, rounds, seed=0, baseline=True, offset=0.0):
    """推小车的 REINFORCE。c 是 11.6 节的熵奖励系数（第 5 节 c = 0）。每一轮：抽 256 个、更新一次。

    baseline=False：不减基线，权重直接用分数本身。
    offset：给每一局的分数都加同一个数（哪个动作更好完全没变；记录下来的 reward 已经把它减回去）。
    """
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.ELU(), torch.nn.Linear(16, 1))   # μ_θ(s)：1 → 16 → 1
    log_std = torch.nn.Parameter(torch.zeros(1))                                                  # σ = exp(log_std)，起点 1
    opt = torch.optim.Adam(list(net.parameters()) + [log_std], lr=3e-3)
    hist = {"sigma": [], "reward": [], "rows": {}, "curves": {}}
    for it in range(rounds + 1):
        s = torch.rand(256, 1) * 2 - 1                 # 256 辆小车，各在 (−1, 1) 里随机一个位置
        mu = net(s)
        std = log_std.exp().expand_as(mu)
        with torch.no_grad():                          # 记录的是这一轮更新之前的策略
            hist["sigma"].append(std[0].item())
            if it in LOG_ROUNDS:
                hist["rows"][it] = net(TEST_S).squeeze(1).tolist()
            if it in SNAP_ROUNDS:
                hist["curves"][it] = net(S_GRID).squeeze(1).numpy().copy()
        dist = torch.distributions.Normal(mu, std)     # 256 口钟：中心 μ_θ(s)，宽度 σ
        a = dist.sample()                              # 每口钟抽一个动作（抽样不带梯度）
        r = -((s + a) ** 2).squeeze(1)                 # 推完的位置 s + a，离目标越远扣得越多
        r = r + offset                                 # 对照实验用：给每一局的分数都加同一个常数（默认 0，不起作用）
        logp = dist.log_prob(a).squeeze(1)             # ln π(a|s)
        adv = r - r.mean()                             # 减基线：用这一批的平均分
        adv = adv if baseline else r                   # 对照实验用：不减基线时，权重直接用分数本身
        loss = -(adv.detach() * logp).mean()           # 权重 × ln π，取平均，再取负号
        loss = loss - c * dist.entropy().mean()        # 11.6 节的熵奖励（c = 0 时这一行不起作用）
        opt.zero_grad(); loss.backward(); opt.step()
        hist["reward"].append(r.mean().item() - offset)   # 记录时把加上去的常数减回去，三条曲线才比得了
    hist["net"] = net
    return hist


ROUNDS_LONG = 3000
run0 = reinforce(0.0, ROUNDS_LONG)
rows = [[it, f(run0["reward"][it], 3), f(run0["sigma"][it], 2)] + [f(v, 2) for v in run0["rows"][it]] for it in LOG_ROUNDS]
table(["第几轮", "平均分", "σ", "μ(s = −0.8)", "μ(s = 0)", "μ(s = 0.6)"], rows)
print("最优是 μ(s) = −s：μ(−0.8) → 0.8，μ(0) → 0，μ(0.6) → −0.6。")
last = run0["rows"][1500]
check("1500 轮后学到 μ(s) ≈ −s（三个点误差都 < 0.1）",
      abs(last[0] - 0.8) < 0.1 and abs(last[1]) < 0.1 and abs(last[2] + 0.6) < 0.1)
check("σ 在训练中一路缩小：1 → 0.63（第 200 轮）→ 0.19（第 1500 轮）",
      run0["sigma"][0] == 1.0 and round(run0["sigma"][200], 2) == 0.63 and round(run0["sigma"][1500], 2) == 0.19)
check("正文引用：平均分 −2.111 → −0.37（第 200 轮）→ −0.034；第 1500 轮 μ 是 0.81、0.00、−0.59",
      round(run0["reward"][0], 3) == -2.111 and round(run0["reward"][200], 2) == -0.37 and round(run0["reward"][1500], 3) == -0.034
      and round(last[0], 2) == 0.81 and round(last[1], 2) == 0 and round(last[2], 2) == -0.59)
check("N：项目里一批 4096 只 × 24 步 = 98,304 条", 4096 * 24 == 98_304)
CODE_LINES = ["dist = torch.distributions.Normal(mu, std)", "a = dist.sample()", "r = -((s + a) ** 2).squeeze(1)",
              "logp = dist.log_prob(a).squeeze(1)", "adv = r - r.mean()", "loss = -(adv.detach() * logp).mean()",
              "loss = loss - c * dist.entropy().mean()", "opt.zero_grad(); loss.backward(); opt.step()"]
check("正文 11.5、11.6 节贴的代码行，原样在本实验的 reinforce() 里、顺序一致",
      lines_in_order(Path(__file__).read_text(encoding="utf-8").split("def reinforce(", 1)[1], CODE_LINES))
src_reinforce = Path(__file__).read_text(encoding="utf-8").split("def reinforce(", 1)[1].split("\nROUNDS_LONG", 1)[0]
check("推小车的设置：μ_θ(s) 是 1 → 16 → 1 的小网络，每一轮 256 辆车",
      "torch.nn.Linear(1, 16), torch.nn.ELU(), torch.nn.Linear(16, 1)" in src_reinforce and "torch.rand(256, 1)" in src_reinforce)

# 把基线拿掉：同一个任务，分数整体加 100（哪个动作更好完全没变），看训练曲线差多少
OFFSET = 100.0
SEEDS_BL = (0, 1, 2)
MARKS_BL = (200, 500, 1500)
CONFIGS = (("减基线", True, 0.0), ("不减基线", False, 0.0),
           (f"减基线（分数 + {fmt(OFFSET)}）", True, OFFSET), (f"不减基线（分数 + {fmt(OFFSET)}）", False, OFFSET))
curves_bl = {}
for label, use_b, off in CONFIGS:
    curves_bl[label] = np.mean([reinforce(0.0, 1500, seed=sd, baseline=use_b, offset=off)["reward"] for sd in SEEDS_BL], axis=0)
table(["3 个随机起点平均"] + [f"第 {k} 轮的平均分" for k in MARKS_BL],
      [[label] + [f(curves_bl[label][k], 3) for k in MARKS_BL] for label, _, _ in CONFIGS])
base_gap = abs(curves_bl["减基线"][1500] - curves_bl[f"减基线（分数 + {fmt(OFFSET)}）"][1500])
r0 = curves_bl["减基线"][0]
print(f"第 0 轮的平均分 {f(r0, 3)}：不减基线等于取 b = 0，离平均分 {abs(r0):.3f}；"
      f"加 {OFFSET:g} 以后平均分成了 {OFFSET + r0:.3f}，离 b = 0 有 {OFFSET + r0:.3f}——远了 {(OFFSET + r0) / abs(r0):.1f} 倍")
check("正文引用：第 200 轮 −0.404 / −0.408 / −0.404 / −1.031（3 个随机起点的平均）",
      [round(curves_bl[label][200], 3) for label, _, _ in CONFIGS] == [-0.404, -0.408, -0.404, -1.031])
check("正文引用：第 1500 轮 −0.037 / −0.038 / −0.037 / −0.494",
      [round(curves_bl[label][1500], 3) for label, _, _ in CONFIGS] == [-0.037, -0.038, -0.037, -0.494])
check("减了基线：加不加 100 分，曲线压在一起（第 1500 轮相差不到 0.002）", base_gap < 0.002)
check("不减基线：加了 100 分就学不动了（第 1500 轮比减基线差 10 倍以上）",
      curves_bl[f"不减基线（分数 + {fmt(OFFSET)}）"][1500] < 10 * curves_bl["减基线"][1500])
check("不减基线、原分数：和减基线几乎一样（第 1500 轮相差不到 0.01）——推小车的分数本来就在 0 附近，底很浅",
      abs(curves_bl["不减基线"][1500] - curves_bl["减基线"][1500]) < 0.01)
check("正文引用：3 个起点第 0 轮的平均分是 −1.748；字面值 100 − 1.748 = 98.252，98.252 / 1.748 = 56.2 倍",
      round(r0, 3) == -1.748 and math.isclose(100 - 1.748, 98.252) and round(98.252 / 1.748, 1) == 56.2)

# ---------------------------------------------------------------------------
banner("5b. 画图：figures/ch11_reinforce_policy.png（钟的中心学成 −s；平均分一路往上）")
fig = plt.figure(figsize=(9.6, 12.8))
fig.suptitle("REINFORCE：只靠抽样和分数，钟的中心学成了 μ(s) ≈ −s", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
ax = fig.add_axes([0.15, 0.585, 0.80, 0.29])
data_axes(ax, "观测 s（小车离目标的位置）", "钟的中心 μ(s)")
sg = S_GRID.squeeze(1).numpy()
ax.plot(sg, -sg, color=GREEN, lw=2.4, ls=(0, (4, 3)), zorder=6)
ax.text(-0.97, -0.75, "绿色虚线：最优 μ = −s", color=GREEN, fontsize=FS_SMALL, va="center", bbox=WHITE_BOX, zorder=6)
for it, color, lw in zip(SNAP_ROUNDS, (FAINT, "#8fb3d9", BLUE, ORANGE), (2.4, 2.4, 2.6, 3.2)):
    ax.plot(sg, run0["curves"][it], color=color, lw=lw, zorder=4)
label_at = {0: (0.55, 0.50), 50: (0.6, -0.02), 200: (-0.45, 0.66), 1500: (0.22, -0.85)}
for it, color in zip(SNAP_ROUNDS, (MUTED, "#6f98c4", BLUE, ORANGE)):
    x, y = label_at[it]
    ax.text(x, y, f"第 {it} 轮", color=color, fontsize=FS_SMALL, va="center", bbox=WHITE_BOX, zorder=6)
ax.set_xlim(-1, 1)
ax.set_ylim(-1.05, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}".replace("-", "−")))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}".replace("-", "−")))
panel_title(fig, [ax], "① 四个时刻的策略：一开始是一条乱线，第 1500 轮压到了 −s 上")
panel_note(fig, [ax], "每一轮只有 256 次“推一下、看分数”，没有人告诉它正确答案是 −s。")

ax = fig.add_axes([0.15, 0.115, 0.80, 0.29])
data_axes(ax, "第几轮", "这一轮的平均分")
ax.plot(np.arange(1501), run0["reward"][:1501], color=BLUE, lw=1.6, zorder=3)
for it in (0, 200, 1500):
    ax.plot([it], [run0["reward"][it]], "o", color=ORANGE, ms=9, zorder=5)
ax.text(40, run0["reward"][0], f"第 0 轮 {f(run0['reward'][0], 2)}", color=ORANGE, fontsize=FS_SMALL, va="center", bbox=WHITE_BOX, zorder=6)
ax.text(230, run0["reward"][200] - 0.18, f"第 200 轮 {f(run0['reward'][200], 2)}", color=ORANGE, fontsize=FS_SMALL, va="top",
        bbox=WHITE_BOX, zorder=6)
ax.text(1480, run0["reward"][1500] - 0.2, f"第 1500 轮 {f(run0['reward'][1500], 3)}", color=ORANGE, fontsize=FS_SMALL, va="top",
        ha="right", bbox=WHITE_BOX, zorder=6)
ax.set_xlim(-30, 1530)
ax.set_ylim(-2.5, 0.25)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}".replace("-", "−")))
panel_title(fig, [ax], "② 平均分：从 −2.1 一路升到接近 0")
panel_note(fig, [ax], "分数到不了 0：σ 还在，每一推都带着一点随机，这份代价 11.6 节再算。")
savefig(fig, "ch11_reinforce_policy")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("5c. 画图：figures/ch11_baseline_training.png（给每一局都加 100 分：减了基线的毫无变化，不减的学不动）")
fig = plt.figure(figsize=(9.6, 7.6))
fig.suptitle("同一个任务，分数整体加 100：减了基线，照学不误", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.975)
ax = fig.add_axes([0.135, 0.175, 0.825, 0.60])
data_axes(ax, "第几轮", "这一轮的平均分")
plot_specs = (("减基线", BLUE, 5.0, "-"), ("减基线（分数 + 100）", GREEN, 2.6, (0, (6, 4))),
              ("不减基线", INK, 1.6, (0, (1, 3))), ("不减基线（分数 + 100）", ORANGE, 3.0, "-"))
for label, color, lw, ls in plot_specs:
    key = label.replace("100", fmt(OFFSET))
    ax.plot(np.arange(1501), curves_bl[key], color=color, lw=lw, ls=ls, zorder=4, label=label)
ax.set_xlim(-30, 1530)
ax.set_ylim(-2.3, 0.35)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}".replace("-", "−")))
ax.legend(loc="lower right", fontsize=FS_SMALL, frameon=False, labelcolor="linecolor", handlelength=2.6,
          borderaxespad=1.2, labelspacing=0.5)
key_bad = f"不减基线（分数 + {fmt(OFFSET)}）"
ax.annotate(f"停在 {f(curves_bl[key_bad][1500], 3)}", xy=(1500, curves_bl[key_bad][1500]), xytext=(1180, -1.05),
            color=ORANGE, fontsize=FS_SMALL, ha="center", va="center", zorder=7,
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.8, shrinkA=2, shrinkB=4))
ax.annotate(f"另外三条压在一起，都到 {f(curves_bl['减基线'][1500], 3)}", xy=(1420, curves_bl["减基线"][1420]),
            xytext=(1050, 0.18), color=INK, fontsize=FS_SMALL, ha="center", va="center", zorder=7, bbox=WHITE_BOX,
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8, shrinkA=2, shrinkB=4))
panel_title(fig, [ax], "推小车练 1500 轮，每条曲线是 3 个随机起点的平均")
panel_note(fig, [ax], "加 100 分不改变哪个动作更好，可它把“分数的底”从 1.7 抬到了 98——\n不减基线的那一条被噪声淹没，减了基线的一点没变。")
savefig(fig, "ch11_baseline_training")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("6. 熵奖励：给“还愿意试”一点分，别让钟缩得太快")
WIDTH_1 = math.sqrt(2 * math.pi * math.e)          # 第 4 章 4.9 节：σ = 1 的钟相当于宽 4.13 的均匀分布


def entropy(sigma):
    return math.log(WIDTH_1 * sigma)


H1, Hhalf = entropy(1.0), entropy(0.5)
H_torch = torch.distributions.Normal(0.0, torch.tensor([1.0, 0.5])).entropy().tolist()
H14 = 14 * 1.41894
bonus = 0.01 * H14
dens = lambda a, s: math.exp(-a ** 2 / (2 * s ** 2)) / (s * math.sqrt(2 * math.pi))
d05, d04 = dens(0.2, 0.5), dens(0.2, 0.4)
print(f"熵：σ = 1 → {H1:.4f}，σ = 0.5 → {Hhalf:.4f}（torch：{H_torch[0]:.4f}、{H_torch[1]:.4f}）；差 {H1 - Hhalf:.3f} = ln 2")
print(f"项目开始时 14 口钟：14 × 1.41894 = {H14:.3f}；乘 0.01 = {bonus:.3f}")
print(f"a = 0.2 处的密度：σ = 0.5 时 {d05:.3f}，σ = 0.4 时 {d04:.3f}（收窄，离中心近的动作更常见）")
check("熵 ln(4.13σ)：σ = 1 → 1.4189，σ = 0.5 → 0.7258，和 torch 一致；σ 减半，熵少 0.693",
      round(H1, 4) == 1.4189 and round(Hhalf, 4) == 0.7258 and math.isclose(H_torch[0], H1, rel_tol=1e-6)
      and math.isclose(H_torch[1], Hhalf, rel_tol=1e-6) and round(H1 - Hhalf, 3) == 0.693)
check("14 × 1.41894 = 19.865；0.01 × 19.865 = 0.199", round(H14, 3) == 19.865 and round(bonus, 3) == 0.199)
check("字面值：ln 4.1327 = 1.4189，ln 2 = 0.6931，1.4189 − 0.6931 = 0.7258；√(2π) = 2.5066",
      round(math.log(4.1327), 4) == 1.4189 and round(math.log(2), 4) == 0.6931 and round(1.4189 - 0.6931, 4) == 0.7258
      and round(math.sqrt(2 * math.pi), 4) == 2.5066)
check("a = 0.2 处：σ = 0.5 时密度 0.737，σ = 0.4 时 0.880", round(d05, 3) == 0.737 and round(d04, 3) == 0.880)
check("字面值：exp(−0.08) / (0.5 × 2.5066) = 0.737，exp(−0.125) / (0.4 × 2.5066) = 0.880",
      round(math.exp(-0.08) / (0.5 * 2.5066), 3) == 0.737 and round(math.exp(-0.125) / (0.4 * 2.5066), 3) == 0.880)

pushes = {s: 0.01 / s for s in (1.0, 0.1, 0.01)}
print(f"熵奖励对 σ 的推力 c/σ（c = 0.01）：σ = 1 → {pushes[1.0]:g}，σ = 0.1 → {pushes[0.1]:g}，σ = 0.01 → {pushes[0.01]:g}")
check("c/σ：c = 0.01 时，σ = 1 推 0.01，σ = 0.1 推 0.1，σ = 0.01 推 1（σ 缩到十分之一，推力大十倍）",
      math.isclose(pushes[1.0], 0.01) and math.isclose(pushes[0.1], 0.1) and math.isclose(pushes[0.01], 1.0))

ENTROPY_COEF = 0.1  # TWEAK-3: 0.5
balance = lambda c: math.sqrt(c / 2)
print(f"两股力相等 2σ = c/σ → σ = √(c/2)：c = 0.1 → {balance(0.1):.4f}，c = 0.01 → {balance(0.01):.4f}，c = 0.02 → {balance(0.02):g}")
check("√(0.1/2) = 0.2236，√(0.01/2) = 0.0707；自测 c = 0.02 → √0.01 = 0.1，两股力 2 × 0.1 = 0.2 和 0.02/0.1 = 0.2",
      round(balance(0.1), 4) == 0.2236 and round(balance(0.01), 4) == 0.0707 and math.isclose(balance(0.02), 0.1)
      and math.isclose(2 * balance(0.02), 0.2) and math.isclose(0.02 / balance(0.02), 0.2))
check("字面值：2 × 0.2236 = 0.4472，0.1 / 0.2236 = 0.4472（两股力在这里相等）",
      math.isclose(2 * 0.2236, 0.4472) and round(0.1 / 0.2236, 4) == 0.4472)

run_small = reinforce(0.01, ROUNDS_LONG)
run_big = reinforce(ENTROPY_COEF, ROUNDS_LONG)
runs = [(0.0, run0), (0.01, run_small), (ENTROPY_COEF, run_big)]
rows = []
for c, run in runs:
    tail = np.mean(run["reward"][-500:])
    rows.append([fmt(c), f(run["sigma"][1500], 3), f(run["sigma"][ROUNDS_LONG], 3),
                 "一直往下" if c == 0 else f(balance(c), 4), f(tail, 3), f(-run["sigma"][ROUNDS_LONG] ** 2, 3)])
table(["熵系数 c", "第 1500 轮的 σ", f"第 {ROUNDS_LONG} 轮的 σ", "停在 √(c/2)", "最后 500 轮平均分", "−σ²"], rows)
sig_end = {c: run["sigma"][ROUNDS_LONG] for c, run in runs}
check(f"c = {fmt(ENTROPY_COEF)}：σ 停在 √(c/2) = {balance(ENTROPY_COEF):.3f} 附近（相差不到 8%）",
      abs(sig_end[ENTROPY_COEF] / balance(ENTROPY_COEF) - 1) < 0.08)
check("c = 0：3000 轮后 σ 缩到 0.10，还在往下走", round(sig_end[0.0], 2) == 0.10 and sig_end[0.0] < run0["sigma"][2500])
check("c = 0.01：比 c = 0 略宽，还没降到 0.0707（它离停下的地方还远）", sig_end[0.0] < sig_end[0.01] and sig_end[0.01] > balance(0.01))
check("正文引用：c = 0.1 停在 0.225（图上 0.22），平均分 −0.051 ≈ −0.225²；三条线在前 200 轮几乎重合（σ 相差不到 0.01）",
      round(run_big["sigma"][ROUNDS_LONG], 3) == 0.225 and round(float(np.mean(run_big["reward"][-500:])), 3) == -0.051
      and round(0.225 ** 2, 4) == 0.0506
      and all(abs(run["sigma"][it] - run0["sigma"][it]) < 0.01 for _, run in runs for it in range(201)))
check("字面值：第 1500 轮 σ = 0.19，0.19² = 0.0361 ≈ 0.036，和平均分 −0.034 差不多", round(0.19 ** 2, 3) == 0.036)
check("推小车的平均分最多到 −σ²：最后 500 轮的平均分和 −σ² 相差不到 0.01（三条都是）",
      all(abs(np.mean(run["reward"][-500:]) + run["sigma"][ROUNDS_LONG] ** 2) < 0.01 for _, run in runs))

# 两座山的老虎机：小山在 a = 0（1 分），大山在 a = 1.5（2 分），两座山都是 0.25 宽；起点 μ = 0、σ = 0.5
PEAK_A, PEAK_W, PEAK_BIG = 1.5, 0.25, 2.0
SIG0 = 0.5


def hill_reward(a):
    """两座山：小山在 0（高 1），大山在 1.5（高 2）。都是 0.25 宽的钟形（第 1 章 1.9 节的钟形打分）。"""
    return torch.exp(-a ** 2 / (2 * PEAK_W ** 2)) + PEAK_BIG * torch.exp(-(a - PEAK_A) ** 2 / (2 * PEAK_W ** 2))


def bandit(c, rounds=3000, seed=0):
    """一个旋钮的老虎机：没有观测，策略就是一口钟（中心 μ、宽度 σ），每一轮抽 256 个动作、更新一次。"""
    torch.manual_seed(seed)
    mu = torch.nn.Parameter(torch.tensor([0.0]))
    log_std = torch.nn.Parameter(torch.tensor([math.log(SIG0)]))
    opt = torch.optim.Adam([mu, log_std], lr=3e-3)
    h = {"mu": [], "sigma": [], "reward": []}
    for _ in range(rounds + 1):
        std = log_std.exp()
        dist = torch.distributions.Normal(mu.expand(256), std.expand(256))
        a = dist.sample()
        r = hill_reward(a)
        adv = r - r.mean()                                          # 还是 11.4 节的基线：这一批的平均分
        loss = -(adv.detach() * dist.log_prob(a)).mean() - c * dist.entropy().mean()
        h["mu"].append(mu.item()); h["sigma"].append(std.item()); h["reward"].append(r.mean().item())
        opt.zero_grad(); loss.backward(); opt.step()
    return {k: np.array(v) for k, v in h.items()}


HILL_CS = (0.0, 0.3, 0.5)
hills = {c: bandit(c) for c in HILL_CS}
table(["熵系数 c", "第 300 轮的 σ", "第 3000 轮的 σ", "第 3000 轮的 μ", "最后 300 轮平均分", "落在哪座山"],
      [[fmt(c), f(hills[c]["sigma"][300], 3), f(hills[c]["sigma"][3000], 3), f(hills[c]["mu"][3000], 2),
        f(float(np.mean(hills[c]["reward"][-300:])), 3),
        "小山（1 分）" if abs(hills[c]["mu"][3000]) < 0.5 else ("大山（2 分）" if hills[c]["sigma"][3000] < 1 else "哪座也没停住")]
       for c in HILL_CS])
dens_far = {s: math.exp(-PEAK_A ** 2 / (2 * s ** 2)) / (s * math.sqrt(2 * math.pi)) for s in (0.5, 0.1)}
hit = {s: 0.5 * (math.erfc((PEAK_A - PEAK_W) / (s * math.sqrt(2))) - math.erfc((PEAK_A + PEAK_W) / (s * math.sqrt(2)))) for s in (0.5, 0.1)}
print(f"中心还在 0 时，a = {PEAK_A:g} 处的密度：σ = 0.5 → {dens_far[0.5]:.5f}；σ = 0.1 → {dens_far[0.1]:.3g}")
print(f"一轮 256 个样本里，落在大山上（1.25 ≤ a ≤ 1.75）的平均个数：σ = 0.5 → {256 * hit[0.5]:.2f}；σ = 0.1 → {256 * hit[0.1]:.1e}")
check("c = 0：σ 缩到 0.022，μ 一直在 0（卡在小山），平均分 1.00",
      round(hills[0.0]["sigma"][3000], 3) == 0.022 and abs(hills[0.0]["mu"][3000]) < 0.1
      and round(float(np.mean(hills[0.0]["reward"][-300:])), 2) == 1.00)
check("c = 0.3：σ 先被顶到 1.025（第 300 轮），μ 挪到 1.50（大山），σ 再收到 0.112，平均分 1.827",
      round(hills[0.3]["sigma"][300], 3) == 1.025 and round(hills[0.3]["mu"][3000], 2) == 1.50
      and round(hills[0.3]["sigma"][3000], 3) == 0.112 and round(float(np.mean(hills[0.3]["reward"][-300:])), 3) == 1.827)
check("c = 0.5：σ 一直往上涨（第 3000 轮 6946），μ 停在 1.02 两山之间，平均分 0.000——一直乱试，定不下来",
      round(hills[0.5]["sigma"][3000], 0) == 6946 and hills[0.5]["sigma"][3000] > hills[0.5]["sigma"][1000]
      and round(hills[0.5]["mu"][3000], 2) == 1.02 and round(float(np.mean(hills[0.5]["reward"][-300:])), 3) == 0.000)
check("三条都从 σ = 0.5、μ = 0 起步（小山的山顶）", all(h["sigma"][0] == SIG0 and h["mu"][0] == 0.0 for h in hills.values()))
check("正文表里的六个数：第 300 轮的 σ 是 0.191 / 1.025 / 1.132；第 3000 轮的 μ 是 0.00 / 1.50 / 1.02",
      [round(hills[c]["sigma"][300], 3) for c in HILL_CS] == [0.191, 1.025, 1.132]
      and [round(hills[c]["mu"][3000], 2) for c in HILL_CS] == [0.0, 1.50, 1.02])
check("正文表里的平均分：0.996 / 1.827 / 0.000",
      [round(float(np.mean(hills[c]["reward"][-300:])), 3) for c in HILL_CS] == [0.996, 1.827, 0.000])
check("字面值：σ 缩到 0.022 之后，大山 1.5 在 1.5 / 0.022 = 68 个 σ 之外", round(1.5 / 0.022) == 68)
check("推小车里熵奖励纯属成本：c = 0 最后 500 轮 −0.013，c = 0.1 掉到 −0.051",
      round(float(np.mean(run0["reward"][-500:])), 3) == -0.013 and round(float(np.mean(run_big["reward"][-500:])), 3) == -0.051)
check("字面值：a = 1.5 处的密度，σ = 0.5 时 exp(−4.5)/(0.5 × 2.5066) = 0.00886；σ = 0.1 时 exp(−112.5)/(0.1 × 2.5066) = 5.5e−49",
      round(dens_far[0.5], 5) == 0.00886 and float(f"{dens_far[0.1]:.1e}") == 5.5e-49
      and round(math.exp(-4.5) / (0.5 * 2.5066), 5) == 0.00886 and round(1.5 ** 2 / (2 * 0.5 ** 2), 1) == 4.5
      and round(1.5 ** 2 / (2 * 0.1 ** 2), 1) == 112.5)
check("σ = 0.5 时一轮 256 个里平均有 1.53 个落在大山上；σ = 0.1 时是 9.6e−34 个（等于永远抽不到）",
      round(256 * hit[0.5], 2) == 1.53 and float(f"{256 * hit[0.1]:.1e}") == 9.6e-34)
print(f"两个密度之比：{dens_far[0.5]:.5f} / {dens_far[0.1]:.3g} = {dens_far[0.5] / dens_far[0.1]:.1e}")
check("正文引用：c = 0 的钟练到第 600 轮左右 σ 就到 0.1 附近（实测 %.3f）" % hills[0.0]["sigma"][600],
      abs(hills[0.0]["sigma"][600] - 0.1) < 0.012)
check("字面值：两个密度差 0.00886 / (5.5e−49) = 1.6e46 倍（直接算出来也是 1.6e46）",
      float(f"{0.00886 / 5.5e-49:.1e}") == 1.6e46 and float(f"{dens_far[0.5] / dens_far[0.1]:.1e}") == 1.6e46)
near = 0.5 * (math.erfc((0.5 - PEAK_W) / (SIG0 * math.sqrt(2))) - math.erfc((0.5 + PEAK_W) / (SIG0 * math.sqrt(2))))
print(f"自测：大山挪到 a = 0.5（1 个 σ）时，一轮 256 个里平均有 {256 * near:.0f} 个落在 0.25 到 0.75 之间，"
      f"是原来 {256 * hit[0.5]:.2f} 个的 {near / hit[0.5]:.0f} 倍")
check("自测：大山挪到 a = 0.5 就在 1 个 σ 处（0.5 / 0.5 = 1），256 个里约 62 个落在它那一带，是原来的 40 倍",
      math.isclose(0.5 / SIG0, 1.0) and round(256 * near) == 62 and round(near / hit[0.5]) == 40)

# ---------------------------------------------------------------------------
banner("6b. 画图：figures/ch11_entropy_sigma.png（σ 停在两股力相等的地方）")
fig = plt.figure(figsize=(9.6, 12.8))
fig.suptitle("熵奖励：σ 越小往回推得越狠，最后停在两股力相等处", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
ax = fig.add_axes([0.15, 0.585, 0.80, 0.29])
data_axes(ax, "第几轮", "σ")
styles = {0.0: (INK, 2.2, "c = 0"), 0.01: (BLUE, 2.4, "c = 0.01（系数和项目相同）"), ENTROPY_COEF: (ORANGE, 2.8, f"c = {fmt(ENTROPY_COEF)}")}
for c, run in runs:
    color, lw, label = styles[c]
    ax.plot(np.arange(ROUNDS_LONG + 1), run["sigma"], color=color, lw=lw, zorder=4, label=label)
ax.legend(loc="upper right", fontsize=FS_SMALL, frameon=False, labelcolor="linecolor", handlelength=1.6)
for c, color in ((ENTROPY_COEF, ORANGE), (0.01, BLUE)):
    ax.axhline(balance(c), color=color, ls="--", lw=1.8, zorder=3)
ax.text(60, balance(ENTROPY_COEF) + 0.012, f"√({fmt(ENTROPY_COEF)}/2) = {balance(ENTROPY_COEF):.3f}", color=ORANGE,
        fontsize=FS_SMALL, ha="left", va="bottom", zorder=6)
ax.text(60, balance(0.01) + 0.012, f"√(0.01/2) = {balance(0.01):.4f}", color=BLUE, fontsize=FS_SMALL, ha="left", va="bottom", zorder=6)
ax.set_xlim(0, ROUNDS_LONG)
ax.set_ylim(0, 1.05)
panel_title(fig, [ax], f"① 推小车练 3000 轮：c = 0 一路缩，c = {fmt(ENTROPY_COEF)} 停在 {sig_end[ENTROPY_COEF]:.2f}")
panel_note(fig, [ax], "推小车里，c = 0.01 在前 3000 轮几乎看不出作用：要等 σ 小到 0.07 附近才顶得住。")

ax = fig.add_axes([0.15, 0.115, 0.80, 0.29])
data_axes(ax, "σ", "对 σ 的推力（导数的大小）")
ss = np.linspace(0.02, 1.0, 300)
ax.plot(ss, 2 * ss, color=GREEN, lw=3, zorder=4)
ax.plot(ss, ENTROPY_COEF / ss, color=ORANGE, lw=2.8, zorder=4)
ax.plot(ss, 0.01 / ss, color=BLUE, lw=2.4, zorder=4)
for c, color in ((ENTROPY_COEF, ORANGE), (0.01, BLUE)):
    x = balance(c)
    ax.plot([x], [2 * x], "o", color=color, ms=10, zorder=6)
    ax.plot([x, x], [0, 2 * x], color=color, ls=":", lw=1.6, zorder=3)
ax.text(0.58, 0.78, f"橙点：σ = {balance(ENTROPY_COEF):.3f}（c = {fmt(ENTROPY_COEF)}）", color=ORANGE, fontsize=FS_SMALL, va="center", zorder=7)
ax.text(0.58, 0.60, f"蓝点：σ = {balance(0.01):.4f}（c = 0.01）", color=BLUE, fontsize=FS_SMALL, va="center", zorder=7)
ax.text(0.62, 1.85, "分数往小里拉：2σ", color=GREEN, fontsize=FS_SMALL, ha="center", va="bottom", bbox=WHITE_BOX, zorder=7)
ax.text(0.24, 1.25, f"熵往大里推：{fmt(ENTROPY_COEF)}/σ", color=ORANGE, fontsize=FS_SMALL, va="center", bbox=WHITE_BOX, zorder=7)
ax.annotate("熵往大里推：0.01/σ", xy=(0.80, 0.0125), xytext=(0.55, 0.36), color=BLUE, fontsize=FS_SMALL, va="center", zorder=7,
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6, shrinkA=3, shrinkB=0))
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 2.1)
panel_title(fig, [ax], "② 两股力：推小车的分数把 σ 往小拉，熵奖励往大推")
panel_note(fig, [ax], "交点就是 σ 停下的地方：2σ = c/σ，σ = √(c/2)。c 越大，停得越宽。")
savefig(fig, "ch11_entropy_sigma")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("6c. 画图：figures/ch11_entropy_escape.png（两座山：c = 0 卡在小山，c = 0.3 翻过去，c = 0.5 定不下来）")
fig = plt.figure(figsize=(9.6, 14.6))
fig.suptitle("熵奖励买的是什么：钟缩得太快，就再也找不到那座大山", fontsize=FS_TITLE - 1, fontweight="bold", color=INK, y=0.988)
HILL_STYLE = {0.0: (INK, "c = 0"), 0.3: (ORANGE, "c = 0.3"), 0.5: ("#7f95a8", "c = 0.5")}
sig_end0 = hills[0.0]["sigma"][3000]

ax = fig.add_axes([0.15, 0.700, 0.80, 0.225])
data_axes(ax, "动作 a", "分数")
aa = np.linspace(-1.2, 2.6, 400)
ax.plot(aa, np.exp(-aa ** 2 / (2 * PEAK_W ** 2)) + PEAK_BIG * np.exp(-(aa - PEAK_A) ** 2 / (2 * PEAK_W ** 2)),
        color=GREEN, lw=3, zorder=5)
ax.text(0.0, 1.06, "小山：1 分", color=GREEN, fontsize=FS_SMALL, ha="center", va="bottom", bbox=WHITE_BOX, zorder=7)
ax.text(PEAK_A, 2.06, "大山：2 分", color=GREEN, fontsize=FS_SMALL, ha="center", va="bottom", bbox=WHITE_BOX, zorder=7)
ax.plot(aa, np.exp(-aa ** 2 / (2 * SIG0 ** 2)) / (SIG0 * math.sqrt(2 * math.pi)) * 0.5, color=BLUE, lw=2.4, zorder=4)
ax.text(-1.15, 1.62, "起点的钟：μ = 0、σ = 0.5\n（高度按 0.5 倍画，只看位置和胖瘦，别读纵轴）", color=BLUE, fontsize=FS_SMALL,
        va="center", linespacing=1.35, zorder=7)
ax.plot([0, 0], [0, 2.45], color=MUTED, lw=2.0, ls=(0, (5, 3)), zorder=6)
ax.text(0.06, 2.42, f"c = 0 练完的钟：σ = {sig_end0:.3f}，细成一根线（高度顶出图外）", color=MUTED, fontsize=FS_SMALL,
        ha="left", va="top", zorder=7)
ax.annotate("", xy=(PEAK_A, 0.30), xytext=(0, 0.30), zorder=6,
            arrowprops=dict(arrowstyle="<|-|>", lw=2.2, color=ORANGE, shrinkA=0, shrinkB=0))
ax.text(PEAK_A / 2 + 0.05, 0.52, f"中心到大山 = 1.5 = 3 个 σ\n256 个里平均 {256 * hit[0.5]:.2f} 个落在大山上",
        color=ORANGE, fontsize=FS_SMALL, ha="center", va="bottom", linespacing=1.35, bbox=WHITE_BOX, zorder=7)
ax.set_xlim(-1.2, 2.6)
ax.set_ylim(0, 2.55)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−")))
panel_title(fig, [ax], "① 这台老虎机：小山 1 分，大山 2 分，起点的钟坐在小山上")
panel_note(fig, [ax], f"σ 一旦缩到 {sig_end0:.3f}，大山就在 1.5 ÷ {sig_end0:.3f} = {PEAK_A / round(sig_end0, 3):.0f} 个 σ 之外——再也抽不到。")

ax = fig.add_axes([0.15, 0.410, 0.80, 0.150])
data_axes(ax, "", "σ")
for c in HILL_CS:
    color, label = HILL_STYLE[c]
    ax.plot(np.arange(3001), hills[c]["sigma"], color=color, lw=2.8, zorder=4, label=label)
ax.set_yscale("log")
ax.set_yticks([0.01, 0.1, 1, 10, 100, 1000, 10000])
ax.set_yticklabels(["0.01", "0.1", "1", "10", "100", "1000", "10000"])
ax.set_ylim(0.012, 30000)
ax.set_xlim(0, 3000)
ax.legend(loc="upper left", fontsize=FS_SMALL, frameon=False, labelcolor="linecolor", handlelength=1.8, ncol=3,
          columnspacing=2.4)
panel_title(fig, [ax], "② σ：c = 0 一路缩到 0.02，c = 0.3 先撑开再收住，c = 0.5 涨到几千")
panel_note(fig, [ax], "纵轴是对数刻度：往上一格是 10 倍，不是加 10——不这么画，0.02 和 6946 放不进同一幅图。")

ax = fig.add_axes([0.15, 0.105, 0.80, 0.165])
data_axes(ax, "第几轮", "这一轮的平均分")
for c in HILL_CS:
    ax.plot(np.arange(3001), hills[c]["reward"], color=HILL_STYLE[c][0], lw=2.6, zorder=4)
for y, lab in ((1.0, "小山 1 分"), (2.0, "大山 2 分")):
    ax.axhline(y, color=GREEN, ls=":", lw=1.8, zorder=3)
    ax.text(2970, y + 0.05, lab, color=GREEN, fontsize=FS_SMALL, ha="right", va="bottom", bbox=WHITE_BOX, zorder=7)
ax.text(1700, 1.63, "c = 0.3", color=ORANGE, fontsize=FS_SMALL, va="center", bbox=WHITE_BOX, zorder=7)
ax.text(1700, 1.12, "c = 0", color=INK, fontsize=FS_SMALL, va="center", bbox=WHITE_BOX, zorder=7)
ax.text(1700, 0.18, "c = 0.5", color=HILL_STYLE[0.5][0], fontsize=FS_SMALL, va="center", bbox=WHITE_BOX, zorder=7)
ax.set_xlim(0, 3000)
ax.set_ylim(-0.12, 2.4)
panel_title(fig, [ax], "③ 平均分：c = 0 停在 1 分，c = 0.3 爬到 1.83，c = 0.5 一直在 0 附近")
panel_note(fig, [ax], "c = 0.3 停在 1.83 而不是 2：σ = 0.11 的钟还有一点宽，\n抽到的动作不全在山顶上——这点差价就是“还愿意试”的租金。")
savefig(fig, "ch11_entropy_escape")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("7. 映射到项目：正文引用的常数和源码行还在不在")
REPO = Path(__file__).resolve().parents[3]
cfg_path = REPO / "src" / "mjlab_microduck" / "tasks" / "microduck_velocity_env_cfg.py"
CFG_LINES = ["distribution_cfg={", '"class_name": "GaussianDistribution",', '"init_std": 1.0,', '"std_type": "scalar",',
             "entropy_coef=0.01,"]
if cfg_path.is_file():
    cfg_text = cfg_path.read_text(encoding="utf-8")
    at = cfg_text.find("MicroduckRlCfg = RslRlOnPolicyRunnerCfg(")
    print("microduck_velocity_env_cfg.py 的 MicroduckRlCfg：", "  ".join(CFG_LINES))
    check("项目配置：actor 输出高斯钟、σ 起点 1.0、σ 直接当旋钮、熵系数 0.01", at >= 0 and lines_in_order(cfg_text[at:], CFG_LINES))
else:
    print("  （没找到项目的 env cfg，跳过这一项）")

rsl_spec = importlib.util.find_spec("rsl_rl")
rsl_dir = Path(rsl_spec.origin).parent if rsl_spec and rsl_spec.origin else None
if rsl_dir and (rsl_dir / "algorithms" / "ppo.py").is_file():
    dist_text = (rsl_dir / "modules" / "distribution.py").read_text(encoding="utf-8")
    DIST_LINES = ["self._distribution = Normal(mean, std)", "return self._distribution.sample()",
                  "return self._distribution.entropy().sum(dim=-1)", "return self._distribution.log_prob(outputs).sum(dim=-1)"]
    at = dist_text.find("class GaussianDistribution(")
    check("rsl_rl/modules/distribution.py 的 GaussianDistribution：搭钟、抽样、熵（14 口相加）、ln 密度（14 个相加）",
          at >= 0 and lines_in_order(dist_text[at:], DIST_LINES))
    ppo_text = (rsl_dir / "algorithms" / "ppo.py").read_text(encoding="utf-8")
    PPO_LINES = ["self.transition.actions = self.actor(obs, stochastic_output=True).detach()",
                 "self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()",
                 "actions_log_prob = self.actor.get_output_log_prob(batch.actions)",
                 "entropy = self.actor.output_entropy[:original_batch_size]",
                 "ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))",
                 "surrogate = -torch.squeeze(batch.advantages) * ratio",
                 "loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()"]
    check(f"rsl_rl/algorithms/ppo.py：正文映射块引用的 {len(PPO_LINES)} 行原样存在，顺序一致（act() 在前，update() 在后）",
          lines_in_order(ppo_text, PPO_LINES) and ppo_text.find("    def act(") < ppo_text.find(PPO_LINES[0])
          and ppo_text.find("    def update(") < ppo_text.find(PPO_LINES[2]))
    check("日志里记的熵就是熵本身（没乘 0.01）：update() 汇总 entropy.mean()，以 \"entropy\" 这个名字交出去",
          "mean_entropy += entropy.mean().item()" in ppo_text and '"entropy": mean_entropy,' in ppo_text)
    logger_text = (rsl_dir / "utils" / "logger.py").read_text(encoding="utf-8")
    check("训练日志：终端打印 Mean action std: 和 Mean entropy loss:（Mean {key} loss:），wandb 里是 Policy/mean_std 和 Loss/entropy",
          '"Mean action std:"' in logger_text and '"Policy/mean_std"' in logger_text and 'f"Loss/{key}"' in logger_text
          and 'f"Mean {key} loss:"' in logger_text and "Mean noise std" not in logger_text)
else:
    print("  （当前 Python 环境里没有 rsl_rl，跳过源码核对；用 uv run 运行就会核对）")

# 两个 torch 事实：sample() 不带梯度（rsample() 才带）；在“新 = 旧”那一刻，−Â × 比率 和 −Â × ln π 的梯度一模一样
mu_t = torch.tensor([0.1, -0.3, 0.2], requires_grad=True)
bell_t = torch.distributions.Normal(mu_t, 0.5)
print(f"sample() 带梯度吗：{bell_t.sample().requires_grad}；rsample() 呢：{bell_t.rsample().requires_grad}")
check("sample() 抽出来的数不带梯度，rsample()（重参数化版）带", not bell_t.sample().requires_grad and bell_t.rsample().requires_grad)
acts = torch.tensor([0.4, -0.1, 0.0])
adv_t = torch.tensor([1.5, -0.5, 0.3])
logp_t = torch.distributions.Normal(mu_t, 0.5).log_prob(acts)
g_reinforce = torch.autograd.grad(-(adv_t * logp_t).mean(), mu_t, retain_graph=True)[0]
g_ratio = torch.autograd.grad(-(adv_t * torch.exp(logp_t - logp_t.detach())).mean(), mu_t)[0]
print(f"−Â × ln π 的梯度：{g_reinforce.numpy()}；−Â × exp(ln π − 旧 ln π) 的梯度：{g_ratio.numpy()}")
check("更新前（新 = 旧，比率 = 1）：PPO 的 −Â × 比率 和本章的 −Â × ln π 梯度相同", torch.allclose(g_reinforce, g_ratio))

done()
