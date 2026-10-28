"""第 4 章实验：分布、密度、期望、方差、大数定律、高斯分布、策略的 14 口钟、log 概率、熵。

运行：uv run python docs/learn-zh/labs/ch04_probability.py
纯 CPU，numpy + torch + matplotlib。第 10 节只读几份源码的文字，不加载机器人、不训练。
小节编号与正文一一对应：实验第 K 节 = 正文 4.K 节（第 10 节对应「映射到项目」）。
随机实验都固定了种子，每次运行打印的数字相同。
正文“改一改”要改的三行都带 `# TWEAK-k:` 标记（第 2、6、7 节各一处）。
"""

import importlib.util
import math
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import torch

import _common
from _common import banner, check, done, lines_in_order, num, savefig, table
from _draw import (BLUE, CELL_EDGE, FAINT, FS_NOTE, FS_SMALL, FS_STEP, FS_TICK, FS_TITLE, GREEN, INK, MUTED, ORANGE,
                   WHITE_BOX, arrow, cell, data_axes, hand, lesson_cells, lesson_figure, lesson_panel, note, panel_note,
                   panel_title, plt)

np.set_printoptions(precision=4, suppress=True)
SQRT_2PI = math.sqrt(2 * math.pi)


def finite(*values) -> bool:
    """画图之前先问一句：这些数都是有限的吗（不是 inf / nan）？"""
    return all(bool(np.all(np.isfinite(np.asarray(v, dtype=float)))) for v in values)


# ---------------------------------------------------------------------------
banner("1. 分布：说不准下一次，说得清很多次（骰子、转盘）")
faces = np.arange(1, 7)
p_dice = np.full(6, 1 / 6)
print("骰子的分布：每个点数的概率")
table(["点数 x", *[str(v) for v in faces]], [["概率 p(x)", *["1/6"] * 6]])
print("六个概率加起来 =", num(float(p_dice.sum())))
check("骰子：六个概率都是 1/6，加起来是 1", math.isclose(float(p_dice.sum()), 1.0) and np.allclose(p_dice, 1 / 6))

N_ROLLS = 6000
rolls = np.random.default_rng(1).integers(1, 7, size=N_ROLLS)          # 固定种子：每次运行掷出同一串点数
counts = np.bincount(rolls, minlength=7)[1:]
print(f"\n真的掷 {N_ROLLS} 次（种子固定）。前 12 次：", " ".join(str(v) for v in rolls[:12]))
table(["点数", "出现次数", "占比"], [[int(v), int(c), c / N_ROLLS] for v, c in zip(faces, counts)], floatfmt=".3f")
check("下一次是几说不准：前 12 次里没有规律可循（至少出现了 4 种不同的点数）", len(set(rolls[:12].tolist())) >= 4)
check("正文引用的数：前 12 次是 3 4 5 6 1 1 5 6 2 2 6 3", rolls[:12].tolist() == [3, 4, 5, 6, 1, 1, 5, 6, 2, 2, 6, 3])
check("很多次却说得清：6000 次里每个点数的占比都在 1/6 ≈ 0.167 上下 0.01 以内", bool(np.all(np.abs(counts / N_ROLLS - 1 / 6) < 0.01)))
check("正文引用的数：6000 次里 1 点出现 1031 次、4 点出现 965 次", counts[0] == 1031 and counts[3] == 965)

wheel_prize = np.array([0.0, 5.0, 20.0])          # 教学构造：一只抽奖转盘，三种奖金（元）
wheel_p = np.array([0.5, 0.3, 0.2])               # 各自的概率：不相等，但加起来还是 1
print("\n概率不必相等。一只抽奖转盘（教学构造）：")
table(["奖金（元）", *[num(v) for v in wheel_prize]], [["概率", *[num(v) for v in wheel_p]]])
check("转盘：三个概率 0.5、0.3、0.2 加起来是 1", math.isclose(float(wheel_p.sum()), 1.0))

# ---------------------------------------------------------------------------
banner("2. 连续的量：均匀分布；高度是密度，面积才是概率")
# 项目给每只仿真机器人抽的电池电压范围（伏）。“改一改”第 1 条改这一行。
VIN_RANGE = (6.5, 8.2)  # TWEAK-1: (6.9, 7.9)
lo, hi = VIN_RANGE
width = hi - lo
density = 1 / width
seg_lo, seg_hi = 7.0, 7.5
p_seg = (seg_hi - seg_lo) / width
print(f"电压 X ~ U({lo}, {hi})：区间宽 {num(width)} V，每个值机会均等")
print(f"  落在 {seg_lo} 到 {seg_hi} 这半伏里的概率 = {num(seg_hi - seg_lo)} / {num(width)} = {p_seg:.3f}")
print(f"  换个算法：高度（密度）= 1 / {num(width)} = {density:.3f}，概率 = 高度 × 宽度 = {density:.3f} × {num(seg_hi - seg_lo)} = {density * (seg_hi - seg_lo):.3f}")
print(f"  整段的面积 = (1 / {num(width)}) × {num(width)} = {num(density * width)}   （{density:.3f} 是四舍五入过的；拿它去乘会得 {round(density, 3) * width:.4f}）")
check("U(6.5, 8.2)：宽 1.7，落在 7.0–7.5 的概率 0.5/1.7 = 0.294（约 29%）", math.isclose(width, 1.7) and round(p_seg, 3) == 0.294)
check("同一个数的另一种算法：密度 1/1.7 = 0.588，面积 0.588 × 0.5 = 0.294；整段面积 = 1",
      round(density, 3) == 0.588 and math.isclose(density * 0.5, p_seg) and math.isclose(density * width, 1.0))
check("字面值重算：0.5 / 1.7 → 0.294；1 / 1.7 → 0.588；0.588 × 0.5 → 0.294；0.588 × 1.7 = 0.9996（所以正文写成 (1/1.7) × 1.7 = 1）",
      round(0.5 / 1.7, 3) == 0.294 and round(1 / 1.7, 3) == 0.588 and round(0.588 * 0.5, 3) == 0.294 and round(0.588 * 1.7, 4) == 0.9996)

N_DRAWS = 100_000
volts = np.random.default_rng(2).uniform(lo, hi, size=N_DRAWS)
frac_seg = float(np.mean((volts >= seg_lo) & (volts <= seg_hi)))
n_exact = int(np.sum(volts == 7.0))
print(f"\n真的抽 {N_DRAWS} 个电压：落在 7.0–7.5 的占 {frac_seg:.3f}；恰好等于 7.000000… 的有 {n_exact} 个")
check(f"抽 10 万个：落在 7.0–7.5 的比例 {frac_seg:.3f} 和理论值 {p_seg:.3f} 相差不到 0.005", abs(frac_seg - p_seg) < 0.005)
check("抽 10 万个：没有一个恰好等于 7", n_exact == 0)
check("正文引用的数：10 万个里落在 7.0–7.5 的占 0.293", round(frac_seg, 3) == 0.293)

# 左端钉在 7.0，右端一点点收窄：概率跟着宽度一起缩，缩成一个点就是 0
SHRINK = (0.5, 0.1, 0.01, 0.001, 0.0001)
shrink_rows, shrink_counts = [], []
for d in SHRINK:
    n_in = int(np.sum((volts >= seg_lo) & (volts < seg_lo + d)))
    shrink_counts.append(n_in)
    p_shown = float(f"{d / width:.3g}")                    # 正文印的是三位有效数字，“10 万 × 概率”就拿它来乘
    expect = N_DRAWS * p_shown
    shrink_rows.append([f"{seg_lo:.1f} 到 {seg_lo + d:g}", f"{d:g}", np.format_float_positional(p_shown, trim="-"), f"{expect:,.0f}" if expect >= 100 else f"{expect:.3g}", f"{n_in:,}"])
shrink_rows.append(["恰好 7", "0", "0", "0", f"{n_exact:,}"])
print("\n把这一段从右边一点点收窄（左端一直是 7.0）：")
table(["这一段", "宽度（伏）", "概率 = 宽度 ÷ 1.7", "10 万 × 概率", "实际落进去"], shrink_rows)
check("正文表里的概率：0.5/1.7 → 0.294，0.1/1.7 → 0.0588，0.01/1.7 → 0.00588，0.001/1.7 → 0.000588，0.0001/1.7 → 0.0000588",
      [f"{d / width:.3g}" for d in SHRINK] == ["0.294", "0.0588", "0.00588", "0.000588", "5.88e-05"])
check("正文表里的个数：10 万个里落进去 29,265、5,878、599、61、7 个，恰好等于 7 的 0 个",
      shrink_counts == [29265, 5878, 599, 61, 7] and n_exact == 0)
check("字面值重算：100000 × 0.294 = 29,400；× 0.0588 = 5,880；× 0.00588 = 588；× 0.000588 = 58.8；× 0.0000588 = 5.88",
      round(100000 * 0.294) == 29400 and round(100000 * 0.0588) == 5880 and round(100000 * 0.00588) == 588
      and round(100000 * 0.000588, 1) == 58.8 and round(100000 * 0.0000588, 2) == 5.88)
check("每一行的实际个数和 10 万 × 概率相差都不到 3 个标准差（抽样的随机起伏，4.5 节）",
      all(abs(c - N_DRAWS * d / width) < 3 * math.sqrt(N_DRAWS * d / width) + 1 for c, d in zip(shrink_counts, SHRINK)))

# 直方图：切格子、数个数、除以 10 万得占比；再除以格宽，就成了密度
HIST_WIDTHS = (0.1, 0.05, 0.01)
hist = {}
for w in HIST_WIDTHS:
    n_bins = max(1, int(round(width / w)))
    edges = lo + w * np.arange(n_bins + 1)
    counts_w = np.histogram(volts, bins=edges)[0]
    hist[w] = (edges, counts_w / N_DRAWS)
    print(f"\n每 {w:g} 伏一格：{n_bins} 格；每格约 {N_DRAWS / n_bins:,.0f} 个（实际 {counts_w.min():,}–{counts_w.max():,}），"
          f"占比约 1/{n_bins} = {1 / n_bins:.3g}；占比 ÷ 格宽 = {1 / n_bins:.3g} ÷ {w:g} = {1 / n_bins / w:.3f}")
edges1, frac1 = hist[0.1]
bin7 = int(round((seg_lo - lo) / 0.1))                     # 7.0–7.1 那一格
check("每 0.1 伏一格：17 格，每格约 100000 ÷ 17 ≈ 5,882 个，占比 1/17 ≈ 0.0588；实际每格 5,743 到 5,998 个，7.0–7.1 那格 5,878 个",
      len(frac1) == 17 and round(N_DRAWS / 17) == 5882 and round(1 / 17, 4) == 0.0588
      and round(frac1.min() * N_DRAWS) == 5743 and round(frac1.max() * N_DRAWS) == 5998 and round(frac1[bin7] * N_DRAWS) == 5878)
check("每 0.05 伏一格：34 格，每格约 100000 ÷ 34 ≈ 2,941 个，占比 1/34 ≈ 0.0294——柱子矮了一半",
      len(hist[0.05][1]) == 34 and round(N_DRAWS / 34) == 2941 and round(1 / 34, 4) == 0.0294 and math.isclose(1 / 34, (1 / 17) / 2))
check("每 0.01 伏一格：170 格，占比约 0.00588", len(hist[0.01][1]) == 170 and round(1 / 170, 5) == 0.00588)
check("除以格宽，三种切法一样高：0.0588 ÷ 0.1 = 0.0294 ÷ 0.05 = 0.00588 ÷ 0.01 = 0.588（字面值重算）",
      round(0.0588 / 0.1, 3) == round(0.0294 / 0.05, 3) == round(0.00588 / 0.01, 3) == 0.588
      and all(math.isclose((1 / len(f)) / w, density) for w, (_, f) in hist.items()))
check("实际的柱子除以格宽后都在 0.588 上下：0.1 伏格在 ±0.03 以内，0.05 伏格在 ±0.04 以内",
      bool(np.all(np.abs(frac1 / 0.1 - density) < 0.03)) and bool(np.all(np.abs(hist[0.05][1] / 0.05 - density) < 0.04)))
check("乘回去：0.588 × 0.1 = 0.0588——柱子的面积（高 × 宽）就是这一格的占比；17 格的面积加起来是 1",
      round(0.588 * 0.1, 4) == 0.0588 and math.isclose(float(np.sum(frac1 / 0.1 * 0.1)), 1.0))
check("自测：每 0.17 伏一格，1.7 ÷ 0.17 = 10 格，每格占比约 0.1，0.1 ÷ 0.17 = 0.588",
      round(1.7 / 0.17) == 10 and round(0.1 / 0.17, 3) == 0.588)

narrow_w = 0.2                                    # 第二个均匀分布 U(0, 0.2)：窄到高度超过 1
narrow_density = 1 / narrow_w
print(f"\n窄一点的 U(0, 0.2)：高度 = 1 / 0.2 = {num(narrow_density)}（超过 1 了）")
print(f"  宽 0.01 的一小段：概率 = 5 × 0.01 = {num(narrow_density * 0.01)}；整段：5 × 0.2 = {num(narrow_density * narrow_w)}")
check("U(0, 0.2) 的密度是 5——高度可以超过 1", math.isclose(narrow_density, 5.0))
check("密度 5 × 区间宽度 0.01 = 0.05（worked_examples.py 里的同一个手算）", math.isclose(5 * 0.01, 0.05))
check("U(0, 0.2) 整段面积 5 × 0.2 = 1", math.isclose(narrow_density * narrow_w, 1.0))
width_mv = width * 1000                          # 同一个分布改用毫伏来记：宽度 1700 毫伏
density_mv = 1 / width_mv
print(f"\n密度带单位：改用毫伏来记，宽度 {width_mv:g} 毫伏，密度 = 1 / {width_mv:g} = {density_mv:.6f}（每毫伏）；"
      f"落在 7000–7500 毫伏的概率 = {density_mv:.6f} × 500 = {density_mv * 500:.3f}，没变")
check("换成毫伏：密度从 0.588 变成 1/1700 ≈ 0.000588，概率仍是 0.294——密度随单位变，概率不变",
      round(density_mv, 6) == 0.000588 and math.isclose(density_mv * 500, p_seg))
check("字面值重算：1 / 1700 → 0.000588；0.000588 × 500 → 0.294；5 × 0.01 = 0.05；5 × 0.2 = 1",
      round(1 / 1700, 6) == 0.000588 and round(0.000588 * 500, 3) == 0.294 and round(5 * 0.01, 2) == 0.05 and 5 * 0.2 == 1)

# 不均匀的密度（教学构造）：一箱旧电池，60% 在 6.5–7.0 伏，40% 在 7.0–8.2 伏，每一段内部处处机会均等
OLD_STEP, OLD_LO, OLD_HI, OLD_P_LOW = 7.0, 6.5, 8.2, 0.6
h_low = OLD_P_LOW / (OLD_STEP - OLD_LO)                    # 0.6 ÷ 0.5 = 1.2
h_high = (1 - OLD_P_LOW) / (OLD_HI - OLD_STEP)             # 0.4 ÷ 1.2 = 1/3
ask_lo, ask_hi = 6.8, 7.2
area_left = h_low * (OLD_STEP - ask_lo)
area_right = h_high * (ask_hi - OLD_STEP)
p_ask = area_left + area_right
p_naive = (ask_hi - ask_lo) / (OLD_HI - OLD_LO)
r_old = np.random.default_rng(8)
n_low = int(round(N_DRAWS * OLD_P_LOW))
old_volts = np.concatenate([r_old.uniform(OLD_LO, OLD_STEP, n_low), r_old.uniform(OLD_STEP, OLD_HI, N_DRAWS - n_low)])
frac_ask = float(np.mean((old_volts >= ask_lo) & (old_volts <= ask_hi)))
print(f"\n一箱旧电池（教学构造）：{OLD_P_LOW:g} 的概率在 {OLD_LO}–{OLD_STEP} 伏，{1 - OLD_P_LOW:g} 在 {OLD_STEP}–{OLD_HI} 伏")
print(f"  两段的高度：{OLD_P_LOW:g} ÷ {OLD_STEP - OLD_LO:g} = {h_low:g}；{1 - OLD_P_LOW:g} ÷ {OLD_HI - OLD_STEP:g} = {h_high:.3f}")
print(f"  落在 {ask_lo}–{ask_hi} 伏：左块 {h_low:g} × 0.2 = {area_left:.2f}，右块 (1/3) × 0.2 = {area_right:.4f}，合计 {p_ask:.4f}；"
      f"只看宽度的比例 0.4 ÷ 1.7 = {p_naive:.3f}（错的）")
print(f"  真的抽 {N_DRAWS} 个（{n_low} 个从左段、{N_DRAWS - n_low} 个从右段均匀抽）：落在 {ask_lo}–{ask_hi} 的占 {frac_ask:.3f}")
check("旧电池：左段高 0.6 ÷ 0.5 = 1.2（超过 1），右段高 0.4 ÷ 1.2 = 1/3 ≈ 0.333；总面积 1.2 × 0.5 + (1/3) × 1.2 = 0.6 + 0.4 = 1",
      math.isclose(h_low, 1.2) and math.isclose(h_high, 1 / 3) and round(h_high, 3) == 0.333
      and math.isclose(h_low * 0.5 + h_high * 1.2, 1.0) and round(1.2 * 0.5, 2) == 0.6 and round(1.2 / 3, 2) == 0.4)
check("旧电池落在 6.8–7.2 伏：1.2 × 0.2 = 0.24，(1/3) × 0.2 = 0.0667，0.24 + 0.0667 = 0.3067 ≈ 0.307（字面值重算也成立）",
      round(area_left, 2) == 0.24 and round(area_right, 4) == 0.0667 and round(p_ask, 3) == 0.307
      and round(1.2 * 0.2, 2) == 0.24 and round(0.2 / 3, 4) == 0.0667 and round(0.24 + 0.0667, 4) == 0.3067)
check("只看宽度的比例 0.4 ÷ 1.7 = 0.235 是错的：差了 0.07 以上", round(p_naive, 3) == 0.235 and p_ask - p_naive > 0.07)
check(f"抽 10 万个旧电池：落在 6.8–7.2 的比例 {frac_ask:.3f} 和面积 {p_ask:.3f} 相差不到 0.005", abs(frac_ask - p_ask) < 0.005)
check("正文引用的数：10 万个旧电池里落在 6.8–7.2 的占 0.307", round(frac_ask, 3) == 0.307)
check("Math.random() 就是 U(0, 1)：宽 1，高度 1", 1 / (1 - 0) == 1)
frac_small = float(np.mean(np.random.default_rng(4).uniform(0, 1, size=N_DRAWS) < 0.1))
check("U(0, 1) 小于 0.1 的概率就是这一段的宽度 0.1（抽 10 万个核对，容差 0.005）——“以指定的概率做一件事”的写法", abs(frac_small - 0.1) < 0.005)
check("自测：U(2, 6) 的高度 0.25，落在 3–4 的概率 0.25", 1 / (6 - 2) == 0.25 and (4 - 3) / (6 - 2) == 0.25)

# ---------------------------------------------------------------------------
banner("2b. 画图：figures/ch04_histogram_to_density.png（格子越细柱子越矮；除以格宽就稳住了）")
if not (finite(lo, hi, density) and 0 < width < 3 and lo <= seg_lo < seg_lo + 0.1 <= hi):
    print("  电压范围改得太离谱，这张图跳过。")
else:
    fig = plt.figure(figsize=(9.4, 18.6))
    gs = fig.add_gridspec(3, 1, hspace=0.66, left=0.13, right=0.95, top=0.925, bottom=0.075)
    fig.suptitle("柱高 ÷ 格宽：怎么切都一样高，这个高度就是密度", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
    X_LIM = (lo - 0.2, hi + 0.2)
    F_TOP = 0.078
    bin7_half = int(round((seg_lo - lo) / 0.05))            # 7.0–7.05 那一格

    ax = fig.add_subplot(gs[0])
    data_axes(ax, "电压（伏）", "占比（个数 ÷ 10 万）")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(0, F_TOP)
    ax.bar(edges1[:-1], frac1, width=0.1, align="edge", color="#dbe7f3", edgecolor=BLUE, lw=1.0, zorder=2)
    ax.bar(edges1[bin7], frac1[bin7], width=0.1, align="edge", color=ORANGE, edgecolor=BLUE, lw=1.0, zorder=3)
    ax.axhline(1 / 17, color=INK, ls="--", lw=1.4, zorder=1)
    ax.text(X_LIM[1], 1 / 17 + 0.002, f"约 1 ÷ 17 = {1 / 17:.4f}", fontsize=FS_SMALL, color=INK, ha="right", va="bottom")
    ax.annotate(f"7.0–7.1 这一格：{round(frac1[bin7] * N_DRAWS):,} 个\n占比 {round(frac1[bin7] * N_DRAWS):,} ÷ 100,000 = {frac1[bin7]:.4f}",
                xy=(seg_lo + 0.05, frac1[bin7] * 0.6), xytext=(lo + 0.05, 0.0035), fontsize=FS_SMALL, color=ORANGE, ha="left", va="bottom",
                linespacing=1.35, bbox=WHITE_BOX, arrowprops=dict(arrowstyle="-", lw=1.4, color=ORANGE, shrinkA=2, shrinkB=2))
    panel_title(fig, [ax], "① 每 0.1 伏一格：17 根柱子，每根占比约 0.0588")
    panel_note(fig, [ax], "17 根柱子的占比加起来是 1：10 万个电压，每个都落在某一格里。")

    ax = fig.add_subplot(gs[1])
    data_axes(ax, "电压（伏）", "占比（个数 ÷ 10 万）")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(0, F_TOP)
    edges2, frac2 = hist[0.05]
    ax.bar(edges2[:-1], frac2, width=0.05, align="edge", color="#dff0e8", edgecolor=GREEN, lw=0.8, zorder=2)
    ax.bar(edges2[bin7_half], frac2[bin7_half], width=0.05, align="edge", color=ORANGE, edgecolor=GREEN, lw=0.8, zorder=3)
    ax.axhline(1 / 17, color=FAINT, ls=":", lw=1.8, zorder=1)
    ax.text(X_LIM[0] + 0.03, 1 / 17 + 0.002, f"上一幅的高度 {1 / 17:.4f}", fontsize=15, color=MUTED, ha="left", va="bottom")
    ax.axhline(1 / 34, color=INK, ls="--", lw=1.4, zorder=1)
    ax.text(X_LIM[0] + 0.03, 1 / 34 + 0.0015, f"约 1 ÷ 34 = {1 / 34:.4f}", fontsize=FS_SMALL, color=INK, ha="left", va="bottom", bbox=WHITE_BOX,
            zorder=5)
    arrow(ax, (hi + 0.1, 1 / 17 - 0.001), (hi + 0.1, 1 / 34 + 0.001), MUTED, lw=2.2)
    ax.text(hi + 0.05, (1 / 17 + 1 / 34) / 2, "矮了一半", fontsize=FS_SMALL, color=MUTED, ha="right", va="center")
    panel_title(fig, [ax], "② 改成每 0.05 伏一格：34 根，每根只剩约 0.0294")
    panel_note(fig, [ax], "格子细一半，每格分到的电压就少一半。切到一个点那么细，柱子就矮成 0。\n"
                          "这样画，图的高矮取决于“怎么切”，不是电压本身的样子。")

    ax = fig.add_subplot(gs[2])
    data_axes(ax, "电压（伏）", "密度（每伏分到多少概率）")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(0, 1.0)
    ax.fill_between([edges1[bin7], edges1[bin7] + 0.1], 0, frac1[bin7] / 0.1, color=ORANGE, alpha=0.85, zorder=2, lw=0)
    ax.stairs(frac2 / 0.05, edges2, color=GREEN, lw=1.6, zorder=3)
    ax.stairs(frac1 / 0.1, edges1, color=BLUE, lw=2.6, zorder=4)
    ax.axhline(density, color=INK, ls="--", lw=1.4, zorder=1)
    ax.text(X_LIM[0] + 0.03, density + 0.025, f"1 ÷ 1.7 = {density:.3f}", fontsize=FS_SMALL, color=INK, ha="left", va="bottom")
    hand(ax, lo + 0.05, 0.91, f"蓝线 = ① 的柱子 ÷ 0.1：0.0588 ÷ 0.1 = {0.0588 / 0.1:.3f}", color=BLUE, fontsize=FS_SMALL, ha="left")
    hand(ax, lo + 0.05, 0.80, f"绿线 = ② 的柱子 ÷ 0.05：0.0294 ÷ 0.05 = {0.0294 / 0.05:.3f}", color=GREEN, fontsize=FS_SMALL, ha="left")
    ax.annotate(f"这一格的面积\n= 0.588 × 0.1 = {0.588 * 0.1:.4f}\n= 它的占比", xy=(seg_lo + 0.05, density * 0.45),
                xytext=(seg_lo + 0.32, 0.26), fontsize=FS_SMALL, color=ORANGE, ha="left", va="center", linespacing=1.35, bbox=WHITE_BOX,
                arrowprops=dict(arrowstyle="-", lw=1.4, color=ORANGE, shrinkA=2, shrinkB=2))
    panel_title(fig, [ax], "③ 每根柱子再除以格宽：两种切法一样高，蓝绿两线几乎重合")
    panel_note(fig, [ax], "除以格宽，得到“每 1 伏分到多少概率”——这就是密度，和怎么切无关。\n"
                          "把格宽乘回去：柱子的面积 = 高 × 宽 = 这一格的概率。")
    savefig(fig, "ch04_histogram_to_density")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("2c. 画图：figures/ch04_area_is_probability.png（面积才是概率；高度可以超过 1）")
if not (finite(lo, hi, density) and 0 < width < 3):
    print("  电压范围改得太离谱，这张图跳过。")
else:
    fig = plt.figure(figsize=(9.4, 13.6))
    gs = fig.add_gridspec(2, 1, hspace=0.62, left=0.12, right=0.95, top=0.90, bottom=0.11)
    fig.suptitle("连续的量：高度是密度，涂色的面积才是概率", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.975)
    X_SPAN, Y_TOP = 2.4, 5.8                        # 两幅用同样的横向跨度和纵轴：两块面积都是 1，才能用眼睛比
    mid = (lo + hi) / 2
    seg_ok = lo <= seg_lo < seg_hi <= hi

    ax = fig.add_subplot(gs[0])
    data_axes(ax, "电压（伏）", "密度（每伏分到多少概率）")
    ax.set_xlim(mid - X_SPAN / 2, mid + X_SPAN / 2)
    ax.set_ylim(0, Y_TOP)
    ax.fill_between([lo, hi], 0, density, color="#dbe7f3", zorder=2)
    if seg_ok:
        ax.fill_between([seg_lo, seg_hi], 0, density, color=ORANGE, alpha=0.85, zorder=3)
    ax.plot([lo, lo, hi, hi], [0, density, density, 0], color=BLUE, lw=3, zorder=4)
    ax.text(hi, density + 0.25, f"高度 = 1 ÷ {num(width)} = {density:.3f}", fontsize=FS_SMALL, color=BLUE, ha="right", va="bottom")
    if seg_ok:
        ax.annotate(f"落在 {seg_lo:g}–{seg_hi:g} 伏的概率\n= 高度 × 宽度 = {density:.3f} × {seg_hi - seg_lo:g} = {p_seg:.3f}",
                    xy=((seg_lo + seg_hi) / 2, density * 0.55), xytext=(mid - 0.05, 3.1), fontsize=FS_SMALL, color=ORANGE,
                    ha="center", va="center", linespacing=1.4,
                    arrowprops=dict(arrowstyle="-", lw=1.4, color=ORANGE, shrinkA=4, shrinkB=2))
    ax.text(mid, 4.85, f"整块面积 = (1 ÷ {num(width)}) × {num(width)} = 1", fontsize=FS_STEP, color=INK, ha="center", va="center")
    panel_title(fig, [ax], f"① 电压 U({lo:g}, {hi:g})：摊在 {num(width)} 伏宽上，高度只有 {density:.3f}")
    panel_note(fig, [ax], "恰好等于某一个值：那是一条没有宽度的线，面积 0，概率 0。")

    ax = fig.add_subplot(gs[1])
    data_axes(ax, "抽到的数 x", "密度")
    ax.set_xlim(narrow_w / 2 - X_SPAN / 2, narrow_w / 2 + X_SPAN / 2)
    ax.set_ylim(0, Y_TOP)
    ax.fill_between([0, narrow_w], 0, narrow_density, color="#dbe7f3", zorder=2)
    ax.fill_between([0.10, 0.11], 0, narrow_density, color=ORANGE, zorder=3)
    ax.plot([0, 0, narrow_w, narrow_w], [0, narrow_density, narrow_density, 0], color=BLUE, lw=3, zorder=4)
    ax.axhline(1.0, color=MUTED, ls=":", lw=1.8, zorder=1)
    ax.text(-0.95, 1.12, "高度 1", fontsize=15, color=MUTED, va="bottom")
    ax.text(0.32, 5.0, f"高度 = 1 ÷ 0.2 = {num(narrow_density)}：超过 1 了", fontsize=FS_SMALL, color=BLUE, va="center")
    ax.annotate(f"宽 0.01 的一小条：概率 = 5 × 0.01 = {num(narrow_density * 0.01)}", xy=(0.105, 2.6), xytext=(0.32, 3.3),
                fontsize=FS_SMALL, color=ORANGE, va="center",
                arrowprops=dict(arrowstyle="-", lw=1.4, color=ORANGE, shrinkA=4, shrinkB=2))
    ax.text(0.32, 1.95, "整块面积 = 5 × 0.2 = 1", fontsize=FS_STEP, color=INK, va="center")
    panel_title(fig, [ax], "② U(0, 0.2)：同样是面积 1，挤在 0.2 宽上，高度就成了 5")
    panel_note(fig, [ax], "两幅的横向跨度、纵轴都一样：两块浅蓝的面积相等，都是 1。\n密度可以超过 1，概率（面积）不会。")
    savefig(fig, "ch04_area_is_probability")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("2d. 画图：figures/ch04_uneven_density.png（密度高低不平时，概率还是面积）")
fig = plt.figure(figsize=(9.4, 13.2))
gs = fig.add_gridspec(2, 1, hspace=0.62, left=0.12, right=0.95, top=0.89, bottom=0.11)
fig.suptitle("密度高低不平时，概率还是面积：跨过台阶就拆成两块", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.975)
X_LIM = (OLD_LO - 0.2, OLD_HI + 0.2)
D_TOP = 1.6
step_x = [OLD_LO, OLD_LO, OLD_STEP, OLD_STEP, OLD_HI, OLD_HI]
step_y = [0, h_low, h_low, h_high, h_high, 0]

ax = fig.add_subplot(gs[0])
data_axes(ax, "电压（伏）", "密度（每伏分到多少概率）")
ax.set_xlim(*X_LIM)
ax.set_ylim(0, D_TOP)
old_edges = OLD_LO + 0.1 * np.arange(int(round((OLD_HI - OLD_LO) / 0.1)) + 1)
old_frac = np.histogram(old_volts, bins=old_edges)[0] / N_DRAWS
ax.bar(old_edges[:-1], old_frac / 0.1, width=0.1, align="edge", color="#dbe7f3", edgecolor=FAINT, lw=0.8, zorder=2)
ax.plot(step_x, step_y, color=BLUE, lw=3, zorder=4)
ax.axhline(1.0, color=MUTED, ls=":", lw=1.8, zorder=1)
ax.text(OLD_HI + 0.18, 1.03, "高度 1", fontsize=15, color=MUTED, ha="right", va="bottom")
ax.text((OLD_LO + OLD_STEP) / 2, h_low + 0.06, f"{OLD_P_LOW:g} ÷ {OLD_STEP - OLD_LO:g} = {h_low:g}", fontsize=FS_SMALL, color=BLUE,
        ha="center", va="bottom")
ax.text((OLD_STEP + OLD_HI) / 2, h_high + 0.06, f"{1 - OLD_P_LOW:g} ÷ {OLD_HI - OLD_STEP:g} = {h_high:.3f}", fontsize=FS_SMALL,
        color=BLUE, ha="center", va="bottom")
panel_title(fig, [ax], "① 10 万个旧电池的直方图，贴着两级台阶")
panel_note(fig, [ax], "浅蓝柱子：每 0.1 伏一格，占比 ÷ 格宽（上一张图 ③ 的画法）。\n"
                      "蓝线是两段的高度。左段高 1.2，比 1 还高：0.6 的概率挤在 0.5 伏里。")

ax = fig.add_subplot(gs[1])
data_axes(ax, "电压（伏）", "密度（每伏分到多少概率）")
ax.set_xlim(*X_LIM)
ax.set_ylim(0, D_TOP)
ax.fill_between([ask_lo, OLD_STEP], 0, h_low, color=ORANGE, alpha=0.85, zorder=2, lw=0)
ax.fill_between([OLD_STEP, ask_hi], 0, h_high, color=GREEN, alpha=0.85, zorder=2, lw=0)
ax.plot(step_x, step_y, color=BLUE, lw=3, zorder=4)
ax.annotate(f"左块：{h_low:g} × 0.2 = {area_left:.2f}", xy=(ask_lo + 0.1, h_low * 0.7), xytext=(OLD_LO + 0.05, 1.45),
            fontsize=FS_SMALL, color=ORANGE, ha="left", va="center", bbox=WHITE_BOX,
            arrowprops=dict(arrowstyle="-", lw=1.4, color=ORANGE, shrinkA=2, shrinkB=2))
ax.annotate(f"右块：1/3 × 0.2 = {area_right:.4f}", xy=(OLD_STEP + 0.1, h_high * 0.5), xytext=(OLD_STEP + 0.35, 0.62),
            fontsize=FS_SMALL, color=GREEN, ha="left", va="center", bbox=WHITE_BOX,
            arrowprops=dict(arrowstyle="-", lw=1.4, color=GREEN, shrinkA=2, shrinkB=2))
hand(ax, OLD_STEP + 0.35, 1.0, f"合计 {area_left:.2f} + {area_right:.4f} ≈ {p_ask:.3f}", color=INK, ha="left")
panel_title(fig, [ax], f"② 落在 {ask_lo}–{ask_hi} 伏的概率：两块面积相加")
panel_note(fig, [ax], f"只看宽度的比例 0.4 ÷ 1.7 = {p_naive:.3f}，是错的：它假装处处一样密。\n"
                      f"真的抽 10 万个旧电池数一数，落在这一段的占 {frac_ask:.3f}——对上的是面积。")
savefig(fig, "ch04_uneven_density")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("3. 期望：按概率加权的平均")
E_dice = float(np.sum(faces * p_dice))
print("骰子：E[X] = 1×1/6 + 2×1/6 + … + 6×1/6 = (1+2+3+4+5+6)/6 =", num(E_dice))
check("骰子的期望 3.5——而 3.5 不是任何一面", math.isclose(E_dice, 3.5) and 3.5 not in faces)
E_wheel = float(np.sum(wheel_prize * wheel_p))
print("转盘：E = 0×0.5 + 5×0.3 + 20×0.2 = 0 + 1.5 + 4 =", num(E_wheel), "元；而直接把三个奖金平均是", num(float(wheel_prize.mean()), 2), "元")
check("转盘的期望 5.5 元；不加权的平均 8.33 元是错的", math.isclose(E_wheel, 5.5) and round(float(wheel_prize.mean()), 2) == 8.33)
check("期望不是“最可能出现的”：转盘最可能出 0 元，期望却是 5.5 元", wheel_prize[int(np.argmax(wheel_p))] == 0)
check("自测：0 元 0.9、100 元 0.1 的抽奖，期望 0 × 0.9 + 100 × 0.1 = 10 元", math.isclose(0 * 0.9 + 100 * 0.1, 10))

STRIP_V = 0.1
n_strips = int(round(width / STRIP_V))
centers = lo + STRIP_V * (np.arange(n_strips) + 0.5)
strip_p = np.full(n_strips, density * STRIP_V)     # 每条的面积 = 高度 × 宽度
E_volt_strips = float(np.sum(centers * strip_p))
E_volt = (lo + hi) / 2
print(f"\n电压：把 {lo}–{hi} 切成 {n_strips} 条、每条宽 {STRIP_V}，每条的概率 = {density:.3f} × {STRIP_V} = {strip_p[0]:.4f}")
print(f"  Σ 条的中点 × 条的概率 = {num(E_volt_strips)}；区间中点 (a + b)/2 = {num(E_volt)}")
check("均匀分布的期望 = 区间中点：电压 7.35 V（切条相加也是 7.35）", math.isclose(E_volt, 7.35) and math.isclose(E_volt_strips, 7.35))
pairs = [(float(centers[k]), float(centers[-1 - k])) for k in range(n_strips // 2)]
print(f"  17 条概率相等，加权平均就是 17 个中点的普通平均；中点两两配对：" + "、".join(f"{a:g} 配 {b:g}" for a, b in pairs[:2]) + "……每对的平均都是 "
      + num((pairs[0][0] + pairs[0][1]) / 2) + f"，正中间那条的中点也是 {num(float(centers[n_strips // 2]))}")
print(f"  （{strip_p[0]:.4f} 是四舍五入过的：17 × 0.0588 = {17 * 0.0588:.4f}；精确值是 1/17）")
check("17 条、每条的概率 0.588 × 0.1 = 0.0588（精确是 1/17），加起来是 1", n_strips == 17 and round(float(strip_p[0]), 4) == 0.0588
      and math.isclose(float(strip_p[0]), 1 / 17) and math.isclose(float(strip_p.sum()), 1.0))
check("正文引用的数：中点从 6.55 到 8.15；6.55 配 8.15、6.65 配 8.05……每对平均都是 7.35；17 × 0.0588 = 0.9996",
      math.isclose(float(centers[0]), 6.55) and math.isclose(float(centers[-1]), 8.15) and math.isclose(pairs[1][0], 6.65) and math.isclose(pairs[1][1], 8.05)
      and all(math.isclose((a + b) / 2, 7.35) for a, b in pairs) and math.isclose(float(centers[8]), 7.35) and round(17 * 0.0588, 4) == 0.9996)

E_2x3 = float(np.sum((2 * faces + 3) * p_dice))
pair_sums = faces[:, None] + faces[None, :]
E_two = float(pair_sums.mean())
E_sq = float(np.sum(faces**2 * p_dice))
print(f"\n线性：E[2X + 3]，逐面算 (5+7+9+11+13+15)/6 = {num(E_2x3)}；直接用 2 × 3.5 + 3 = {num(2 * E_dice + 3)}")
print(f"      两颗骰子点数之和：36 种组合的平均 = {num(E_two)} = 3.5 + 3.5")
print(f"别顺手推广到平方：E[X²] = (1+4+9+16+25+36)/6 = {E_sq:.4f}，而 (E[X])² = {E_dice**2:g}")
check("E[2X + 3] = 2 × 3.5 + 3 = 10", math.isclose(E_2x3, 10) and math.isclose(2 * E_dice + 3, 10))
check("E[X + Y] = E[X] + E[Y]：两颗骰子之和的期望是 7", math.isclose(E_two, 7))
check("E[X²] = 91/6 ≈ 15.17，不等于 3.5² = 12.25", round(E_sq, 2) == 15.17 and E_dice**2 == 12.25)
check("字面值重算：0 + 1.5 + 4 = 5.5；(0 + 5 + 20)/3 → 8.33；(5+7+9+11+13+15)/6 = 10；91/6 → 15.17；15.17 − 12.25 = 2.92",
      0 + 1.5 + 4 == 5.5 and round((0 + 5 + 20) / 3, 2) == 8.33 and (5 + 7 + 9 + 11 + 13 + 15) / 6 == 10 and round(91 / 6, 2) == 15.17
      and round(15.17 - 12.25, 2) == 2.92)
check("一个和除以 6 = 每项各除以 6 再相加：(2 + 4) ÷ 2 = 3 = 2 ÷ 2 + 4 ÷ 2；(1+…+6) ÷ 6 = Σ x × 1/6",
      (2 + 4) / 2 == 3 == 2 / 2 + 4 / 2 and math.isclose(float(faces.sum()) / 6, float(np.sum(faces * (1 / 6)))))
check("每条细条的概率不舍入是 (1/1.7) × 0.1 = 0.1/1.7 = 1/17", math.isclose((1 / 1.7) * 0.1, 0.1 / 1.7) and math.isclose(0.1 / 1.7, 1 / 17))
first_half = float(np.sum(2 * faces * p_dice))                     # 拆成两堆：“2 × x × 1/6”那一堆
second_half = float(np.sum(3 * p_dice))                            # “3 × 1/6”那一堆
check("线性为什么成立：逐面的 (2x+3)×1/6 拆成两堆，2 × (Σ x/6) = 2 × 3.5 = 7，3 × (Σ 1/6) = 3 × 1 = 3，合起来 10；分配律 (2+3)×4 = 2×4 + 3×4 = 20",
      math.isclose(first_half, 7) and math.isclose(second_half, 3) and math.isclose(first_half + second_half, E_2x3)
      and (2 + 3) * 4 == 2 * 4 + 3 * 4 == 20 and 2 * 3.5 == 7 and 3 * 1 == 3)
first_die_total = int(np.sum(np.broadcast_to(faces[:, None], (6, 6))))   # 36 种组合里“第一颗的点数”加起来
second_die_total = int(np.sum(np.broadcast_to(faces[None, :], (6, 6))))
check("两颗骰子：36 种组合；“第一颗的点数”加起来 6 × 21 = 126，第二颗也是 126；36 个和加起来 252，252 ÷ 36 = 7；126 ÷ 36 = 3.5",
      pair_sums.size == 36 == 6 * 6 and first_die_total == 126 == 6 * 21 and second_die_total == 126 and int(pair_sums.sum()) == 252 == 126 + 126
      and 252 / 36 == 7 and 126 / 36 == 3.5 and int(faces.sum()) == 21)

# ---------------------------------------------------------------------------
banner("4. 方差与标准差：散得多开")
class_a = np.array([70.0, 70.0])                   # 全班 70
class_b = np.array([40.0, 100.0])                  # 一半 40、一半 100
print("两个班平均分都是 70：")
table(["", "分数", "平均", "偏差（分数 − 平均）", "偏差²", "方差", "标准差"],
      [["甲班", "全是 70", num(float(class_a.mean())), "0", "0", num(float(class_a.var())), num(float(class_a.std()))],
       ["乙班", "一半 40、一半 100", num(float(class_b.mean())), "−30、+30", "900、900", num(float(class_b.var())), num(float(class_b.std()))]])
check("甲班方差 0；乙班方差 900、标准差 30 分", class_a.var() == 0 and class_b.var() == 900 and class_b.std() == 30)

two = np.array([1.0, 3.0])
print("\n为什么要平方：等概率的 1 和 3。均值", num(float(two.mean())), "，偏差", (two - two.mean()).tolist(),
      "，偏差直接平均 =", num(float((two - two.mean()).mean())), "，平方后再平均 =", num(float(two.var())))
check("[1, 3] 的总体方差 = 1，标准差 = 1（worked_examples.py 里的同一个手算）；偏差直接平均是 0",
      two.var() == 1 and two.std() == 1 and (two - two.mean()).mean() == 0)

dev = faces - E_dice
print("\n骰子（期望 3.5）：")
table(["点数 x", "偏差 x − 3.5", "偏差²"], [[int(x), num(d), num(d * d)] for x, d in zip(faces, dev)])
var_dice = float(np.sum(p_dice * dev**2))
std_dice = math.sqrt(var_dice)
print(f"方差 = (6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25) / 6 = 17.5 / 6 = {var_dice:.4f}；标准差 = √{var_dice:.4f} = {std_dice:.4f}")
check("骰子：偏差² 加起来 17.5，方差 17.5/6 = 2.92，标准差 1.71",
      math.isclose(float(np.sum(dev**2)), 17.5) and round(var_dice, 2) == 2.92 and round(std_dice, 2) == 1.71)
check("换一种算法也对得上：方差 = E[X²] − (E[X])² = 15.17 − 12.25 = 2.92", math.isclose(E_sq - E_dice**2, var_dice))
check("字面值重算：17.5 / 6 → 2.92；√2.92 → 1.71；乙班 (900 + 900)/2 = 900，√900 = 30",
      round(17.5 / 6, 2) == 2.92 and round(math.sqrt(2.92), 2) == 1.71 and (900 + 900) / 2 == 900 and math.sqrt(900) == 30)

std_volt = width / math.sqrt(12)
dev17 = centers - E_volt                                   # 第 3 节那 17 条：中点离期望多远
one_side = [round(float(d) ** 2, 2) for d in dev17 if d > 1e-9]
var17 = float(np.sum(dev17**2 * strip_p))
print(f"\n电压 U({lo}, {hi})：还用第 3 节那 17 条。中点离 {num(E_volt)} 的偏差是 0、±0.1、±0.2、……、±{num(float(dev17.max()), 1)}")
print("  一侧的偏差²：" + " + ".join(num(v, 2) for v in one_side) + f" = {num(sum(one_side), 2)}；两侧一共 {num(2 * sum(one_side), 2)}（中间那条是 0）")
print(f"  17 条概率相等 → 方差 = {num(2 * sum(one_side), 2)} / 17 = {var17:.2f}，标准差 = √{var17:.2f} = {math.sqrt(var17):.2f} V")
fine = lo + 0.001 * (np.arange(int(round(width / 0.001))) + 0.5)       # 切成宽 0.001 的细条
var_volt_strips = float(np.sum((fine - E_volt) ** 2 * density * 0.001))
print(f"  切得更细（每条宽 0.001）：方差 {var_volt_strips:.4f}，开根号 {math.sqrt(var_volt_strips):.4f}；"
      f"现成公式 宽度² / 12 = {num(width)}² / 12 = {width**2 / 12:.4f}")
print(f"  第 2 节那 10 万个样本直接量：标准差 = {float(volts.std()):.4f}")
check("17 条手算：一侧偏差² 的和 2.04，两侧 4.08，方差 4.08/17 = 0.24，标准差 0.49 V",
      math.isclose(sum(one_side), 2.04) and round(var17, 2) == 0.24 and round(math.sqrt(var17), 2) == 0.49)
check("字面值重算：0.01 + 0.04 + 0.09 + 0.16 + 0.25 + 0.36 + 0.49 + 0.64 = 2.04；2 × 2.04 = 4.08；4.08 / 17 = 0.24；√0.24 → 0.49",
      round(0.01 + 0.04 + 0.09 + 0.16 + 0.25 + 0.36 + 0.49 + 0.64, 2) == 2.04 and round(2 * 2.04, 2) == 4.08
      and round(4.08 / 17, 2) == 0.24 and round(math.sqrt(0.24), 2) == 0.49)
check("切得越细越准：17 条给 0.24，1700 条给 0.2408，和公式 1.7²/12 = 0.2408 一致；开根号 0.4907",
      round(var_volt_strips, 4) == 0.2408 and round(1.7**2 / 12, 4) == 0.2408 and round(math.sqrt(0.2408), 4) == 0.4907 and round(std_volt, 4) == 0.4907)
check(f"三种办法一致：公式 {std_volt:.4f}、1700 条细条 {math.sqrt(var_volt_strips):.4f}（差不到 0.0001）、10 万个样本 {float(volts.std()):.4f}（差不到 0.005）",
      abs(math.sqrt(var_volt_strips) - std_volt) < 1e-4 and abs(float(volts.std()) - std_volt) < 0.005)
check("正文引用的数：10 万个样本直接量，标准差也是 0.49", round(float(volts.std()), 2) == 0.49)
var_n = float(np.var([1.0, 3.0]))                         # numpy 默认除以 n：本章的公式
var_n1 = torch.var(torch.tensor([1.0, 3.0])).item()       # torch 默认除以 n − 1
print(f"\n进阶：同样的 [1, 3]，np.var 给 {var_n:g}（除以 n），torch.var 给 {var_n1:g}（除以 n − 1）；"
      f"torch.var(..., correction=0) 给 {torch.var(torch.tensor([1.0, 3.0]), correction=0).item():g}")
check("代码里的小坑：np.var([1, 3]) = 1（除以 n），torch.var 默认给 2（除以 n − 1），加 correction=0 才是 1",
      var_n == 1 and var_n1 == 2 and torch.var(torch.tensor([1.0, 3.0]), correction=0).item() == 1)
big = np.random.default_rng(6).uniform(lo, hi, size=10_000)
check("样本多了两种算法几乎没差别：1 万个电压，除以 n 和除以 n − 1 的标准差相差不到 0.0001",
      abs(float(big.std()) - float(big.std(ddof=1))) < 1e-4)

print("\n另一种算法：方差 = 平方的平均 − 平均的平方。先把 (x − 3.5)² 展开成 x² − 7x + 12.25：")
table(["点数 x", "(x − 3.5)²", "x² − 7x + 12.25"], [[int(x), num((x - 3.5) ** 2), num(x * x - 7 * x + 12.25)] for x in faces])
check("展开括号：六个面上 (x − 3.5)² 都等于 x² − 7x + 12.25；x = 1 时两边都是 6.25（1 − 7 + 12.25 = 6.25）",
      all(math.isclose((x - 3.5) ** 2, x * x - 7 * x + 12.25) for x in faces) and (1 - 3.5) ** 2 == 6.25 == 1 - 7 + 12.25)
check("三块各自求平均：E[X²] − 7 × E[X] + 12.25 = 方差；字面值 15.17 − 24.5 + 12.25 = 2.92；−7 × 3.5 = −24.5；7 = 2 × 3.5，12.25 = 3.5²",
      math.isclose(E_sq - 7 * E_dice + 12.25, var_dice) and round(15.17 - 24.5 + 12.25, 2) == 2.92 and -7 * 3.5 == -24.5
      and 7 == 2 * 3.5 and 12.25 == 3.5**2 and math.isclose(E_sq - 2 * E_dice * E_dice + E_dice**2, E_sq - E_dice**2))
new_vals = 2 * faces + 3
new_dev = new_vals - float(np.mean(new_vals))
var_new = float(np.mean(new_dev**2))
print(f"平移和拉伸：2X + 3 的六个值 {new_vals.tolist()}，平均 {num(float(np.mean(new_vals)))}，偏差 {[int(d) for d in new_dev]}，"
      f"偏差² 之和 {num(float(np.sum(new_dev**2)))}，方差 70/6 = {var_new:.2f}，标准差 {math.sqrt(var_new):.2f}")
check("2X + 3：新的偏差 −5、−3、−1、1、3、5，正好是原来的 2 倍；偏差² 25、9、1、1、9、25，加起来 70",
      [int(d) for d in new_dev] == [-5, -3, -1, 1, 3, 5] and np.allclose(new_dev, 2 * dev)
      and [int(d * d) for d in new_dev] == [25, 9, 1, 1, 9, 25] and math.isclose(float(np.sum(new_dev**2)), 70))
check("方差乘 2² = 4、标准差乘 2：70/6 → 11.67，70 = 4 × 17.5；√11.67 → 3.42 = 2 × 1.71；平移的 +3 不改变方差",
      math.isclose(var_new, 4 * var_dice) and round(70 / 6, 2) == 11.67 and 70 == 4 * 17.5 and round(math.sqrt(11.67), 2) == 3.42
      and round(2 * 1.71, 2) == 3.42 and math.isclose(math.sqrt(var_new), 2 * std_dice)
      and math.isclose(float(np.var(faces + 3)), var_dice) and math.isclose(float(np.var(-faces)), var_dice) and abs(-2) == 2)

mids10 = (np.arange(10) + 0.5) / 10                        # 进阶：U(0, 1) 切成 10 条、100 条，算“平方的平均”
mids100 = (np.arange(100) + 0.5) / 100
sq10, sq100 = float(np.sum(mids10**2 * 0.1)), float(np.sum(mids100**2 * 0.01))
print(f"进阶：U(0, 1) 的平方的平均——10 条：{sq10:.4f}，100 条：{sq100:.6f}（越切越接近 1/3）；1/3 − 1/4 = 1/12 = {1 / 12:.4f}；"
      f"× 1.7² = 2.89 → {2.89 / 12:.4f}")
check("宽度² ÷ 12 的来历：10 条得 0.3325，100 条得 0.333325（→ 1/3）；1/3 − 1/4 = 4/12 − 3/12 = 1/12 ≈ 0.0833；2.89 ÷ 12 → 0.2408；1.7² = 2.89",
      round(sq10, 4) == 0.3325 and round(sq100, 6) == 0.333325 and math.isclose(1 / 3 - 1 / 4, 1 / 12) and round(1 / 12, 4) == 0.0833
      and round(2.89 / 12, 4) == 0.2408 and round(1.7**2, 2) == 2.89 and math.isclose(4 / 12 - 3 / 12, 1 / 12))
check("自测：等概率的 2、4、9 → 均值 5，方差 (9 + 1 + 16)/3 ≈ 8.67，标准差 ≈ 2.94（字面值：26/3 → 8.67，√8.67 → 2.94）",
      np.mean([2, 4, 9]) == 5 and round(float(np.var([2, 4, 9])), 2) == 8.67 and round(float(np.std([2, 4, 9])), 2) == 2.94
      and round(26 / 3, 2) == 8.67 and round(math.sqrt(8.67), 2) == 2.94)

# ---------------------------------------------------------------------------
banner("4b. 画图：figures/ch04_mean_and_spread.png（期望是平衡点，标准差是散开的宽度）")
fig = plt.figure(figsize=(9.4, 13.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.74, wspace=0.22, left=0.10, right=0.96, top=0.845, bottom=0.115)
fig.suptitle("期望说“中心在哪”，标准差说“散得多开”", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.975)
class_axes = []
for k, (scores, title_, color) in enumerate([(class_a, "甲班全是 70 分", BLUE), (class_b, "乙班一半 40、一半 100", GREEN)]):
    ax = fig.add_subplot(gs[0, k])
    data_axes(ax, "分数", "占全班的比例" if k == 0 else "")
    values, shares = np.unique(scores, return_counts=True)
    ax.bar(values, shares / len(scores), width=7, color=color, zorder=3)
    ax.axvline(70, color=ORANGE, lw=2.5, ls="--", zorder=4)
    ax.set_xlim(20, 120)
    ax.set_ylim(0, 1.25)
    ax.set_xticks([40, 70, 100])
    ax.set_yticks([0, 0.5, 1])
    ax.text(73, 1.12, "平均 70", fontsize=FS_SMALL, color=ORANGE, ha="left", va="center", bbox=WHITE_BOX)
    ax.set_title(f"{title_}\n标准差 {num(float(scores.std()))} 分", fontsize=FS_SMALL, color=color, pad=10, linespacing=1.4)
    class_axes.append(ax)
panel_title(fig, class_axes, "① 平均一样，散的程度完全不同", pad=0.012)
panel_note(fig, class_axes, "橙色虚线是平均分。乙班每个人都离平均 30 分：标准差就是 30。")

ax = fig.add_subplot(gs[1, :])
data_axes(ax, "骰子的点数", "概率")
ax.bar(faces, p_dice, width=0.62, color="#dbe7f3", edgecolor=BLUE, lw=2, zorder=3)
ax.axvline(E_dice, color=ORANGE, lw=3, zorder=4)
ax.set_xlim(0.2, 6.8)
ax.set_ylim(0, 0.36)
ax.set_xticks(faces)
ax.set_yticks([0, 1 / 6])
ax.set_yticklabels(["0", "1/6"])
ax.text(E_dice + 0.08, 0.335, f"期望 {num(E_dice)}（不是任何一面）", fontsize=FS_SMALL, color=ORANGE, va="center", bbox=WHITE_BOX, zorder=6)
span_y = 0.245
arrow(ax, (E_dice, span_y), (E_dice + std_dice, span_y), GREEN, lw=3)
arrow(ax, (E_dice, span_y), (E_dice - std_dice, span_y), GREEN, lw=3)
ax.text(E_dice, span_y + 0.028, f"左右各一个标准差：{std_dice:.2f}", fontsize=FS_SMALL, color=GREEN, ha="center", va="center", bbox=WHITE_BOX, zorder=6)
for x, d in zip(faces, dev):
    ax.text(x, 1 / 6 + 0.012, f"偏差² {num(d * d)}", fontsize=13, color=MUTED, ha="center", va="bottom")
panel_title(fig, [ax], f"② 骰子：期望 {num(E_dice)}，方差 17.5 ÷ 6 = {var_dice:.2f}，标准差 {std_dice:.2f}")
panel_note(fig, [ax], "六根柱子一样高（各 1/6）。橙线是期望；绿色箭头的长度是标准差，\n两根都按横轴的刻度画：从 3.5 出发各伸出 1.71。")
savefig(fig, "ch04_mean_and_spread")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("5. 大数定律：不知道概率，就多抽几次取平均")
run = np.random.default_rng(0).uniform(lo, hi, size=100_000)       # 一条抽样流：一个接一个地抽 10 万个电压
running = np.cumsum(run) / np.arange(1, run.size + 1)              # 抽到第 n 个时，“到目前为止的平均”
rows = [[n, float(running[n - 1]), float(running[n - 1] - E_volt)] for n in [1, 10, 100, 1000, 10000, 100000]]
lln_rows = rows
print(f"从 U({lo}, {hi}) 一个接一个地抽电压，抽到第 n 个时看一眼“到目前为止的平均”（理论期望 {num(E_volt)}）：")
table(["样本数 n", "样本均值", "与期望的差"], rows, floatfmt=".4f")
check("正文引用的数：抽 1 个差 0.23，抽 10 万个只差 0.0007", round(rows[0][2], 2) == 0.23 and round(rows[-1][2], 4) == -0.0007)
check(f"十万个样本的均值 {rows[-1][1]:.4f} 和期望 {num(E_volt)} 相差不到 0.01", abs(rows[-1][2]) < 0.01)
check("误差并不是整整齐齐地缩小：样本从 10 个加到 100 个（10 倍），误差几乎没动（0.086 → 0.082）；而抽 100 个“一般”只差 0.049",
      round(rows[1][2], 3) == 0.086 and round(rows[2][2], 3) == 0.082 and abs(rows[2][2]) > std_volt / math.sqrt(100)
      and round(std_volt / math.sqrt(100), 3) == 0.049)
check("字面值重算：7.5828 − 7.35 → 0.2328；7.4321 − 7.35 → 0.0821；7.3493 − 7.35 → −0.0007",
      round(7.5828 - 7.35, 4) == 0.2328 and round(7.4321 - 7.35, 4) == 0.0821 and round(7.3493 - 7.35, 4) == -0.0007
      and [f"{r[1]:.4f}" for r in (rows[0], rows[2], rows[5])] == ["7.5828", "7.4321", "7.3493"])

N_REPEAT = 2000
rng_rep = np.random.default_rng(5)
spread_rows, mean_sets = [], {}
for n in [1, 4, 100, 10000]:
    means = rng_rep.uniform(lo, hi, size=(N_REPEAT, n)).mean(axis=1)    # 整个“抽 n 个取平均”重做 2000 遍
    mean_sets[n] = means
    spread_rows.append([n, float(means.std()), std_volt / math.sqrt(n), math.sqrt(n)])
printed = [float(f"{r[1]:.4g}") for r in spread_rows]                 # 表里印出来的（四位有效数字的）那几个数
shrink = [printed[0] / v for v in printed]                           # 用印出来的数算“缩小了几倍”：读者按计算器得到的就是它
print(f"\n把“抽 n 个取平均”整个重做 {N_REPEAT} 遍，看这 {N_REPEAT} 个平均值自己散得多开：")
table(["样本数 n", "平均值的标准差（实测）", "比只抽 1 个时缩小了几倍"],
      [[r[0], r[1], "1" if k == 0 else f"{shrink[k]:.1f}"] for k, r in enumerate(spread_rows)], floatfmt=".4g")
print(f"按 σ / √n 算（σ = {std_volt:.4f}）：" + "、".join(f"{r[2]:.4g}" for r in spread_rows))
check("正文引用的数：缩小了 2.0、10.1、99.5 倍（字面值：0.492/0.2445、0.492/0.0487、0.492/0.004943）",
      [f"{v:.1f}" for v in shrink[1:]] == ["2.0", "10.1", "99.5"] and printed == [0.492, 0.2445, 0.0487, 0.004943]
      and round(0.492 / 0.2445, 1) == 2.0 and round(0.492 / 0.0487, 1) == 10.1 and round(0.492 / 0.004943, 1) == 99.5)
check("正文引用的数：0.4907 ÷ √n 依次是 0.4907、0.2454、0.04907、0.004907", [f"{r[2]:.4g}" for r in spread_rows] == ["0.4907", "0.2454", "0.04907", "0.004907"])
check("字面值重算（四舍五入）：0.4907 ÷ 2 → 0.2454；0.4907 ÷ 10 = 0.04907；0.4907 ÷ 100 = 0.004907",
      (Decimal("0.4907") / 2).quantize(Decimal("0.0001"), ROUND_HALF_UP) == Decimal("0.2454")
      and Decimal("0.4907") / 10 == Decimal("0.04907") and Decimal("0.4907") / 100 == Decimal("0.004907"))
check("自测：σ = 2，要平均值的标准差降到 0.2 得抽 100 个；降到 0.02 得抽 10000 个", 2 / math.sqrt(100) == 0.2 and 2 / math.sqrt(10000) == 0.02)
check("样本数 × 4，平均值的散布 ÷ 2；× 100，÷ 10；× 10000，÷ 100（和理论 σ/√n 相差不到 6%）",
      all(abs(r[1] - r[2]) / r[2] < 0.06 for r in spread_rows))
check("σ 大一倍、样本要多 4 倍：(2σ)/√(4n) = 2σ/(2√n) = σ/√n（σ = 0.4907、n = 100 代进去核对）",
      math.isclose(2 * std_volt / math.sqrt(4 * 100), std_volt / math.sqrt(100)) and 2 * 2 == 4)
coins = np.array([[a, b] for a in (1, -1) for b in (1, -1)])            # 进阶：两枚 ±1 硬币的四种结果
coin_sum = coins.sum(axis=1)
cross = coins[:, 0] * coins[:, 1]
print(f"\n进阶：两枚 ±1 硬币。四种结果的和 {coin_sum.tolist()}，和的平方 {(coin_sum**2).tolist()}，"
      f"和的方差 = {float(np.mean(coin_sum**2)):g}；交叉项“第一枚 × 第二枚” {cross.tolist()}，平均 {float(cross.mean()):g}")
check("一枚 ±1 硬币：平均 0，方差 1；两枚的和：方差 (4 + 0 + 0 + 4)/4 = 2 = 1 + 1；交叉项 +1、−1、−1、+1 平均是 0",
      float(np.mean([1, -1])) == 0 and float(np.var([1, -1])) == 1 and float(np.mean(coin_sum**2)) == 2 == (4 + 0 + 0 + 4) / 4
      and cross.tolist() == [1, -1, -1, 1] and float(cross.mean()) == 0 and coin_sum.tolist() == [2, 0, 0, -2])
same = np.array([2, -2])                                                 # 第二枚总跟第一枚一样：和只有 2 和 −2
check("n 枚的和方差是 n、标准差 √n；平均值的标准差 √n ÷ n = 1/√n（n = 100：10 ÷ 100 = 1/10）；两枚总一样时方差是 4 不是 2",
      all(math.isclose(float(np.var(np.random.default_rng(11).choice([1, -1], size=(20000, n)).sum(axis=1))), n, rel_tol=0.05)
          for n in (2, 4, 9))
      and math.isclose(math.sqrt(100) / 100, 1 / math.sqrt(100)) and 10 / 100 == 0.1 and float(np.mean(same**2)) == 4)
check("正文引用的数：0.49 → 0.245 → 0.049 → 0.0049",
      round(spread_rows[0][1], 2) == 0.49 and round(spread_rows[1][1], 3) == 0.245
      and round(spread_rows[2][1], 3) == 0.049 and round(spread_rows[3][1], 4) == 0.0049)
n_records = 4096 * 24
print(f"\n项目里一次迭代有 4096 × 24 = {n_records:,} 条记录；√{n_records} = {math.sqrt(n_records):.1f}。")
print("但同一只机器人相邻两帧很像，它们不是独立的样本，误差不会真的缩到 1/313。")
check("4096 × 24 = 98,304，√98304 ≈ 313.5（所以“1/313”）", n_records == 98304 and round(math.sqrt(n_records), 1) == 313.5)

# ---------------------------------------------------------------------------
banner("5b. 画图：figures/ch04_law_of_large_numbers.png（抽得越多，平均值越稳）")
if not finite(lo, hi, std_volt):
    print("  数字不是有限值，这张图跳过。")
else:
    fig = plt.figure(figsize=(9.4, 16.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.6, 5.2], hspace=0.42, left=0.13, right=0.95, top=0.915, bottom=0.085)
    gs_hist = gs[1].subgridspec(3, 1, hspace=0.34)
    fig.suptitle("多抽几次取平均：样本越多，平均值越贴近期望", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)

    ax = fig.add_subplot(gs[0])
    data_axes(ax, "已经抽了几个（对数刻度：每一大格 ×10）", "到目前为止的平均（伏）")
    ax.plot(np.arange(1, run.size + 1), running, color=BLUE, lw=2.6, zorder=3)      # 就是上面那张表读的同一条流
    ax.axhline(E_volt, color=ORANGE, lw=2.4, ls="--", zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 130_000)
    ax.set_ylim(E_volt - 0.42 * width, E_volt + 0.42 * width)
    ax.set_xticks([1, 10, 100, 1000, 10_000, 100_000])
    ax.set_xticklabels(["1", "10", "100", "1000", "1 万", "10 万"])
    ax.minorticks_off()
    ax.text(1.0e5, E_volt + 0.035 * width, f"期望 {num(E_volt)}", fontsize=FS_SMALL, color=ORANGE, ha="right", va="bottom")
    ax.plot(1, running[0], "o", color=BLUE, ms=9, zorder=5)
    ax.text(1.25, running[0], f"只抽 1 个：{running[0]:.4f}", fontsize=FS_SMALL, color=BLUE, va="center", bbox=WHITE_BOX)
    ax.plot(run.size, running[-1], "o", color=BLUE, ms=9, zorder=5)
    ax.text(1.0e5, running[-1] - 0.06 * width, f"抽了 10 万个：{running[-1]:.4f}", fontsize=FS_SMALL, color=BLUE, ha="right", va="top", bbox=WHITE_BOX)
    for n_mark in (10, 100, 1000, 10_000):                             # 表里的另外四行：同一条线上的四个点
        ax.plot(n_mark, running[n_mark - 1], "o", color=BLUE, ms=7, zorder=5)
    panel_title(fig, [ax], "① 一次抽样：一边抽、一边算“到目前为止的平均”")
    panel_note(fig, [ax], "圆点就是正文六行表里的六个数。开头上蹿下跳，后来贴着橙色虚线走。\n横轴是对数刻度：1、10、100……每往右一大格，样本数乘 10。")

    hist_axes = []
    edges = np.linspace(lo, hi, 86)
    for k, (n, color) in enumerate([(1, MUTED), (100, BLUE), (10000, GREEN)]):
        ax = fig.add_subplot(gs_hist[k])
        data_axes(ax, "2000 个平均值落在哪（伏）" if k == 2 else "", "")
        ax.hist(mean_sets[n], bins=edges, color=color, zorder=3)
        ax.axvline(E_volt, color=ORANGE, lw=2, ls="--", zorder=4)
        ax.set_xlim(lo - 0.02 * width, hi + 0.02 * width)
        ax.set_yticks([])
        ax.set_ylim(0, ax.get_ylim()[1] * 1.55)                        # 给两个标签留出头部空间，不压在柱子上
        ax.text(0.015, 0.80, f"每次抽 {n} 个", transform=ax.transAxes, fontsize=FS_SMALL, color=color, fontweight="bold", va="center", bbox=WHITE_BOX)
        ax.text(0.985, 0.80, f"平均值的标准差：{float(mean_sets[n].std()):.2g}", transform=ax.transAxes, fontsize=FS_SMALL,
                color=color, ha="right", va="center", bbox=WHITE_BOX)
        hist_axes.append(ax)
    panel_title(fig, hist_axes, "② 把整个实验重做 2000 遍：平均值自己散得多开")
    panel_note(fig, hist_axes, "柱高 = 落在这一格里的个数（不是密度）；三幅纵向各自缩放，所以别比高矮和面积，\n只比宽窄：每次抽的样本 × 100，平均值的标准差就 ÷ 10：0.49 → 0.049 → 0.0049。")
    savefig(fig, "ch04_law_of_large_numbers")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("6. 高斯分布：先手算一个点的高度，再核对“面积是 1”")
# 这口钟的中心和标准差。“改一改”第 2 条改 BELL_SIGMA 这一行。
BELL_MU = 0.0
BELL_SIGMA = 1.0  # TWEAK-2: 2.0


def gauss_pdf(x, mu, sigma):
    """钟的高度：exp(−(x − μ)² / (2σ²)) 再除以 σ√(2π)。x 可以是一个数，也可以是一串数。"""
    return np.exp(-((np.asarray(x, dtype=float) - mu) ** 2) / (2 * sigma**2)) / (sigma * SQRT_2PI)


x_hand = BELL_MU + 1.0
expo = -((x_hand - BELL_MU) ** 2) / (2 * BELL_SIGMA**2)
shape_part = math.exp(expo)
p_hand = shape_part / (BELL_SIGMA * SQRT_2PI)
print(f"手算 x = {num(x_hand)} 处的高度（μ = {num(BELL_MU)}，σ = {num(BELL_SIGMA)}）：")
print(f"  ① 离中心多远：{num(x_hand - BELL_MU)}；平方 {num((x_hand - BELL_MU) ** 2)}；除以 2σ² = {num(2 * BELL_SIGMA**2)} → {num((x_hand - BELL_MU) ** 2 / (2 * BELL_SIGMA**2))}；取负号、过 exp：exp({num(expo)}) = {shape_part:.4f}")
print(f"  ② 除以 σ × √(2π) = {num(BELL_SIGMA)} × {SQRT_2PI:.4f} = {BELL_SIGMA * SQRT_2PI:.4f} → {shape_part:.4f} / {BELL_SIGMA * SQRT_2PI:.4f} = {p_hand:.4f}")
check("手算：exp(−0.5) = 0.6065，√(2π) = 2.5066，高度 0.6065 / 2.5066 = 0.2420",
      round(shape_part, 4) == 0.6065 and round(SQRT_2PI, 4) == 2.5066 and round(p_hand, 4) == 0.2420)

bell = torch.distributions.Normal(BELL_MU, BELL_SIGMA)
rows = []
for k in [-2, -1, 0, 1, 2]:
    x = BELL_MU + k * BELL_SIGMA
    rows.append([num(x), float(gauss_pdf(x, BELL_MU, BELL_SIGMA)), bell.log_prob(torch.tensor(float(x))).exp().item()])
print("\n公式手写 与 torch 现成的 Normal 对比：")
table(["x", "高度 p(x) 手写", "高度 p(x) torch"], rows, floatfmt=".4f")
check("手写高斯 = torch.distributions.Normal", all(abs(r[1] - r[2]) < 1e-6 for r in rows))
check("正文引用的数：x = −2、−1、0、1、2 处依次是 0.0540、0.2420、0.3989、0.2420、0.0540（六位：0.053991、0.241971、0.398942）",
      [f"{r[1]:.4f}" for r in rows] == ["0.0540", "0.2420", "0.3989", "0.2420", "0.0540"]
      and round(rows[2][1], 6) == 0.398942 and round(rows[1][1], 6) == round(rows[3][1], 6) == 0.241971
      and round(rows[0][1], 6) == round(rows[4][1], 6) == 0.053991)
check("字面值重算：Math.exp(-0.5) → 0.6065；0.6065 / 2.5066 → 0.2420；1 / 2.5066 → 0.3989；√6.2832 → 2.5066",
      round(math.exp(-0.5), 4) == 0.6065 and round(0.6065 / 2.5066, 4) == 0.2420 and round(1 / 2.5066, 4) == 0.3989
      and round(math.sqrt(6.2832), 4) == 2.5066)

STRIP = 0.1
strip_mid = BELL_MU + BELL_SIGMA * (np.arange(-50, 50) + 0.5) * STRIP          # 从 μ − 5σ 到 μ + 5σ 切成 100 条
strip_w = BELL_SIGMA * STRIP
raw_area = float(np.sum(np.exp(-((strip_mid - BELL_MU) ** 2) / (2 * BELL_SIGMA**2)) * strip_w))   # 不带常数的钟
area = float(np.sum(gauss_pdf(strip_mid, BELL_MU, BELL_SIGMA) * strip_w))
in1 = np.abs(strip_mid - BELL_MU) < BELL_SIGMA
in2 = np.abs(strip_mid - BELL_MU) < 2 * BELL_SIGMA
area1 = float(np.sum(gauss_pdf(strip_mid[in1], BELL_MU, BELL_SIGMA) * strip_w))
area2 = float(np.sum(gauss_pdf(strip_mid[in2], BELL_MU, BELL_SIGMA) * strip_w))
print(f"\n把钟切成宽 {num(strip_w)} 的细条，每条面积 = 高度 × 宽度，再全部加起来：")
print(f"  不带常数的钟 exp(−(x−μ)²/(2σ²))：面积 = {raw_area:.4f}   （σ × √(2π) = {BELL_SIGMA * SQRT_2PI:.4f}）")
print(f"  除以这个常数之后：面积 = {area:.4f}")
print(f"  只加 μ ± 1σ 以内的条：{area1:.4f}；μ ± 2σ 以内：{area2:.4f}")
check("不带常数的钟面积 2.5066 = √(2π)——所以要除以它，面积才是 1", round(raw_area, 4) == 2.5066 and abs(area - 1) < 1e-4)
check("μ ± 1σ 内的面积约 68%，μ ± 2σ 内约 95%", round(area1, 2) == 0.68 and round(area2, 2) == 0.95)
check("正文引用的数：细条加出来是 0.6829 和 0.9546（精确值 0.6827、0.9545，细条越细越接近）",
      round(area1, 4) == 0.6829 and round(area2, 4) == 0.9546 and abs(area1 - 0.6827) < 5e-4 and abs(area2 - 0.9545) < 5e-4)

var_strips = float(np.sum((strip_mid - BELL_MU) ** 2 * gauss_pdf(strip_mid, BELL_MU, BELL_SIGMA) * strip_w))
mean_strips = float(np.sum(strip_mid * gauss_pdf(strip_mid, BELL_MU, BELL_SIGMA) * strip_w))
print(f"式子里的 μ、σ 真的是 4.3、4.4 节的期望和标准差吗？还是切细条：Σ x × 条的面积 = {num(mean_strips)}；"
      f"Σ (x − μ)² × 条的面积 = {var_strips:.4f}，开根号 {math.sqrt(var_strips):.4f}")
check(f"细条核对：这口钟的期望是 μ = {num(BELL_MU)}，方差是 σ² = {num(BELL_SIGMA**2)}（容差 0.001）——分母里那个 2 保证了这一点",
      abs(mean_strips - BELL_MU) < 1e-3 and abs(var_strips - BELL_SIGMA**2) < 1e-3 * BELL_SIGMA**2)
check("正文引用的数：细条加出来的方差 1.0000", f"{var_strips:.4f}" == "1.0000")
z = np.random.default_rng(3).normal(BELL_MU, BELL_SIGMA, size=100_000)
frac1 = float(np.mean(np.abs(z - BELL_MU) <= BELL_SIGMA))
frac2 = float(np.mean(np.abs(z - BELL_MU) <= 2 * BELL_SIGMA))
print(f"真的从这口钟里抽 10 万个数：落在 ±1σ 内的占 {frac1:.3f}，±2σ 内的占 {frac2:.3f}")
check("正文引用的数：10 万个样本里占 0.681 和 0.955", round(frac1, 3) == 0.681 and round(frac2, 3) == 0.955)
check("10 万个样本：±1σ 内约 68%、±2σ 内约 95%（容差 0.01）", abs(frac1 - 0.6827) < 0.01 and abs(frac2 - 0.9545) < 0.01)

narrow_bell_peak = float(gauss_pdf(0.0, 0.0, 0.1))
print(f"\n窄的钟 σ = 0.1：中心高度 = 1 / (0.1 × 2.5066) = {narrow_bell_peak:.3f}——又超过 1 了；面积照样是 1")
check("σ = 0.1 的钟中心密度 3.989", round(narrow_bell_peak, 3) == 3.989)
check("正文引用的数：2π ≈ 6.2832；打分的 σ 要是分布的 σ 的 √2 ≈ 1.41 倍；图 ② 橙色那条 0.30 × 0.5 = 0.15（字面值也成立）",
      round(2 * math.pi, 4) == 6.2832 and round(math.sqrt(2), 2) == 1.41
      and round(float(gauss_pdf(0.75, 0.0, 1.0)), 2) == 0.30 and round(float(gauss_pdf(0.75, 0.0, 1.0)) * 0.5, 2) == 0.15 and round(0.30 * 0.5, 2) == 0.15)
draw_mid_std = (np.arange(-8, 8) + 0.5) * 0.5                       # 图 ② 画出来的 16 条（宽 0.5，从 −4 到 4）
draw_area_std = float(np.sum(gauss_pdf(draw_mid_std, 0.0, 1.0) * 0.5))
check("正文引用的数：图 ② 画出来的 16 条加起来 0.9999；实验用的 0.1 宽细条（−5 到 5）加起来 1.0000", f"{draw_area_std:.4f}" == "0.9999" and f"{area:.4f}" == "1.0000")
check("字面值重算：1 / (0.1 × 2.5066) → 3.989；2 × 0.01 = 0.02", round(1 / (0.1 * 2.5066), 3) == 3.989 and round(2 * 0.01, 2) == 0.02)
check("通勤那口钟 μ = 35、σ = 5：±1σ 是 30–40，±2σ 是 25–45，50 分钟在 3 个 σ 处", (35 - 5, 35 + 5, 35 - 10, 35 + 10) == (30, 40, 25, 45) and (50 - 35) / 5 == 3)
check("自测：密度 2 不是 200% 的概率。宽 0.01 的一小段，概率约 2 × 0.01 = 0.02", math.isclose(2 * 0.01, 0.02))
sigma_demo = 0.5
s_kernel = math.sqrt(2) * sigma_demo
xs_demo = np.array([0.0, 0.3, 1.0])
check("和第 1 章钟形打分同形：exp(−x²/s²) 与 exp(−x²/(2σ²)) 在 s = √2 σ 时处处相等",
      np.allclose(np.exp(-(xs_demo**2) / s_kernel**2), np.exp(-(xs_demo**2) / (2 * sigma_demo**2))))
kx = (np.arange(-600, 600) + 0.5) * 0.01                          # σ = 1 的“打分钟” exp(−x²)，当成分布量它的标准差
kw = np.exp(-(kx**2))
kernel_std = math.sqrt(float(np.sum(kw * kx**2) / np.sum(kw)))
print(f"σ = 1 的打分钟 exp(−x²) 要是当成分布：细条量出的标准差 = {kernel_std:.4f}（1 ÷ √2 = {1 / math.sqrt(2):.4f}）")
check("开根号的规矩 √(4 × 9) = √36 = 6 = 2 × 3；打分钟 exp(−x²) 的标准差是 1/√2 = 0.7071（细条量出来也是 0.7071）；2 × 1² = 2",
      math.sqrt(4 * 9) == 6 == 2 * 3 and round(kernel_std, 4) == 0.7071 and round(1 / math.sqrt(2), 4) == 0.7071 and 2 * 1**2 == 2)

# ---------------------------------------------------------------------------
banner("6b. 画图：figures/ch04_gaussian_68_95.png（一口钟：高度、面积、68% 和 95%）")
fig = plt.figure(figsize=(9.4, 18.0))
gs = fig.add_gridspec(3, 1, hspace=0.60, left=0.13, right=0.95, top=0.92, bottom=0.085)
fig.suptitle("高斯分布：μ 定中心，σ 管有多宽，面积一共是 1", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
xs = np.linspace(BELL_MU - 4 * BELL_SIGMA, BELL_MU + 4 * BELL_SIGMA, 600)
ys = gauss_pdf(xs, BELL_MU, BELL_SIGMA)
peak = float(gauss_pdf(BELL_MU, BELL_MU, BELL_SIGMA))
sigma_ticks = [BELL_MU + k * BELL_SIGMA for k in range(-4, 5)]


def bell_axes(ax, ylabel="密度（钟的高度）"):
    data_axes(ax, f"抽到的数 x（这口钟 μ = {num(BELL_MU)}，σ = {num(BELL_SIGMA)}）", ylabel)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(0, peak * 1.42)
    ax.set_xticks(sigma_ticks)
    ax.set_xticklabels([num(t) for t in sigma_ticks])


ax = fig.add_subplot(gs[0])
bell_axes(ax)
ax.plot(xs, ys, color=BLUE, lw=3.5, zorder=3)
ax.plot(BELL_MU, peak, "o", color=ORANGE, ms=10, zorder=5)
ax.text(BELL_MU + 0.18 * BELL_SIGMA, peak * 1.06, f"中心最高：{peak:.4f}", fontsize=FS_SMALL, color=ORANGE, va="bottom")
ax.plot([x_hand, x_hand], [0, p_hand], color=ORANGE, lw=2, ls=":", zorder=4)
ax.plot(x_hand, p_hand, "o", color=ORANGE, ms=10, zorder=5)
ax.text(x_hand + 0.18 * BELL_SIGMA, p_hand * 1.04, f"x = {num(x_hand)}：exp({num(expo)}) ÷ {BELL_SIGMA * SQRT_2PI:.4f}\n= {shape_part:.4f} ÷ {BELL_SIGMA * SQRT_2PI:.4f} = {p_hand:.4f}".replace("-", "−"),
        fontsize=FS_SMALL, color=ORANGE, va="bottom", linespacing=1.35)
panel_title(fig, [ax], "① 钟的高度：离中心越远越低（橙点是正文手算的那个点）")
panel_note(fig, [ax], "形状就是第 1 章的钟形打分；多出来的“÷ 2.5066”只管把面积调成 1。")

ax = fig.add_subplot(gs[1])
bell_axes(ax)
DRAW_STRIP = 0.5 * BELL_SIGMA                     # 画图用粗一点的条，看得清；算数用的是宽 0.1σ 的细条
draw_mid = BELL_MU + BELL_SIGMA * (np.arange(-8, 8) + 0.5) * 0.5
ax.bar(draw_mid, gauss_pdf(draw_mid, BELL_MU, BELL_SIGMA), width=DRAW_STRIP, color="#dbe7f3", edgecolor=BLUE, lw=1.4, zorder=3)
ax.plot(xs, ys, color=BLUE, lw=2.5, zorder=4)
hot = BELL_MU + 0.75 * BELL_SIGMA
ax.bar([hot], [float(gauss_pdf(hot, BELL_MU, BELL_SIGMA))], width=DRAW_STRIP, color=ORANGE, alpha=0.85, zorder=5)
ax.annotate(f"这一条：高度 {float(gauss_pdf(hot, BELL_MU, BELL_SIGMA)):.2f} × 宽度 {num(DRAW_STRIP)}\n= 面积 {float(gauss_pdf(hot, BELL_MU, BELL_SIGMA)) * DRAW_STRIP:.2f}",
            xy=(hot, float(gauss_pdf(hot, BELL_MU, BELL_SIGMA)) * 0.6), xytext=(BELL_MU + 1.55 * BELL_SIGMA, peak * 0.95),
            fontsize=FS_SMALL, color=ORANGE, va="center", linespacing=1.35,
            arrowprops=dict(arrowstyle="-", lw=1.4, color=ORANGE, shrinkA=4, shrinkB=2))
draw_area = float(np.sum(gauss_pdf(draw_mid, BELL_MU, BELL_SIGMA) * DRAW_STRIP))
ax.text(xs[0] + 0.15 * BELL_SIGMA, peak * 1.25, f"图上这 16 条的面积加起来 = {draw_area:.4f}", fontsize=FS_STEP, color=INK, va="center")
panel_title(fig, [ax], "② 面积 = 把钟切成细条，条的“高度 × 宽度”全加起来")
panel_note(fig, [ax], f"图上为了看得清，条画成了 {num(DRAW_STRIP)} 宽，两头还各漏了一点点尾巴；\n实验用宽 {num(strip_w)} 的细条、多加到两头更远处，得 {area:.4f}。")

ax = fig.add_subplot(gs[2])
bell_axes(ax)
ax.fill_between(xs, ys, where=np.abs(xs - BELL_MU) <= 2 * BELL_SIGMA, color="#cfe0f1", zorder=2)
ax.fill_between(xs, ys, where=np.abs(xs - BELL_MU) <= BELL_SIGMA, color="#7fa9d3", zorder=3)
ax.plot(xs, ys, color=BLUE, lw=3, zorder=4)
for k, frac, y_at, color in [(1, area1, peak * 1.06, BLUE), (2, area2, peak * 1.26, INK)]:
    arrow(ax, (BELL_MU, y_at), (BELL_MU + k * BELL_SIGMA, y_at), color, lw=2.2)
    arrow(ax, (BELL_MU, y_at), (BELL_MU - k * BELL_SIGMA, y_at), color, lw=2.2)
    ax.text(BELL_MU, y_at + peak * 0.015, f"μ ± {k}σ 以内：{frac * 100:.0f}%", fontsize=FS_SMALL, color=color, ha="center", va="bottom", zorder=6)
panel_title(fig, [ax], "③ 两个值得记的面积：±1σ 内约 68%，±2σ 内约 95%")
panel_note(fig, [ax], "深蓝 = 离中心不超过 1 个 σ；浅蓝再往外加到 2 个 σ。\n剩下约 5% 在两条尾巴里：抽到 2σ 以外的数，二十次里才一次。")
savefig(fig, "ch04_gaussian_68_95")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("7. 策略吐出的是一口钟：动作 = 均值 + 标准差 × 噪声")
# 14 个关节共用的初始标准差（项目里 init_std = 1.0）。“改一改”第 3 条改这一行。
POLICY_STD = 1.0  # TWEAK-3: 0.1
torch.manual_seed(0)                   # 固定种子：每次运行抽到同一组数
mean = torch.randn(14) * 0.3           # 假装是策略网络看了某一帧观测后，给 14 个关节算出的均值
std = POLICY_STD * torch.ones(14)      # 14 个标准差：不看观测，是 14 个单独的旋钮
eps = torch.randn(14)                  # 14 个噪声：每个都从“标准钟”（μ = 0，σ = 1）里抽
action = mean + std * eps              # 动作 = 均值 + 标准差 × 噪声
print("先看 1 个关节（迷你版）：")
print(f"  均值 {mean[0]:.4f} + 标准差 {POLICY_STD:g} × 噪声 {eps[0]:.4f} = 动作 {action[0]:.4f}")
print("再看前 4 个关节（一共 14 个，做法相同）：")
table(["关节", "均值 μ", "噪声 ε", f"动作 = μ + {POLICY_STD:g} × ε"],
      [[i, float(mean[i]), float(eps[i]), float(action[i])] for i in range(4)], floatfmt=".4f")
check("动作 = 均值 + 标准差 × 噪声，14 个关节各算各的", torch.allclose(action, mean + POLICY_STD * eps) and action.shape == (14,))
check("正文引用的数：第 0 个关节 0.4623 + 1 × (−1.0712) = −0.6089",
      round(mean[0].item(), 4) == 0.4623 and round(eps[0].item(), 4) == -1.0712 and round(action[0].item(), 4) == -0.6089)
check("正文引用的数：前 4 个动作 −0.6089、0.0347、−1.2200、0.5436",
      [round(v, 4) for v in action[:4].tolist()] == [-0.6089, 0.0347, -1.2200, 0.5436])
check("字面值重算：0.4623 + 1 × (−1.0712) = −0.6089；(−0.6089 − 0.4623) / 1 = −1.0712；35 + 5 × (−1.0712) → 29.6",
      round(0.4623 + 1 * -1.0712, 4) == -0.6089 and round((-0.6089 - 0.4623) / 1, 4) == -1.0712 and round(35 + 5 * -1.0712, 1) == 29.6)
check("表里的数都四舍五入过：关节 2 按字面值相加 −0.6536 + (−0.5663) = −1.2199，和表里的 −1.2200 差在最后一位；另外三行按字面值相加正好对上",
      round(-0.6536 + -0.5663, 4) == -1.2199 and f"{action[2]:.4f}" == "-1.2200" and round(-0.0880 + 0.1227, 4) == 0.0347
      and round(0.1705 + 0.3731, 4) == 0.5436)
check("自测：μ = 0.2、σ = 0.5、ε = −2 → 动作 0.2 + 0.5 × (−2) = −0.8；动作 0.7 对应 ε = (0.7 − 0.2)/0.5 = 1",
      math.isclose(0.2 + 0.5 * -2, -0.8) and math.isclose((0.7 - 0.2) / 0.5, 1.0))
back = (action - mean) / std
check("反过来 (动作 − 均值) / 标准差 就把噪声找回来了（这一步叫归一化）", torch.allclose(back, eps, atol=1e-6))

many_eps = np.random.default_rng(7).normal(0, 1, size=100_000)
many_actions = float(mean[0]) + POLICY_STD * many_eps
print(f"\n同一口钟反复抽 10 万次：平均 {many_actions.mean():.3f}（均值 {mean[0]:.3f}），标准差 {many_actions.std():.3f}（σ = {POLICY_STD:g}）")
check("同一个 ε 放进通勤那口钟：35 + 5 × (−1.0712) = 29.6 分钟", round(35 + 5 * float(eps[0]), 1) == 29.6)
check("正文引用的数：10 万次的平均 0.461、标准差 0.998", round(float(many_actions.mean()), 3) == 0.461 and round(float(many_actions.std()), 3) == 0.998)
check("抽 10 万次：样本的平均 ≈ μ，样本的标准差 ≈ σ（容差 0.01）",
      abs(many_actions.mean() - float(mean[0])) < 0.01 and abs(many_actions.std() - POLICY_STD) < 0.01 * max(1.0, POLICY_STD))
renorm = (many_actions - many_actions.mean()) / many_actions.std()     # 倒着用 4.4 节的两条规则：先减均值，再除以标准差
check("任何一批数减去自己的均值、再除以自己的标准差，均值变 0、标准差变 1（拉伸规则：σ × 1/σ = 1）",
      abs(float(renorm.mean())) < 1e-9 and abs(float(renorm.std()) - 1) < 1e-9 and math.isclose(POLICY_STD * (1 / POLICY_STD), 1.0))

print("\n“给定观测”：同一张网络，观测换了，钟的中心跟着换。迷你网络 μ(s) = 0.5 × s（教学构造）：")
for s_obs in (0.2, -1.0):
    print(f"  观测 s = {s_obs:g} → 钟的中心 μ = 0.5 × {s_obs:g} = {0.5 * s_obs:g}，σ 不变")
check("迷你网络：s = 0.2 → μ = 0.1；s = −1 → μ = −0.5", math.isclose(0.5 * 0.2, 0.1) and 0.5 * -1.0 == -0.5)

layer_sizes = [(61, 512), (512, 256), (256, 128), (128, 14)]
n_mlp = sum(i * o + o for i, o in layer_sizes)
print(f"\nactor 的旋钮：四层网络 {n_mlp:,} 个 + 14 个标准差 = {n_mlp + 14:,} 个")
check("197,774 + 14 = 197,788", n_mlp == 197_774 and n_mlp + 14 == 197_788)

# ---------------------------------------------------------------------------
banner("7b. 画图：figures/ch04_policy_bells.png（1 口钟 → 14 口钟）与 figures/ch04_overview.png（4.0 节总览）")
if not finite(mean.numpy(), action.numpy()) or POLICY_STD <= 0:
    print("  标准差不是正数，这两张图跳过。")
else:
    fig = plt.figure(figsize=(9.4, 17.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.4, 3.4, 4.4], hspace=0.58, left=0.10, right=0.96, top=0.92, bottom=0.075)
    fig.suptitle("策略吐出的不是动作，是钟：每个关节一口，动作从里面抽", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.985)
    X_LIM = (-3.6, 3.6)
    bx = np.linspace(*X_LIM, 500)
    peak_p = float(gauss_pdf(0, 0, POLICY_STD))

    def minus(text):
        return text.replace("-", "−")

    def draw_bell(ax, j, color, label=None, arrow_at=None, arrow_color=None):
        """一口钟：曲线、均值虚线、抽到的点（橙芯，描上所属那口钟的颜色）。
        label = (ha, dx) 时在钟顶写均值；arrow_at 给出高度时，从均值到抽到的点画一根“σ × ε”箭头。"""
        mu_j, a_j = float(mean[j]), float(action[j])
        ax.plot(bx, gauss_pdf(bx, mu_j, POLICY_STD), color=color, lw=3, zorder=3)
        ax.plot([mu_j, mu_j], [0, peak_p], color=color, lw=1.8, ls="--", zorder=2)
        ax.plot(a_j, 0, "o", color=ORANGE, mec=color, mew=2.6, ms=12, zorder=6, clip_on=False)
        if label:
            ha, dx = label
            ax.text(mu_j + dx, peak_p * 1.07, minus(f"均值 {mu_j:.4f}"), fontsize=FS_SMALL, color=color, ha=ha, va="bottom")
        if arrow_at is not None:
            ax.plot([a_j, a_j], [0, arrow_at], color=arrow_color or color, lw=1.5, ls=":", zorder=2)
            arrow(ax, (mu_j, arrow_at), (a_j, arrow_at), arrow_color or color, lw=2.6)

    ax = fig.add_subplot(gs[0])
    data_axes(ax, "关节 0 的动作", "密度")
    a0, m0 = float(action[0]), float(mean[0])
    draw_bell(ax, 0, BLUE, label=("center", 0.0), arrow_at=peak_p * 0.42, arrow_color=ORANGE)
    lab_dx = 0.25 if a0 < m0 else -0.25                                 # 两个标签都往均值那一侧挪：写在钟的里面
    ax.text((m0 + a0) / 2 + lab_dx, peak_p * 0.385, minus(f"{POLICY_STD:g} × ({eps[0]:.4f})"), fontsize=FS_SMALL, color=ORANGE,
            ha="center", va="top", bbox=WHITE_BOX, zorder=6)            # 箭头下面：白底只盖住均值虚线的一小段，不压蓝色曲线和点到箭头的竖线
    ax.text(a0 + lab_dx, peak_p * 0.06, minus(f"抽到的动作 {a0:.4f}"), fontsize=FS_SMALL, color=ORANGE,
            ha="left" if a0 < m0 else "right", va="bottom", bbox=WHITE_BOX, zorder=6)   # 钟外那一侧曲线贴着地，标签写在那里白底会压住曲线
    ax.set_xlim(*X_LIM)
    ax.set_ylim(0, peak_p * 1.32)
    panel_title(fig, [ax], minus(f"① 一个关节：{m0:.4f} + {POLICY_STD:g} × ({eps[0]:.4f}) = {a0:.4f}"))
    panel_note(fig, [ax], "虚线是网络给的均值（“我认为该做的”）；橙色箭头是“标准差 × 噪声”，\n把动作从均值推到圆点：真正执行的动作（“顺便试试旁边的”）。")

    ax = fig.add_subplot(gs[1])
    data_axes(ax, "动作", "密度")
    draw_bell(ax, 0, BLUE, label=("left", -0.30), arrow_at=peak_p * 0.42)
    draw_bell(ax, 2, GREEN, label=("right", 0.30), arrow_at=peak_p * 0.20)
    ax.set_xlim(*X_LIM)
    ax.set_ylim(0, peak_p * 1.32)
    panel_title(fig, [ax], "② 两个关节：两口钟，中心不同，各抽各的")
    panel_note(fig, [ax], minus(f"蓝箭头：关节 0 从均值 {mean[0]:.4f} 被推到 {action[0]:.4f}；绿箭头：关节 2 从 {mean[2]:.4f} 被推到 {action[2]:.4f}。\n"
                                "圆点描着自己那口钟的颜色。一口钟抽到什么，不影响另一口。"))

    inner = gs[2].subgridspec(2, 7, hspace=0.62, wspace=0.12)
    small_axes = []
    for j in range(14):
        ax = fig.add_subplot(inner[j // 7, j % 7])
        draw_bell(ax, j, BLUE if j % 2 == 0 else GREEN)
        ax.set_xlim(*X_LIM)
        ax.set_ylim(0, peak_p * 1.45)
        ax.set_xticks([-3, 0, 3])                                      # 保留刻度：和 ①② 是同一把尺子
        ax.set_xticklabels(["−3", "0", "3"])
        ax.tick_params(labelsize=13, colors=MUTED, length=3, pad=5)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(CELL_EDGE)
        ax.text(0.5, 0.97, f"关节 {j}", transform=ax.transAxes, fontsize=13, color=MUTED, ha="center", va="top")
        small_axes.append(ax)
    panel_title(fig, small_axes, "③ 项目里：14 个关节 = 14 口钟，一次抽出 14 个数")
    panel_note(fig, small_axes, f"为了排下 14 口，每口横向压缩到了 ①② 的约 1/7：看刻度，它们和 ①② 是同样宽的钟（σ 都是 {POLICY_STD:g}），\n只是中心各不相同。14 个圆点合起来，就是这一步的 14 维动作。", pad=0.014)
    savefig(fig, "ch04_policy_bells")
    plt.close(fig)

    fig, axes = lesson_figure(4, "随机的东西：说不准下一次，说得清很多次", panel_height=3.25, width=9.4)
    HOT = dict(facecolor="#fff0d9", edgecolor=ORANGE, color=ORANGE)
    ax = axes[0]
    lesson_panel(ax, "① 分布：每种结果各有多大可能（4.1–4.2 节）")
    lesson_cells(ax, [[1, 2, 3, 4, 5, 6]], 1.75, 2.05, 0.95, 0.62)
    lesson_cells(ax, [["1/6"] * 6], 1.75, 1.43, 0.95, 0.62, facecolor="white")
    ax.text(1.6, 2.36, "点数", ha="right", va="center", fontsize=FS_SMALL, color=MUTED)
    ax.text(1.6, 1.74, "概率", ha="right", va="center", fontsize=FS_SMALL, color=MUTED)
    hand(ax, 7.75, 1.74, "加起来 = 1")
    note(ax, 0.4, 0.55, "下一次掷出几，说不准；每个点数占六分之一，说得清。")
    ax = axes[1]
    lesson_panel(ax, "② 期望与标准差：中心在哪、散得多开（4.3–4.4 节）")
    hand(ax, 0.4, 2.55, "(1 + 2 + 3 + 4 + 5 + 6) ÷ 6 =")
    cell(ax, 7.75, 2.18, num(E_dice), width=1.3, height=0.75, **HOT)
    hand(ax, 0.4, 1.55, "离 3.5 多远 → 平方 → 平均 → 开根号 =", color=GREEN)
    cell(ax, 7.75, 1.18, f"{std_dice:.2f}", width=1.3, height=0.75, facecolor="#e6f2ec", edgecolor=GREEN, color=GREEN)
    note(ax, 0.4, 0.45, "3.5 叫期望（长期的平均），1.71 叫标准差（一般偏开多少）。")
    ax = axes[2]
    lesson_panel(ax, "③ 不知道概率：多抽几次，取平均（4.5 节）")
    lesson_cells(ax, [[f"{lln_rows[k][1]:.4f}" for k in (0, 2, 5)]], 2.3, 1.55, 1.75, 0.8)
    for k, label in enumerate(["抽 1 个", "抽 100 个", "抽 10 万个"]):
        ax.text(2.3 + (k + 0.5) * 1.75, 2.62, label, ha="center", fontsize=FS_SMALL, color=MUTED)
    ax.text(2.15, 1.95, "平均电压", ha="right", va="center", fontsize=FS_SMALL, color=MUTED)
    hand(ax, 7.7, 1.95, f"→ 期望 {num(E_volt)}", color=ORANGE)
    note(ax, 0.4, 0.55, "电压在 6.5–8.2 伏里随机抽。抽得越多，平均越贴近期望 7.35。")
    ax = axes[3]
    lesson_panel(ax, "④ 策略吐出一口钟，动作从钟里抽（4.6–4.9 节）")
    bell_x = np.linspace(-3, 3, 200)
    ax.plot(2.0 + 0.5 * bell_x, 1.25 + 3.2 * gauss_pdf(bell_x, 0, 1), color=BLUE, lw=3)
    ax.plot([0.5, 3.5], [1.25, 1.25], color=CELL_EDGE, lw=1.5)
    ax.plot([2.0, 2.0], [1.25, 1.25 + 3.2 * 0.3989], color=BLUE, lw=1.6, ls="--")
    ax.plot(2.0 + 0.5 * float(eps[0]), 1.25, "o", color=ORANGE, ms=11, zorder=5)
    hand(ax, 4.0, 2.55, "动作 = 均值 + 标准差 × 噪声")
    hand(ax, 4.0, 1.75, f"{mean[0]:.4f} + {POLICY_STD:g} × ({eps[0]:.4f}) = {action[0]:.4f}".replace("-", "−"), color=ORANGE)
    note(ax, 0.4, 0.45, "这口钟叫高斯分布。最后两节再问：这个动作有多常见？这口钟有多难猜？")
    savefig(fig, "ch04_overview")
    plt.close(fig)

# ---------------------------------------------------------------------------
banner("8. log 概率：先取对数，14 个密度相乘就变成相加")
print("ln 是 exp 的逆运算：ln(x) 问的是“e 的几次方等于 x”")
ln_rows = [[label, value, math.log(value)] for label, value in
           [("e² ≈ 7.389", math.e**2), ("e ≈ 2.718", math.e), ("1", 1.0), ("0.4", 0.4), ("0.3", 0.3), ("0.12", 0.12)]]
table(["x", "ln x"], [[label, lnv] for label, _, lnv in ln_rows], floatfmt=".4g")
check("ln(e²) = 2，ln(e) = 1，ln(1) = 0；小于 1 的数，ln 是负的", math.isclose(math.log(math.e**2), 2) and math.log(1.0) == 0
      and math.log(0.4) < 0)
e2, e3 = math.e**2, math.e**3
print(f"\n指数相加 = 数相乘：e² × e³ = {e2:.3f} × {e3:.3f} ≈ {e2 * e3:.1f}，正是 e⁵ ≈ {math.e**5:.1f}")
check("e² × e³ = e⁵：7.389 × 20.086 ≈ 148.4，e⁵ ≈ 148.4（字面值重算：7.389 × 20.086 = 148.415…，印两位会是 148.42，和 e⁵ = 148.41 对不上，所以正文只印一位）",
      math.isclose(e2 * e3, math.e**5) and round(math.e**5, 1) == 148.4 and round(e2, 3) == 7.389 and round(e3, 3) == 20.086
      and round(7.389 * 20.086, 1) == 148.4 and round(7.389 * 20.086, 2) == 148.42 and round(math.e**5, 2) == 148.41)
check("自测：ln 1 = 0，ln(e³) = 3；ln 4 = ln 2 + ln 2 = 0.693 + 0.693 = 1.386；ln 0.5 = ln 1 − ln 2 = 0 − 0.693 = −0.693",
      math.log(1.0) == 0 and math.isclose(math.log(math.e**3), 3) and round(math.log(2), 3) == 0.693 and round(math.log(4), 3) == 1.386
      and round(0.693 + 0.693, 3) == 1.386 and round(math.log(0.5), 3) == -0.693 and round(0 - 0.693, 3) == -0.693)

p1, p2 = 0.3, 0.4
print(f"\n2 个数的迷你版：{p1} × {p2} = {num(p1 * p2)}")
print(f"  ln {p1} + ln {p2} = {math.log(p1):.3f} + ({math.log(p2):.3f}) = {math.log(p1) + math.log(p2):.3f}；ln {num(p1 * p2)} = {math.log(p1 * p2):.3f}")
print(f"  再用 exp 变回去：exp({math.log(p1) + math.log(p2):.3f}) = {math.exp(math.log(p1) + math.log(p2)):.2f}")
check("ln 0.3 + ln 0.4 = −1.204 − 0.916 = −2.120 = ln 0.12",
      round(math.log(p1), 3) == -1.204 and round(math.log(p2), 3) == -0.916 and round(math.log(p1) + math.log(p2), 3) == -2.120
      and math.isclose(math.log(p1) + math.log(p2), math.log(p1 * p2)))
check("JS 注释里的数：Math.log(0.3) + Math.log(0.4) = −2.1203", round(math.log(p1) + math.log(p2), 4) == -2.1203)
check("字面值重算：−1.204 + (−0.916) = −2.120；exp(−2.120) → 0.12", round(-1.204 + -0.916, 3) == -2.12 and round(math.exp(-2.120), 2) == 0.12)
check("除法变减法：ln(0.3 / 0.4) = ln 0.3 − ln 0.4", math.isclose(math.log(p1 / p2), math.log(p1) - math.log(p2)))
check("自测：ln 0.5 + ln 0.4 = −0.693 + (−0.916) = −1.609 = ln 0.2；exp(−1.609) = 0.2（字面值也成立）",
      round(math.log(0.5), 3) == -0.693 and round(math.log(0.4), 3) == -0.916 and round(math.log(0.2), 3) == -1.609
      and round(-0.693 + -0.916, 3) == -1.609 and math.isclose(0.5 * 0.4, 0.2) and round(math.exp(-1.609), 3) == 0.2
      and math.isclose(math.log(0.5) + math.log(0.4), math.log(0.2)))
check("两件互不影响的事同时发生，概率相乘：两颗骰子掷出 (3, 5) 是 36 种组合里的 1 种，1/6 × 1/6 = 1/36",
      math.isclose(float(np.mean((faces[:, None] == 3) & (faces[None, :] == 5))), 1 / 6 * 1 / 6))

P_EACH = 0.4
print(f"\n14 个 {P_EACH} 连乘 = {P_EACH**14:.7f}（约 {P_EACH**14:.1e}）；取 ln 再相加 = 14 × ({math.log(P_EACH):.4f}) = {14 * math.log(P_EACH):.2f}")
print(f"1000 个 {P_EACH} 连乘：计算机得到 {P_EACH**1000}（小到存不下，变成了 0）；取 ln 再相加 = {1000 * math.log(P_EACH):.1f}，好好的")
check("0.4 的 14 次方 ≈ 0.0000027；14 × ln 0.4 = −12.83", round(P_EACH**14, 7) == 0.0000027 and round(14 * math.log(P_EACH), 2) == -12.83)
check("1000 个 0.4 连乘下溢成 0；log 相加是 −916.3，没事", P_EACH**1000 == 0.0 and round(1000 * math.log(P_EACH), 1) == -916.3)
check("每个正数都能写成 e 的（它的 ln）次方：e^(−1.204) → 0.3000，e^(−0.916) → 0.400；e⁵ ÷ e² = e³（5 个 e 约掉 2 个）；e^(−1) = 1/e → 0.37",
      round(math.exp(-1.204), 4) == 0.3000 and round(math.exp(-0.916), 3) == 0.400 and math.isclose(math.e**5 / math.e**2, math.e**3)
      and math.isclose(math.exp(-1), 1 / math.e) and round(1 / math.e, 2) == 0.37)
check("规则表：ln(0.12 ÷ 0.4) = −2.120 − (−0.916) = −1.204 = ln 0.3；ln(1/0.4) = ln 2.5 → 0.916；ln(0.4¹⁴) = 14 × ln 0.4；"
      "ln √9 = ln 3 → 1.0986，½ × ln 9 = ½ × 2.1972 = 1.0986",
      math.isclose(math.log(0.12 / 0.4), math.log(0.3)) and round(-2.120 - -0.916, 3) == -1.204 and round(math.log(0.3), 3) == -1.204
      and math.isclose(math.log(1 / 0.4), -math.log(0.4)) and math.isclose(math.log(1 / 0.4), math.log(2.5)) and round(math.log(2.5), 3) == 0.916
      and math.isclose(math.log(0.4**14), 14 * math.log(0.4)) and math.isclose(math.log(math.sqrt(9)), 0.5 * math.log(9))
      and round(math.log(3), 4) == 1.0986 and round(math.log(9), 4) == 2.1972 and round(0.5 * 2.1972, 4) == 1.0986)
check("字面值重算：14 × (−0.9163) → −12.83；1000 × (−0.9163) = −916.3",
      round(math.log(P_EACH), 4) == -0.9163 and round(14 * -0.9163, 2) == -12.83 and round(1000 * -0.9163, 1) == -916.3)

# ---------------------------------------------------------------------------
banner("8b. 画图：figures/ch04_ln_curve.png（ln 的曲线；相乘 ↔ 相加）")
fig = plt.figure(figsize=(9.4, 14.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.46, left=0.12, right=0.95, top=0.905, bottom=0.075)
fig.suptitle("ln 问“e 的几次方等于 x”：它把相乘变成相加", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.975)
ax = fig.add_subplot(gs[0])
data_axes(ax, "x", "ln x")
ax.axvspan(0, 1, color="#eef2f6", zorder=0)                         # x 比 1 小的那一段
lx = np.linspace(0.06, 8.1, 700)
ax.plot(lx, np.log(lx), color=BLUE, lw=3.4, zorder=3)
ax.axhline(0, color=MUTED, lw=1.6, ls=":", zorder=2)
LN_LABEL_AT = {"e² ≈ 7.389": (7.2, 2.4, "right"), "e ≈ 2.718": (3.05, 0.5, "left"), "1": (0.12, 0.85, "left"),
               "0.4": (1.5, -0.7, "left"), "0.3": (1.5, -1.35, "left"), "0.12": (1.5, -2.0, "left")}
for label, value, lnv in ln_rows:                                    # 表里的六行：六个橙点
    tx, ty, ha = LN_LABEL_AT[label]
    ax.plot(value, lnv, "o", color=ORANGE, ms=11, zorder=5)
    ax.annotate(f"ln {label.split()[0]} = {lnv:.4g}".replace("-", "−"), xy=(value, lnv), xytext=(tx, ty), fontsize=FS_SMALL,
                color=ORANGE, ha=ha, va="center", bbox=WHITE_BOX, zorder=6,
                arrowprops=dict(arrowstyle="-", lw=1.2, color=ORANGE, shrinkA=2, shrinkB=7))
ax.set_xlim(0, 8.2)
ax.set_ylim(-2.9, 2.8)
ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
panel_title(fig, [ax], "① 曲线过 (1, 0)：x 比 1 小，ln 是负的，越靠近 0 越负")
panel_note(fig, [ax], "橙点就是 ln 表的六行。浅灰那一段是 x 比 1 小的部分，曲线全在虚线（0）以下；\nx 比 1 大，ln 是正的，涨得越来越慢。x 是 0 或负数时，ln 没有定义。")

ax = fig.add_subplot(gs[1])
lesson_panel(ax, xmax=10.0, ymax=4.0)
l1, l2 = math.log(p1), math.log(p2)
top_y, bot_y, cw, ch = 2.85, 0.45, 2.0, 0.8
top_txt = [num(p1), num(p2), num(p1 * p2)]
bot_txt = [f"{l1:.3f}", f"({l2:.3f})", f"{l1 + l2:.3f}"]
for k, x0 in enumerate([1.0, 3.7, 6.4]):                             # 三个格子的左边
    style = dict(facecolor="#fff0d9", edgecolor=ORANGE, color=ORANGE) if k == 2 else {}
    cell(ax, x0, top_y, top_txt[k], width=cw, height=ch, **style)
    cell(ax, x0, bot_y, bot_txt[k].replace("-", "−"), width=cw, height=ch, facecolor="#e6f2ec", edgecolor=GREEN, color=GREEN)
    xc, y_mid = x0 + cw / 2, (top_y + bot_y + ch) / 2
    if k < 2:
        arrow(ax, (xc, top_y - 0.08), (xc, bot_y + ch + 0.08), BLUE, lw=2.6)
        hand(ax, xc + 0.12, y_mid, "取 ln", color=BLUE, fontsize=FS_SMALL)
    else:
        arrow(ax, (xc, bot_y + ch + 0.08), (xc, top_y - 0.08), ORANGE, lw=2.6)
        hand(ax, xc + 0.12, y_mid, "过 exp", color=ORANGE, fontsize=FS_SMALL)
for x_op, top_op, bot_op in [(3.35, "×", "+"), (6.05, "=", "=")]:
    ax.text(x_op, top_y + ch / 2, top_op, fontsize=FS_STEP, color=INK, ha="center", va="center")
    ax.text(x_op, bot_y + ch / 2, bot_op, fontsize=FS_STEP, color=INK, ha="center", va="center")
ax.text(8.65, top_y + ch / 2, "相乘", fontsize=FS_SMALL, color=MUTED, ha="left", va="center")
ax.text(8.65, bot_y + ch / 2, "相加", fontsize=FS_SMALL, color=MUTED, ha="left", va="center")
panel_title(fig, [ax], "② 相乘 ↔ 相加：每个数往下取 ln，× 就变成了 +")
panel_note(fig, [ax], f"上一行 {num(p1)} × {num(p2)} = {num(p1 * p2)}；下一行把每个数换成它的 ln，乘号变成加号。\n算完往上过一次 exp，又回到 {num(p1 * p2)}：什么也没丢。")
savefig(fig, "ch04_ln_curve")
plt.close(fig)

dist = torch.distributions.Normal(mean, std)
dens = gauss_pdf(action.numpy(), mean.numpy(), std.numpy())        # 14 个一维密度：第 6 节的公式
print("\n回到第 7 节抽到的 14 维动作。每个关节：这个动作在自己那口钟下的密度，以及它的 ln（前 4 个关节）：")
table(["关节", "动作", "均值", "密度 p", "ln p"], [[i, float(action[i]), float(mean[i]), float(dens[i]), math.log(dens[i])] for i in range(4)],
      floatfmt=".4f")
logp_manual = float(np.sum(np.log(dens)))
logp_torch = dist.log_prob(action).sum(dim=-1).item()
prod_dens = float(np.prod(dens))
print("14 个合成一个：")
table(["算法", "结果"], [["14 个密度连乘", f"{prod_dens:.3g}"], ["ln(连乘的结果)", f"{math.log(prod_dens):.4f}"],
                       ["手写：14 个 ln p 相加", f"{logp_manual:.4f}"], ["torch：log_prob(a).sum(dim=-1)", f"{logp_torch:.4f}"]])
check("14 维 log 概率：手写 Σ ln p_i = torch 的 log_prob(...).sum(dim=-1) = ln(连乘)",
      abs(logp_manual - logp_torch) < 1e-4 and abs(math.log(prod_dens) - logp_manual) < 1e-9)
check("正文引用的数：log π = −19.0465，对应连乘的密度 5.35e-09", round(logp_torch, 4) == -19.0465 and f"{prod_dens:.3g}" == "5.35e-09")
check("5.35e-09 就是 5.35 × 10⁻⁹ = 0.00000000535：小数点后第 9 位才开始有数", f"{5.35e-9:.11f}" == "0.00000000535" and f"{5.35e-9:.11f}".index("5") == 10)
check("关节 0 的动作离均值 1.07 个 σ，是前 4 个里最远的", round(abs(float(eps[0])), 2) == 1.07 and int(torch.argmax(eps[:4].abs())) == 0)
check("前 4 个关节的密度都在 0.22 到 0.40 之间：单个都很普通，小的是 14 个连乘", all(0.22 < float(d) < 0.40 for d in dens[:4]))
check("正文引用的数：关节 0 的密度 0.2248、ln p = −1.4927", round(float(dens[0]), 4) == 0.2248 and round(math.log(dens[0]), 4) == -1.4927)
check("手算关节 0（字面值重算）：−0.6089 − 0.4623 = −1.0712；1.0712² → 1.14747；1.14747 ÷ 2 → 0.5737；exp(−0.5737) → 0.5634；0.5634 ÷ 2.5066 → 0.2248",
      round(-0.6089 - 0.4623, 4) == -1.0712 and round(1.0712**2, 5) == 1.14747 and round(1.14747 / 2, 4) == 0.5737
      and round(math.exp(-0.5737), 4) == 0.5634 and round(0.5634 / 2.5066, 4) == 0.2248)

log_formula = -((action - mean) ** 2) / (2 * std**2) - torch.log(std) - 0.5 * math.log(2 * math.pi)
check("一维 log 密度的三项式 −(a−μ)²/(2σ²) − ln σ − ½ ln(2π) 和 torch 逐个相同", torch.allclose(log_formula, dist.log_prob(action), atol=1e-5))
print(f"\nlog“概率”其实是 log 密度，可以是正数：σ = 0.1 的钟，中心密度 {narrow_bell_peak:.3f}，ln = {math.log(narrow_bell_peak):.3f}")
check("ln 3.989 = 1.384 > 0：log 密度为正并不奇怪", round(math.log(narrow_bell_peak), 3) == 1.384)

commute = torch.distributions.Normal(35.0, 5.0)        # 4.0 节的通勤时间：一般 35 分钟，上下差 5 分钟
lp35, lp50 = commute.log_prob(torch.tensor(35.0)).item(), commute.log_prob(torch.tensor(50.0)).item()
print(f"通勤那口钟（μ = 35，σ = 5）：花 35 分钟的 log 密度 {lp35:.2f}，花 50 分钟的 {lp50:.2f}；差 {lp50 - lp35:.2f}，"
      f"密度之比 exp({lp50 - lp35:.1f}) = 1/{math.exp(lp35 - lp50):.0f}")
check("通勤：log 密度 −2.53 和 −7.03，差 4.5，密度差约 90 倍", round(lp35, 2) == -2.53 and round(lp50, 2) == -7.03
      and round(lp35 - lp50, 2) == 4.5 and round(math.exp(lp35 - lp50)) == 90)
check("通勤手算（字面值重算）：5 × 2.5066 = 12.533；1 ÷ 12.533 → 0.0798；ln 0.0798 → −2.53；15² ÷ (2 × 5²) = 225 ÷ 50 = 4.5；−2.53 + (−4.5) = −7.03",
      math.isclose(5 * 2.5066, 12.533) and round(1 / 12.533, 4) == 0.0798 and round(math.log(0.0798), 2) == -2.53
      and 15**2 / (2 * 5**2) == 225 / 50 == 4.5 and round(-2.53 + -4.5, 2) == -7.03)

mean_new = mean + 0.1 * (action - mean)            # 假想更新了一小步：钟的中心朝这个动作挪了 10%
logp_new = torch.distributions.Normal(mean_new, std).log_prob(action).sum(dim=-1).item()
lp_old4, lp_new4 = round(logp_torch, 4), round(logp_new, 4)   # 正文印的是这两个 4 位小数：“差”就拿它们相减，读者按计算器得到的也是它
diff4 = round(lp_new4 - lp_old4, 4)
print(f"两口钟比较：中心朝这个动作挪 10% 之后，log π 从 {lp_old4:.4f} 变成 {lp_new4:.4f}；差 {diff4:.4f}，"
      f"exp({diff4:.4f}) = {math.exp(diff4):.2f}（新钟下的密度是旧钟的这么多倍）")
check("两个 log π 相减再 exp = 两个密度相除", math.isclose(math.exp(logp_new - logp_torch),
      float(np.prod(gauss_pdf(action.numpy(), mean_new.numpy(), std.numpy()))) / prod_dens, rel_tol=1e-4))
check("正文引用的数：log π 从 −19.0465 变成 −17.8721，差 1.1744（拿印出来的两个数相减），exp(1.1744) = 3.24，密度比 3.24",
      lp_old4 == -19.0465 and lp_new4 == -17.8721 and diff4 == 1.1744 and round(-17.8721 - -19.0465, 4) == 1.1744
      and round(math.exp(1.1744), 2) == 3.24 and round(math.exp(logp_new - logp_torch), 2) == 3.24)

std_bell = torch.distributions.Normal(0.0, 1.0)
lp_at1 = std_bell.log_prob(torch.tensor(1.0)).item()
print(f"\n进阶：log 密度三项式在 σ = 1、x = 1 处：−0.5 − ln 1 − ½ ln(2π) = −0.5 − 0 − {0.5 * math.log(2 * math.pi):.4f} = {-0.5 - 0.5 * math.log(2 * math.pi):.4f}；"
      f"torch：{lp_at1:.4f}；ln(0.241971) = {math.log(0.241971):.4f}")
check("log 密度三项式核对（字面值）：½ × ln 6.2832 → 0.9189；−0.5 − 0.9189 = −1.4189；ln(0.241971) → −1.4189；torch 给 −1.4189",
      round(0.5 * math.log(6.2832), 4) == 0.9189 and round(-0.5 - 0.9189, 4) == -1.4189 and round(math.log(0.241971), 4) == -1.4189
      and round(lp_at1, 4) == -1.4189 and math.log(1) == 0)
first_term = lambda m: -((1 - m) ** 2) / 2                              # x = 1、σ = 1 时 log 密度的第一项，m 是均值
nudge = (first_term(0.001) - first_term(0.0)) / 0.001
print(f"对 μ 的导数：μ 从 0 拧到 0.001，第一项 {first_term(0.0):g} → {first_term(0.001):.7f}，涨了 {first_term(0.001) - first_term(0.0):.7f}，÷ 0.001 = {nudge:.4f}（公式 (x − μ)/σ² = 1）")
check("对 μ 的导数缩 h 核对：−(1 − 0.001)² ÷ 2 = −0.4990005，涨了 0.0009995，÷ 0.001 = 0.9995，靠近 (1 − 0)/1² = 1",
      math.isclose(first_term(0.001), -0.4990005) and math.isclose(first_term(0.001) - first_term(0.0), 0.0009995) and round(nudge, 4) == 0.9995
      and round(-(1 - 0.001) ** 2 / 2, 7) == -0.4990005)
ln_slopes = [(x, (math.log(x + 0.001) - math.log(x)) / 0.001) for x in (0.5, 1, 2, 4)]
print("ln 的导数：h = 0.001 时量到的斜率 " + "、".join(f"x = {x:g}：{s:.4f}" for x, s in ln_slopes) + "——规律是 1/x")
check("正文表里的四个斜率：1.9980、0.9995、0.4999、0.2500（都靠近 1/x）",
      [f"{s:.4f}" for _, s in ln_slopes] == ["1.9980", "0.9995", "0.4999", "0.2500"] and all(abs(s - 1 / x) < 0.003 / x for x, s in ln_slopes))
check("ln(1 + t) ≈ t：第 1 章的 e^0.01 = 1.01005；Math.log(1.001) → 0.0009995；(x + h) ÷ x = 1 + h/x",
      round(math.exp(0.01), 5) == 1.01005 and round(math.log(1.001), 7) == 0.0009995 and math.isclose((2 + 0.001) / 2, 1 + 0.001 / 2))
h = 1e-6
dln = (math.log(2 + h) - math.log(2)) / h
print(f"缩 h 量 x = 2 处的斜率：(ln 2.001 − ln 2) / 0.001 = {(math.log(2.001) - math.log(2)) / 0.001:.4f}；h = 0.000001 时 {dln:.4f} = 1/2")
check("(ln x)′ = 1/x：在 x = 2 处是 0.5", abs(dln - 0.5) < 1e-5 and round((math.log(2.001) - math.log(2)) / 0.001, 4) == 0.4999)
mu_t = mean.clone().requires_grad_(True)
torch.distributions.Normal(mu_t, std).log_prob(action).sum().backward()
check("进阶：log π 对均值的梯度 = (a − μ)/σ²（第 11 章会用）", torch.allclose(mu_t.grad, (action - mean) / std**2, atol=1e-5))

# ---------------------------------------------------------------------------
banner("9. 熵：这个分布有多“随机”")
print("意外程度 = −ln p：概率越小越意外。熵 = 意外程度的期望（按概率加权平均）")
coin_fair = -(0.5 * math.log(0.5) + 0.5 * math.log(0.5))
coin_biased = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1))
print(f"  公平硬币：两面的意外程度都是 −ln 0.5 = {-math.log(0.5):.3f}，熵 = 0.5 × 0.693 + 0.5 × 0.693 = {coin_fair:.3f}")
print(f"  偏心硬币（0.9 / 0.1）：意外程度 {-math.log(0.9):.3f} 和 {-math.log(0.1):.3f}，熵 = 0.9 × 0.105 + 0.1 × 2.303 = {coin_biased:.3f}")
print(f"  骰子：每面 −ln(1/6) = ln 6 = {math.log(6):.3f}，熵也是 {math.log(6):.3f}")
check("公平硬币的熵 0.693（= ln 2）；偏心硬币 0.325，更好猜；骰子 1.792（= ln 6），更难猜",
      round(coin_fair, 3) == 0.693 and round(coin_biased, 3) == 0.325 and round(math.log(6), 3) == 1.792)
check("偏心硬币手算用到的两个数：−ln 0.9 = 0.105，−ln 0.1 = 2.303", round(-math.log(0.9), 3) == 0.105 and round(-math.log(0.1), 3) == 2.303)
check("为什么用 ln：两枚公平硬币都是正面，概率 0.5 × 0.5 = 0.25，意外程度 −ln 0.25 → 1.386 = 0.693 + 0.693（独立的意外相加）",
      0.5 * 0.5 == 0.25 and round(-math.log(0.25), 3) == 1.386 and round(0.693 + 0.693, 3) == 1.386
      and math.isclose(-math.log(0.25), -math.log(0.5) - math.log(0.5)))
check("字面值重算：−ln(1/6) = −(ln 1 − ln 6) = −(0 − 1.792) = 1.792；0.5 × 0.693 + 0.5 × 0.693 = 0.693；0.9 × 0.105 + 0.1 × 2.303 → 0.325",
      round(-math.log(1 / 6), 3) == 1.792 and -(0 - 1.792) == 1.792 and round(0.5 * 0.693 + 0.5 * 0.693, 3) == 0.693
      and round(0.9 * 0.105 + 0.1 * 2.303, 3) == 0.325)

H_volt = math.log(width)
print(f"\n均匀分布：密度处处是 1/宽度，意外程度处处是 −ln(1/宽度) = ln(宽度)，熵就是 ln(宽度)")
table(["分布", "宽度", "熵 = ln(宽度)"], [[f"U({lo:g}, {hi:g})", width, H_volt], ["U(0, 1)", 1.0, math.log(1.0) + 0.0],
                                       ["U(0, 0.2)", narrow_w, math.log(narrow_w)]], floatfmt=".4g")
check("电压 U(6.5, 8.2) 的熵 = ln 1.7 ≈ 0.53", round(H_volt, 2) == 0.53 and round(H_volt, 4) == 0.5306)
check("宽度小于 1 时熵是负数：U(0, 0.2) 的熵 = ln 0.2 = −1.609；U(0, 1) 的熵恰好是 0",
      round(math.log(narrow_w), 3) == -1.609 and math.log(1.0) == 0)
torch_H_volt = torch.distributions.Uniform(lo, hi).entropy().item()
check("和 torch 的 Uniform(...).entropy() 一致", abs(torch_H_volt - H_volt) < 1e-5)
check("字面值重算：−ln(1/1.7) = −(ln 1 − ln 1.7) = ln 1.7 → 0.53", math.isclose(-math.log(1 / 1.7), math.log(1.7)) and round(math.log(1.7), 2) == 0.53)

ln_c = math.log(SQRT_2PI)                            # σ = 1 的钟：p(x) = exp(−x²/2) ÷ 2.5066，取 ln 以后剩下的那个常数
H_std_torch = torch.distributions.Normal(0.0, 1.0).entropy().item()
print(f"\nσ = 1 的钟，手算熵：ln p(x) = −x²/2 − ln 2.5066 = −x²/2 − {ln_c:.4f}；意外程度 = x²/2 + {ln_c:.4f}；"
      f"x² 的平均就是方差 1，所以熵 = 0.5 + {ln_c:.4f} = {0.5 + ln_c:.4f}（torch 给的：{H_std_torch:.4f}）")
check("手算 σ = 1 的钟的熵：ln 2.5066 → 0.9189，0.5 + 0.9189 = 1.4189，和 torch 给的一样（字面值也成立）",
      round(math.log(2.5066), 4) == 0.9189 and round(0.5 + math.log(2.5066), 4) == 1.4189 and round(0.5 + 0.9189, 4) == 1.4189
      and abs(0.5 + ln_c - H_std_torch) < 1e-6)
EQUIV_WIDTH = math.sqrt(2 * math.pi * math.e)       # 一口 σ = 1 的钟，和宽 4.13 的均匀分布一样“随机”
SIGMAS = [0.1, 0.5, 1.0, 2.0]
ent_rows = [[s, torch.distributions.Normal(0.0, s).entropy().item(), EQUIV_WIDTH * s, math.log(EQUIV_WIDTH * s)] for s in SIGMAS]
print(f"\n高斯分布的熵只和 σ 有关：熵 = ln({EQUIV_WIDTH:.4f} × σ)——相当于一段宽 {EQUIV_WIDTH:.2f}σ 的均匀分布")
table(["σ", "熵（torch 算的）", "相当的宽度 4.13σ", "ln(相当的宽度)"], [[num(r[0]), *r[1:]] for r in ent_rows], floatfmt=".4f")
check("torch 的 Normal(...).entropy() = ln(4.1327 σ) = 折叠块里的 ½ ln(2πe σ²)；2πe = 17.079",
      all(abs(r[1] - r[3]) < 1e-5 and abs(r[1] - 0.5 * math.log(2 * math.pi * math.e * r[0] ** 2)) < 1e-5 for r in ent_rows) and round(EQUIV_WIDTH, 4) == 4.1327
      and round(2 * math.pi * math.e, 3) == 17.079)
check("正文引用的数：σ = 0.1、0.5、1、2 的熵是 −0.8836、0.7258、1.4189、2.1121",
      [round(r[1], 4) for r in ent_rows] == [-0.8836, 0.7258, 1.4189, 2.1121])
check("σ 翻倍，熵加 ln 2 = 0.693（0.5 → 1 → 2）", abs(ent_rows[2][1] - ent_rows[1][1] - math.log(2)) < 1e-5
      and abs(ent_rows[3][1] - ent_rows[2][1] - math.log(2)) < 1e-5 and round(math.log(2), 3) == 0.693)
check("折叠块的合并五步：½ = ½ ln e；ln σ = ½ ln σ²；½(ln e + ln σ² + ln 2π) = ½ ln(2πeσ²) = ln(√(2πe) σ)；2πe → 17.079，√17.079 → 4.1327；σ = 1 时 0.5 + 0.9189 = 1.4189",
      all(math.isclose(0.5 + math.log(s) + 0.5 * math.log(2 * math.pi), 0.5 * (math.log(math.e) + math.log(s**2) + math.log(2 * math.pi)))
          and math.isclose(0.5 * math.log(2 * math.pi * math.e * s**2), math.log(math.sqrt(2 * math.pi * math.e) * s)) for s in SIGMAS)
      and math.isclose(math.log(0.5**2), 2 * math.log(0.5)) and round(2 * math.pi * math.e, 3) == 17.079 and round(math.sqrt(17.079), 4) == 4.1327
      and round(0.5 + 0.9189, 4) == 1.4189)
sigma_zero_entropy = 1 / EQUIV_WIDTH
check("σ = 0.1 的钟相当于宽度 0.41（比 1 小，所以熵为负）；exp(1.4189) = 4.13", round(ent_rows[0][2], 2) == 0.41 and round(math.exp(1.4189), 2) == 4.13)
check("通勤（按分钟算）：天天准点 σ = 1，熵 ln 4.13 → 1.42；很随缘 σ = 10，熵 ln 41.3 → 3.72；相差 ln 10 → 2.30",
      round(math.log(4.13), 2) == 1.42 and round(math.log(41.3), 2) == 3.72 and round(3.72 - 1.42, 2) == 2.30 and round(math.log(10), 2) == 2.30
      and round(math.log(EQUIV_WIDTH * 1), 2) == 1.42 and round(math.log(EQUIV_WIDTH * 10), 2) == 3.72)
check("自测：σ = 4 的钟，熵 = 2.1121 + 0.6931 = 2.8052；U(0, 0.5) 的熵 = ln 0.5 = −0.693",
      round(math.log(EQUIV_WIDTH * 4), 4) == 2.8052 and round(math.log(0.5), 3) == -0.693)
check("熵恰好为 0 的那口钟：σ = 1/4.1327 ≈ 0.242", round(sigma_zero_entropy, 3) == 0.242)

z9 = np.random.default_rng(9).normal(0, 1, size=100_000)
H_mc = float(np.mean(-np.log(gauss_pdf(z9, 0.0, 1.0))))
print(f"\n用第 5 节的办法核对定义：从 σ = 1 的钟里抽 10 万个数，把 −ln p(x) 取平均 = {H_mc:.3f}（公式给 {ent_rows[2][1]:.3f}）")
check("“多抽几次取平均”估出来的熵 ≈ 1.419（容差 0.01）；正文引用的数 1.422", abs(H_mc - ent_rows[2][1]) < 0.01 and round(H_mc, 3) == 1.422)

H_policy_manual = float(sum(math.log(EQUIV_WIDTH * s) for s in std.tolist()))
H_policy_torch = dist.entropy().sum(dim=-1).item()
print(f"\n14 口钟的熵相加：14 × {math.log(EQUIV_WIDTH * POLICY_STD):.5f} = {H_policy_manual:.3f}；torch：{H_policy_torch:.3f}")
check("14 维策略的熵：手写 = torch 的 entropy().sum(dim=-1)", abs(H_policy_manual - H_policy_torch) < 1e-4)
check("正文引用的数：σ 都是 1 时，14 × 1.41894 = 19.865", round(H_policy_torch, 3) == 19.865 and round(ent_rows[2][1], 5) == 1.41894)

# ---------------------------------------------------------------------------
banner("9b. 画图：figures/ch04_entropy_vs_sigma.png（钟越宽，熵越大）")
fig = plt.figure(figsize=(9.4, 13.6))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.58, left=0.12, right=0.95, top=0.90, bottom=0.115)
fig.suptitle("熵：钟越宽越难猜，熵越大；σ 每翻一倍，熵加 0.693", fontsize=FS_TITLE, fontweight="bold", color=INK, y=0.975)
ax = fig.add_subplot(gs[0])
data_axes(ax, "抽到的数 x", "密度")
ex = np.linspace(-6, 6, 700)
for s, color, (tx, ty) in [(0.5, ORANGE, (0.75, 0.70)), (1.0, BLUE, (1.55, 0.27)), (2.0, GREEN, (3.35, 0.115))]:
    ax.plot(ex, gauss_pdf(ex, 0, s), color=color, lw=3.2, zorder=3)
    ax.text(tx, ty, f"σ = {s:g}：熵 {0.5 * math.log(2 * math.pi * math.e * s**2):.3f}", fontsize=FS_SMALL, color=color, va="center", bbox=WHITE_BOX)
ax.set_xlim(-6, 6)
ax.set_ylim(0, 0.88)
panel_title(fig, [ax], "① 三口钟，面积都是 1：越宽越矮，抽出来的数越难猜")
panel_note(fig, [ax], "σ = 0.5 的钟几乎总抽到 0 附近；σ = 2 的钟从 −4 到 4 都常见。")

ax = fig.add_subplot(gs[1])
data_axes(ax, "标准差 σ", "一口钟的熵")
ss = np.linspace(0.05, 2.25, 300)
ax.plot(ss, np.log(EQUIV_WIDTH * ss), color=BLUE, lw=3.5, zorder=3)
ax.axhline(0, color=MUTED, lw=1.8, ls=":", zorder=2)
LABEL_AT = {0.1: (0.10, -0.05, "left"), 0.5: (0.09, -0.22, "left"), 1.0: (0.08, -0.24, "left"), 2.0: (0.0, -0.50, "right")}
for s, H, *_ in ent_rows:
    dx, dy, ha = LABEL_AT[s]
    ax.plot(s, H, "o", color=ORANGE, ms=11, zorder=5)
    ax.text(s + dx, H + dy, f"σ = {s:g}：{H:.3f}".replace("-", "−"), fontsize=FS_SMALL, color=ORANGE, ha=ha, va="center", bbox=WHITE_BOX, zorder=6)
ax.plot(sigma_zero_entropy, 0, "o", color=MUTED, ms=9, zorder=5)
ax.text(sigma_zero_entropy + 0.16, -0.50, f"σ = {sigma_zero_entropy:.3f}：熵恰好 0", fontsize=15, color=MUTED, va="center", bbox=WHITE_BOX)
ax.plot([sigma_zero_entropy + 0.01, sigma_zero_entropy + 0.15], [-0.05, -0.42], color=MUTED, lw=1.2)
ax.set_xlim(0, 2.3)
ax.set_ylim(-1.75, 2.6)
ax.set_xticks([0, 0.5, 1, 1.5, 2])
panel_title(fig, [ax], "② 熵随 σ 变大而变大；钟很窄时熵是负数")
panel_note(fig, [ax], "0.5 → 1 → 2：σ 每翻一倍，熵都加 0.693（0.726 → 1.419 → 2.112）。\n虚线以下是负数：只说明这口钟比“宽 1 的均匀分布”还集中，不是算错了。")
savefig(fig, "ch04_entropy_vs_sigma")
plt.close(fig)

# ---------------------------------------------------------------------------
banner("10. 映射到项目：正文引用的常数和源码行还在不在")
REPO = Path(_common.__file__).resolve().parents[3]      # 从 _common.py 的位置找仓库根目录：把本脚本复制到别处运行也找得到
DIST_LINES = ["self.std_param = nn.Parameter(init_std * torch.ones(output_dim))",
              "std = self.std_param.expand_as(mean)",
              "self._distribution = Normal(mean, std)",
              "return self._distribution.sample()",
              "return self._distribution.entropy().sum(dim=-1)",
              "return self._distribution.log_prob(outputs).sum(dim=-1)"]
CFG_LINES = ['"class_name": "GaussianDistribution",', '"init_std": 1.0,', '"std_type": "scalar",', "entropy_coef=0.01,"]
CMD_LINES = ["for i, (lo, hi) in enumerate(self.cfg.ranges):",
             "self._command[env_ids, i] = r.uniform_(lo, hi)",
             "if self.cfg.zero_command_prob > 0.0:",
             "zero_mask = torch.rand(n, device=self.device) < self.cfg.zero_command_prob",
             "self._command[env_ids[zero_mask]] = 0.0"]

rsl_spec = importlib.util.find_spec("rsl_rl")
rsl_dir = Path(rsl_spec.origin).parent if rsl_spec and rsl_spec.origin else None
if rsl_dir and (rsl_dir / "modules" / "distribution.py").is_file():
    dist_text = (rsl_dir / "modules" / "distribution.py").read_text(encoding="utf-8")
    at = dist_text.find("class GaussianDistribution(Distribution):")
    print("rsl_rl/modules/distribution.py 的 GaussianDistribution：14 个标准差是可训练参数 → 造钟 → 抽样 → 熵 → log 概率")
    check(f"正文映射块引用的 {len(DIST_LINES)} 行源码都原样存在，且顺序一致", at >= 0 and lines_in_order(dist_text[at:], DIST_LINES))
    ppo_text = (rsl_dir / "algorithms" / "ppo.py").read_text(encoding="utf-8")
    logger_text = (rsl_dir / "utils" / "logger.py").read_text(encoding="utf-8")
    check("PPO 的损失里减去 entropy_coef × 熵的平均（熵大有奖励）", "- self.entropy_coef * entropy.mean()" in ppo_text)
    check("训练日志里那一行叫 Mean action std，wandb 里叫 Policy/mean_std", '"Mean action std:"' in logger_text and '"Policy/mean_std"' in logger_text)
else:
    print("  （当前 Python 环境里没有 rsl_rl，跳过源码核对；用 uv run 运行就会核对）")

mjlab_spec = importlib.util.find_spec("mjlab")
vel_cmd = Path(mjlab_spec.origin).parent / "tasks" / "velocity" / "mdp" / "velocity_command.py" if mjlab_spec and mjlab_spec.origin else None
if vel_cmd and vel_cmd.is_file():
    check("走路任务的速度指令用同一种写法留出“原地站着”的份额：U(0, 1) 抽一个数，和 rel_standing_envs 比大小",
          "self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs" in vel_cmd.read_text(encoding="utf-8"))
else:
    print("  （当前 Python 环境里没有 mjlab，跳过这一项）")

cfg_path = REPO / "src" / "mjlab_microduck" / "tasks" / "microduck_velocity_env_cfg.py"
mdp_path = REPO / "src" / "mjlab_microduck" / "tasks" / "mdp.py"
const_path = REPO / "src" / "mjlab_microduck" / "robot" / "microduck_constants.py"
if cfg_path.is_file() and mdp_path.is_file() and const_path.is_file():
    cfg_text = cfg_path.read_text(encoding="utf-8")
    at = cfg_text.find("MicroduckRlCfg = RslRlOnPolicyRunnerCfg(")
    print("microduck_velocity_env_cfg.py 的 MicroduckRlCfg：", "  ".join(CFG_LINES))
    check("项目配置：高斯策略、初始标准差 1.0、熵系数 0.01", at >= 0 and lines_in_order(cfg_text[at:], CFG_LINES))
    mdp_text = mdp_path.read_text(encoding="utf-8")
    at = mdp_text.find("class UniformPoseCommand(CommandTerm):")
    print("mdp.py 的 UniformPoseCommand._resample_command：每一维从均匀分布抽；再以 zero_command_prob 的概率把整条指令设成 0")
    check(f"正文映射块引用的 {len(CMD_LINES)} 行抽指令的源码都原样存在，且顺序一致", at >= 0 and lines_in_order(mdp_text[at:], CMD_LINES))
    standup_path = REPO / "src" / "mjlab_microduck" / "tasks" / "microduck_standup_env_cfg.py"
    standup_text = standup_path.read_text(encoding="utf-8") if standup_path.is_file() else ""
    check("“站起”任务里，全零指令的概率设的是 0.3（走路任务没有设，默认 0）",
          re.search(r"^BODY_CMD_ZERO_PROB\s*=\s*0\.3\s*$", standup_text, flags=re.M) is not None
          and "zero_command_prob=BODY_CMD_ZERO_PROB," in standup_text and "zero_command_prob" not in cfg_text)
    print("microduck_constants.py：vin_range=(6.5, 8.2),   ← 4.2 节的电压范围")
    check("电池电压的范围 vin_range=(6.5, 8.2) 还在", "\n    vin_range=(6.5, 8.2)," in const_path.read_text(encoding="utf-8"))
else:
    print("  （没找到项目源码，跳过这几项）")

done()
