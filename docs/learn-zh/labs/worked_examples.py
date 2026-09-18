"""复算详解版正文中的手算数字，不加载机器人、不训练、不联网。

运行：python3 docs/learn-zh/labs/worked_examples.py
只用 Python 标准库。各节同时检查等价计算或数值差分，帮助发现抄错公式。
"""

import math
from statistics import mean, pvariance


def close(label, actual, expected, tolerance=1e-6):
    assert math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), (
        label, actual, expected
    )
    print(f"✓ {label}: {actual:.9g}")


def reward(error, sigma):
    return math.exp(-(error / sigma) ** 2)


print("第 1 章：误差改善、局部导数与最敏感位置")
sigma = math.sqrt(0.1)
for old, new in [(0.05, 0), (0.25, 0.2), (1, 0.95)]:
    print(f"  {old} → {new}: ΔR={reward(new, sigma)-reward(old, sigma):.9f}")
derivative = lambda x: -2*x/sigma**2 * reward(x, sigma)
h = 1e-6
for x in [-0.3, 0, 0.1, sigma/math.sqrt(2), 0.6]:
    numeric = (reward(x+h, sigma)-reward(x-h, sigma))/(2*h)
    close(f"R'({x:.6f}) 与中心差分", derivative(x), numeric)
peak = sigma/math.sqrt(2)
close("最敏感误差", peak, math.sqrt(0.05))
close("该处奖励", reward(peak, sigma), math.exp(-0.5))
assert abs(derivative(peak)) > abs(derivative(peak-0.01))
assert abs(derivative(peak)) > abs(derivative(peak+0.01))
close("0.2 → 0.19 的奖励改善", reward(0.19, sigma)-reward(0.2, sigma), 0.026658952)
close("0.100 → 0.101 的实际奖励变化", reward(0.101, sigma)-reward(0.1, sigma), -0.001816897)

print("\n第 2–4 章：矩阵、梯度、密度与方差")
X = [[1, 2], [3, 4]]
W = [[2, 0], [1, -1], [0, 3]]
out = [[sum(x*w for x, w in zip(row, weights)) for weights in W] for row in X]
assert out == [[2, -1, 6], [6, -1, 12]]
print("✓ 批量矩阵乘法:", out)
close("小步梯度下降后的损失", 0.98**2 + 3*1.88**2, 11.5636)
close("大步梯度下降后的损失", (-1)**2 + 3*(-10)**2, 301)
close("密度 5 × 区间宽度 0.01", 5*0.01, 0.05)
close("[1,3] 的总体方差", pvariance([1, 3]), 1)

print("\n第 5–8 章：学习、前向、Adam、在线统计")
x, target = [1, 2], [3, 5]
w, b = 1, 0
err = [w*xi+b-yi for xi, yi in zip(x, target)]
dw, db = mean([2*e*xi for e, xi in zip(err, x)]), mean([2*e for e in err])
close("旧 MSE", mean([e**2 for e in err]), 6.5)
w, b = w-0.1*dw, b-0.1*db
close("新 w", w, 1.8)
close("新 b", b, 0.5)
close("新 MSE", mean([(w*xi+b-yi)**2 for xi, yi in zip(x, target)]), 0.65)
close("两层教学网络输出", 0.6-2*(math.exp(-0.7)-1), 1.606829392)
close("Adam 第一阶偏差修正", 0.2/(1-0.9), 2)
close("Adam 第二阶偏差修正", 0.004/(1-0.999), 4)
normalized = (6-2)/(4+0.01)
close("正确归一化", normalized, 0.997506234)
close("重复归一化的错误结果", (normalized-2)/4.01, -0.249998445)
old, batch = [0, 2], [4, 6]
n, m = len(old), len(batch)
delta = mean(batch)-mean(old)
merged_var = (n*pvariance(old)+m*pvariance(batch))/(n+m) + n*m/(n+m)**2*delta**2
close("在线合并方差 = 全量计算", merged_var, pvariance(old+batch))

print("\n第 9–11 章：回报、价值和不对奖励求导的策略梯度")
close("三步 [0,0,1] 折扣回报", 0+0.9*0+0.9**2*1, 0.81)
close("贝尔曼期望", 0.8*(0.1+0.9*2)+0.2*0.1, 1.54)
close("一次 TD 更新", 1+0.1*(0.1+0.9*1.5-1), 1.045)
close("按策略加权的优势均值", 0.75*0.5+0.25*(-1.5), 0)
p = 0.5
J = lambda probability: 3*probability + (1-probability)
score_gradient = p*(1/p)*3 + (1-p)*(-1/(1-p))*1
close("策略梯度 = 期望奖励有限差分", score_gradient, (J(p+h)-J(p-h))/(2*h))
close("减基线后的 A 样本贡献", (1/p)*(3-2), 2)
close("减基线后的 B 样本贡献", (-1/(1-p))*(1-2), 2)
close("高斯均值的单样本梯度贡献", (0.2-0)/0.5**2*1.5, 1.2)

print("\n第 12 章：GAE 的递推、展开与 n 步混合")
rewards, values = [0.1, 0.2, 1], [0.5, 0.4, 0.3, 0]
gamma, lam = 0.9, 0.8
deltas = [r+gamma*values[t+1]-values[t] for t, r in enumerate(rewards)]
advantages, running = [0.0]*3, 0.0
for t in reversed(range(3)):
    running = deltas[t]+gamma*lam*running
    advantages[t] = running
for t, expected in enumerate([0.37328, 0.574, 0.7]):
    close(f"GAE[{t}]", advantages[t], expected)
close("GAE 展开", sum((gamma*lam)**t*d for t, d in enumerate(deltas)), advantages[0])
one = rewards[0]+gamma*values[1]-values[0]
two = rewards[0]+gamma*rewards[1]+gamma**2*values[2]-values[0]
full = sum(gamma**t*r for t, r in enumerate(rewards))-values[0]
close("n 步目标混合 = GAE", (1-lam)*one+(1-lam)*lam*two+lam**2*full, advantages[0])
close("λ=1 的望远镜消去", sum(gamma**t*d for t, d in enumerate(deltas)), full)
close("rsl_rl 超时修正示例", 0.1+0.99*0.8, 0.892)

print("\n第 13 章：PPO 四个方向及平坦区")
def objective(ratio, advantage):
    return min(ratio*advantage, min(1.2, max(0.8, ratio))*advantage)

for ratio, adv, expected in [(1.5, 2, 2.4), (0.5, 2, 1), (0.5, -2, -1.6), (1.5, -2, -3)]:
    close(f"ratio={ratio}, A={adv}", objective(ratio, adv), expected)
    slope = (objective(ratio+h, adv)-objective(ratio-h, adv))/(2*h)
    plateau = (adv>0 and ratio>1.2) or (adv<0 and ratio<0.8)
    close("该项对 ratio 的导数", slope, 0 if plateau else adv)
close("同标准差高斯均值移动 0.2 的 KL", (1+0.2**2)/2-0.5, 0.02)

print("\n第 14–18 章：日志、奖励、执行器与课程的单位")
close("完整 20s 回合的奖励日志示例", 0.5*2*0.02*1000/20, 1)
close("只有 5s 回合的同项日志", 0.5*2*0.02*250/20, 0.25)
velocity_error_sq = (0.3-0.2)**2+(0-0.1)**2+0.05**2
close("速度平方误差和", velocity_error_sq, 0.0225)
close("速度核单步贡献", math.exp(-velocity_error_sq/0.1)*2*0.02, 0.031940649)
close("动作变化代价", 0.1**2+(-0.2)**2, 0.05)
ema = 0
for _ in range(50):
    ema = 0.98*ema+0.02*0.2
close("EMA 递推 = 几何级数", ema, 0.2*(1-0.98**50))
close("50 步 EMA", ema, 0.127166064)
close("目标角", 0.35+0.1-0.01, 0.44)
voltage = 0.04*200*0.00288*7.5
close("近似位置环电压", voltage, 0.1728)
close("含反电动势的近似电磁力矩", 0.366*voltage/2.81-0.366**2*0.2/2.81, 0.012972811)
close("3 个物理步的毫秒数", 3*0.005*1000, 15)
close("6 个物理步的毫秒数", 6*0.005*1000, 30)
close("DR 混合回报", 0.9*10+0.1*(-20), 7)
close("课程 500 次迭代对应控制步", 500*24, 12000)

print("\n全部手算复核通过。这里只验证公式与数字，不证明训练或真机行为。")
