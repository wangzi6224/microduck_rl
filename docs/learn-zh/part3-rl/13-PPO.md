# 第 13 章 · PPO

> **上一章我们有了什么**：优势 Â（GAE）和策略梯度 $\text{mean}[\nabla\ln\pi\cdot\hat A]$。
>
> **这一章要补什么**：最后一块拼图。策略梯度有个隐含前提：数据必须是**当前策略**跑出来的。
> 可第 7 章说过同一批数据要切 4 份、过 5 遍——第二遍开始，策略已经变了，数据"过期"了。
> 硬用会怎样？策略一步跨太大，直接崩掉，而且崩了之后新数据也是崩的策略跑的，救不回来。
> **PPO**（Proximal Policy Optimization，近端策略优化）的全部内容就是给这一步加一根**保险丝**：变可以，别变太多。
> 本章末尾你能逐行读 rsl_rl 的 `update()`。
>
> **本章新词**：比率 r、裁剪、代理目标、价值损失、KL。
> **需要的前提**：第 4 章的 ln π，第 11 章策略梯度，第 12 章的 Â 和 returns。

---

## 13.1 数据过期的问题

策略梯度是 $\mathbb{E}_{a\sim\pi_\theta}[\cdots]$——期望下标写明"动作按**现在的** $\pi_\theta$ 抽"。
第一次更新时数据确实是它抽的。更新一次后 θ 变了，同一批数据变成"旧策略 $\pi_{\text{old}}$ 抽的"，
再用它算梯度就名不正言不顺。

这就是 **on-policy**（同策略）数据分布的要求，不能简化为“每条数据物理上只能使用一次”。PPO 在限制策略变化的前提下有限重用一批新采数据。

## 13.2 比率：新旧策略对同一个动作的看法差多少

修正的办法：给每个样本乘上一个"权重"，把旧策略的数据折算成新策略的（叫**重要性采样**，折叠里推导）。权重是：

$$r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)} = \exp\big(\ln\pi_\theta - \ln\pi_{\text{old}}\big)$$

| 符号 | 怎么读 | 就是 |
|---|---|---|
| $r_t(\theta)$ | "比率" | 新策略给这个动作的概率 ÷ 旧策略给的。**注意**：这个 r 不是奖励，是 ratio，同一个字母的不幸撞车 |
| $\pi_{\text{old}}$ | | 抽这个动作时的策略（第 11 章 📍 说过 `act()` 时把 ln π 存下来了，就是为了现在） |
| exp(ln − ln) | | 第 1 章：两个 log 相减再 exp，等于两个数相除；用 log 是为了数值稳定 |

实验 `ch13_ppo_clip.py` 第 1 节：

```
log π_old  log π_new     ratio
---------  ---------  --------
       -1         -1         1        ← 没变
       -1       -0.8    1.2214        ← 新策略更爱这个动作
       -1       -1.4   0.67032        ← 更不爱
```

第一次更新时新旧相同，比率全是 1；之后每更新一次，比率就偏离 1 一点。

<details>
<summary>重要性采样一行推导（可跳过）</summary>

$\mathbb{E}_{a\sim\pi_\theta}[f(a)] = \sum_a \pi_\theta(a) f(a) = \sum_a \pi_{\text{old}}(a)\frac{\pi_\theta(a)}{\pi_{\text{old}}(a)} f(a) = \mathbb{E}_{a\sim\pi_{\text{old}}}\big[r(\theta) f(a)\big]$

——想要"按新策略的期望"，可以用"按旧策略抽的样本"乘上比率来估。把 $f = \hat A$ 代进去，就得到 13.3 节的第一个式子。
</details>

## 13.3 加保险丝：裁剪

带比率的策略梯度目标是 $\mathbb{E}[r_t(\theta)\hat A_t]$——"优势为正的动作，把比率推高；为负的，推低"。
问题：推多少算够？没有上限，梯度会一直推，比率飙到 5、10，策略就跨崩了。

PPO 的做法：**对某个样本，向有利方向越过阈值后，代理目标不再增加**（ε = 0.2）。环境奖励不受这个裁剪操作影响：

$$L^{\text{CLIP}} = \mathbb{E}\Big[\min\big(r_t\hat A_t,\ \ \text{clip}(r_t,\,1-\varepsilon,\,1+\varepsilon)\,\hat A_t\big)\Big]$$

| 符号 | 就是 |
|---|---|
| $r_t\hat A_t$ | 不裁剪的目标 |
| $\text{clip}(r_t, 0.8, 1.2)$ | 把比率夹在 0.8 到 1.2 之间（第 0 章符号表） |
| $\min(\cdot,\cdot)$ | 两个里取小的——取"悲观"的那个 |

为什么是 min？分两种情况看实验第 2 节的表：

```
ratio  Â=+1 时的目标  Â=−1 时的目标
-----  -------------  -------------
  0.5            0.5           -0.8
  0.8            0.8           -0.8
    1              1             -1
  1.2            1.2           -1.2
  1.5            1.2           -1.5
```

![PPO 裁剪目标](../figures/ch13_ppo_clip.png)

- **Â > 0（好动作）**：比率涨到 1.2 以后，目标停在 1.2 不再涨——梯度为 0，"够了，别再更爱它了"。
  比率往下跌时不裁（左半边是斜线）——万一它跌了，还能拉回来。
- **Â < 0（坏动作）**：比率跌到 0.8 以后目标不再变——"够了，别再更讨厌它了"。往上涨时不裁。

**min 的作用**：两种情况下都只在"往有利方向走过头"时踩刹车，"往不利方向走"时不踩（保留纠错能力）。这就是"悲观取值"的含义。

处于平坦区域的样本，其裁剪代理项对比率的导数为 0；不是整个网络的总梯度为 0。实验第 4 节模拟了一个 mini-batch：策略稍微变了一点后，约 9% 的样本被裁掉。高比例提示需要查看更新是否过大；低比例也可能只是更新保守，不能单凭它判定训练好坏。

## 13.4 完整的损失：三项

PPO 一次更新同时训练 actor 和 critic，损失有三项：

$$\text{loss} = \underbrace{-L^{\text{CLIP}}}_{\text{actor}} + c_v\underbrace{(V_\theta - \text{returns})^2}_{\text{critic}} - c_e\underbrace{H}_{\text{熵}}$$

| 项 | 干什么 | 系数 |
|---|---|---|
| $-L^{\text{CLIP}}$ | 策略梯度（取负因为 torch 只会最小化） | 1 |
| 价值损失 | critic 的 MSE，目标是第 12 章的 returns。项目用了"裁剪版"（折叠） | `value_loss_coef=1.0` |
| 熵 | 第 11 章：别让 σ 缩太快 | `entropy_coef=0.01` |

实验第 5 节把三项算出来：

```
            项         值  系数
--------------  ---------  ----
surrogate_loss  -0.556254     1
    value_loss     0.0455     1
entropy (减去)      19.76  0.01
   总损失 loss  -0.708354
```

$L^{\text{CLIP}}$ 在论文里叫**代理目标**（surrogate objective），rsl_rl 的变量叫 `surrogate_loss`——"代理"是说它不是真正的 J，而是一个更便于用当前样本优化的替身，但不提供绝对稳定性保证。

<details>
<summary>裁剪版价值损失（可跳过）</summary>
和策略一样，critic 一步也不该变太多：$V_{\text{clipped}} = V_{\text{old}} + \text{clip}(V_\theta - V_{\text{old}}, -\varepsilon, \varepsilon)$，
损失取 $\max\big((V_\theta - \text{returns})^2,\ (V_{\text{clipped}} - \text{returns})^2\big)$。`use_clipped_value_loss=True`。
</details>

## 13.5 第二根保险丝：KL 自适应学习率

裁剪管的是单个样本。整体上"新旧策略差多远"还有一把尺子：**KL 散度**（Kullback–Leibler divergence），
两口钟之间的距离，两个高斯之间有闭式公式（实验第 6 节手算和 torch 对上）。

第 3 章 📍 见过的规则：每个 mini-batch 更新前比较当前分布与采集时的旧分布，量 KL，超过目标（0.01）的 2 倍 → 学习率 ÷1.5；不到一半 → ×1.5。

```
这次更新测到的 KL  调整后的学习率
-----------------  --------------
             0.03     0.000666667     ← 变太多，缩
            0.025     0.000444444     ← 再缩一次
            0.012     0.000444444     ← 在范围内，保持上次值
            0.004     0.000666667     ← 变太少，放
            0.003           0.001     ← 再放一次
             0.02           0.001     ← 恰好 2×目标，不触发严格大于
```

裁剪与 KL 自适应帮助控制变化，本项目据此把同一批数据用 5 遍；仍需观察训练是否稳定。

## 13.6 数一数

| | |
|---|---|
| 一次 rollout | 4096 env × 24 步 = 98,304 条 |
| 切 4 份 | 每份 24,576 条 |
| 过 5 遍 | 4 × 5 = **20 次更新**（20 次 backward、20 次 Adam step） |
| 然后 | 扔掉数据，用新策略再跑 24 步 |

---

## 13.7 四种情况逐格代数：min 到底裁掉哪一边

先固定一条旧数据：观测、已执行动作、旧 log 密度、优势都不变。
更新的只有新策略。设裁剪参数 ε=0.2，记 $r$ 为**概率或密度比率**。
本节 $r$ 不是环境奖励，也不是关节角。

当优势 $\hat A=2$，新策略把这个动作的密度提高到旧策略的 1.5 倍：

$$r\hat A=1.5\times2=3,$$
$$\operatorname{clip}(r,0.8,1.2)\hat A=1.2\times2=2.4,$$
$$\min(3,2.4)=2.4.$$

把 r 继续提高到 1.6，目标还是 2.4；该样本的这一项不再奖励继续提高。
但如果 r 掉到 0.5，未裁剪项为 1，裁剪项为 1.6，取小得到 1。
这时仍保留斜率，因为好动作的概率正在往坏方向走，需要有纠正信号。

负优势容易弄反，再完整算两次。设 $\hat A=-2$：

| 比率 r | 未裁剪 rÂ | 裁剪后的比率 × Â | min 结果 | 含义 |
|---:|---:|---:|---:|---|
| 0.5 | −1.0 | −1.6 | −1.6 | 已经大幅减少坏动作，不再奖励继续减少 |
| 1.5 | −3.0 | −2.4 | −3.0 | 反而增加坏动作，仍保留纠正信号 |

注意 −3 比 −2.4 **小**。如果这里拿错 min/max，就会把想保护的方向全倒过来。
代码采用最小化损失，整体取负，于是 $-\min(u,v)=\max(-u,-v)$。
所以论文的 min 和源码的 max 没有矛盾。

### “这一项没梯度”不等于参数不再变化

一个网络同时服务很多状态。当前样本在平坦区域，别的样本仍可能更新同一批权重，
熵项也可能改变分布，Adam 还保存着历史状态。
因此更新后此样本的真实比率仍可能超过 1.2 或低于 0.8。
**裁剪移除了过度改善这一样本的激励，没有给真实概率比率设置硬边界。**
这一边界也见 [OpenAI Spinning Up 的 PPO 说明](https://spinningup.openai.com/en/latest/algorithms/ppo.html)。

## 13.8 为什么“只用旧动作”却每次都要重新跑网络

旧动作假设是 `a=0.2`。旧策略的均值可能是 0，新策略均值可能已变成 0.1。
现在要问的是：“新策略给**当时做过的 0.2**多少密度？”
不是重新抽一个 0.3，再把它的密度与旧动作 0.2 的密度相比。
因为奖励和优势属于 0.2 那次经历，换了动作就失去了对应关系。

代码的核心语义是：

```text
旧 obs ──当前 actor──→ 新分布参数
旧 action ──────────→ 在新分布下求 log 密度
旧 log 密度 ────────→ 相减并 exp，得到 ratio
旧 advantage ───────→ 与 ratio 一起构造目标
```

重要性采样在固定状态下可修正动作分布的加权。
但新策略长期运行时，遇到的**状态分布**也会变。
PPO 的单步比率并没有完整修正整条轨迹的分布，所以目标称为“代理”，
不是保证等于新策略真实长期回报的精确公式。这也是不能无限重用旧样本的原因。

## 13.9 KL、优势归一化、loss：不要把三个数当成同一把尺

KL 衡量两份动作分布的差异。对一维高斯，旧分布为 $(\mu_o,\sigma_o)$，新分布为 $(\mu_n,\sigma_n)$：

$$D_{KL}(o\|n)=\log\frac{\sigma_n}{\sigma_o}
+\frac{\sigma_o^2+(\mu_o-\mu_n)^2}{2\sigma_n^2}-\frac12.$$

若两边标准差都为 1，均值从 0 移到 0.2，第一项为 0，
第二项是 $(1+0.04)/2=0.52$，最后减 0.5，KL=0.02。
若只移到 0.1，则 KL=0.005。相同标准差下，均值变化翻倍，KL 翻四倍。
KL 不对称，因此是“散度”，不满足通常距离的全部性质；这里用它作变化程度指标。

优势归一化回答“这条记录在这一批中相对好多少”。
如果所有比率都为 1，归一化优势均值约为 0，代理目标数值也约为 0。
**目标数值为 0 不等于梯度为 0**：例如函数 $f(x)=x$ 在 0 处值为 0，导数却为 1。

因此，surrogate loss 接近 0 不能独立证明学好了或没有学习。
要结合 KL、任务成功率、回合长度和实际轨迹判断。
clip fraction 低也未必有问题，可能只是更新足够保守；高则提示需进一步查看学习率和优势分布。

**自测**：ε=0.2、Â=−1、r=1.4，这个样本的策略目标是多少，是否落在平坦区？

**解释**：min(−1.4,−1.2)=−1.4，仍有斜率。它在增加比预期差的动作，不能在这边停止纠正。

## 📍 映射到项目

> 现在可以逐行读 `rsl_rl/algorithms/ppo.py` 的 `update()` 核心段了。

**比率与裁剪**（注意 rsl_rl 先取了负号，所以教科书的 min 变成 max）：

```python
ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))   # 13.2 节：exp(ln π_new − ln π_old)
surrogate = -torch.squeeze(batch.advantages) * ratio                               # −r·Â
surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param                            # −clip(r, 0.8, 1.2)·Â
)
surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()                    # max(−a, −b) = −min(a, b)
```

实验第 3 节验证了 `rsl_rl 的 max(−…) = −教科书的 min(…)`。

**价值损失与总损失**：

```python
value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
value_losses = (values - batch.returns).pow(2)
value_losses_clipped = (value_clipped - batch.returns).pow(2)
value_loss = torch.max(value_losses, value_losses_clipped).mean()
loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()   # 13.4 节三项
```

**KL 自适应**（同一函数，更新前）：

```python
kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
kl_mean = torch.mean(kl)
if kl_mean > self.desired_kl * 2.0:
    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
```

**五个数**在 `microduck_velocity_env_cfg.py`：

```python
clip_param=0.2,            # ε
entropy_coef=0.01,         # c_e
value_loss_coef=1.0,       # c_v
desired_kl=0.01,           # KL 目标
num_learning_epochs=5, num_mini_batches=4,
```

## 🧪 动手实验

```bash
uv run python docs/learn-zh/labs/ch13_ppo_clip.py
```

1. 比率。
2. 裁剪目标的形状，画图。
3. rsl_rl 的 max/负号写法 = 教科书的 min。
4. 一个 mini-batch 里被裁剪的比例。
5. 三项损失。
6. KL 自适应学习率；两个高斯的 KL 手算。
7. 样本量算术。

**改一改**：第 4 节把 `0.15`（新旧策略的差异）改成 `0.5`，看被裁剪比例升到多少。

## 本章小结

- 同一批数据用多遍 → 数据"过期"；用**比率** $r = \pi_\theta/\pi_{\text{old}}$ 折算。
- **裁剪**：min 只在样本“往有利方向走过头”时让该代理项变平；真实比率仍可能超出区间。
- 总损失 = $-L^{\text{CLIP}}$ + 1.0 × 价值损失 − 0.01 × 熵。
- **KL 自适应学习率**是第二根保险丝。
- 每次 rollout：98,304 条，切 4 份过 5 遍，20 次更新。

**下一章**：所有零件都齐了。把它们按时间顺序串成一个循环，对照冒烟训练打印出来的日志，
一行一行说清每个数字是哪个公式算出来的——[第 14 章 · 训练回路全貌](14-训练回路全貌.md)。
