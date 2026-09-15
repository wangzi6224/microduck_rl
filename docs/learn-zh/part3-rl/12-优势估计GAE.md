# 第 12 章 · 优势估计 GAE

> **上一章我们有了什么**：策略梯度 $\nabla J \approx \text{mean}[\nabla\ln\pi(a_i|s_i)\cdot\hat A_i]$。
>
> **这一章要补什么**：$\hat A_i$ 从哪来。两个极端各有毛病：用完整回报 $G_t - V(s_t)$ 要等一局结束、噪声大；
> 用一步 TD 误差 $\delta_t$ 噪声小、但全靠 critic 估得准。**GAE**（广义优势估计）用一个旋钮 λ 在两者之间连续调节，
> 项目用 λ = 0.95。本章末尾你会逐行读懂 rsl_rl 的 `compute_returns()`——它不到 15 行。
>
> **本章新词**：n 步回报、GAE、λ、偏差与方差、自举（time-out）。
> **需要的前提**：第 9 章回报、第 10 章 TD 误差。

---

## 12.1 造一段数据

用合成数据讲最清楚。实验 `ch12_gae.py` 造了 8 步：每步的奖励 $r_t$、critic 的估值 $V_t$、是否结束 $\text{done}_t$：

```
t  r_t  V_t  done_t
-  ---  ---  ------
0  0.1  0.5       0
1  0.1  0.6       0
2  0.1  0.7       0
3    1  0.4       1      ← 第 3 步拿到 +1，回合结束
4  0.1  0.3       0      ← 新回合
5  0.1  0.2       0
6   -1  0.1       1      ← 第 6 步 −1（摔了），结束
7  0.1  0.3       0      ← 又一个新回合，rollout 到此截断
```

rollout 之后那一步（第 8 步）critic 的估值 $V_{\text{last}} = 0.35$——最后一步的"下一格"要用它。

## 12.2 第一块积木：TD 误差 δ

第 10 章的 $\delta_t = r_t + \gamma V_{t+1} - V_t$，回合结束时 $V_{t+1}$ 当 0：

```
t     δ_t
-  ------
0   0.194
1   0.193
2  -0.204
3     0.6      ← 1.0 + 0 − 0.4
4  -0.002
5  -0.001
6    -1.1      ← −1.0 + 0 − 0.1
7  0.1465      ← 0.1 + 0.99×0.35 − 0.3
```

δ 本身就是一个（很粗的）优势估计："这一步过后比预期好/差多少"。它只用了一步真实奖励，其余全靠 critic。

## 12.3 两个极端

**λ = 0：只信一步。** $\hat A_t = \delta_t$。噪声最小（只有一个 $r_t$ 是随机的），但 critic 估错了它就错。

**λ = 1：全信真实奖励。** $\hat A_t = G_t - V_t$，回报 $G_t$ 用真实奖励一路加到回合结束（或 rollout 末尾用 $V_{\text{last}}$ 补上）。
不依赖中间任何 critic 估值，但把每一步的噪声全加进来了。

实验第 4 节验证了 λ=0 时 Â = δ，λ=1 时 returns = 蒙特卡洛回报——两个极端就是第 10 章的"TD" 和 "蒙特卡洛"。

## 12.4 GAE：用 λ 把 δ 串起来

**广义优势估计**（Generalized Advantage Estimation, GAE）把往后的 δ 用 $(\gamma\lambda)$ 打折加起来：

$$\hat A_t = \delta_t + (\gamma\lambda)\,\delta_{t+1} + (\gamma\lambda)^2\,\delta_{t+2} + \cdots$$

| 符号 | 就是 |
|---|---|
| $\delta_{t+k}$ | 往后第 k 步的 TD 误差 |
| $(\gamma\lambda)^k$ | 打 k 次折。γ 是第 9 章的折扣，λ 是新旋钮，两个相乘 |
| λ = 0 | 只剩 $\delta_t$ |
| λ = 1 | 全部 δ 相加，展开后中间的 V 两两抵消（"望远镜"），只剩 $G_t - V_t$ |

和第 9 章回报的递推一样，从后往前算最省事：

$$\hat A_t = \delta_t + \gamma\lambda\,\hat A_{t+1}\qquad\text{（回合结束处 } \hat A_{t+1} \text{ 当 0）}$$

实验第 3 节分别用"按定义逐项求和"和"从后往前递推"算，结果逐位相同：

```
t  Â (按定义求和)  Â (rsl_rl 递推)  returns = Â + V
-  --------------  ---------------  ---------------
0        0.694216         0.694216          1.19422
1        0.531862         0.531862          1.13186
2          0.3603           0.3603           1.0603
3             0.6              0.6                1
4       -0.975935        -0.975935        -0.675935
5        -1.03555         -1.03555         -0.83555
6            -1.1             -1.1               -1
7          0.1465           0.1465           0.4465
```

看第 4、5 步：奖励只有 0.1，可 Â 是 −0.98、−1.04——因为第 6 步的 −1 通过 $(\gamma\lambda)^k$ 传回来了。**GAE 把"两步之后要摔"的信息提前给了第 4 步。**

最后一列 **returns** = Â + V，它是训练 critic 的目标（第 10 章说过 critic 是监督学习，这就是它的"标准答案"）。

![不同 λ 的优势估计](../figures/ch12_gae_lambda.png)

## 12.5 为什么是 0.95：偏差和方差

实验第 5 节用 critic 的真值、给奖励加噪声，看不同 λ 下 $\hat A_0$ 散得多开：

```
   λ   Â_0 的均值  Â_0 的标准差
----  -----------  ------------
   0  -0.00693473       0.47955
 0.5    0.0407499      0.579298
0.95    0.0254169       1.39274
   1     0.129735       2.03155
```

λ 越大，标准差越大——因为把越多步的噪声加进来了。反过来，λ 越小越依赖 critic，critic 早期不准时估计就系统性地偏。
**偏差-方差权衡**（bias-variance tradeoff）：0.95 是大量实践里的折中，几乎所有 PPO 实现都用 0.9–0.97。

> **停一下，自测**：训练刚开始 critic 一塌糊涂时，λ=0 和 λ=1 哪个更靠谱？
> <details><summary>答案</summary>λ=1：它不依赖中间的 critic 估值（只在 rollout 末尾用一次）。这也是为什么 λ 取 0.95 而不是 0.5——宁可多一点噪声，少一点对 critic 的依赖。</details>

## 12.6 超时不是失败：time_out 自举

第 9 章留的伏笔。回合因为"20 秒到了"结束，后面其实还有分，但 rollout 里 done = 1，δ 的下一格价值被当成 0——机器人会被冤枉："走到 20 秒就被扣光了"。

rsl_rl 的修正：在超时那一步，**把 $\gamma V(s_t)$ 加回奖励**，再截断。实验第 6 节：

```
某一步超时：原始 r = 0.1，critic 估 V = 0.8
  rsl_rl 修正后的 r = r + γV = 0.8920
```

效果：这一步的 δ = (r + γV) + 0 − V ≈ r，好像后面还有 V 那么多分——不惩罚"只是时间到了"。摔倒（`fell_over`）没有这一步：后面真的没分。
这就是 cfg 里 `time_out=True` 标记的用途。

## 12.7 最后一步：优势归一化

算完全部 98,304 个 Â，减均值、除标准差：

$$\hat A \leftarrow \frac{\hat A - \text{mean}(\hat A)}{\text{std}(\hat A) + 10^{-8}}$$

这样不管奖励的绝对尺度是 0.01 还是 100，送进策略梯度的优势都在 ±1 量级，学习率才有统一的意义（第 8 章归一化的同一个道理）。

---

## 📍 映射到项目

> 现在可以逐行读了。

`rsl_rl/algorithms/ppo.py` 的 `compute_returns()`，一字不改：

```python
last_values = self.critic(obs).detach()                       # 12.1 节的 V_last
advantage = 0
for step in reversed(range(st.num_transitions_per_env)):      # 从后往前，24 步
    next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
    next_is_not_terminal = 1.0 - st.dones[step].float()       # 结束了就是 0
    # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
    delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]   # 12.2 节
    # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
    advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage                   # 12.4 节递推
    # Return: R_t = A(s_t, a_t) + V(s_t)
    st.returns[step] = advantage + st.values[step]             # critic 的训练目标
st.advantages = st.returns - st.values
st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)   # 12.7 节
```

实验第 3 节的 `gae_rsl_rl()` 就是把这段抄成 numpy。

**time_out 自举**在同一文件的 `process_env_step()`：

```python
if "time_outs" in extras:
    self.transition.rewards += self.gamma * torch.squeeze(
        self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1)   # 12.6 节：r += γV，只对超时的 env
```

**两个旋钮**在 `microduck_velocity_env_cfg.py`：`gamma=0.99`，`lam=0.95`。

## 🧪 动手实验

```bash
uv run python docs/learn-zh/labs/ch12_gae.py
```

1. 造 8 步数据。
2. 算 δ。
3. GAE：按定义求和 vs 递推，逐位对比。
4. λ 的两个极端。
5. 方差随 λ 增大。
6. time_out 自举。
7. 优势归一化。

**改一改**：第 1 节把第 6 步的 −1 改成 0，看第 4、5 步的 Â 怎么变——"未来的惩罚传回现在"就是 GAE 在做的事。

## 本章小结

- **δ** 是一步的优势估计；**GAE** 把往后的 δ 用 $(\gamma\lambda)^k$ 打折加起来，递推 $\hat A_t = \delta_t + \gamma\lambda\hat A_{t+1}$。
- λ=0 只信一步（低方差、靠 critic）；λ=1 全信真实奖励（高方差、不靠 critic）；**0.95 是折中**。
- **returns = Â + V** 是 critic 的训练目标。
- **超时**那一步把 γV 加回奖励，不冤枉"时间到了"；**摔倒**不加。
- 最后**归一化**优势到 ±1 量级。

**下一章**：有了 Â，策略梯度就能算了。可是同一批数据要用 5 遍（第 7 章的 epoch），
第二遍时策略已经变了，梯度公式的前提（数据来自当前策略）就不成立了。PPO 给这个问题加了一根"保险丝"——
[第 13 章 · PPO](13-PPO.md)。
