# 附录 A · 符号总表

全书用到的数学符号，按首次出现的章节排列。每个符号在正文首次出现处有完整讲解，这里只是速查。
同一字母可能有不同含义，尤其注意下表后面的同名符号辨析。

| 符号 | 读音 | 含义 | 首次出现 |
|---|---|---|---|
| $f(x)$ | f of x | 函数 f 在 x 处的值 | §1.1 |
| $\Delta x$ | 德尔塔 x | x 的变化量 | §1.2 |
| $f'(x)$、$\frac{dy}{dx}$ | f 撇、dy 比 dx | 导数 | §1.2 |
| $\lim_{h\to0}$ | h 趋近 0 的极限 | 让 h 无限小 | §1.2 |
| $\frac{dy}{du}\cdot\frac{du}{dx}$ | | 链式法则 | §1.6 |
| $e$、$\exp(x)$、$e^x$ | | 自然常数 2.718…、指数函数 | §1.7 |
| $\ln x$、$\log x$ | 自然对数 | exp 的逆运算；ML 里 log 默认以 e 为底 | §1.7 |
| $\exp(-\text{err}^2/\sigma^2)$ | | 高斯核（项目奖励形状） | §1.8 |
| $\mathbf{v}$、$v_i$ | 向量 v、v 下标 i | 一串数、第 i 个元素 | §2.1 |
| $\mathbb{R}^n$ | R 的 n 次方 | n 维实数向量的集合 | §2.1 |
| $\mathbf{a}\cdot\mathbf{b}$、$\mathbf{a}^\top\mathbf{b}$ | a 点 b | 点积 $\sum a_ib_i$ | §2.2 |
| $\sum_{i=1}^n$ | 对 i 从 1 到 n 求和 | 累加 | §2.2 |
| $\|\mathbf{v}\|$ | v 的范数 | 长度 $\sqrt{\sum v_i^2}$ | §2.3 |
| $\hat{\mathbf{v}}$ | v 帽 | 单位向量 $\mathbf{v}/\|\mathbf{v}\|$（帽子在别处也表示"估计值"） | §2.3 |
| $W$、$W_{ij}$、$m\times n$ | | 矩阵、第 i 行 j 列元素、m 行 n 列 | §2.5 |
| $W^\top$ | W 转置 | 行列互换 | §2.6 进阶 |
| $W\mathbf{x}+\mathbf{b}$ | | 一层神经网络 | §2.6 |
| $\partial f/\partial x$ | f 对 x 的偏导 | 其他变量当常数 | §3.2 |
| $\nabla f$、$\nabla_\theta L$ | 纳布拉 f、L 对 θ 的梯度 | 全部偏导数排成的向量 | §3.3 |
| $\theta$ | 西塔 | 全部参数 | §3.4 |
| $\theta\leftarrow\theta-\alpha\nabla L$ | | 梯度下降 | §3.4 |
| $\alpha$ | 阿尔法 | 学习率 | §3.4 |
| $\hat y$ | y 帽 | 预测值 | §3.4 |
| $L(\theta)$ | | 损失函数 | §3.4 |
| $\min(1, c/\|\mathbf{g}\|)$ | | 梯度裁剪系数 | §3.6 |
| $X\sim U(a,b)$ | X 服从均匀分布 | 密度 $1/(b-a)$ | §4.1 |
| $p(x)$ | | 概率（离散）/ 概率密度（连续） | §4.1 |
| $\mathbb{E}[X]$、$\mathbb{E}_{x\sim p}[f(x)]$ | X 的期望 | 概率加权平均 | §4.2 |
| $\int\cdots dx$ | 对 x 积分 | 连续版求和 | §4.2 |
| $\mu$、$\mathrm{Var}[X]$、$\sigma$ | 缪、方差、西格玛 | 均值、方差、标准差 | §4.2 |
| $\bar x_n$ | x 杠 | 样本均值 | §4.3 |
| $\mathcal{N}(\mu,\sigma^2)$ | 正态分布 | 高斯分布，第二个参数是方差 | §4.4 |
| $\pi$（数） | 派 | 圆周率 | §4.4 |
| $z=(x-\mu)/\sigma$ | | 标准化 | §4.4 |
| $\mu+\sigma\epsilon$ | | 重参数化采样 | §4.4 |
| $\prod_{i=1}^n$ | 连乘 | | §4.5 |
| $p(a\mid s)$ | 给定 s 下 a 的概率 | 条件概率 | §4.5 |
| $\pi_\theta(a\mid s)$ | 策略 | 参数为 θ 的条件分布 | §4.5 |
| $\mu_\theta(s)$ | | 策略网络输出的均值 | §4.5 |
| $H[p]$ | 熵 | $-\mathbb{E}[\log p]$ | §4.6 |
| $\text{ELU}(z)$、$\text{ReLU}(z)$ | | 激活函数（门） | §6.1 |
| $\mathbf{h}_k$ | | 第 k 个隐藏层的输出 | §6.3 |
| $\mathbb{R}^{61}\to\mathbb{R}^{14}$ | | 策略网络的输入输出维度 | §6.5 |
| $\partial L/\partial w_1 = \prod(\text{本地汇率})$ | | 反向传播 | §7.2 |
| $m$、$v$ | | Adam 的两本账：梯度平均、梯度平方平均 | §7.4 |
| $\theta \leftarrow \theta - \alpha\, m/(\sqrt v+\varepsilon)$ | | Adam 更新 | §7.4 |
| $\tilde x_j = (x_j-\mu_j)/(\sigma_j+\varepsilon)$ | x 波浪 | 归一化后的第 j 项 | §8.3 |
| $\mu_{\text{新}} = \mu_{\text{旧}} + \text{rate}\,(\mu_{\text{批}}-\mu_{\text{旧}})$ | | 在线均值更新 | §8.4 |
| $s_t$、$o_t$、$a_t$、$r_t$ | | 状态、观测、动作、奖励（第 t 步） | §9.3 |
| $\tau$ | 陶 | 轨迹：一回合的整串 (s, a, r) | §9.2 |
| $G_t = \sum_k \gamma^k r_{t+k}$ | | 回报 | §9.4 |
| $\gamma$ | 伽马 | 折扣因子 0.99，视界 ≈ 1/(1−γ) 步 | §9.4 |
| $J(\pi) = \mathbb{E}[G_0]$ | | 策略的目标值 | §9.5 |
| $V^\pi(s)$ | | 状态价值 | §10.1 |
| $V(s) = \mathbb{E}[r + \gamma V(s')]$ | | 贝尔曼方程 | §10.2 |
| $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ | 德尔塔 | TD 误差 | §10.3 |
| $Q^\pi(s,a)$、$A^\pi = Q - V$ | | 动作价值、优势 | §10.4 |
| $\nabla\pi = \pi\nabla\ln\pi$ | | log-derivative trick | §11.2 |
| $\nabla J = \mathbb{E}[\nabla\ln\pi(a)\cdot\hat A]$ | | 策略梯度 | §11.4 |
| $b$ | | 基线 | §11.4 |
| $\hat A_t = \sum_k (\gamma\lambda)^k\delta_{t+k}$ | A 帽 | GAE | §12.4 |
| $\lambda$ | 拉姆达 | GAE 的偏差-方差旋钮 0.95 | §12.5 |
| returns $= \hat A + V$ | | critic 的训练目标 | §12.4 |
| $r_t(\theta) = \pi_\theta/\pi_{\text{old}}$ | | 比率（不是奖励！） | §13.2 |
| $\text{clip}(r, 1-\varepsilon, 1+\varepsilon)$ | | 裁剪，ε = 0.2 | §13.3 |
| $L^{\text{CLIP}}$ | | PPO 代理目标 | §13.3 |
| $\text{KL}$ | | 两个分布的散度，不是对称距离；目标 0.01 | §13.5 |
| $q^{\text{target}} = \text{HOME} + a\cdot\text{scale} - \text{bias}$ | | 动作 → 目标角 | §17.1 |
| $V = v_{in}\,\text{clip}(k_p\,g\,(q^*-q), -1, 1)$ | | 固件位置环 → 电压 | §17.2 |
| $\tau = k_tV/R - k_t^2\dot q/R$ | 陶 | 直流电机力矩，反电动势 | §17.2 |
| $\xi\sim\text{DR}$ | 克西 | 随机抽的一套物理参数 | §17.3 |
| $\max_\theta\mathbb{E}_\xi[\mathbb{E}[G\mid\xi]]$ | | 域随机化下的训练目标 | §17.3 |
| 环境步 $= $ 迭代 $\times 24$ | | 课程表的横轴 | §18.2 |
| $\Delta\cos(\text{tilt})$ | | 势能式塑形：只付进步 | §16.6 |


## 同名字母不代表同一个量

| 容易混淆的符号 | 区别 | 详解 |
|---|---|---|
| 奖励宽度 s 与动作标准差 σ | 前者是评分参数，后者是探索分布参数；奖励核分母 s² 与正态分母 2σ² 还差一个 2 | §4.10 |
| r：奖励与 PPO 比率 | 环境奖励评估转移；比率比较同一旧动作的新旧密度 | §13.7 |
| α：学习率与 EMA 系数 | 一个控制参数更新，另一个控制新误差占统计值的比例 | §7.7、§16.9 |
| V：价值与电压 | 第 10 章预测回报，第 17 章的 V 是伏特；看单位与上下文 | §10.6、§17.4 |
| σ：归一化统计与探索噪声 | 前者按观测采样统计，后者通过策略目标训练 | §8.7、§4.10 |
| step：物理、控制、样本、优化器 | 0.005 s 子步；0.02 s 控制步；跨环境条数；参数更新次数 | §14.6、§17.5 |
| Â 与归一化后的 Â | critic 目标使用归一化前的 Â+V；actor 可使用随后归一化的优势 | §12.8 |
| $\Delta\Phi$ 与 $\gamma\Phi(s')-\Phi(s)$ | 前者是普通进步差分，后者与折扣目标的势函数塑形对应；还需处理终止边界 | §16.10 |
