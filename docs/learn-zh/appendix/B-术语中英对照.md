# 附录 B · 术语中英对照

按拼音排序。每个术语在正文首次出现处有根源定义，这里给一句话提示和位置。
（随各部完成逐步补全；覆盖全书。）

| 中文 | English | 一句话 | 首次出现 |
|---|---|---|---|
| 标准差 | standard deviation | 方差的平方根，"散得多开"的尺子 | §4.2 |
| 标准化 | standardization | $(x-\mu)/\sigma$ | §4.4 |
| 标准正态分布 | standard normal | $\mathcal{N}(0,1)$ | §4.4 |
| 参数 | parameter | 训练要调的数，全部打包记作 θ | §3.1 |
| 重参数化 | reparameterization | 采样写成 $\mu+\sigma\epsilon$ | §4.4 |
| 大数定律 | law of large numbers | 样本均值 → 期望 | §4.3 |
| 单位向量 | unit vector | 长度为 1 的向量 | §2.3 |
| 导数 | derivative | 输入动一点、输出动多少的极限 | §1.2 |
| 点积 / 内积 | dot product / inner product | 对应元素相乘再相加 | §2.2 |
| 独立 | independent | 互不影响，联合概率相乘 | §4.5 |
| 多维高斯 | multivariate Gaussian | 每维独立时密度相乘、log 相加 | §4.5 |
| 方差 | variance | 离均值距离平方的期望 | §4.2 |
| 范数 | norm | 向量长度 | §2.3 |
| 复合函数 | composition | 函数套函数 | §1.6 |
| 高斯分布 / 正态分布 | Gaussian / normal distribution | 钟形曲线 | §4.4 |
| 高斯核 | Gaussian kernel | $\exp(-\text{err}^2/\sigma^2)$，项目奖励形状 | §1.8 |
| 概率密度 | probability density | 连续变量的"概率高度" | §4.1 |
| 概率分布 | probability distribution | 每种结果的概率 | §4.1 |
| 函数 | function | 确定的输入→输出规则 | §1.1 |
| 均方误差 | mean squared error, MSE | 误差平方的平均 | §3.4 |
| 均匀分布 | uniform distribution | 区间内密度处处相等 | §4.1 |
| 矩阵 | matrix | 二维数组 | §2.5 |
| 极限 | limit | 无限逼近的那个值 | §1.2 |
| 链式法则 | chain rule | 复合函数的导数 = 各层导数相乘 | §1.3 |
| 梯度 | gradient | 全部偏导数排成的向量，上升最快方向 | §3.3 |
| 梯度裁剪 | gradient clipping | 梯度太长就按比例缩短 | §3.6 |
| 梯度下降 | gradient descent | 沿梯度反方向走一小步，重复 | §3.4 |
| 条件概率 | conditional probability | 给定某事后另一事的概率 | §4.5 |
| 蒙特卡洛估计 | Monte Carlo estimate | 用样本平均估期望 | §4.3 |
| 期望 | expectation | 概率加权平均 | §4.2 |
| 偏导数 | partial derivative | 只对一个变量求导 | §3.2 |
| 偏置 | bias | 一层网络里的 $\mathbf{b}$ | §2.6 |
| 权重 | weight | 一层网络里的 $W$ | §2.6 |
| 熵 | entropy | 分布的不确定程度 | §4.6 |
| 损失函数 | loss function | 输入参数，输出"多差" | §1.1, §3.4 |
| 随机变量 | random variable | 结果不确定但概率确定的量 | §4.1 |
| 探索 | exploration | 故意加噪声试新动作 | §4.5 |
| 投影重力 | projected gravity | 从机器人自己看重力指哪 | §2.4 |
| 维度 | dimension | 向量的长度 | §2.1 |
| 向量 | vector | 定长有序数组 | §2.1 |
| 学习率 | learning rate | 梯度下降的步长 α | §3.4 |
| 因变量 / 自变量 | dependent / independent variable | 输出 / 输入 | §1.1 |
| 指数函数 | exponential function | $e^x$ | §1.7 |
| 转置 | transpose | 矩阵行列互换 | §2.6 进阶 |
| 自然对数 | natural logarithm | $\ln$，exp 的逆 | §1.7 |
| 坐标系 | coordinate frame | 世界系 / 身体系 | §2.4 进阶 |
| 策略 | policy | 给定观测输出动作分布的函数 π | §4.5 |
| 四元数 | quaternion | 用 4 个数表示旋转 | §2.4 进阶 |
| 数据 / 模型 / 损失 / 优化 | data / model / loss / optimization | 学习的四个零件 | §5.2 |
| 训练 | training | 循环"算损失→backward→拧旋钮" | §5.2 |
| 泛化 | generalization | 没见过的数据上也准 | §5.4 |
| 过拟合 | overfitting | 把噪声背下来了 | §5.4 |
| 监督 / 无监督 / 强化学习 | supervised / unsupervised / reinforcement learning | 有答案 / 找结构 / 只有分数 | §5.5 |
| 神经元 | neuron | 点积 + 运费 + 过门 | §6.1 |
| 激活函数 | activation function | 那道门：ReLU、ELU | §6.1 |
| 层 / 隐藏层 | layer / hidden layer | 一排神经元 / 中间的层 | §6.3 |
| 多层感知机 | MLP, multi-layer perceptron | 层叠层的网络 | §6.3 |
| 前向传播 | forward pass | 从输入算到输出 | §6.5 |
| actor / critic | actor / critic | 出动作的网络 / 估分数的网络 | §6.6 |
| 计算图 | computational graph | 算式画成流水线 | §7.1 |
| 反向传播 | backpropagation | 从损失倒着乘汇率 = `backward()` | §7.2 |
| SGD | stochastic gradient descent | 用一小批数据的梯度下降 | §7.3, §7.5 |
| Adam | Adam | 每个旋钮自适应步长的优化器 | §7.4 |
| mini-batch / epoch | mini-batch / epoch | 一小批 / 全部批过一遍 | §7.5 |
| 归一化 | normalization | 每项减均值除标准差 | §8.3 |
| 在线统计 | running statistics | 每批挪一点，不存历史 | §8.4 |
| 冻结 | freeze (eval mode) | 推理时不再更新 μ、σ | §8.5 |
| 烘焙 | bake in | 归一化器和网络合成一个模块导出 | §8.6 |
| 强化学习 | reinforcement learning, RL | 没有答案只有分数，数据自己跑 | §9.1 |
| 智能体 / 环境 | agent / environment | 做决定的 / 其他一切 | §9.2 |
| 步 / 轨迹 | step / trajectory | 一圈看-做-得分 / 一回合的整串 | §9.2 |
| 状态 / 观测 | state / observation | 全部真相 / 能看到的部分 | §9.3 |
| 回合 | episode | 从开始到结束的一局 | §9.3 |
| 终止 / 超时 | termination / time-out (truncation) | 真的完了 / 只是时间到 | §9.3 |
| 马尔可夫性质 / MDP | Markov property / Markov decision process | 下一步只看现在 / 五元组打包 | §9.3 进阶 |
| 回报 | return | 往后奖励的折扣总和 | §9.4 |
| 折扣因子 | discount factor | γ，远处奖励打折 | §9.4 |
| 价值函数 | value function | 从 s 出发的平均回报 | §10.1 |
| 贝尔曼方程 | Bellman equation | 价值的递归关系 | §10.2 |
| 蒙特卡洛 / TD | Monte Carlo / temporal difference | 跑完取平均 / 走一步修一点 | §10.1, §10.3 |
| 自举 | bootstrapping | 目标里用自己的估计 | §10.3 |
| 动作价值 / 优势 | action value Q / advantage A | 先做 a 再按策略 / 比平均好多少 | §10.4 |
| 非对称 actor-critic | asymmetric actor-critic | critic 看特权信息 | §10.5 |
| 策略梯度 | policy gradient | J 对 θ 的梯度 | §11.1 |
| log-derivative trick | log-derivative (likelihood ratio) trick | ∇π = π∇ln π | §11.2 |
| 基线 | baseline | 减去不改期望、降方差 | §11.4 |
| REINFORCE | REINFORCE | 最基本的策略梯度算法 | §11.5 |
| GAE | generalized advantage estimation | 用 λ 串起来的 δ | §12.4 |
| 偏差-方差权衡 | bias-variance tradeoff | λ 小偏、λ 大抖 | §12.5 |
| on-policy | on-policy | 数据只能来自当前策略 | §13.1 |
| 重要性采样 / 比率 | importance sampling / ratio | 用旧数据估新期望 | §13.2 |
| 裁剪 | clipping | 比率超出 [0.8,1.2] 停止奖励 | §13.3 |
| 代理目标 | surrogate objective | 可安全优化的替身 L^CLIP | §13.4 |
| KL 散度 | KL divergence | 两个分布的距离 | §13.5 |
| PPO | proximal policy optimization | 带保险丝的策略梯度 | 第 13 章 |
| rollout / 迭代 | rollout / iteration | 跑一段记下来 / 训练循环的一圈 | §14.1 |
| 冒烟测试 | smoke test | 64 env × 5 iter 的快速检查 | §14.2 |
| 本体感受 | proprioception | 自己身上传感器能测到的量 | §15.1 |
| 命令块 | command block | 观测里的 13 维指令 | §15.3 |
| 零填充 / 死权重 | zero-padding / dead weights | 不用的槽保留并喂小随机值 | §15.3 |
| 特权观测 | privileged observation | 只有仿真知道、critic 专用 | §15.4 |
| 热切换 | hot-swap | 多个 ONNX 共用 61 维格式互换 | §15.3, §19.6 |
| 代价 / 自负号惩罚 | cost / self-negating penalty | ≥0 配负权 / ≤0 配正权 | §16.2 |
| 盒式指示器 | box indicator | 在区间内为 1 否则 0（air_time） | §16.3 |
| 势能式塑形 | potential-based shaping | 只付"进步"，不付"待着" | §16.6 |
| 目标角 | target joint position | HOME + 动作偏移 | §17.1 |
| 固件位置环 | firmware position loop | 舵机芯片里的 P 控制 | §17.2 |
| 反电动势 | back-EMF | 转得越快推力越小 | §17.2 |
| BAM | better actuator models | 用真舵机数据拟合的电压模型 | §17.2 |
| 域随机化 | domain randomization, DR | 每只机器人抽一套参数 | §17.3 |
| 课程学习 | curriculum learning | 先易后难的时间表 | §18.1 |
| 环境步 | environment steps | 迭代 × 24 | §18.2 |
| 检查点 | checkpoint | model_N.pt，全部旋钮 + 统计 | §19.1 |
| ONNX | Open Neural Network Exchange | 跨框架的计算图文件 | §19.2 |
