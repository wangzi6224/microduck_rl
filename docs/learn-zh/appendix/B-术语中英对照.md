# 附录 B · 术语中英对照

按拼音排序。每个术语在正文首次出现处有根源定义，这里给一句话提示和位置。
（随各部完成逐步补全；覆盖全书。）

| 中文 | English | 一句话 | 首次出现 |
|---|---|---|---|
| 标准差 | standard deviation | 方差的平方根，"散得多开"的尺子 | §4.4 |
| 标准化 | standardization | $(x-\mu)/\sigma$；本书主名是"归一化"（见第 8 章） | §4.7 |
| 标准正态分布 | standard normal | $\mathcal{N}(0,1)$，正文叫"标准钟" | §4.7（折叠里给名字） |
| 参数 | parameter | 训练要调的数，全部打包记作 θ | §3.1 |
| 重参数化 | reparameterization | 采样写成 $\mu+\sigma\epsilon$ | §4.7（折叠里给名字） |
| 大数定律 | law of large numbers | 样本均值 → 期望 | §4.5 |
| 单位向量 | unit vector | 长度为 1 的向量 | §2.3 |
| 导数 | derivative | 输入动一点、输出动多少的极限 | §1.4 |
| 点积 / 内积 | dot product / inner product | 对应元素相乘再相加 | §2.2 |
| 独立 | independent | 互不影响，联合概率相乘 | §4.5 |
| 多维高斯 | multivariate Gaussian | 每维独立时密度相乘、log 相加 | §4.7（折叠里）、§4.8 |
| 变化率 | rate of change | 输出变了多少 ÷ 输入变了多少；直线上处处相同，就是斜率 | §1.3 |
| 方差 | variance | 离均值距离平方的期望 | §4.4 |
| 范数 | norm | 向量长度 | §2.3 |
| 复合函数 | composition | 函数套函数 | §1.7 |
| 高斯分布 / 正态分布 | Gaussian / normal distribution | 钟形曲线 | §4.6 |
| 高斯核 | Gaussian kernel | $\exp(-\text{err}^2/\sigma^2)$，项目奖励形状（正文叫钟形打分） | §1.9 |
| 概率密度 | probability density | 连续变量每单位宽度的可能性；高度不是概率，面积才是 | §4.2 |
| 概率分布 | probability distribution | 每种结果的概率 | §4.1 |
| 函数 | function | 确定的输入→输出规则 | §1.1 |
| 均方误差 | mean squared error, MSE | 误差平方的平均 | §3.5 |
| 均匀分布 | uniform distribution | 区间内密度处处相等 | §4.2 |
| 矩阵 | matrix | 二维数组 | §2.5 |
| 极限 | limit | 无限逼近的那个值 | §1.4（折叠里） |
| log 概率 | log-probability | 概率（密度）取自然对数；14 个相乘变成 14 个相加，代码里叫 `log_prob` | §4.8 |
| 链式法则 | chain rule | 复合函数的导数 = 各层导数相乘 | §1.7 |
| 梯度 | gradient | 全部偏导数排成的向量，上升最快方向 | §3.3 |
| 梯度裁剪 | gradient clipping | 梯度太长就按比例缩短 | §3.7 |
| 梯度下降 | gradient descent | 沿梯度反方向走一小步，重复 | §3.4 |
| 条件概率 | conditional probability | 给定某事后另一事的概率 | §4.7 |
| 蒙特卡洛估计 | Monte Carlo estimate | 用样本平均估期望（"多抽几次取平均"） | §4.5 |
| 期望 | expectation | 概率加权平均 | §4.3 |
| 优化器 | optimizer | 拿到梯度后真正去拧旋钮的那段程序（SGD、Adam） | §7.3 |
| 偏导数 | partial derivative | 只对一个变量求导 | §3.2 |
| 偏置 | bias | 一层网络里的 $\mathbf{b}$ | §2.6 |
| 权重 | weight | 一层网络里的 $W$ | §2.6 |
| 熵 | entropy | 分布的不确定程度 | §4.9 |
| 损失函数 | loss function | 输入参数，输出"多差" | §1.1（先见名字）, §3.1（正式介绍） |
| 随机变量 | random variable | 结果不确定但概率确定的量 | §4.1 |
| 探索 | exploration | 故意加噪声试新动作 | §9.4（第 11 章再细讲；§4.7 先见这个词） |
| 投影重力 | projected gravity | 从机器人自己看重力指哪 | §2.4 |
| 维度 | dimension | 向量有几个分量（"几维向量"）；也可能在问数组有几条轴 | §2.8（"几维向量"的说法先见 §2.1） |
| 向量 | vector | 定长有序数组 | §2.1 |
| 学习率 | learning rate | 梯度下降的步长 α | §3.4 |
| 因变量 / 自变量 | dependent / independent variable | 输出 / 输入 | §1.1 |
| 指数函数 | exponential function | $e^x$ | §1.8 |
| 转置 | transpose | 矩阵行列互换 | §2.6 进阶 |
| 自然对数 | natural logarithm | $\ln$，exp 的逆 | §1.8（先见名字）、§4.8（正式讲） |
| 坐标系 | coordinate frame | 世界系 / 身体系 | §2.4 进阶 |
| 策略 | policy | 给定观测输出动作分布的函数 π | §9.4（正式定义；§4.7 先见记号 π(a∣s)） |
| 四元数 | quaternion | 用 4 个数表示旋转 | §2.4 进阶 |
| 数据 / 模型 / 损失 / 优化 | data / model / loss / optimization | 学习的四个零件 | §5.2 |
| 训练 | training | 循环"算损失→backward→拧旋钮" | §5.3 |
| 泛化 | generalization | 没见过的数据上也准 | §5.5 |
| 过拟合 | overfitting | 把噪声背下来了 | §5.5 |
| 监督 / 无监督 / 强化学习 | supervised / unsupervised / reinforcement learning | 有答案 / 找结构 / 只有分数 | §5.7（强化学习只点到，第 9 章正式讲） |
| 神经元 | neuron | 点积 + 运费 + 过门 | §6.1 |
| 激活函数 | activation function | 那道门：ReLU、ELU | §6.1 |
| 层 / 隐藏层 | layer / hidden layer | 一排神经元 / 中间的层 | §6.2（层）、§6.3（隐藏层） |
| 多层感知机 | MLP, multi-layer perceptron | 层叠层的网络 | §6.3 |
| 滑动平均 | moving average | 新值占一点、旧值占大头的平均；Adam 的两本账都是它 | §7.4 |
| 起步修正 | bias correction（资料上常译"偏差修正"） | 滑动平均从 0 起步会偏低，除以 1 − βᵗ 补回来；本书为了不和"偏差 = 指令 − 实际"混，改叫起步修正 | §7.4 |
| 前向传播 | forward pass | 从输入算到输出 | §6.6 |
| actor / critic | actor / critic | 出动作的网络 / 估分数的网络 | §6.7 |
| 计算图 | computational graph | 算式画成流水线 | §7.1 |
| 反向传播 | backpropagation | 从损失倒着乘汇率 = `backward()` | §7.2 |
| SGD | stochastic gradient descent | 用一小批数据的梯度下降 | §7.3（"随机"的来历在 §7.6） |
| Adam | Adam | 每个旋钮自适应步长的优化器 | §7.5 |
| mini-batch / epoch | mini-batch / epoch | 一小批 / 全部批过一遍 | §7.6 |
| 归一化 | normalization | 每项减均值除标准差 | §8.3 |
| 在线统计 | running statistics | 每批挪一点，不存历史 | §8.4 |
| 冻结 | freeze (eval mode) | 推理时不再更新 μ、σ | §8.5 |
| 烘焙 | bake in | 归一化器和网络合成一个模块导出 | §8.6 |
| 强化学习 | reinforcement learning, RL | 没有答案只有分数，数据自己跑 | §9.1 |
| 智能体 / 环境 | agent / environment | 做决定的 / 其他一切 | §9.2 |
| 步 / 轨迹 | step / trajectory | 一圈看-做-得分 / 一回合的整串 | §9.5 |
| 状态 / 观测 | state / observation | 全部真相 / 能看到的部分 | §9.3 |
| 回合 | episode | 从开始到结束的一局 | §9.5 |
| 终止 / 超时 | termination / time-out (truncation) | 真的完了 / 只是时间到 | §9.6 |
| 马尔可夫性质 / MDP | Markov property / Markov decision process | 下一步只看现在 / 五元组打包 | §9.3 进阶 |
| 回报 | return | 往后奖励的折扣总和 | §9.7 |
| 折扣因子 | discount factor | γ，远处奖励打折 | §9.7 |
| 价值函数 | value function | 从 s 出发的平均回报 | §10.1 |
| 贝尔曼方程 | Bellman equation | 价值的递归关系 | §10.2 |
| 蒙特卡洛 / TD | Monte Carlo / temporal difference | 跑完取平均 / 走一步修一点 | §10.1, §10.3 |
| 自举 | bootstrapping | 目标里用自己的估计 | §10.3 |
| 动作价值 / 优势 | action value Q / advantage A | 先做 a 再按策略 / 比平均好多少 | §10.4 |
| 非对称 actor-critic | asymmetric actor-critic | critic 看特权信息 | §10.5 |
| 策略梯度 | policy gradient | J 对 θ 的梯度 | §11.1 |
| 梯度上升 | gradient ascent | 顺着梯度走、让 J 变大（第 3 章的减号换成加号） | §11.1 |
| 对数导数技巧 | log-derivative (likelihood ratio) trick | ∇π = π∇ln π：抽到谁就推谁，推多少看分数 | §11.2 |
| 基线 | baseline | 减去不改期望、降方差 | §11.4 |
| REINFORCE | REINFORCE | 最基本的策略梯度算法 | §11.5 |
| 熵奖励 | entropy bonus | 给"还愿意试"一点分，别让钟缩得太快 | §11.6 |
| n 步回报 | n-step return | 先记 n 步真实奖励，剩下的用 critic 的估计补上 | §12.2 |
| 广义优势估计 / GAE | generalized advantage estimation | 用 λ 串起来的 δ | §12.3 |
| 偏差 | bias | 统计意义：估计的平均值系统性地偏离真值（第 1–2 章的"偏差 = 指令 − 实际"不是一回事）；第 7 章 7.4 的起步修正修的就是这一种 | §12.5 |
| 偏差-方差权衡 | bias-variance tradeoff | λ 小偏、λ 大抖 | §12.5 |
| 超时自举 | time-out bootstrapping | 到点截断的那一步，给奖励补上 γ × critic 的估计 | §12.6 |
| 优势归一化 | advantage normalization | 一批优势减均值、除以标准差再给 actor | §12.7 |
| 同策略 | on-policy | 数据只能来自当前策略 | §13.1 |
| 重要性采样 / 比率 | importance sampling / ratio | 用旧数据估新期望 | §13.2 |
| 裁剪 | clipping | 比率超出 [0.8,1.2] 停止奖励 | §13.3 |
| 代理目标 | surrogate objective | 可安全优化的替身 L^CLIP | §13.4 |
| KL 散度 | KL divergence | 两个分布的距离 | §13.5 |
| 自适应学习率 | adaptive learning rate | 按 KL 自动调学习率：超过目标的 2 倍就 ÷ 1.5，不到一半就 × 1.5 | §13.5 |
| PPO | proximal policy optimization | 加了一道"别改太多"保险的策略梯度 | §13.3 |
| 迭代 | iteration | 训练循环的一圈：采一批记录 + 拧 20 次（名字在 §7.6 给出；第 14 章 14.1 节把它摆进四只钟里） | §7.6 |
| 采集 | rollout / collection | 用当前的策略原样走 24 步，每一步记一行 | §14.2 |
| 仿真时间 | simulated time | 仿真世界里过去的时间 = 环境步 × 0.02 秒 | §14.1 |
| 墙上时间 | wall-clock time | 你手表上真的过去的时间（编译、日志都算在内） | §14.1 |
| 符号约定 | sign convention | 函数的正负 × 权重的正负必须为负：惩罚项的日志读数一定 ≤ 0 | §14.6 |
| 冒烟测试 | smoke test | 64 env × 5 iter 的快速检查 | §14.10 |
| 半开区间 | half-open interval | [a, b)：含 a、不含 b；这一块有 b − a 项，边界编号只归后一块 | §15.1 |
| 本体感受 | proprioception | 自己身上传感器能测到的量 | §15.2 |
| 右手定则 | right-hand rule | 大拇指指向转轴的正方向，四指弯的方向就是"正着转" | §15.2 |
| 命令块 | command block | 观测里的 13 维指令 | §15.5 |
| 零填充 / 死权重 | zero-padding / dead weights | 不用的槽保留并喂小随机值 | §15.7 |
| 特权观测 | privileged observation | 只有仿真知道、critic 专用 | §15.8 |
| 热切换 | hot-swap | 多个 ONNX 共用 61 维格式互换 | §15.6（另见第 19 章） |
| 代价 / 自负号惩罚 | cost / self-negating penalty | ≥0 配负权 / ≤0 配正权 | §16.2 |
| 盒式指示器 | box indicator | 在区间内为 1 否则 0（air_time） | §16.3 |
| 势能式塑形 | potential-based shaping | 用势函数构造附加奖励，需匹配折扣与终止条件 | §16.10 |
| 目标角 | target joint position | HOME + 动作偏移 | §17.1 |
| 占空比 | duty cycle | 每一小段时间里开关开着的比例（−1 到 1），决定给电机多少电压 | §17.2 |
| 固件位置环 | firmware position loop | 舵机芯片里的 P 控制 | §17.2 |
| 反电动势 | back-EMF | 转得越快推力越小 | §17.3 |
| 执行器 | actuator | 把目标角变成电机力矩的那一整套（每个 0.005 s 物理步算一次） | §17.4 |
| BAM | better actuator models | 用真舵机数据拟合的电压模型 | §17.4 |
| 域随机化 | domain randomization, DR | 每只机器人抽一套参数 | §17.7 |
| 课程学习 | curriculum learning | 先易后难的时间表 | §18.1 |
| 环境步 | environment steps | 迭代 × 24 | §18.2 |
| 检查点 | checkpoint | model_N.pt，全部旋钮 + 统计 | §19.1 |
| ONNX | Open Neural Network Exchange | 跨框架的计算图文件 | §19.2 |


## 容易混用的词，放在一起比较

| 词组 | 一句话分清 | 去哪里手算 |
|---|---|---|
| 奖励 / 回报 / 价值 / 优势 | 一步评分 / 一段实际折扣总分 / 给定策略的期望回报 / 动作比当前处境平均好多少 | §9.7、§9.9、§10.1、§10.4 |
| 概率 / 密度 | 离散结果的可能性 / 连续区间每单位宽度的可能性；密度高度本身不是概率 | §4.2 |
| 参数 / 超参数 / 统计量 | 梯度更新的权重 / 人设的训练规则 / 从数据累积的均值与方差 | §5.3、§8.4 |
| 前向 / 反向 / 优化器更新 | 算输出 / 算导数 / 改参数 | §7.3、§7.7 |
| rollout / episode / epoch / iteration | 收集一段 / 从开始到结束一局 / 一批数据用一遍 / 一次采集与更新循环 | §14.1、§14.2 |
| 终止 / 超时 / 采集截断 | 任务真的结束 / 持续任务时间上限 / 只是收够一批 | §12.6 |
| 梯度裁剪 / PPO 裁剪 | 限制梯度范数 / 让某些样本的代理项进入平坦区 | §3.7、§13.3 |
| 数值等价 / 接口一致 / 行为成功 | 同输入计算相近 / 输入输出含义相同 / 所测任务实际完成 | §19.8–19.10 |
