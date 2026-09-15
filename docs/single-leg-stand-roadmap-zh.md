# Microduck 单腿站立(金鸡独立)训练路线图

> 面向 RL 新手的从 0 到 1 实操指南。
> 目标:让 Microduck 抬起一条腿,用另一条腿站住并保持数秒(**先只要求仿真里能站**)。
>
> 本文里的每一个文件路径、行号、函数名、物理数字都是在本仓库当场核实/实测过的,
> 不是凭印象写的。核实时间:2026-09-04,分支 `develop`。

---

## 目录

- [写在最前:这个任务有多难](#写在最前这个任务有多难)
- [第 0 章 · 术语表](#第-0-章--术语表)
- [第 1 章 · 先跑通、先看见](#第-1-章--先跑通先看见)
- [第 2 章 · 读懂一个任务是怎么拼出来的](#第-2-章--读懂一个任务是怎么拼出来的)
- [第 3 章 · 训练前的物理可行性验证](#第-3-章--训练前的物理可行性验证)
- [第 4 章 · 奖励设计](#第-4-章--奖励设计)
- [第 5 章 · 课程学习](#第-5-章--课程学习)
- [第 6 章 · 代码落地清单](#第-6-章--代码落地清单)
- [第 7 章 · 训练与读日志](#第-7-章--训练与读日志)
- [第 8 章 · 评估](#第-8-章--评估)
- [第 9 章 · 常见失败模式速查表](#第-9-章--常见失败模式速查表)
- [附录](#附录)

---

## 写在最前:这个任务有多难

在动手之前,先看一组我用 MuJoCo 从 `scene.xml` 的 `STAND` 关键帧**实测**出来的数字。
这组数字决定了后面所有的设计选择。

| 量 | 实测值 | 为什么重要 |
|---|---|---|
| 总质量 | **737 g** | |
| 双脚横向间距 | **8.36 cm**(中线各 ±4.18 cm) | 单腿站立要把质心横移 4.18 cm |
| 质心(CoM)高度 | **14.2 cm** | 比躯干原点(12 cm)还高 —— 因为头很重 |
| 单只脚掌尺寸 | **5.4 cm(前后) × 4.12 cm(横向)** | 单脚支撑区横向只有 **±2.06 cm** |
| 头部质量占比 | `jaw_soft` 188.8 g + `neck` 36.8 g + `yaw_roll_motion` 48.6 g = **37%** | 一根杆子上顶着 1/3 的体重 |
| 单腿自由度 | hip_yaw / hip_roll / hip_pitch / knee / **ankle(只有俯仰)** | **没有踝关节横滚** |
| `hip_roll` 硬限位 | **±0.384 rad = ±22°** | 横向平衡的唯一主动关节 |

### 由此得到的四条硬结论

**1. 静态倾覆裕度极小。**

```
倾覆角 = atan(脚掌横向半宽 / 质心高度) = atan(0.0206 / 0.1417) ≈ 8.3°
```

质心横向偏出支撑脚约 2 cm,机器人就翻。人类单腿站立的裕度大概是这个的两倍以上
(脚更长更宽、质心相对更低)。

**2. 没有 ankle roll(踝关节横滚),这是最关键的一条。**

人类单腿站立主要靠**踝关节**微调:身体往左倒,踝关节就往左压地,产生一个把你推
回来的力矩。Microduck 每条腿只有 5 个关节,`ankle` 只有俯仰一个自由度,
**支撑脚在冠状面上完全无法主动产生恢复力矩**。

横向平衡只能靠 `hip_roll`(±22°)把躯干整体压过去,加上头部和摆动腿的惯性反作用。
在真实机器人领域这叫 **hip strategy(髋策略)**,比 ankle strategy 反应更慢、
更容易过冲。**这就是为什么这个任务不是"调调奖励就能成"。**

**3. 光靠 hip_roll 不够 —— 必须全身协同。**

我做了实测(双脚都在地上,缓慢把 `hip_roll` 推到极限):

| `hip_roll` 偏置 | 右脚承重比例 |
|---|---|
| +20°(压向左) | 34% |
| 0°(HOME) | 54% |
| −20°(压向右) | **65%(上限)** |

**把 `hip_roll` 用到极限,也只能把 65% 的体重转移到一只脚上。** 单腿站立需要 100%。
所以光靠"侧倾躯干"这一招是不够的。

**4. 但是 —— 全身协同下,静平衡解确实存在。**

我在全部 14 个关节上(严格遵守硬限位)做了全局优化,判据是严格的
**"质心投影落在支撑脚接触面凸包内"**:

| 允许活动的关节 | 最大平衡裕度 | 结论 |
|---|---|---|
| 全部 14 个(软限位,即训练实际范围) | **+11.5 mm** | ✅ 可行 |
| 只有两条腿(头颈锁死在 HOME) | **+4.25 mm** | ✅ 勉强可行 |
| 只有 `hip_roll` / `hip_yaw` / 摆动腿 | **−2.0 mm** | ❌ 不可行 |

**三个可操作的推论:**

- **任务是可行的**,但裕度只有约 1 cm —— 大约是一枚硬币的宽度。
- **头部必须参与平衡。** 把头颈锁死会让裕度**减半**(11.5 → 4.25 mm)。
  所以在奖励里**不要给 `head_pose` 跟踪项太高的权重、也不要用太紧的 std** ——
  那等于没收了策略一半的平衡能力。(AGENTS.md 记录过一次同类事故:
  一个过紧的头部跟踪 std 把走路任务压到策略干脆站着不动。)
- **最优姿态会顶到多个关节限位**(`hip_yaw`、`hip_roll` 都在极限)。
  这意味着 `joint_pos_limit_proximity` 惩罚要**温和**(margin 别设太大),
  否则你会把唯一可行的解也惩罚掉。

> **这些结论的适用范围**:以上是**静力学**判据(质心投影 + 支撑多边形),
> 用正运动学把支撑脚摆平在地面上算出来的。它**没有**验证:
> (a) 维持这些姿态所需的关节力矩是否在舵机的 ±0.96 Nm 之内;
> (b) 这些平衡点是否**动态可镇定**。
> 但它给出的是必要条件 —— 如果连静平衡解都不存在,那就不用往下做了。

### 复算这些数字的脚本

```bash
uv run python -c "
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path('src/mjlab_microduck/robot/microduck/scene.xml')
d = mujoco.MjData(m)
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(m.nkey)]
mujoco.mj_resetDataKeyframe(m, d, names.index('STAND')); mujoco.mj_forward(m, d)

def site(n):
    return d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)].copy()

lf, rf = site('left_foot'), site('right_foot')
print('双脚间距        ', np.round(np.abs(lf - rf), 4))
print('总质量 (kg)     ', round(float(m.body_mass.sum()), 4))
mujoco.mj_comPos(m, d)
print('质心 CoM        ', np.round(np.array(d.subtree_com[0]), 4))

gi = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'left_foot_collision')
mi = m.geom_dataid[gi]
V = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]]
W = V @ d.geom_xmat[gi].reshape(3, 3).T + d.geom_xpos[gi]
print('脚掌尺寸 (x,y,z)', np.round(W.max(0) - W.min(0), 4))
"
```

---

## 第 0 章 · 术语表

新手最大的障碍不是数学,是**听不懂话**。这一章把你后面会反复撞到的词一次讲完。
每个词都配一个 Microduck 的具体例子。

### 0.1 强化学习基础

| 术语 | 一句话 | 在 Microduck 里具体是什么 |
|---|---|---|
| **策略 policy** | 一个神经网络:看到状态 → 输出动作 | 输入 61 个数,输出 14 个数。就是最后导出的那个 `.onnx` |
| **观测 observation** | 策略每一步"看到"的东西 | 61 维向量:48 维本体感受 + 13 维命令。**不是**上帝视角,只有机器人自己能测到的量 |
| **动作 action** | 策略每一步"做"的事 | 14 个舵机的**目标角度**(不是力矩)。舵机自己有 PD 控制器去追这个目标 |
| **奖励 reward** | 每一步给策略打的分 | 一堆"奖励项"加权求和,比如"站得直 +2 分""抖动 -0.1 分" |
| **回合 episode** | 一次从重置到结束的完整尝试 | 本仓库通常 5–12 秒。到时间了就重置,或者摔了就重置 |
| **PPO** | 目前最常用的 RL 算法之一 | 本仓库固定用它(`rsl_rl` 库)。你基本不用改它的超参 |
| **Actor / Critic** | Actor = 策略本身;Critic = 一个只在训练时用的"打分员" | Critic 可以看到**作弊信息**(比如真实速度),Actor 不能。导出 ONNX 时只导出 Actor |
| **熵 entropy** | 策略的"随机程度",越高探索越多 | 日志里的 `Mean noise std`。训练初期大(乱试),后期自动变小(定型) |
| **优势 advantage** | "这个动作比平均水平好多少" | PPO 真正用来更新的量。**关键推论:PPO 看的是相对值,不是绝对值** —— 所以奖励项的**相对大小**比绝对大小重要得多 |

> **新手最容易误解的一点**:奖励不是"目标",是"梯度的来源"。
> 你写 `站得直 +2 分`,策略学到的不是"我要站直",而是"往站直的方向挪一点点会涨分"。
> 所以奖励函数必须**处处有梯度**,一个只在"完全成功时才给分"的奖励等于没写。

### 0.2 本仓库的计量单位(读日志和写课程表必须搞清楚)

| 术语 | 值 / 含义 |
|---|---|
| **并行环境 `num_envs`** | 同时仿真多少个机器人。本仓库典型值 4096。它们共享一个策略,只是初始状态不同 |
| **`NUM_STEPS_PER_ENV`** | **24**(定义在 `microduck_velocity_env_cfg.py:22`)。每次策略更新前,每个环境往前推 24 步 |
| **迭代 iteration** | 一次"收集数据 + 更新网络"的循环。日志里每行就是一次迭代 |
| **环境步数 env step** | **= 迭代 × 24**。⚠️ **所有课程表(curriculum)用的都是这个刻度,不是迭代数** |
| **控制频率** | 50 Hz(每步 20 ms) |
| **`episode_length_s`** | 一个回合几秒。StandUp 是 6.0 秒 = 300 步 |

**换算例子**:你想让某个奖励在第 600 次迭代生效 → 课程表里要写 `"step": 600*24 = 14400`。

**时间预算换算**(基于你这台 RTX 5070 Laptop 的实测 ~2.6 秒/迭代 @ 4096 环境):

| 迭代数 | 大约耗时 |
|---|---|
| 5(冒烟测试,64 环境) | < 1 分钟 |
| 1000 | ~43 分钟 |
| 3000 | ~2.2 小时 |
| 15000 | ~11 小时(你上次 StandUp 就是这个) |

### 0.3 mjlab 的六个 Manager —— 读懂任何配置文件的钥匙

mjlab 把一个 RL 环境拆成六个"管理器",每个管理器就是配置文件里的一个字典。
**看懂这六个字典 = 看懂这个任务**。

| Manager | 配置里的字段 | 干什么 | 单腿站立里的例子 |
|---|---|---|---|
| **Observation** | `cfg.observations` | 定义策略能看到什么 | 关节角、角速度、投影重力、上一步动作…… |
| **Action** | `cfg.actions` | 定义策略输出怎么变成控制信号 | 14 维 → 14 个舵机目标角 |
| **Reward** | `cfg.rewards` | 每一项奖励 = 一个函数 × 一个权重 | `{"单支撑": 函数 × 3.0, "抖动": 函数 × -0.1}` |
| **Termination** | `cfg.terminations` | 什么时候提前结束回合 | 摔了、出界了、数值 NaN 了 |
| **Event** | `cfg.events` | 在特定时机改变仿真 | **reset 时**摆姿势;**startup 时**随机化质量;**interval**随机推一把 |
| **Command** | `cfg.commands` | 给策略下的"指令",会进观测 | 走路任务是速度指令;站立任务是头部姿态指令 |
| **Curriculum** | `cfg.curriculum` | 随训练进度改上面任何东西 | "第 600 迭代后把单支撑奖励权重从 0 拉到 3.0" |

每一项都长这样:

```python
cfg.rewards["某个名字"] = RewardTermCfg(
    func=某个函数,        # 一个 (env, **params) -> Tensor[num_envs] 的函数
    weight=2.0,           # 权重
    params={"std": 0.02}, # 传给函数的额外参数
)
```

日志里 `Episode_Reward/某个名字` 打印的就是 **`func` 的输出 × `weight`**(加权后的值)。

### 0.4 本项目特有的名词

| 术语 | 解释 |
|---|---|
| **BAM 执行器** | 本仓库不用理想 PD 电机模型,而是用 [BAM](https://github.com/Rhoban/bam) 模拟 XL330 舵机的**电压控制律**(反电动势、库仑/Stribeck 摩擦、负载压降)。因为在这个尺度上,舵机的非理想性就是 sim2real 差距的大头 |
| **域随机化 DR (Domain Randomization)** | 每个并行环境用略微不同的物理参数(质量、摩擦、质心、电池电压、延迟)。目的是让策略学会"对参数不敏感",这样换到真机上也能用 |
| **齿隙 backlash** | 齿轮间隙(每个舵机 ±1°)。仓库为每个任务都做了一个 `-Backlash-` 孪生任务 |
| **`passive_*` 前缀** | 所有**不受驱动**的关节(轮子、齿隙铰链)都以此命名。所有选择器都用 `^(?!passive_).*` 排除它们 |
| **61 维观测合同** | 所有策略共用同一个观测布局,这样真机运行时可以在走路/站立/翻滚策略之间**热切换**。布局:48 维本体感受 + 13 维命令 `[twist(3), head_pose(4), body_pose(6)]`。**用不到的命令槽要补零,不能删** |
| **投影重力 projected gravity** | 重力向量在机器人身体坐标系里的分量。3 个数就完整表达了"我现在歪了多少、往哪歪"。站得笔直时是 `(0, 0, -1)` |
| **接触传感器 / air time** | 检测脚有没有踩地。`air_time` = 这只脚已经腾空多久 |
| **质心 CoM / 支撑多边形** | 静态平衡的充要条件:**质心的垂直投影落在支撑多边形内**。双脚站立时支撑多边形是两只脚围出的区域;单脚站立时就只剩一只脚掌那么大 |
| **势能式塑形 potential-based shaping** | 一种**不可被刷分**的奖励写法:只奖励"进步量 Δ",不奖励"当前状态"。上升给钱、维持给零、下降扣钱。理论保证不改变最优策略 |
| **ONNX** | 一种跨框架的模型格式。真机上跑的就是它。`scripts/export.py` 会把观测归一化器一起烤进去 |

---

## 第 1 章 · 先跑通、先看见

**目标**:在写任何新代码之前,把"训练 → 回放 → 读日志"这条链路在你机器上跑顺,
建立直觉。**全部用已有任务,零改动。**

**预计耗时**:半天。

### 1.1 冒烟测试(Smoke Test)

```bash
uv run train Mjlab-StandUp-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
```

64 个环境、跑 5 次迭代就停。**AGENTS.md 说这一条能用几分钱抓住约 95% 的配置错误。**
以后每次改完配置都先跑它,再跑正式训练。

它验证的是:环境能构建、能推进不出 NaN、观测维度对、每个奖励项都能算出来。

### 1.2 回放你已有的策略

你的 `logs/` 里已经有一个训练了 15000 次迭代的 StandUp 策略。先看看它长什么样:

```bash
uv run play Mjlab-StandUp-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/microduck_stand/2026-09-01_18-34-30_microduck_stand/model_14999.pt \
  --num-envs 1
```

> **注意**:README 里写的是 `--wandb-run-path`,那需要配置 wandb 账号。
> 你没配 wandb,**用 `--checkpoint-file` 直接读本地文件即可**,功能完全一样。
> `scripts/export.py` 同样支持 `--checkpoint-file`。

试着调不同的 checkpoint(`model_2750.pt`、`model_7500.pt`、`model_14999.pt`),
**亲眼看到策略是怎么一步步变好的** —— 这个直觉在你后面判断"我的训练是失败了
还是只是还没到时候"时非常重要。

### 1.3 逐行读懂训练日志

训练时每次迭代会打印一屏。挑重要的看:

| 字段 | 含义 | 怎么判断好坏 |
|---|---|---|
| `Mean reward` | 每回合总奖励的平均 | **应该持续上升**。但注意它可能靠正则项虚涨(见下) |
| `Mean episode length` | 平均回合长度(步) | 取决于任务。有摔倒终止的任务里,它上升 = 摔得少了 |
| `Episode_Reward/<项名>` | **该项加权后**的回合累计值 | ⚠️ 这是**乘过权重的**。权重为 0 的项恒显示 0,不代表行为没发生 |
| `Value function loss` | Critic 的预测误差 | 前期大很正常,应逐渐下降并稳定 |
| `Surrogate loss` | PPO 的策略损失 | 在 0 附近小幅波动是正常的 |
| `Mean noise std` | 策略的探索噪声大小 | 从 ~1.0 开始自动衰减。**如果它掉得太快(几百迭代就到 0.1),说明策略过早定型了** |

**最重要的看法**(AGENTS.md 反复强调):

> 只看 `Mean reward` 是不够的。**必须单独盯住"主任务奖励项"在不在涨。**
> 总奖励完全可以靠"少抖动"这类正则项一路上涨,而机器人从头到尾没做过那个动作。

以及:

> **每一个惩罚项的 `Episode_Reward/` 都必须 ≤ 0。**
> 有任何一个是正的,说明你的权重符号写反了(详见 §2.3)。

### 1.4 看历史曲线

```bash
uv run --with tensorboard tensorboard --logdir logs/rsl_rl
```

打开 <http://localhost:6006>,把你那次 15000 迭代的 StandUp 当参照物。
后面训练单腿站立时,曲线形状可以跟它对比。

---

## 第 2 章 · 读懂一个任务是怎么拼出来的

**教材**:`src/mjlab_microduck/tasks/microduck_standup_env_cfg.py`(1210 行)。
选它是因为它是"回合式姿态任务"的标准模板 —— 你的单腿站立本质上就是这一类。

**预计耗时**:一天。目标不是背下来,而是**知道每一段在干什么、要改的时候去哪改**。

### 2.1 文件解剖图

| 行号 | 内容 | 你要做的事 |
|---|---|---|
| 1–19 | 模块 docstring:讲清任务定义、初始状态、目标、奖励哲学 | **认真读**。这是作者留给你的设计说明 |
| 26–38 | `ENABLE_*` 开关(对称性、各类域随机化) | 知道有哪些开关 |
| 40–58 | DR 数值范围 + `EPISODE_LENGTH_S = 6.0` | |
| 60–95 | 任务常量:`SITTING_JOINT_OVERRIDES`、`_LEG_JOINTS`(:82)、`_NECK_JOINTS`(:83)、`SIT_Z = 0.060`(:89)、`STAND_Z = 0.115`(:95) | **重点**:这是"调参面板" |
| 97–120 | 身体姿态命令的常量 | |
| 122–152 | play 时的 spawn 覆盖(见 §5.5) | |
| 155–183 | import(故意放在常量后面) | |
| 186–226 | 工厂函数开头:两个接触传感器 + `cfg = make_velocity_env_cfg()` | **重点** |
| 228–232 | Actions | |
| 233–245 | Rewards:先 `del` 掉走路专用项 | **重点**:这个"先继承再删改"的套路 |
| 246–512 | Rewards:堆任务奖励项 | **重点中的重点** |
| 513–559 | Rewards:正则项(与 velocity 对齐) | |
| 560–641 | Observations:61 维布局、噪声、延迟、IMU 失准、编码器偏差 | 单腿任务基本原样抄 |
| 642–704 | Commands:`head_pose` / `body_pose` / 被"阉割"但保留的 `twist` | **重点**:零填充的技巧 |
| 705–714 | Terminations:只有 `nan_state` | |
| 715–869 | Events:`expand_bam_friction_fields`、`set_random_ground_state`、各类 DR | **重点**:初始姿态在这里定 |
| 870–881 | Terrain | |
| 882–1168 | Curriculum:阶段化权重表和范围表 | **重点** |
| 1173–1210 | `MicroduckStandUpRlCfg`:PPO 超参 | 基本照抄,只改名字 |

### 2.2 "先继承再删改" —— 本仓库的核心套路

几乎所有任务都是这么写的:

```python
cfg = make_velocity_env_cfg()        # 1. 拿 mjlab 的走路模板(它自带一整套东西)

del cfg.rewards["track_linear_velocity"]   # 2. 删掉走路专用的
del cfg.rewards["air_time"]
# ...

cfg.rewards["height_stand"] = RewardTermCfg(...)   # 3. 加自己任务的
```

**为什么不从零开始?** 因为 `make_velocity_env_cfg()` 顺手给了你:
观测组的完整定义、噪声模型、终止条件、地形、以及一堆你想不到但必须有的东西。
从零写你会漏掉一半。

本仓库有两种建法:

- **派生式**:`cfg = make_microduck_velocity_env_cfg(...)` 然后改。
  好处是域随机化/观测噪声/延迟**自动保持同步**。
  只有 `microduck_velstand_env_cfg.py` 和轮滑家族这么干。
- **独立式**:`cfg = make_velocity_env_cfg()`(mjlab 的基础模板)然后自己把
  整套 DR + 观测噪声 + NaN 守卫**手工移植一遍**。
  standup / sitstand / ground_pick / ball_kick / roulade / spin 都是这样。

> **对你的建议**:直接**整体复制 `microduck_standup_env_cfg.py`** 当骨架。
> 它虽然是独立式的,但那套移植工作**已经做完了**,你复制过来就白拿。
> 从零走独立式路线的话,AGENTS.md 列了你必须自己补的清单:
> `_safe` 后缀的 critic 观测项、带 `sensor_names` 的 `nan_state` 终止、
> `expand_bam_friction_fields` 事件、编码器偏差、IMU 失准。

### 2.3 ⚠️ 奖励符号约定 —— 本仓库最容易翻车的一条

`mdp.py` 里有**两种风格**的惩罚函数:

| 风格 | 函数返回值 | 该配的权重 | 例子 |
|---|---|---|---|
| **mjlab 自带的 cost 函数** | ≥ 0(是"代价") | **负**权重 | `mdp.action_rate_l2` → `weight=-0.1` |
| **microduck 自己写的 `*_penalty` / `*_l1`** | ≤ 0(**已经自带负号**) | **正**权重 | `microduck_mdp.height_l1_penalty` → `weight=7.5` |

**给一个自带负号的惩罚函数配负权重,就是双重否定 → 变成了奖励违规行为。**
策略一定会去刷它(历史上出现过"屁股蹦跳""故意坐着摔"这类行为)。

配置文件 `microduck_standup_env_cfg.py:401-406` 有一段注释专门警告这件事。

**万无一失的检验方法**:

> 每次训练,wandb / tensorboard 里每一个 `Episode_Reward/<惩罚项名>` 都必须 ≤ 0。
> 有正的就是符号写反了。

### 2.4 14 个舵机的关节表(从 MJCF 提取)

| 索引 | 关节名 | 硬限位(rad) | 硬限位(度) |
|---|---|---|---|
| 0 | `left_hip_yaw` | −0.4363 … 0.5236 | −25° … +30° |
| 1 | `left_hip_roll` | ±0.3840 | **±22°** |
| 2 | `left_hip_pitch` | ±1.5708 | ±90° |
| 3 | `left_knee` | ±1.5708 | ±90° |
| 4 | `left_ankle` | ±1.5708 | ±90° |
| 5 | `neck_pitch` | −1.5708 … 1.0472 | −90° … +60° |
| 6 | `head_pitch` | ±1.5708 | ±90° |
| 7 | `head_yaw` | ±2.9671 | ±170° |
| 8 | `head_roll` | ±0.4363 | ±25° |
| 9 | `right_hip_yaw` | −0.5236 … 0.4363 | −30° … +25° |
| 10 | `right_hip_roll` | ±0.3840 | **±22°** |
| 11 | `right_hip_pitch` | ±1.5708 | ±90° |
| 12 | `right_knee` | ±1.5708 | ±90° |
| 13 | `right_ankle` | ±1.5708 | ±90° |

所以每个配置文件里都有这一行:

```python
_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]   # 十个腿关节
_NECK_JOINTS = [5, 6, 7, 8]                          # 四个头颈关节
```

**HOME 姿态**(`microduck_constants.py` 的 `HOME_FRAME`,即站立默认姿态)注意
**左右是镜像的**(符号相反):

```python
r".*left_hip_roll.*":  -0.0873    r".*right_hip_roll.*":   0.0873
r".*left_hip_pitch.*": -0.4579    r".*right_hip_pitch.*":  0.4579
r".*left_knee.*":      -0.0049    r".*right_knee.*":       0.0049
r".*left_ankle.*":      0.4530    r".*right_ankle.*":     -0.4530
```

### 2.5 ⚠️ 绝不要硬编码关节索引

上面那张表**只在 `walk` 和 `groundcontact` 模型上成立**。在轮滑模型上,
被动轮关节 `passive_*wheel` 会**插进来**,右腿就从索引 9 变成 11。

`mdp.py` 提供了两个助手来解决这个问题:

```python
_servo_joint_ids(env, asset)      # mdp.py:126 — 返回所有非 passive_ 关节的索引
_servo_joint_pos(env, asset)      # mdp.py:146 — 直接返回这些关节的角度
```

**规则**:mdp 函数里所有的 `joint_indices` / `target_overrides` 参数,
都是按"14 个舵机的标准布局"写的,函数内部通过这两个助手转换成模型的真实索引。
你写新奖励函数时必须遵守同样的约定。


---

## 第 3 章 · 训练前的物理可行性验证

AGENTS.md 把这一步称为**"单个最大的时间节省者"**。对单腿站立尤其如此 ——
第 0 章开头那些数字就是这么来的,它们直接改变了后面奖励设计的多个决定。

**核心原则**:在花几个小时训练之前,先花二十分钟确认**你要的动作在物理上做得到**。

### 3.1 ⚠️ 教科书式的"保持测试"在这个机器人上不成立

AGENTS.md 建议的做法是:"把目标姿态当 ctrl 目标保持 3 秒,从带噪声的初始状态出发,
检查倾斜角"。

**我实测发现,这个机器人连已知正确的双腿 `STAND` 姿态都撑不过 1 秒:**

```
[STAND] 开环保持 3 秒:
    t= 0.0s  z= 120.0mm  倾斜=  0.00°
    t= 0.5s  z= 112.2mm  倾斜= 15.08°
    t= 1.0s  z=  51.1mm  倾斜= 83.50°   ← 已经摔了
    t= 3.0s  z=  37.2mm  倾斜= 81.30°
```

**原因**:舵机 `kp = 0.55 Nm/rad`,力矩上限 `±0.96 Nm`(见
`joints_properties.xml` 的 `chosen_actuator` 类)。这个刚度太低了 ——
关节偏差 0.1 rad 只产生 0.055 Nm 的恢复力矩,而整机重力是 7.23 N。

**这正是 `ctrlrange` 被设成 ±10 rad 的原因**:策略必须**大幅超调指令**
(命令一个远超实际可达的角度)才能榨出力矩。AGENTS.md 里那句
"wide ctrlrange is intentional — low-kp servos need overshoot" 说的就是这件事。

**推论**:
1. 这个机器人**没有任何开环稳定的姿态**。所有站立都是**主动反馈平衡**的结果 ——
   这正是 RL 要学的东西。
2. 不要用"保持 ctrl 3 秒"来判断姿态可行性。**要用静力学判据**(下一节)。
3. 如果你想做保持测试,得先把 `kp` 临时调高。我用 `kp=30` 做过基线验证:
   双腿 STAND 能稳住(z=116.7 mm,倾斜 0.03°),维持它只需要 **0.068 Nm** 峰值力矩,
   远在 0.96 Nm 之内 —— 说明**姿态本身没问题,问题在于低刚度下需要反馈**。

### 3.2 正确的判据:质心投影 vs 支撑多边形

静态平衡的充要条件:

> **质心的垂直投影,落在支撑多边形内。**

单腿站立时,支撑多边形就是那一只脚的接触面。所以要算的是:

1. 把支撑脚**刚性地摆平**在地面上(用正运动学,不跑动力学)
2. 算出脚掌接触点的**凸包**
3. 算质心投影到这个凸包的**有符号距离**(在内为正 = 裕度)
4. 在关节限位内**最大化**这个裕度

我按这个流程做了全局优化(`scipy.differential_evolution`,14 维,严格遵守限位),
结果就是第 0 章的那张表。**关键数字重复一遍:**

| 允许活动的关节 | 最大平衡裕度 |
|---|---|
| 全部 14 个(软限位 0.9,即训练实际范围) | **+11.5 mm** ✅ |
| 只有两条腿(头颈锁死) | **+4.25 mm** ✅ |
| 只有 `hip_roll` / `hip_yaw` / 摆动腿 | **−2.0 mm** ❌ |
| (参考)双腿 STAND 相对单只右脚 | −30.2 mm |

### 3.3 复现脚本

```python
# 保存为 check_balance.py,用 `uv run python check_balance.py` 跑
import mujoco, numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
from matplotlib.path import Path

m = mujoco.MjModel.from_xml_path("src/mjlab_microduck/robot/microduck/scene.xml")
d = mujoco.MjData(m)
KEYS = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(m.nkey)]
STAND = KEYS.index("STAND")

NAMES = ["left_hip_yaw","left_hip_roll","left_hip_pitch","left_knee","left_ankle",
         "neck_pitch","head_pitch","head_yaw","head_roll",
         "right_hip_yaw","right_hip_roll","right_hip_pitch","right_knee","right_ankle"]
JID = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in NAMES]
QA  = [m.jnt_qposadr[j] for j in JID]
LIM = np.array([m.jnt_range[j] for j in JID])
mid = LIM.mean(1, keepdims=True); half = (LIM[:,1]-LIM[:,0])[:,None]/2
SLIM = np.hstack([mid - half*0.9, mid + half*0.9])   # soft_joint_pos_limit_factor

GR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")  # 支撑脚
GL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")   # 摆动脚
def mesh(g):
    i = m.geom_dataid[g]; a = m.mesh_vertadr[i]
    return m.mesh_vert[a:a + m.mesh_vertnum[i]]
MR, ML = mesh(GR), mesh(GL)

mujoco.mj_resetDataKeyframe(m, d, STAND); mujoco.mj_forward(m, d)
R0 = d.geom_xmat[GR].reshape(3,3)
NAX  = int(np.argmax([abs(R0[:,i] @ [0,0,1.]) for i in range(3)]))   # 脚掌法线是局部哪个轴
NSGN = np.sign(R0[:,NAX] @ [0,0,1.])

def rot_align(a, b):                       # 把向量 a 转到 b 的旋转矩阵
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9: return np.eye(3)
    vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)

def seg_dist(p, a, b):                     # 点到线段的距离(注意不是到直线!)
    ab = b - a; t = np.clip(((p-a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t*ab))

def evaluate(q):
    """返回 (平衡裕度 m, 摆动脚离地 m, 质心高 m)"""
    mujoco.mj_resetDataKeyframe(m, d, STAND)
    for a, v in zip(QA, q): d.qpos[a] = v
    mujoco.mj_forward(m, d); mujoco.mj_comPos(m, d)
    com = np.array(d.subtree_com[0])
    Rg, pg = d.geom_xmat[GR].reshape(3,3), d.geom_xpos[GR]
    Rc = rot_align(NSGN * Rg[:,NAX], np.array([0,0,1.]))     # 把支撑脚摆平
    sole = ((MR @ Rg.T + pg) - pg) @ Rc.T
    comF = Rc @ (com - pg)
    lsole = ((ML @ d.geom_xmat[GL].reshape(3,3).T + d.geom_xpos[GL]) - pg) @ Rc.T
    contact = sole[sole[:,2] < sole[:,2].min() + 0.002][:, :2]   # 最低 2mm = 接触面
    if len(contact) < 3: return -1., 0., 0.
    poly = contact[ConvexHull(contact).vertices]
    p = comF[:2]
    dmin = min(seg_dist(p, poly[i], poly[(i+1) % len(poly)]) for i in range(len(poly)))
    margin = dmin if Path(poly).contains_point(p) else -dmin
    return margin, lsole[:,2].min() - sole[:,2].min(), comF[2]

def cost(q):
    margin, clearance, comz = evaluate(q)
    pen = 0.
    if clearance < 0.02: pen += 10 * (0.02 - clearance)   # 摆动脚至少离地 2cm
    if comz < 0.10:      pen += 10 * (0.10 - comz)        # 别蹲成一团
    return -margin + pen

r = differential_evolution(cost, list(map(tuple, SLIM)), seed=0,
                           maxiter=400, popsize=25, tol=1e-8, polish=True)
margin, clearance, comz = evaluate(r.x)
print(f"最大平衡裕度 = {margin*1000:+.2f} mm  摆动脚离地 {clearance*1000:.1f} mm  质心高 {comz*1000:.1f} mm")
print("姿态:", ", ".join(f"{n}={np.degrees(v):.0f}°" for n, v in zip(NAMES, r.x)))
```

**两个容易写错的地方**(我都踩过):

1. **点到多边形的距离必须用"到线段",不能用"到直线"。** 对多边形外的点,
   某条边的**延长线**可能离它很近,导致裕度被严重低估(我一开始算出双腿 STAND
   相对单脚的裕度是 −1.5 mm,实际是 −30.2 mm)。
2. **不能用轴对齐包围盒代替凸包。** 脚掌被 `hip_yaw` 转过角度之后,
   它的轴对齐包围盒会明显大于真实接触面,判据会过于宽松。

### 3.4 这一步还该量什么

**量出单腿站立时真实的躯干高度。** 别直接抄 `STAND_Z = 0.115`(那是双腿值)。
上面脚本里的 `comz` 和躯干 z 都能读出来。AGENTS.md 记录过:
"5 毫米不对的 `STAND_Z` 把目标变成不可能任务,浪费了好几天。"

**检查力矩是否够。** 把上面找到的姿态用高增益 PD 稳住,读 `d.actuator_force`,
跟真实的 ±0.96 Nm 比。我验证过双腿 STAND 只需 0.068 Nm,单腿姿态**我没有单独验证** ——
这是留给你的一步。

### 3.5 如果结论是"不可行"怎么办

本任务的结论是**可行**,所以你可以直接往下做。但万一你改了目标(比如要求
保持 10 秒、或者站在斜坡上)导致不可行,退路有两条:

- **放宽成动态平衡**:允许支撑脚小幅踏步/滑动。这在真实机器人上也是标准做法 ——
  很多"单腿站立"其实是持续的微小调整,而不是绝对静止。
- **缩短保持时长目标**:从 3 秒降到 1 秒。有总比没有强,而且短时的成功
  能给逆向课程(§5.6)提供出生状态。

---

## 第 4 章 · 奖励设计

这一章是整个任务里**最难、最需要经验**的部分。好消息是:`mdp.py` 里已经有 220 个
写好的函数,你要做的大部分是**挑选和组装**,而不是从头写。

### 4.1 先做目标分解

把模糊的"单腿站立"拆成**可测量的量**。这是奖励设计的第一步,也是最关键的一步。

| # | 要什么 | 现成函数 | 位置 | 备注 |
|---|---|---|---|---|
| 1 | 支撑脚踩实 | `single_foot_grounded_reward(env, sensor_name)` | `mdp.py:5809` | 二值:该脚触地就给 1 |
| 2 | 摆动脚离地 | 同一函数配**负**权重(用另一个传感器) | `mdp.py:5809` | |
| 3 | **质心投影落在支撑脚上** | `com_over_support_foot` | `mdp.py:2743` | ⚠️ 见 §4.4 |
| 4 | 躯干竖直 | `body_upright_linear` + `body_upright_gaussian` | `mdp.py:683` / `:710` | 两个要一起用,见下 |
| 5 | 高度维持 | `height_target_gaussian` + `height_l1_penalty` | `mdp.py:2440` / `:2454` | |
| 6 | 摆动腿摆成指定姿态 | `pose_target_match` + `pose_l1_penalty` | `mdp.py:2396` / `:2421` | 配 `target_overrides` |
| 7 | 综合"站好了" | `standing_composite_score` / `standing_success_bonus` | `mdp.py:810` / `:856` | 见 §4.5 |
| 8 | **不可刷分的进步塑形** | `upright_progress` / `height_progress` | `mdp.py:546` / `:575` | 见 §4.6 |

**为什么第 4 项要两个函数一起用?**

- `body_upright_linear` 返回 `cos(倾斜角)`,在整个范围内都有梯度,但**在完全竖直
  处梯度为零**(cos 在 0 附近是平的)—— 它能把你从躺着拉到大致竖直,但最后几度
  没有动力。
- `body_upright_gaussian` 是 `exp(-倾斜²/std²)`,在竖直附近**特别陡**,负责最后
  那几度。

这是本仓库的通用套路:**一个宽的项负责"从远处拉过来",一个窄的项负责"最后一公里"**。
高度也是一样(`height_stand` std=0.04 配 `height_stand_sharp` std=0.015)。

### 4.2 防作弊的辅助项

RL 会**照着字面意思**优化你的奖励。每一个你没约束住的自由度都会被利用。
下面每一项都是针对一种具体的作弊方式:

| 会出现的作弊 | 用什么挡 | 位置 |
|---|---|---|
| 关节顶在硬限位上白嫖(低 kp 舵机很爱这么干) | `joint_pos_limit_proximity(margin=0.15)` | `mdp.py:1989` |
| 用膝盖 / 下巴当"第三只脚" | 新建 `robot_ground_contact` 传感器 + `body_impact_cost` | `mdp.py:1644`;传感器抄 `microduck_roulade_env_cfg.py:190` |
| 高频抖脚来刷"单支撑"计数 | `contact_frequency_penalty` | `mdp.py:1761` |
| 两脚并到一起,退化成双支撑 | `feet_distance_penalty` | `mdp.py:5571` |
| 支撑脚踮脚尖 / 侧翻 | `feet_flat_penalty`(支持按脚接触门控) | `mdp.py:1463` |
| 整体抖动 | `mdp.action_rate_l2`、`joint_torque_rate_l2` | mjlab / `mdp.py:1601` |

> `joint_pos_limit_proximity` 对本任务**特别重要**。第 3 章会看到,最优单腿姿态
> 需要把 `hip_roll` 用到接近 ±22° 的极限。如果不加这个惩罚,策略会直接顶死限位
> (硬限位提供了"免费的支撑力矩"),得到一个在真机上必然过热/抖动的策略。
>
> 注意:**要用 qpos 侧的限位惩罚,不是 ctrl 侧的**。这个机器人的 `ctrlrange` 是
> 故意开到 ±10 rad 的 —— kp 只有 0.55 Nm/rad,策略必须靠"超调指令"才能拿到力矩。
> 惩罚指令值会直接废掉它的控制权威。

### 4.3 接触传感器怎么拿

**MJCF 里根本没有接触传感器。** `sensors.xml` 里总共只有 6 个传感器:
`framequat`/`gyro`×2/`velocimeter`/`accelerometer`(都挂在 `imu` 站点上)
加一个 `subtreeangmom`(躯干子树角动量)。没有任何 `<touch>` 或 `<force>`。
所有接触检测都是在 Python 侧用 `ContactSensorCfg` 声明的。

**方式 A —— 用共享的双脚传感器**(每个任务都有,直接抄
`microduck_standup_env_cfg.py:194-206`):

```python
feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom",
        pattern=r"^(left_foot_collision|right_foot_collision)$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"), reduce="netforce", num_slots=1,
    track_air_time=True,
)
```

**slot 顺序是(左, 右)** —— 这一点在 `microduck_velocity_env_cfg.py` 的注释里
写死了("LEFT foot first, RIGHT foot second"),`gait_symmetry_penalty` 也依赖它。
读法:

```python
s = env.scene.sensors["feet_ground_contact"]
left_down  = s.data.found[:, 0]                # 左脚是否触地
right_down = s.data.found[:, 1]
n_down     = (s.data.current_contact_time > 0).sum(dim=1)   # 几只脚触地
left_air   = s.data.current_air_time[:, 0]     # 左脚已腾空多久
```

**方式 B —— 专用单脚传感器**(BallKick 已经这么干,直接抄
`microduck_ball_kick_env_cfg.py:160-172`):

```python
support_foot_ground_cfg = ContactSensorCfg(
    name="support_foot_ground_contact",
    primary=ContactMatch(mode="geom",
        pattern=rf"^{support_foot}_foot_collision$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",), reduce="netforce", num_slots=1,
)
```

然后注册到 `cfg.scene.sensors = (feet_ground_cfg, support_foot_ground_cfg, self_collision_cfg)`。

**你会用到的名字**(已核实):

| 用途 | 名字 |
|---|---|
| 脚部碰撞几何体 | `left_foot_collision` / `right_foot_collision` |
| 脚部站点(site) | `left_foot` / `right_foot`(挂在 `ankle_left` / `ankle_right` 上) |
| 地形 | body 名 `terrain`,scene 里的几何体名 `floor` |
| 整机子树 | `trunk_base` |
| 头部(下巴)碰撞体 | `jaw_soft` |

### 4.4 ⚠️ `com_over_support_foot` 要改才能用

这个函数是**本任务最关键的一项**,但你**不能直接拿来用**,原因有二:

1. **它目前没有任何任务在使用**(我 grep 过整个 `tasks/` 目录,只有定义没有调用)。
   它是为 BallKick 写的,后来被移除了。所以它没有被现役测试覆盖 —— 你用之前
   要自己验证。
2. **它被踢球相位门控着**:

```python
cmd = env.command_manager.get_command(command_name)
phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0
gate = kick_engagement(phase, windup_end, return_end)
return gate * reward
```

在零命令的站立任务里,`cmd` 全是 ~0,`atan2(0,0)=0` → phase=0 → gate 很可能恒为 0,
**这一项会永远返回 0**,你却看不出任何报错。

**要做的**:在 `mdp.py` 里写一个去门控版本。核心只有 4 行:

```python
asset = env.scene[asset_cfg.name]
com_xy  = asset.data.root_com_pos_w[:, :2]
foot_xy = asset.data.site_pos_w[:, asset_cfg.site_ids[0], :2]
return torch.exp(-((com_xy - foot_xy) ** 2).sum(-1) / std ** 2)
```

调用时 `asset_cfg` 必须带站点:

```python
params={"asset_cfg": SceneEntityCfg("robot", site_names=["right_foot"]), "std": 0.02}
```

**`std` 取多少?** 默认是 0.04。但第 3 章会量出,这个机器人单脚支撑区横向半宽
只有约 ±20 mm,所以 **0.02 更合适**。AGENTS.md 的原则是"std ≈ 你还在乎的误差,
不是最大误差";0.04 意味着偏 4 cm 还有 37% 的分,那已经摔了。

### 4.5 复合项 vs 加和项

`standing_composite_score`(`mdp.py:810`)是**三个高斯相乘**(高度 × 竖直 × 姿态),
而不是相加。区别很关键:

- **相加**:策略可以找到一个"每项拿 80% 分"的折中姿势(比如整体前倾一点,
  高度、竖直、姿态都差一点但都不差太多),然后就赖在那儿。
- **相乘**:任何一项塌了,总分就塌了。逼策略把每一项都做好。

**但有个前提**:std 必须**足够宽,让当前策略拿得到可见的分数**。如果三项都很窄,
乘起来就是个几乎处处为 0 的函数,梯度看不见,什么都学不到。

**实践建议**:早期用宽 std 的加和项建立基本行为,后期再用窄 std 的复合项打磨。
用课程(第 5 章)在两者之间过渡。

### 4.6 势能式塑形 —— 唯一不会被刷的奖励写法

普通奖励("站得越直分越高")的问题:策略可以**待在一个中等好的状态里一直领工资**。

势能式塑形只付**进步量**:

```python
reward_t = φ(s_t) - φ(s_{t-1})       # φ 是"势能",比如 cos(倾斜角)
```

- 变直了 → 正分
- 保持不动 → **零分**(不管当前多直)
- 变歪了 → 负分

一个回合的总和 = `φ(终点) - φ(起点)`,**跟中间怎么走无关**,所以无法通过
"在好状态里磨蹭"来刷分。理论上(Ng et al. 1999)它还保证不改变最优策略。

仓库里现成的两个:

| 函数 | 势能 φ | 位置 |
|---|---|---|
| `upright_progress` | `cos(倾斜角)` | `mdp.py:546` |
| `height_progress` | `min(躯干z, ceiling)` | `mdp.py:575` |

它们内部维护 `env._upright_potential_prev`,并在 `episode_length_buf <= 1` 时重置。

**你可能想为"单腿"再写一个**:φ = 摆动脚离地高度(带上限),这样"抬腿"这个动作
本身有梯度,但"一直举着"不给钱(靠别的项给)。

### 4.7 避坑清单

每一条都对应本仓库历史上真实翻过的车(记录在 AGENTS.md),或者我在代码里
发现的陷阱。

**1. 符号搞反 = 奖励机器人违规。**
检验:每个 `Episode_Reward/<惩罚项>` 都必须 ≤ 0。见 §2.3。

**2. 绝不要把正奖励门控在"坏状态"上。**
反例:"高度低于 X 时,抬腿给分" → 策略会蹲到最低点然后疯狂抬腿刷分。
正确做法是用势能式塑形(§4.6)。

**3. 不要发"头奖"(jackpot)。**
反例:"只要进入单腿状态,每步 +5 分" → 策略会用最暴力的方式(猛地甩腿)
一秒进入,然后躺着收钱 5 秒。
正确做法:跟踪一个**匀速爬升的内部目标** —— 比"进度条"快不给额外的钱,
于是"慢慢来"本身就是最优解。只加个速度上限惩罚是没用的,它积分出来是有界的成本,
打不过无界的收益。

**4. `ENABLE_SYMMETRY` 必须是 `False`。**
PPO 的镜像增广(`symmetry.py`)会把"左腿站"和"右腿站"当成同一个样本喂进去。
单腿站立是**非对称任务**,这会直接毁掉学习。
参考 `microduck_spin_env_cfg.py:24` 的写法,并在测试里断言
`MicroduckSingleLegRlCfg.algorithm.symmetry_cfg is None`。

**5. 正则项要等技能出现之后再加。**
`action_rate`、`joint_torque_rate` 这类"平滑税"在技能探索期全额生效时,
"站着不动"就成了最优解(不动 = 不扣钱)。用课程从 ~0 慢慢加(第 5 章)。

注意区分两类正则项:
- **运动阻断型**(`body_ang_vel`、`angular_momentum`、姿态 std):惩罚的是动态动作
  **物理上必须有**的东西。单腿平衡需要持续的小幅角速度来纠偏,所以这类要给**低权重**。
- **平滑型**(`action_rate`、`joint_torque_rate`):压抖动但不阻止慢的大动作,可以给较高权重 ——
  但仍然要**等技能出现之后**再加。

**6. 命令槽零填充,不能删观测项。**
即使你完全不用 `head_pose` / `body_pose`,也要保留这些观测项,用
`microduck_mdp.zero_command_padding(dim=6)`(`mdp.py:5115`)补零。
理由:61 维观测合同是全策略族共享的,删了就没法热切换。

顺带一条:**一个永远为零的命令输入,它对应的神经元权重永远学不到东西**。
所以本仓库的惯例是每个命令槽从第 0 步就保持一个**很小的非零采样范围**
(即使奖励权重是 0),让这些输入神经元保持"活着",方便后面的课程接管。

**7. ⚠️ `pose_target_match` 在 `mdp.py` 里被定义了两次**(L2059 和 L2396)。
后者覆盖前者,而且**位置参数顺序不同**:

```python
# L2059: (env, asset_cfg, std, joint_indices, target_overrides)
# L2396: (env, target_overrides, asset_cfg, std, joint_indices)   ← 实际生效的
```

**永远用关键字参数调用它。**

**8. 有两个函数硬编码了传感器名** `"feet_ground_contact"`:
`foot_step_penalty_when_standing`(`mdp.py:3892`)和
`recovery_stepping_reward`(`mdp.py:3931`)。用它们就必须保留这个名字的传感器。

**9. 比较"奖励质量",不是"权重数值"。**
PPO 看的是相对优势。同一个 `action_rate` 权重,在一个正奖励总量大 4 倍的任务里,
实际强度只有 1/4。从别的任务抄正则项权重时,要先看两边正奖励的总量。

**10. 别用人类的速度直觉设上限。**
一个 25 cm 高的机器人翻倒时的自然角速度就是 3.5–5.5 rad/s。给角速度设一个
"看起来合理"的上限会直接禁止它做必要的快速纠偏。反暴力的压力应该加在
**冲击**(`|a_z|`)和**抖动**(`action_rate`)上,而不是转速上。

---

## 第 5 章 · 课程学习

### 5.1 为什么需要课程

单腿站立的典型失败是:**策略学会了"抬一下腿"这个开头,但永远学不会"站住"这个结尾**。
原因是它从来没有机会体验"已经单腿站稳"这个状态 —— 每次刚抬腿就摔了,
所以"站稳之后该怎么办"这部分从来没有 on-policy 数据。

课程学习就是人为地控制"任务有多难",让策略按顺序学。

### 5.2 三段式课程建议

| 阶段 | 迭代范围 | 环境步数(× 24) | 干什么 |
|---|---|---|---|
| **1** | 0 → 600 | 0 → 14400 | 双腿站稳。高度 + 竖直度主导,单腿相关项权重 **0** |
| **2** | 600 → 2500 | 14400 → 60000 | 单支撑 + 质心横移的权重从 0 拉到主导;要求的单腿保持时长 0.3 s → 3 s |
| **3** | 2500 起 | 60000 起 | 加平滑正则:`action_rate` 从 -0.1 → -1.0 |

> **注意**:阶段 1 里单腿相关的奖励项**必须已经存在**(只是权重为 0),
> 观测项也必须已经接上。否则那些输入神经元在整个阶段 1 里没有梯度,
> 到阶段 2 才开始学等于浪费了 600 次迭代。

### 5.3 权重课程的机制

用 `microduck_mdp.reward_weight`(`mdp.py:3442`):

```python
cfg.curriculum["single_support_weight"] = CurriculumTermCfg(
    func=microduck_mdp.reward_weight,
    params={
        "reward_name": "single_support",
        "weight_stages": [
            {"step":     0, "weight": 0.0},
            {"step": 14400, "weight": 1.0},
            {"step": 24000, "weight": 2.0},
            {"step": 36000, "weight": 3.0},
        ],
    },
)
```

⚠️ **`reward_weight` 是阶跃函数,不是线性插值。** 想要"斜坡"就得手动切成好几段。
抄 `microduck_standup_env_cfg.py:882-1166`,那里有一堆现成例子。

事件参数和命令范围有各自的课程函数:

| 函数 | 位置 | 改什么 |
|---|---|---|
| `event_param_curriculum` | `mdp.py:4453` | 任意事件的任意参数 |
| `pose_command_range_curriculum` | `mdp.py:5512` | 姿态命令的采样范围 |
| `push_curriculum` | `mdp.py:3378` | 推力大小 |
| `com_range_curriculum` | `mdp.py:3464` | 质心随机化范围 |
| `termination_param_curriculum` | `mdp.py:5370` | 终止条件的阈值 |

### 5.4 ⚠️ 必须通过 manager 改,不能改 cfg

```python
# ✅ 对
env.reward_manager.get_term_cfg("single_support").weight = 3.0
env.event_manager.get_term_cfg("set_ground_state").params["standing_prob"] = 0.5

# ❌ 错(静默无效,不报错)
env.cfg.rewards["single_support"].weight = 3.0
```

原因:各个 manager 在 `__init__` 时**深拷贝**了 cfg。之后写 `env.cfg` 改的是
那份没人再看的原件。这个坑也会咬到自定义的评估脚本(强制设置 spawn 状态时)。

### 5.5 ⚠️ play 时课程会把你的设置冲掉

课程按 `env.common_step_counter` 计数,而 **play 会话里它从 0 开始**。
结果:`uv run play` 永远只能看到**第 0 阶段**的配置。

`microduck_standup_env_cfg.py` 里那份(你工作区里未提交的)diff 干的正是这件事:

```python
if play:
    play_face_up = _resolve_play_face_up()          # 读环境变量 STANDUP_PLAY_FACE_UP
    if play_face_up is not None:
        cfg.events["set_ground_state"].params.update({...})
        del cfg.curriculum["ground_state_mix"]       # ← 关键:课程会在 reset 事件之前跑,不删掉就会把上面的设置覆盖回去
```

**从第一天就把这个套路抄进去**(`:122-152` 定义,`:909-924` 使用),
否则你后面每次 play 都在看阶段 0 的行为,还以为策略没学会。

### 5.6 逆向课程(reverse curriculum)—— 治"学不会最后一米"

让一部分环境**直接从"已经单腿站好"的姿态开始**。这样"站稳之后怎么维持"
就有了 on-policy 数据。

机制:`set_random_ground_state`(`mdp.py:4157`)已经支持按概率从不同姿态出生,
并支持 `sitting_joint_overrides`(一个 `{关节索引: 角度}` 字典)+ 噪声。
你可以照它的样子加一个 `one_leg_prob` + `one_leg_joint_overrides`。

配合课程逐步调整比例:

| 阶段 | 双腿站立出生 | 单腿姿态出生 |
|---|---|---|
| 1 | 100% | 0% |
| 2 | 70% | 30% |
| 3 | 40% | 60% |

**AGENTS.md 的节奏原则**:
> 每个阶段都要和策略**实际学会的东西**对齐。当某个 wandb 指标在课程阶段边界上
> **正好向下一跳**,说明节奏太快了 —— 应该**拉长阶段或推迟引入**,绝不要提前。

---

## 第 6 章 · 代码落地清单

这一章是你自己动手时的检查表。**不给实现代码**,只说改哪几个文件、抄谁。

### 6.1 新建配置文件

`src/mjlab_microduck/tasks/microduck_singleleg_env_cfg.py`

**骨架从 `microduck_standup_env_cfg.py` 整体复制。** 它虽然是"独立式"建法,
但那套 DR + 观测噪声 + 延迟 + NaN 守卫的移植工作**已经做完了**,你复制过来白拿。

**奖励配料从 `microduck_ball_kick_env_cfg.py` 抄。** 它有你需要的两样东西:
第二个单脚接触传感器(`:160-172`)和 `single_foot_grounded_reward` 的用法(`:254`)。

机器人模型用 `MICRODUCK_STANDUP_ROBOT_CFG`(`microduck_constants.py:181`)。
它对应 `robot_groundcontact.xml` —— 身体各部位都有碰撞体,摔倒时会真的躺在地上,
这正是你需要的(纯 `walk` 模型只有两只脚有碰撞体,摔倒"很便宜")。

> **命名注意**:仓库最近把 `allcollisions` 改名成了 `groundcontact`
> (`MICRODUCK_GROUNDCONTACT_XML`),同时新增了一个真正的全碰撞模型
> `robot_allcollisions.xml`(70 个几何体,暂无任务使用)。别搞混。

文件结构照抄这个顺序:
1. 模块 docstring(**认真写** —— 写清任务定义、初始状态、目标、以及你的奖励哲学)
2. `ENABLE_*` 开关 + 调参常量
3. import
4. `def make_microduck_singleleg_env_cfg(play: bool = False, rough: bool = False)`
5. `MicroduckSingleLegRlCfg`

### 6.2 追加 mdp 函数

在 `src/mjlab_microduck/tasks/mdp.py` **末尾**追加,加一个任务分组注释。
至少需要两个:

- `com_over_support_foot` 的**去门控版本**(见 §4.4)
- `single_support_reward` 的**去命令门控版本** —— 原版(`mdp.py:4701`)用
  `cmd_x`(前进速度命令)做缩放,零命令站立时恒为 0

### 6.3 注册任务

`src/mjlab_microduck/tasks/__init__.py`:

```python
# 1) import 块(在其他 cfg import 附近)
from mjlab_microduck.tasks.microduck_singleleg_env_cfg import (
    make_microduck_singleleg_env_cfg, MicroduckSingleLegRlCfg,
)

# 2) 注册(在其他 register_mjlab_task 附近)
register_mjlab_task(
    task_id="Mjlab-SingleLeg-Flat-MicroDuck",
    env_cfg=make_microduck_singleleg_env_cfg(),
    play_env_cfg=make_microduck_singleleg_env_cfg(play=True),
    rl_cfg=MicroduckSingleLegRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
```

任务 ID 命名惯例:`Mjlab-<任务名>-<地形>-MicroDuck`。

`_BACKLASH_TASKS` 那一行**先别加** —— 你现在不上真机,加了只是让每次改动
都要维护两份。

### 6.4 RL 配置

抄 `microduck_standup_env_cfg.py:1173-1210`,只改三处:

```python
MicroduckSingleLegRlCfg = RslRlOnPolicyRunnerCfg(
    ...
    experiment_name="microduck_singleleg",   # ← 改
    run_name="microduck_singleleg",          # ← 改
    max_iterations=3000,                     # ← 先设 3000
    algorithm=PpoWithSymmetryCfg(
        ...,
        symmetry_cfg=None,                   # ← 必须 None(非对称任务)
    ),
)
```

PPO 的其他超参(`learning_rate=1e-3`、`schedule="adaptive"`、`gamma=0.99`、
`clip_param=0.2`、网络 `(512,256,128)` + ELU)**照抄别动** —— 全仓库所有任务
用的都是同一套,不是你该调的地方。

### 6.5 写测试

`tests/test_singleleg_cfg.py`,照 `tests/test_roller_crouch_cfg.py`(48 行)或
`tests/test_spin_cfg.py`(110 行)的模式。**纯 CPU,几秒跑完,不需要 GPU。**

模式很简单:调工厂函数拿到 cfg,然后对字典做断言:

```python
def test_cfg_has_the_single_leg_rewards():
    cfg = make_microduck_singleleg_env_cfg()
    for name in ("support_grounded", "swing_lifted", "com_over_support"):
        assert name in cfg.rewards, name
    # 惩罚项必须是负权重(或自带负号的函数配正权重)
    assert cfg.rewards["swing_lifted"].weight < 0.0

def test_symmetry_is_off_for_this_asymmetric_task():
    assert MicroduckSingleLegRlCfg.algorithm.symmetry_cfg is None

def test_obs_layout_matches_standup():
    a = make_microduck_singleleg_env_cfg()
    b = make_microduck_standup_env_cfg()
    for g in a.observations:
        assert list(a.observations[g].terms) == list(b.observations[g].terms)
```

最后一个测试(观测布局一致性)是本仓库的**惯用做法**,它锁住的正是 61 维观测合同。

跑测试:

```bash
uv run --with pytest pytest tests/test_singleleg_cfg.py -v
```

> **已知的既有失败**:`tests/test_aarch64_cuda_torch.py::test_x86_64_resolution_stays_on_pypi`
> 目前是红的,因为你本地的 `uv.lock` 把 torch 的源换成了清华镜像。
> 这**不影响本地训练**,但如果哪天想用 `--hf-jobs` 上云,云端会拉到不同的 wheel。
> 跑全量测试时忽略这一条即可(其余 165 个是绿的)。

---

## 第 7 章 · 训练与读日志

### 7.1 冒烟测试(每次改完都要跑)

```bash
uv run train Mjlab-SingleLeg-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
```

它验证:环境能构建、能推进不出 NaN、观测是 61 维、每个奖励项都能算出来。
AGENTS.md:**几分钱抓住约 95% 的配置错误。**

### 7.2 正式训练

```bash
uv run train Mjlab-SingleLeg-Flat-MicroDuck --env.scene.num-envs 4096
```

**时间预算**(基于你这台机器实测的 ~2.6 秒/迭代):

| 目标 | 迭代 | 耗时 |
|---|---|---|
| 看出苗头 | 500 | ~22 分钟 |
| 基本成型 | 1500 | ~1.1 小时 |
| 第一次完整尝试 | 3000 | ~2.2 小时 |
| 打磨 | 6000 | ~4.3 小时 |

显存 8 GB 跑 4096 环境你已经验证过可行(上次 StandUp 就是)。OOM 就降到 2048。

**中断续训**:

```bash
uv run train Mjlab-SingleLeg-Flat-MicroDuck --env.scene.num-envs 4096 \
  --agent.run-name resume --agent.load-checkpoint model_1500.pt --agent.resume True
```

### 7.3 盯三件事

1. **平均奖励在涨**
2. **主任务项在涨** —— 打开 `Episode_Reward/com_over_support` 和
   `Episode_Reward/support_grounded`。总奖励完全可以靠正则项一路上涨而动作从没发生过
3. **每个惩罚项 ≤ 0**

还要注意:`Episode_Reward/` 打印的是**加权后**的值。权重为 0 的项恒显示 0,
**不代表行为没发生**。要对着你的课程表来解读。

### 7.4 心理准备

AGENTS.md 的原话:**头 2–5 轮会经历"奖励作弊打地鼠"—— 这是正常流程,不是失败。**
你会看到策略发明出各种你没想到的偷懒方式,然后你补一个惩罚项,它再发明一个新的。
第 9 章的表格列了本任务能预见的几种。

---

## 第 8 章 · 评估

### 8.1 看视频

```bash
uv run play Mjlab-SingleLeg-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/microduck_singleleg/<时间戳>/model_3000.pt \
  --num-envs 1
```

多看几个 checkpoint(1000 / 2000 / 3000),看它是怎么演化的。

### 8.2 先测量,再理论化

这是 AGENTS.md 反复强调的铁律:

> 当一次训练"失败"时,**先跑一次 headless 评估**(按出生类型分组的批量测试、
> 末态聚类、角速度曲线),再动奖励函数。

历史上多次"失败"其实是:
- 看错了 checkpoint(拿早期的当最终的)
- 成功判据把同一种行为切成了两半(比如"保持 3 秒"的阈值正好卡在分布中间)
- 惩罚上限在跟真实物理打架(策略想做的事被你的上限禁止了)

对单腿站立,值得统计的量:
- **单腿保持时长的分布**(不是平均值 —— 平均值会把"一半 5 秒一半 0 秒"和
  "全部 2.5 秒"混为一谈)
- **摔倒方向**(向支撑腿侧摔 vs 向摆动腿侧摔 —— 两者的成因完全不同)
- **哪个几何体先着地**
- **`hip_roll` 的分布**(是不是长期贴在限位上)

### 8.3 仿真指标通过 ≠ 视频好看

两个都要看。AGENTS.md:
> 报告 rollout 实际展现的东西("能翻但三次里有一次脸着地"),
> 而不是"它работает了!"。什么时候算够好,是使用者决定的。

---

## 第 9 章 · 常见失败模式速查表

| 症状 | 大概率原因 | 去改哪里 |
|---|---|---|
| 两只脚从头到尾都不抬 | 单支撑奖励太弱,或正则项上得太早 | 阶段 2 的权重表;把 `action_rate` 的引入往后推 |
| 抬腿了但立刻**向摆动腿侧**摔 | 质心没有横移过去 | 加/加强 `com_over_support_foot`,`std` 收到 0.02 |
| 抬腿了但**向支撑腿侧**摔(过冲) | 横移过头,缺阻尼 | 加 `body_ang_vel_at_height`,或降低质心项权重 |
| 站着不动,什么也不干 | `action_rate` 在技能出现前就全额生效("不动 = 不扣钱") | 课程推迟到阶段 3 |
| 疯狂高频抖脚 | 单支撑奖励被当成可刷分项 | 加 `contact_frequency_penalty` + 最短保持时长门控 |
| 用膝盖 / 下巴当第三只脚 | 没有惩罚非脚部位接地 | 加 `robot_ground_contact` 传感器 + `body_impact_cost` |
| `hip_roll` 长期贴在 ±22° 限位上 | 没有限位邻近惩罚 | 加 `joint_pos_limit_proximity`(qpos 侧,**不是** ctrl 侧) |
| 两脚并到一起当双支撑 | 没约束脚间距 | 加 `feet_distance_penalty` |
| 奖励一直涨但视频很丑 | 主任务项没涨,靠正则项虚涨 | 逐项看 `Episode_Reward/` |
| 某个惩罚项的 `Episode_Reward/` 是**正的** | 权重符号写反了 | 见 §2.3 |
| 左右腿行为诡异地耦合 | 忘了关对称性增广 | `symmetry_cfg=None` |
| 某个指标**正好在课程阶段边界向下跳** | 课程节奏太快 | 拉长该阶段,或把引入时机往后推 |
| play 时行为和训练日志对不上 | 课程在 play 会话里被重置到阶段 0 | 抄 §5.5 的 play 覆盖套路 |

---

## 附录

### A. 两周学习节奏建议

| 天 | 做什么 | 产出 |
|---|---|---|
| 1 | 读第 0 章术语表;跑第 1 章的冒烟测试和回放 | 能看懂日志的每一行 |
| 2–3 | 精读 `microduck_standup_env_cfg.py`,对照第 2 章的解剖图 | 能说出每个 manager 在干什么 |
| 4 | 跑第 3 章的可行性脚本,自己改改参数玩 | 对这个机器人的物理极限有手感 |
| 5–6 | 读 `mdp.py` 里第 4 章表格提到的那十几个函数的源码 | 知道每个函数返回什么形状、什么范围 |
| 7 | 写配置文件骨架 + 测试,跑冒烟测试 | 环境能跑起来(哪怕奖励还很烂) |
| 8 | 第一次正式训练 3000 迭代 | 一条曲线 + 一段视频 |
| 9–14 | 打地鼠循环:看视频 → 定位作弊 → 补惩罚 → 再训练 | 能站住的策略 |

**给新手的两条心态建议:**

1. **第一次训练一定会失败,这是流程的一部分,不是你的问题。** AGENTS.md 明确说
   要预期 2–5 轮的"奖励作弊打地鼠"。
2. **不要一次改五个奖励项。** 一次改一个,否则你不知道是哪个起了作用。

### B. 外部学习材料

| 主题 | 推荐 |
|---|---|
| PPO 直觉理解 | OpenAI Spinning Up 的 PPO 章节(有中文翻译版) |
| 势能式塑形的理论 | Ng, Harada, Russell (1999), *Policy Invariance Under Reward Transformations* |
| MuJoCo 基础 | 官方文档的 Overview + Computation 两章;`mujoco.viewer` 交互式玩模型 |
| mjlab | 仓库 <https://github.com/mujocolab/mjlab>,重点看 `tasks/velocity/velocity_env_cfg.py` |
| 双足平衡的控制学背景 | 搜 "ZMP / capture point / ankle strategy vs hip strategy" |
| 本仓库的经验总结 | **`AGENTS.md`(以及你已经翻译的 `AGENTS-zh.md`)—— 这是最重要的一份** |

### C. 文件 · 行号索引

> 核实于 2026-09-04,`develop` 分支,commit `29e887e`。
> 仓库更新后行号可能漂移,用 `grep -n "^def <函数名>"` 重新定位。

**MDP 函数**(全部在 `src/mjlab_microduck/tasks/mdp.py`,共 7188 行 / 220 个函数):

| 函数 | 行 | 用途 |
|---|---|---|
| `_servo_joint_ids` | 126 | 非 `passive_` 关节的索引 |
| `_servo_joint_pos` | 146 | 同上,直接取角度 |
| `upright_progress` | 546 | 势能式:Δcos(倾斜) |
| `height_progress` | 575 | 势能式:Δ高度 |
| `body_upright_linear` | 683 | cos(倾斜),处处有梯度 |
| `body_upright_gaussian` | 710 | 竖直附近很陡 |
| `standing_composite_score` | 810 | 高度×竖直×姿态 三高斯连乘 |
| `standing_success_bonus` | 856 | 二值成功奖励 |
| `feet_flat_penalty` | 1463 | 脚掌没踩平 |
| `body_impact_cost` | 1644 | 接触力惩罚 |
| `contact_frequency_penalty` | 1761 | 接触切换过频 |
| `joint_pos_limit_proximity` | 1989 | qpos 侧限位邻近惩罚 |
| `pose_target_match` | **2059 / 2396** | ⚠️ 定义了两次,用关键字参数 |
| `pose_l1_penalty` | 2421 | |
| `height_target_gaussian` | 2440 | |
| `height_l1_penalty` | 2454 | |
| `com_over_support_foot` | 2743 | ⚠️ 目前无任务使用,且被相位门控 |
| `reward_weight` | 3442 | 权重课程(阶跃) |
| `set_random_ground_state` | 4157 | 按概率从不同姿态出生 |
| `event_param_curriculum` | 4453 | 事件参数课程 |
| `single_support_reward` | 4701 | ⚠️ 被前进速度门控 |
| `zero_command_padding` | 5115 | 命令槽补零 |
| `pose_command_range_curriculum` | 5512 | 命令范围课程 |
| `feet_distance_penalty` | 5571 | 两脚太近 |
| `single_foot_grounded_reward` | 5809 | 单只脚是否触地(二值) |

**配置文件**:

| 位置 | 内容 |
|---|---|
| `microduck_standup_env_cfg.py:26` | `ENABLE_SYMMETRY = False` |
| `microduck_standup_env_cfg.py:82` | `_LEG_JOINTS` |
| `microduck_standup_env_cfg.py:89,95` | `SIT_Z = 0.060`,`STAND_Z = 0.115` |
| `microduck_standup_env_cfg.py:122-152` | play 时的 spawn 覆盖套路 |
| `microduck_standup_env_cfg.py:186` | 工厂函数入口 |
| `microduck_standup_env_cfg.py:194` | `feet_ground_contact` 传感器 |
| `microduck_standup_env_cfg.py:233-559` | 奖励段 |
| `microduck_standup_env_cfg.py:560-641` | 观测段 |
| `microduck_standup_env_cfg.py:705-714` | 终止段 |
| `microduck_standup_env_cfg.py:715-869` | 事件段 |
| `microduck_standup_env_cfg.py:882-1168` | 课程段 |
| `microduck_standup_env_cfg.py:1173-1210` | `MicroduckStandUpRlCfg` |
| `microduck_ball_kick_env_cfg.py:160` | 单脚接触传感器模板 |
| `microduck_ball_kick_env_cfg.py:254` | `single_foot_grounded_reward` 用法 |
| `microduck_roulade_env_cfg.py:190` | `robot_ground_contact` 传感器模板 |
| `microduck_spin_env_cfg.py:24` | 非对称任务关闭 symmetry 的写法 |
| `microduck_velocity_env_cfg.py:22` | `NUM_STEPS_PER_ENV = 24` |
| `microduck_constants.py:181` | `MICRODUCK_STANDUP_ROBOT_CFG` |
| `robot_groundcontact.xml:166,325` | 脚部碰撞几何体(`robot_walk.xml` 里是 164/319) |
| `robot_groundcontact.xml:174,331` | 脚部站点(`robot_walk.xml` 里是 172/325) |
| `joints_properties.xml:23-31` | `chosen_actuator`:kp=0.55,力矩 ±0.96 Nm,ctrlrange ±10 |

**测试模板**:`tests/test_roller_crouch_cfg.py`(48 行,最短)、
`tests/test_spin_cfg.py`(110 行)、`tests/test_roller_standup_cfg.py`(539 行,最详尽)。

### D. 本文档的实测数据是怎么来的

所有物理数字都用 MuJoCo 在 `src/mjlab_microduck/robot/microduck/scene.xml`
(它 include 的是 `robot_groundcontact.xml`,正是本任务要用的模型)的 `STAND`
关键帧上算出来的:

| 数字 | 方法 |
|---|---|
| 双脚间距、质心、质量分布 | `mj_forward` + `mj_comPos` + `body_mass` |
| 脚掌尺寸 | 把 `sole_left` 网格顶点变换到世界系取包围盒 |
| 关节限位 | `m.jnt_range` |
| 开环保持失败 | 3 秒仿真,`ctrl = 关键帧关节角` |
| 维持姿态所需力矩 | 临时把 `kp` 改成 30、力矩上限放开,稳定后读 `d.actuator_force` |
| 承重转移上限 | 双脚触地,2 秒匀速过渡到目标 `hip_roll`,用 `mj_contactForce` 读两脚法向力 |
| 平衡裕度 | 支撑脚摆平 → 接触点凸包 → 质心投影的有符号距离 → `differential_evolution` 在 14 维上最大化 |
