# AGENTS.md（代理指南）

MicroDuck机器人的强化学习训练环境—一个约800克、约25厘米高的双足
机器人，配备14个Dynamixel XL330伺服—基于[mjlab](https://github.com/mujocolab/mjlab)
（MuJoCo Warp）使用PPO（rsl_rl）构建。策略在此处以50 Hz频率训练，导出为
ONNX格式，由`pollen-robotics/microduck`仓库中的运行时在真实机器人上部署。
仿真到实物转移是重点：下面的每个惯例都存在是因为打破它会产生一个在查看器中工作而在硬件上失败的策略。

## 命令

```bash
uv run list-envs                                    # 实时任务注册表
uv run train <TASK_ID> --env.scene.num-envs 4096    # 训练（添加--hf-jobs以获取Hugging Face Jobs）
uv run train <TASK_ID> --env.scene.num-envs 64 --agent.max_iterations 5   # 烟雾测试—始终先运行
uv run play <TASK_ID> --wandb-run-path <entity/project/run_id>
uv run scripts/export.py <TASK_ID> --wandb-run-path <...>   # → ONNX（烘烤观察归一化器—强制路径）
uv run scripts/infer_policy.py --walking out.onnx   # CPU MuJoCo部署排练
uv run --with pytest pytest tests/
```

在64个环境上进行5次迭代的烟雾测试以几分钱的成本捕获~95%的配置错误。
永远不要在没有烟雾测试的情况下启动长时间运行。

## 仓库地图

- `src/mjlab_microduck/tasks/mdp.py` — 所有自定义MDP函数（奖励、事件、
  观察、命令、课程）。在此处添加新函数，按任务分组。
- `src/mjlab_microduck/tasks/microduck_*_env_cfg.py` — 每个任务
  族一个cfg模块。`microduck_velocity_env_cfg.py`是主要行走配方AND共享
  基础（机器人、DR、观察、命令），其他环境构建或镜像其上。
- `src/mjlab_microduck/tasks/__init__.py` — 任务注册（基础+`-Backlash-`变种）。
- `src/mjlab_microduck/tasks/backlash.py` — 将任何环境cfg包装成其齿隙孪生。
- `src/mjlab_microduck/robot/microduck_constants.py` — 机器人配置、HOME框架、BAM执行器配置。
- `src/mjlab_microduck/robot/microduck/` — 来自Onshape的MJCF导出
  （onshape-to-robot，每个模型一个`config_mjcf_*.json`）+场景+`add_backlash.py`。
- `src/mjlab_microduck/actuator/friction_dr_bam.py` — BAM执行器+摩擦DR+齿隙编码器。
- `scripts/` — 导出、推理、仿真到实物比较、wandb助手。
- `tests/` — 配置不变和mdp函数回归测试（CPU，无需GPU）。

## 不变量—不要打破这些

- **观察布局是61D（执行器）并在整个策略族中共享**，所以
  策略在运行时是热可交换的：48个基础本体感受+
  13D命令块`[twist(3), head_pose(4), body_pose(6)]`，按这个顺序。
  不使用命令槽的环境会ZERO-PAD它（保持观察项，
  采样微小范围）—永远不要删除槽。
- **关节布局**（14个伺服，ctrl idx = walk/allcollisions上的关节idx
  模型）：0–4左腿（髋部偏航、髋部滚动、髋部俯仰、膝盖、脚踝），5–8
  颈部/头部（颈部俯仰、头部俯仰、头部偏航、头部滚动），9–13右腿。
  在轮滑/齿隙模型上，被动关节交错—永远不要在mdp函数中硬编码关节
  索引；使用mdp.py中的`_servo_joint_ids`/`_servo_joint_pos`
  助手（在简单模型上是恒等式，在其他地方正确）。
- **未激励的关节都命名为`passive_*`**（轮、齿隙铰链）。
  每个执行器/观察/奖励选择器使用`^(?!passive_).*`—添加关节时保持前缀
  惯例，新的`passive_`正则表达式不能
  意外匹配齿隙关节（`^passive_.*wheel`，不是`^passive_.*`）。
- **执行器是BAM**（压力控制的XL330模型，摩擦由
  执行器计算）。两个后果：任何STANDALONE环境cfg必须注册
  `expand_bam_friction_fields`启动事件，关节摩擦DR必须缩放
  执行器的`friction_scale`—`dof_frictionloss`在BAM下为零，所以
  随机化它是无声的no-op。
- **观察归一化ON**→归一化器必须烘烤到ONNX中。
  `scripts/export.py`执行此操作；仿真中播放隐藏错误（它无论如何都应用
  归一化器），所以永远不要手动转换检查点。
- **策略未过滤**（训练中无行动低通）。在没有匹配的运行时标志和转移测试的情况下不要添加EMA
  过滤—trained-with/deployed-without（任一方向）打破转移。
- **域随机化不能在重置时累积。** mjlab 1.3.0的
  `dr.*`操作与`operation="add"/"scale"`本身是非累积的（它们
  重新读取编译时默认值）；自定义DR函数必须restore-then-apply。
  一个累积的CoM随机化器曾经让每次长时间运行恶化长达数月。
- 如果观察被重新映射到传感器视图（齿隙编码器、偏差），任何追踪
  同一量的REWARD必须测量相同的视图—否则策略
  被惩罚以纠正其所看到的。
- `-Backlash-`任务变种必须镜像其基础任务的机器人模型
  （walk/allcollisions/rollers），所以齿隙A/B比较是无混淆的。

## 构建新环境—工作流程

1. **选择最接近的模板**并在其基础上构建，不要从头开始：
   运动→速度配方；以姿态结束的情节技巧→
   起立；命令的两态→坐站；动态机动→翻滚
   （阅读其cfg文档字符串—它编码一个5次运行的经验弧）。基于
   `make_microduck_velocity*_env_cfg`构建保持DR/观察/噪声/延迟同步
   免费；如果你从mjlab的基础模板独立构建，你必须移植
   整个DR+观察噪声+NaN保护堆栈自己（grep速度有什么导线：
   `_safe`批评者观察项、`nan_state`终止与sensor_names、
   `expand_bam_friction_fields`、编码器偏差、IMU错位）。
2. **在训练前验证仿真中的物理假设**—这是单一
   最大的时间节省器：
   - 目标/休息姿态必须是稳定的平衡：从有噪声的初始化中
     保持其ctrl 3秒并检查倾斜，而不仅仅是高度（只记录z的
     安定测试将摔倒状态报告为"休息很好"）。
   - 从仿真中实际机器人测量目标高度（例如站立时的躯干z
     策略），永远不要在模型修订中携带它们。一个5毫米错误的
     STAND_Z曾经将目标变成不可能的目标长达数天。
3. **配置惯例**：`ENABLE_*`切换+cfg文件顶部的调整常数；
   工厂`make_..._env_cfg(play: bool, rough: bool)`；在
   `tasks/__init__.py`中注册（+`_BACKLASH_TASKS`表（如适用））；拥有
   `RslRl...RunnerCfg`与不同的`experiment_name`。对称镜像损失
   可用（`symmetry.py`中的61D表）—默认关闭，从不用于
   非对称任务。
4. **写cfg测试**（见`tests/test_*_cfg.py`）：关节索引在
   实际模型上解析，奖励权重有预期符号，门
   按预期打开/关闭。这些在CPU上运行并锁定不变量。
5. **烟雾测试**（64个环境，5次迭代）：构建、步骤NaN-free、观察是61D、
   每个奖励项计算、ONNX导出。
6. 训练、看日志（下面）并期望2–5次奖励黑客whack-a-mole迭代—这是
   正常的，下面的经验快捷方式大部分。

## 奖励设计—每条都是艰难学到的规则

- **符号惯例（四个环境钻研）：** mdp.py有两个惩罚风格。mjlab-base
  成本函数返回≥0→负权重。自否定microduck函数
  （`*_penalty`、`*_l1`返回≤0）→POSITIVE权重。在自否定惩罚上的负权重
  双否定为违规奖励，策略会耕种它（屁股跳跃、崩溃坐）。**绝对检查：在
  每次运行时，wandb中的每个`Episode_Reward/<penalty>`必须≤0。**
- **RL优化奖励的字母。** 每个未指定的自由度
  会被利用（弹道鞭子而不是滚动、肩滚而不是矢状、头三脚架而不是站立）。
  编码什么计为在硬状态基门中的机动（支持接触、方向轴
  检查、闩锁），不在小惩罚戳。
- **没有头奖**：任何"到达X"奖励必须是速率限制或倾斜的。
  提前到达然后按步骤支付的目标状态是头奖，买任意暴力。
  对于命令的转换，追踪一个倾斜的内部
  目标（常数速率混合）—在坡道前面支付零，所以慢IS
  argmax。单独速度上限惩罚集成为有限成本并失败。
- **永远不要在处于坏状态时门正奖励**（摔倒、低）—策略
  停泊在最便宜的合格姿态中并耕种它。使用
  基于势的塑形（支付Δ进度，例如Δcos(倾斜)：上升支付，
  保持支付零，不可耕种）。对于休息任务，审计每个正项
  对抗每个稳定拍打（在背/脸/侧面）：如果拍打保持大部分
  堆栈，策略会拍打。
- **情节姿态着陆任务**：从t=0单一固定目标（关节和高度上的高斯+L1、
  慷慨的标准）+|a_z|影响惩罚+两层
  竖立—NOT关键帧/路点轨迹（策略在
  路点处驻营）。路径是RL应该发现的。
- **正则化器有两种**。运动阻滞（body_ang_vel、
  angular_momentum、姿态标准）惩罚动态运动物理上
  需要—对动态任务保持低。光滑（action_rate、
  joint_torque_rate）抑制抖动而不阻止慢大运动—安全
  加权，但在技能发现后引入（课程从~0）：任何
  尝试税活跃而硬技能正被探索使"什么都不做"赢。
  缓慢仔细任务（到达）想要比行走更重的光滑。
- **比较奖励质量，而不是权重，在环境之间复制正则化器时。**
  PPO看到相对优势：相同的action_rate权重在4×下弱4倍
  更大的正任务堆栈。
- **追踪高斯标准**：≈你仍然关心的错误，不是最大
  错误—太松在小错误处没有梯度。BUT在收紧前，
  询问错误是否可以通过策略逃脱或内在于行为
  你想要的（一个占身体质量38%的头必须在行走时振荡；一个紧
  瞬间头追踪标准对行走征税如此之硬策略站立
  静止）。仅价格可逃脱部分—例如1秒EMA上的L1费用
  DC偏差并让振荡相消。
- **乘法合成在目标状态处击败加法和。** 当
  加法堆栈有一个折衷盆（通过倾斜80%每项），一个
  高斯的产品在任何单一不足因子上崩溃—但选择标准
  足够宽，CURRENT策略得分明显，或梯度
  看不见没有变化。
- **关节在硬限制处驻营**：用qpos-side限制接近
  对违规关节的惩罚来修复；库存`dof_pos_limits`仅在
  范围的最后~7.5%激发，和命令侧惩罚不工作（宽ctrlrange是
  故意的—低kp伺服需要超射）。

## 命令、观察、死亡权重

- **从不非零的命令输入永远有死权重。** 每个
  命令槽从步骤0保持一个小的非零采样范围（即使在
  奖励权重0）所以其输入神经元保持活力以获得后来的课程。
- **零命令行为必须被明确训练**（`zero_command_prob`风格
  精确零采样）：统一采样本质上从不产生全零
  命令，这正是部署空闲状态。
- 稀有但重要的命令区域需要显式桶—例如原地转
  （`rel_turn_in_place_envs`）：独立统一采样使旋转~2%
  经验和它从不训练。

## 课程

- 步骤是环境步骤：`迭代×24`（`NUM_STEPS_PER_ENV = 24`）。
- 使用证明的拆分：`microduck_mdp.reward_weight`对于权重计划，一个
  专门的params-curriculum对于命令/事件范围。`mdp.reward_weight`是
  步函数，不是插值—离散化坡道为阶段。
- 通过管理器变异项cfg（`env.event_manager.get_term_cfg(...)`），
  永远不是`env.cfg.events[...]`—管理器deepcopy他们的cfg在init，所以写入
  `env.cfg`是静默no-ops（这也咬评估脚本强制
  生成状态）。
- **将每个阶段与策略实际学到的内容相位对齐**：不要
  在当前切片固化前硬化生成混合；不要引入
  税在技能存在之前。当wandb指标正好在
  课程阶段边界向下踏步时，步速错误—拉伸阶段或移动
  引入后来，从不更早。
- 反向课程生成（从机动中途开始情节，
  包括几乎完成）是可靠的修复对于"学习开始，永不
  最后一英里"—前沿否则得不到按时数据。

## 训练操作和阅读运行

- wandb项目`mjlab_microduck`；日志在`logs/<experiment_name>/`；恢复
  与`--agent.load-checkpoint model_XXXX.pt --agent.resume True`。
- 观察每次迭代：平均奖励上升AND情节长度表现
  按任务需求；每个惩罚项≤0；主任务项实际增长
  （总奖励可以纯粹通过正则化器上升而技巧从不发生）。
  `Episode_Reward/<term>`记录加权值—项在权重0处读取0
  不管行为如何，所以根据权重计划解释。
- 预算：简单情节技巧≈1000次迭代在4096环境；步态和
  课程重任务恢复需要4000–6000。
- **在理论化前测量。** 当运行"失败"时，运行无头评估
  实际检查点（每生成类型电池、末态簇、角速率
  配置文件）在改变奖励前：过去"失败"转向为早期
  检查点、成功标准分裂一个行为簇为一半、和
  支付上限对抗测量物理。仿真度量可以通过而视频
  失败人眼—观看视频AND检查哪个几何/轴接触。
- 报告rollouts实际显示（"滚动但面部植物1在3"），不是
  "它工作！"。用户决定什么时候足够好。

## 仿真到实物陷阱（耗费真实调试周）

- 一个新鲜`uv sync`是地面真实（HF Jobs运行一个）：任何仅
  通过手动安装本地包工作的东西会远程死亡。保持
  `pyproject.toml`诚实。
- **轮是每架构。** 在linux-`aarch64`（DGX Spark/GB10、Jetson）PyPI的
  torch wheel是CPU-ONLY（`2.9.1+cpu`、`torch.version.cuda is None`），所以
  `torch.cuda.device_count() == 0`和mjlab的`select_gpus()`索引空
  列表→`IndexError`在迭代0前。`[tool.uv.sources]`路由torch到
  cu129索引对于`aarch64`仅（cu129匹配CUDA工具包warp
  束；x86_64/HF Jobs停留在PyPI）。两个静默破点，都锁定
  由`tests/test_aarch64_cuda_torch.py`：torch必须停留为DIRECT依赖
  （uv应用`[tool.uv.sources]`到直接deps仅—删除
  冗余看`torch==`引脚使路由no-op），和引脚必须
  停留`==`，因为CUDA索引携带较新的构建比PyPI（一个`>=`
  静默拖torch 2.9.1→2.13.0）。
- 物理对齐限制：一个25厘米机器人在3.5–5.5弧度/秒自然翻滚—
  不要通过上限强加人类规模的速度直觉；把反暴力
  压力在影响和激烈上（|a_z|、action_rate、支持门），不是
  旋转速度。
- IMU DR是零中心的—它训练对错位幅度的容忍，和
  无法补偿系统安装偏差（这是运行时校准）。
- 真实部署热切换ONNX策略（行走/站立/技巧）与共享
  观察合同—在`scripts/infer_policy.py`中排练在触及
  机器人前，与正确命令槽写入（一个姿态标志住在
  twist vx槽；馈送全零意味着"站立"，看起来像"策略
  忽略按钮"）。
