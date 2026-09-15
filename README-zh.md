# Microduck RL

<img width="2215" height="884" alt="image" src="https://github.com/user-attachments/assets/5db7cc83-b3ce-4f7c-83f0-0572a63baed7" />

MicroDuck（一个约800克、约25厘米高的双足机器人）的强化学习训练环境，基于
[mjlab](https://github.com/mujocolab/mjlab)（MuJoCo Warp）使用PPO算法构建。
策略在此处以50 Hz频率训练，导出为ONNX格式，然后由[pollen-robotics/microduck](https://github.com/pollen-robotics/microduck)仓库中的运行时在真实机器人上部署。

<!-- 英雄视频 — 真实机器人合集：行走、起立、翻滚、轮滑。
     保持简短（约30秒）并优先展示真实机器人：这是"为什么我应该关心"的镜头。 -->

https://github.com/user-attachments/assets/50c3d537-8db2-4005-9d9c-3472faeec4d0

该仓库编码了完整的仿真到实物（sim2real）的配方：[BAM](https://github.com/Rhoban/bam)
执行器物理模型、域随机化、齿隙仿真以及使其工作的奖励设计经验
（详见[AGENTS.md](AGENTS.md)中的精炼播放手册）。

## 快速开始

需要CUDA GPU（训练通过MuJoCo Warp运行）和[uv](https://docs.astral.sh/uv/)。

> **在ARM盒子上（DGX Spark / GB10、Jetson）：** `uv sync`首次运行会下载约2GB的CUDA
> wheels，uv的默认30秒HTTP超时可能会在下载中途中止。
> 第一次同步时导出`UV_HTTP_TIMEOUT=600`。

```bash
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl

# 训练行走策略（使用你的GPU；在4096个环境下约1-2小时可得到可用的步态）
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096

# 在查看器中观看训练好的策略
uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity/project/run_id>

# 导出为ONNX供部署
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <...>

# 用键盘驱动导出的策略在CPU MuJoCo中运行
uv run scripts/infer_policy.py --walking output.onnx
```

从检查点恢复：

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.run-name resume --agent.load-checkpoint model_29999.pt --agent.resume True
```

没有GPU？在任何train命令中添加`--hf-jobs`可在Hugging Face Jobs上运行
而不是在本地运行（详见[scripts/hf/README.md](scripts/hf/README.md)）。

## 任务

`uv run list-envs`打印实时任务注册表。平坦/粗糙地形变种存在于注记位置。

<!-- 展示网格 — 每个任务族一个简短的GIF（仿真或真实），每行3个。
     如果你只能录制几个，优先级顺序：Velocity、VelStand（跌倒+恢复）、
     Roulade、SitStand、Rollers/Swizzle、BallKick。 -->

| 任务 ID | 地形 | 描述 |
|---|---|---|
| `Mjlab-Velocity-{Flat,Rough}-MicroDuck` | 平坦/粗糙 | **主要任务**：带速度命令和头部姿态命令的行走 |
| `Mjlab-VelStand-{Flat,Rough}-MicroDuck` | 平坦/粗糙 | 行走+一个策略中的跌倒恢复 |
| `Mjlab-StandUp-{Flat,Rough}-MicroDuck` | 平坦/粗糙 | 从面朝下/面朝上/坐下起立，然后保持站立+身体姿态控制 |
| `Mjlab-SitStand-{Flat,Rough}-MicroDuck` | 平坦/粗糙 | 一个策略中的温和坐↔站命令，头部可命令 |
| `Mjlab-GroundPick-{Flat,Rough}-MicroDuck` | 平坦/粗糙 | 蹲下并用嘴尖触地，然后返回站立 |
| `Mjlab-BallKick-Flat-MicroDuck` | 平坦 | 踢70毫米/15克球向前（执行器看不到球） |
| `Mjlab-Roulade-Flat-MicroDuck` | 平坦 | 在头上向前滚动，落地回到脚上 |
| `Mjlab-Velocity-Flat-MicroDuck-Rollers` | 平坦 | 轮滑速度追踪（脚下的被动轮） |
| `Mjlab-Velocity-Swizzle-MicroDuck` | 平坦 | 经典对称摇摆滑冰 |
| `Mjlab-RollerCrouch-Flat-MicroDuck` | 平坦 | 在轮滑时蹲下 |
| `Mjlab-RollerSlope-Flat-MicroDuck` | 斜坡 | 在轮滑时沿斜坡滑下 |
| `Mjlab-RollerStandUp-Flat-MicroDuck` | 平坦 | 从地面起立到轮子上 |
| `Mjlab-Spin-Flat-MicroDuck` | 平坦 | 在轮滑时原地快速旋转 |

在部署时，运行时在一个共享的61维观察合同后面热切换这些策略
（行走/恢复/技巧），所以其中任何一个都可以随时接管机器人。`scripts/infer_policy.py`
正好排练这一点：

```bash
uv run scripts/infer_policy.py --walking walk.onnx --standing stand.onnx \
    --sitstand sitstand.onnx --roulade roulade.onnx --new-cmd-obs
```

键盘驱动（速度命令、`G`地面接触、`Y`坐/站、`R`翻滚、
`K`/`L`踢）；`--debug`、`--save-csv`、`--record`支持仿真到实物比较。

### 齿隙变种

每个主要任务都有一个**齿隙**孪生任务，在一个模型上训练，该模型在每个14个伺服关节上都有±1°的齿轮间隙（总共2°）：
在任务ID中的`MicroDuck`之前插入`-Backlash`，例如`Mjlab-Velocity-Flat-Backlash-MicroDuck`。

齿隙对仿真到实物进行了正确建模：每个伺服获得一个未激励的
`passive_<joint>_backlash`铰链，因为真实编码器位于间隙的输出端，
固件PD仿真（`BacklashEncoderBamActuator`）和`joint_pos`/`joint_vel`观察
都通过齿隙读取（`qpos[servo] + qpos[backlash]`）。观察和
行动维数不变，所以ONNX导出和运行时无需更改。
见`src/mjlab_microduck/tasks/backlash.py`。

## 执行器模型

所有任务都为Dynamixel XL330使用[BAM](https://github.com/Rhoban/bam) M6执行器模型
（电压控制规律、反电动势、库仑/Stribeck/负载相关摩擦），
在每个环境上具有电池电压、负载下电压下降、命令延迟和摩擦幅度的逐环境域随机化
（`src/mjlab_microduck/actuator/`中的`FrictionDRBamActuator`）。

在这个规模—微型伺服驱动约800克的双足机器人—执行器保真度是
仿真到实物差距的大部分，这就是为什么执行器被建模为其电压
控制规律而不是理想PD。

## 机器人模型

MJCF模型位于`src/mjlab_microduck/robot/microduck/`中，
使用[onshape-to-robot](https://github.com/Rhoban/onshape-to-robot)从Onshape导出，
每个模型一个`config_mjcf_*.json`：

| XML | 用于 |
|---|---|
| `robot_walk.xml` | 速度（去除躯干/头接触—跌倒廉价） |
| `robot_allcollisions.xml` | VelStand、StandUp、SitStand、GroundPick、BallKick、Roulade（身体可以物理上躺在地上） |
| `robot_allcollisions_rollers.xml` | 轮滑任务（被动轮） |
| `robot_*_backlash.xml` | 齿隙任务变种（由`add_backlash.py`生成） |

`scene*.xml`文件使用地板+关键帧（STAND/SIT/FOLD）
来快速查看和用于`infer_policy.py`。

<!-- 图像 —并排渲染：行走模型vs轮滑模型（或碰撞几何可视化）。
     这里的一张图像使模型变种故事瞬间理解。 -->

## 项目结构

```
src/mjlab_microduck/
├── robot/
│   ├── microduck/                    # MJCF导出、导出配置、场景、add_backlash.py
│   └── microduck_constants.py        # 机器人配置、HOME框架、BAM执行器配置
├── actuator/friction_dr_bam.py       # BAM+摩擦DR+齿隙编码器反馈
├── tasks/
│   ├── __init__.py                   # 任务注册（基础+齿隙变种）
│   ├── mdp.py                        # 奖励、事件、观察、自定义类
│   ├── backlash.py                   # make_backlash_variant()环境配置包装器
│   └── microduck_*_env_cfg.py        # 每个任务族一个配置模块
├── train_cli.py                      # `train`入口点（+--hf-jobs）
└── hf_jobs.py                        # Hugging Face Jobs提交
```

值得了解的惯例：

- 观察布局在每个策略中共享（61维执行器观察：
  48个本体感受+命令`[twist(3), head_pose(4), body_pose(6)]`），这是
  什么使运行时策略热切换成为可能。不使用命令槽的环境会
  用零填充它而不是删除它。
- 未激励的关节都命名为`passive_*`（轮滑轮、齿隙
  铰链）；执行器、关节观察和姿态奖励使用`^(?!passive_).*`选择伺服关节。
- 域随机化切换是每个环境配置文件顶部的`ENABLE_*`布尔值。
- 关节布局（14个伺服）：0–4左腿（髋部偏航、髋部滚动、髋部俯仰、膝盖、
  脚踝），5–8颈部/头部（颈部俯仰、头部俯仰、头部偏航、头部滚动），
  9–13右腿。
- 导出器将观察归一化器烘烤到ONNX图中—始终
  部署由`scripts/export.py`生成的ONNX，永远不要手动转换
  检查点，否则策略在运行时看到未归一化的观察。

[AGENTS.md](AGENTS.md)记录了环境构建工作流程和在整个项目中学到的奖励设计
规则（也针对在该仓库中工作的AI编码代理）。

## 测试

```bash
uv run --with pytest pytest tests/
```

CPU唯一的配置不变和奖励函数回归测试—它们锁定
关节索引映射、奖励符号惯例和NaN守卫。

## 相关项目

- [microduck](https://github.com/pollen-robotics/microduck) — MicroDuck项目主页，包括运行导出策略的船上运行时
- [mjlab](https://github.com/mujocolab/mjlab) — 训练框架（MuJoCo Warp + rsl_rl）
- [BAM](https://github.com/Rhoban/bam) — 更好的执行器模型，由Rhoban提供

## 许可证

该项目在Apache 2.0许可证下获得许可。详见[LICENSE](LICENSE)文件。
3D模型文件在Creative Commons BY-SA-NC下获得许可。
