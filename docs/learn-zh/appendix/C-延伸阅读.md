# 附录 C · 延伸阅读

按"读完本书哪一部之后适合看"分组。

## 第 1 部之后（数学直觉）
- **3Blue1Brown**《微积分的本质》《线性代数的本质》（B 站有官方中文字幕）——用动画把导数、链式法则、矩阵乘法讲成可以"看见"的东西。每集 10–20 分钟，强烈推荐配合第 1–3 章。
- Khan Academy 微积分 / 线性代数 / 概率入门：想做题巩固时用。

## 第 2 部之后（神经网络）
- 3Blue1Brown《深度学习》系列（4 集）：神经网络、梯度下降、反向传播的动画版。
- PyTorch 官方教程 "Learn the Basics"：tensor、autograd、`nn.Module` 的最小实践。

## 第 3 部之后（强化学习）
- Sutton & Barto《Reinforcement Learning: An Introduction》（第 2 版，有中文译本）：RL 的圣经。第 3 章（MDP）、第 6 章（TD）、第 13 章（策略梯度）对应本书第 9–11 章。
- OpenAI **Spinning Up in Deep RL**（spinningup.openai.com）：policy gradient 推导、GAE、PPO 各有一页讲清楚，附代码。本书第 11–13 章的推导顺序参考了它。
- 论文：Schulman et al. 2017, *Proximal Policy Optimization Algorithms*（PPO 原文，只有 12 页）；Schulman et al. 2016, *High-Dimensional Continuous Control Using Generalized Advantage Estimation*（GAE 原文）。
- rsl_rl 源码：`.venv/lib/python3.12/site-packages/rsl_rl/algorithms/ppo.py`，不到 500 行，读完第 13 章后可以通读。

## 第 4 部之后（机器人与 sim2real）
- mjlab 文档与源码（`.venv/lib/python3.12/site-packages/mjlab/`），尤其 `managers/` 目录：六个 Manager 的实现。
- MuJoCo 文档 "Computation" 一章：四元数、坐标系、接触求解器。
- Rhoban **BAM**（Better Actuator Models）论文与仓库：第 17 章电机方程的来源。
- 本仓库 `docs/single-leg-stand-roadmap-zh.md`：把本书的原理用到一个新任务上的完整实操。

## 四元数（第 2 章 §2.4 承诺的补充）
- 3Blue1Brown 与 Ben Eater 合作的交互式四元数可视化：eater.net/quaternions。
- 只需记住：四元数 $q = (w, x, y, z)$，$\|q\|=1$；旋转向量 $v$ 是 $q\,v\,q^{-1}$；MuJoCo 用 (w, x, y, z) 顺序。


## 本轮详解的算法核对入口

- [OpenAI Spinning Up：PPO-Clip](https://spinningup.openai.com/en/latest/algorithms/ppo.html)：对照第 13 章，尤其是“移除过度改变的激励”不等于硬约束。该页示例使用 KL 提前停止，本项目用 KL 自适应学习率，不能混写实现。
- 本地 `rsl_rl/algorithms/ppo.py`：对照 `act`、`process_env_step`、`compute_returns`、`update`。本书第 12 章专门区分超时自举的理论形式与安装版本的实际近似。
- 本地 `mjlab/actuator/actuator.py` 的 `ActuatorCfg`、`mjlab/entity/entity.py` 的 `_apply_actuator_controls` 与环境的 decimation 循环：对照第 17 章，确认执行器延迟按物理步计数。
