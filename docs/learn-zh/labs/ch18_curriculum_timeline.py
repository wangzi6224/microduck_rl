"""第 18 章实验：从真实的 env cfg 里读出全部课程项，画成一张时间线；验证“步 = 迭代 × 24”。

运行：uv run python docs/learn-zh/labs/ch18_curriculum_timeline.py
纯 CPU（只构造 cfg，不建环境）。
"""

import numpy as np

from _common import banner, check, done, savefig, table, use_headless_matplotlib

# ---------------------------------------------------------------------------
banner("1. 构造 velocity 任务的 cfg，列出课程项")
from mjlab_microduck.tasks.microduck_velocity_env_cfg import NUM_STEPS_PER_ENV, make_microduck_velocity_env_cfg  # noqa: E402

cfg = make_microduck_velocity_env_cfg()
print("NUM_STEPS_PER_ENV =", NUM_STEPS_PER_ENV, "（每次迭代每个环境走的步数）")
for name, term in cfg.curriculum.items():
    print(f"  {name:22s} func={term.func.__name__}")
check("有 7 个课程项", len(cfg.curriculum) == 7)

# ---------------------------------------------------------------------------
banner("2. 权重课程：action_rate_l2 和 head_pose_bias 的阶梯")
series = {}
for name in ("action_rate_weight", "head_pose_bias_weight"):
    stages = cfg.curriculum[name].params["weight_stages"]
    rows = [[s["step"], s["step"] // NUM_STEPS_PER_ENV, s["weight"]] for s in stages]
    print(f"\n  {name}  (reward_name = {cfg.curriculum[name].params['reward_name']})")
    table(["环境步 step", "= 迭代", "权重"], rows)
    series[name] = [(s["step"] // NUM_STEPS_PER_ENV, s["weight"]) for s in stages]
check("action_rate 权重最终 −1.0", series["action_rate_weight"][-1][1] == -1.0)
check("head_pose_bias 在 600 次迭代前权重为 0", series["head_pose_bias_weight"][0] == (600, 1.0) or series["head_pose_bias_weight"][0][1] == 0.0)

# ---------------------------------------------------------------------------
banner("3. 范围课程：standing_envs、com_range、head_com_range、head_pose_range")
def stages_of(name, key):
    p = cfg.curriculum[name].params
    for k in p:
        if k.endswith("stages"):
            return p[k], key
    raise KeyError(name)

for name, key in (("standing_envs", "rel_standing_envs"), ("com_range", None), ("head_com_range", None)):
    stages, _ = stages_of(name, key)
    print(f"\n  {name}")
    rows = []
    for s in stages:
        val = s.get(key) if key else {k: v for k, v in s.items() if k != "step"}
        rows.append([s["step"], s["step"] // NUM_STEPS_PER_ENV, str(val)])
    table(["环境步", "= 迭代", "取值"], rows)

stages, _ = stages_of("head_pose_range", None)
print("\n  head_pose_range（4 个头部关节的命令范围，弧度）")
rows = []
for s in stages:
    rng_ = s.get("ranges") or {k: v for k, v in s.items() if k != "step"}
    rows.append([s["step"] // NUM_STEPS_PER_ENV, str(rng_)[:90]])
table(["迭代", "范围"], rows)

# ---------------------------------------------------------------------------
banner("4. 画时间线")
plt = use_headless_matplotlib()
fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
it_max = 2500
def step_plot(ax, pts, label):
    xs, ys = [], []
    for i, (it, v) in enumerate(pts):
        nxt = pts[i + 1][0] if i + 1 < len(pts) else it_max
        xs += [it, nxt]; ys += [v, v]
    ax.plot(xs, ys, drawstyle="steps-post", label=label)
step_plot(axes[0], series["action_rate_weight"], "action_rate_l2 权重")
step_plot(axes[0], series["head_pose_bias_weight"], "head_pose_bias 权重")
axes[0].set_ylabel("权重"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
st, _ = stages_of("standing_envs", "rel_standing_envs")
step_plot(axes[1], [(s["step"] // NUM_STEPS_PER_ENV, s["rel_standing_envs"]) for s in st], "站立环境比例")
axes[1].set_ylabel("比例"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
for name in ("com_range", "head_com_range"):
    st, _ = stages_of(name, None)
    pts = []
    for s in st:
        v = [v for k, v in s.items() if k != "step"][0]
        v = v[1] if isinstance(v, (tuple, list)) else v
        pts.append((s["step"] // NUM_STEPS_PER_ENV, v))
    step_plot(axes[2], pts, name + " (±m)")
axes[2].set_ylabel("质心偏移"); axes[2].set_xlabel("迭代"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
axes[0].set_title("velocity 任务的课程时间线（横轴 = 迭代，环境步 ÷ 24）")
savefig(fig, "ch18_curriculum_timeline")

done()
