# 改写 TODO：全书已完成，这里留作后续维护的说明书

> 2026-09-23：全书 19 章 + 附录按第 2 章的标准改写完毕。这份文件从"接着做"的清单，变成"以后再改教材时照着做"的说明书。
> 配套：[改写台账](改写台账.md)（每章状态、旧 → 新小节对照、承诺表、施工须知）和 [编写规范](编写规范.md)（标准全文 + 自检清单）。
> 这两份文件要不要留在仓库里，由用户定。

## 0. 完成情况

| 章 | 状态 |
|---|---|
| 第 2 章 | 范文，全程一字未改 |
| 第 1、3–19 章 | **全部已验收**：作者改写 → 主编机械检查 → 独立评审兼修订 → 主编验收 → 同步附录 / README |
| 附录 A / B / D、README | 已按各章的新小节号同步 |
| 第 5–14 章 | **另做过一轮 max 档深度复审（2026-09-23/24）**：逐章验算手算、核实源码、补图补手算补自测、拆密节、修接缝。详见改写台账的「深度复审」一节 |

验证（2026-09-23）：`tests/test_learn_docs.py` + `tests/test_learn_labs.py` 全量 **220 passed、2 skipped、0 failed**（跳过的是没有 GPU 时的两个 GPU 实验）；
第 2 章与 `src/` 未被改动；本轮改动全部提交在 `develop`（上游 `fork/zh-docs`）。

## 1. 剩下的事

1. `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` 里 `COM_RANDOMIZATION_RANGE` 旁的注释写 "ramped to ±8mm"，实际课程到 ±15 mm。不归教材管，本轮没动——要不要顺手改，由用户定。
2. 以后改教材：照第 2–5 节的标准、流程和提示词模板做；改完跑 `uv run --with pytest pytest tests/test_learn_docs.py tests/test_learn_labs.py -q`。
3. 改了某一章的小节号，记得同步附录 A/B、附录 D 和 README 里指向它的引用（检查器会报附录 B 的错位，附录 A 和 README 要人工核对）。

## 2. 标准是什么（摘要；全文见编写规范，范文是第 2 章）

**读者**：做过 Web 前端、数学停在初中、没读过 torch 代码，一个人自学。**用户的要求**：省 token，但以内容高效优质为主——手算、讲解图、读图段、自测、check 一样都不能为省调用而砍。

**章骨架（顺序固定）**：`# 第 N 章 · 标题：副标题` → 开篇引用块四件套（上一章有了什么 / 这一章要补什么 / 本章新词 / 需要的前提）→ `## N.0 先看地图`（一句加粗主旨、生活场景、总览图、概念表、易混词先贴标签、第一遍怎么读）→ 正文各节 → `## 📍 映射到项目`（开场白 `> 只需看一眼。语法看不懂没关系。`；先中文说再贴逐行注释的代码；写文件名 + 函数名，不写行号；代码行要能在源文件里原样找到）→ `## 🧪 动手实验`（编号清单 = 实验的整数 banner；三条"改一改"，先预测再运行、不泄底，实验里用 `# TWEAK-k:` 标记）→ `## 本章小结` → `**下一章**` 链接。

**每节节奏**：大白话 + 生活例子 → 一个小数字手算 → 才给名字、公式、`| 符号 | 怎么读 | 就是 |` 表 → 回到实验验证（"实验第 K 节"）。每个 H2 里 `**中文**（English）` 式的主词不超过 2 个；先小后大并明说"这是迷你版"；进阶和补课放折叠。

**全章下限**：折叠自测至少 3 处（`> **停一下，自测**` + 折叠答案）；本章真实存在的每个经典坑各有一个 ⚠️；长章给歇脚点。

**图**：课堂式讲解图（INK / MUTED / BLUE / GREEN / ORANGE 配色，①②③ 分步，手算写进图里，标题就是结论，中文标注带单位）；视觉量成比例；要读的数落在刻度上或单独标出；alt 是完整描述句；图后 3 行内有读图段；图里的数和 check 同源。

**实验**：整数 `banner("K. …")` 顺序 = 正文顺序；正文引用的每个数都有 `check`；随机数固定种子并对照理论值；无语言标记的代码块 = 实验原样输出；注释不写源码行号；结尾 `done()`；映射块的代码行用 `lines_in_order` 核对。

**几条评审反复抓到的规矩**：
- 字面值重算：写成等式的手算（含自测答案），拿印出来的已舍入的数重算也必须成立。
- 字母或词中途换含义要 ⚠️ 或换字母（交稿前列"字母 × 小节 × 含义"表自查）。
- 不先用后讲；前面章节讲过的回指小节号、不重教（回指前 grep 确认那一节真讲了）。
- 带小节号的引用只能指向本章或更早定稿的章（检查器按 WAVE 判）；指向后面的章只写"第 N 章（概念名）"。
- 项目事实（文件、函数、常数）逐条对源码核实；第 4 部的项目常数由实验从 env cfg 读出并 check。

**事实与术语（各章不得自行"统一"）**：197,774 = actor MLP 旋钮数（也是 ONNX 里的）；+14 个 σ = 197,788；检查点另含 critic 和 Adam 的两本账。归一化是主名。"偏差"有三义要分清：指令 − 实际（第 1–2 章）、结果 − 平均（第 4 章 4.4 节）、统计意义的 bias（第 12 章 12.5 节）；网络里的 b 叫"偏置"。log-derivative trick 叫"对数导数技巧"。PPO 的比率记作 ρ。

## 3. 流程与验收清单

**流水线**（每章）：全新作者代理 → 主编机械检查 → 主编同步附录 / README → 全新评审兼修订代理（不看旧稿）→ 主编验收 → 改写台账 → 快照。派评审前先用 `ListAgents` 确认作者已收工。

**主编机械检查 / 验收**（`$S/accNN` 是一次性目录）：

```bash
LEARNZH_FIG_DIR=$S/accNN uv run python docs/learn-zh/labs/chNN_*.py > $S/accNN/out.txt 2>&1   # 0 个 ✗
for f in $S/accNN/chNN_*.png; do cmp -s $f docs/learn-zh/figures/$(basename $f) || echo DIFF $f; done
uv run --with pytest python tests/test_learn_docs.py chNN --lab-output $S/accNN/out.txt       # 必须修 0
uv run --with pytest pytest tests/test_learn_labs.py -q -p no:cacheprovider -k chNN            # 含 TWEAK 测试
```

再逐张看改过的图、抽读几段、手算几个关键数。附录 B 的节号按"本章第一次加粗介绍它"的那一节登记（检查器会报错位）。

**硬规则**：子代理不跑任何改状态的 git 命令（add / commit / stash / checkout / restore / clean / reset / rm）；提交只在用户要求时由主编做；第 2 章（正文、实验、图）不改；`labs/worked_examples.py` 冻结；共享文件（README、appendix/、tests/、编写规范、改写台账、`_common.py`、`_draw.py`、任务书）只由主编改，子代理只改本章的 `.md`、`labs/chNN_*.py`、`figures/chNN_*.png`；验证性运行一律带 `LEARNZH_FIG_DIR`；子代理不跑 pytest。

**用量上限**：会打断代理（09-22 已发生五次）。成果都在磁盘上：先看文件修改时间、带 `LEARNZH_FIG_DIR` 复跑实验、跑检查器，弄清停在哪，再用 SendMessage 让同一个代理续做（附上磁盘状态）。同时跑的代理越多，上限来得越快。

## 4. 代理提示词模板

**作者**（每章一个全新代理）：

```text
你是中文教材《从零到 PPO：跟着 Microduck 学机器学习》第 NN 章（标题）的作者。你是全新代理，只写这一章。
仓库 /home/joyin/Projects/microduck_rl，教材在 docs/learn-zh/。下面 S = <本会话 scratchpad>
按顺序读：1. S/briefs/COMMON.md（通用任务书，全部照做）2. docs/learn-zh/编写规范.md（全文一次）
3. S/briefs/EXEMPLAR.md 4. S/briefs/chNN.md（本章任务书，末尾"补充"是定稿章节号速查）5. S/brief_chNN_inputs.md（冻结输入）
然后读旧稿（本章 .md 和 labs/chNN_*.py）。草稿和运行输出放 S/chNN/。
另外几位作者正在同时改第 X、Y 章：不要碰它们的文件，也不要带小节号引用它们。
不要运行任何改状态的 git 命令；看改动用 git status 或 git diff HEAD。
用户的要求：质量第一，讲清、讲准；在这个前提下不浪费 token。你可能被用量上限打断，成果都在磁盘上。
完成后按 COMMON.md"交稿报告"的 10 项交回，不超过 60 行。
```

**评审兼修订**（每章一个全新代理）：

```text
你是……第 NN 章的独立评审，同时负责把评审意见改掉。你是全新代理。
先读 S/briefs/REVIEWER.md（全部照做），再读 docs/learn-zh/编写规范.md 和 S/briefs/EXEMPLAR.md。
被评审的是：本章 .md、labs/chNN_*.py、figures/chNN_*.png。不要看这一章的旧稿（不要 git show / git diff 历史版本）。
评审意见写进 S/chNN/REVIEW.md，运行输出放 S/chNN/rev/。小节号不许动（主编已同步附录和 README）。
同时在改的章：不碰、不带小节号引用。
主编请你特别核对：（从作者交稿报告的"评审指引"和"不够好的地方"里挑 5–7 条，写具体数字和源码位置）
质量第一；完成后按 REVIEWER.md 的"交回"格式交回，不超过 50 行。
```

## 5. 文件在哪

- 仓库里：标准 `docs/learn-zh/编写规范.md`；状态 `docs/learn-zh/改写台账.md`；检查器 `tests/test_learn_docs.py`（文件开头的 REWRITTEN、PENDING_SYNC、B_SYNCED、WAVE 四个变量就是波次状态）；实验测试 `tests/test_learn_labs.py`；公共实验工具 `docs/learn-zh/labs/_common.py`、`_draw.py`。
- 持久目录 `~/.claude/projects/-home-joyin-Projects-microduck-rl/learn-zh-rewrite/`：
  - `briefs/`：COMMON.md、EXEMPLAR.md、REVIEWER.md（精简版，现行）、COMMON_v1.md、REVIEWER_v1.md（旧版）、ch01–ch19 各章任务书（含未开工的 ch16、ch18、ch19）。
  - `brief_chNN_inputs.md`：已生成的冻结输入（第 1、3–15、17 章）；第 16、18、19 章的要等第 13、14 章定稿后再生成。
  - `reviews/`：已完成的评审意见。`gpu_archive/`：GPU 实验输出存档（第 14、15 章）。`drafts/`：第 13、14 章作者的笔记和中间产物。
  - `snapshot-latest/`：最新快照（`07-handoff`，docs/learn-zh 与 tests 的整份拷贝）。
- 批准的总计划：`~/.claude/plans/home-joyin-projects-microduck-rl-docs-l-foamy-sloth.md`（含逐章要点）。
