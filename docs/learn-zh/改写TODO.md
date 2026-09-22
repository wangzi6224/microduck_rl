# 改写 TODO：下次从哪开始、标准是什么

> 2026-09-22 用户要求收尾时写。下次开新会话接着做：**先读这份**，再读 [改写台账](改写台账.md)（每章状态、旧 → 新小节对照、承诺表、施工须知）和 [编写规范](编写规范.md)（标准全文 + 自检清单）。
> 全部做完后，这份文件和改写台账都可以删掉（先问用户）。

## 0. 现在在哪

| 章 | 状态 |
|---|---|
| 第 2 章 | 范文，不改 |
| 第 1、3–12、15、17 章 | **已验收**（作者 → 评审兼修订 → 主编验收），附录 A/B 与 README 里指向它们的小节号已同步 |
| 第 13 章 PPO | **作者稿完成，待评审**（588 行、5 张图、72 处 check，机械检查全过） |
| 第 14 章 训练回路全貌 | **作者稿基本完成，待评审**（740 行、6 张图；CPU 伴生实验 80 处 check 全过、TWEAK 测试通过、检查器 0）。作者在最后跑 TWEAK 时被停下，**没交稿报告** |
| 第 16、18、19 章 | **未开工**，任务书已写好 |
| 阶段 3（全书收尾） | 未开始 |

收尾时的验证：全量 pytest 204 passed、2 skipped、3 xfailed（xfail 就是还没改写的第 16、18、19 章）；第 2 章与 `src/` 未被改动。快照 `07-handoff`。09-22 按用户要求全部提交到`develop` 分支（上游是 `fork/zh-docs`）。

## 1. 下一次按这个顺序做

### 1.1 开工前（每次新会话都要做）

1. `ListAgents` 确认没有别的会话在当主编（同一时间只能有一个主编会话，见改写台账"施工须知"）。
2. scratchpad 每个会话不同、会话结束就没了。先把持久目录里的材料拷进新会话的 scratchpad（下面记作 `$S`）：

```bash
P=~/.claude/projects/-home-joyin-Projects-microduck-rl/learn-zh-rewrite
cp -r $P/briefs $P/gpu_archive $P/drafts $S/ && cp $P/brief_ch*_inputs.md $S/
```

3. 看一眼当前分支和 `git status`：09-22 的成果已按用户要求提交在`develop` 分支（上游是 `fork/zh-docs`）。在这个分支上接着做；新的改动先留在工作区，用户说提交再提交。

### 1.2 第 13 章：派评审兼修订

- 提示词用第 4 节的评审模板。"主编请你特别核对"填这几条（出自作者的交稿报告）：
  1. 比率全章记作 ρ（论文写 r_t(θ)，和奖励 r 撞字母），正文、图、实验、附录 A 要一致。
  2. "首次更新时比率只是接近 1"那个折叠：归一化器在采集中每一步都在更新（rsl_rl `ppo.py` 里调用 `update_normalization`），演示用的是 3 个输入的小网络，只是定性结论，正文要说清。
  3. 全章"一步"只指环境步，拧旋钮一律叫"更新"。
  4. 冻结的四格 2.4 / 1 / −1.6 / −3；一条记录的总账 −2.58865；比率 1.0833 = exp(0.08)；KL 0.005 → 0.02；价值裁剪的三种 V 手算——逐个验算。
  5. KL 自适应规则的上下限 1e-5 / 1e-2 和 `kl_mean > 0` 条件；📍 的代码与 rsl_rl 5.0.1 原样一致（含 `if self.gpu_global_rank == 0:`）。
  6. "日志里 Mean surrogate loss 常是小负数"是推理加构造例子（−0.044）加第 14 章冒烟日志，要标清口径。
- 同时在改的章（第 14 章）只写到章，不带小节号。

### 1.3 第 14 章：先补对照表，再派评审兼修订

- 作者没交报告：**旧 → 新小节对照由主编自己补**（评审不看旧稿）。旧大纲用 `git show 64faf9b:docs/learn-zh/part3-rl/14-训练回路全貌.md | grep '^## '`，新大纲已记在改写台账的对照表里。作者的草稿和笔记在 `$S/drafts/ch14/`。
- 检查器的 `--lab-output` 用"GPU 存档 + CPU 输出"拼起来：`cat $S/gpu_archive/ch14_smoke_train.out <CPU 输出> > all.txt`。GPU 实验打印的内容已由主编定稿（作者只改了 docstring），**不要重跑 GPU 实验去替换存档**，否则正文里的原样输出块会对不上。
- 评审特别核对：
  1. 14.1 的四只钟要和第 17 章 17.5 节的"物理步 / 环境步"接上，不重教。
  2. 14.5 的 Episode_Reward 单位：0.5 × 2 × 0.02 × 1000 ÷ 20 = 1；只有 5 s 的回合是 0.25（worked_examples 冻结）；"除以 20 秒"对 mjlab 源码核实。
  3. 14.6 符号约定的 ⚠️ 和 AGENTS.md 一致（返回 ≥ 0 的代价配负权重；自带负号的 `*_penalty` / `*_l1` 配正权重）。
  4. "4096 × 24 是真乘法（记录条数），4096 × 61 是形状"的 ⚠️。
  5. 14.9 检查点里有什么（actor、critic、Adam 的两本账；197,774 / 197,788 的口径）。
  6. 14.10：冒烟测试只证明"能跑通"，不证明"学得会"。
  7. 第 13 章对本章的承诺（surrogate 常为小负数；rsl_rl 不记平坦区比例）。
- 附录 B 可补登本章新词：仿真时间、墙上时间 §14.1；采集 §14.2；符号约定 §14.6（先确认它们在那一节第一次加粗出现）。

### 1.4 D 波收尾（第 13、14 章都验收之后）

1. `tests/test_learn_docs.py` 的 WAVE 按定稿顺序拆开（例如 13: 10、14: 11），再给第 14 章指回第 13 章的"第 13 章（概念名）"补小节号。
2. 改写台账：状态行、对照表、承诺表（标"已兑现"）、同步记录。
3. 快照：`docs/learn-zh` 和 `tests/test_learn_*.py` 整份拷到 `$S/snapshots/07-waveD-accepted/`，再覆盖持久目录的 `snapshot-latest/`。

### 1.5 E 波：第 16、18、19 章

1. 开工前：三章加进 `REWRITTEN` 和 `PENDING_SYNC`，WAVE 取比 D 波更大的数（三章同一个数）。
2. 给 `briefs/ch16.md`、`ch18.md`、`ch19.md` 末尾各补一节"定稿章节号速查"（第 13、14 章的小节号；做法照 `ch13.md` 末尾的样子）。
3. 生成冻结输入：`uv run --with pytest python tests/test_learn_docs.py --brief ch16 > $S/brief_ch16_inputs.md`（18、19 同理）。
4. 用第 4 节的作者模板同时派三位作者；**同时跑的代理不超过 3 个**。
5. 各章注意：
   - 第 16 章：实验的 `--env` 小节要 GPU，保留为可选；CPU 部分必须独立 0 个 ✗。
   - 第 19 章：实验固定用运行目录 `logs/rsl_rl/velocity/2026-09-22_13-21-10_learnzh-ch14`（被 gitignore；若不在了，重跑一次第 14 章 GPU 实验生成新的，并改任务书里的路径）；以 `## 全书结束` 收尾（成文例外）。
   - 第 18 章："四只钟"回指第 14 章，不重教。
6. 每章交稿后照第 3 节流程：机械检查 → 同步附录 / README → 评审兼修订 → 验收 → 快照 `08-waveE-accepted`。

### 1.6 阶段 3：全书收尾

1. 附录 A/B 按各章清单重建：补齐第 16、18、19 章的条目；附录 A 里仍指旧号的行改正（`uv run --with pytest python tests/test_learn_docs.py chNN` 会报附录 B 的错位；附录 A 靠人工核对）。
2. 附录 D 指向第 16、17 章的 § 引用改号（§16.6 / §16.10 等要按第 16 章的新大纲改）。
3. README：讲法顺序改成"手算先于符号"；代码引用约定改成"文件名 + 函数名，不写行号"；目录表；卡点表剩余行（第 16、18、19 章）；新写一段带日期的"本轮校验范围"（如实写跑了什么、没跑什么）；链到编写规范。
4. 全量验证：`uv run --with pytest pytest tests/test_learn_docs.py tests/test_learn_labs.py -q -p no:cacheprovider`（约 4 分钟，有 GPU 时会跑 GPU 实验）；第 2 章未改：`git diff --quiet 64faf9b -- docs/learn-zh/part1-math/02-向量与矩阵.md docs/learn-zh/labs/ch02_vectors.py 'docs/learn-zh/figures/ch02_*.png'`。
5. 给用户写最终报告；问改写台账、这份 TODO 留不留。

### 1.7 等用户决定

1. （已解决）09-22 暂存区被整体 `git add` 的事：用户随后要求全部提交，已一并提交。
2. `src/` 里 `COM_RANDOMIZATION_RANGE` 旁的注释写"ramped to ±8mm"，实际课程到 ±15 mm。不归教材管，没动。

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
