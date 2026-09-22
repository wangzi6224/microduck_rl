"""Structural checks for the Chinese textbook under docs/learn-zh/.

Chapter 2 (part1-math/02-向量与矩阵.md) is the gold standard.  Every rule below is
calibrated so that chapter 2 passes UNMODIFIED; docs/learn-zh/编写规范.md is the prose
version of the same standard.

Two tiers:
  U (universal)  every file, always: balanced fences / <details> / $$, links and images
                 exist, every PNG is referenced and has exactly one savefig source,
                 "实验第 K 节" and the numbered lab list match the lab's integer banners,
                 every section reference (N.M 节 / §N.M / 第 N 章 N.M / 附录 D.k) resolves.
  S (strict)     only chapters in REWRITTEN: skeleton, N.0 map, end blocks, folded
                 self-tests, figure alt + reading guide, new-term density, and the wave
                 rule for cross-chapter section references.  Chapters not yet rewritten
                 are xfail(strict=True): the day one passes by accident it must be moved
                 into REWRITTEN.

Readable per-chapter report (hard + soft findings), for authors:
    uv run python tests/test_learn_docs.py ch03 [ch04 ...] [--lab-output FILE]
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BOOK = REPO / "docs" / "learn-zh"
LABS = BOOK / "labs"
FIGS = BOOK / "figures"

# --- rewrite state (edit as chapters move through the pipeline) ---------------------
ALL_CHAPTERS = set(range(1, 20))
REWRITTEN: set[int] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17}   # held to the strict tier
PENDING_SYNC: set[int] = set()   # rewritten, but README/appendix refs to them not remapped yet
B_SYNCED: set[int] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17}  # appendix B "首次出现" rows rebuilt for these chapters
WAVE: dict[int, int] = {2: 0, 3: 1, 1: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 3, 9: 4, 10: 5, 15: 6, 11: 7, 12: 8, 17: 9, 13: 10, 14: 10}   # a section-numbered ref X→Y (X≠Y) needs WAVE[Y] < WAVE[X]

# --- waivers: keyed by file name + substring, never by line number -------------------
WAIVERS = {
    "fig-guide": {"ch02_axis_rotation.png", "ch02_axes.png"},
    "lab-list": set(),
    "map-opener": {12: "现在可以逐行读", 13: "现在可以逐行读"},
    "closer": {19: "## 全书结束"},
    # worked_examples literals allowed to be absent from the prose (none left)
    "echo": set(),
}

# --- regexes --------------------------------------------------------------------------
FENCE = re.compile(r"^\s*(?:>\s*)*(```|~~~)(.*)$")
SEC = r"(?:1[0-9]|[1-9])\.\d{1,2}(?![\d.])"
R1 = re.compile(rf"第\s*(?P<ch>\d{{1,2}})\s*章(?:的)?\s*(?P<list>{SEC}(?:\s*[–、]\s*{SEC})*)(?:\s*节)?")
R2 = re.compile(rf"(?<![\d.§])(?P<list>{SEC}(?:\s*[–、]\s*{SEC})*)\s*节")
_ITEM = r"\d{1,2}\.\d{1,2}"
R3 = re.compile(rf"§\s*(?:{_ITEM}|\d{{1,2}})(?:\s*(?:–\s*§?\s*{_ITEM}|[、,/]\s*§\s*(?:{_ITEM}|\d{{1,2}})))*")
R4 = re.compile(r"第\s*(?P<k>\d{1,2})\s*节")
R5 = re.compile(r"附录\s*(?P<app>[A-D])\.(?P<s>\d{1,2})")
H1 = re.compile(r"^# 第 (?P<n>\d{1,2}) 章 · (?P<t>[^：]+)(?:：(?P<sub>.+))?$")
H2 = re.compile(r"^## (?P<c>1[0-9]|[1-9])\.(?P<s>\d{1,2}) (?P<title>\S.*)$")
H2_APP = re.compile(r"^## (?P<app>[A-D])\.(?P<s>\d{1,2}) ")
BAN = re.compile(r"^\s*banner\(\s*f?[\"'](?P<k>\d+)(?P<sub>[a-z]?)\.")
SAVEFIG = re.compile(r"savefig\(\s*[\w.]+\s*,\s*f?[\"'](?P<name>[^\"']+)[\"']")
IMAGE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<target>[^)\s]+)\)")
LINK = re.compile(r"(?<!!)\[[^\]]*\]\((?P<target>[^)\s]+)\)")
GUIDE = re.compile(r"^(?:>\s*)?(?:\*\*|#{3,4}\s).{0,14}(?:读图|怎么读|跟着图|看图)")
NEW_TERM = re.compile(r"\*\*[^*\n]+\*\*（[A-Za-z][^）\n]*）")
LAB_PATH = re.compile(r"labs/((?:ch\d\d|appendix)_\w+\.py)")
OPENER_PREV = re.compile(r"(上一章|上一部|前五章|第 \d 部)我们有了什么")
OPENER_GOAL = re.compile(r"(这一章要补什么|这一部要补什么|第 \d 部要做什么)")


# --- document model ---------------------------------------------------------------------
@dataclass
class Doc:
    path: Path
    raw: list[str]
    masked: list[str] = field(default_factory=list)  # fenced code blanked out
    prose: list[str] = field(default_factory=list)   # masked, minus math / inline code / link targets
    fences: list[tuple[str, int, int]] = field(default_factory=list)  # (lang, first, last) 0-based
    unbalanced_fence: bool = False

    @property
    def name(self) -> str:
        return self.path.name


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO))


@lru_cache(maxsize=None)
def load(path: Path) -> Doc:
    raw = path.read_text(encoding="utf-8").splitlines()
    doc = Doc(path=path, raw=raw)
    masked, marker, start, lang = [], None, 0, ""
    for i, line in enumerate(raw):
        m = FENCE.match(line)
        if marker is None:
            if m:
                marker, start, lang = m.group(1), i, m.group(2).strip()
                masked.append("")
            else:
                masked.append(line)
        else:
            masked.append("")
            if m and m.group(1) == marker and not m.group(2).strip():
                doc.fences.append((lang, start, i))
                marker = None
    doc.unbalanced_fence = marker is not None
    doc.masked = masked
    text = "\n".join(masked)
    text = re.sub(r"\$\$.*?\$\$", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    prose = []
    for line in text.split("\n"):
        line = re.sub(r"\$[^$\n]*\$", " ", line)
        line = re.sub(r"`[^`\n]*`", " ", line)
        line = re.sub(r"\]\([^)]*\)", "]", line)
        prose.append(line)
    doc.prose = prose
    return doc


@lru_cache(maxsize=None)
def chapter_files() -> dict[int, Path]:
    return {int(p.name[:2]): p for p in sorted(BOOK.glob("part*/[0-9][0-9]-*.md"))}


def appendix_files() -> list[Path]:
    return sorted((BOOK / "appendix").glob("*.md"))


def other_files() -> list[Path]:
    return [p for p in sorted(BOOK.glob("*.md"))]


def all_files() -> list[Path]:
    return list(chapter_files().values()) + appendix_files() + other_files()


def file_id(p: Path) -> str:
    for n, q in chapter_files().items():
        if p == q:
            return f"ch{n:02d}"
    return ("app" + p.name[0]) if p.parent.name == "appendix" else p.stem


def chapter_of(p: Path) -> int | None:
    for n, q in chapter_files().items():
        if p == q:
            return n
    return None


@lru_cache(maxsize=None)
def sections(ch: int) -> dict[int, tuple[int, str]]:
    """section number -> (0-based line, title) for the numbered H2s of a chapter."""
    out = {}
    for i, line in enumerate(load(chapter_files()[ch]).masked):
        m = H2.match(line)
        if m and int(m.group("c")) == ch:
            out[int(m.group("s"))] = (i, m.group("title"))
    return out


@lru_cache(maxsize=None)
def appendix_sections() -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for p in appendix_files():
        for line in load(p).masked:
            m = H2_APP.match(line)
            if m:
                out.setdefault(m.group("app"), set()).add(int(m.group("s")))
    return out


# --- references -------------------------------------------------------------------------
def _pair(tok: str) -> tuple[int, int]:
    c, s = tok.split(".")
    return int(c), int(s)  # never float: 4.10 is not 4.1


def _expand(listing: str) -> list[tuple[int, int]]:
    out = []
    for item in re.split(r"\s*、\s*", listing.strip()):
        ends = re.split(r"\s*–\s*", item)
        a = _pair(ends[0])
        if len(ends) == 1:
            out.append(a)
            continue
        b = _pair(ends[1])
        if a[0] != b[0] or b[1] < a[1]:
            out.append((-1, -1))  # malformed range
            continue
        out.extend((a[0], s) for s in range(a[1], b[1] + 1))
    return out


def section_refs(doc: Doc) -> list[tuple[int, int, int | None, str]]:
    """(0-based line, chapter, section-or-None, shown text) for every reference in prose."""
    refs = []
    for i, line in enumerate(doc.prose):
        work = line
        for m in R1.finditer(work):
            ch = int(m.group("ch"))
            for c, s in _expand(m.group("list")):
                refs.append((i, ch if c == ch else -1, s, m.group(0)))
        work = R1.sub(lambda m: " " * len(m.group(0)), work)
        for m in R2.finditer(work):
            for c, s in _expand(m.group("list")):
                refs.append((i, c, s, m.group(0)))
        for m in R3.finditer(work):
            prev = None
            for sep, tok in re.findall(r"([–、,/])?\s*§?\s*(\d{1,2}(?:\.\d{1,2})?)", m.group(0)):
                if "." not in tok:
                    refs.append((i, int(tok), None, m.group(0)))
                    prev = None
                    continue
                cur = _pair(tok)
                if sep == "–" and prev and prev[0] == cur[0] and cur[1] > prev[1]:
                    refs.extend((i, cur[0], s, m.group(0)) for s in range(prev[1] + 1, cur[1] + 1))
                else:
                    refs.append((i, cur[0], cur[1], m.group(0)))
                prev = cur
    return refs


def ref_problems(doc: Doc) -> list[str]:
    if doc.name in ("改写台账.md", "改写TODO.md"):
        return []  # the ledger records OLD → NEW section maps, so it necessarily names sections that no longer exist
    src = chapter_of(doc.path)
    problems = []
    for i, ch, sec, shown in section_refs(doc):
        where = f"{doc.name}:{i + 1} 「{shown.strip()}」"
        if ch not in ALL_CHAPTERS:
            problems.append(f"{where} 章号或范围写法不对")
            continue
        if src is None and ch in PENDING_SYNC:
            continue  # shared file, target chapter awaiting the renumbering pass
        if src is not None and src not in REWRITTEN and ch != src and ch in REWRITTEN and ch != 2:
            continue  # legacy chapter pointing at a rewritten one: fixed when it is rewritten
        if sec is not None and sec not in sections(ch):
            problems.append(f"{where} → 第 {ch} 章没有 {ch}.{sec} 节")
    for i, line in enumerate(doc.prose):
        for m in R5.finditer(line):
            if int(m.group("s")) not in appendix_sections().get(m.group("app"), set()):
                problems.append(f"{doc.name}:{i + 1} 「{m.group(0)}」 → 附录里没有这一节")
    return problems


def wave_problems(doc: Doc) -> list[str]:
    src = chapter_of(doc.path)
    problems = []
    for i, ch, sec, shown in section_refs(doc):
        if sec is None or ch == src or ch not in ALL_CHAPTERS:
            continue
        if ch not in REWRITTEN or WAVE.get(ch, 99) >= WAVE.get(src, 99):
            problems.append(
                f"{doc.name}:{i + 1} 「{shown.strip()}」 带小节号指向尚未定稿的第 {ch} 章——改写成「第 {ch} 章（概念名）」"
            )
    return problems


# --- labs and figures ---------------------------------------------------------------------
@lru_cache(maxsize=None)
def banners(lab: str) -> tuple[int, ...]:
    path = LABS / lab
    if not path.exists():
        return ()
    ks = {int(m.group("k")) for line in path.read_text(encoding="utf-8").splitlines() if (m := BAN.match(line))}
    return tuple(sorted(ks))


def block(doc: Doc, heading_prefix: str) -> tuple[int, int] | None:
    """(first, last+1) 0-based line span of the H2 block whose heading starts with the prefix."""
    start = None
    for i, line in enumerate(doc.masked):
        if line.startswith("## "):
            if start is not None:
                return start, i
            if line.startswith(heading_prefix):
                start = i
    return (start, len(doc.masked)) if start is not None else None


def lab_problems(doc: Doc) -> list[str]:
    labs = list(dict.fromkeys(LAB_PATH.findall("\n".join(doc.raw))))
    if not labs:
        return []
    problems = [f"{doc.name} 引用了不存在的实验 labs/{lab}" for lab in labs if not (LABS / lab).exists()]
    valid = {k for lab in labs for k in banners(lab)}
    for i, line in enumerate(doc.prose):
        for m in R4.finditer(line):
            if int(m.group("k")) not in valid:
                problems.append(f"{doc.name}:{i + 1} 「{m.group(0)}」 实验里没有这个整数小节（现有 {sorted(valid)}）")
    span = block(doc, "## 🧪 动手实验") or block(doc, "## 动手实验")
    if span and doc.name not in WAIVERS["lab-list"]:
        listed_labs = list(dict.fromkeys(LAB_PATH.findall("\n".join(doc.raw[span[0]:span[1]]))))
        expected = sum(len(banners(lab)) for lab in listed_labs)
        items = 0
        for line in doc.masked[span[0]:span[1]]:
            if line.startswith("**改一改**"):
                break
            items += bool(re.match(r"^\d+\. ", line))
        if items != expected:
            problems.append(f"{doc.name} 动手实验清单 {items} 条，实验整数 banner 共 {expected} 个")
    return problems


def balance_problems(doc: Doc) -> list[str]:
    text = "\n".join(doc.prose)  # prose: a `<details>` mentioned in inline code is not a tag
    problems = []
    if doc.unbalanced_fence:
        problems.append(f"{doc.name} 代码围栏没有成对")
    if text.count("<details") != text.count("</details>"):
        problems.append(f"{doc.name} <details> 开 {text.count('<details')} 闭 {text.count('</details>')}")
    if text.count("$$") % 2:
        problems.append(f"{doc.name} $$ 个数是奇数")
    return problems


def link_problems(doc: Doc) -> list[str]:
    problems = []
    for i, line in enumerate(doc.masked):
        for m in list(IMAGE.finditer(line)) + list(LINK.finditer(line)):
            target = m.group("target").split("#")[0]
            if not target or re.match(r"^[a-z]+://", target) or target.startswith("mailto:"):
                continue
            if not (doc.path.parent / target).exists():
                problems.append(f"{doc.name}:{i + 1} 链接目标不存在：{target}")
    return problems


def figure_inventory_problems() -> list[str]:
    referenced: dict[str, int] = {}
    for p in all_files():
        for line in load(p).masked:
            for m in IMAGE.finditer(line):
                name = Path(m.group("target")).name
                referenced[name] = referenced.get(name, 0) + 1
    sources: dict[str, list[str]] = {}
    for lab in sorted(LABS.glob("*.py")):
        for m in SAVEFIG.finditer(lab.read_text(encoding="utf-8")):
            sources.setdefault(m.group("name") + ".png", []).append(lab.name)
    problems = []
    for png in sorted(p.name for p in FIGS.glob("*.png")):
        if png not in referenced:
            problems.append(f"figures/{png} 没有任何正文引用")
        if len(sources.get(png, [])) != 1:
            problems.append(f"figures/{png} 的 savefig 来源应恰好 1 个，现在是 {sources.get(png, [])}")
    return problems


# --- strict tier ------------------------------------------------------------------------------
def strict_problems(ch: int) -> list[str]:
    doc = load(chapter_files()[ch])
    P: list[str] = []
    lines, name = doc.masked, doc.name

    m = H1.match(lines[0]) if lines else None
    if not m or int(m.group("n")) != ch:
        P.append(f"{name}:1 H1 应为「# 第 {ch} 章 · 标题：副标题」")
    elif not m.group("sub"):
        P.append(f"{name}:1 H1 缺副标题（「：」后的一句话）")

    head = "\n".join(lines[:25])
    for label, ok in [
        ("上一章/上一部我们有了什么", OPENER_PREV.search(head)),
        ("这一章要补什么", OPENER_GOAL.search(head)),
        ("本章新词", "本章新词" in head),
        ("需要的前提", "需要的前提" in head),
    ]:
        if not ok:
            P.append(f"{name} 开篇引用块缺「{label}」")

    secs = sections(ch)
    if sorted(secs) != list(range(len(secs))) or not secs:
        P.append(f"{name} 小节号应从 {ch}.0 起连续，现在是 {sorted(secs)}")
    if 0 in secs and not re.match(r"^先看地图[：:]", secs[0][1]):
        P.append(f"{name}:{secs[0][0] + 1} {ch}.0 的标题应以「先看地图：」开头")

    depth = 0
    for i, line in enumerate(doc.prose):
        depth += line.count("<details") - line.count("</details>")
        if depth > 0 and line.startswith("## "):
            P.append(f"{name}:{i + 1} <details> 里不能放 H2")

    h2 = [(i, line) for i, line in enumerate(lines) if line.startswith("## ")]
    tail = ["## 📍 映射到项目", "## 🧪 动手实验", "## 本章小结"]
    closer = WAIVERS["closer"].get(ch)
    got = [line for _, line in h2[-(len(tail) + bool(closer)):]]
    if got != tail + ([closer] if closer else []):
        P.append(f"{name} 章末 H2 应依次是 {tail + ([closer] if closer else [])}，现在是 {got}")

    span = block(doc, "## 📍 映射到项目")
    if span:
        first = next((line for line in lines[span[0] + 1:span[1]] if line.strip()), "")
        want = WAIVERS["map-opener"].get(ch)
        if not (first.startswith("> 只需看一眼。语法看不懂没关系。") or (want and want in first)):
            P.append(f"{name}:{span[0] + 2} 映射块应以「> 只需看一眼。语法看不懂没关系。」开场")
    span = block(doc, "## 🧪 动手实验")
    if span:
        body = lines[span[0]:span[1]]
        at = next((i for i, line in enumerate(body) if line.startswith("**改一改**")), None)
        if at is None or sum(line.startswith("- ") for line in body[at:]) < 3:
            P.append(f"{name} 「**改一改**」应是 3 条以上的列表（先预测再运行）")

    if not closer:
        last = next((line for line in reversed(lines) if line.strip()), "")
        nxt = f"{ch + 1:02d}-"
        if not any(Path(m.group("target")).name.startswith(nxt) for m in LINK.finditer(last)):
            P.append(f"{name} 最后一段应链接到第 {ch + 1} 章")

    for i, line in enumerate(lines):
        for m in IMAGE.finditer(line):
            png = Path(m.group("target")).name
            if len(m.group("alt")) < 12:
                P.append(f"{name}:{i + 1} {png} 的 alt 不足 12 字（要写成完整描述句）")
            following = [x for x in lines[i + 1:i + 12] if x.strip()][:3]
            if png not in WAIVERS["fig-guide"] and not any(GUIDE.match(x) for x in following):
                P.append(f"{name}:{i + 1} {png} 后 3 行内缺读图段（**这样读图：** / **读图顺序：** / ### 怎么读…）")

    P.extend(wave_problems(doc))

    for i, line in enumerate(doc.raw):
        if re.search(r"\.py:\d+", line):
            P.append(f"{name}:{i + 1} 不写源码行号（文件名 + 函数名即可）")
    for i, line in enumerate(lines):
        if re.match(r"^(?:>\s*)?\*\*自测\*\*", line):
            P.append(f"{name}:{i + 1} 自测要用「停一下」写法并把答案折叠")

    stops = [i for i, line in enumerate(lines) if "停一下" in line and re.match(r"^(?:>\s*|#{2,4}\s)", line)]
    if len(stops) < 3:
        P.append(f"{name} 折叠自测不足 3 处（现在 {len(stops)}）")
    for i in stops:
        if not any("<details" in x for x in lines[i:i + 13]):
            P.append(f"{name}:{i + 1} 「停一下」之后 12 行内没有折叠的答案")

    bounds = [i for i, _ in h2] + [len(lines)]
    for (i, title), end in zip(h2, bounds[1:]):
        n = len(NEW_TERM.findall("\n".join(lines[i:end])))
        if n > 2:
            P.append(f"{name}:{i + 1} 「{title[3:]}」一节里「**词**（English）」式新词 {n} 个，超过 2 个")
    return P


def soft_problems(ch: int, lab_output: str | None = None) -> list[str]:
    doc = load(chapter_files()[ch])
    W: list[str] = []
    for i, line in enumerate(doc.masked):
        for m in IMAGE.finditer(line):
            if len(m.group("alt")) < 20:
                W.append(f"{doc.name}:{i + 1} alt 偏短（{len(m.group('alt'))} 字）")
    if not any(lang.lower() in {"js", "javascript", "ts"} for lang, _, _ in doc.fences):
        W.append(f"{doc.name} 全章没有前端（js）类比代码块")
    for s, (i, title) in sections(ch).items():
        if "：" not in title and ":" not in title:
            W.append(f"{doc.name}:{i + 1} 标题「{title}」不是「概念：比喻/论断」形式")
    ks = [int(m.group("k")) for line in doc.prose for m in re.finditer(r"实验[^。\n]{0,24}?第\s*(?P<k>\d{1,2})\s*节", line)]
    first = list(dict.fromkeys(ks))  # later back-references to an earlier lab section are fine
    if first != sorted(first):
        W.append(f"{doc.name} 各实验小节第一次被提到的顺序不是递增的（实验顺序应与正文一致）：{first}")
    if lab_output is not None:
        out_lines = {x.rstrip() for x in lab_output.splitlines()}
        for lang, a, b in doc.fences:
            if lang:
                continue
            missing = [x for x in doc.raw[a + 1:b] if x.strip() and "..." not in x and x.rstrip().lstrip("> ") not in out_lines
                       and x.rstrip() not in out_lines]
            if missing:
                W.append(f"{doc.name}:{a + 1} 无语言标记的代码块有 {len(missing)} 行不在实验输出里（示意图请用 ```text），如「{missing[0].strip()}」")
    span = block(doc, "## 📍 映射到项目")
    if span:
        body = "\n".join(doc.raw[span[0]:span[1]])
        roots = [REPO, REPO / "src" / "mjlab_microduck", *sorted(REPO.glob(".venv/lib/python3.*/site-packages"))]
        sources = ""
        for rel in dict.fromkeys(re.findall(r"`([\w./-]+\.py)`", body)):
            for root in roots:
                if (root / rel).is_file():
                    sources += (root / rel).read_text(encoding="utf-8", errors="ignore")
                    break
        if sources:
            for lang, a, b in doc.fences:
                if span[0] <= a < span[1] and lang.lower() in {"python", "py"}:
                    for x in doc.raw[a + 1:b]:
                        code = x.split("#")[0].strip()
                        if len(code) > 8 and code != "..." and code not in sources:
                            W.append(f"{doc.name}:{a + 1} 映射块代码行在所引文件里找不到原样：{code}")
    return W


# --- appendix B: a term must be bold inside the section it is registered under -------------------
def appendix_b_rows() -> list[tuple[str, int, int]]:
    rows = []
    for p in appendix_files():
        if not p.name.startswith("B-"):
            continue
        for line in load(p).raw:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 4 or not (m := re.search(r"§\s*(\d{1,2})\.(\d{1,2})", cells[-1])):
                continue
            rows.append((cells[0], int(m.group(1)), int(m.group(2))))
    return rows


def appendix_b_problems(ch: int) -> list[str]:
    """A row is stale when the chapter bold-introduces the term in a different section first
    (a reused section number still "resolves", so reference checking alone cannot see this)."""
    doc = load(chapter_files()[ch])
    secs = sections(ch)
    order = sorted(secs)
    end_of_body = next((i for i, line in enumerate(doc.masked) if line.startswith("## 📍")), len(doc.raw))
    span = {}
    for k in order:
        later = [secs[j][0] for j in order if secs[j][0] > secs[k][0]]
        span[k] = (secs[k][0], min(later + [end_of_body]))
    problems = []
    for term, c, s in appendix_b_rows():
        if c != ch:
            continue
        if s not in secs:
            problems.append(f"附录 B「{term}」登记在不存在的 §{c}.{s}")
            continue
        variants = [v.strip() for v in re.split(r"[/、]|（[^）]*）", term) if v.strip()]
        # exact bold span only: "**梯度下降**" must not count as an introduction of "梯度"
        bold_in = [k for k in order if k != 0 and any(
            b.strip() in variants for b in re.findall(r"\*\*([^*\n]+)\*\*", "\n".join(doc.raw[slice(*span[k])])))]
        if bold_in and bold_in[0] != s:
            problems.append(f"附录 B「{term}」登记在 §{c}.{s}，但本章第一次加粗介绍它是在 §{c}.{bold_in[0]}")
        elif not bold_in and not any(v in "\n".join(doc.raw[slice(*span[s])]) for v in variants):
            problems.append(f"附录 B「{term}」登记在 §{c}.{s}，但该节正文里没有出现这个词")
    return problems


# --- worked_examples.py: every hand-computed literal must still be quoted in the prose --------------
def echo_problems() -> list[str]:
    src = (LABS / "worked_examples.py").read_text(encoding="utf-8").splitlines()
    chapters: list[int] = []
    problems = []
    texts = {n: "\n".join(load(p).raw).replace("−", "-").replace(",", "") for n, p in chapter_files().items()}
    for line in src:
        if m := re.match(r'print\("\\?n?第 (\d+)(?:–(\d+))? 章', line.replace('"\\n', '"')):
            a, b = int(m.group(1)), int(m.group(2) or m.group(1))
            chapters = list(range(a, b + 1))
            continue
        m = re.match(r'\s*close\(.*,\s*(-?\d+\.\d+)\)\s*$', line)
        if not m or not chapters:
            continue
        literal = m.group(1)
        value = float(literal)
        if literal in WAIVERS["echo"] or len(literal.lstrip("-0.").replace(".", "")) < 2:
            continue
        # the prose may quote fewer digits than the script; accept any rounding that is still the same number
        forms = {f"{abs(value):.{k}g}" for k in range(3, 10)} | {f"{abs(value):.{d}f}" for d in range(2, 10)}
        forms = {f for f in forms if abs(float(f) - abs(value)) <= 5e-4 * abs(value)} | {literal.lstrip("-")}
        pats = [re.compile(rf"(?<![\d.]){re.escape(f)}(?!\d)") for f in forms]
        if not any(p.search(texts[n]) for n in chapters for p in pats):
            problems.append(f"worked_examples.py 的 {literal} 在第 {chapters[0]}–{chapters[-1]} 章正文里找不到")
    return problems


# --- tests -----------------------------------------------------------------------------------------
_FILES = all_files()
_CHAPTERS = sorted(chapter_files())


@pytest.mark.parametrize("path", _FILES, ids=[file_id(p) for p in _FILES])
def test_balanced(path: Path) -> None:
    assert not (problems := balance_problems(load(path))), "\n".join(problems)


@pytest.mark.parametrize("path", _FILES, ids=[file_id(p) for p in _FILES])
def test_links_and_images_exist(path: Path) -> None:
    assert not (problems := link_problems(load(path))), "\n".join(problems)


@pytest.mark.parametrize("path", _FILES, ids=[file_id(p) for p in _FILES])
def test_section_references_resolve(path: Path) -> None:
    assert not (problems := ref_problems(load(path))), "\n".join(problems)


@pytest.mark.parametrize("path", _FILES, ids=[file_id(p) for p in _FILES])
def test_lab_sections_match(path: Path) -> None:
    assert not (problems := lab_problems(load(path))), "\n".join(problems)


def test_every_figure_is_referenced_and_has_one_source() -> None:
    assert not (problems := figure_inventory_problems()), "\n".join(problems)


def test_worked_examples_are_still_quoted() -> None:
    assert not (problems := echo_problems()), "\n".join(problems)


@pytest.mark.parametrize(
    "ch",
    [
        pytest.param(n, id=f"ch{n:02d}", marks=() if n in REWRITTEN else pytest.mark.xfail(
            strict=True, reason="not yet rewritten to the chapter-2 standard"))
        for n in _CHAPTERS
    ],
)
def test_strict_standard(ch: int) -> None:
    assert not (problems := strict_problems(ch)), "\n".join(problems)


@pytest.mark.parametrize("ch", sorted(B_SYNCED) or [pytest.param(0, marks=pytest.mark.skip(reason="no chapter synced yet"))],
                         ids=lambda n: f"ch{n:02d}")
def test_appendix_b_terms_are_introduced_where_registered(ch: int) -> None:
    assert not (problems := appendix_b_problems(ch)), "\n".join(problems)


# --- frozen inputs for a chapter-rewrite brief ----------------------------------------------------------
BASE_COMMIT = "64faf9b"  # the generation of the book that is being rewritten


def _old_text(path: Path) -> list[str]:
    import subprocess

    out = subprocess.run(["git", "show", f"{BASE_COMMIT}:{_rel(path)}"], cwd=REPO, capture_output=True, text=True)
    return out.stdout.splitlines() if out.returncode == 0 else []


def _brief(ch: int) -> None:
    path = chapter_files()[ch]
    old = _old_text(path)
    old_doc = Doc(path=path, raw=old)
    fenced, prose = False, []
    for line in old:
        if FENCE.match(line):
            fenced = not fenced
            prose.append("")
        else:
            prose.append("" if fenced else re.sub(r"\$[^$\n]*\$|`[^`\n]*`", " ", line))
    old_doc.masked = old_doc.prose = prose

    print(f"# 第 {ch} 章任务书的冻结输入（旧稿 = {BASE_COMMIT}:{_rel(path)}，{len(old)} 行）\n")
    print("## 旧大纲（对照表必须覆盖这里的每一个编号小节）")
    for i, line in enumerate(prose):
        if re.match(r"^#{1,3} ", line):
            print(f"  L{i + 1:<4} {line}")

    print("\n## 旧稿里带小节号的跨章引用（重新核对目标，不要照抄）")
    for i, c, s, shown in section_refs(old_doc):
        if c != ch and s is not None:
            print(f"  L{i + 1:<4} {shown.strip()}")

    allowed = sorted(n for n in REWRITTEN if n != ch and WAVE.get(n, 99) < WAVE.get(ch, 99))
    print(f"\n## 可以带小节号引用的章（已定稿且波次更早）：{allowed or '无'}")
    for n in allowed:
        print(f"  第 {n} 章：" + "；".join(f"{n}.{k} {t}" for k, (_, t) in sorted(sections(n).items())))
    print("  其余章节一律只写到章，并带概念名。")

    print("\n## 已定稿章节里提到本章的句子（= 对本章的承诺，必须兑现）")
    for n in sorted(REWRITTEN - {ch}):
        for i, line in enumerate(load(chapter_files()[n]).prose):
            if re.search(rf"第\s*{ch}\s*章", line):
                print(f"  第 {n} 章 L{i + 1}: {line.strip()[:160]}")

    for letter, label in (("A", "符号"), ("B", "术语")):
        print(f"\n## 附录 {letter} 里登记在本章的{label}（逐条核对：还在吗？现在在哪一节？）")
        for p in appendix_files():
            if not p.name.startswith(f"{letter}-"):
                continue
            for line in load(p).raw:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) >= 4 and re.search(rf"§\s*{ch}\.\d", cells[-1]):
                    print(f"  {cells[0]} | {cells[1]} | {cells[-1]}")

    print("\n## labs/worked_examples.py 里属于本章范围的手算（冻结：数字原样搬进正文，并在本章实验里再 check 一遍）")
    keep = False
    for i, line in enumerate((LABS / "worked_examples.py").read_text(encoding="utf-8").splitlines()):
        if m := re.match(r'print\("(?:\\n)?第 (\d+)(?:–(\d+))? 章', line):
            keep = int(m.group(1)) <= ch <= int(m.group(2) or m.group(1))
        if keep:
            print(f"  {i + 1:>3}: {line}")


# --- author-facing report -------------------------------------------------------------------------------
def _report(argv: list[str]) -> int:
    if argv[:1] == ["--brief"]:
        for a in argv[1:]:
            _brief(int(a.lower().removeprefix("ch")))
        return 0
    lab_output = None
    if "--lab-output" in argv:
        at = argv.index("--lab-output")
        lab_output = Path(argv[at + 1]).read_text(encoding="utf-8")
        argv = argv[:at] + argv[at + 2:]
    wanted = [int(a.lower().removeprefix("ch")) for a in argv] or sorted(REWRITTEN)
    hard_total = 0
    for ch in wanted:
        doc = load(chapter_files()[ch])
        hard = (balance_problems(doc) + link_problems(doc) + ref_problems(doc) + lab_problems(doc)
                + strict_problems(ch) + appendix_b_problems(ch))
        soft = soft_problems(ch, lab_output)
        hard_total += len(hard)
        print(f"\n===== 第 {ch} 章 · {doc.name} =====")
        print(f"必须修（{len(hard)}）：" + ("" if hard else " 无"))
        for x in hard:
            print("  ✗", x)
        print(f"建议看一眼（{len(soft)}）：" + ("" if soft else " 无"))
        for x in soft:
            print("  ·", x)
    for x in figure_inventory_problems():
        print("  ✗", x)
    return 1 if hard_total else 0


if __name__ == "__main__":
    sys.exit(_report(sys.argv[1:]))
