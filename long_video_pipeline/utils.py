from __future__ import annotations

# 这个文件收纳“通用小工具函数”，主要被 pipeline、retrieval 等模块复用。
# 可以按 4 类来理解：
# 1. 时间格式转换：秒数 <-> `HH:MM:SS`
# 2. JSON / JSONL 读写
# 3. Markdown / 证据片段解析
# 4. 失败重试与退避（backoff）
#
# 如果你是 Python 新手，先记住几个这里会反复出现的写法：
# - `from __future__ import annotations`：
#   让类型提示延后求值。简单理解：写类型标注时更灵活，不容易遇到“名字还没定义”的问题。
# - `Path`：
#   `pathlib` 里的路径对象，比直接拼字符串更适合处理文件路径。
# - `Dict[str, Any]` / `List[str]` / `Iterable[...]`：
#   这些是类型提示，主要给人和 IDE 看，帮助理解“这个参数/返回值大概长什么样”。
# - `-> str` / `-> Dict[str, Any]`：
#   表示函数“预计返回什么类型”，不会强制改变函数运行逻辑。

import json
import re
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, TypeVar


def seconds_to_hhmmss(seconds: float) -> str:
    """把秒数转换成 `HH:MM:SS` 字符串。"""

    # `int(seconds)` 会把小数秒直接截断成整数秒。
    # `max(0, ...)` 用来防止出现负时间。
    seconds = max(0, int(seconds))
    # `timedelta(seconds=...)` 会生成一个时间差对象，
    # `str(...)` 后通常会得到类似 `0:03:15` 的字符串。
    # `.rjust(8, "0")` 表示：如果总长度不够 8，就在左边补 `0`，
    # 于是 `0:03:15` 会变成 `00:03:15`。
    return str(timedelta(seconds=seconds)).rjust(8, "0")


def hhmmss_to_seconds(hhmmss: str) -> float:
    """把 `HH:MM:SS`、`MM:SS` 或纯秒数字符串转成秒数。"""

    # `hhmmss or ""` 的意思是：
    # 如果传进来的是空值（例如 `None`），先退回到空字符串，避免后面报错。
    s = str(hhmmss or "").strip()
    if not s:
        return 0.0

    # `split(":")` 会按冒号切开字符串。
    # 例如 `01:02:03` -> `["01", "02", "03"]`
    parts = s.split(":")
    try:
        if len(parts) == 3:
            # 这里先 `float(...)` 再 `int(...)`，是为了兼容像 `"01.0"` 这种输入。
            h = int(float(parts[0]))
            m = int(float(parts[1]))
            sec = float(parts[2])
            return float(h * 3600 + m * 60) + sec
        if len(parts) == 2:
            m = int(float(parts[0]))
            sec = float(parts[1])
            return float(m * 60) + sec
        return float(s)
    except Exception:
        # 这里选择“解析失败就返回 0”，而不是把异常继续往外抛。
        return 0.0


def ensure_json_obj(text: str) -> Dict[str, Any]:
    """尽量从文本里解析出一个 JSON 对象（字典）。"""

    text = text.strip()
    try:
        # 最理想的情况：整段文本本身就是合法 JSON。
        return json.loads(text)
    except Exception:
        # `pass` 表示“忽略这次异常，继续执行后面的兜底逻辑”。
        pass

    # 如果整段文本不是纯 JSON，就尝试用正则从中间捞出 `{ ... }` 这部分。
    # `[\s\S]` 表示“任意字符（包括换行）”。
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}

    # `group(0)` 表示“整段匹配到的文本”。
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return {}


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """把多条字典数据写成 JSONL 文件。"""

    # `path.parent` 是父目录。
    # `mkdir(parents=True, exist_ok=True)` 表示：
    # - 父目录不存在就递归创建
    # - 如果目录已经存在，也不要报错
    path.parent.mkdir(parents=True, exist_ok=True)
    # `with ... as f` 是上下文管理器。
    # 进入代码块时打开文件，离开代码块时自动关闭文件。
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            # JSONL（JSON Lines）的格式是：每一行都是一个独立 JSON。
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件，返回由多个字典组成的列表。"""

    # `items: List[Dict[str, Any]] = []` 是“带类型提示的初始化”。
    # 实际上它还是一个普通空列表，只是额外告诉读代码的人：
    # 这个列表里预期放的是字典。
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def safe_float(v: Any, default: float = 0.0) -> float:
    """安全地把值转成浮点数；失败时返回默认值。"""

    try:
        return float(v)
    except Exception:
        # `default` 有默认参数值 `0.0`，但调用方也可以传入别的兜底值。
        return default


def compact_text(s: str, max_len: int = 1200) -> str:
    """压缩空白字符，并在超长时截断文本。"""

    # `\s+` 匹配连续空白字符（空格、换行、制表符等）。
    # 这里把它们统一替换成单个空格，便于后续拼 prompt 或展示。
    s = re.sub(r"\s+", " ", s).strip()
    # 这是 Python 的条件表达式，格式是：
    # `A if 条件 else B`
    return s if len(s) <= max_len else s[:max_len] + "..."


def extract_claim_candidates(markdown_text: str, max_claims: int = 12) -> List[str]:
    """从 Markdown 列表里提取可能的“论断/要点”文本。"""

    claims: List[str] = []
    for line in markdown_text.splitlines():
        ln = line.strip()
        if not ln:
            continue
        # `startswith("- ")` / `startswith("* ")` 用来识别无序列表项。
        if ln.startswith("- ") or ln.startswith("* "):
            # `ln[2:]` 是切片，表示“从索引 2 一直到结尾”。
            # 这里就是去掉前面的列表符号和空格。
            c = ln[2:].strip()
            if len(c) >= 8:
                claims.append(c)
        # `^\d+\.\s+` 匹配像 `1. xxx`、`23. xxx` 这样的有序列表项。
        elif re.match(r"^\d+\.\s+", ln):
            c = re.sub(r"^\d+\.\s+", "", ln).strip()
            if len(c) >= 8:
                claims.append(c)

    # 用 `set()` 做去重。
    # 这里故意保留原始大小写显示，但去重时用小写版本来比较，
    # 这样 `Hello` 和 `hello` 会被视为同一条。
    seen = set()
    uniq: List[str] = []
    for c in claims:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
        if len(uniq) >= max_claims:
            break
    return uniq


def merge_texts(texts: List[str], sep: str = "\n") -> str:
    """把多个非空文本拼成一个字符串。"""

    # 这是一个列表推导式（list comprehension）：
    # `[t for t in texts if ...]`
    # 含义是：遍历 `texts`，只保留“不是空字符串、去掉空白后也不为空”的项。
    return sep.join([t for t in texts if t and t.strip()])


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    """把单个字典写成普通 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    # `indent=2` 会让输出 JSON 更易读，适合人工查看。
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    """读取普通 JSON 文件，并解析成字典。"""

    return json.loads(path.read_text(encoding="utf-8"))


def normalize_hhmmss(value: Any) -> str:
    """把各种时间输入统一整理成标准 `HH:MM:SS`。"""

    s = str(value or "").strip()
    if not s:
        return ""
    # 这里复用了前面两个函数：
    # 先把输入转成秒数，再把秒数格式化回统一字符串。
    return seconds_to_hhmmss(hhmmss_to_seconds(s))


def parse_evidence_json_lines(text: str) -> List[Dict[str, Any]]:
    """从形如 `- {...}` 的多行文本中提取 JSON 对象列表。"""

    rows: List[Dict[str, Any]] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        # 去掉 Markdown 列表前缀 `- `，剩下的才是可能的 JSON。
        payload = line[2:].strip()
        if not payload.startswith("{"):
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if isinstance(obj, dict):
            # 这里只接受 JSON 对象（Python 里对应 `dict`），
            # 不接受 JSON 数组、字符串、数字等其他类型。
            rows.append(obj)
    return rows


def dedupe_time_segments(segments: List[Dict[str, str]], max_items: int = 5) -> List[Dict[str, str]]:
    """时间片段去重，并把时间格式整理统一。"""

    uniq: List[Dict[str, str]] = []
    seen = set()
    for seg in segments:
        # `dict.get("key", "")` 的意思是：
        # 如果字典里有这个键，就取它的值；否则退回默认值 `""`。
        t_start = normalize_hhmmss(seg.get("t_start", ""))
        # `A or B` 的意思是：如果 A 是“真值”就用 A，否则用 B。
        # 这里表示：如果 `t_end` 为空，就回退成和 `t_start` 一样。
        t_end = normalize_hhmmss(seg.get("t_end", "")) or t_start
        source_job_id = str(seg.get("source_job_id", "")).strip()
        source_video_path = str(seg.get("source_video_path", "")).strip()
        if not t_start:
            continue
        # 用元组(tuple)作为去重键：
        # `(开始时间, 结束时间, 来源)` 完全相同就视为重复片段。
        key = (t_start, t_end, source_job_id or source_video_path)
        if key in seen:
            continue
        seen.add(key)
        item = {
            "t_start": t_start,
            "t_end": t_end,
            "reason": str(seg.get("reason", "")).strip(),
        }
        for extra_key in ("source_job_id", "source_video_name", "source_video_path"):
            extra_value = str(seg.get(extra_key, "")).strip()
            if extra_value:
                item[extra_key] = extra_value
        uniq.append(item)
        if len(uniq) >= max_items:
            break
    return uniq


def extract_time_segments_from_markdown(markdown_text: str, max_items: int = 5) -> List[Dict[str, str]]:
    """从 Markdown 文本中抽取时间片段和对应描述。"""

    # 这里先编译一个正则，用来匹配：
    # - `00:01:23`
    # - `00:01:23-00:01:40`
    # - `00:01:23 ~ 00:01:40`
    #
    # `(?P<start>...)` / `(?P<end>...)` 是“命名分组”，
    # 后面可以通过 `match.group("start")` 这种方式按名字取值。
    pattern = re.compile(
        r"(?P<start>\d{1,2}:\d{2}:\d{2})(?:\s*[-~–—]\s*(?P<end>\d{1,2}:\d{2}:\d{2}))?"
    )
    segments: List[Dict[str, str]] = []

    for raw_line in str(markdown_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = pattern.search(line)
        if not match:
            continue
        t_start = normalize_hhmmss(match.group("start"))
        t_end = normalize_hhmmss(match.group("end") or match.group("start"))
        # `match.end()` 表示“本次匹配结束的位置索引”。
        # `line[match.end() :]` 就是把时间戳后面的说明文字切出来。
        desc = line[match.end() :].strip()
        # 去掉说明文字前面常见的分隔符，例如 `:`、`-`、空格等。
        desc = re.sub(r"^[：:：\-*\s]+", "", desc).strip()
        segments.append(
            {
                "t_start": t_start,
                "t_end": t_end,
                "reason": desc,
            }
        )

    return dedupe_time_segments(segments, max_items=max_items)


def build_qa_segment_section(segments: List[Dict[str, str]]) -> str:
    """把时间片段列表渲染成 Markdown 小节。"""

    if not segments:
        return "## 对应片段\n- 证据不足"

    lines = ["## 对应片段"]
    for seg in dedupe_time_segments(segments):
        if seg["t_start"] == seg["t_end"]:
            time_label = seg["t_start"]
        else:
            # 这是 f-string，方便把变量嵌进字符串里。
            time_label = f"{seg['t_start']}-{seg['t_end']}"
        reason = seg.get("reason", "").strip() or "与问题相关的讲解片段"
        lines.append(f"- {time_label}：{reason}")
    return "\n".join(lines)


def inject_qa_segment_section(markdown_text: str, segments: List[Dict[str, str]]) -> str:
    """把“对应片段”小节插入回答 Markdown 中。"""

    text = str(markdown_text or "").strip()
    if not text:
        return build_qa_segment_section(segments)
    if "## 对应片段" in text:
        # 如果原文里已经有这个小节，就直接返回，避免重复插入。
        return text

    segment_section = build_qa_segment_section(segments)
    anchor = "## 证据"
    if anchor in text:
        # `replace(..., 1)` 里的 `1` 表示“只替换第一次出现的锚点”。
        # 这里相当于把新小节插到 `## 证据` 前面。
        return text.replace(anchor, segment_section + "\n\n" + anchor, 1)
    return text + "\n\n" + segment_section


# `TypeVar("T")` 可以理解成“占位类型”：
# 这个函数返回什么类型，外层拿到的就还是什么类型。
T = TypeVar("T")


def retry_with_backoff(
    op_name: str,
    fn: Callable[[], T],
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> T:
    """执行一个操作；失败时按指数退避策略自动重试。"""

    # `Callable[[], T]` 的意思是：
    # `fn` 是一个“可调用对象”（通常是函数），不接收参数，返回值类型是 `T`。
    attempt = 0
    # `while True` 表示“无限循环”，通常会在循环体内部 `return` 或 `raise` 结束。
    while True:
        try:
            return fn()
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                last_err = str(e).strip()
                if len(last_err) > 320:
                    last_err = last_err[:320] + "..."
                # `raise ... from e` 会保留原始异常链。
                # 这样报错信息里既有新的业务提示，也能看出底层最初是哪里失败。
                raise RuntimeError(
                    f"{op_name} failed after {max_retries} retries: {last_err}"
                ) from e
            # 指数退避（exponential backoff）：
            # 第 1 次失败等 `base_delay`
            # 第 2 次失败等 `base_delay * 2`
            # 第 3 次失败等 `base_delay * 4`
            # ...
            # 同时用 `min(max_delay, ...)` 限制最大等待时间，避免越等越久没有上限。
            sleep_s = min(max_delay, base_delay * (2 ** (attempt - 1)))
            print(f"[Retry] {op_name} failed (attempt {attempt}/{max_retries}), wait {sleep_s:.1f}s: {e}")
            time.sleep(sleep_s)
