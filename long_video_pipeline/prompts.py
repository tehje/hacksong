from __future__ import annotations

"""集中存放“提示词构造函数”。

这个文件里的函数有一个共同点：
- 输入：普通 Python 变量，例如字符串、数字、布尔值。
- 输出：一段给大模型看的长字符串（prompt）。

如果你是 Python 新手，先抓住这几个语法点：

1. `def 函数名(...):`
   这是定义函数。函数像一个“可重复使用的小工具”。
2. `参数名: str` / `参数名: int`
   这是“类型提示”，告诉读代码的人：这个参数通常应该是什么类型。
   它主要帮助 IDE 和维护者理解代码，本身不强制限制运行。
3. `-> str`
   表示“这个函数最终会返回一个字符串”。
4. `f"..."` 或 `f\"\"\"...\"\"\"`
   前面的 `f` 表示格式化字符串，可以把变量值直接塞进字符串里。
   例如 `f"你好，{name}"` 会把 `name` 的值替换到花括号里。
5. `\"\"\"...\"\"\"`
   三引号字符串，适合写多行文本。这里非常适合构造 prompt。
6. `.strip()`
   用来去掉字符串开头和结尾多余的空白和换行，避免 prompt 前后出现无意义空行。

所以，这个文件本质上不是在“做复杂计算”，而是在“拼装给模型的指令模板”。
"""

from typing import List


# -------------------------
# Core summarization prompts
# -------------------------


def build_llm_chunk_plan_prompt(
    asr_timeline_text: str,
    total_duration_sec: float,
    min_chunk_seconds: int,
    max_chunk_seconds: int,
    overlap_seconds: int,
    section_minutes: int,
) -> str:
    """构造“切分视频 chunk”的提示词。

    参数说明：
    - `asr_timeline_text`：整段视频的 ASR 时间线文本。
    - `total_duration_sec`：视频总时长，单位是秒。
    - `min_chunk_seconds` / `max_chunk_seconds`：期望的 chunk 时长范围。
    - `overlap_seconds`：相邻 chunk 之间允许保留多少秒重叠。
    - `section_minutes`：更粗粒度 section 的参考分钟数。

    返回值：
    - 一段完整 prompt，供大模型生成 JSON 格式的切段方案。
    """

    # `return f"""..."""` 的意思是：
    # 直接返回一个“多行格式化字符串”。
    # 其中 `{变量名}` 会被替换成真实值。
    #
    # 例如 `{overlap_seconds}` 会被替换成传进来的秒数。
    #
    # 注意下面 JSON 示例里的 `{{` 和 `}}`：
    # 在 f-string 中，如果你真的想输出花括号本身，而不是插入变量，
    # 就要写成双花括号。
    return f"""
你是“长视频结构规划助手”。
请根据全片 ASR 时间线内容，规划语义片段（chunk）边界。

目标：
- 按语义事件切分，而不是固定秒数。
- 相邻 chunk 保持约 {overlap_seconds} 秒时间重叠（用于上下文衔接）。
- 每个 chunk 时长应尽量在 {min_chunk_seconds}~{max_chunk_seconds} 秒。
- section 用于更粗粒度聚类，按每 {section_minutes} 分钟为基准编号即可。

视频总时长（秒）：{total_duration_sec:.3f}

ASR 时间线：
{asr_timeline_text}

严格输出 JSON（不要 markdown，不要解释）：
{{
  "chunks": [
    {{
      "t_start": "HH:MM:SS",
      "t_end": "HH:MM:SS",
      "title": "该片段主题",
      "reason": "为何在此切分",
      "key_moments": ["HH:MM:SS", "HH:MM:SS"]
    }}
  ]
}}

硬约束：
1) t_start/t_end 必须在视频时长内，且 t_start < t_end；
2) 按时间顺序输出，不要重叠错乱；
3) 若 ASR 信息不足，也必须给出覆盖全片的合理分段；
4) 不要编造未出现的人名/地名，仅描述语义结构。
""".strip()


def build_chunk_card_prompt(
    t_start: str,
    t_end: str,
    asr_text: str,
    ocr_text: str,
) -> str:
    """构造单个 chunk 的信息抽取提示词。

    这个函数会把某一段视频的时间范围、ASR 文本、OCR 文本打包起来，
    让模型输出结构化的 chunk card。
    """

    # 这里的逻辑很直接：
    # 1. 把函数参数插入模板；
    # 2. 明确要求模型只输出 JSON；
    # 3. 给一个固定 schema，减少模型跑偏。
    return f"""
你是“长视频多模态信息抽取器”。
请根据给定的时间段信息、ASR转写、OCR文本和关键帧图片，输出该片段的 chunk card。

时间范围: {t_start} - {t_end}
ASR:
{asr_text}

OCR:
{ocr_text}

输出要求：
1. 只输出 JSON，对象结构必须严格匹配下方 schema。
2. 不要输出 markdown 代码块，不要解释。
3. importance 为 0~1 的浮点数。
4. quotes 中每条必须包含 t(时间戳) 和 text(原句)。
5. 如果信息缺失，对应字段用空数组或空字符串，不要编造。

JSON schema:
{{
  "one_liner": "",
  "bullets": ["", ""],
  "entities": ["", ""],
  "visual_facts": ["", ""],
  "quotes": [{{"t": "HH:MM:SS", "text": ""}}],
  "importance": 0.0,
  "tags": ["", ""]
}}
""".strip()


def build_section_summary_prompt(
    section_title: str,
    t_start: str,
    t_end: str,
    chunk_cards_text: str,
) -> str:
    """构造“章节总结”的提示词。

    它的输入不再是原始视频内容，而是多个 chunk card 的整理结果。
    也就是说，这一步是在“摘要的基础上继续摘要”。
    """

    return f"""
你是章节总结助手。请把多个 micro summaries 合并成一个章节总结，去重但保留推理/叙事演进。

章节: {section_title}
时间范围: {t_start} - {t_end}
输入 chunk cards:
{chunk_cards_text}

请输出 Markdown，格式严格如下：
## {section_title}（{t_start} - {t_end}）
- 章结论：
- 关键论据/演示：
- 争议/限制：
- 可执行清单：
- 关键证据时间戳：
""".strip()


def build_global_summary_prompt(section_summaries: str) -> str:
    """构造“全片总结”的提示词。

    输入是一组章节总结，输出目标是两版更高层的总摘要。
    """

    return f"""
你是全片总结助手。请基于章节总结输出两版大总结：

输入章节总结：
{section_summaries}

输出 Markdown，必须包含以下结构：
# Executive Summary（忙人版）
- 一句话结论
- 核心要点（3~7条）
- 关键亮点/争议
- 适合谁看

# Structured Summary（结构化版）
- 全片结构（章节索引+时间戳）
- 分章节要点
- 术语表/概念关系
- 行动清单/下一步

要求：
1. 优先保留带时间戳的可追溯信息；
2. 不要编造未出现的细节。
""".strip()


def build_thinking_review_prompt(
    draft_summary: str,
    claim_evidence_pack: str,
    mystery_mode: bool,
) -> str:
    """构造“复核总结草稿”的提示词。

    这里开始出现一个常见写法：
    - 先准备一个默认值 `mystery_rules = ""`
    - 再根据条件决定要不要补充额外规则

    这样做的好处是：主 prompt 模板不用写两份，只需要把可选部分插进去。
    """

    # 默认情况下，不附加任何“悬疑/推理模式”规则。
    mystery_rules = ""

    # `if mystery_mode:` 的意思是：
    # 如果 `mystery_mode` 为 True，就执行下面缩进的代码块。
    if mystery_mode:
        mystery_rules = """
额外要求（悬疑/推理模式）：
- 给出“线索链”：线索 -> 中间推断 -> 结论；
- 区分“已证实”和“待验证”；
- 对关键反转点给出证据时间戳。
"""

    # 最终把“固定模板”和“可选规则”拼成一个完整 prompt。
    return f"""
你是“复核与证据化推理助手”。
请对 Instruct 阶段的全片总结草稿进行复核：纠错、补证据、标注置信度。

【草稿】
{draft_summary}

【检索证据包】
{claim_evidence_pack}

请输出 Markdown，结构固定：
# 复核后最终总结
## 1) 修订后的核心结论
## 2) 逐条结论证据表
| 结论 | 证据时间戳 | 证据摘录 | 置信度(高/中/低) |
|---|---|---|---|
## 3) 不确定项与待补证据
## 4) 叙事/论证链路（若适用）
{mystery_rules}

硬约束：
1. 只能使用证据包或草稿中可追溯的信息；
2. 遇到证据不足，必须明确写“证据不足”；
3. 不要输出思维链过程，只输出结论与证据映射。
""".strip()


def build_claim_extraction_prompt(draft_summary: str) -> str:
    """构造“从总结草稿中提取可检索结论句”的提示词。

    这里的目标很单一：
    让模型把一段较长总结，拆成若干条更适合向量检索的短结论句。
    """

    return f"""
请从下面总结草稿中提取 6~12 条“可检索结论句”（每条一句话，便于向量检索）。
只输出 JSON：
{{"claims": ["...", "..."]}}

草稿：
{draft_summary}
""".strip()


def build_video_qa_prompt(
    question: str,
    summary_context: str,
    evidence_pack: str,
    mystery_mode: bool,
) -> str:
    """构造“视频问答”的提示词。

    输入包括：
    - 用户问题
    - 已有摘要上下文
    - 和当前问题最相关的证据
    - 是否开启悬疑/推理模式
    """

    # 这段写法和 `build_thinking_review_prompt()` 是同一个套路：
    # 先给可选部分一个空字符串，再按条件补内容。
    mystery_rules = ""
    if mystery_mode:
        mystery_rules = """
额外要求（悬疑/推理模式）：
- 必须明确“线索 -> 推断 -> 结论”；
- 标注“已证实”与“待验证”；
- 关键结论尽量附时间戳。
"""

    return f"""
你是“视频问答助手”。
用户会基于同一段视频提出问题，请结合摘要与“当前问题直接相关”的证据回答。
回答要尽量短、准、可定位，不要复述整篇总结。

【用户问题】
{question}

【摘要上下文】
{summary_context}

【问题相关证据】
{evidence_pack}

请输出 Markdown，格式如下：
## 回答
- 先给直接结论，控制在 1~3 句话内

## 对应片段
- 最多 3 条，每条一行，格式必须是 `HH:MM:SS-HH:MM:SS：原因`

{mystery_rules}

硬约束：
1) 仅使用上面的摘要与证据；
2) 若证据不足，明确写“证据不足”；
3) 不输出思维链，仅输出简短结论；
4) 最多输出 3 个最相关的“视频片段范围”；
5) 如果只能定位到单个时间点，也应结合相邻证据补成时间范围，不要只给孤立时间点；
6) 不要输出“证据”“不确定性”等额外小节，不要大段复述摘要，不要输出超过 6 条列表项。
""".strip()


def format_chunk_cards_for_prompt(chunk_cards: List[dict], max_items: int = 50) -> str:
    """把多个 chunk card 压缩成适合放进 prompt 的文本。

    参数：
    - `chunk_cards`：一个列表，里面每个元素都是字典。
    - `max_items`：最多取前多少个 card，避免 prompt 太长。

    返回：
    - 多行字符串。每一行对应一个 chunk card 的简化版。
    """

    # 先创建一个空列表，准备逐行收集文本。
    lines = []

    # `chunk_cards[:max_items]` 是“切片”语法，表示只取前 `max_items` 个元素。
    #
    # `enumerate(..., start=1)` 会在循环时同时给出：
    # - `i`：序号，从 1 开始
    # - `card`：当前这张 chunk card
    for i, card in enumerate(chunk_cards[:max_items], start=1):
        # `dict.get("键", 默认值)` 的意思是：
        # 如果字典里有这个键，就取它的值；
        # 如果没有，就返回后面的默认值。
        #
        # 这样写比 `card["t_start"]` 更稳妥，因为缺键时不会直接报错。
        lines.append(
            f"[{i}] {card.get('t_start', '')}-{card.get('t_end', '')} | "
            f"one_liner={card.get('one_liner', '')} | "
            f"bullets={card.get('bullets', [])} | "
            f"quotes={card.get('quotes', [])}"
        )

    # `"\n".join(lines)` 会把列表里的多行字符串，用换行符连接成一个大字符串。
    return "\n".join(lines)
