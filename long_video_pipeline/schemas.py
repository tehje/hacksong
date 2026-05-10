"""项目里统一使用的数据结构定义。

这个文件本身几乎不做“计算”，它更像是在定义：
“系统里有哪些标准数据长什么样子”。

你可以把它理解成 3 张小表：
1. `TranscriptSegment`：一条字幕片段。
2. `ChunkMeta`：一个视频切片的基础信息。
3. `ChunkCard`：模型为某个切片整理出的摘要卡片。

如果你是 Python 新手，这里有几个语法特别值得先认识：

- `@dataclass`
  这是一个装饰器，用来把“主要用于存数据的类”写得更简洁。
  它会自动帮你生成 `__init__` 等常用方法，所以不用手写构造函数。

- `name: str`
  这是“类型标注”，意思是“这个属性通常应该是字符串”。
  它主要是给人和 IDE 看，帮助阅读和检查，不会强制改变运行逻辑。

- `field(default_factory=list)`
  这是给列表字段设置默认值的安全写法。
  不能直接写成 `[]`，否则多个对象可能会意外共享同一个列表。

- `asdict(self)`
  这是 dataclasses 提供的工具，可以把 dataclass 对象转换成普通字典，
  方便后续保存成 JSON，或者传给别的函数处理。
"""

# 让类型标注先按“字符串”处理，避免类还没定义完就被提前求值。
# 对新手来说，可以简单理解成：让类型提示写起来更灵活。
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class TranscriptSegment:
    """表示一条带时间范围的字幕/转写片段。"""

    # 这条字幕开始的时间，单位通常是“秒”。
    start: float
    # 这条字幕结束的时间，单位通常也是“秒”。
    end: float
    # 这段时间里识别出来的文本内容。
    text: str


@dataclass
class ChunkMeta:
    """表示一个视频切片在“原始素材阶段”的信息。

    这里保存的是比较底层、比较客观的数据：
    比如切片时间范围、关键帧路径、ASR 文本、OCR 文本。

    可以把它理解成：
    “这个切片本身有什么基础资料可供后续模型使用？”
    """

    # 每个切片的唯一编号，例如 `chunk_0003`。
    chunk_id: str
    # 切片在整条视频中的顺序编号，通常从 0 开始递增。
    index: int
    # 切片开始时间，单位秒。
    t_start: float
    # 切片结束时间，单位秒。
    t_end: float
    # 这个切片属于哪个 section（更大一级的分组）。
    section_id: str

    # 该切片对应的关键帧图片路径列表。
    # `field(default_factory=list)` 表示：
    # 如果创建对象时没有传值，就自动给一个“新的空列表”。
    keyframe_paths: List[str] = field(default_factory=list)
    # 关键帧对应的时间戳列表，通常和 `keyframe_paths` 一一对应。
    keyframe_timestamps: List[float] = field(default_factory=list)
    # ASR（语音识别）得到的文本，默认先留空，后面再填充。
    asr_text: str = ""
    # OCR（图片文字识别）得到的文本，默认先留空，后面再填充。
    ocr_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """把当前对象转成普通字典。

        返回普通字典的好处是更容易：
        - 写入 JSON
        - 打日志
        - 传给只接受 dict 的代码
        """
        return asdict(self)


@dataclass
class ChunkCard:
    """表示模型对一个切片整理后的“信息卡片”。

    和 `ChunkMeta` 相比，`ChunkCard` 更偏“理解结果”而不是“原始素材”。
    它通常由大模型生成，目的是把一个切片里的重点信息结构化保存下来。
    """

    # 对应的切片 ID，用来和 `ChunkMeta` 对上。
    chunk_id: str
    # 切片顺序编号。
    index: int
    # 所属 section 的编号。
    section_id: str

    # 这里的开始/结束时间被保存成字符串，而不是 float。
    # 常见原因是：展示给用户时，`00:01:23` 这种格式更直观。
    t_start: str
    t_end: str

    # 一句话概括这个切片的核心内容。
    one_liner: str
    # 多条要点摘要。
    bullets: List[str]
    # 切片里出现的重要实体，比如人名、地名、品牌名等。
    entities: List[str]
    # 画面中可以直接观察到的事实。
    visual_facts: List[str]
    # 引用内容列表。
    # 这里写成 `List[Dict[str, str]]`，意思是：
    # 列表中的每一项都是“键和值都为字符串”的字典。
    quotes: List[Dict[str, str]]
    # 重要性分数，通常用于排序、筛选或后续汇总。
    importance: float
    # 标签列表，便于分类和检索。
    tags: List[str]

    # 这里也保留 ASR/OCR 文本，方便后续检索和回溯证据。
    asr_text: str = ""
    ocr_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """把信息卡片转换成普通字典，便于序列化保存。"""
        return asdict(self)
