"""项目配置中心。

这个文件主要做三件事：
1. 定义默认路径和默认参数；
2. 读取 `service_runtime.json` 中的运行时覆盖配置；
3. 用 `dataclass` 把配置对象、输出目录对象组织起来，方便其他模块使用。

如果你是新手，可以把它理解成：
- 上半部分：准备“默认值”；
- 中间部分：把外部 json 配置读进来，并做基础类型转换；
- 下半部分：定义两个“配置类”。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


# `Path(__file__)` 是当前这个 settings.py 文件的路径。
# `resolve()` 会把它变成绝对路径。
# `parents[1]` 表示“往上找两层目录”：
# - `parents[0]` 是当前文件所在目录 `long_video_pipeline`
# - `parents[1]` 是项目根目录 `LLM_Project`
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 这里的 `/` 不是数学除法，而是 `pathlib.Path` 重载后的“拼接路径”写法。
# 例如 `PROJECT_ROOT / "models"` 会得到 `LLM_Project/models`。
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_DATABASE_DIR = PROJECT_ROOT / "DataBase"
RUNTIME_PROFILE_PATH = PROJECT_ROOT / "service_runtime.json"

# 下面这些是项目里各类模型的默认路径。
# `str(...)` 是把 Path 对象转换成普通字符串，因为后面很多库更习惯接收字符串路径。
DEFAULT_INSTRUCT_MODEL_PATH = str(MODELS_DIR / "qwen3-vl-8b-instruct")
DEFAULT_THINKING_MODEL_PATH = str(MODELS_DIR / "qwen3-vl-8b-thinking")
DEFAULT_QA_MODEL_PATH = DEFAULT_INSTRUCT_MODEL_PATH
DEFAULT_EMBEDDING_MODEL_NAME = str(MODELS_DIR / "paraphrase-multilingual-MiniLM-L12-v2")

# 这是“运行时配置”的默认字典。
# 如果项目根目录下存在 `service_runtime.json`，其中同名字段会覆盖这里的默认值。
DEFAULT_RUNTIME_PROFILE = {
    "enable_model_staging": True,  # 是否启用模型缓存/暂存目录
    "model_staging_dir": "/var/tmp/long_video_pipeline_model_cache",  # 模型暂存目录
    "startup_warmup_mode": "background",  # 启动预热模式：background / blocking
    "startup_warm_models": True,  # 服务启动时是否预热模型
    "startup_preload_qa": True,  # 启动时是否顺便预加载问答模型
    "keep_qa_ready_after_job": True,  # 单次任务结束后是否保留 QA 模型常驻内存
    "enable_parallel_loading": True,  # 是否并行加载多个资源
    "parallel_loading_workers": 4,  # 并行加载线程/worker 数
    "prepare_keyframe_workers": 3,  # 提取关键帧时使用的 worker 数
    "prepare_ocr_workers": 4,  # OCR 预处理时使用的 worker 数
    "allow_embedding_download": False,  # 本地没有 embedding 模型时，是否允许自动下载
}


def load_runtime_profile() -> dict[str, Any]:
    """读取运行时配置，并与默认值合并。

    `dict[str, Any]` 的意思是：
    - 键（key）必须是字符串 `str`
    - 值（value）可以是任意类型 `Any`
    """

    # 先复制一份默认配置，后面如果读到用户自定义配置，就在这份副本上覆盖。
    profile = dict(DEFAULT_RUNTIME_PROFILE)

    # 如果配置文件不存在，直接返回默认配置。
    if not RUNTIME_PROFILE_PATH.exists():
        return profile

    try:
        # `read_text(encoding="utf-8")` 读取文本文件内容。
        # `json.loads(...)` 把 JSON 字符串解析成 Python 对象。
        obj = json.loads(RUNTIME_PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        # 这里选择“静默兜底”：
        # 只要读取失败、JSON 格式错误等异常发生，就退回默认配置。
        return profile

    # 如果解析出来的不是字典，也无法按“键值对配置”方式使用，因此直接返回默认值。
    if not isinstance(obj, dict):
        return profile

    # 用外部配置覆盖默认配置。
    # `obj.items()` 会依次取出字典里的 `(key, value)`。
    for key, value in obj.items():
        # 这里把 key 强制转成字符串，避免 JSON 里写出非字符串键导致后续取值不统一。
        profile[str(key)] = value
    return profile


# 模块导入时就执行一次加载，这样其他文件可以直接使用合并后的结果。
RUNTIME_PROFILE = load_runtime_profile()


def runtime_value(name: str, default: Any = None) -> Any:
    """读取某个运行时配置项。

    参数：
    - `name`: 配置项名字
    - `default`: 找不到时返回的默认值
    """
    if name in RUNTIME_PROFILE:
        return RUNTIME_PROFILE[name]
    return default


def runtime_bool(name: str, default: bool) -> bool:
    """把配置项尽量转换成布尔值 `True/False`。"""

    value = runtime_value(name, default)

    # 如果本来就是布尔值，就直接返回。
    if isinstance(value, bool):
        return value

    # 数字也可以转布尔：
    # `0 -> False`，非 0 -> True
    if isinstance(value, (int, float)):
        return bool(value)

    # 其余情况先转成字符串，再做常见写法兼容。
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False

    # 如果实在识别不了，就退回调用者给的默认值。
    return bool(default)


def runtime_int(name: str, default: Optional[int] = None) -> Optional[int]:
    """把配置项尽量转换成整数。

    `Optional[int]` 表示：返回值可能是 `int`，也可能是 `None`。
    """

    value = runtime_value(name, default)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        # 转换失败时返回默认值，而不是直接报错。
        return default


def runtime_str(name: str, default: str = "") -> str:
    """把配置项转换成字符串。"""

    value = runtime_value(name, default)
    if value is None:
        return default
    return str(value)


@dataclass
class PipelineConfig:
    """整条视频处理流水线的主配置。

    `@dataclass` 是 Python 提供的一个语法糖。
    加上它之后，Python 会自动帮这个类生成常见方法，比如：
    - `__init__`：初始化函数
    - `__repr__`：打印对象时显示内容

    例如：
    `PipelineConfig(video_path="demo.mp4")`
    就能快速创建一个配置对象。
    """

    # 字段写法 `video_path: str` 的意思是：
    # - 这个属性叫 `video_path`
    # - 预期类型是 `str`
    #
    # 如果写成 `output_dir: str = "outputs"`，
    # 就表示它有默认值；创建对象时可以不手动传入。

    # 输入/输出
    video_path: str  # 要处理的视频路径。这个字段没有默认值，因此创建对象时必须传入。
    output_dir: str = "outputs"  # 所有输出文件默认放在 outputs 目录

    # 本地模型路径
    instruct_model_path: str = DEFAULT_INSTRUCT_MODEL_PATH  # 指令型模型
    thinking_model_path: str = DEFAULT_THINKING_MODEL_PATH  # 推理/思考型模型
    qa_model_path: str = DEFAULT_QA_MODEL_PATH  # 问答模型

    # 分段 / 关键帧采样
    # 为 True 时，会让 LLM 根据 ASR 时间轴推断“语义上更合理”的分段边界，
    # 而不只是简单按固定时长切片。
    enable_llm_guided_chunking: bool = True
    chunk_seconds: int = 60  # 普通切块长度（秒）
    overlap_seconds: int = 10  # 相邻切块之间重叠的秒数，避免边界信息丢失
    section_minutes: int = 10  # 更大粒度 section 的长度（分钟）
    keyframes_per_chunk: int = 2  # 每个 chunk 默认保留多少张关键帧
    llm_chunk_min_seconds: int = 30  # LLM 分段时允许的最短 chunk
    llm_chunk_max_seconds: int = 120  # LLM 分段时允许的最长 chunk
    llm_chunk_overlap_seconds: int = 1  # LLM 分段时相邻 chunk 的最小重叠
    keyframe_candidates_per_chunk: int = 6  # 每个 chunk 先挑多少候选帧，再从中筛选
    keyframe_max_width: int = 1280  # 关键帧图片最大宽度，避免图像过大

    # ASR / OCR
    asr_model_size: str = "large-v3"  # 语音识别模型规格
    asr_language: Optional[str] = "zh"  # `Optional[str]` 表示可以是字符串，也可以是 None
    use_vad: bool = True  # 是否启用 VAD（语音活动检测），帮助切分语音片段
    ocr_lang: str = "chi_sim+eng"  # OCR 识别语言：简体中文 + 英文

    # 检索
    chroma_dir: str = "outputs/chroma_db"  # 向量数据库目录
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME  # 向量化模型
    retrieval_top_k: int = 5  # 检索时取最相关的前 k 条
    retrieval_neighbor_k: int = 1  # 额外带上前后相邻片段，增强上下文连续性

    # 文本生成
    max_new_tokens_chunk: int = 512  # chunk 级总结的最大生成 token 数
    max_new_tokens_section: int = 900  # section 级总结的最大生成 token 数
    max_new_tokens_global: int = 1200  # 全局总结的最大生成 token 数
    max_new_tokens_review: int = 1400  # 复盘/审阅类文本的最大生成 token 数
    temperature: float = 0.2  # 越低越稳定，越高越发散
    top_p: float = 0.9  # nucleus sampling 参数，控制采样范围

    # 逻辑开关
    mystery_mode: bool = False  # 项目里的特殊模式开关；默认关闭

    # 问答（QA）
    qa_max_new_tokens: int = 384
    qa_temperature: float = 0.1
    qa_top_p: float = 0.9
    qa_retrieval_top_k: int = 3
    qa_retrieval_neighbor_k: int = 0

    # 运行时
    device_map: str = "none"  # 模型如何映射到设备，如 CPU / GPU
    torch_dtype: str = "float16"  # PyTorch 张量默认精度
    attn_implementation: Optional[str] = None  # 注意力实现方式，可留空让底层自行决定

    # 多 GPU 负载均衡
    enable_dual_gpu_balance: bool = False  # 是否启用双卡显存分配策略
    gpu0_mem_fraction: float = 0.40  # GPU 0 预计使用的显存比例
    gpu1_mem_fraction: float = 0.55  # GPU 1 预计使用的显存比例
    use_free_gpu_memory: bool = True  # 是否根据空闲显存动态调整
    gpu_reserve_memory_mib: int = 1536  # 预留多少 MiB 显存，避免把卡占满
    cpu_offload_max_memory: str = "64GiB"  # CPU 卸载时最多允许使用的内存

    # 稳定性 / 重试
    resume: bool = True  # 如果存在中间结果，是否从断点继续
    max_retries: int = 2  # 失败后最多重试几次
    retry_base_delay_sec: float = 2.0  # 第一次重试前的基础等待时间
    retry_max_delay_sec: float = 20.0  # 重试等待时间上限

    # 上传统计信息
    upload_duration_sec: Optional[float] = None  # 上传耗时（秒）
    upload_size_bytes: Optional[int] = None  # 上传文件大小（字节）
    upload_filename: Optional[str] = None  # 上传文件名

    def output_path(self) -> Path:
        """把输出目录字符串转换成 `Path` 对象，便于后续做路径拼接。"""
        return Path(self.output_dir)

    def chroma_path(self) -> Path:
        """返回向量数据库目录的 `Path` 对象。"""
        return Path(self.chroma_dir)


@dataclass
class RuntimePaths:
    """运行过程中的各类输出目录。

    这个类的作用是把一堆相关目录打包到一起，
    避免在其他文件里反复手写 `"xxx/yyy"` 这种字符串拼接。
    """

    root: Path  # 整个输出根目录
    chunks_dir: Path  # 分段结果目录
    keyframes_dir: Path  # 关键帧目录
    cards_dir: Path  # chunk 卡片/摘要目录
    sections_dir: Path  # section 级结果目录
    checkpoints_dir: Path  # 断点续跑用的检查点目录
    final_dir: Path  # 最终产物目录
    qa_clips_dir: Path  # QA 裁剪片段目录
    timings_dir: Path  # 耗时统计目录

    @staticmethod
    def build(root: str) -> "RuntimePaths":
        """根据根目录，一次性构造出所有子目录路径。

        `@staticmethod` 表示这是“类里的普通函数”：
        - 可以写在类里，方便归类管理
        - 但它不依赖 `self`（实例对象）
        """
        root_path = Path(root)
        return RuntimePaths(
            root=root_path,
            chunks_dir=root_path / "chunks",
            keyframes_dir=root_path / "keyframes",
            cards_dir=root_path / "chunk_cards",
            sections_dir=root_path / "sections",
            checkpoints_dir=root_path / "checkpoints",
            final_dir=root_path / "final",
            qa_clips_dir=root_path / "qa_clips",
            timings_dir=root_path / "timings",
        )

    def mkdirs(self) -> None:
        """把所有需要的目录创建出来。"""

        for p in [
            self.root,
            self.chunks_dir,
            self.keyframes_dir,
            self.cards_dir,
            self.sections_dir,
            self.checkpoints_dir,
            self.final_dir,
            self.qa_clips_dir,
            self.timings_dir,
        ]:
            # `parents=True`：如果上级目录不存在，也一并创建。
            # `exist_ok=True`：如果目录已经存在，不报错。
            p.mkdir(parents=True, exist_ok=True)
