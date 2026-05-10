"""ASR（语音识别）和 OCR（图片文字识别）相关工具。

这份文件主要负责 4 件事：
1. 找到并加载 faster-whisper 的语音识别模型。
2. 把视频转写成带时间戳的字幕片段。
3. 把字幕片段分配到视频切片 `ChunkMeta` 上。
4. 对关键帧图片做 OCR，并把文字结果写回切片对象。

下面的注释尽量面向新手解释“这行代码在做什么”和“为什么这么写”。
"""

# 这句的作用是：把类型标注先当成普通字符串处理，避免某些“类型提前引用”问题。
# 对新手来说，可以简单理解为“让类型标注更灵活”。
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from PIL import Image
import pytesseract

from .schemas import ChunkMeta, TranscriptSegment


@dataclass
class ASRResult:
    # `@dataclass` 会自动帮类生成 `__init__`、`__repr__` 等常见方法，
    # 所以这种“只存数据”的类可以写得很简洁。
    #
    # `segments: List[TranscriptSegment]` 的意思是：
    # `segments` 这个属性的值应该是一个列表，
    # 列表里的每个元素都是 `TranscriptSegment` 对象。
    segments: List[TranscriptSegment]


def _project_root() -> Path:
    """返回项目根目录。"""
    # `__file__` 是当前这个 `.py` 文件自己的路径。
    # `resolve()` 会把它变成绝对路径。
    # `parents[1]` 表示向上找两层目录。
    return Path(__file__).resolve().parents[1]


def _resolve_asr_model_size_or_path(model_size_or_path: str) -> str:
    """把用户传入的模型名/路径，解析成更明确的结果。

    这个函数兼容两种输入：
    - 模型名字，比如 `"large-v3"`
    - 本地模型目录，比如 `"/data/models/whisper-large-v3"`

    如果找到了真实存在的本地路径，就返回那个路径；
    如果没找到，就返回原始字符串，让后续库按“模型名”继续处理。
    """
    # 这行写法比较紧凑，拆开理解就是：
    # 1. 如果 `model_size_or_path` 是空值，先变成空字符串。
    # 2. `str(...)` 确保它一定是字符串。
    # 3. `strip()` 去掉首尾空格。
    # 4. 如果最终还是空字符串，就使用默认值 `"large-v3"`。
    raw = str(model_size_or_path or "").strip() or "large-v3"

    project_root = _project_root()

    # `Path(raw)` 把字符串路径包装成 Path 对象，便于拼路径和判断文件是否存在。
    # `expanduser()` 可以把 `~` 展开成用户家目录。
    p = Path(raw).expanduser()

    # 先准备一批最可能的候选路径。
    candidates = [
        p,
        (project_root / raw).expanduser(),
    ]

    models_dir = project_root / "models"

    # 再补充一些常见命名方式，提升兼容性。
    candidates.extend(
        [
            models_dir / raw,
            models_dir / f"faster-whisper-{raw}",
            models_dir / f"faster_whisper-{raw}",
            models_dir / f"whisper-{raw}",
        ]
    )

    # 依次检查候选路径，只要某个路径存在，就立刻返回它。
    for c in candidates:
        if c.exists():
            return str(c)

    # 如果所有候选路径都不存在，就把原始字符串返回。
    return raw


def _asr_download_root() -> Path:
    """返回 ASR 模型缓存目录；如果目录不存在就创建。"""
    root = _project_root() / "models" / ".faster_whisper_cache"

    # `parents=True`：如果父目录不存在，也一起创建。
    # `exist_ok=True`：如果目录已经存在，不要报错。
    root.mkdir(parents=True, exist_ok=True)
    return root


def warm_asr_model_file_cache(
    model_size_or_path: str, read_chunk_bytes: int = 8 * 1024 * 1024
) -> str:
    """顺序读取模型文件，提前预热磁盘缓存。

    这个函数不会真正“加载模型”，只是把模型目录里的文件先读一遍。
    这样操作系统可能会提前把文件放进缓存里，后续加载速度会更好。
    """
    resolved = _resolve_asr_model_size_or_path(model_size_or_path)
    root = Path(resolved).expanduser()

    # 只有当目标存在并且是一个目录时，才有“预热文件”的意义。
    if not root.exists() or not root.is_dir():
        return resolved

    # `rglob("*")` 会递归遍历目录中的所有文件和子目录。
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            # 以二进制模式打开文件。
            with path.open("rb") as f:
                # 按块读取，而不是一次性读完整个大文件，能减少内存压力。
                while True:
                    chunk = f.read(read_chunk_bytes)
                    if not chunk:
                        break
        except Exception:
            # 如果某个文件读取失败，直接跳过，不让整个预热过程报错中断。
            continue
    return str(root)


def _load_whisper_model(model_size_or_path: str, device: str, compute_type: str) -> Any:
    """加载 faster-whisper 模型。

    加载顺序是：
    1. 如果传入的是本地目录，优先直接从本地目录加载。
    2. 否则尝试从本地缓存加载。
    3. 如果缓存没有，再尝试联网下载。
    """
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        # `raise ... from e` 表示抛出一个更友好的新错误，
        # 同时保留原始错误信息，便于排查问题。
        raise RuntimeError(
            "faster-whisper is required for ASR. Please install it first."
        ) from e

    resolved = _resolve_asr_model_size_or_path(model_size_or_path)
    resolved_path = Path(resolved).expanduser()

    # 情况 1：如果这是一个实际存在的本地路径，优先按本地目录加载。
    if resolved_path.exists():
        print(f"[ASR] 使用本地模型目录: {resolved}")
        try:
            return WhisperModel(
                resolved,
                device=device,
                compute_type=compute_type,
                local_files_only=True,
            )
        except Exception as e:
            raise RuntimeError(
                "ASR 本地模型目录不可用（文件不完整或格式错误）。"
                f" path={resolved}"
            ) from e

    download_root = _asr_download_root()

    try:
        # 情况 2：先只查本地缓存，不联网。
        print(
            f"[ASR] 尝试从本地缓存加载: model={resolved}, "
            f"download_root={download_root}"
        )
        return WhisperModel(
            resolved,
            device=device,
            compute_type=compute_type,
            download_root=str(download_root),
            local_files_only=True,
        )
    except Exception as local_err:
        print(f"[ASR] 本地缓存未命中，尝试联网下载: {local_err}")

    try:
        # 情况 3：允许联网下载模型。
        return WhisperModel(
            resolved,
            device=device,
            compute_type=compute_type,
            download_root=str(download_root),
            local_files_only=False,
        )
    except Exception as online_err:
        raise RuntimeError(
            "ASR 模型加载失败（本地缓存+联网下载均失败）。"
            f" model={resolved}, download_root={download_root}. "
            "请检查网络，或预先下载模型并通过 --asr-model-size 传入本地目录。"
        ) from online_err


def transcribe_video(
    video_path: str,
    model_size: str = "large-v3",
    language: Optional[str] = "zh",
    use_vad: bool = True,
) -> ASRResult:
    """使用 faster-whisper 对视频做转写。

    Install:
      pip install faster-whisper
    """
    import torch

    # 如果当前机器能用 CUDA（通常是 NVIDIA GPU），优先走 GPU；
    # 否则就退回 CPU。
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 推理精度会影响速度和资源消耗。
    # 常见做法是：GPU 用 float16，CPU 用 int8。
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[ASR] device={device}, compute_type={compute_type}, model={model_size}")

    model = _load_whisper_model(
        model_size_or_path=model_size,
        device=device,
        compute_type=compute_type,
    )

    try:
        # `transcribe()` 返回两个值：
        # - `segments`：识别得到的字幕分段
        # - 第二个值：附加信息（这里暂时不需要，所以用 `_` 接收）
        segments, _ = model.transcribe(
            video_path,
            language=language,
            vad_filter=use_vad,
            beam_size=5,
        )

        # 把第三方库返回的数据，转换成项目内部统一使用的 TranscriptSegment。
        rows: List[TranscriptSegment] = []
        for s in segments:
            rows.append(
                TranscriptSegment(
                    # 显式转成 float，后面做时间比较更稳。
                    start=float(s.start),
                    end=float(s.end),
                    # `s.text or ""` 可以避免文本是 None 时出错。
                    # `strip()` 去掉前后空格和换行。
                    text=(s.text or "").strip(),
                )
            )
        return ASRResult(segments=rows)
    finally:
        # `finally` 中的代码无论前面成功还是失败，都会执行。
        # 这里用来及时释放模型和 GPU 显存。
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def assign_asr_to_chunks(chunks: List[ChunkMeta], segs: List[TranscriptSegment]) -> None:
    """把 ASR 结果按时间区间分配到每个视频 chunk 上。

    理解方式：
    - `chunks`：视频已经切好的时间段
    - `segs`：语音识别得到的一句句字幕

    目标是找出“哪些字幕和这个 chunk 在时间上有重叠”，
    然后把它们的文字拼起来写入 `chunk.asr_text`。
    """
    # `i` 和 `j` 是两个索引指针。
    # 这种写法常被叫做“双指针”，适合处理按时间顺序排列的数据。
    i = 0
    n = len(segs)

    for c in chunks:
        texts = []

        # 先把那些“已经完全结束在当前 chunk 开始之前”的字幕跳过。
        while i < n and segs[i].end < c.t_start:
            i += 1

        # 从 `i` 开始继续向后扫描，找所有可能和当前 chunk 重叠的字幕。
        j = i
        while j < n and segs[j].start <= c.t_end:
            # 这里再判断一次结束时间，确保这条字幕确实和 chunk 有交集。
            if segs[j].end >= c.t_start:
                texts.append(segs[j].text)
            j += 1

        # 过滤掉空字符串后，用空格拼起来。
        c.asr_text = " ".join([t for t in texts if t])


def ocr_image(image_path: str, lang: str = "chi_sim+eng") -> str:
    """对单张图片做 OCR，返回识别到的文字。"""
    # `with ... as ...` 是推荐写法。
    # 它会在代码块结束后自动关闭图片文件。
    with Image.open(image_path) as img:
        txt = pytesseract.image_to_string(img, lang=lang)
    return (txt or "").strip()


def ocr_for_chunk_keyframes(chunk: ChunkMeta, lang: str = "chi_sim+eng") -> None:
    """对一个 chunk 的所有关键帧做 OCR，并保存到 `chunk.ocr_text`。"""
    texts: List[str] = []

    for p in chunk.keyframe_paths:
        try:
            t = ocr_image(p, lang=lang)
        except Exception:
            # 某一张图识别失败时，不影响其他图片继续处理。
            t = ""
        if t:
            texts.append(t)

    # 用换行拼接多张关键帧的 OCR 文本，阅读起来更清楚。
    chunk.ocr_text = "\n".join(texts)
