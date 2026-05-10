# 这行用于“延迟解析类型注解”。
# 对新手来说，可以先简单理解为：它让 `-> argparse.ArgumentParser`
# 这类类型标注在更多场景下更稳定，也更方便大型项目组织代码。
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import LongVideoSummaryPipeline
from .settings import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_INSTRUCT_MODEL_PATH,
    DEFAULT_QA_MODEL_PATH,
    DEFAULT_THINKING_MODEL_PATH,
    PipelineConfig,
)

"""命令行入口文件。

这个文件主要负责 4 件事：
1. 定义用户在终端里可以传入的参数。
2. 把这些参数整理成 PipelineConfig 配置对象。
3. 调用 LongVideoSummaryPipeline 执行整条处理流程。
4. 把最终结果保存成 JSON 文件并打印到终端。
"""


def build_parser() -> argparse.ArgumentParser:
    """创建并返回命令行参数解析器。

    `argparse` 是 Python 标准库里专门处理命令行参数的模块。
    例如用户执行：
    `python -m long_video_pipeline --video demo.mp4 --output-dir outputs`
    这里的 `--video`、`--output-dir` 都是在下面通过 `add_argument` 定义出来的。

    函数名后面的 `-> argparse.ArgumentParser` 叫“返回类型标注”，
    表示这个函数预计会返回一个 `ArgumentParser` 对象。
    """

    # `ArgumentParser(...)` 相当于创建一个“参数说明书”对象。
    # 程序后面会用它读取命令行输入，并自动生成 `--help` 帮助信息。
    p = argparse.ArgumentParser(description="Long video multimodal summarization pipeline")

    # 最基础的输入/输出参数。
    # `required=True` 表示这个参数必须传，不传程序会直接报错。
    # `default="outputs"` 表示如果用户没写这个参数，就用默认值。
    p.add_argument("--video", required=True, help="Path to local video file")
    p.add_argument("--output-dir", default="outputs", help="Output directory")

    # 模型路径相关参数。
    # 这里的 `default=...` 使用的是 settings.py 里预先定义好的默认路径。
    p.add_argument(
        "--instruct-model",
        default=DEFAULT_INSTRUCT_MODEL_PATH,
        help="Local path to Qwen3-VL instruct model",
    )
    p.add_argument(
        "--thinking-model",
        default=DEFAULT_THINKING_MODEL_PATH,
        help="Local path to Qwen3-VL thinking model",
    )
    p.add_argument(
        "--qa-model",
        default=DEFAULT_QA_MODEL_PATH,
        help="Local path to the QA model; default uses the lighter instruct model",
    )

    # 分块策略相关参数。
    # `dest="enable_llm_guided_chunking"` 的意思是：
    # 不管用户写的是启用还是禁用选项，最终都会存到同一个变量名里。
    # `action="store_true"` 表示“只要写了这个参数，就把值设为 True”。
    # `action="store_false"` 则相反，“只要写了这个参数，就把值设为 False”。
    p.add_argument(
        "--enable-llm-guided-chunking",
        dest="enable_llm_guided_chunking",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--disable-llm-guided-chunking",
        dest="enable_llm_guided_chunking",
        action="store_false",
    )
    p.add_argument("--llm-chunk-min-seconds", type=int, default=30)
    p.add_argument("--llm-chunk-max-seconds", type=int, default=120)
    p.add_argument("--llm-chunk-overlap-seconds", type=int, default=1)

    # 普通切块、关键帧、ASR、OCR 等处理参数。
    # `type=int` / `type=float` 表示 argparse 会自动把字符串转成整数/浮点数。
    p.add_argument("--chunk-seconds", type=int, default=60)
    p.add_argument("--overlap-seconds", type=int, default=10)
    p.add_argument("--section-minutes", type=int, default=10)
    p.add_argument("--keyframes-per-chunk", type=int, default=2)
    p.add_argument("--keyframe-candidates-per-chunk", type=int, default=6)
    p.add_argument(
        "--keyframe-max-width",
        type=int,
        default=1280,
        help="Resize extracted keyframes to this max width; 0 disables downscale",
    )

    p.add_argument("--asr-model-size", default="large-v3")
    p.add_argument("--asr-language", default="zh")
    p.add_argument("--disable-vad", action="store_true")
    p.add_argument("--ocr-lang", default="chi_sim+eng")

    # 向量检索相关参数。
    p.add_argument("--chroma-dir", default="outputs/chroma_db")
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL_NAME)
    p.add_argument("--retrieval-top-k", type=int, default=5)
    p.add_argument("--retrieval-neighbor-k", type=int, default=1)

    # 摘要生成相关参数。
    # `max_new_tokens_*` 控制模型最多生成多少新 token，可理解为输出长度上限。
    p.add_argument("--max-new-tokens-chunk", type=int, default=512)
    p.add_argument("--max-new-tokens-section", type=int, default=900)
    p.add_argument("--max-new-tokens-global", type=int, default=1200)
    p.add_argument("--max-new-tokens-review", type=int, default=1400)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.9)

    # 问答（QA）相关参数。
    p.add_argument("--qa-max-new-tokens", type=int, default=384)
    p.add_argument("--qa-temperature", type=float, default=0.1)
    p.add_argument("--qa-top-p", type=float, default=0.9)
    p.add_argument("--qa-retrieval-top-k", type=int, default=3)
    p.add_argument("--qa-retrieval-neighbor-k", type=int, default=0)

    p.add_argument(
        "--mystery-mode",
        action="store_true",
        help="Enable suspense reasoning review mode",
    )

    # 失败重试与断点续跑。
    # `resume` / `no-resume` 也是一对“写到同一个变量”的布尔开关。
    p.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Enable resume from checkpoints",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume and rerun from scratch",
    )
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--retry-base-delay-sec", type=float, default=2.0)
    p.add_argument("--retry-max-delay-sec", type=float, default=20.0)

    # 推理硬件相关参数，例如显卡映射、精度和注意力实现。
    p.add_argument("--device-map", default="none")
    p.add_argument(
        "--torch-dtype",
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    p.add_argument("--attn-implementation", default=None)

    # 双 GPU 负载均衡相关参数。
    p.add_argument(
        "--enable-dual-gpu-balance",
        dest="enable_dual_gpu_balance",
        action="store_true",
        default=False,
        help="Enable custom two-GPU memory balancing (default: disabled, single-GPU preferred)",
    )
    p.add_argument(
        "--disable-dual-gpu-balance",
        dest="enable_dual_gpu_balance",
        action="store_false",
        help="Disable custom two-GPU memory balancing",
    )
    p.add_argument(
        "--gpu0-mem-fraction",
        type=float,
        default=0.40,
        help="GPU0 memory fraction when dual-GPU balancing enabled",
    )
    p.add_argument(
        "--gpu1-mem-fraction",
        type=float,
        default=0.55,
        help="GPU1 memory fraction when dual-GPU balancing enabled",
    )
    p.add_argument(
        "--disable-free-gpu-memory-aware",
        action="store_true",
        help="Use total memory fractions only, ignore current free GPU memory",
    )
    p.add_argument(
        "--gpu-reserve-memory-mib",
        type=int,
        default=1536,
        help="Headroom reserved on each GPU for generation spikes",
    )
    p.add_argument(
        "--cpu-offload-max-memory",
        default="64GiB",
        help="CPU max_memory for accelerate offload, e.g. 64GiB",
    )

    # 把配置好的解析器返回给调用方。
    return p


def main() -> None:
    """程序真正的入口函数。

    这里的 `-> None` 表示这个函数不返回有意义的结果，
    它主要通过“执行动作”来完成任务，比如运行流程、写文件、打印输出。
    """

    # `build_parser()` 先创建参数解析器。
    # `.parse_args()` 再去读取用户在命令行输入的内容。
    # 返回值 `args` 是一个对象，可以用 `args.xxx` 的形式取到每个参数。
    args = build_parser().parse_args()

    # 把命令行参数转换成项目内部统一使用的配置对象。
    # 这样后面的 pipeline 就不用直接关心命令行细节，只需要读 cfg 即可。
    cfg = PipelineConfig(
        video_path=args.video,
        output_dir=args.output_dir,
        instruct_model_path=args.instruct_model,
        thinking_model_path=args.thinking_model,
        qa_model_path=args.qa_model,
        enable_llm_guided_chunking=args.enable_llm_guided_chunking,
        llm_chunk_min_seconds=args.llm_chunk_min_seconds,
        llm_chunk_max_seconds=args.llm_chunk_max_seconds,
        llm_chunk_overlap_seconds=args.llm_chunk_overlap_seconds,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        section_minutes=args.section_minutes,
        keyframes_per_chunk=args.keyframes_per_chunk,
        keyframe_candidates_per_chunk=args.keyframe_candidates_per_chunk,
        keyframe_max_width=args.keyframe_max_width,
        asr_model_size=args.asr_model_size,
        # 如果用户传的是字符串 "none"，这里就转成真正的 None。
        # `None` 在 Python 里表示“空值 / 没有值”。
        # `lower()` 会把字符串转成小写，避免用户写成 `None`、`NONE` 时匹配失败。
        asr_language=None if args.asr_language.lower() == "none" else args.asr_language,
        # `not args.disable_vad` 的意思是：
        # 用户如果写了 `--disable-vad`，那么 `disable_vad=True`，
        # 再取反后 `use_vad=False`；否则默认启用 VAD。
        use_vad=not args.disable_vad,
        ocr_lang=args.ocr_lang,
        chroma_dir=args.chroma_dir,
        embedding_model_name=args.embedding_model,
        retrieval_top_k=args.retrieval_top_k,
        retrieval_neighbor_k=args.retrieval_neighbor_k,
        max_new_tokens_chunk=args.max_new_tokens_chunk,
        max_new_tokens_section=args.max_new_tokens_section,
        max_new_tokens_global=args.max_new_tokens_global,
        max_new_tokens_review=args.max_new_tokens_review,
        temperature=args.temperature,
        top_p=args.top_p,
        mystery_mode=args.mystery_mode,
        qa_max_new_tokens=args.qa_max_new_tokens,
        qa_temperature=args.qa_temperature,
        qa_top_p=args.qa_top_p,
        qa_retrieval_top_k=args.qa_retrieval_top_k,
        qa_retrieval_neighbor_k=args.qa_retrieval_neighbor_k,
        resume=args.resume,
        max_retries=args.max_retries,
        retry_base_delay_sec=args.retry_base_delay_sec,
        retry_max_delay_sec=args.retry_max_delay_sec,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        enable_dual_gpu_balance=args.enable_dual_gpu_balance,
        gpu0_mem_fraction=args.gpu0_mem_fraction,
        gpu1_mem_fraction=args.gpu1_mem_fraction,
        use_free_gpu_memory=not args.disable_free_gpu_memory_aware,
        gpu_reserve_memory_mib=args.gpu_reserve_memory_mib,
        cpu_offload_max_memory=args.cpu_offload_max_memory,
    )

    # 用配置对象创建主流程实例。
    pipeline = LongVideoSummaryPipeline(cfg)

    # 真正开始执行整条长视频处理流程。
    # `run()` 的返回值通常是一个 Python 字典，里面存放最终结果。
    outputs = pipeline.run()

    # `Path(...) / "final" / "run_outputs.json"` 是 pathlib 的常见写法，
    # `/` 在这里不是除法，而是“拼接路径”。
    out_file = Path(args.output_dir) / "final" / "run_outputs.json"

    # `out_file.parent` 表示文件所在的上一级目录，也就是 `.../final`。
    # `mkdir(parents=True, exist_ok=True)` 的意思是：
    # 1. 如果父目录不存在，就一并创建。
    # 2. 如果目录已经存在，也不要报错。
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # `json.dumps(...)` 会把 Python 对象转成 JSON 字符串。
    # `ensure_ascii=False` 可以让中文直接正常保存，而不是变成 `\u4e2d\u6587`。
    # `indent=2` 表示格式化缩进 2 个空格，文件更易读。
    out_file.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")

    # 同时把结果打印到终端，方便立刻查看。
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


# 这是 Python 中非常常见的“脚本入口判断”。
# 只有当这个文件被直接运行时，`__name__` 才会等于 `"__main__"`。
# 如果它是被别的文件 import 进来的，就不会自动执行 `main()`。
if __name__ == "__main__":
    main()
