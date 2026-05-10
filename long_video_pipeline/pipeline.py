from __future__ import annotations

"""长视频摘要流水线的主调度文件。

如果你是第一次读这个文件，可以先抓住这几点：

1. `run()` 是总入口，按阶段串起整条流水线。
2. `_prepare_chunks()` 负责把视频变成适合模型处理的小片段。
3. `_run_instruct_stage()` 负责先生成 chunk card、再生成 section summary、最后生成全局初稿。
4. `_build_vector_store()` 和 `_run_thinking_stage()` 负责“检索证据 + 二次复核”。
5. `answer_question()` 是在摘要已经生成后，基于摘要和证据回答用户问题。

这份代码里有几个 Python 语法会反复出现：

- `Optional[X]`：表示“这个值可能是 X，也可能是 None”。
- `-> Dict[str, str]`：这是返回值类型提示，主要给人和 IDE 看，不会改变运行逻辑。
- `with ...:`：上下文管理器。这里常用来做“进入一个阶段时开始计时，离开时自动收尾”。
- `@staticmethod`：静态方法，不依赖实例状态，所以调用时不需要 `self`。
- `lambda: ...`：匿名函数。这里主要是把“真正执行模型调用的动作”包起来，交给重试函数处理。
"""

# `concurrent.futures` 是 Python 标准库里的并行工具。
# 这里主要用线程池把“每个 chunk 都要做一遍”的任务并发跑起来。
# `FIRST_EXCEPTION` 表示：只要有任意一个任务先报错，就提前结束等待。
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
# `Callable[[...], ...]` 表示“可调用对象”的类型提示。
# 例如 `Callable[[ChunkMeta, int, int], None]` 可以理解成：
# “一个函数，接收 3 个参数，最终不返回值”。
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .asr_ocr import assign_asr_to_chunks, ocr_for_chunk_keyframes, transcribe_video
from .model_runtime import GenerationConfig, QwenVLRunner
from .profiling import RunProfiler
from .prompts import (
    build_chunk_card_prompt,
    build_claim_extraction_prompt,
    build_global_summary_prompt,
    build_llm_chunk_plan_prompt,
    build_section_summary_prompt,
    build_thinking_review_prompt,
    build_video_qa_prompt,
    format_chunk_cards_for_prompt,
)
from .retrieval import VectorStore, build_claim_evidence_pack, build_question_evidence_pack
from .schemas import ChunkCard, ChunkMeta, TranscriptSegment
from .settings import PipelineConfig, RuntimePaths, runtime_int
from .utils import (
    compact_text,
    dedupe_time_segments,
    ensure_json_obj,
    extract_claim_candidates,
    extract_time_segments_from_markdown,
    hhmmss_to_seconds,
    inject_qa_segment_section,
    merge_texts,
    parse_evidence_json_lines,
    read_json,
    read_jsonl,
    retry_with_backoff,
    safe_float,
    seconds_to_hhmmss,
    write_json,
    write_jsonl,
)
from .video_ops import (
    extract_video_clip,
    extract_keyframes_at_timestamps,
    extract_keyframes_for_chunk,
    make_chunks,
    normalize_keyframe_timestamps,
    probe_video_duration,
)


class LongVideoSummaryPipeline:
    """长视频总结系统的核心编排类。

    你可以把它理解成一个“总导演”：

    - 它不直接做 ASR、OCR、抽帧、向量检索这些底层工作。
    - 它负责决定这些步骤以什么顺序执行、何时复用缓存、何时回退。
    - 它还负责把最终结果写到磁盘，并把耗时统计交给 profiler。
    """

    # 处理若干个 chunk card 后，把当前结果重新压实写回 checkpoint，避免断点文件越来越乱。
    CHUNK_CARD_CHECKPOINT_COMPACT_EVERY = 12
    # section summary 不必每次都立刻写盘，累计到一定数量再 flush，可以减少 IO 次数。
    SECTION_SUMMARY_CHECKPOINT_FLUSH_EVERY = 2

    def __init__(
        self,
        cfg: PipelineConfig,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """初始化流水线对象，并准备运行目录、计时器和断点续跑状态。"""

        self.cfg = cfg
        self.progress_callback = progress_callback
        self.paths = RuntimePaths.build(cfg.output_dir)
        self.paths.mkdirs()
        self.profiler = RunProfiler(self.paths.timings_dir)
        self.video_path = cfg.video_path
        # `stem` 表示“去掉扩展名后的文件名”，常被用作一个稳定的 video_id。
        self.video_id = Path(cfg.video_path).stem
        self._staged_instruct_runner: Optional[QwenVLRunner] = None

        # `cp_` 前缀表示 checkpoint（断点文件）。
        # 这样当中途失败时，下次可以从已有中间结果继续跑。
        self.cp_run_meta = self.paths.checkpoints_dir / "run_meta.json"
        self.cp_chunk_plan = self.paths.checkpoints_dir / "chunk_plan.json"
        self.cp_chunks_prepared = self.paths.checkpoints_dir / "chunks_prepared.jsonl"
        self.cp_asr_segments = self.paths.checkpoints_dir / "asr_segments.jsonl"
        self.cp_chunk_cards = self.paths.checkpoints_dir / "chunk_cards.jsonl"
        self.cp_sections = self.paths.checkpoints_dir / "section_summaries.json"
        self.cp_global_draft = self.paths.checkpoints_dir / "global_summary_draft.md"
        self.cp_claims = self.paths.checkpoints_dir / "claims.json"
        self.cp_evidence_pack = self.paths.checkpoints_dir / "claim_evidence_pack.md"
        self.cp_final_reviewed = self.paths.checkpoints_dir / "global_summary_final_reviewed.md"

        self.resume_enabled = cfg.resume
        self._initialize_resume_mode()
        self._record_upload_timing()

    def run(self) -> Dict[str, str]:
        """执行整条摘要流水线，并返回最终产物路径。

        返回值是一个字典，里面存放各类输出文件的路径，例如：
        - chunk manifest
        - chunk cards
        - section summaries
        - global summary draft / final
        """

        try:
            with self.profiler.stage("pipeline.run", category="pipeline", meta={"video_id": self.video_id}):
                # 第 1 阶段：把长视频拆成 chunk，并补齐 ASR / 关键帧 / OCR 这些基础素材。
                self._emit_progress(stage_index=1, stage_name="prepare", message="[1/6] 模型切段 + 抽帧 + ASR/OCR...")
                print("[1/6] 模型切段 + 抽帧 + ASR/OCR...")
                with self.profiler.stage("stage.prepare", category="stage"):
                    chunks = self._prepare_chunks()

                # 第 2 阶段：让 instruct 模型先做“整理信息”的工作。
                self._emit_progress(stage_index=2, stage_name="instruct", message="[2/6] Instruct 模型生成 chunk cards / section summaries / global summary...")
                print("[2/6] Instruct 模型生成 chunk cards / section summaries / global summary...")
                with self.profiler.stage("stage.instruct", category="stage", meta={"chunk_count": len(chunks)}):
                    chunk_cards, section_summaries, draft_summary = self._run_instruct_stage(chunks)

                # 第 3 阶段：把中间结果放进向量库，方便后续“按问题找证据”。
                self._emit_progress(stage_index=3, stage_name="retrieval", message="[3/6] 建立向量库（chunk card + raw text 双粒度）...")
                print("[3/6] 建立向量库（chunk card + raw text 双粒度）...")
                with self.profiler.stage("stage.retrieval", category="stage", meta={"chunk_card_count": len(chunk_cards)}):
                    store = self._build_vector_store(chunk_cards)

                # 第 4 阶段：让 thinking 模型基于证据复核初稿，得到更可靠的最终稿。
                self._emit_progress(stage_index=4, stage_name="thinking", message="[4/6] Thinking 模型进行复核和证据化推理...")
                print("[4/6] Thinking 模型进行复核和证据化推理...")
                with self.profiler.stage("stage.thinking", category="stage"):
                    final_summary = self._run_thinking_stage(draft_summary, store)

                # 第 5 阶段：把所有关键产物写到磁盘。
                self._emit_progress(stage_index=5, stage_name="save", message="[5/6] 输出落盘...")
                print("[5/6] 输出落盘...")
                with self.profiler.stage("stage.save", category="stage"):
                    out = self._save_outputs(chunks, chunk_cards, section_summaries, draft_summary, final_summary)

                self._emit_progress(
                    stage_index=6,
                    stage_name="done",
                    message="[6/6] 完成。",
                    force_percent=100.0,
                )
                print("[6/6] 完成。")
                return out
        finally:
            # `finally` 里的代码无论成功还是失败都会执行，适合做收尾工作。
            self.profiler.flush_summary()
            self._cleanup_staged_instruct_runner()

    def answer_question(
        self,
        question: str,
        mystery_mode: Optional[bool] = None,
        qa_runner: Optional[QwenVLRunner] = None,
        summary_context_override: Optional[str] = None,
        evidence_pack_override: Optional[str] = None,
        evidence_rows_override: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """基于已生成的摘要和证据，回答用户提出的问题。

        这个方法可以理解为“摘要生成完以后，再开一个问答模式”：
        1. 先读取已经落盘的 summary / evidence。
        2. 再去向量库里检索与当前问题最相关的证据。
        3. 最后调用 QA 模型组织答案，并尽量给出对应时间片段。
        """

        # `question or ""` 是一个常见兜底写法：
        # 如果 question 是空值（例如 `None`），就先退回空字符串，避免 `.strip()` 报错。
        q = str(question or "").strip()
        if not q:
            raise ValueError("question is empty")

        # 先从最终稿、初稿和证据包里读取可复用的上下文。
        final_summary = self._read_first_text([
            self.paths.final_dir / "global_summary_final_reviewed.md",
            self.cp_final_reviewed,
        ])
        draft_summary = self._read_first_text([
            self.paths.final_dir / "global_summary_draft.md",
            self.cp_global_draft,
        ])
        global_evidence_pack = self._read_first_text([self.cp_evidence_pack])
        default_fallback_rows = parse_evidence_json_lines(global_evidence_pack)
        default_summary_context = self._build_qa_summary_context(final_summary, draft_summary)
        summary_context = str(summary_context_override or "").strip() or default_summary_context

        if evidence_rows_override is None:
            # 正常路径：自动检索与当前问题有关的证据。
            fallback_evidence_rows = default_fallback_rows
            evidence_pack, evidence_rows = self._load_question_evidence_pack(
                question=q,
                fallback_rows=fallback_evidence_rows,
            )
        else:
            # 覆盖路径：外部已经给了证据，就直接使用，不再检索。
            # 这是列表推导式：
            # `[dict(row) for row in evidence_rows_override if isinstance(row, dict)]`
            # 可以理解成“遍历传入列表，只保留字典项，并顺手复制一份”。
            evidence_rows = [dict(row) for row in evidence_rows_override if isinstance(row, dict)]
            fallback_evidence_rows = list(evidence_rows)
            evidence_pack = str(evidence_pack_override or "").strip()
            if not evidence_pack and evidence_rows:
                evidence_pack, _ = self._build_compact_evidence_pack_from_rows(
                    evidence_rows,
                    max_items=max(3, self.cfg.qa_retrieval_top_k * 2),
                )

        # 这里用了 Python 的条件表达式：
        # `A if 条件 else B`
        # 含义是：如果调用方没有显式传 `mystery_mode`，就沿用全局配置。
        qa_mystery_mode = self.cfg.mystery_mode if mystery_mode is None else bool(mystery_mode)
        prompt = build_video_qa_prompt(
            question=q,
            summary_context=summary_context,
            evidence_pack=evidence_pack,
            mystery_mode=qa_mystery_mode,
        )

        runner = qa_runner
        # `owns_runner` 表示“这个 runner 是不是当前函数自己创建的”。
        # 如果是自己创建的，最后就要自己负责 unload。
        owns_runner = runner is None
        if runner is None:
            runner = self._new_runner(self.cfg.qa_model_path)
            with self.profiler.stage("qa.model_load", category="qa_model", meta={"model_path": self.cfg.qa_model_path}):
                runner.load()
        else:
            self.profiler.record(
                name="qa.model_reuse",
                category="qa_model",
                duration_sec=0.0,
                status="reused",
                meta={"model_path": self.cfg.qa_model_path},
            )
        try:
            with self.profiler.stage(
                "qa.generate",
                category="qa",
                meta={
                    "prompt_chars": len(prompt),
                    "summary_chars": len(summary_context),
                    "evidence_rows": len(evidence_rows),
                    "max_new_tokens": self.cfg.qa_max_new_tokens,
                },
            ):
                ans = retry_with_backoff(
                    op_name="video_qa_generate",
                    fn=lambda: runner.generate(
                        prompt=prompt,
                        images=None,
                        gen_cfg=GenerationConfig(
                            max_new_tokens=self.cfg.qa_max_new_tokens,
                            temperature=self.cfg.qa_temperature,
                            top_p=self.cfg.qa_top_p,
                        ),
                    ),
                    max_retries=self.cfg.max_retries,
                    base_delay=self.cfg.retry_base_delay_sec,
                    max_delay=self.cfg.retry_max_delay_sec,
                )
            answer_md = ans.strip()
            if not answer_md:
                answer_md = "证据不足，当前无法给出可靠回答。"

            # `A or B` 的意思是：A 为空时回退到 B。
            # 这里表示：如果本轮检索没拿到证据，就用之前准备好的兜底证据。
            source_rows = evidence_rows or fallback_evidence_rows
            # 如果 evidence row 自带来源信息，优先直接从证据里提取片段；
            # 否则再尝试从模型回答的 markdown 中解析时间段。
            if self._has_source_aware_rows(source_rows):
                answer_segments = self._segments_from_evidence_rows(source_rows, max_items=3)
            else:
                answer_segments = extract_time_segments_from_markdown(answer_md, max_items=3)
                if not answer_segments:
                    answer_segments = self._segments_from_evidence_rows(source_rows, max_items=3)
            with self.profiler.stage(
                "qa.materialize_clips",
                category="qa",
                meta={"segment_count": len(answer_segments)},
            ):
                answer_clips = self._materialize_answer_clips(answer_segments, max_items=3)
            return {
                "answer_markdown": answer_md,
                "answer_segments": self._public_answer_segments(answer_segments),
                "answer_clips": answer_clips,
            }
        finally:
            if owns_runner:
                with self.profiler.stage("qa.model_unload", category="qa_model"):
                    runner.unload()
            self.profiler.flush_summary()

    @staticmethod
    def _has_source_aware_rows(rows: List[Dict[str, Any]]) -> bool:
        """判断证据行里是否已经记录了来源视频信息。"""

        for row in rows:
            # `(row or {})` 表示：如果 row 意外是空值，就先退回空字典，
            # 这样后面的 `.get(...)` 不会因为 `None` 报错。
            source_job_id = str((row or {}).get("source_job_id", "")).strip()
            source_video_path = str((row or {}).get("source_video_path", "")).strip()
            if source_job_id or source_video_path:
                return True
        return False

    @staticmethod
    def _public_answer_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """把内部 answer segment 清洗成对外返回的轻量结构。"""

        public_segments: List[Dict[str, str]] = []
        for seg in segments:
            row = {
                "t_start": str(seg.get("t_start", "")).strip(),
                "t_end": str(seg.get("t_end", "")).strip(),
                "reason": str(seg.get("reason", "")).strip(),
            }
            for extra_key in ("source_job_id", "source_video_name"):
                extra_value = str(seg.get(extra_key, "")).strip()
                if extra_value:
                    row[extra_key] = extra_value
            public_segments.append(row)
        return public_segments

    def _build_qa_summary_context(self, final_summary: str, draft_summary: str) -> str:
        """压缩最终稿和初稿，生成问答时要喂给模型的摘要上下文。"""

        blocks: List[str] = []
        final_text = compact_text(final_summary, 1800)
        draft_text = compact_text(draft_summary, 900)

        if final_text:
            # `f"...{变量}..."` 是 f-string，作用是把变量值直接插进字符串。
            blocks.append(f"[final summary]\n{final_text}")
        if draft_text and draft_text != final_text:
            blocks.append(f"[draft summary supplement]\n{draft_text}")

        if not blocks:
            return "(暂无摘要上下文)"
        # `"\n\n".join(blocks)` 表示：用两个换行把多个文本块拼起来。
        return "\n\n".join(blocks)

    def _load_question_evidence_pack(
        self,
        question: str,
        fallback_rows: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """优先从向量库检索证据，失败时退回到缓存证据。"""

        try:
            with self.profiler.stage("qa.vector_store_init", category="qa"):
                store = VectorStore(
                    persist_dir=str(self.cfg.chroma_path()),
                    embedding_model_name=self.cfg.embedding_model_name,
                )
            with self.profiler.stage(
                "qa.question_retrieval",
                category="qa",
                meta={
                    "top_k": self.cfg.qa_retrieval_top_k,
                    "neighbor_k": self.cfg.qa_retrieval_neighbor_k,
                },
            ):
                evidence_pack, evidence_rows = build_question_evidence_pack(
                    store=store,
                    question=question,
                    top_k=self.cfg.qa_retrieval_top_k,
                    neighbor_k=self.cfg.qa_retrieval_neighbor_k,
                )
            if evidence_rows:
                return evidence_pack, evidence_rows
        except Exception as e:
            # 问答阶段不能因为检索失败就整体中断，所以这里属于“失败兜底逻辑”：
            # 向量库不可用时，退回已经缓存过的证据。
            print(f"[QA][WARN] question retrieval unavailable, fallback to cached evidence pack: {e}")

        return self._build_compact_evidence_pack_from_rows(
            fallback_rows,
            max_items=max(3, self.cfg.qa_retrieval_top_k * 2),
        )

    @staticmethod
    def _build_compact_evidence_pack_from_rows(
        rows: List[Dict[str, Any]],
        max_items: int,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """把较长的证据行裁剪成紧凑版，减少 prompt 长度。"""

        compact_rows: List[Dict[str, Any]] = []
        for row in rows:
            # `row.get("evidence", "") or row.get("document", "")` 是典型兜底写法：
            # 优先使用 `evidence` 字段；如果它为空，再回退到 `document`。
            evidence_text = compact_text(
                str(row.get("evidence", "") or row.get("document", "")).strip(),
                320,
            )
            if not evidence_text:
                continue

            compact_rows.append(
                {
                    "chunk_id": str(row.get("chunk_id", "")).strip(),
                    "t_start": str(row.get("t_start", "")).strip(),
                    "t_end": str(row.get("t_end", "")).strip(),
                    "section_id": str(row.get("section_id", "")).strip(),
                    "modality": str(row.get("modality", "")).strip(),
                    "evidence": evidence_text,
                }
            )
            if len(compact_rows) >= max(1, int(max_items)):
                break

        if not compact_rows:
            return "", []

        lines = ["## Question Evidence"]
        for row in compact_rows:
            lines.append("- " + json.dumps(row, ensure_ascii=False))
        return "\n".join(lines), compact_rows

    def _prepare_chunks(self) -> List[ChunkMeta]:
        """准备后续摘要所需的 chunk 基础数据。

        这一阶段主要做“底层素材准备”：
        1. ASR 转写
        2. 切段
        3. 给每段挂上对应文本
        4. 抽关键帧
        5. OCR
        """

        if self.resume_enabled and self.cp_chunks_prepared.exists():
            print("[Resume] 使用已缓存的 chunks_prepared.jsonl")
            self.profiler.record(
                name="prepare.resume_chunks",
                category="prepare",
                duration_sec=0.0,
                status="cached",
                meta={"path": str(self.cp_chunks_prepared)},
            )
            rows = read_jsonl(self.cp_chunks_prepared)
            return [self._chunk_meta_from_dict(r) for r in rows]

        self._emit_progress(stage_index=1, stage_name="prepare", message="[1/6] ASR 转写中...")
        print("[Prepare] ASR 转写开始...")
        with self.profiler.stage("prepare.asr", category="prepare"):
            segments = self._load_or_run_asr()
        print(f"[Prepare] ASR 转写完成，segments={len(segments)}")

        chunks: List[ChunkMeta] = []
        if self.cfg.enable_llm_guided_chunking:
            # 优先尝试语义切段，让 chunk 边界更贴近内容变化点。
            self._emit_progress(stage_index=1, stage_name="prepare", message="[1/6] LLM 引导语义切段中...")
            print("[Prepare] LLM 引导语义切段...")
            try:
                with self.profiler.stage("prepare.llm_chunking", category="prepare", meta={"segment_count": len(segments)}):
                    chunks = self._build_chunks_with_llm_guidance(segments)
            except Exception as e:
                print(f"[Prepare][WARN] LLM 引导语义切段失败，回退固定切段: {e}")
                chunks = []

        if not chunks:
            # 如果模型没有给出可靠切段，就退回稳定的固定时长切段。
            print("[Chunking] 回退到固定长度切段")
            with self.profiler.stage("prepare.fixed_chunking", category="prepare"):
                chunks = make_chunks(
                    video_path=self.video_path,
                    chunk_seconds=self.cfg.chunk_seconds,
                    overlap_seconds=self.cfg.overlap_seconds,
                    section_minutes=self.cfg.section_minutes,
                )
            if self.cp_chunk_plan.exists():
                try:
                    self.cp_chunk_plan.unlink()
                except Exception:
                    pass

        print(f"[Prepare] 切段完成，chunks={len(chunks)}")
        with self.profiler.stage(
            "prepare.assign_asr_to_chunks",
            category="prepare",
            meta={"chunk_count": len(chunks), "segment_count": len(segments)},
        ):
            assign_asr_to_chunks(chunks, segments)

        self._emit_progress(stage_index=1, stage_name="prepare", message="[1/6] 关键帧抽取中...")
        print("[Prepare] 关键帧抽取开始...")
        total_chunks = max(1, len(chunks))
        self._run_chunk_workers(
            chunks=chunks,
            total_chunks=total_chunks,
            worker_count=self._parallel_worker_count("prepare_keyframe_workers", default_limit=3),
            worker_fn=self._prepare_keyframes_for_chunk,
        )

        self._emit_progress(stage_index=1, stage_name="prepare", message="[1/6] OCR 处理中...")
        print("[Prepare] OCR 开始...")
        self._run_chunk_workers(
            chunks=chunks,
            total_chunks=total_chunks,
            worker_count=self._parallel_worker_count("prepare_ocr_workers", default_limit=4),
            worker_fn=self._ocr_chunk_keyframes,
        )

        write_jsonl(self.cp_chunks_prepared, [c.to_dict() for c in chunks])
        return chunks

    def _build_chunks_with_llm_guidance(self, segments: List[TranscriptSegment]) -> List[ChunkMeta]:
        """根据 LLM 的 chunk plan 真正构建 `ChunkMeta` 对象。

        可以把这个函数理解成“两步走”：
        1. 先拿到模型产出的 chunk plan（本质上是一组普通字典）。
        2. 再把这些字典转换成项目内部统一使用的 `ChunkMeta`。
        """

        with self.profiler.stage("prepare.probe_video_duration", category="prepare"):
            duration = probe_video_duration(self.video_path)

        plan_rows: List[Dict[str, Any]] = []
        if self.resume_enabled and self.cp_chunk_plan.exists():
            obj = read_json(self.cp_chunk_plan)
            # `obj.get("chunks", [])` 的意思是：
            # 如果 `obj` 这个字典里有 `chunks` 键，就取它；
            # 否则回退到空列表，避免后面遍历时报错。
            rows = obj.get("chunks", []) if isinstance(obj, dict) else []
            if isinstance(rows, list):
                plan_rows = self._normalize_chunk_plan_rows(rows, duration)

        if not plan_rows:
            with self.profiler.stage("prepare.request_chunk_plan", category="prepare", meta={"segment_count": len(segments)}):
                plan_rows = self._request_chunk_plan_from_instruct(segments=segments, duration=duration)
            if plan_rows:
                write_json(self.cp_chunk_plan, {"chunks": plan_rows})

        if not plan_rows:
            return []

        chunks: List[ChunkMeta] = []
        section_len = max(60, self.cfg.section_minutes * 60)

        for idx, row in enumerate(plan_rows):
            # `max/min` 的组合是常见写法，用来把数值“夹”在合法范围内。
            t_start = max(0.0, min(duration, float(row.get("t_start", 0.0))))
            t_end = max(t_start + 1.0, min(duration, float(row.get("t_end", t_start + 1.0))))
            section_idx = int(t_start // section_len)
            chunk_id = f"chunk_{idx:05d}"
            section_id = f"section_{section_idx:03d}"

            # 模型可能会给一组“重要时间点”，这里把它们整理成合法的抽帧时间戳。
            key_moments = row.get("key_moments", [])
            key_ts = normalize_keyframe_timestamps(
                chunk_start=t_start,
                chunk_end=t_end,
                candidate_timestamps=key_moments if isinstance(key_moments, list) else [],
                desired_count=max(self.cfg.keyframes_per_chunk, self.cfg.keyframe_candidates_per_chunk),
            )

            chunks.append(
                ChunkMeta(
                    chunk_id=chunk_id,
                    index=idx,
                    t_start=t_start,
                    t_end=t_end,
                    section_id=section_id,
                    keyframe_timestamps=key_ts,
                )
            )

        return chunks

    def _request_chunk_plan_from_instruct(
        self,
        segments: List[TranscriptSegment],
        duration: float,
    ) -> List[Dict[str, Any]]:
        """向 instruct 模型请求语义切段方案。

        这里不是直接返回最终 `ChunkMeta`，而是先让模型给一份“切段计划”。
        这样好处是后面还能统一做清洗、修正和兜底。
        """

        timeline_text = self._build_asr_timeline_text(segments)
        prompt = build_llm_chunk_plan_prompt(
            asr_timeline_text=timeline_text,
            total_duration_sec=duration,
            min_chunk_seconds=self.cfg.llm_chunk_min_seconds,
            max_chunk_seconds=self.cfg.llm_chunk_max_seconds,
            overlap_seconds=self.cfg.llm_chunk_overlap_seconds,
            section_minutes=self.cfg.section_minutes,
        )

        runner = self._staged_instruct_runner
        if runner is None:
            runner = self._new_runner(self.cfg.instruct_model_path)
            with self.profiler.stage("instruct.chunk_plan_model_load", category="model"):
                runner.load()
            self._staged_instruct_runner = runner
        else:
            # 这里复用已经加载好的 instruct 模型，避免在同一阶段反复加载/卸载。
            self.profiler.record(
                name="instruct.chunk_plan_model_reuse",
                category="model",
                duration_sec=0.0,
                status="reused",
                meta={"model_path": self.cfg.instruct_model_path},
            )
        with self.profiler.stage("instruct.chunk_plan_generate", category="generation"):
            txt = retry_with_backoff(
                op_name="llm_chunk_plan",
                fn=lambda: runner.generate(
                    prompt=prompt,
                    images=None,
                    gen_cfg=GenerationConfig(
                        max_new_tokens=max(800, self.cfg.max_new_tokens_global),
                        temperature=0.0,
                        top_p=1.0,
                    ),
                ),
                max_retries=self.cfg.max_retries,
                base_delay=self.cfg.retry_base_delay_sec,
                max_delay=self.cfg.retry_max_delay_sec,
            )

        obj = ensure_json_obj(txt)
        # 模型输出未必完全可靠，所以这里先尽量解析 JSON，
        # 再只取我们关心的 `chunks` 字段。
        rows = obj.get("chunks", []) if isinstance(obj, dict) else []
        if not isinstance(rows, list):
            rows = []

        return self._normalize_chunk_plan_rows(rows, duration)

    def _build_asr_timeline_text(self, segments: List[TranscriptSegment], max_chars: int = 18000) -> str:
        """把 ASR 结果整理成带时间戳的时间线文本。"""

        if not segments:
            return "(ASR empty)"

        lines: List[str] = []
        total = 0
        for s in segments:
            # 这里把每条转写整理成：
            # `[开始时间-结束时间] 文本`
            # 方便后续模型理解“内容发生在什么时候”。
            line = (
                f"[{seconds_to_hhmmss(s.start)}-{seconds_to_hhmmss(s.end)}] "
                f"{compact_text(s.text, 180)}"
            )
            size = len(line) + 1
            if total + size > max_chars:
                break
            lines.append(line)
            total += size
        return "\n".join(lines) if lines else "(ASR empty)"

    def _normalize_chunk_plan_rows(self, rows: List[Any], duration: float) -> List[Dict[str, Any]]:
        """清洗、修正并必要时拆分模型给出的 chunk plan。

        这是一个典型的“格式归一化 + 边界修正 + 超长拆分”函数：
        - 先把模型输出中各种可能的时间格式统一成秒数
        - 再保证每段起止时间合法、顺序合理
        - 如果某一段过长，再继续拆成多个较稳定的小段
        """

        min_sec = max(10, int(self.cfg.llm_chunk_min_seconds))
        max_sec = max(min_sec + 10, int(self.cfg.llm_chunk_max_seconds))
        overlap = max(0.0, float(self.cfg.llm_chunk_overlap_seconds))

        parsed: List[Dict[str, Any]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue

            # 模型给出的时间可能是数字，也可能是 "00:01:23" 这样的字符串。
            s = self._parse_time_like(item.get("t_start"))
            e = self._parse_time_like(item.get("t_end"))
            s = max(0.0, min(float(duration), s))
            e = max(0.0, min(float(duration), e))
            if e <= s + 1.0:
                continue

            km = self._parse_time_list(item.get("key_moments", []), s, e)
            parsed.append(
                {
                    "t_start": s,
                    "t_end": e,
                    "title": str(item.get("title", "")).strip(),
                    "reason": str(item.get("reason", "")).strip(),
                    "key_moments": km,
                }
            )

        if not parsed:
            return []

        # `key=lambda x: (...)` 表示“按什么规则排序”。
        # 这里先按开始时间排，再按结束时间排，确保 chunk plan 顺序稳定。
        parsed = sorted(parsed, key=lambda x: (x["t_start"], x["t_end"]))

        fixed: List[Dict[str, Any]] = []
        for i, row in enumerate(parsed):
            s = float(row["t_start"])
            e = float(row["t_end"])

            if i == 0:
                # 第一段强制从 0 秒起，避免视频开头被漏掉。
                s = 0.0
            else:
                prev_end = float(fixed[-1]["t_end"])
                target_start = max(0.0, prev_end - overlap)
                s = min(s, target_start)
                if s > prev_end + min_sec:
                    s = target_start

            if e <= s + 1.0:
                e = min(duration, s + min_sec)
            e = min(duration, e)

            fixed.append(
                {
                    "t_start": s,
                    "t_end": e,
                    "title": row["title"],
                    "reason": row["reason"],
                    "key_moments": [float(x) for x in row.get("key_moments", [])],
                }
            )

        if not fixed:
            return []
        fixed[-1]["t_end"] = float(duration)

        expanded: List[Dict[str, Any]] = []
        for row in fixed:
            s = float(row["t_start"])
            e = float(row["t_end"])
            length = e - s
            if length <= max_sec:
                expanded.append(row)
                continue

            # 如果一段过长，就继续拆成多个更容易处理的小段。
            step = max(float(min_sec), float(max_sec) - overlap)
            cursor = s
            part_idx = 1
            while cursor < e - 1.0:
                part_end = min(e, cursor + max_sec)
                k = self._parse_time_list(row.get("key_moments", []), cursor, part_end)
                expanded.append(
                    {
                        "t_start": cursor,
                        "t_end": part_end,
                        "title": f"{row['title']}#{part_idx}" if row["title"] else "",
                        "reason": row["reason"],
                        "key_moments": k,
                    }
                )
                if part_end >= e:
                    break
                # 下一段的起点会和上一段保留少量 overlap，
                # 这是切片任务里常见的“重叠切段”模式，可以减少边界信息丢失。
                cursor = max(cursor + step, part_end - overlap)
                part_idx += 1

        normalized: List[Dict[str, Any]] = []
        for i, row in enumerate(expanded):
            s = float(row["t_start"])
            e = float(row["t_end"])

            if i == 0:
                s = 0.0
            else:
                prev_end = float(normalized[-1]["t_end"])
                s = min(s, max(0.0, prev_end - overlap))

            if e <= s + 1.0:
                e = min(duration, s + min_sec)
            if i == len(expanded) - 1:
                e = float(duration)

            if normalized and (e - s) < min_sec * 0.4:
                # 太短的尾巴段会并回上一段，避免产生没有信息量的碎片。
                prev = normalized[-1]
                prev["t_end"] = max(float(prev["t_end"]), e)
                prev["key_moments"] = normalize_keyframe_timestamps(
                    chunk_start=float(prev["t_start"]),
                    chunk_end=float(prev["t_end"]),
                    candidate_timestamps=list(prev.get("key_moments", [])) + list(row.get("key_moments", [])),
                    desired_count=max(2, self.cfg.keyframe_candidates_per_chunk),
                )
                continue

            km = row.get("key_moments", [])
            if not isinstance(km, list):
                km = []
            km = normalize_keyframe_timestamps(
                chunk_start=s,
                chunk_end=e,
                candidate_timestamps=km,
                desired_count=max(2, self.cfg.keyframe_candidates_per_chunk),
            )

            normalized.append(
                {
                    "t_start": max(0.0, s),
                    "t_end": min(float(duration), e),
                    "title": str(row.get("title", "")),
                    "reason": str(row.get("reason", "")),
                    "key_moments": km,
                }
            )

        if normalized:
            normalized[0]["t_start"] = 0.0
            normalized[-1]["t_end"] = float(duration)

        return normalized

    @staticmethod
    def _parse_time_like(v: Any) -> float:
        """把数字/字符串时间统一解析为秒数。"""

        # `isinstance(v, (int, float))` 表示：
        # 如果 `v` 是整数或浮点数中的任意一种，就按数字时间直接处理。
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v or "").strip()
        if not s:
            return 0.0
        if ":" in s:
            try:
                return float(hhmmss_to_seconds(s))
            except Exception:
                return 0.0
        try:
            return float(s)
        except Exception:
            return 0.0

    def _parse_time_list(self, v: Any, t_start: float, t_end: float) -> List[float]:
        """解析时间点列表，并过滤掉 chunk 范围外的时间。"""

        if not isinstance(v, list):
            return []
        out: List[float] = []
        for x in v:
            t = self._parse_time_like(x)
            if t_start <= t <= t_end:
                out.append(float(t))
        # 这里用了“集合去重 + 排序”的组合：
        # - `set(...)` 去掉重复时间点
        # - `round(x, 3)` 避免极小的小数误差导致本来一样的值被当成不同值
        # - `sorted(...)` 让最终顺序稳定
        return sorted(list({round(x, 3) for x in out}))

    def _run_instruct_stage(
        self,
        chunks: List[ChunkMeta],
    ) -> tuple[List[ChunkCard], List[str], str]:
        """执行 instruct 阶段：chunk card -> section summary -> global draft。

        这里做的是“多级摘要”：
        - 先总结每个 chunk
        - 再总结每个 section
        - 最后把 section 汇总成全局初稿
        """

        chunk_cards = self._load_chunk_cards_checkpoint()
        section_map = self._load_section_summaries_checkpoint_map()
        section_summaries = [section_map[k] for k in sorted(section_map.keys())]
        draft_summary = self.cp_global_draft.read_text(encoding="utf-8").strip() if (self.resume_enabled and self.cp_global_draft.exists()) else ""

        # 下面这几组 `need_xxx` 判断，本质上是在算“哪些中间结果还是脏的，需要重算”。
        need_chunk_cards = len(chunk_cards) < len(chunks)
        expected_section_ids = sorted({c.section_id for c in chunk_cards}) if chunk_cards else []
        need_sections = (not expected_section_ids) or any(sec_id not in section_map for sec_id in expected_section_ids)
        need_draft = (not draft_summary) or need_sections

        if not (need_chunk_cards or need_sections or need_draft):
            print("[Resume] Instruct 阶段已完成，直接复用缓存")
            self.profiler.record(name="instruct.resume", category="instruct", duration_sec=0.0, status="cached")
            return chunk_cards, section_summaries, draft_summary

        runner = self._staged_instruct_runner
        if runner is None:
            runner = self._new_runner(self.cfg.instruct_model_path)
            with self.profiler.stage("instruct.model_load", category="model"):
                runner.load()
        else:
            self.profiler.record(
                name="instruct.model_reuse",
                category="model",
                duration_sec=0.0,
                status="reused",
                meta={"model_path": self.cfg.instruct_model_path},
            )

        try:
            cards_were_updated = False
            if need_chunk_cards:
                # 如果本轮生成了新的 chunk card，后续 section / draft 也要视为脏数据重新计算。
                chunk_cards = self._generate_chunk_cards(runner, chunks, existing=chunk_cards)
                cards_were_updated = True

            section_map = {} if cards_were_updated else self._load_section_summaries_checkpoint_map()
            expected_section_ids = sorted({c.section_id for c in chunk_cards}) if chunk_cards else []
            need_sections = (not expected_section_ids) or any(sec_id not in section_map for sec_id in expected_section_ids)

            if need_sections:
                section_summaries = self._generate_section_summaries(
                    runner,
                    chunk_cards,
                    force_regen=cards_were_updated,
                )
            else:
                section_summaries = [section_map[k] for k in sorted(section_map.keys())]

            need_draft = (not draft_summary) or need_sections
            if need_draft:
                draft_summary = self._generate_global_summary(runner, section_summaries)
                self.cp_global_draft.write_text(draft_summary, encoding="utf-8")
                self._invalidate_thinking_checkpoints()
        finally:
            with self.profiler.stage("instruct.model_unload", category="model"):
                runner.unload()
            if self._staged_instruct_runner is runner:
                self._staged_instruct_runner = None

        return chunk_cards, section_summaries, draft_summary

    def _generate_chunk_cards(
        self,
        runner: QwenVLRunner,
        chunks: List[ChunkMeta],
        existing: List[ChunkCard],
    ) -> List[ChunkCard]:
        """为每个 chunk 生成结构化摘要卡片（chunk card）。"""

        card_map = {c.chunk_id: c for c in existing}

        for c in chunks:
            if c.chunk_id in card_map:
                continue

            prompt = build_chunk_card_prompt(
                t_start=seconds_to_hhmmss(c.t_start),
                t_end=seconds_to_hhmmss(c.t_end),
                asr_text=c.asr_text,
                ocr_text=c.ocr_text,
            )

            with self.profiler.stage(
                "instruct.chunk_card_generate",
                category="generation",
                meta={"chunk_id": c.chunk_id, "chunk_index": c.index, "keyframe_count": len(c.keyframe_paths)},
            ):
                text = retry_with_backoff(
                    op_name=f"chunk_card_generate:{c.chunk_id}",
                    # 这里写成 `lambda prompt=prompt, imgs=c.keyframe_paths: ...`
                    # 是为了把“当前循环里的 prompt / 图片列表”固定下来。
                    # 否则在某些闭包场景里，后面循环继续变化时容易拿错值。
                    fn=lambda prompt=prompt, imgs=c.keyframe_paths: runner.generate(
                        prompt=prompt,
                        images=imgs,
                        gen_cfg=GenerationConfig(
                            max_new_tokens=self.cfg.max_new_tokens_chunk,
                            temperature=self.cfg.temperature,
                            top_p=self.cfg.top_p,
                        ),
                    ),
                    max_retries=self.cfg.max_retries,
                    base_delay=self.cfg.retry_base_delay_sec,
                    max_delay=self.cfg.retry_max_delay_sec,
                )

            obj = ensure_json_obj(text)
            # 这里把模型输出的 JSON 转成内部的 `ChunkCard` 数据结构。
            card = ChunkCard(
                chunk_id=c.chunk_id,
                index=c.index,
                section_id=c.section_id,
                t_start=seconds_to_hhmmss(c.t_start),
                t_end=seconds_to_hhmmss(c.t_end),
                one_liner=str(obj.get("one_liner", "")).strip(),
                bullets=[str(x) for x in obj.get("bullets", []) if str(x).strip()],
                entities=[str(x) for x in obj.get("entities", []) if str(x).strip()],
                visual_facts=[str(x) for x in obj.get("visual_facts", []) if str(x).strip()],
                quotes=[q for q in obj.get("quotes", []) if isinstance(q, dict)],
                importance=min(1.0, max(0.0, safe_float(obj.get("importance", 0.0)))),
                tags=[str(x) for x in obj.get("tags", []) if str(x).strip()],
                asr_text=c.asr_text,
                ocr_text=c.ocr_text,
            )
            card_map[c.chunk_id] = card
            self._append_chunk_card_checkpoint(card)

            if len(card_map) % self.CHUNK_CARD_CHECKPOINT_COMPACT_EVERY == 0:
                # 先排序再整体写回，避免断点文件里出现重复或顺序混乱。
                ordered_cards = self._order_cards(card_map.values())
                write_jsonl(self.cp_chunk_cards, [x.to_dict() for x in ordered_cards])

        ordered_cards = self._order_cards(card_map.values())
        write_jsonl(self.cp_chunk_cards, [x.to_dict() for x in ordered_cards])
        return ordered_cards

    def _generate_section_summaries(
        self,
        runner: QwenVLRunner,
        cards: List[ChunkCard],
        force_regen: bool = False,
    ) -> List[str]:
        """把多个 chunk card 按 section 聚合，再生成章节摘要。"""

        chunk_by_section: Dict[str, List[ChunkCard]] = {}
        for c in cards:
            # `setdefault(key, [])` 是字典分组时很常见的写法：
            # 如果这个键不存在，就先放一个空列表；然后再把当前元素 append 进去。
            chunk_by_section.setdefault(c.section_id, []).append(c)

        cached = {} if force_regen else self._load_section_summaries_checkpoint_map()
        dirty_updates = 0

        for sec_id in sorted(chunk_by_section.keys()):
            if sec_id in cached:
                continue

            sec_cards = chunk_by_section[sec_id]
            # section 的时间范围由“本章第一张卡”和“本章最后一张卡”共同决定。
            t_start = sec_cards[0].t_start
            t_end = sec_cards[-1].t_end
            section_title = f"第{int(sec_id.split('_')[-1]) + 1}章"

            prompt = build_section_summary_prompt(
                section_title=section_title,
                t_start=t_start,
                t_end=t_end,
                chunk_cards_text=format_chunk_cards_for_prompt([c.to_dict() for c in sec_cards], max_items=120),
            )
            with self.profiler.stage(
                "instruct.section_summary_generate",
                category="generation",
                meta={"section_id": sec_id, "chunk_count": len(sec_cards)},
            ):
                txt = retry_with_backoff(
                    op_name=f"section_summary_generate:{sec_id}",
                    fn=lambda prompt=prompt: runner.generate(
                        prompt=prompt,
                        images=None,
                        gen_cfg=GenerationConfig(
                            max_new_tokens=self.cfg.max_new_tokens_section,
                            temperature=self.cfg.temperature,
                            top_p=self.cfg.top_p,
                        ),
                    ),
                    max_retries=self.cfg.max_retries,
                    base_delay=self.cfg.retry_base_delay_sec,
                    max_delay=self.cfg.retry_max_delay_sec,
                )
            cached[sec_id] = txt.strip()
            dirty_updates += 1
            if dirty_updates >= self.SECTION_SUMMARY_CHECKPOINT_FLUSH_EVERY:
                # 累积到一定数量再写盘，属于常见的“批量 flush”优化。
                write_json(self.cp_sections, cached)
                dirty_updates = 0

        if dirty_updates:
            write_json(self.cp_sections, cached)

        return [cached[k] for k in sorted(cached.keys())]

    def _generate_global_summary(self, runner: QwenVLRunner, section_summaries: List[str]) -> str:
        """把所有章节摘要再汇总成全局摘要初稿。"""

        prompt = build_global_summary_prompt(merge_texts(section_summaries, sep="\n\n"))
        with self.profiler.stage("instruct.global_summary_generate", category="generation", meta={"section_count": len(section_summaries)}):
            txt = retry_with_backoff(
                op_name="global_summary_generate",
                fn=lambda: runner.generate(
                    prompt=prompt,
                    images=None,
                    gen_cfg=GenerationConfig(
                        max_new_tokens=self.cfg.max_new_tokens_global,
                        temperature=self.cfg.temperature,
                        top_p=self.cfg.top_p,
                    ),
                ),
                max_retries=self.cfg.max_retries,
                base_delay=self.cfg.retry_base_delay_sec,
                max_delay=self.cfg.retry_max_delay_sec,
            )
        return txt.strip()

    def _build_vector_store(self, cards: List[ChunkCard]) -> VectorStore:
        """根据 chunk cards 构建或复用向量库。"""

        manifest_path = Path(self.cfg.chroma_path()) / "collection_manifest.json"
        card_fingerprint = self._chunk_cards_fingerprint(cards)

        with self.profiler.stage("retrieval.vector_store_init", category="retrieval", meta={"chunk_card_count": len(cards)}):
            store = VectorStore(
                persist_dir=str(self.cfg.chroma_path()),
                embedding_model_name=self.cfg.embedding_model_name,
            )

        try:
            existing_manifest = read_json(manifest_path) if manifest_path.exists() else {}
        except Exception:
            existing_manifest = {}
        if (
            isinstance(existing_manifest, dict)
            and str(existing_manifest.get("video_id", "")) == self.video_id
            and str(existing_manifest.get("chunk_card_fingerprint", "")) == card_fingerprint
            and str(existing_manifest.get("embedding_model_name", "")) == str(self.cfg.embedding_model_name)
            and store.chunk_ids_sorted
        ):
            # 指纹一致说明“视频内容 + 配置 + 嵌入模型”都没变，可以直接复用旧向量库。
            self.profiler.record(
                name="retrieval.vector_store_reuse",
                category="retrieval",
                duration_sec=0.0,
                status="cached",
                meta={"chunk_card_count": len(cards)},
            )
            return store

        with self.profiler.stage("retrieval.vector_store_reset", category="retrieval"):
            store.reset()
        with self.profiler.stage("retrieval.vector_store_add_cards", category="retrieval", meta={"chunk_card_count": len(cards)}):
            store.add_chunk_cards(video_id=self.video_id, chunk_cards=cards)
        write_json(
            manifest_path,
            {
                "video_id": self.video_id,
                "chunk_card_fingerprint": card_fingerprint,
                "chunk_card_count": len(cards),
                "embedding_model_name": str(self.cfg.embedding_model_name),
            },
        )
        return store

    def _run_thinking_stage(self, draft_summary: str, store: VectorStore) -> str:
        """执行 thinking 复核阶段，输出最终审校版摘要。

        可以把这一阶段理解成“先抽待核验论点，再给论点找证据，最后让模型复核初稿”。
        它的目标不是重新总结一遍，而是提高摘要的可靠性。
        """

        if self.resume_enabled and self.cp_final_reviewed.exists():
            print("[Resume] Thinking 阶段已完成，直接复用缓存")
            self.profiler.record(name="thinking.resume", category="thinking", duration_sec=0.0, status="cached")
            return self.cp_final_reviewed.read_text(encoding="utf-8").strip()

        runner = self._new_runner(self.cfg.thinking_model_path)
        with self.profiler.stage("thinking.model_load", category="model"):
            runner.load()

        try:
            claims = self._load_or_extract_claims(runner, draft_summary)
            if not claims:
                # 如果模型没抽出 claim，就退回一个更简单的启发式方案。
                claims = extract_claim_candidates(draft_summary, max_claims=10)

            evidence_pack = self._load_or_build_evidence_pack(claims, store)

            prompt = build_thinking_review_prompt(
                draft_summary=draft_summary,
                claim_evidence_pack=evidence_pack,
                mystery_mode=self.cfg.mystery_mode,
            )
            with self.profiler.stage("thinking.review_generate", category="generation", meta={"claim_count": len(claims)}):
                final = retry_with_backoff(
                    op_name="thinking_review_generate",
                    fn=lambda: runner.generate(
                        prompt=prompt,
                        images=None,
                        gen_cfg=GenerationConfig(
                            max_new_tokens=self.cfg.max_new_tokens_review,
                            temperature=self.cfg.temperature,
                            top_p=self.cfg.top_p,
                        ),
                    ),
                    max_retries=self.cfg.max_retries,
                    base_delay=self.cfg.retry_base_delay_sec,
                    max_delay=self.cfg.retry_max_delay_sec,
                )
            final = final.strip()
            self.cp_final_reviewed.write_text(final, encoding="utf-8")
            return final
        finally:
            with self.profiler.stage("thinking.model_unload", category="model"):
                runner.unload()

    def _extract_claims(self, runner: QwenVLRunner, draft_summary: str) -> List[str]:
        """从摘要初稿中提取待核验的 claim 列表。"""

        prompt = build_claim_extraction_prompt(draft_summary)
        with self.profiler.stage("thinking.claim_extraction", category="generation"):
            txt = retry_with_backoff(
                op_name="claim_extraction",
                fn=lambda: runner.generate(
                    prompt=prompt,
                    images=None,
                    gen_cfg=GenerationConfig(max_new_tokens=500, temperature=0.0, top_p=1.0),
                ),
                max_retries=self.cfg.max_retries,
                base_delay=self.cfg.retry_base_delay_sec,
                max_delay=self.cfg.retry_max_delay_sec,
            )
        obj = ensure_json_obj(txt)
        # `obj.get("claims", [])` 是字典安全取值；
        # 如果模型没按预期输出 `claims` 字段，就退回空列表。
        claims = obj.get("claims", []) if isinstance(obj, dict) else []
        return [str(c).strip() for c in claims if str(c).strip()]

    def _load_or_extract_claims(self, runner: QwenVLRunner, draft_summary: str) -> List[str]:
        """优先复用 claims checkpoint，否则重新抽取。"""

        if self.resume_enabled and self.cp_claims.exists():
            self.profiler.record(name="thinking.resume_claims", category="thinking", duration_sec=0.0, status="cached")
            obj = read_json(self.cp_claims)
            claims = obj.get("claims", []) if isinstance(obj, dict) else []
            return [str(c).strip() for c in claims if str(c).strip()]

        claims = self._extract_claims(runner, draft_summary)
        write_json(self.cp_claims, {"claims": claims})
        return claims

    def _load_or_build_evidence_pack(self, claims: List[str], store: VectorStore) -> str:
        """优先复用证据包，否则根据 claim 去向量库里检索证据。"""

        if self.resume_enabled and self.cp_evidence_pack.exists():
            self.profiler.record(name="thinking.resume_evidence_pack", category="thinking", duration_sec=0.0, status="cached")
            return self.cp_evidence_pack.read_text(encoding="utf-8")

        with self.profiler.stage("thinking.build_evidence_pack", category="retrieval", meta={"claim_count": len(claims)}):
            evidence_pack = build_claim_evidence_pack(
                store=store,
                claims=claims,
                top_k=self.cfg.retrieval_top_k,
                neighbor_k=self.cfg.retrieval_neighbor_k,
            )
        self.cp_evidence_pack.write_text(evidence_pack, encoding="utf-8")
        return evidence_pack

    def _save_outputs(
        self,
        chunks: List[ChunkMeta],
        cards: List[ChunkCard],
        section_summaries: List[str],
        draft_summary: str,
        final_summary: str,
    ) -> Dict[str, str]:
        """把本次运行的关键结果统一写入 `final_dir`。"""

        manifest_path = self.paths.final_dir / "chunk_manifest.jsonl"
        cards_path = self.paths.final_dir / "chunk_cards.jsonl"
        sections_path = self.paths.final_dir / "section_summaries.md"
        draft_path = self.paths.final_dir / "global_summary_draft.md"
        final_path = self.paths.final_dir / "global_summary_final_reviewed.md"
        chunk_plan_path = self.paths.final_dir / "chunk_plan.json"

        write_jsonl(manifest_path, [c.to_dict() for c in chunks])
        write_jsonl(cards_path, [c.to_dict() for c in cards])

        sections_path.write_text("\n\n".join(section_summaries), encoding="utf-8")
        draft_path.write_text(draft_summary, encoding="utf-8")
        final_path.write_text(final_summary, encoding="utf-8")

        if self.cp_chunk_plan.exists():
            # 如果这次走过了 LLM 语义切段，就把 chunk plan 也同步到最终输出目录，
            # 方便后续排查“为什么是这样切段的”。
            chunk_plan_path.write_text(self.cp_chunk_plan.read_text(encoding="utf-8"), encoding="utf-8")
        elif chunk_plan_path.exists():
            try:
                chunk_plan_path.unlink()
            except Exception:
                pass

        return {
            "manifest": str(manifest_path),
            "chunk_cards": str(cards_path),
            "section_summaries": str(sections_path),
            "global_draft": str(draft_path),
            "global_final": str(final_path),
            "chunk_plan": str(chunk_plan_path) if chunk_plan_path.exists() else "",
            "checkpoints_dir": str(self.paths.checkpoints_dir),
            "timing_events": str(self.profiler.events_path),
            "timing_summary": str(self.profiler.summary_path),
        }

    def _load_or_run_asr(self) -> List[TranscriptSegment]:
        """优先读取缓存 ASR；没有缓存时才真正执行转写。"""

        if self.resume_enabled and self.cp_asr_segments.exists():
            self.profiler.record(name="prepare.resume_asr", category="prepare", duration_sec=0.0, status="cached")
            rows = read_jsonl(self.cp_asr_segments)
            # 这里把 JSONL 里的普通字典重新恢复成 `TranscriptSegment` 对象。
            return [
                TranscriptSegment(start=float(x["start"]), end=float(x["end"]), text=str(x.get("text", "")))
                for x in rows
            ]

        asr_result = retry_with_backoff(
            op_name="asr_transcribe",
            fn=lambda: transcribe_video(
                video_path=self.video_path,
                model_size=self.cfg.asr_model_size,
                language=self.cfg.asr_language,
                use_vad=self.cfg.use_vad,
            ),
            max_retries=self.cfg.max_retries,
            base_delay=self.cfg.retry_base_delay_sec,
            max_delay=self.cfg.retry_max_delay_sec,
        )

        rows = [{"start": s.start, "end": s.end, "text": s.text} for s in asr_result.segments]
        write_jsonl(self.cp_asr_segments, rows)
        return asr_result.segments

    def _read_first_text(self, candidates: List[Path]) -> str:
        """按顺序寻找第一个存在的文本文件并读取。"""

        for p in candidates:
            if p.exists():
                return p.read_text(encoding="utf-8").strip()
        return ""

    def _materialize_answer_clips(self, segments: List[Dict[str, str]], max_items: int = 4) -> List[Dict[str, str]]:
        """把问答阶段选出的时间片段真正裁成视频 clip。

        这属于“把结构化结果落成实际文件”的一步：
        前面只得到时间范围，这里才真正调用视频裁剪函数生成可播放片段。
        """

        clips: List[Dict[str, str]] = []
        # 先去重，再限制最多输出多少个 clip，避免生成太多重复片段。
        for idx, seg in enumerate(dedupe_time_segments(segments, max_items=max_items), start=1):
            t_start_text = str(seg.get("t_start", "")).strip()
            t_end_text = str(seg.get("t_end", "")).strip() or t_start_text
            start_sec = max(0.0, hhmmss_to_seconds(t_start_text))
            end_sec = max(start_sec + 0.5, hhmmss_to_seconds(t_end_text))
            if end_sec - start_sec < 6.0:
                # 如果片段太短，向后扩到 10 秒左右，方便播放时看清上下文。
                end_sec = start_sec + 10.0
                t_end_text = seconds_to_hhmmss(end_sec)
            reason = str(seg.get("reason", "")).strip() or f"相关片段 {idx}"
            source_job_id = str(seg.get("source_job_id", "")).strip()
            source_video_name = str(seg.get("source_video_name", "")).strip()
            source_video_path = str(seg.get("source_video_path", "")).strip() or self.video_path

            clip_id = self._build_answer_clip_id(
                idx=idx,
                t_start=t_start_text,
                t_end=t_end_text,
                reason=reason,
                source_ref=source_job_id or source_video_path,
            )
            clip_path = self.paths.qa_clips_dir / f"{clip_id}.mp4"
            try:
                # `stat().st_size` 是文件大小（字节）。
                # 这里把“文件不存在”或“文件是空的”都视为需要重新裁剪。
                if (not clip_path.exists()) or clip_path.stat().st_size <= 0:
                    extract_video_clip(
                        video_path=source_video_path,
                        t_start=start_sec,
                        t_end=end_sec,
                        out_path=clip_path,
                    )
            except Exception as e:
                print(f"[QA][WARN] extract clip failed: clip_id={clip_id}, err={e}")
                continue

            clips.append(
                {
                    "clip_id": clip_id,
                    "t_start": t_start_text,
                    "t_end": t_end_text,
                    "reason": reason,
                }
            )
            if source_job_id:
                clips[-1]["source_job_id"] = source_job_id
            if source_video_name:
                clips[-1]["source_video_name"] = source_video_name
        return clips

    @staticmethod
    def _build_answer_clip_id(idx: int, t_start: str, t_end: str, reason: str, source_ref: str = "") -> str:
        """根据时间段和原因生成稳定的 clip 文件名。"""

        # 这里不是直接把整段 reason 拼进文件名，而是先做哈希：
        # 这样文件名更短，也能避免特殊字符带来的路径问题。
        base = f"{source_ref}|{idx}|{t_start}|{t_end}|{reason}".encode("utf-8")
        digest = hashlib.sha1(base).hexdigest()[:10]
        safe_start = str(t_start).replace(":", "")
        safe_end = str(t_end).replace(":", "")
        return f"clip_{idx:02d}_{safe_start}_{safe_end}_{digest}"

    @staticmethod
    def _segments_from_evidence_rows(rows: List[Dict[str, Any]], max_items: int = 5) -> List[Dict[str, str]]:
        """把结构化 evidence rows 转成 answer segment 列表。"""

        segments: List[Dict[str, str]] = []
        for row in rows:
            t_start = str(row.get("t_start", "")).strip()
            t_end = str(row.get("t_end", "")).strip() or t_start
            evidence = compact_text(str(row.get("evidence", "")).strip(), 180)
            segments.append(
                {
                    "t_start": t_start,
                    "t_end": t_end,
                    "reason": evidence or "与问题相关的证据片段",
                    "source_job_id": str(row.get("source_job_id", "")).strip(),
                    "source_video_name": str(row.get("source_video_name", "")).strip(),
                    "source_video_path": str(row.get("source_video_path", "")).strip(),
                }
            )
        return dedupe_time_segments(segments, max_items=max_items)

    def _load_chunk_cards_from_any(self) -> List[ChunkCard]:
        """优先从 final 输出读取 chunk cards，拿不到再读 checkpoint。"""

        cp = self.paths.final_dir / "chunk_cards.jsonl"
        if cp.exists():
            # 优先读 `final_dir`，因为那里通常代表“本次成功运行后整理好的最终版本”。
            rows = read_jsonl(cp)
            card_map: Dict[str, ChunkCard] = {}
            for row in rows:
                card = self._chunk_card_from_dict(row)
                card_map[card.chunk_id] = card
            return self._order_cards(card_map.values())
        if self.cp_chunk_cards.exists():
            rows = read_jsonl(self.cp_chunk_cards)
            card_map: Dict[str, ChunkCard] = {}
            for row in rows:
                card = self._chunk_card_from_dict(row)
                card_map[card.chunk_id] = card
            return self._order_cards(card_map.values())
        return []

    def _expected_keyframe_paths(self, chunk: ChunkMeta) -> List[str]:
        """按固定命名规则推导一个 chunk 应该拥有的关键帧路径。"""

        # 这是一个列表推导式，作用是批量生成：
        # `chunk_xxx_kf1.jpg`, `chunk_xxx_kf2.jpg`, ...
        return [
            str(self.paths.keyframes_dir / f"{chunk.chunk_id}_kf{i+1}.jpg")
            for i in range(self.cfg.keyframes_per_chunk)
        ]

    def _recover_keyframe_timestamps(self, chunk: ChunkMeta) -> List[float]:
        """在缺少时间戳时，重新估算该 chunk 的关键帧采样点。"""

        if chunk.keyframe_timestamps:
            return chunk.keyframe_timestamps
        # 如果历史数据里只有图片路径没有时间戳，就按 chunk 范围重新均匀估一个。
        return normalize_keyframe_timestamps(
            chunk_start=chunk.t_start,
            chunk_end=chunk.t_end,
            candidate_timestamps=[],
            desired_count=self.cfg.keyframes_per_chunk,
        )

    def _keyframes_exist(self, chunk: ChunkMeta) -> bool:
        """判断一个 chunk 预期的关键帧文件是否已经全部存在。"""

        paths = self._expected_keyframe_paths(chunk)
        # `all(...)` 表示：只有当里面每一项都为真时，结果才为真。
        return all(Path(p).exists() for p in paths)

    @staticmethod
    def _chunk_meta_from_dict(d: Dict) -> ChunkMeta:
        """把普通字典恢复成 `ChunkMeta` 对象。"""

        # 这里显式做 `str(...)` / `int(...)` / `float(...)` 转换，
        # 是为了把磁盘里的 JSON 数据重新规整成项目内部预期的类型。
        return ChunkMeta(
            chunk_id=str(d["chunk_id"]),
            index=int(d["index"]),
            t_start=float(d["t_start"]),
            t_end=float(d["t_end"]),
            section_id=str(d["section_id"]),
            keyframe_paths=[str(x) for x in d.get("keyframe_paths", [])],
            keyframe_timestamps=[float(x) for x in d.get("keyframe_timestamps", [])],
            asr_text=str(d.get("asr_text", "")),
            ocr_text=str(d.get("ocr_text", "")),
        )

    @staticmethod
    def _chunk_card_from_dict(d: Dict) -> ChunkCard:
        """把普通字典恢复成 `ChunkCard` 对象。"""

        return ChunkCard(
            chunk_id=str(d["chunk_id"]),
            index=int(d["index"]),
            section_id=str(d["section_id"]),
            t_start=str(d["t_start"]),
            t_end=str(d["t_end"]),
            one_liner=str(d.get("one_liner", "")),
            bullets=[str(x) for x in d.get("bullets", [])],
            entities=[str(x) for x in d.get("entities", [])],
            visual_facts=[str(x) for x in d.get("visual_facts", [])],
            quotes=[x for x in d.get("quotes", []) if isinstance(x, dict)],
            importance=safe_float(d.get("importance", 0.0)),
            tags=[str(x) for x in d.get("tags", [])],
            asr_text=str(d.get("asr_text", "")),
            ocr_text=str(d.get("ocr_text", "")),
        )

    @staticmethod
    def _order_cards(cards: Iterable[ChunkCard]) -> List[ChunkCard]:
        """按 chunk 的原始顺序排序。"""

        return sorted(list(cards), key=lambda x: x.index)

    @staticmethod
    def _chunk_cards_fingerprint(cards: List[ChunkCard]) -> str:
        """计算 chunk cards 的内容指纹，用来判断向量库是否还能复用。"""

        # 这里先把对象整理成稳定的普通数据结构，再做哈希。
        # `sort_keys=True` 的作用是让字典键顺序固定，避免“内容没变但哈希变了”。
        payload = [
            {
                "chunk_id": card.chunk_id,
                "index": card.index,
                "section_id": card.section_id,
                "t_start": card.t_start,
                "t_end": card.t_end,
                "one_liner": card.one_liner,
                "bullets": list(card.bullets),
                "entities": list(card.entities),
                "visual_facts": list(card.visual_facts),
                "quotes": list(card.quotes),
                "importance": card.importance,
                "tags": list(card.tags),
                "asr_text": card.asr_text,
                "ocr_text": card.ocr_text,
            }
            for card in cards
        ]
        digest = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return digest

    def _load_chunk_cards_checkpoint(self) -> List[ChunkCard]:
        """读取 chunk card 的断点缓存。"""

        if self.resume_enabled and self.cp_chunk_cards.exists():
            rows = read_jsonl(self.cp_chunk_cards)
            card_map: Dict[str, ChunkCard] = {}
            for row in rows:
                card = self._chunk_card_from_dict(row)
                card_map[card.chunk_id] = card
            return self._order_cards(card_map.values())
        return []

    def _append_chunk_card_checkpoint(self, card: ChunkCard) -> None:
        """以追加模式写入单条 chunk card checkpoint。"""

        self.cp_chunk_cards.parent.mkdir(parents=True, exist_ok=True)
        # 这里使用追加模式 `"a"`，这样每生成一张卡就能立刻落盘，
        # 中途失败时，前面已经完成的结果也不会丢。
        with self.cp_chunk_cards.open("a", encoding="utf-8") as f:
            f.write(json.dumps(card.to_dict(), ensure_ascii=False) + "\n")

    @staticmethod
    def _parallel_worker_count(runtime_name: str, default_limit: int) -> int:
        """决定某类并行任务应该开几个 worker。"""

        configured = runtime_int(runtime_name, None)
        if configured is not None:
            # 允许通过运行时配置手动覆盖 worker 数。
            return max(1, int(configured))
        cpu_count = max(1, (os.cpu_count() or 1))
        return max(1, min(default_limit, cpu_count))

    def _run_chunk_workers(
        self,
        chunks: List[ChunkMeta],
        total_chunks: int,
        worker_count: int,
        worker_fn: Callable[[ChunkMeta, int, int], None],
    ) -> None:
        """并行执行 chunk 级任务，例如抽帧或 OCR。

        这里封装了一个通用的“并行跑每个 chunk 的小任务”的模式：
        - 如果任务很少，就串行跑
        - 如果任务较多，就交给线程池并发执行
        - 只要有一个 worker 报错，就尽快中止其他任务
        """

        if not chunks:
            return

        tasks = [(idx, chunk) for idx, chunk in enumerate(chunks, start=1)]
        actual_workers = max(1, min(int(worker_count), len(tasks)))
        if actual_workers <= 1:
            # worker 数只有 1 时直接串行执行，逻辑更简单。
            for idx, chunk in tasks:
                worker_fn(chunk, idx, total_chunks)
            return

        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            future_map = {
                executor.submit(worker_fn, chunk, idx, total_chunks): (idx, chunk.chunk_id)
                for idx, chunk in tasks
            }
            # `wait(..., return_when=FIRST_EXCEPTION)` 的意思是：
            # 只要有任意一个 future 先抛异常，就立刻返回，不再傻等全部完成。
            done, pending = wait(future_map.keys(), return_when=FIRST_EXCEPTION)

            first_error: Optional[BaseException] = None
            first_error_meta: Optional[Tuple[int, str]] = None
            for future in done:
                try:
                    future.result()
                except Exception as exc:
                    first_error = exc
                    first_error_meta = future_map[future]
                    break

            if first_error is not None:
                # 任意一个 worker 失败，就尽快取消剩余任务，避免继续浪费资源。
                for future in pending:
                    future.cancel()
                chunk_index, chunk_id = first_error_meta or (-1, "")
                # `raise ... from first_error` 会保留原始异常链。
                # 这样外层既能看到“统一包装后的报错信息”，也能追到真正的根因。
                raise RuntimeError(
                    f"parallel chunk worker failed: chunk_id={chunk_id}, chunk_index={chunk_index}"
                ) from first_error

            for future in pending:
                future.result()

    def _prepare_keyframes_for_chunk(self, chunk: ChunkMeta, chunk_index: int, total_chunks: int) -> None:
        """为单个 chunk 抽取关键帧。"""

        print(f"[Prepare] 抽帧 {chunk_index}/{total_chunks}: {chunk.chunk_id}")
        try:
            with self.profiler.stage(
                "prepare.extract_keyframes.chunk",
                category="prepare_chunk",
                meta={
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk_index,
                    "chunk_start_sec": chunk.t_start,
                    "chunk_end_sec": chunk.t_end,
                },
            ):
                if self.resume_enabled and self._keyframes_exist(chunk):
                    # 断点续跑场景下，如果关键帧图片已经存在，就直接复用。
                    chunk.keyframe_paths = self._expected_keyframe_paths(chunk)
                    if not chunk.keyframe_timestamps:
                        chunk.keyframe_timestamps = self._recover_keyframe_timestamps(chunk)
                    return

                if chunk.keyframe_timestamps:
                    # 如果上游已经决定了候选时间点，就按指定时间戳抽帧。
                    paths, selected_ts = retry_with_backoff(
                        op_name=f"extract_keyframes:{chunk.chunk_id}",
                        fn=lambda chunk=chunk: extract_keyframes_at_timestamps(
                            video_path=self.video_path,
                            chunk=chunk,
                            timestamps=chunk.keyframe_timestamps,
                            out_dir=self.paths.keyframes_dir,
                            keyframes_per_chunk=self.cfg.keyframes_per_chunk,
                            keyframe_max_width=self.cfg.keyframe_max_width,
                        ),
                        max_retries=self.cfg.max_retries,
                        base_delay=self.cfg.retry_base_delay_sec,
                        max_delay=self.cfg.retry_max_delay_sec,
                    )
                    chunk.keyframe_paths = paths
                    chunk.keyframe_timestamps = selected_ts
                    return

                # 否则走“自动从该 chunk 中挑关键帧”的常规路径。
                chunk.keyframe_paths = retry_with_backoff(
                    op_name=f"extract_keyframes:{chunk.chunk_id}",
                    fn=lambda chunk=chunk: extract_keyframes_for_chunk(
                        video_path=self.video_path,
                        chunk=chunk,
                        keyframes_per_chunk=self.cfg.keyframes_per_chunk,
                        out_dir=self.paths.keyframes_dir,
                        keyframe_max_width=self.cfg.keyframe_max_width,
                    ),
                    max_retries=self.cfg.max_retries,
                    base_delay=self.cfg.retry_base_delay_sec,
                    max_delay=self.cfg.retry_max_delay_sec,
                )
                chunk.keyframe_timestamps = self._recover_keyframe_timestamps(chunk)
        except Exception as e:
            print(f"[Prepare][WARN] 抽帧失败 {chunk.chunk_id}: {e}")
            fallback_paths = [p for p in self._expected_keyframe_paths(chunk) if Path(p).exists()]
            if fallback_paths:
                chunk.keyframe_paths = fallback_paths
                chunk.keyframe_timestamps = normalize_keyframe_timestamps(
                    chunk_start=chunk.t_start,
                    chunk_end=chunk.t_end,
                    candidate_timestamps=chunk.keyframe_timestamps,
                    desired_count=len(fallback_paths),
                )
                print(f"[Prepare][WARN] 使用已有关键帧 {len(fallback_paths)} 张继续")
            else:
                chunk.keyframe_paths = []
                chunk.keyframe_timestamps = []
                print(f"[Prepare][WARN] {chunk.chunk_id} 无可用关键帧，回退纯文本继续")

    def _ocr_chunk_keyframes(self, chunk: ChunkMeta, chunk_index: int, total_chunks: int) -> None:
        """对单个 chunk 的关键帧执行 OCR。"""

        print(f"[Prepare] OCR {chunk_index}/{total_chunks}: {chunk.chunk_id}")
        with self.profiler.stage(
            "prepare.ocr.chunk",
            category="prepare_chunk",
            meta={"chunk_id": chunk.chunk_id, "chunk_index": chunk_index, "keyframe_count": len(chunk.keyframe_paths)},
        ):
            retry_with_backoff(
                op_name=f"ocr:{chunk.chunk_id}",
                fn=lambda chunk=chunk: ocr_for_chunk_keyframes(chunk, lang=self.cfg.ocr_lang),
                max_retries=self.cfg.max_retries,
                base_delay=self.cfg.retry_base_delay_sec,
                max_delay=self.cfg.retry_max_delay_sec,
            )

    def _load_section_summaries_checkpoint_map(self) -> Dict[str, str]:
        """读取 section summary 的断点缓存，并保持 section_id -> summary 的映射。"""

        if self.resume_enabled and self.cp_sections.exists():
            obj = read_json(self.cp_sections)
            if isinstance(obj, dict):
                return {str(k): str(v) for k, v in obj.items()}
        return {}

    def _load_section_summaries_checkpoint(self) -> List[str]:
        """读取 section summary 断点缓存，并按 section_id 排序输出。"""

        m = self._load_section_summaries_checkpoint_map()
        return [m[k] for k in sorted(m.keys())]

    def _invalidate_thinking_checkpoints(self) -> None:
        """当 draft 变化后，清掉依赖旧 draft 的 thinking 阶段缓存。"""

        # 这是典型的“下游缓存失效”逻辑：
        # 上游 draft 一旦变化，基于旧 draft 算出的 claim / evidence / final 都不能直接复用。
        for p in [self.cp_claims, self.cp_evidence_pack, self.cp_final_reviewed]:
            if p.exists():
                p.unlink()

    def _build_run_signature(self) -> Dict:
        """根据视频文件和关键配置生成运行签名，用于判断能否 resume。"""

        p = Path(self.video_path)
        stat = p.stat()
        payload = {
            "video_path": str(p.resolve()),
            "video_size": int(stat.st_size),
            "video_mtime": int(stat.st_mtime),
            "enable_llm_guided_chunking": self.cfg.enable_llm_guided_chunking,
            "chunk_seconds": self.cfg.chunk_seconds,
            "overlap_seconds": self.cfg.overlap_seconds,
            "llm_chunk_min_seconds": self.cfg.llm_chunk_min_seconds,
            "llm_chunk_max_seconds": self.cfg.llm_chunk_max_seconds,
            "llm_chunk_overlap_seconds": self.cfg.llm_chunk_overlap_seconds,
            "section_minutes": self.cfg.section_minutes,
            "keyframes_per_chunk": self.cfg.keyframes_per_chunk,
            "keyframe_candidates_per_chunk": self.cfg.keyframe_candidates_per_chunk,
            "asr_model_size": self.cfg.asr_model_size,
            "asr_language": self.cfg.asr_language,
            "use_vad": self.cfg.use_vad,
            "ocr_lang": self.cfg.ocr_lang,
            "instruct_model_path": self.cfg.instruct_model_path,
            "thinking_model_path": self.cfg.thinking_model_path,
        }
        # 这里把“视频本身 + 关键配置”一起做哈希，
        # 目的是判断当前运行环境是否与上一次保持一致。
        h = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        return {"signature": h, "payload": payload}

    def _new_runner(self, model_path: str) -> QwenVLRunner:
        """按当前配置创建一个模型运行器。"""

        return QwenVLRunner(
            model_path=model_path,
            device_map=self.cfg.device_map,
            torch_dtype=self.cfg.torch_dtype,
            attn_implementation=self.cfg.attn_implementation,
            enable_dual_gpu_balance=self.cfg.enable_dual_gpu_balance,
            gpu0_mem_fraction=self.cfg.gpu0_mem_fraction,
            gpu1_mem_fraction=self.cfg.gpu1_mem_fraction,
            use_free_gpu_memory=self.cfg.use_free_gpu_memory,
            gpu_reserve_memory_mib=self.cfg.gpu_reserve_memory_mib,
            cpu_offload_max_memory=self.cfg.cpu_offload_max_memory,
        )

    def _cleanup_staged_instruct_runner(self) -> None:
        """安全释放暂存的 instruct runner。"""

        runner = self._staged_instruct_runner
        self._staged_instruct_runner = None
        if runner is None:
            return
        try:
            runner.unload()
        except Exception as e:
            print(f"[Cleanup][WARN] staged instruct runner unload failed: {e}")

    def _initialize_resume_mode(self) -> None:
        """决定本次运行是否允许断点续跑。

        断点续跑不是“只要有缓存就继续”，而是要先确认：
        当前视频和关键配置是否与上次一致。
        否则继续复用旧中间结果，反而可能得到错误输出。
        """

        if not self.cfg.resume:
            self.resume_enabled = False
            return

        current = self._build_run_signature()
        if not self.cp_run_meta.exists():
            # 第一次运行时没有旧签名，先写入当前签名，后续才能拿来比较。
            write_json(self.cp_run_meta, current)
            self.resume_enabled = True
            return

        try:
            old = read_json(self.cp_run_meta)
        except Exception:
            old = {}

        old_sig = str(old.get("signature", ""))
        new_sig = str(current.get("signature", ""))
        if old_sig == new_sig:
            # 只有签名完全一致，才说明“视频和关键参数都没变”。
            self.resume_enabled = True
            return

        print("[Resume] 检测到视频/关键配置变化，自动禁用续跑并执行全量重跑。")
        self.resume_enabled = False
        # 即使这次禁用了 resume，也要把新的签名写回去，
        # 这样下一次再运行时，比较基准就是最新的配置。
        write_json(self.cp_run_meta, current)

    def _emit_progress(
        self,
        stage_index: int,
        stage_name: str,
        message: str,
        total_stages: int = 6,
        force_percent: Optional[float] = None,
    ) -> None:
        """把当前阶段进度通过回调函数发给外部界面。"""

        if self.progress_callback is None:
            return

        if force_percent is None:
            # `max(..., min(...))` 是把值夹在 0 到 100 之间的常见写法，
            # 避免出现负数或大于 100 的进度。
            progress_percent = round(max(0.0, min(100.0, stage_index / total_stages * 100.0)), 2)
        else:
            progress_percent = round(max(0.0, min(100.0, force_percent)), 2)

        payload = {
            "stage_index": stage_index,
            "total_stages": total_stages,
            "stage_name": stage_name,
            "message": message,
            "progress_percent": progress_percent,
        }
        try:
            self.progress_callback(payload)
        except Exception:
            # 进度回调失败不应该影响主流程，所以这里选择吞掉异常。
            pass

    def _record_upload_timing(self) -> None:
        """如果这次任务来自上传文件，则把上传耗时也记进 profiler。"""

        if self.cfg.upload_duration_sec is None:
            return

        meta: Dict[str, Any] = {}
        if self.cfg.upload_size_bytes is not None:
            meta["upload_size_bytes"] = int(self.cfg.upload_size_bytes)
        if self.cfg.upload_filename:
            meta["upload_filename"] = str(self.cfg.upload_filename)

        # 这条记录不是模型阶段本身，而是 API 层的“上传耗时”埋点。
        self.profiler.record(
            name="api.upload",
            category="api",
            duration_sec=float(self.cfg.upload_duration_sec),
            status="ok",
            meta=meta,
        )
