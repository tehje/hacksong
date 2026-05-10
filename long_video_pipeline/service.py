from __future__ import annotations

"""FastAPI 服务层。

如果把整个项目想成一家工厂：

1. `pipeline.py` 像生产线，负责真正处理视频。
2. `service.py` 像前台 + 调度室，负责接请求、排队、记录任务状态、把结果再返回给前端。

新手读这个文件，建议先抓住 3 条主线：

1. 先看 `CreateJobRequest`、`JobStatusResponse` 这些 `BaseModel`。
   它们定义了“接口收什么数据、回什么数据”。
2. 再看 `JobManager`。
   它是核心调度器，负责提交任务、运行任务、保存任务状态、回答 QA。
3. 最后看文件底部的 `@app.get(...)` / `@app.post(...)`。
   这些是 FastAPI 路由，决定了 HTTP 接口地址。

这个文件里会反复出现几种 Python 语法：

- `x: str = "..."`：类型提示 + 默认值。主要帮助人和 IDE 理解代码。
- `Optional[X]`：值可能是 `X`，也可能是 `None`。
- `Dict[str, Any]`：字典，键是字符串，值可以是任意类型。
- `Field(...)`：Pydantic 的字段声明写法。`...` 表示“这是必填项”。
- `func(**data)`：把一个字典拆开成关键字参数传进去，常见于“把请求字典重新组装成模型对象”。
- `A if 条件 else B`：Python 条件表达式。满足条件时取 `A`，否则取 `B`。
- `with self.lock:`：加锁。意思是“接下来这一小段代码，同一时刻只允许一个线程进入”。
- `ThreadPoolExecutor`：线程池。这里用它在后台跑视频任务，避免 HTTP 请求线程被长时间阻塞。
- `async def` / `await`：异步函数写法，适合文件上传、网络请求这类 I/O 操作。
- `raise HTTPException(...)`：主动返回一个 HTTP 错误给前端。
"""

import os
import shutil
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
from threading import Event, Lock, Thread
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .settings import (
    DEFAULT_DATABASE_DIR,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_INSTRUCT_MODEL_PATH,
    DEFAULT_QA_MODEL_PATH,
    DEFAULT_THINKING_MODEL_PATH,
    PipelineConfig,
    RUNTIME_PROFILE_PATH,
    runtime_bool,
    runtime_str,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# API 任务默认输出目录，例如 `outputs/api_jobs/<job_id>/...`
DEFAULT_JOB_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "api_jobs"
# 全局任务注册表。服务重启后，会优先从这里恢复任务信息。
JOB_REGISTRY_PATH = PROJECT_ROOT / "outputs" / "job_registry.json"
# 每个任务目录内自己的任务记录文件名。
JOB_RECORD_FILENAME = "job_record.json"
# 老版本项目路径里可能含有旧项目名，这里用来做兼容替换。
LEGACY_PROJECT_MARKER = PROJECT_ROOT.name


def _warm_asr_model_file_cache(model_size_or_path: str) -> str:
    from .asr_ocr import warm_asr_model_file_cache

    return warm_asr_model_file_cache(model_size_or_path)


def _prime_retrieval_import():
    from . import retrieval as retrieval_module

    return retrieval_module


def _prepare_model_artifacts(model_path: str) -> str:
    _prime_retrieval_import()
    from .model_runtime import prepare_model_artifacts

    return prepare_model_artifacts(model_path)


def _warm_model_file_cache(model_path: str) -> str:
    _prime_retrieval_import()
    from .model_runtime import warm_model_file_cache

    return warm_model_file_cache(model_path)


def _build_multi_store_question_evidence_pack(*args, **kwargs):
    from .retrieval import build_multi_store_question_evidence_pack

    return build_multi_store_question_evidence_pack(*args, **kwargs)


def _preload_embedding_model(model_name_or_path: str) -> str:
    from .retrieval import preload_embedding_model

    return preload_embedding_model(model_name_or_path)


def _vector_store_cls():
    from .retrieval import VectorStore

    return VectorStore


def _long_video_summary_pipeline_cls():
    _prime_retrieval_import()
    from .pipeline import LongVideoSummaryPipeline

    return LongVideoSummaryPipeline


def _shared_qwen_vl_runner_pool_cls():
    _prime_retrieval_import()
    from .model_runtime import SharedQwenVLRunnerPool

    return SharedQwenVLRunnerPool


def _rewrite_legacy_project_path(path_value: Any) -> str:
    """把旧工程路径改写成当前工程路径。

    这个函数主要是做“兼容老数据”：
    如果磁盘上的历史记录里写的是旧目录，就尽量改写到当前项目根目录下。
    """

    # `path_value or ""` 表示：如果传进来的是空值（如 `None`），
    # 就先退回空字符串，避免后面 `Path(...)` 直接报错。
    text = str(path_value or "").strip()
    if not text:
        return text

    try:
        normalized = str(Path(text).expanduser())
        # `Path(...).parts` 会把路径拆成一个元组，便于后面按目录层级查找旧项目名。
        path_parts = Path(normalized).parts
    except Exception:
        return text

    if LEGACY_PROJECT_MARKER not in path_parts:
        return normalized

    # `path_parts[::-1]` 是切片写法，表示“把序列倒过来”。
    # 这里从后往前找旧项目名，是为了兼容路径里可能出现多次同名目录的情况。
    marker_index = len(path_parts) - 1 - path_parts[::-1].index(LEGACY_PROJECT_MARKER)
    suffix_parts = path_parts[marker_index + 1 :]
    if not suffix_parts:
        return str(PROJECT_ROOT)

    return str(PROJECT_ROOT.joinpath(*suffix_parts))


def _rewrite_record_request_paths(request_obj: Dict[str, Any]) -> Dict[str, Any]:
    """批量修正任务请求对象里的路径字段。"""

    if not isinstance(request_obj, dict):
        return {}

    normalized = dict(request_obj)
    path_like_keys = [
        "video_path",
        "output_root",
        "chroma_dir",
        "embedding_model_name",
        "instruct_model_path",
        "thinking_model_path",
        "qa_model_path",
    ]
    for key in path_like_keys:
        if normalized.get(key) is not None:
            normalized[key] = _rewrite_legacy_project_path(normalized[key])
    return normalized


def _force_single_gpu_request(request_obj: Dict[str, Any]) -> Dict[str, Any]:
    """强制把任务请求改成单卡模式。

    这里不是“用户不能配置”，而是服务层统一收口，避免多任务情况下 GPU 分配更复杂。
    """

    if not isinstance(request_obj, dict):
        return {"device_map": "none", "enable_dual_gpu_balance": False}

    normalized = dict(request_obj)
    normalized["device_map"] = "none"
    normalized["enable_dual_gpu_balance"] = False
    return normalized


def _rewrite_output_paths(outputs_obj: Dict[str, Any]) -> Dict[str, str]:
    """修正输出结果字典里的路径。"""

    if not isinstance(outputs_obj, dict):
        return {}

    normalized: Dict[str, str] = {}
    for key, value in outputs_obj.items():
        if value is None:
            normalized[str(key)] = ""
        else:
            normalized[str(key)] = _rewrite_legacy_project_path(value)
    return normalized


def _resolve_video_path(path_value: Any) -> Path:
    """把输入的视频路径解析成 `Path` 对象。

    注意：
    - `Path` 是 `pathlib` 里的路径对象，比纯字符串更适合做路径拼接。
    - 这里会先尝试原路径，再尝试“相对项目根目录”的路径。
    """

    raw = _rewrite_legacy_project_path(path_value)
    candidate = Path(raw).expanduser()

    if candidate.exists():
        return candidate

    if not candidate.is_absolute():
        project_candidate = PROJECT_ROOT / candidate
        if project_candidate.exists():
            return project_candidate

    return candidate


def _resolve_existing_or_candidate_path(path_value: Any) -> Optional[Path]:
    """把值转成路径；如果输入为空，就返回 `None`。"""

    raw = _rewrite_legacy_project_path(path_value)
    if not raw:
        return None
    return Path(raw).expanduser()


def _safe_resolve(path_value: Path) -> Path:
    """尽量把路径转成绝对规范路径；失败就退回原值。"""

    try:
        return path_value.resolve()
    except Exception:
        return path_value


def _is_path_within(path_value: Path, root: Path) -> bool:
    """判断一个路径是否位于另一个根目录下面。"""

    try:
        _safe_resolve(path_value).relative_to(_safe_resolve(root))
        return True
    except Exception:
        return False


def _remove_tree_if_exists(path_value: Optional[Path]) -> str:
    """如果目录存在，就整棵删除，并返回删除的路径字符串。"""

    if path_value is None:
        return ""
    target = Path(path_value)
    if not target.exists():
        return ""
    shutil.rmtree(target)
    return str(target)


def _utc_now() -> str:
    """返回 UTC 时间字符串，例如 `2026-04-04T12:34:56Z`。"""

    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _model_dump_compat(m: BaseModel) -> Dict[str, Any]:
    """兼容 Pydantic v1 / v2 的导出方法。

    - v2 常用 `model_dump()`
    - v1 常用 `dict()`
    """

    if hasattr(m, "model_dump"):
        return m.model_dump()
    return m.dict()


def _to_bool(v: Any, default: bool = False) -> bool:
    """把各种输入尽量转换成布尔值。

    例如 `"true"`、`1`、`"on"` 都会被当成 `True`。
    """

    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _normalize_retrieval_scope(value: Any) -> str:
    """把不同写法的检索范围统一成标准值。"""

    raw = str(value or "job").strip().lower()
    if raw in {"job", "single", "current", "local"}:
        return "job"
    if raw in {"all", "global", "full"}:
        return "all"
    # 这里故意抛错而不是静默兜底，
    # 因为检索范围写错时，继续执行反而更容易让调用方误解结果。
    raise ValueError("retrieval_scope must be one of: job, all")


class CreateJobRequest(BaseModel):
    """创建视频任务时的请求体。

    `BaseModel` 是 Pydantic 的数据模型：
    - 它会帮你校验字段类型
    - 也会自动生成接口文档
    - FastAPI 能直接拿它作为请求参数
    """

    # `Field(..., ...)` 里的 `...` 不是省略号文本，而是 Pydantic 约定的“必填”标记。
    video_path: str = Field(..., description="Local path of video")
    output_root: str = Field(default="outputs/api_jobs", description="Root dir for job outputs")
    mystery_mode: bool = False
    resume: bool = True

    # 模型引导的切分与关键帧规划。
    # 这一组参数控制“更智能”的分段方式。
    enable_llm_guided_chunking: bool = True
    llm_chunk_min_seconds: int = 30
    llm_chunk_max_seconds: int = 120
    llm_chunk_overlap_seconds: int = 1

    # 固定时长切片的兜底参数。
    # 当不走模型引导分段时，就主要依赖这里。
    chunk_seconds: int = 120
    overlap_seconds: int = 1
    section_minutes: int = 10

    # 关键帧抽取相关参数。
    keyframes_per_chunk: int = 2
    keyframe_candidates_per_chunk: int = 6
    keyframe_max_width: int = 1280

    # ASR / OCR：语音识别与画面文字识别。
    asr_model_size: str = "large-v3"
    # `Optional[str]` 表示这里既可以是字符串，也可以是 `None`。
    asr_language: Optional[str] = "zh"
    use_vad: bool = True
    ocr_lang: str = "chi_sim+eng"

    # 检索相关参数。
    chroma_dir: Optional[str] = None
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    retrieval_top_k: int = 5
    retrieval_neighbor_k: int = 1

    # 文本生成参数。
    max_new_tokens_chunk: int = 512
    max_new_tokens_section: int = 900
    max_new_tokens_global: int = 1200
    max_new_tokens_review: int = 1400
    temperature: float = 0.2
    top_p: float = 0.9

    # 问答生成参数。
    qa_max_new_tokens: int = 384
    qa_temperature: float = 0.1
    qa_top_p: float = 0.9
    qa_retrieval_top_k: int = 3
    qa_retrieval_neighbor_k: int = 0

    # 稳定性 / 重试参数。
    max_retries: int = 2
    retry_base_delay_sec: float = 2.0
    retry_max_delay_sec: float = 20.0

    # 上传统计信息。
    # 这组字段主要是服务端记录上传耗时和文件大小，不是流水线核心逻辑。
    upload_duration_sec: Optional[float] = None
    upload_size_bytes: Optional[int] = None
    upload_filename: Optional[str] = None

    # 运行时 / 设备参数。
    enable_dual_gpu_balance: bool = False
    device_map: str = "none"
    torch_dtype: str = "float16"
    attn_implementation: Optional[str] = None
    gpu0_mem_fraction: float = 0.40
    gpu1_mem_fraction: float = 0.55
    use_free_gpu_memory: bool = True
    gpu_reserve_memory_mib: int = 1536
    cpu_offload_max_memory: str = "64GiB"

    instruct_model_path: str = DEFAULT_INSTRUCT_MODEL_PATH
    thinking_model_path: str = DEFAULT_THINKING_MODEL_PATH
    qa_model_path: str = DEFAULT_QA_MODEL_PATH


class JobStatusResponse(BaseModel):
    """任务状态接口返回的数据结构。

    前端轮询 `/jobs/{job_id}` 时，主要拿到的就是这一类数据。
    """

    job_id: str
    status: str
    message: str = ""
    created_at: str
    updated_at: str
    request: Dict[str, Any]
    # `default_factory=dict` 表示“默认给一个新的空字典”。
    # 这样可以避免多个对象意外共享同一个默认字典。
    outputs: Dict[str, str] = Field(default_factory=dict)
    error: str = ""

    stage_name: str = "queued"
    stage_index: int = 0
    total_stages: int = 6
    progress_percent: float = 0.0


class QARequest(BaseModel):
    """问答接口的请求体。"""

    question: str = Field(..., description="自然语言问题")
    # 这里写成 `Optional[bool]`，是因为前端可以“不传这个值”，
    # 让它沿用任务本身的配置。
    mystery_mode: Optional[bool] = Field(default=None, description="可选覆盖任务的 mystery_mode")


class QASegment(BaseModel):
    """答案里引用到的时间片段。"""

    t_start: str
    t_end: str
    reason: str = ""
    source_job_id: str = ""
    source_video_name: str = ""


class QAClip(BaseModel):
    """答案里生成的视频片段信息。"""

    clip_id: str
    t_start: str
    t_end: str
    reason: str = ""
    clip_job_id: str = ""
    source_job_id: str = ""
    source_video_name: str = ""


class QAResponse(BaseModel):
    """问答接口返回的数据结构。"""

    job_id: str
    question: str
    retrieval_scope: str = "job"
    answer_markdown: str
    answer_segments: List[QASegment] = Field(default_factory=list)
    answer_clips: List[QAClip] = Field(default_factory=list)
    generated_at: str


class DeleteJobResponse(BaseModel):
    """删除任务接口返回的数据结构。"""

    job_id: str
    deleted: bool = True
    status_before_delete: str
    deleted_output_dir: str = ""
    deleted_upload_dir: str = ""
    retained_video_path: str = ""
    deleted_at: str


class RuntimePreloadRequest(BaseModel):
    """预热运行时的请求体。"""

    job_id: Optional[str] = Field(default=None, description="可选任务 ID；若已完成则按该任务的 QA 配置预热")


class JobManager:
    """任务调度器。

    可以把这个类理解成“服务层的大管家”：

    - `submit()`：接收新任务
    - `_run_job()`：在后台真正执行任务
    - `get()` / `list()`：查询任务状态
    - `ask()`：对已完成任务发起问答
    - `delete()` / `cancel()`：删除或取消任务

    这里使用单 worker（`max_workers=1`）的核心原因是：
    大模型推理非常吃显存，同时跑多个任务容易把 GPU 撑爆。
    """

    def __init__(self, max_workers: int = 1) -> None:
        # 线程池：真正的视频处理任务会被丢到后台线程执行。
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        # `lock` 保护任务字典 `self.jobs`，避免多线程同时修改。
        self.lock = Lock()
        # `model_lock` 保护模型加载 / 卸载过程，避免多个线程争抢模型资源。
        self.model_lock = Lock()
        # 运行时预热状态需要和任务状态分开管理，避免互相阻塞。
        self.runtime_lock = Lock()
        self.runtime_warmup_done = Event()
        self.runtime_warmup_done.set()
        self.runtime_warmup_thread: Optional[Thread] = None
        # `Dict[str, Dict[str, Any]]` 可以理解成：
        # 最外层是“job_id -> 任务记录”的映射。
        self.jobs: Dict[str, Dict[str, Any]] = {}
        # `Future` 代表一个“已经提交到后台、未来会有结果”的任务句柄。
        self.futures: Dict[str, Future] = {}
        self.default_output_root = DEFAULT_JOB_OUTPUT_ROOT
        self.registry_path = JOB_REGISTRY_PATH
        self.qa_runtime_pool = None
        self.runtime_state = self._default_runtime_state()
        self._load_persisted_jobs()

    def _default_runtime_state(self) -> Dict[str, Any]:
        return {
            "warmup_status": "idle",
            "started_at": None,
            "finished_at": None,
            "last_error": "",
            "requested_job_id": None,
            "job_id": None,
            "source": None,
            "reason": "",
            "qa_model_path": "",
            "embedding_model_name": "",
            "embedding_model_resolved": "",
            "runner_reused": False,
            "qa_preloaded": False,
            "updated_at": _utc_now(),
        }

    def _set_runtime_state_locked(self, **updates: Any) -> None:
        self.runtime_state.update(updates)
        self.runtime_state["updated_at"] = _utc_now()

    def _runtime_snapshot_locked(self) -> Dict[str, Any]:
        return dict(self.runtime_state)

    def runtime_status(self) -> Dict[str, Any]:
        with self.runtime_lock:
            return self._runtime_snapshot_locked()

    def _runtime_response_from_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": str(snapshot.get("warmup_status", "idle")),
            "source": snapshot.get("source"),
            "reason": snapshot.get("reason", ""),
            "requested_job_id": snapshot.get("requested_job_id"),
            "job_id": snapshot.get("job_id"),
            "qa_model_path": snapshot.get("qa_model_path", ""),
            "embedding_model_name": snapshot.get("embedding_model_name", ""),
            "embedding_model_resolved": snapshot.get("embedding_model_resolved", ""),
            "runner_reused": bool(snapshot.get("runner_reused", False)),
            "qa_preloaded": bool(snapshot.get("qa_preloaded", False)),
            "time": snapshot.get("updated_at"),
            **snapshot,
        }

    def _mark_runtime_warming_locked(
        self,
        cfg: PipelineConfig,
        requested_job_id: Optional[str],
        resolved_job_id: Optional[str],
        source: str,
        reason: str,
    ) -> None:
        self.runtime_warmup_done.clear()
        self._set_runtime_state_locked(
            warmup_status="warming",
            started_at=_utc_now(),
            finished_at=None,
            last_error="",
            requested_job_id=requested_job_id,
            job_id=resolved_job_id,
            source=source,
            reason=reason,
            qa_model_path=cfg.qa_model_path,
            embedding_model_name=cfg.embedding_model_name,
            embedding_model_resolved="",
            runner_reused=False,
            qa_preloaded=False,
        )

    def _finish_runtime_warmup(
        self,
        *,
        status: str,
        last_error: str = "",
        embedder_resolved: str = "",
        runner_reused: bool = False,
        qa_preloaded: bool = False,
    ) -> Dict[str, Any]:
        with self.runtime_lock:
            self._set_runtime_state_locked(
                warmup_status=status,
                finished_at=_utc_now(),
                last_error=last_error,
                embedding_model_resolved=embedder_resolved,
                runner_reused=bool(runner_reused),
                qa_preloaded=bool(qa_preloaded),
            )
            self.runtime_warmup_thread = None
            snapshot = self._runtime_snapshot_locked()
            self.runtime_warmup_done.set()
        return self._runtime_response_from_snapshot(snapshot)

    def _default_preload_cfg(self) -> PipelineConfig:
        """构造一个“默认预热配置”。

        作用：即使用户没有指定某个已完成任务，也能先把默认 QA 模型和 embedding 模型加载起来。
        """

        return PipelineConfig(
            video_path="",
            output_dir=str(self.default_output_root),
            qa_model_path=DEFAULT_QA_MODEL_PATH,
            embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
        )

    def _resolve_preload_request(
        self,
        job_id: Optional[str] = None,
    ) -> tuple[PipelineConfig, Optional[str], str, Optional[str]]:
        cfg, resolved_job_id, source = self._resolve_preload_cfg(job_id)
        requested_job_id = str(job_id or "").strip() or None
        return cfg, resolved_job_id, source, requested_job_id

    def _resolve_preload_cfg(self, job_id: Optional[str] = None) -> tuple[PipelineConfig, Optional[str], str]:
        """决定预热时应该使用哪份配置。

        返回一个三元组：
        1. `PipelineConfig`
        2. 实际命中的任务 ID（可能为 `None`）
        3. 来源字符串：`"job"` 或 `"default"`
        """

        requested_job_id = str(job_id or "").strip()
        if not requested_job_id:
            return self._default_preload_cfg(), None, "default"

        with self.lock:
            # 这里复制一份记录，避免锁释放后原对象又被别的线程改掉。
            rec = self.jobs.get(requested_job_id)
            rec_copy = dict(rec) if rec is not None else None

        if rec_copy and str(rec_copy.get("status", "")) == "succeeded":
            return self._cfg_from_record(rec_copy), requested_job_id, "job"

        return self._default_preload_cfg(), None, "default"

    def _get_qa_runtime_pool(self):
        with self.runtime_lock:
            if self.qa_runtime_pool is None:
                self.qa_runtime_pool = _shared_qwen_vl_runner_pool_cls()()
            return self.qa_runtime_pool

    def _unload_all_qa_runtimes(self) -> int:
        with self.runtime_lock:
            pool = self.qa_runtime_pool
        if pool is None:
            return 0
        return pool.unload_all()

    def _get_shared_qa_runner(self, cfg: PipelineConfig):
        """获取一个可复用的 QA 模型运行时。

        `SharedQwenVLRunnerPool` 会尽量复用已经加载过的模型，
        避免每次问答都重新加载一次大模型。
        """

        return self._get_qa_runtime_pool().get_or_load(
            model_path=cfg.qa_model_path,
            device_map=cfg.device_map,
            torch_dtype=cfg.torch_dtype,
            attn_implementation=cfg.attn_implementation,
            enable_dual_gpu_balance=cfg.enable_dual_gpu_balance,
            gpu0_mem_fraction=cfg.gpu0_mem_fraction,
            gpu1_mem_fraction=cfg.gpu1_mem_fraction,
            use_free_gpu_memory=cfg.use_free_gpu_memory,
            gpu_reserve_memory_mib=cfg.gpu_reserve_memory_mib,
            cpu_offload_max_memory=cfg.cpu_offload_max_memory,
        )

    def _runtime_ready_for_cfg_locked(
        self,
        cfg: PipelineConfig,
        *,
        include_embedding: bool,
        include_qa: bool,
    ) -> bool:
        if str(self.runtime_state.get("warmup_status", "")) != "ready":
            return False
        if include_embedding and str(self.runtime_state.get("embedding_model_name", "")) != str(cfg.embedding_model_name):
            return False
        if include_qa:
            if not bool(self.runtime_state.get("qa_preloaded", False)):
                return False
            if str(self.runtime_state.get("qa_model_path", "")) != str(cfg.qa_model_path):
                return False
        return True

    def _wait_for_active_runtime_warmup(self) -> None:
        while True:
            with self.runtime_lock:
                thread = self.runtime_warmup_thread
                is_warming = bool(
                    thread is not None
                    and thread.is_alive()
                    and str(self.runtime_state.get("warmup_status", "")) == "warming"
                )
            if not is_warming:
                return
            self.runtime_warmup_done.wait()

    def _note_runtime_released(self, reason: str) -> None:
        with self.runtime_lock:
            self._set_runtime_state_locked(
                warmup_status="deferred",
                finished_at=_utc_now(),
                last_error="",
                reason=reason,
                runner_reused=False,
                qa_preloaded=False,
            )

    def _record_runtime_ready(
        self,
        *,
        cfg: PipelineConfig,
        requested_job_id: Optional[str],
        resolved_job_id: Optional[str],
        source: str,
        reason: str,
        runner_reused: bool,
        embedder_resolved: Optional[str] = None,
    ) -> None:
        with self.runtime_lock:
            self._set_runtime_state_locked(
                warmup_status="ready",
                started_at=self.runtime_state.get("started_at") or _utc_now(),
                finished_at=_utc_now(),
                last_error="",
                requested_job_id=requested_job_id,
                job_id=resolved_job_id,
                source=source,
                reason=reason,
                qa_model_path=cfg.qa_model_path,
                embedding_model_name=cfg.embedding_model_name,
                embedding_model_resolved=embedder_resolved
                if embedder_resolved is not None
                else str(self.runtime_state.get("embedding_model_resolved", "")),
                runner_reused=bool(runner_reused),
                qa_preloaded=True,
            )

    def _execute_runtime_warmup(
        self,
        *,
        cfg: PipelineConfig,
        requested_job_id: Optional[str],
        resolved_job_id: Optional[str],
        source: str,
        reason: str,
        include_model_file_warm: bool,
        include_asr_warm: bool,
        include_embedding: bool,
        include_qa: bool,
        allow_deferred: bool,
        mark_started: bool,
        raise_on_error: bool,
    ) -> Dict[str, Any]:
        if mark_started:
            with self.runtime_lock:
                self._mark_runtime_warming_locked(
                    cfg=cfg,
                    requested_job_id=requested_job_id,
                    resolved_job_id=resolved_job_id,
                    source=source,
                    reason=reason,
                )

        embedder_resolved = ""
        runner_reused = False
        qa_preloaded = False

        try:
            if include_model_file_warm:
                warm_model_paths: List[str] = []
                for model_path in [DEFAULT_QA_MODEL_PATH, DEFAULT_INSTRUCT_MODEL_PATH, DEFAULT_THINKING_MODEL_PATH]:
                    normalized = str(model_path).strip()
                    if normalized and normalized not in warm_model_paths:
                        warm_model_paths.append(normalized)

                for model_path in warm_model_paths:
                    try:
                        cached_path = _prepare_model_artifacts(model_path)
                        print(
                            "[Runtime Warmup] model cache ready: "
                            f"reason={reason}, source={model_path}, cache={cached_path}"
                        )
                    except Exception as e:
                        print(
                            "[Runtime Warmup][WARN] prepare model cache failed: "
                            f"reason={reason}, model={model_path}, err={e}"
                        )

                    try:
                        started = time.perf_counter()
                        print(f"[Runtime Warmup] warm model files begin: reason={reason}, model={model_path}")
                        resolved = _warm_model_file_cache(model_path)
                        elapsed = time.perf_counter() - started
                        print(
                            "[Runtime Warmup] warm model files done: "
                            f"reason={reason}, source={model_path}, cache={resolved}, elapsed={elapsed:.2f}s"
                        )
                    except Exception as e:
                        print(
                            "[Runtime Warmup][WARN] warm model files failed: "
                            f"reason={reason}, model={model_path}, err={e}"
                        )

            if include_asr_warm:
                default_asr_model_size = str(PipelineConfig(video_path="").asr_model_size).strip()
                if default_asr_model_size:
                    try:
                        warmed_asr_path = _warm_asr_model_file_cache(default_asr_model_size)
                        print(
                            "[Runtime Warmup] ASR model files warmed: "
                            f"reason={reason}, source={default_asr_model_size}, resolved={warmed_asr_path}"
                        )
                    except Exception as e:
                        print(
                            "[Runtime Warmup][WARN] warm ASR model files failed: "
                            f"reason={reason}, model={default_asr_model_size}, err={e}"
                        )

            if include_embedding:
                embedder_resolved = _preload_embedding_model(cfg.embedding_model_name)
                print(
                    "[Runtime Warmup] embedding ready: "
                    f"reason={reason}, model={cfg.embedding_model_name}, resolved={embedder_resolved}"
                )

            if include_qa:
                if allow_deferred:
                    acquired = self.model_lock.acquire(blocking=False)
                    if not acquired:
                        print(
                            "[Runtime Warmup] QA preload deferred: "
                            f"reason={reason}, model={cfg.qa_model_path}"
                        )
                        return self._finish_runtime_warmup(
                            status="deferred",
                            embedder_resolved=embedder_resolved,
                            runner_reused=False,
                            qa_preloaded=False,
                        )
                    try:
                        _, runner_reused = self._get_shared_qa_runner(cfg)
                    finally:
                        self.model_lock.release()
                else:
                    with self.model_lock:
                        _, runner_reused = self._get_shared_qa_runner(cfg)
                qa_preloaded = True

            return self._finish_runtime_warmup(
                status="ready",
                embedder_resolved=embedder_resolved,
                runner_reused=runner_reused,
                qa_preloaded=qa_preloaded,
            )
        except Exception as e:
            err = f"{e}\n{traceback.format_exc()}"
            print(f"[Runtime Warmup][WARN] failed: reason={reason}, err={e}")
            result = self._finish_runtime_warmup(
                status="failed",
                last_error=err,
                embedder_resolved=embedder_resolved,
                runner_reused=runner_reused,
                qa_preloaded=False,
            )
            if raise_on_error:
                raise RuntimeError(err) from e
            return result

    def schedule_runtime_warmup(
        self,
        *,
        job_id: Optional[str] = None,
        include_model_file_warm: bool,
        include_asr_warm: bool,
        include_embedding: bool,
        include_qa: bool,
        allow_deferred: bool,
        reason: str,
    ) -> Dict[str, Any]:
        cfg, resolved_job_id, source, requested_job_id = self._resolve_preload_request(job_id)

        with self.runtime_lock:
            thread = self.runtime_warmup_thread
            if thread is not None and thread.is_alive():
                return self._runtime_response_from_snapshot(self._runtime_snapshot_locked())

            if self._runtime_ready_for_cfg_locked(
                cfg,
                include_embedding=include_embedding,
                include_qa=include_qa,
            ):
                return self._runtime_response_from_snapshot(self._runtime_snapshot_locked())

            self._mark_runtime_warming_locked(
                cfg=cfg,
                requested_job_id=requested_job_id,
                resolved_job_id=resolved_job_id,
                source=source,
                reason=reason,
            )
            thread_name = f"runtime-warmup-{reason}-{resolved_job_id or 'default'}"
            worker = Thread(
                target=self._execute_runtime_warmup,
                kwargs={
                    "cfg": cfg,
                    "requested_job_id": requested_job_id,
                    "resolved_job_id": resolved_job_id,
                    "source": source,
                    "reason": reason,
                    "include_model_file_warm": include_model_file_warm,
                    "include_asr_warm": include_asr_warm,
                    "include_embedding": include_embedding,
                    "include_qa": include_qa,
                    "allow_deferred": allow_deferred,
                    "mark_started": False,
                    "raise_on_error": False,
                },
                name=thread_name,
                daemon=True,
            )
            self.runtime_warmup_thread = worker
            snapshot = self._runtime_snapshot_locked()

        worker.start()
        return self._runtime_response_from_snapshot(snapshot)

    def preload_runtime(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        """提前加载 embedding 模型和 QA 模型，减少第一次请求时的等待。"""

        self._wait_for_active_runtime_warmup()
        cfg, resolved_job_id, source, requested_job_id = self._resolve_preload_request(job_id)

        with self.runtime_lock:
            if self._runtime_ready_for_cfg_locked(cfg, include_embedding=True, include_qa=True):
                return self._runtime_response_from_snapshot(self._runtime_snapshot_locked())

        return self._execute_runtime_warmup(
            cfg=cfg,
            requested_job_id=requested_job_id,
            resolved_job_id=resolved_job_id,
            source=source,
            reason="manual",
            include_model_file_warm=False,
            include_asr_warm=False,
            include_embedding=True,
            include_qa=True,
            allow_deferred=False,
            mark_started=True,
            raise_on_error=True,
        )

    def submit(self, req: CreateJobRequest) -> Dict[str, Any]:
        """提交一个新任务。

        主流程：
        1. 规范化请求参数
        2. 检查视频路径是否存在
        3. 生成 `job_id`
        4. 先把任务状态写入内存和磁盘
        5. 再把真正执行逻辑提交到后台线程池
        """

        # 这一长串调用可以按“从里到外”理解：
        # 1. 先把 Pydantic 对象转成普通字典
        # 2. 修正历史路径
        # 3. 强制收口到单卡模式
        normalized_payload = _force_single_gpu_request(_rewrite_record_request_paths(_model_dump_compat(req)))
        video_path = _resolve_video_path(normalized_payload.get("video_path", req.video_path))
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        normalized_payload["video_path"] = str(video_path)
        normalized_payload["output_root"] = _rewrite_legacy_project_path(
            normalized_payload.get("output_root", req.output_root)
        )

        # `**normalized_payload` 会把字典拆成关键字参数，
        # 相当于 `CreateJobRequest(video_path=..., output_root=..., ...)`。
        req_runtime = CreateJobRequest(**normalized_payload)

        # `uuid.uuid4().hex[:12]` 表示生成一个随机 ID，
        # 再用切片只取前 12 个十六进制字符，作为更短的任务编号。
        job_id = uuid.uuid4().hex[:12]
        now = _utc_now()
        output_dir = str(Path(req_runtime.output_root) / job_id)

        record = {
            "job_id": job_id,
            "status": "queued",
            "message": "queued",
            "created_at": now,
            "updated_at": now,
            "request": normalized_payload,
            "outputs": {},
            "error": "",
            "output_dir": output_dir,
            "stage_name": "queued",
            "stage_index": 0,
            "total_stages": 6,
            "progress_percent": 0.0,
        }
        with self.lock:
            # 先登记任务，再启动后台线程。
            # 这样即使线程刚启动就出错，前端也还能查到这条任务记录。
            self.jobs[job_id] = record
            self._persist_state_locked(job_id)

        # `self.executor.submit(...)` 不会阻塞当前 HTTP 请求；
        # 它只是把任务扔到后台，然后立刻返回一个 `Future` 对象。
        fut = self.executor.submit(self._run_job, job_id, req_runtime, output_dir)
        with self.lock:
            self.futures[job_id] = fut
        return self._public_record(job_id)

    def _build_cfg(self, req: CreateJobRequest, output_dir: str) -> PipelineConfig:
        """把 API 请求对象转换成流水线真正使用的 `PipelineConfig`。"""

        # 这是条件表达式的多行写法：
        # 如果用户显式传了 `chroma_dir`，就用它；
        # 否则回退到当前任务输出目录下的默认位置。
        chroma_dir = (
            _rewrite_legacy_project_path(req.chroma_dir)
            if req.chroma_dir
            else str(Path(output_dir) / "chroma_db")
        )
        return PipelineConfig(
            video_path=str(_resolve_video_path(req.video_path)),
            output_dir=_rewrite_legacy_project_path(output_dir),
            instruct_model_path=_rewrite_legacy_project_path(req.instruct_model_path),
            thinking_model_path=_rewrite_legacy_project_path(req.thinking_model_path),
            qa_model_path=_rewrite_legacy_project_path(req.qa_model_path),
            enable_llm_guided_chunking=req.enable_llm_guided_chunking,
            llm_chunk_min_seconds=req.llm_chunk_min_seconds,
            llm_chunk_max_seconds=req.llm_chunk_max_seconds,
            llm_chunk_overlap_seconds=req.llm_chunk_overlap_seconds,
            chunk_seconds=req.chunk_seconds,
            overlap_seconds=req.overlap_seconds,
            section_minutes=req.section_minutes,
            keyframes_per_chunk=req.keyframes_per_chunk,
            keyframe_candidates_per_chunk=req.keyframe_candidates_per_chunk,
            keyframe_max_width=req.keyframe_max_width,
            asr_model_size=req.asr_model_size,
            asr_language=req.asr_language,
            use_vad=req.use_vad,
            ocr_lang=req.ocr_lang,
            chroma_dir=chroma_dir,
            embedding_model_name=_rewrite_legacy_project_path(req.embedding_model_name),
            retrieval_top_k=req.retrieval_top_k,
            retrieval_neighbor_k=req.retrieval_neighbor_k,
            max_new_tokens_chunk=req.max_new_tokens_chunk,
            max_new_tokens_section=req.max_new_tokens_section,
            max_new_tokens_global=req.max_new_tokens_global,
            max_new_tokens_review=req.max_new_tokens_review,
            temperature=req.temperature,
            top_p=req.top_p,
            mystery_mode=req.mystery_mode,
            qa_max_new_tokens=req.qa_max_new_tokens,
            qa_temperature=req.qa_temperature,
            qa_top_p=req.qa_top_p,
            qa_retrieval_top_k=req.qa_retrieval_top_k,
            qa_retrieval_neighbor_k=req.qa_retrieval_neighbor_k,
            resume=req.resume,
            max_retries=req.max_retries,
            retry_base_delay_sec=req.retry_base_delay_sec,
            retry_max_delay_sec=req.retry_max_delay_sec,
            upload_duration_sec=req.upload_duration_sec,
            upload_size_bytes=req.upload_size_bytes,
            upload_filename=req.upload_filename,
            device_map="none",
            torch_dtype=req.torch_dtype,
            attn_implementation=req.attn_implementation,
            enable_dual_gpu_balance=False,
            gpu0_mem_fraction=req.gpu0_mem_fraction,
            gpu1_mem_fraction=req.gpu1_mem_fraction,
            use_free_gpu_memory=req.use_free_gpu_memory,
            gpu_reserve_memory_mib=req.gpu_reserve_memory_mib,
            cpu_offload_max_memory=req.cpu_offload_max_memory,
        )

    def _cfg_from_record(self, rec: Dict[str, Any]) -> PipelineConfig:
        """根据历史任务记录，重新构建一份 `PipelineConfig`。

        这个方法主要用于：
        - 服务重启后恢复旧任务
        - 已完成任务再次 QA
        """

        # `dict(...)` 这里是在复制一份请求字典，
        # 避免下面修正字段时直接改到原始记录。
        req = _force_single_gpu_request(_rewrite_record_request_paths(dict(rec.get("request", {}))))
        output_dir = _rewrite_legacy_project_path(rec.get("output_dir", ""))
        # `A or B` 是很常见的兜底写法：
        # 如果历史记录里没有 `chroma_dir`，就回退到默认目录。
        chroma_dir = req.get("chroma_dir") or str(Path(output_dir) / "chroma_db")

        instruct_default = (
            # `hasattr(...)` 是“判断对象有没有这个属性”。
            # 这里是在兼容 Pydantic v1 / v2 的字段访问方式。
            CreateJobRequest.model_fields["instruct_model_path"].default
            if hasattr(CreateJobRequest, "model_fields")
            else CreateJobRequest.__fields__["instruct_model_path"].default
        )
        thinking_default = (
            CreateJobRequest.model_fields["thinking_model_path"].default
            if hasattr(CreateJobRequest, "model_fields")
            else CreateJobRequest.__fields__["thinking_model_path"].default
        )
        qa_default = (
            CreateJobRequest.model_fields["qa_model_path"].default
            if hasattr(CreateJobRequest, "model_fields")
            else CreateJobRequest.__fields__["qa_model_path"].default
        )

        return PipelineConfig(
            video_path=str(_resolve_video_path(req.get("video_path", ""))),
            output_dir=output_dir,
            instruct_model_path=_rewrite_legacy_project_path(req.get("instruct_model_path", instruct_default)),
            thinking_model_path=_rewrite_legacy_project_path(req.get("thinking_model_path", thinking_default)),
            qa_model_path=_rewrite_legacy_project_path(
                # 这里连写了三个 `or`，可以理解成“多级回退链”：
                # 先用专门的 QA 模型；没有的话退回 instruct 模型；再没有才用类默认值。
                req.get("qa_model_path")
                or req.get("instruct_model_path")
                or qa_default
            ),
            enable_llm_guided_chunking=_to_bool(req.get("enable_llm_guided_chunking"), True),
            llm_chunk_min_seconds=int(req.get("llm_chunk_min_seconds", 30)),
            llm_chunk_max_seconds=int(req.get("llm_chunk_max_seconds", 120)),
            llm_chunk_overlap_seconds=int(req.get("llm_chunk_overlap_seconds", 1)),
            chunk_seconds=int(req.get("chunk_seconds", 120)),
            overlap_seconds=int(req.get("overlap_seconds", 1)),
            section_minutes=int(req.get("section_minutes", 10)),
            keyframes_per_chunk=int(req.get("keyframes_per_chunk", 2)),
            keyframe_candidates_per_chunk=int(req.get("keyframe_candidates_per_chunk", 6)),
            keyframe_max_width=int(req.get("keyframe_max_width", 1280)),
            asr_model_size=str(req.get("asr_model_size", "large-v3")),
            asr_language=req.get("asr_language", "zh"),
            use_vad=_to_bool(req.get("use_vad"), True),
            ocr_lang=str(req.get("ocr_lang", "chi_sim+eng")),
            chroma_dir=_rewrite_legacy_project_path(chroma_dir),
            embedding_model_name=_rewrite_legacy_project_path(
                req.get("embedding_model_name", DEFAULT_EMBEDDING_MODEL_NAME)
            ),
            retrieval_top_k=int(req.get("retrieval_top_k", 5)),
            retrieval_neighbor_k=int(req.get("retrieval_neighbor_k", 1)),
            max_new_tokens_chunk=int(req.get("max_new_tokens_chunk", 512)),
            max_new_tokens_section=int(req.get("max_new_tokens_section", 900)),
            max_new_tokens_global=int(req.get("max_new_tokens_global", 1200)),
            max_new_tokens_review=int(req.get("max_new_tokens_review", 1400)),
            temperature=float(req.get("temperature", 0.2)),
            top_p=float(req.get("top_p", 0.9)),
            mystery_mode=_to_bool(req.get("mystery_mode"), False),
            qa_max_new_tokens=int(req.get("qa_max_new_tokens", 384)),
            qa_temperature=float(req.get("qa_temperature", 0.1)),
            qa_top_p=float(req.get("qa_top_p", 0.9)),
            qa_retrieval_top_k=int(req.get("qa_retrieval_top_k", 3)),
            qa_retrieval_neighbor_k=int(req.get("qa_retrieval_neighbor_k", 0)),
            resume=_to_bool(req.get("resume"), True),
            max_retries=int(req.get("max_retries", 2)),
            retry_base_delay_sec=float(req.get("retry_base_delay_sec", 2.0)),
            retry_max_delay_sec=float(req.get("retry_max_delay_sec", 20.0)),
            # 这里的 `A if 条件 else B` 是条件表达式。
            # 只有原字段确实不为 `None` 时才做数值转换，避免 `float(None)` 报错。
            upload_duration_sec=float(req["upload_duration_sec"]) if req.get("upload_duration_sec") is not None else None,
            upload_size_bytes=int(req["upload_size_bytes"]) if req.get("upload_size_bytes") is not None else None,
            upload_filename=str(req.get("upload_filename", "")) or None,
            device_map="none",
            torch_dtype=str(req.get("torch_dtype", "float16")),
            attn_implementation=str(req.get("attn_implementation")) if req.get("attn_implementation") is not None else None,
            enable_dual_gpu_balance=False,
            gpu0_mem_fraction=float(req.get("gpu0_mem_fraction", 0.40)),
            gpu1_mem_fraction=float(req.get("gpu1_mem_fraction", 0.55)),
            use_free_gpu_memory=_to_bool(req.get("use_free_gpu_memory"), True),
            gpu_reserve_memory_mib=int(req.get("gpu_reserve_memory_mib", 1536)),
            cpu_offload_max_memory=str(req.get("cpu_offload_max_memory", "64GiB")),
        )

    def _run_job(self, job_id: str, req: CreateJobRequest, output_dir: str) -> None:
        """后台线程真正执行任务的入口。

        这是一个典型的“状态机式”方法：
        - 先把任务标记为 `running`
        - 调用 pipeline
        - 成功则标记 `succeeded`
        - 失败则标记 `failed`
        """

        self._update(
            job_id,
            status="running",
            message="pipeline running",
            progress={
                "stage_name": "starting",
                "stage_index": 0,
                "total_stages": 6,
                "progress_percent": 0.0,
            },
        )
        try:
            cfg = self._build_cfg(req=req, output_dir=output_dir)
            pipeline_cls = _long_video_summary_pipeline_cls()
            pipeline = pipeline_cls(
                cfg,
                # `lambda payload: ...` 是一个匿名函数。
                # pipeline 每汇报一次进度，就会回调这里，把进度同步到任务状态里。
                progress_callback=lambda payload: self._on_pipeline_progress(job_id, payload),
            )

            with self.model_lock:
                # 任务正式跑之前，先把预热用的 QA runtime 卸掉，
                # 避免它占住显存影响主 pipeline。
                released_runtimes = self._unload_all_qa_runtimes()
                if released_runtimes:
                    print(f"[QA Runtime] released {released_runtimes} preloaded runner(s) before pipeline run")
                self._note_runtime_released(reason="pipeline_running")
                outputs = pipeline.run()

            self._update(
                job_id,
                status="succeeded",
                message="done",
                outputs=outputs,
                progress={
                    "stage_name": "done",
                    "stage_index": 6,
                    "total_stages": 6,
                    "progress_percent": 100.0,
                },
            )
            if _keep_qa_ready_after_job_enabled():
                # 任务结束后，异步把 QA 模型再预热回来，方便用户马上追问。
                self._preload_qa_runtime_after_job(cfg=cfg, job_id=job_id)
        except Exception as e:
            # 这里用“大网兜式”的 `try/except` 包住整段后台任务，
            # 目的是无论哪一步失败，都能把任务状态明确标成 `failed`。
            # `traceback.format_exc()` 会拿到完整报错栈，比只保存 `str(e)` 更利于排查问题。
            err = f"{e}\n{traceback.format_exc()}"
            with self.lock:
                # 复制一份当前记录，后面即使别的线程更新了原对象，这里也有稳定快照可用。
                rec = dict(self.jobs.get(job_id, {}))

            # 这里不是简单地把进度清零，而是尽量保留“失败前走到了哪一步”，
            # 方便前端和开发者判断任务卡在什么阶段。
            preserved_message = str(rec.get("message", "")).strip()
            preserved_stage_name = str(rec.get("stage_name", "")).strip() or "starting"
            preserved_stage_index = max(0, int(rec.get("stage_index", 0) or 0))
            preserved_total_stages = max(1, int(rec.get("total_stages", 6) or 6))
            preserved_progress_percent = max(
                0.0,
                min(100.0, float(rec.get("progress_percent", 0.0) or 0.0)),
            )
            self._update(
                job_id,
                status="failed",
                # `preserved_message or "pipeline failed"` 表示：
                # 如果之前已经有更具体的提示，就优先保留；否则再给一个通用失败提示。
                message=preserved_message or "pipeline failed",
                error=err,
                progress={
                    "stage_name": preserved_stage_name,
                    "stage_index": preserved_stage_index,
                    "total_stages": preserved_total_stages,
                    "progress_percent": preserved_progress_percent,
                },
            )

    def _preload_qa_runtime_after_job(self, cfg: PipelineConfig, job_id: str) -> None:
        """任务完成后，异步预热 QA runtime。"""

        def _worker() -> None:
            # 内部再定义一个小函数，是为了把“后台预热”这段逻辑封装起来，
            # 然后交给新线程执行。
            try:
                time.sleep(0.25)
                info = self.schedule_runtime_warmup(
                    job_id=job_id,
                    include_model_file_warm=False,
                    include_asr_warm=False,
                    include_embedding=True,
                    include_qa=True,
                    allow_deferred=True,
                    reason="after_job",
                )
                if str(info.get("status", "")) == "warming":
                    self._wait_for_active_runtime_warmup()
                    snapshot = self.runtime_status()
                    if str(snapshot.get("warmup_status", "")) != "ready":
                        info = self.schedule_runtime_warmup(
                            job_id=job_id,
                            include_model_file_warm=False,
                            include_asr_warm=False,
                            include_embedding=True,
                            include_qa=True,
                            allow_deferred=True,
                            reason="after_job",
                        )
                print(f"[QA Runtime] async warmup scheduled after job: job={job_id}, status={info.get('status')}")
            except Exception as e:
                print(f"[QA Runtime][WARN] async preload after job failed: job={job_id}, err={e}")

        # `daemon=True` 表示这是守护线程；主进程退出时，不会因为它还活着而卡住退出。
        Thread(target=_worker, name=f"qa-preload-{job_id}", daemon=True).start()

    def ask(
        self,
        job_id: str,
        question: str,
        mystery_mode: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """对一个已经成功完成的任务发起问答。"""

        q = str(question or "").strip()
        if not q:
            raise ValueError("question is empty")
        # 目前代码把范围固定成单任务 QA。
        scope = "job"

        with self.lock:
            rec = self.jobs.get(job_id)
            if rec is None:
                raise KeyError(job_id)
            status = str(rec.get("status", ""))
            if status != "succeeded":
                raise RuntimeError(f"job not ready for QA: {status}")
            # 拿到锁时顺手复制一份，避免锁释放后原记录被并发修改。
            rec_copy = dict(rec)

        # 这里重新创建 pipeline，不是为了重跑摘要，
        # 而是为了复用其中的问答与证据组织逻辑。
        cfg = self._cfg_from_record(rec_copy)
        pipeline_cls = _long_video_summary_pipeline_cls()
        pipeline = pipeline_cls(cfg)

        self._wait_for_active_runtime_warmup()

        with self.model_lock:
            qa_runner, runner_reused = self._get_shared_qa_runner(cfg)
            self._record_runtime_ready(
                cfg=cfg,
                requested_job_id=job_id,
                resolved_job_id=job_id,
                source="job",
                reason="qa_request",
                runner_reused=runner_reused,
            )
            if scope == "all":
                # 这个分支是“跨任务检索”的保留能力。
                # 当前默认不会走到，但代码结构先预留好了。
                qa_result = self._ask_across_all_jobs(
                    job_id=job_id,
                    cfg=cfg,
                    pipeline=pipeline,
                    question=q,
                    mystery_mode=mystery_mode,
                    qa_runner=qa_runner,
                )
            else:
                qa_result = pipeline.answer_question(
                    question=q,
                    mystery_mode=mystery_mode,
                    qa_runner=qa_runner,
                )

        qa_result = self._annotate_qa_result(
            job_id=job_id,
            rec=rec_copy,
            cfg=cfg,
            qa_result=qa_result,
        )

        self._update(job_id, message="qa answered")
        return {
            "job_id": job_id,
            "question": q,
            "retrieval_scope": scope,
            "answer_markdown": str(qa_result.get("answer_markdown", "")),
            "answer_segments": qa_result.get("answer_segments", []),
            "answer_clips": qa_result.get("answer_clips", []),
            "generated_at": _utc_now(),
        }

    @staticmethod
    def _job_video_name(rec: Dict[str, Any], cfg: Optional[PipelineConfig] = None) -> str:
        """尽量找出一个适合展示给前端看的“视频名”。"""

        # 这里按优先级依次回退：
        # 上传时的原始文件名 -> 视频路径的文件名 -> job_id。
        request = rec.get("request", {}) if isinstance(rec.get("request"), dict) else {}
        upload_filename = str(request.get("upload_filename", "") or "").strip()
        if upload_filename:
            return upload_filename

        video_path = str(request.get("video_path", "") or "").strip()
        if not video_path and cfg is not None:
            video_path = str(cfg.video_path or "").strip()

        name = Path(video_path).name
        return name or str(rec.get("job_id", "")).strip()

    @staticmethod
    def _read_first_text(candidates: List[Path]) -> str:
        """按顺序读取第一个存在的文本文件。"""

        for path in candidates:
            if path.exists():
                return path.read_text(encoding="utf-8").strip()
        return ""

    def _read_job_summary_texts(self, rec: Dict[str, Any]) -> Tuple[str, str]:
        """读取某个任务的最终摘要和草稿摘要。"""

        output_dir = Path(str(rec.get("output_dir", "")).strip())
        final_summary = self._read_first_text(
            [
                output_dir / "final" / "global_summary_final_reviewed.md",
                output_dir / "checkpoints" / "global_summary_final_reviewed.md",
            ]
        )
        draft_summary = self._read_first_text(
            [
                output_dir / "final" / "global_summary_draft.md",
                output_dir / "checkpoints" / "global_summary_draft.md",
            ]
        )
        return final_summary, draft_summary

    def _build_global_qa_sources(self, anchor_job_id: str) -> List[Dict[str, Any]]:
        """收集“跨任务问答”需要用到的全部候选来源。"""

        with self.lock:
            # 这是一个列表推导式：
            # 遍历所有任务 ID，并把每条记录复制一份放进列表。
            records = [dict(self.jobs[jid]) for jid in self.jobs]

        sources: List[Dict[str, Any]] = []
        for rec in records:
            if str(rec.get("status", "")).strip() != "succeeded":
                continue

            source_job_id = str(rec.get("job_id", "")).strip()
            if not source_job_id:
                continue

            try:
                cfg = self._cfg_from_record(rec)
                vector_store_cls = _vector_store_cls()
                store = vector_store_cls(
                    persist_dir=str(cfg.chroma_path()),
                    embedding_model_name=cfg.embedding_model_name,
                )
            except Exception as e:
                # 这里属于“跳过坏数据继续收集”的兜底逻辑：
                # 某个任务的向量库坏了，不应该拖垮整个全局 QA。
                print(f"[QA][WARN] skip global source {source_job_id}: {e}")
                continue

            if not store.chunk_ids_sorted:
                continue

            sources.append(
                {
                    "source_job_id": source_job_id,
                    "source_video_name": self._job_video_name(rec, cfg),
                    "source_video_path": str(cfg.video_path),
                    "record": rec,
                    "store": store,
                }
            )

        # 排序时把当前任务优先放到最前面，其它任务再按视频名和 job_id 排。
        sources.sort(
            # `key=lambda item: (...)` 表示告诉 `sort()`：
            # “请按这个匿名函数返回的元组来排序”。
            key=lambda item: (
                0 if item["source_job_id"] == anchor_job_id else 1,
                str(item.get("source_video_name", "")).lower(),
                str(item.get("source_job_id", "")),
            )
        )
        return sources

    def _build_global_summary_context(
        self,
        pipeline: LongVideoSummaryPipeline,
        sources_by_job_id: Dict[str, Dict[str, Any]],
        evidence_rows: List[Dict[str, Any]],
        fallback_job_id: str,
        max_jobs: int = 4,
    ) -> str:
        """为跨视频 QA 组装一个摘要上下文。"""

        ordered_job_ids: List[str] = []
        seen_job_ids = set()

        for row in evidence_rows:
            source_job_id = str(row.get("source_job_id", "")).strip()
            if source_job_id and source_job_id not in seen_job_ids:
                # 这里一边去重，一边保留“第一次出现的顺序”，
                # 这样最终上下文更贴近检索结果的原始相关性顺序。
                seen_job_ids.add(source_job_id)
                ordered_job_ids.append(source_job_id)

        if fallback_job_id and fallback_job_id in sources_by_job_id and fallback_job_id not in seen_job_ids:
            ordered_job_ids.append(fallback_job_id)

        blocks: List[str] = []
        # `ordered_job_ids[: max(1, int(max_jobs))]` 是切片写法，
        # 表示“最多只取前 max_jobs 个任务”，避免上下文无限变长。
        for source_job_id in ordered_job_ids[: max(1, int(max_jobs))]:
            source = sources_by_job_id.get(source_job_id)
            if not source:
                continue

            final_summary, draft_summary = self._read_job_summary_texts(source["record"])
            summary_block = pipeline._build_qa_summary_context(final_summary, draft_summary)
            if not summary_block or summary_block == "(暂无摘要上下文)":
                continue

            video_name = str(source.get("source_video_name", "")).strip() or source_job_id
            blocks.append(f"[video: {video_name} | job: {source_job_id}]\n{summary_block}")

        if not blocks:
            return "(暂无跨视频摘要上下文)"
        # `"\n\n".join(blocks)` 会用两个换行把多个文本块拼起来。
        return "\n\n".join(blocks)

    def _ask_across_all_jobs(
        self,
        job_id: str,
        cfg: PipelineConfig,
        pipeline: LongVideoSummaryPipeline,
        question: str,
        mystery_mode: Optional[bool],
        qa_runner: Any,
    ) -> Dict[str, Any]:
        """跨多个已完成任务进行检索和问答。

        当前 `ask()` 里默认没有走到这里，但保留了完整能力。
        """

        sources = self._build_global_qa_sources(anchor_job_id=job_id)
        if not sources:
            raise RuntimeError("no succeeded jobs available for global QA")

        evidence_pack, evidence_rows = _build_multi_store_question_evidence_pack(
            sources=sources,
            question=question,
            top_k=cfg.qa_retrieval_top_k,
            neighbor_k=cfg.qa_retrieval_neighbor_k,
        )
        # 这是一个字典推导式：
        # 把 `sources` 转成“job_id -> source 信息”的映射，后面按 job_id 查会更方便。
        sources_by_job_id = {
            str(source.get("source_job_id", "")).strip(): source
            for source in sources
            if str(source.get("source_job_id", "")).strip()
        }
        summary_context = self._build_global_summary_context(
            pipeline=pipeline,
            sources_by_job_id=sources_by_job_id,
            evidence_rows=evidence_rows,
            fallback_job_id=job_id,
        )
        return pipeline.answer_question(
            question=question,
            mystery_mode=mystery_mode,
            qa_runner=qa_runner,
            summary_context_override=summary_context,
            evidence_pack_override=evidence_pack,
            evidence_rows_override=evidence_rows,
        )

    def _annotate_qa_result(
        self,
        job_id: str,
        rec: Dict[str, Any],
        cfg: PipelineConfig,
        qa_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """给 QA 结果补齐来源信息，方便前端显示。"""

        source_video_name = self._job_video_name(rec, cfg)

        answer_segments: List[Dict[str, Any]] = []
        # `qa_result.get("answer_segments", []) or []` 的用意是双重兜底：
        # 即使字段缺失，或者字段值意外是 `None`，这里也能安全遍历。
        for seg in qa_result.get("answer_segments", []) or []:
            row = dict(seg)
            row["source_job_id"] = str(row.get("source_job_id", "")).strip() or job_id
            row["source_video_name"] = str(row.get("source_video_name", "")).strip() or source_video_name
            # `pop("key", None)` 表示“如果这个键存在就删掉；不存在也别报错”。
            row.pop("source_video_path", None)
            answer_segments.append(row)

        answer_clips: List[Dict[str, Any]] = []
        for clip in qa_result.get("answer_clips", []) or []:
            row = dict(clip)
            row["clip_job_id"] = str(row.get("clip_job_id", "")).strip() or job_id
            row["source_job_id"] = str(row.get("source_job_id", "")).strip() or job_id
            row["source_video_name"] = str(row.get("source_video_name", "")).strip() or source_video_name
            answer_clips.append(row)

        normalized = dict(qa_result)
        normalized["answer_segments"] = answer_segments
        normalized["answer_clips"] = answer_clips
        return normalized

    def delete(self, job_id: str) -> Dict[str, Any]:
        """永久删除任务记录及其输出文件。

        这里故意限制只能删除失败 / 取消 / 中断的任务，
        防止误删一条已经成功完成的结果。
        """

        with self.lock:
            rec = self.jobs.get(job_id)
            if rec is None:
                raise KeyError(job_id)

            status = str(rec.get("status", "")).strip().lower()
            if status not in {"failed", "canceled", "interrupted"}:
                raise RuntimeError("only failed, canceled, or interrupted jobs can be permanently deleted")

            fut = self.futures.get(job_id)
            if fut is not None and not fut.done():
                raise RuntimeError("job is still running and cannot be deleted")

            # 这里拿一份“包含内部字段”的副本，后面即使从 `self.jobs` 里删掉，
            # 也还能继续用它来决定要删哪些目录。
            rec_copy = self._serialize_record(rec, public=False)

        request = rec_copy.get("request", {}) if isinstance(rec_copy.get("request"), dict) else {}
        output_dir_raw = str(rec_copy.get("output_dir", "")).strip()
        # 这里的条件表达式表示：
        # 有路径字符串才转成 `Path`，否则保持 `None`。
        output_dir = Path(output_dir_raw) if output_dir_raw else None
        video_path = _resolve_existing_or_candidate_path(request.get("video_path", ""))
        upload_dir: Optional[Path] = None

        if video_path and str(request.get("upload_filename", "")).strip() and video_path.name:
            candidate_upload_dir = video_path.parent
            # 只允许删除系统托管的上传目录，避免误删用户自己磁盘上的任意目录。
            if candidate_upload_dir != video_path and _is_path_within(candidate_upload_dir, UPLOAD_VIDEO_ROOT):
                upload_dir = candidate_upload_dir

        deleted_output_dir = ""
        deleted_upload_dir = ""

        deleted_output_dir = _remove_tree_if_exists(output_dir)

        if upload_dir and upload_dir != output_dir:
            deleted_upload_dir = _remove_tree_if_exists(upload_dir)

        with self.lock:
            self.jobs.pop(job_id, None)
            self.futures.pop(job_id, None)
            self._persist_registry_locked()

        retained_video_path = ""
        if video_path and not deleted_upload_dir:
            retained_video_path = str(video_path)

        return {
            "job_id": job_id,
            "deleted": True,
            "status_before_delete": status,
            "deleted_output_dir": deleted_output_dir,
            "deleted_upload_dir": deleted_upload_dir,
            "retained_video_path": retained_video_path,
            "deleted_at": _utc_now(),
        }

    def cancel(self, job_id: str) -> Dict[str, Any]:
        """尝试取消尚未开始执行的任务。"""

        with self.lock:
            fut = self.futures.get(job_id)
            if fut is None:
                raise KeyError(job_id)
            # `Future.cancel()` 只有在任务“尚未真正开始”时才会成功。
            canceled = fut.cancel()

        if canceled:
            self._update(
                job_id,
                status="canceled",
                message="canceled before start",
                progress={
                    "stage_name": "canceled",
                    "stage_index": 0,
                    "total_stages": 6,
                    "progress_percent": 0.0,
                },
            )
        else:
            self._update(job_id, message="unable to cancel (already running or done)")
        return self._public_record(job_id)

    def get(self, job_id: str) -> Dict[str, Any]:
        """读取单个任务状态。"""

        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
        return self._public_record(job_id)

    def list(self) -> Dict[str, Dict[str, Any]]:
        """列出全部任务。"""

        with self.lock:
            job_ids = list(self.jobs.keys())
        # 这里用字典推导式把每个 `job_id` 映射成公开版任务记录。
        return {job_id: self._public_record(job_id) for job_id in job_ids}

    def _update(
        self,
        job_id: str,
        status: Optional[str] = None,
        message: Optional[str] = None,
        outputs: Optional[Dict[str, str]] = None,
        error: Optional[str] = None,
        progress: Optional[Dict[str, Any]] = None,
    ) -> None:
        """统一更新任务记录，并立即写盘。"""

        with self.lock:
            rec = self.jobs[job_id]
            # 这里一组 `if xxx is not None` 的写法很常见：
            # 只有调用方真的传了这个参数，才覆盖原记录，没传就保持原值不动。
            if status is not None:
                rec["status"] = status
            if message is not None:
                rec["message"] = message
            if outputs is not None:
                rec["outputs"] = outputs
            if error is not None:
                rec["error"] = error
            if progress is not None:
                # `dict.get("key", 默认值)` 的意思是：
                # 进度字典里有新值就用新值，没有就沿用旧值。
                rec["stage_name"] = str(progress.get("stage_name", rec.get("stage_name", "")))
                rec["stage_index"] = int(progress.get("stage_index", rec.get("stage_index", 0)))
                rec["total_stages"] = int(progress.get("total_stages", rec.get("total_stages", 6)))
                rec["progress_percent"] = float(progress.get("progress_percent", rec.get("progress_percent", 0.0)))
            rec["updated_at"] = _utc_now()
            self._persist_state_locked(job_id)

    def _public_record(self, job_id: str) -> Dict[str, Any]:
        """返回适合对外暴露的任务记录。"""

        with self.lock:
            rec = self.jobs[job_id]
            return self._serialize_record(rec, public=True)

    def _on_pipeline_progress(self, job_id: str, payload: Dict[str, Any]) -> None:
        """接收 pipeline 的进度回调，再转存到任务状态里。"""

        message = str(payload.get("message", "pipeline running"))
        self._update(job_id, message=message, progress=payload)

    def _serialize_record(self, rec: Dict[str, Any], public: bool) -> Dict[str, Any]:
        """序列化任务记录；`public=True` 时去掉不该暴露的内部字段。"""

        # `dict(rec)` 是浅拷贝：会生成一个新的外层字典，
        # 这样后面删字段时不会直接改到原对象。
        obj = dict(rec)
        if public:
            obj.pop("output_dir", None)
        return obj

    def _persist_state_locked(self, job_id: str) -> None:
        """把单个任务状态写入它自己的目录，并刷新全局注册表。"""

        rec = self.jobs[job_id]
        output_dir = Path(str(rec.get("output_dir", "")))
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            record_path = output_dir / JOB_RECORD_FILENAME
            record_path.write_text(
                json.dumps(self._serialize_record(rec, public=False), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        self._persist_registry_locked()

    def _persist_registry_locked(self) -> None:
        """把所有任务统一写入全局注册表文件。"""

        registry = {
            "updated_at": _utc_now(),
            "jobs": {
                # `sorted(..., key=lambda item: item[0])` 这里是按字典键（也就是 job_id）排序，
                # 主要是为了让写出来的 JSON 更稳定、可读。
                jid: self._serialize_record(job, public=False)
                for jid, job in sorted(self.jobs.items(), key=lambda item: item[0])
            },
        }
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_persisted_jobs(self) -> None:
        """服务启动时，从磁盘恢复历史任务。"""

        loaded: Dict[str, Dict[str, Any]] = {}
        candidate_dirs: List[Path] = []

        if self.registry_path.exists():
            try:
                obj = json.loads(self.registry_path.read_text(encoding="utf-8"))
            except Exception:
                # 注册表坏了时，不让整个服务启动失败，继续尝试从任务目录恢复。
                obj = {}
            jobs = obj.get("jobs", {}) if isinstance(obj, dict) else {}
            if isinstance(jobs, dict):
                for rec in jobs.values():
                    if isinstance(rec, dict):
                        output_dir = str(rec.get("output_dir", "")).strip()
                        if output_dir:
                            candidate_dirs.append(Path(_rewrite_legacy_project_path(output_dir)))

        if self.default_output_root.exists():
            # 这里把输出根目录下的每个子目录都当成“潜在任务目录”加入候选集。
            candidate_dirs.extend([p for p in self.default_output_root.iterdir() if p.is_dir()])

        seen = set()
        for output_dir in candidate_dirs:
            try:
                resolved = output_dir.resolve()
            except Exception:
                resolved = output_dir
            key = str(resolved)
            if key in seen:
                continue
            # `seen` 集合是一个很典型的去重模式，
            # 用来避免同一个目录同时从注册表和磁盘扫描里被重复加载。
            seen.add(key)

            rec = self._load_job_record_from_dir(output_dir)
            if rec is None:
                continue
            loaded[str(rec["job_id"])] = rec

        with self.lock:
            self.jobs = loaded

        with self.lock:
            # 重新写一遍，是为了把旧格式数据规范化成当前格式。
            for job_id in sorted(self.jobs.keys()):
                self._persist_state_locked(job_id)

    def _load_job_record_from_dir(self, output_dir: Path) -> Optional[Dict[str, Any]]:
        """优先读现成的 `job_record.json`，读不到再尝试从目录结构反推。"""

        record_path = output_dir / JOB_RECORD_FILENAME
        if record_path.exists():
            try:
                rec = json.loads(record_path.read_text(encoding="utf-8"))
            except Exception:
                rec = None
            if isinstance(rec, dict):
                rec["output_dir"] = str(output_dir)
                return self._normalize_loaded_record(rec, output_dir)

        return self._reconstruct_job_record(output_dir)

    def _normalize_loaded_record(self, rec: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """把从磁盘恢复出来的记录整理成统一格式。"""

        status = str(rec.get("status", "unknown"))
        if status in {"running", "queued"}:
            # 如果服务重启前任务处于运行中，那么恢复后它不可能继续跑，
            # 所以这里明确标成 `interrupted`。
            status = "interrupted"
            rec["message"] = "service restarted before job finished"

        normalized_output_dir = Path(_rewrite_legacy_project_path(str(output_dir)))
        normalized_request = _force_single_gpu_request(_rewrite_record_request_paths(
            dict(rec.get("request", {})) if isinstance(rec.get("request"), dict) else {}
        ))
        normalized_outputs = _rewrite_output_paths(
            dict(rec.get("outputs", {})) if isinstance(rec.get("outputs"), dict) else {}
        )

        normalized = {
            # `A or B` 在这里再次用作兜底：
            # 记录里没有 job_id 时，就退回目录名。
            "job_id": str(rec.get("job_id") or normalized_output_dir.name),
            "status": status,
            "message": str(rec.get("message", "")),
            "created_at": str(rec.get("created_at") or _utc_now()),
            "updated_at": str(rec.get("updated_at") or rec.get("created_at") or _utc_now()),
            "request": normalized_request,
            "outputs": normalized_outputs,
            "error": str(rec.get("error", "")),
            "output_dir": str(normalized_output_dir),
            # 这里嵌了条件表达式：
            # 成功任务默认视为已经走到最后一阶段；否则沿用状态名或从 0 开始。
            "stage_name": str(rec.get("stage_name", "done" if status == "succeeded" else status)),
            "stage_index": int(rec.get("stage_index", 6 if status == "succeeded" else 0)),
            "total_stages": int(rec.get("total_stages", 6)),
            "progress_percent": float(rec.get("progress_percent", 100.0 if status == "succeeded" else 0.0)),
        }
        return normalized

    def _reconstruct_job_record(self, output_dir: Path) -> Optional[Dict[str, Any]]:
        """当缺少显式任务记录时，尝试从产物目录反推出一条任务记录。"""

        checkpoints_dir = output_dir / "checkpoints"
        final_dir = output_dir / "final"
        run_meta_path = checkpoints_dir / "run_meta.json"
        if not checkpoints_dir.exists() and not final_dir.exists():
            return None

        request: Dict[str, Any] = {}
        # 如果找不到正式记录文件，就退而求其次：
        # 先用目录的修改时间猜一个创建时间。
        created_at = datetime.utcfromtimestamp(output_dir.stat().st_mtime).isoformat(timespec="seconds") + "Z"
        if run_meta_path.exists():
            try:
                meta_obj = json.loads(run_meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta_obj = {}
            payload = meta_obj.get("payload", {}) if isinstance(meta_obj, dict) else {}
            if isinstance(payload, dict):
                # 这里是在做“尽力恢复”：
                # 只从 `run_meta.json` 里提取最关键的那部分配置。
                video_path = str(payload.get("video_path", ""))
                request = {
                    "video_path": video_path,
                    "output_root": str(output_dir.parent),
                    "resume": True,
                    "enable_llm_guided_chunking": bool(payload.get("enable_llm_guided_chunking", True)),
                    "chunk_seconds": int(payload.get("chunk_seconds", 120)),
                    "overlap_seconds": int(payload.get("overlap_seconds", 1)),
                    "llm_chunk_min_seconds": int(payload.get("llm_chunk_min_seconds", 30)),
                    "llm_chunk_max_seconds": int(payload.get("llm_chunk_max_seconds", 120)),
                    "llm_chunk_overlap_seconds": int(payload.get("llm_chunk_overlap_seconds", 1)),
                    "section_minutes": int(payload.get("section_minutes", 10)),
                    "keyframes_per_chunk": int(payload.get("keyframes_per_chunk", 2)),
                    "keyframe_candidates_per_chunk": int(payload.get("keyframe_candidates_per_chunk", 6)),
                    "asr_model_size": str(payload.get("asr_model_size", "large-v3")),
                    "asr_language": payload.get("asr_language", "zh"),
                    "use_vad": bool(payload.get("use_vad", True)),
                    "ocr_lang": str(payload.get("ocr_lang", "chi_sim+eng")),
                    "instruct_model_path": str(payload.get("instruct_model_path", "")),
                    "thinking_model_path": str(payload.get("thinking_model_path", "")),
                }
                try:
                    created_at = datetime.utcfromtimestamp(run_meta_path.stat().st_mtime).isoformat(timespec="seconds") + "Z"
                except Exception:
                    pass

        request = _force_single_gpu_request(_rewrite_record_request_paths(request))

        outputs = self._collect_outputs(output_dir)
        has_final = bool(outputs.get("global_final"))
        has_partial = bool(outputs.get("global_draft") or outputs.get("chunk_cards") or outputs.get("manifest"))
        # 这里是在“看目录里留下了哪些文件”来猜测任务状态。
        # 这是一个嵌套条件表达式：
        # 有最终摘要 -> succeeded
        # 否则如果有部分中间产物 -> interrupted
        # 再否则 -> unknown
        status = "succeeded" if has_final else ("interrupted" if has_partial else "unknown")
        message = "restored from disk" if status == "succeeded" else "restored from disk; previous run did not finish"
        stage_name = "done" if status == "succeeded" else "interrupted"
        stage_index = 6 if status == "succeeded" else 0
        progress_percent = 100.0 if status == "succeeded" else 0.0

        rec = {
            "job_id": output_dir.name,
            "status": status,
            "message": message,
            "created_at": created_at,
            "updated_at": created_at,
            "request": request,
            "outputs": outputs,
            "error": "",
            "output_dir": str(output_dir),
            "stage_name": stage_name,
            "stage_index": stage_index,
            "total_stages": 6,
            "progress_percent": progress_percent,
        }
        return self._normalize_loaded_record(rec, output_dir)

    def _collect_outputs(self, output_dir: Path) -> Dict[str, str]:
        """扫描任务输出目录，收集所有可对外暴露的结果文件路径。"""

        final_dir = output_dir / "final"
        checkpoints_dir = output_dir / "checkpoints"
        timings_dir = output_dir / "timings"

        final_path = final_dir / "global_summary_final_reviewed.md"
        draft_path = final_dir / "global_summary_draft.md"
        # 这里属于“从最终目录回退到 checkpoints 目录”的兜底逻辑。
        # 任务中断或还没完全整理完时，结果可能只留在 checkpoints 里。
        if not final_path.exists():
            cp = checkpoints_dir / "global_summary_final_reviewed.md"
            if cp.exists():
                final_path = cp
        if not draft_path.exists():
            cp = checkpoints_dir / "global_summary_draft.md"
            if cp.exists():
                draft_path = cp

        outputs = {
            # 这一整段是“文件存在就返回路径，不存在就返回空字符串”的归一化写法。
            # 好处是前端拿到的数据结构始终稳定，不需要再判断键是否缺失。
            "manifest": str(final_dir / "chunk_manifest.jsonl") if (final_dir / "chunk_manifest.jsonl").exists() else "",
            "chunk_cards": str(final_dir / "chunk_cards.jsonl") if (final_dir / "chunk_cards.jsonl").exists() else "",
            "section_summaries": str(final_dir / "section_summaries.md") if (final_dir / "section_summaries.md").exists() else "",
            "global_draft": str(draft_path) if draft_path.exists() else "",
            "global_final": str(final_path) if final_path.exists() else "",
            "chunk_plan": str(final_dir / "chunk_plan.json") if (final_dir / "chunk_plan.json").exists() else "",
            "checkpoints_dir": str(checkpoints_dir) if checkpoints_dir.exists() else "",
            "timing_events": str(timings_dir / "timing_events.jsonl") if (timings_dir / "timing_events.jsonl").exists() else "",
            "timing_summary": str(timings_dir / "timing_summary.json") if (timings_dir / "timing_summary.json").exists() else "",
        }
        return outputs

    def get_summary_preview(self, job_id: str) -> Dict[str, Any]:
        """读取摘要预览内容，给前端直接展示。"""

        with self.lock:
            rec = self.jobs.get(job_id)
            if rec is None:
                raise KeyError(job_id)
            output_dir = Path(str(rec.get("output_dir", "")))

        final_final = output_dir / "final" / "global_summary_final_reviewed.md"
        cp_final = output_dir / "checkpoints" / "global_summary_final_reviewed.md"
        final_draft = output_dir / "final" / "global_summary_draft.md"
        cp_draft = output_dir / "checkpoints" / "global_summary_draft.md"

        # 这里也是多级回退：
        # 优先读 `final/`，没有的话再回退读 `checkpoints/`。
        final_path = final_final if final_final.exists() else (cp_final if cp_final.exists() else None)
        draft_path = final_draft if final_draft.exists() else (cp_draft if cp_draft.exists() else None)

        final_markdown = final_path.read_text(encoding="utf-8") if final_path else ""
        draft_markdown = draft_path.read_text(encoding="utf-8") if draft_path else ""

        return {
            "job_id": job_id,
            "status": rec.get("status", ""),
            "stage_name": rec.get("stage_name", ""),
            "stage_index": rec.get("stage_index", 0),
            "total_stages": rec.get("total_stages", 6),
            "progress_percent": rec.get("progress_percent", 0.0),
            "final_markdown": final_markdown,
            "draft_markdown": draft_markdown,
            "final_path": str(final_path) if final_path else "",
            "draft_path": str(draft_path) if draft_path else "",
        }

    def get_qa_clip_path(self, job_id: str, clip_id: str) -> Path:
        """根据任务 ID 和 clip ID 找到对应的视频片段文件。"""

        with self.lock:
            rec = self.jobs.get(job_id)
            if rec is None:
                raise KeyError(job_id)
            output_dir = Path(str(rec.get("output_dir", "")))

        # `Path(...).name` 的作用是只保留最后一段文件名，顺手规避路径穿越问题。
        safe_clip_id = Path(str(clip_id)).name
        clip_path = output_dir / "qa_clips" / f"{safe_clip_id}.mp4"
        if not clip_path.exists() or not clip_path.is_file():
            raise FileNotFoundError(str(clip_path))
        return clip_path

    def get_job_video_path(self, job_id: str) -> Path:
        """返回任务对应的原始视频路径。"""

        with self.lock:
            rec = self.jobs.get(job_id)
            if rec is None:
                raise KeyError(job_id)
            request = rec.get("request", {}) if isinstance(rec.get("request"), dict) else {}

        video_path = _resolve_video_path(request.get("video_path", ""))
        if not video_path.exists() or not video_path.is_file():
            raise FileNotFoundError(str(video_path))
        return video_path


# `app` 是整个 FastAPI 应用对象。
# 所有 `@app.get(...)`、`@app.post(...)` 都是在往这个对象上注册路由。
app = FastAPI(title="Long Video Summary API", version="2.0.0")
# 全局只创建一个 `JobManager`，整个服务进程共享它。
manager = JobManager(max_workers=1)

WEB_DIR = Path(__file__).resolve().parent / "web"
# 上传的视频会先落到这个根目录下。
UPLOAD_VIDEO_ROOT = DEFAULT_DATABASE_DIR
# 把 `/ui/...` 这个 URL 映射到前端静态文件目录。
# `mount(...)` 可以理解成“把一整段子路径交给静态文件服务处理”。
app.mount("/ui", StaticFiles(directory=str(WEB_DIR)), name="ui")


def _startup_warm_models_enabled() -> bool:
    """读取是否启用“启动即预热模型”的配置开关。"""

    return runtime_bool("startup_warm_models", True)


def _startup_preload_qa_enabled() -> bool:
    """读取是否在启动时预加载 QA runtime。"""

    return runtime_bool("startup_preload_qa", True)


def _startup_warmup_mode() -> str:
    """读取启动预热模式：`background` 或 `blocking`。"""

    raw = runtime_str("startup_warmup_mode", "background").strip().lower()
    if raw in {"blocking", "sync", "synchronous"}:
        return "blocking"
    return "background"


def _keep_qa_ready_after_job_enabled() -> bool:
    """读取是否在任务完成后保持 QA runtime 处于预热状态。"""

    return runtime_bool("keep_qa_ready_after_job", True)


@app.on_event("startup")
def warm_startup_runtime() -> None:
    """服务启动时执行一次的预热逻辑。

    目标很简单：把“第一次请求特别慢”的工作提前做掉。
    """

    warm_models_enabled = _startup_warm_models_enabled()
    preload_qa_enabled = _startup_preload_qa_enabled()
    if not warm_models_enabled and not preload_qa_enabled:
        return

    print(f"[Startup] runtime profile: {RUNTIME_PROFILE_PATH}")
    mode = _startup_warmup_mode()

    if mode == "blocking":
        try:
            info = manager._execute_runtime_warmup(
                cfg=manager._default_preload_cfg(),
                requested_job_id=None,
                resolved_job_id=None,
                source="default",
                reason="startup",
                include_model_file_warm=warm_models_enabled,
                include_asr_warm=warm_models_enabled,
                include_embedding=warm_models_enabled or preload_qa_enabled,
                include_qa=preload_qa_enabled,
                allow_deferred=False,
                mark_started=True,
                raise_on_error=False,
            )
            print(f"[Startup] blocking warmup finished: status={info.get('status')}")
        except Exception as e:
            print(f"[Startup][WARN] blocking warmup failed: {e}")
        return

    info = manager.schedule_runtime_warmup(
        include_model_file_warm=warm_models_enabled,
        include_asr_warm=warm_models_enabled,
        include_embedding=warm_models_enabled or preload_qa_enabled,
        include_qa=preload_qa_enabled,
        allow_deferred=True,
        reason="startup",
    )
    print(
        "[Startup] async warmup scheduled: "
        f"status={info.get('status')}, warm_models={warm_models_enabled}, preload_qa={preload_qa_enabled}"
    )


@app.middleware("http")
async def disable_frontend_cache(request, call_next):
    """给前端静态资源加“不要缓存”的响应头。"""

    # `await` 用在异步函数里，表示“等这个异步操作执行完成再继续”。
    response = await call_next(request)
    path = request.url.path
    # 这里连用了多个 `or` 条件：
    # 只要命中首页、静态资源或 favicon 中的任意一种，就禁用缓存。
    if path == "/" or path.startswith("/ui/") or path == "/favicon.ico":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/")
def index() -> FileResponse:
    """返回前端首页。"""

    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/favicon.ico")
def favicon() -> Response:
    """浏览器请求网站图标时，直接返回空响应。"""

    return Response(status_code=204)


@app.get("/health")
def health() -> Dict[str, Any]:
    """最简单的健康检查接口。"""

    return {
        "status": "ok",
        "time": _utc_now(),
        "runtime": manager.runtime_status(),
    }


@app.post("/jobs", response_model=JobStatusResponse)
def create_job(req: CreateJobRequest) -> Dict[str, Any]:
    """用“本地已有视频路径”创建任务。"""

    try:
        return manager.submit(req)
    except FileNotFoundError as e:
        # 把 Python 的文件不存在异常翻译成 HTTP 400，前端更容易处理。
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/jobs/upload", response_model=JobStatusResponse)
async def create_job_with_upload(
    file: UploadFile = File(...),
    output_root: str = Form("outputs/api_jobs"),
    mystery_mode: bool = Form(False),
    resume: bool = Form(True),
) -> Dict[str, Any]:
    """先上传视频，再立即创建任务。

    这是一个异步函数，因为上传文件本身是 I/O 密集型操作。
    """

    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")

    upload_id = uuid.uuid4().hex[:10]
    upload_dir = UPLOAD_VIDEO_ROOT / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    local_video = upload_dir / file.filename

    upload_started = time.perf_counter()
    total_bytes = 0
    try:
        with local_video.open("wb") as f:
            # `with ... as f` 的含义是：
            # 打开文件后把句柄取名为 `f`，离开代码块时自动关闭文件。
            while True:
                # 每次读 8MB，避免一次性把整个大视频都塞进内存。
                chunk = await file.read(8 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                total_bytes += len(chunk)
    finally:
        # `finally` 里的代码无论前面成功还是失败都会执行，
        # 适合做“收尾清理”，比如关闭上传文件句柄。
        await file.close()
    upload_duration_sec = time.perf_counter() - upload_started

    req = CreateJobRequest(
        video_path=str(local_video),
        output_root=output_root,
        mystery_mode=mystery_mode,
        resume=resume,
        upload_duration_sec=upload_duration_sec,
        upload_size_bytes=total_bytes,
        upload_filename=file.filename,
    )
    return manager.submit(req)


@app.get("/jobs")
def list_jobs() -> Dict[str, Dict[str, Any]]:
    """列出全部任务。"""

    return manager.list()


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> Dict[str, Any]:
    """查询单个任务状态。"""

    try:
        return manager.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")


@app.post("/jobs/{job_id}/cancel", response_model=JobStatusResponse)
def cancel_job(job_id: str) -> Dict[str, Any]:
    """尝试取消任务。"""

    try:
        return manager.cancel(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")


@app.delete("/jobs/{job_id}", response_model=DeleteJobResponse)
def delete_job(job_id: str) -> Dict[str, Any]:
    """删除失败 / 中断 / 已取消的任务。"""

    try:
        return manager.delete(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")
    except RuntimeError as e:
        # 这里用 409（冲突）表示“请求格式没错，但当前任务状态不允许这样做”。
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"delete failed: {e}")


@app.get("/jobs/{job_id}/result")
def get_job_result(job_id: str) -> Dict[str, Any]:
    """获取任务最终产物路径。"""

    try:
        rec = manager.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")

    if rec.get("status") != "succeeded":
        raise HTTPException(status_code=409, detail=f"job not succeeded: {rec.get('status')}")

    return {
        "job_id": rec["job_id"],
        "outputs": rec.get("outputs", {}),
    }


@app.get("/jobs/{job_id}/summary")
def get_job_summary(job_id: str) -> Dict[str, Any]:
    """读取摘要内容本身，而不仅仅是结果文件路径。"""

    try:
        data = manager.get_summary_preview(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")

    if not data.get("final_markdown") and not data.get("draft_markdown"):
        raise HTTPException(status_code=404, detail="summary markdown not available yet")

    return data


@app.post("/runtime/preload")
def preload_runtime(req: RuntimePreloadRequest) -> Dict[str, Any]:
    """手动触发运行时预热。"""

    try:
        return manager.preload_runtime(job_id=req.job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"runtime preload failed: {e}")


@app.post("/jobs/{job_id}/qa", response_model=QAResponse)
def ask_job_question(job_id: str, req: QARequest) -> Dict[str, Any]:
    """针对某个已完成任务提问。"""

    try:
        return manager.ask(
            job_id=job_id,
            question=req.question,
            mystery_mode=req.mystery_mode,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"qa failed: {e}")


@app.get("/jobs/{job_id}/qa/clips/{clip_id}")
def get_job_qa_clip(job_id: str, clip_id: str) -> FileResponse:
    """下载或播放 QA 生成的视频片段。"""

    try:
        clip_path = manager.get_qa_clip_path(job_id=job_id, clip_id=clip_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="qa clip not found")
    return FileResponse(str(clip_path), media_type="video/mp4", filename=clip_path.name)


@app.get("/jobs/{job_id}/video")
def get_job_video(job_id: str) -> FileResponse:
    """获取任务关联的原始视频文件。"""

    try:
        video_path = manager.get_job_video_path(job_id=job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="job video not found")
    return FileResponse(str(video_path), filename=video_path.name)
