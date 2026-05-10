from __future__ import annotations

"""Qwen3-VL 运行时封装。

这个文件主要负责 5 件事：
1. 准备模型目录：把模型文件同步到更快的本地缓存目录，并提前预热文件缓存。
2. 选择加载设备：决定模型放到 CPU、单张 GPU，还是交给 transformers 自动分配。
3. 加载和卸载模型：封装 `from_pretrained(...)`、`processor` 初始化，以及显存清理。
4. 执行多模态生成：把图片和文本整理成模型能理解的输入，然后调用模型生成回答。
5. 做异常兜底：例如显存不足（OOM）时，自动降低参数后重试。

注释尽量面向新手，重点解释：
- Python 语法在这里扮演什么角色
- 每个函数的输入/输出是什么
- 代码为什么这样组织
"""

from contextlib import contextmanager
import importlib.util
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

# 这里要在 `import torch` 之前设置环境变量。
# `expandable_segments:True` 可以让 PyTorch 的显存分配更灵活，
# 某些场景下能减少显存碎片带来的 OOM。
if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gc
from threading import Lock

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .settings import runtime_bool, runtime_int, runtime_str

# `fcntl` 常用于 Linux/Unix 下的文件锁。
# 这里用 `try/except` 是为了兼容没有这个模块的平台。
try:
    import fcntl
except Exception:
    fcntl = None


@dataclass
class GenerationConfig:
    """生成参数配置。

    `@dataclass` 适合这种“主要用来存数据”的类，
    Python 会自动帮它生成初始化方法 `__init__` 等常见样板代码。
    """

    max_new_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.9
    use_cache: bool = True


def _has_accelerate() -> bool:
    """检查 `accelerate` 包是否已安装。"""

    try:
        return importlib.util.find_spec("accelerate") is not None
    except Exception:
        return False


def _read_mem_available_bytes() -> Optional[int]:
    """读取系统当前“可用内存”字节数。

    返回 `Optional[int]` 的意思是：
    - 成功时返回整数
    - 失败时返回 `None`
    """

    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    except Exception:
        pass
    return None


def _cpu_count() -> int:
    """获取当前进程实际可用的 CPU 核心数。"""

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


def _list_weight_files(model_path: str) -> List[Path]:
    """列出模型权重文件。

    这个函数会按常见格式依次尝试：
    1. 分片 safetensors 索引文件
    2. 目录下直接存在的 `.safetensors`
    3. 传统的 `pytorch_model.bin`
    """

    root = Path(model_path)
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        try:
            obj = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = sorted(set(obj.get("weight_map", {}).values()))
            files = [root / name for name in shard_names if (root / name).exists()]
            if files:
                return files
        except Exception:
            pass

    safetensors = sorted(root.glob("*.safetensors"))
    if safetensors:
        return safetensors

    model_bin = root / "pytorch_model.bin"
    if model_bin.exists():
        return [model_bin]

    return []


def _estimate_weight_bytes(model_path: str) -> int:
    """估算模型权重总大小。"""

    total = 0
    for file_path in _list_weight_files(model_path):
        try:
            total += int(file_path.stat().st_size)
        except Exception:
            continue
    return total


def _format_gib(num_bytes: Optional[int]) -> str:
    """把字节数格式化成更适合人看的 GiB 字符串。"""

    if not num_bytes:
        return "unknown"
    return f"{num_bytes / (1024 ** 3):.2f}GiB"


# 下面这些模块级变量属于“进程内共享状态”。
# 这样做的目的，是让整个程序运行期间可以复用已经做过的选择和缓存结果。
_SINGLE_GPU_SELECTION_LOCK = Lock()
_SINGLE_GPU_SELECTION: Optional[int] = None
_MODEL_STAGE_ROOT_LOCK = Lock()
_MODEL_STAGE_ROOT: Optional[Path] = None
_MODEL_STAGE_ROOT_RESOLVED = False
_MODEL_STAGE_MANIFEST = ".model_stage_manifest.json"
_PREPARED_MODEL_CACHE_LOCK = Lock()
_PREPARED_MODEL_CACHE: Dict[str, str] = {}


def _parse_cuda_index(device: str) -> Optional[int]:
    """把 `"cuda"` / `"cuda:1"` 这类字符串解析成 GPU 编号。"""

    value = str(device or "").strip().lower()
    if not value.startswith("cuda"):
        return None
    if ":" not in value:
        return 0
    try:
        return int(value.split(":", 1)[1])
    except Exception:
        return None


def _gpu_matches_preferred_target(name: str) -> bool:
    """判断某块 GPU 是否属于“优先使用”的型号。

    当前规则比较简单：如果显卡名字里含有 `4090`，就认为它是偏好的目标。
    """

    return "4090" in str(name or "").lower()


def _get_fixed_single_gpu_index() -> Optional[int]:
    """选择一张固定的 GPU 作为“单卡模式”的目标。

    设计思路：
    1. 如果配置里显式指定了 GPU 编号，优先听配置。
    2. 否则扫描所有 GPU，挑一张更合适的卡。
    3. 选中后缓存到模块变量里，后面重复使用，不再每次重新选。
    """

    global _SINGLE_GPU_SELECTION

    if not torch.cuda.is_available():
        return None

    # 允许通过外部配置强制指定 GPU 编号。
    override = runtime_int("single_gpu_index", None)
    if override is not None:
        try:
            idx = int(override)
        except Exception:
            idx = -1
        if 0 <= idx < torch.cuda.device_count():
            return idx
        print(f"[QwenVLRunner] invalid configured single_gpu_index={override!r}, ignore override")

    with _SINGLE_GPU_SELECTION_LOCK:
        if _SINGLE_GPU_SELECTION is not None and 0 <= _SINGLE_GPU_SELECTION < torch.cuda.device_count():
            return _SINGLE_GPU_SELECTION

        # `candidates` 保存所有 GPU 的信息；
        # `preferred` 则只收集“优先型号”的 GPU。
        candidates = []
        preferred = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            name = str(props.name)
            total_bytes = int(props.total_memory)
            try:
                free_bytes, _ = torch.cuda.mem_get_info(idx)
            except Exception:
                free_bytes = total_bytes
            used_bytes = max(0, total_bytes - int(free_bytes))
            item = {
                "index": idx,
                "name": name,
                "free": int(free_bytes),
                "used": used_bytes,
                "total": total_bytes,
            }
            candidates.append(item)
            if _gpu_matches_preferred_target(name):
                preferred.append(item)

        pool = preferred or candidates
        if not pool:
            return None

        # `key=lambda item: ...` 的作用是告诉 `max(...)`“按什么标准比较”。
        # 这里的优先级是：
        # 1. 空闲显存更多
        # 2. 已用显存更少
        # 3. 总显存更大
        # 4. GPU 编号更小
        chosen = max(pool, key=lambda item: (item["free"], -item["used"], item["total"], -item["index"]))
        _SINGLE_GPU_SELECTION = int(chosen["index"])
        print(
            "[QwenVLRunner] fixed single GPU selected: "
            f"cuda:{_SINGLE_GPU_SELECTION} ({chosen['name']}), "
            f"free={_format_gib(chosen['free'])}, used={_format_gib(chosen['used'])}, total={_format_gib(chosen['total'])}"
        )
        return _SINGLE_GPU_SELECTION


def _dir_signature_rows(root: Path) -> List[Tuple[str, int, int]]:
    """把目录内容整理成“签名原料”列表。

    每一项都是一个三元组 `(相对路径, 文件大小, 修改时间)`，
    后面会据此判断模型目录有没有变化。
    """

    rows: List[Tuple[str, int, int]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        stat = path.stat()
        rows.append((rel, int(stat.st_size), int(stat.st_mtime_ns)))
    return rows


def _dir_signature(rows: List[Tuple[str, int, int]]) -> str:
    """根据目录信息计算一个稳定的哈希签名。"""

    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _is_writable_directory(path: Path) -> bool:
    """检查某个目录是否可写。"""

    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def _resolve_model_stage_root() -> Optional[Path]:
    """决定“模型本地缓存目录”应该放在哪里。"""

    global _MODEL_STAGE_ROOT, _MODEL_STAGE_ROOT_RESOLVED

    if not runtime_bool("enable_model_staging", True):
        return None

    with _MODEL_STAGE_ROOT_LOCK:
        if _MODEL_STAGE_ROOT_RESOLVED:
            return _MODEL_STAGE_ROOT

        candidates: List[Path] = []
        configured_root = runtime_str("model_staging_dir", "").strip()
        if configured_root:
            candidates.append(Path(configured_root).expanduser())
        candidates.extend(
            [
                Path("/var/tmp/long_video_pipeline_model_cache"),
                Path("/tmp/long_video_pipeline_model_cache"),
            ]
        )

        resolved: Optional[Path] = None
        for candidate in candidates:
            if _is_writable_directory(candidate):
                resolved = candidate
                break

        _MODEL_STAGE_ROOT = resolved
        _MODEL_STAGE_ROOT_RESOLVED = True
        return _MODEL_STAGE_ROOT


@contextmanager
def _file_lock(lock_path: Path):
    """提供一个简单的文件锁。

    `@contextmanager` 让这个函数可以像下面这样使用：

    ```python
    with _file_lock(path):
        ...
    ```

    `yield` 前面的代码相当于“进入 with 块时执行”，
    `yield` 后面的 `finally` 相当于“退出 with 块时执行”。
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _read_stage_manifest(dst_root: Path) -> Dict[str, Any]:
    """读取模型缓存目录里的 manifest 文件。"""

    manifest_path = dst_root / _MODEL_STAGE_MANIFEST
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sync_model_tree(src_root: Path, dst_root: Path, rows: List[Tuple[str, int, int]]) -> None:
    """把源模型目录同步到目标缓存目录。

    这里做的是“增量同步”：
    - 已经一样的文件不重复复制
    - 目标目录里多余的旧文件会被删除
    """

    expected_files = {rel for rel, _, _ in rows}

    dst_root.mkdir(parents=True, exist_ok=True)
    for rel, size, mtime_ns in rows:
        src_path = src_root / rel
        dst_path = dst_root / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # 只有当目标文件不存在，或大小/修改时间不同，才需要重新复制。
        copy_needed = True
        if dst_path.exists():
            try:
                dst_stat = dst_path.stat()
                copy_needed = not (
                    int(dst_stat.st_size) == int(size)
                    and int(dst_stat.st_mtime_ns) == int(mtime_ns)
                )
            except Exception:
                copy_needed = True

        if copy_needed:
            shutil.copy2(src_path, dst_path)

    # 反向清理：把目标目录里“源目录已经没有了”的文件删掉。
    for path in sorted(dst_root.rglob("*"), reverse=True):
        rel = path.relative_to(dst_root).as_posix()
        if rel == _MODEL_STAGE_MANIFEST:
            continue
        if path.is_file() and rel not in expected_files:
            path.unlink()
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def prepare_model_artifacts(model_path: str) -> str:
    """准备模型可加载目录，并尽量落到本地高速缓存中。

    返回值仍然是“可用于加载模型的路径字符串”。
    如果缓存准备失败，会优雅地退回原始模型目录。
    """

    src_root = Path(model_path).expanduser()
    if not src_root.exists() or not src_root.is_dir():
        return str(model_path)

    stage_root = _resolve_model_stage_root()
    if stage_root is None:
        return str(src_root)

    try:
        resolved_src = src_root.resolve()
    except Exception:
        resolved_src = src_root

    cache_key_source = str(resolved_src)
    with _PREPARED_MODEL_CACHE_LOCK:
        cached_path = _PREPARED_MODEL_CACHE.get(cache_key_source)
    if cached_path:
        cached_root = Path(cached_path)
        if cached_root.exists() and cached_root.is_dir():
            return cached_path
        with _PREPARED_MODEL_CACHE_LOCK:
            _PREPARED_MODEL_CACHE.pop(cache_key_source, None)

    # 如果模型本身已经在 staging 目录里，就没必要再复制一份。
    if resolved_src == stage_root or stage_root in resolved_src.parents:
        resolved_path = str(resolved_src)
        with _PREPARED_MODEL_CACHE_LOCK:
            _PREPARED_MODEL_CACHE[cache_key_source] = resolved_path
        return resolved_path

    cache_key = f"{resolved_src.name}-{hashlib.sha1(str(resolved_src).encode('utf-8')).hexdigest()[:12]}"
    dst_root = stage_root / cache_key
    lock_path = stage_root / f".{cache_key}.lock"

    try:
        rows = _dir_signature_rows(resolved_src)
        source_signature = _dir_signature(rows)
    except Exception as e:
        print(f"[QwenVLRunner][WARN] failed to inspect model directory, skip fast local cache: {e}")
        return str(resolved_src)

    with _file_lock(lock_path):
        manifest = _read_stage_manifest(dst_root)
        if (
            manifest.get("source_path") == str(resolved_src)
            and manifest.get("source_signature") == source_signature
        ):
            # 源目录没变化，直接复用旧缓存。
            resolved_path = str(dst_root)
            with _PREPARED_MODEL_CACHE_LOCK:
                _PREPARED_MODEL_CACHE[cache_key_source] = resolved_path
            return resolved_path

        print(
            "[QwenVLRunner] sync model artifacts to fast local cache: "
            f"source={resolved_src}, cache={dst_root}"
        )
        try:
            _sync_model_tree(resolved_src, dst_root, rows)
            manifest = {
                "source_path": str(resolved_src),
                "source_signature": source_signature,
                "file_count": len(rows),
                "updated_at": int(time.time()),
            }
            (dst_root / _MODEL_STAGE_MANIFEST).write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            resolved_path = str(dst_root)
            with _PREPARED_MODEL_CACHE_LOCK:
                _PREPARED_MODEL_CACHE[cache_key_source] = resolved_path
            return resolved_path
        except Exception as e:
            print(f"[QwenVLRunner][WARN] fast local cache sync failed, fallback to source model path: {e}")
            resolved_path = str(resolved_src)
            with _PREPARED_MODEL_CACHE_LOCK:
                _PREPARED_MODEL_CACHE[cache_key_source] = resolved_path
            return resolved_path


def warm_model_file_cache(model_path: str, read_chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """预热模型文件缓存。

    它不会真正构建模型对象，而是把关键文件顺序读一遍，
    让操作系统更可能提前把数据放进页缓存中。
    """

    resolved_model_path = prepare_model_artifacts(model_path)
    root = Path(resolved_model_path)
    if not root.exists() or not root.is_dir():
        return resolved_model_path

    candidate_files: List[Path] = []
    candidate_files.extend(_list_weight_files(str(root)))

    # 除了权重文件，也顺手预热常见配置文件。
    for name in [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
    ]:
        p = root / name
        if p.exists():
            candidate_files.append(p)

    seen = set()
    ordered_files: List[Path] = []
    for path in candidate_files:
        key = str(path)
        if key in seen or not path.exists() or not path.is_file():
            continue
        seen.add(key)
        ordered_files.append(path)

    started = time.perf_counter()
    total_bytes = 0
    for path in ordered_files:
        try:
            # 如果系统支持 `posix_fadvise`，就给操作系统一个“我马上要读这个文件”的提示；
            # 否则退回到手动分块读取。
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_WILLNEED"):
                with path.open("rb") as f:
                    os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_WILLNEED)
            else:
                with path.open("rb", buffering=0) as f:
                    while True:
                        chunk = f.read(max(1, int(read_chunk_bytes)))
                        if not chunk:
                            break
        except Exception:
            with path.open("rb", buffering=0) as f:
                while True:
                    chunk = f.read(max(1, int(read_chunk_bytes)))
                    if not chunk:
                        break
        try:
            total_bytes += int(path.stat().st_size)
        except Exception:
            pass

    elapsed = time.perf_counter() - started
    print(
        "[QwenVLRunner] warm model file cache done: "
        f"model={resolved_model_path}, files={len(ordered_files)}, "
        f"bytes={_format_gib(total_bytes)}, elapsed={elapsed:.2f}s"
    )
    return resolved_model_path


class QwenVLRunner:
    """单个 Qwen3-VL 模型的运行时包装器。

    你可以把它理解成“一个会管理模型生命周期的对象”：
    - `load()` 负责加载模型
    - `generate()` 负责做一次推理
    - `unload()` 负责释放资源

    设计上，一个 `QwenVLRunner` 实例只管理一个模型。
    这样更容易控制显存，不会把 instruct / thinking 等多个模型同时堆在内存里。
    """

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        attn_implementation: Optional[str] = None,
        enable_dual_gpu_balance: bool = True,
        gpu0_mem_fraction: float = 0.75,
        gpu1_mem_fraction: float = 0.25,
        use_free_gpu_memory: bool = True,
        gpu_reserve_memory_mib: int = 1536,
        cpu_offload_max_memory: str = "64GiB",
    ) -> None:
        # 这里只是保存“配置”，并不真正加载模型。
        # 真正重量级的操作放在 `load()` 中，方便延迟初始化。
        self.source_model_path = model_path
        self.model_path = model_path
        self.device_map = self._normalize_device_map(device_map)
        self.torch_dtype = self._resolve_dtype(torch_dtype)
        self.attn_implementation = attn_implementation
        self.enable_dual_gpu_balance = enable_dual_gpu_balance
        self.gpu0_mem_fraction = gpu0_mem_fraction
        self.gpu1_mem_fraction = gpu1_mem_fraction
        self.use_free_gpu_memory = use_free_gpu_memory
        self.gpu_reserve_memory_mib = max(256, int(gpu_reserve_memory_mib))
        self.cpu_offload_max_memory = cpu_offload_max_memory

        self.model: Optional[Qwen3VLForConditionalGeneration] = None
        self.processor: Optional[AutoProcessor] = None

    @staticmethod
    def _resolve_dtype(dtype_str: str):
        """把字符串形式的 dtype 解析成 PyTorch 能识别的对象。"""

        if dtype_str == "auto":
            return "auto"
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(dtype_str.lower(), "auto")

    @staticmethod
    def _normalize_device_map(device_map: Optional[str]) -> str:
        """把外部传入的 `device_map` 统一整理成程序内部的标准写法。"""

        value = str(device_map or "").strip().lower()
        if value in {"", "none", "null", "single", "single_gpu"}:
            return "single"

        allowed_policies = {"auto", "balanced", "balanced_low_0", "sequential"}
        if value in allowed_policies:
            return value

        if value == "cpu" or value.startswith("cuda"):
            return value

        print(f"[QwenVLRunner] invalid device_map={device_map!r}, fallback to 'single'")
        return "single"

    def load(self) -> None:
        """加载模型和处理器。

        整体步骤是：
        1. 准备模型目录（必要时先同步到本地高速缓存）
        2. 决定加载策略（CPU / 单卡 / 自动分配）
        3. 组装 `from_pretrained(...)` 所需参数
        4. 加载模型权重
        5. 加载 `AutoProcessor`
        """

        load_model_path = prepare_model_artifacts(self.source_model_path)
        self.model_path = load_model_path
        configured_low_cpu_mem_usage = self._resolve_low_cpu_mem_usage()
        parallel_enabled, parallel_workers = self._configure_parallel_loading()
        estimated_weight_bytes = _estimate_weight_bytes(load_model_path)
        mem_available_bytes = _read_mem_available_bytes()
        resolved_device_map, target_device = self._resolve_load_device_map()
        requested_single_gpu = self.device_map == "single" or self.device_map.startswith("cuda")
        accelerate_available = _has_accelerate()
        single_gpu_target = (
            isinstance(target_device, str)
            and target_device.startswith("cuda")
            and requested_single_gpu
        )
        direct_device_map_single_gpu = accelerate_available and single_gpu_target
        manual_single_gpu_move = single_gpu_target and not direct_device_map_single_gpu
        low_cpu_mem_usage = configured_low_cpu_mem_usage or direct_device_map_single_gpu

        # `kwargs` 是一个普通字典，后面会通过 `**kwargs` 展开成关键字参数。
        # 例如：`from_pretrained(**kwargs)` 相当于手写
        # `from_pretrained(pretrained_model_name_or_path=..., torch_dtype=..., ...)`
        kwargs: Dict[str, Any] = {
            "pretrained_model_name_or_path": load_model_path,
            "torch_dtype": "auto" if manual_single_gpu_move else self.torch_dtype,
            "low_cpu_mem_usage": low_cpu_mem_usage,
        }
        if not manual_single_gpu_move:
            kwargs["device_map"] = resolved_device_map
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        # 当 `device_map="auto"` 且开启双卡均衡时，额外给出每张卡可用的上限。
        if self.device_map == "auto" and self.enable_dual_gpu_balance:
            max_memory = self._build_balanced_max_memory()
            if max_memory:
                print(f"[QwenVLRunner] dual-gpu balance enabled, max_memory={max_memory}")
                kwargs["max_memory"] = max_memory

        if requested_single_gpu and not single_gpu_target:
            print(
                "[QwenVLRunner][WARN] CUDA unavailable for single-GPU load; "
                f"fallback target_device={target_device}. Model will load on CPU and be much slower."
            )

        print(
            "[QwenVLRunner] load strategy: "
            f"source_model={self.source_model_path}, load_model={load_model_path}, "
            f"requested_device_map={self.device_map}, target_device={target_device}, "
            f"dtype={self.torch_dtype}, low_cpu_mem_usage={low_cpu_mem_usage}, "
            f"accelerate_available={accelerate_available}, direct_device_map_single_gpu={direct_device_map_single_gpu}, "
            f"manual_single_gpu_move={manual_single_gpu_move}, "
            f"parallel_loading={parallel_enabled}, parallel_workers={parallel_workers}, "
            f"estimated_weight_size={_format_gib(estimated_weight_bytes)}, mem_available={_format_gib(mem_available_bytes)}"
        )

        load_started = time.perf_counter()

        # 真正开始加载模型权重。
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(**kwargs)
        model_loaded_at = time.perf_counter()
        moved_to_device_at = model_loaded_at

        # 某些环境下，如果没法在 `from_pretrained` 阶段直接定位到单卡，
        # 就先按更保守的方式加载，再手动 `.to(...)` 移动到目标设备。
        if manual_single_gpu_move:
            move_kwargs: Dict[str, Any] = {"device": target_device}
            if self.torch_dtype != "auto":
                move_kwargs["dtype"] = self.torch_dtype
            self.model = self.model.to(**move_kwargs)
            moved_to_device_at = time.perf_counter()

        # `processor` 负责把“图片+文本”处理成模型输入，也负责把输出 token 解码回文本。
        self.processor = AutoProcessor.from_pretrained(load_model_path)
        finished_at = time.perf_counter()

        print(
            "[QwenVLRunner] load timings: "
            f"weights={model_loaded_at - load_started:.2f}s, "
            f"device_move={moved_to_device_at - model_loaded_at:.2f}s, "
            f"processor={finished_at - moved_to_device_at:.2f}s, total={finished_at - load_started:.2f}s"
        )

    def unload(self) -> None:
        """卸载模型并尽量回收显存/内存。"""

        self.model = None
        self.processor = None
        self._release_cuda_memory()

    def _resolve_load_device_map(self) -> Tuple[Any, str]:
        """把当前 `device_map` 配置转换成真正的加载参数和目标设备字符串。"""

        if self.device_map == "single":
            gpu_index = _get_fixed_single_gpu_index()
            if gpu_index is not None:
                return {"": gpu_index}, f"cuda:{gpu_index}"
            return {"": "cpu"}, "cpu"

        if self.device_map.startswith("cuda"):
            gpu_index = _parse_cuda_index(self.device_map)
            if gpu_index is not None:
                return {"": gpu_index}, f"cuda:{gpu_index}"
            return {"": self.device_map}, self.device_map

        if self.device_map == "cpu":
            return {"": "cpu"}, "cpu"

        return self.device_map, self.device_map

    def _resolve_low_cpu_mem_usage(self) -> bool:
        """预留的扩展点。

        现在固定返回 `False`，表示默认不额外启用这一策略。
        以后如果想按模型大小或机器配置动态决定，可以改这里。
        """

        return False

    def _configure_parallel_loading(self) -> Tuple[bool, int]:
        """配置 Hugging Face 的并行加载开关。"""

        enabled = runtime_bool("enable_parallel_loading", True)
        override = runtime_int("parallel_loading_workers", None)
        if override is not None:
            try:
                workers = max(1, int(override))
            except Exception:
                workers = 1
        else:
            shard_count = max(1, len(_list_weight_files(self.model_path)))
            workers = min(shard_count, max(1, min(_cpu_count(), 4)))

        actual_enabled = enabled and workers > 1

        # 这里是通过环境变量告诉 transformers/Hugging Face：
        # 是否开启并行加载，以及最多用多少 worker。
        os.environ["HF_ENABLE_PARALLEL_LOADING"] = "true" if actual_enabled else "false"
        os.environ["HF_PARALLEL_LOADING_WORKERS"] = str(max(1, workers))
        return actual_enabled, max(1, workers)

    def generate(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        gen_cfg: Optional[GenerationConfig] = None,
    ) -> str:
        """执行一次多模态生成。

        输入：
        - `prompt`：文本提示词
        - `images`：图片路径列表，可以为空
        - `gen_cfg`：生成参数；如果不传，就用默认配置

        输出：
        - 返回模型最终生成的文本字符串
        """

        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        cfg = gen_cfg or GenerationConfig()
        attempts = self._build_generation_attempts(
            max_new_tokens=cfg.max_new_tokens,
            image_count=len(images or []),
            use_cache=cfg.use_cache,
        )
        last_err: Optional[Exception] = None

        for idx, attempt in enumerate(attempts, start=1):
            active_images = list(images or [])[: attempt[2]]
            inputs = None
            generated_ids = None
            generated_ids_trimmed = None
            try:
                # 按 Qwen-VL 的多模态聊天格式组织内容：
                # 前面是若干张图片，最后拼上文本提示词。
                content = []
                for img in active_images:
                    content.append({"type": "image", "image": img})
                content.append({"type": "text", "text": prompt})

                messages = [{"role": "user", "content": content}]

                # `apply_chat_template(...)` 会做两件事：
                # 1. 按模型要求拼接对话格式
                # 2. 把结果转成 token / tensor
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )

                if torch.cuda.is_available():
                    target_device = self._pick_input_device()
                    if target_device:
                        # 这是一个“字典推导式”：
                        # 把 `inputs` 里的每一项都遍历一遍，能 `.to(...)` 的就移动到目标设备。
                        inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}

                do_sample = cfg.temperature > 0

                # `inference_mode()` 表示“这里只做推理，不做训练”，
                # 通常能减少额外开销。
                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=attempt[0],
                        do_sample=do_sample,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        use_cache=attempt[1],
                    )

                # `generate(...)` 返回的是“输入 token + 新生成 token”的完整序列。
                # 这里通过列表推导式，把前面的输入部分切掉，只保留新生成的内容。
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                return output_text[0] if output_text else ""
            except Exception as e:
                last_err = e
                if idx >= len(attempts) or not self._is_cuda_oom(e):
                    raise
                print(
                    "[QwenVLRunner] CUDA OOM, retry with lower memory mode: "
                    f"attempt={idx}/{len(attempts)}, "
                    f"max_new_tokens={attempt[0]}, use_cache={attempt[1]}, images={attempt[2]}"
                )
                self._release_cuda_memory()
            finally:
                # `del` 不是必须的，但这里显式删除大对象，有助于更快释放引用。
                del inputs
                del generated_ids
                del generated_ids_trimmed
                self._release_cuda_memory()

        if last_err is not None:
            raise last_err
        return ""

    def _build_balanced_max_memory(self) -> Optional[Dict[Any, str]]:
        """为自动分卡加载构建 `max_memory` 参数。"""

        if not torch.cuda.is_available():
            return None

        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            return None

        def _fraction_to_mib(device_idx: int, fraction: float) -> str:
            # 这是一个内部辅助函数，只在当前方法里使用。
            total_bytes = int(torch.cuda.get_device_properties(device_idx).total_memory)
            frac = max(0.05, min(0.98, float(fraction)))
            limit_bytes = int(total_bytes * frac)

            if self.use_free_gpu_memory:
                try:
                    free_bytes, _ = torch.cuda.mem_get_info(device_idx)
                except Exception:
                    free_bytes = total_bytes
                reserve_bytes = int(self.gpu_reserve_memory_mib) * 1024 ** 2
                limit_bytes = min(limit_bytes, max(0, int(free_bytes) - reserve_bytes))

            mib = max(256, int(limit_bytes / (1024 ** 2)))
            return f"{mib}MiB"

        max_memory: Dict[Any, str] = {}
        max_memory[0] = _fraction_to_mib(0, self.gpu0_mem_fraction)
        max_memory[1] = _fraction_to_mib(1, self.gpu1_mem_fraction)

        for i in range(2, gpu_count):
            max_memory[i] = _fraction_to_mib(i, 0.3)

        max_memory["cpu"] = self.cpu_offload_max_memory
        return max_memory

    def _pick_input_device(self) -> Optional[str]:
        """推断输入张量应该放到哪个设备上。"""

        if self.model is None:
            return None

        # 如果模型是由 Hugging Face 自动切分的，`hf_device_map` 往往最可靠。
        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for dev in device_map.values():
                if isinstance(dev, int):
                    return f"cuda:{dev}"
                if isinstance(dev, str) and dev.startswith("cuda"):
                    return dev

        dev = getattr(self.model, "device", None)
        if dev is not None:
            dev_str = str(dev)
            if dev_str.startswith("cuda"):
                return dev_str

        if torch.cuda.is_available():
            return "cuda"
        return None

    @staticmethod
    def _is_cuda_oom(err: Exception) -> bool:
        """判断异常是否属于显存不足（OOM）。"""

        msg = str(err).lower()
        return isinstance(err, torch.cuda.OutOfMemoryError) or "cuda out of memory" in msg or "out of memory" in msg

    def _build_generation_attempts(
        self,
        max_new_tokens: int,
        image_count: int,
        use_cache: bool,
    ) -> List[Tuple[int, bool, int]]:
        """构建“逐步降级”的生成尝试方案。

        返回的列表里，每个元素都是：
        `(max_new_tokens, use_cache, image_count)`

        如果第一次生成 OOM，后面就会按这个列表依次尝试更省显存的组合。
        """

        base_tokens = max(64, int(max_new_tokens))
        base_images = max(0, int(image_count))
        attempts: List[Tuple[int, bool, int]] = []

        def _add(tokens: int, cache_flag: bool, images: int) -> None:
            # 用内部函数统一做“归一化 + 去重”。
            item = (max(64, int(tokens)), bool(cache_flag), max(0, int(images)))
            if item not in attempts:
                attempts.append(item)

        _add(base_tokens, use_cache, base_images)
        if use_cache:
            _add(base_tokens, False, base_images)

        for factor in (0.8, 0.65, 0.5):
            _add(max(256, int(base_tokens * factor)), False, base_images)

        if base_images > 1:
            _add(max(384, int(base_tokens * 0.65)), False, 1)
            _add(max(256, int(base_tokens * 0.5)), False, 1)

        return attempts

    @staticmethod
    def _release_cuda_memory() -> None:
        """尽量触发 Python 和 CUDA 的内存回收。"""

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


class SharedQwenVLRunnerPool:
    """共享的 `QwenVLRunner` 池。

    作用是：相同配置的模型只加载一次，后续直接复用。
    这样可以减少重复加载带来的时间和显存波动。
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._runners: Dict[Tuple[Any, ...], QwenVLRunner] = {}

    @staticmethod
    def _build_key(
        model_path: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        attn_implementation: Optional[str] = None,
        enable_dual_gpu_balance: bool = True,
        gpu0_mem_fraction: float = 0.75,
        gpu1_mem_fraction: float = 0.25,
        use_free_gpu_memory: bool = True,
        gpu_reserve_memory_mib: int = 1536,
        cpu_offload_max_memory: str = "64GiB",
    ) -> Tuple[Any, ...]:
        """把一组加载参数整理成可哈希的元组键。

        之所以用元组，是因为它可以作为字典的 key；
        而列表、字典本身不能直接当 key。
        """

        return (
            str(model_path),
            QwenVLRunner._normalize_device_map(device_map),
            str(torch_dtype or "auto").lower(),
            str(attn_implementation or ""),
            bool(enable_dual_gpu_balance),
            round(float(gpu0_mem_fraction), 6),
            round(float(gpu1_mem_fraction), 6),
            bool(use_free_gpu_memory),
            int(gpu_reserve_memory_mib),
            str(cpu_offload_max_memory),
        )

    def get_or_load(
        self,
        model_path: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        attn_implementation: Optional[str] = None,
        enable_dual_gpu_balance: bool = True,
        gpu0_mem_fraction: float = 0.75,
        gpu1_mem_fraction: float = 0.25,
        use_free_gpu_memory: bool = True,
        gpu_reserve_memory_mib: int = 1536,
        cpu_offload_max_memory: str = "64GiB",
    ) -> Tuple[QwenVLRunner, bool]:
        """获取已加载的 runner；如果没有，就创建并加载。

        返回值是一个二元组：
        - 第 1 项：`QwenVLRunner` 实例
        - 第 2 项：`bool`，表示这次是不是直接复用了已加载实例
        """

        key = self._build_key(
            model_path=model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            enable_dual_gpu_balance=enable_dual_gpu_balance,
            gpu0_mem_fraction=gpu0_mem_fraction,
            gpu1_mem_fraction=gpu1_mem_fraction,
            use_free_gpu_memory=use_free_gpu_memory,
            gpu_reserve_memory_mib=gpu_reserve_memory_mib,
            cpu_offload_max_memory=cpu_offload_max_memory,
        )

        with self._lock:
            runner = self._runners.get(key)
            if runner is not None and runner.model is not None and runner.processor is not None:
                return runner, True

            # 没有现成实例时，先创建一个空壳 runner，再执行实际加载。
            if runner is None:
                runner = QwenVLRunner(
                    model_path=model_path,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_implementation,
                    enable_dual_gpu_balance=enable_dual_gpu_balance,
                    gpu0_mem_fraction=gpu0_mem_fraction,
                    gpu1_mem_fraction=gpu1_mem_fraction,
                    use_free_gpu_memory=use_free_gpu_memory,
                    gpu_reserve_memory_mib=gpu_reserve_memory_mib,
                    cpu_offload_max_memory=cpu_offload_max_memory,
                )
                self._runners[key] = runner

            try:
                runner.load()
            except Exception:
                self._runners.pop(key, None)
                raise

            return runner, False

    def unload_all(self) -> int:
        """卸载池中所有 runner，并返回卸载数量。"""

        with self._lock:
            runners = list(self._runners.values())
            self._runners = {}

        for runner in runners:
            runner.unload()

        return len(runners)
