from __future__ import annotations

"""
这个模块负责“检索”相关的核心逻辑，可以把它理解成 4 步：

1. 找到可用的 embedding 模型（本地模型优先，必要时降级）。
2. 把文本转成向量。
3. 把视频分块后的文本写入 ChromaDB 向量库。
4. 用户提问时，找出最相关的 chunk，并按需要补上相邻 chunk 作为上下文。

如果你是新手，读这个文件时可以重点抓住两条主线：
- 数据写入：`add_chunk_cards()`
- 数据查询：`query()` / `build_question_evidence_pack()`
"""

import hashlib
import json
import math
import os
import re
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

from .schemas import ChunkCard
from .settings import DEFAULT_EMBEDDING_MODEL_NAME, runtime_bool
from .utils import compact_text


def _project_root() -> Path:
    """返回项目根目录。

    `__file__` 是当前文件路径。
    `resolve()` 会把它变成绝对路径。
    `parents[1]` 表示“向上数第 2 层目录”：
    - `parents[0]` 是 `long_video_pipeline/`
    - `parents[1]` 是整个项目根目录
    """
    return Path(__file__).resolve().parents[1]


# 这里记录“项目根目录名”，用于兼容旧版本保存下来的路径。
LEGACY_PROJECT_MARKER = _project_root().name

# 下面两个全局变量实现“embedding 模型缓存”：
# - `_EMBEDDER_CACHE_LOCK`：多线程时加锁，避免重复加载同一个模型
# - `_EMBEDDER_CACHE`：把已经加载好的模型实例缓存起来
_EMBEDDER_CACHE_LOCK = Lock()
_EMBEDDER_CACHE: Dict[str, Any] = {}


def _rewrite_legacy_project_path(path_value: str) -> str:
    """把旧项目路径重写成当前项目路径。

    例如以前路径里可能带着旧的项目根目录名，迁移后目录位置变了，
    这个函数会尽量把“后半段相对路径”接到当前项目根目录下面。
    """
    # `path_value or ""` 的意思是：
    # 如果 `path_value` 是空字符串 / None / False，就先退回到空字符串。
    # 再 `str(...)` 保证后面一定是在处理字符串。
    raw = str(path_value or "").strip()
    if not raw:
        return raw

    try:
        # `expanduser()` 会把 `~` 展开成用户家目录。
        normalized = str(Path(raw).expanduser())
        # `parts` 会把路径拆成一个个片段组成的元组(tuple)。
        # 例如 `/a/b/c` 会变成 `("/", "a", "b", "c")`
        path_parts = Path(normalized).parts
    except Exception:
        # 如果路径格式有问题，就直接返回原值，不让程序在这里中断。
        return raw

    if LEGACY_PROJECT_MARKER not in path_parts:
        return normalized

    # `path_parts[::-1]` 是“倒序切片”，表示把整个序列反过来。
    # 这里的目的，是找到“最后一次出现旧项目根目录名”的位置。
    marker_index = len(path_parts) - 1 - path_parts[::-1].index(LEGACY_PROJECT_MARKER)
    # 切片 `marker_index + 1 :` 表示取标记后面的所有路径片段。
    suffix_parts = path_parts[marker_index + 1 :]
    if not suffix_parts:
        return str(_project_root())

    # `*suffix_parts` 是“解包”语法，会把列表/元组里的多个元素
    # 依次作为位置参数传给 `joinpath(...)`。
    return str(_project_root().joinpath(*suffix_parts))


def _resolve_embedding_model_name_or_path(model_name_or_path: str) -> str:
    """解析 embedding 模型名字或路径，尽量定位到实际存在的本地路径。"""
    # 如果传入值为空，就退回默认模型名。
    raw_input = str(model_name_or_path or "").strip() or DEFAULT_EMBEDDING_MODEL_NAME
    raw = _rewrite_legacy_project_path(raw_input)

    project_root = _project_root()
    p = Path(raw).expanduser()
    # `candidates` 表示“候选路径列表”，后面会按顺序挨个尝试。
    candidates = [
        p,
        (project_root / raw).expanduser(),
    ]

    models_dir = project_root / "models"
    # 这是一个列表推导式(list comprehension)：
    # `raw.split("/")` 先按 `/` 分割，`if part` 会过滤掉空字符串。
    repo_parts = [part for part in raw.split("/") if part]
    if repo_parts:
        basename = repo_parts[-1]
        # `extend(...)` 是把一个列表里的多个元素批量追加进来。
        candidates.extend(
            [
                models_dir / basename,
                models_dir / raw.replace("/", "--"),
                models_dir / raw.replace("/", "_"),
            ]
        )
    if len(repo_parts) >= 2:
        org = repo_parts[0]
        name = repo_parts[-1]
        candidates.extend(
            [
                models_dir / org / name,
                models_dir / f"{org}--{name}",
            ]
        )

    # 用 `set()` 去重，避免同一路径被重复检查。
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return str(candidate)

    # 如果上面的候选路径都不存在，就保留原始值，
    # 让后面的加载逻辑继续尝试（例如把它当成 Hugging Face 模型名）。
    return raw


class HashingEmbedder:
    """离线兜底的 embedding 实现。

    当 `sentence-transformers` 模型不可用时，程序仍然需要一个“把文本变成向量”
    的对象，于是用这个类做降级方案。

    它不是真正的语义模型，效果通常不如正式 embedding 模型，但优点是：
    - 不依赖联网下载
    - 不依赖额外模型文件
    - 接口和真正的 embedder 足够像，便于替换
    """

    # 这个正则表达式会提取：
    # - 单个中文字符：`[\u4e00-\u9fff]`
    # - 英文/数字/下划线组成的 token：`[A-Za-z0-9_]+`
    TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")

    def __init__(self, dim: int = 384) -> None:
        # `max(64, int(dim))` 表示向量维度至少为 64，避免维度太小。
        self.dim = max(64, int(dim))

    def encode(self, texts: List[str], normalize_embeddings: bool = True) -> List[List[float]]:
        """把多条文本编码成多个向量。

        返回值是“二维列表”：
        - 外层列表：每条文本对应一个向量
        - 内层列表：这个向量里的每一维数值
        """
        rows: List[List[float]] = []
        for text in texts:
            # 先创建一个固定长度、全是 0 的向量。
            vec = [0.0] * self.dim
            # `findall()` 会把正则匹配到的所有 token 都找出来。
            # `lower()` 转小写，减少大小写差异带来的影响。
            tokens = self.TOKEN_PATTERN.findall(str(text or "").lower())
            if not tokens:
                # 如果整段文本一个 token 都没提出来，就塞一个占位 token。
                tokens = ["(empty)"]

            for tok in tokens:
                # `sha1` 会把 token 变成固定长度的哈希结果。
                digest = hashlib.sha1(tok.encode("utf-8")).digest()
                # 取哈希前 4 个字节映射到某个维度下标。
                idx = int.from_bytes(digest[:4], byteorder="big", signed=False) % self.dim
                # 再用第 5 个字节决定加正号还是负号，减少哈希碰撞偏差。
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign

            if normalize_embeddings:
                # 这里在做 L2 归一化，让向量长度约等于 1。
                norm = math.sqrt(sum(v * v for v in vec))
                if norm > 0:
                    # 这是一个列表推导式：把每个元素都除以 `norm`。
                    vec = [v / norm for v in vec]
            rows.append(vec)
        return rows


def _build_embedder(embedding_model_name: str, resolved: str):
    """按“本地优先，联网可选，最后降级”的顺序构建 embedder。"""
    allow_online = runtime_bool("allow_embedding_download", False)
    tried: List[str] = []

    def _try_build(name_or_path: str, local_only: bool):
        # 这是一个“条件表达式（三元表达式）”：
        # `A if 条件 else B`
        kwargs = {"local_files_only": True} if local_only else {}
        try:
            return SentenceTransformer(name_or_path, **kwargs)
        except TypeError:
            # 某些版本的 `SentenceTransformer` 可能不接受 `local_files_only` 参数。
            # 如果目标本身是一个本地目录，就退一步直接只传路径。
            if local_only and Path(name_or_path).exists():
                return SentenceTransformer(name_or_path)
            raise

    try:
        embedder = _try_build(resolved, local_only=True)
        print(f"[Retrieval] using sentence-transformers embedder: {resolved} (offline/local)")
        return embedder
    except Exception as e:
        tried.append(f"local_only failed: {e}")

    if allow_online:
        try:
            embedder = _try_build(resolved, local_only=False)
            print(f"[Retrieval] using sentence-transformers embedder: {resolved} (online)")
            return embedder
        except Exception as e:
            tried.append(f"online failed: {e}")

    print(
        "[Retrieval][WARN] sentence-transformers model unavailable; "
        "falling back to offline hashing embedder. "
        f"model={embedding_model_name}, resolved={resolved}, details={' | '.join(tried)}"
    )
    return HashingEmbedder(dim=384)


def get_shared_embedder(embedding_model_name: str):
    """获取共享的 embedder 实例。

    这里用了缓存，所以同一个模型不会被反复加载。
    """
    resolved = _resolve_embedding_model_name_or_path(embedding_model_name)
    # `with _EMBEDDER_CACHE_LOCK:` 表示进入一个“加锁代码块”。
    # 代码块执行完后会自动释放锁，这样多线程下更安全。
    with _EMBEDDER_CACHE_LOCK:
        embedder = _EMBEDDER_CACHE.get(resolved)
        if embedder is None:
            embedder = _build_embedder(embedding_model_name=embedding_model_name, resolved=resolved)
            _EMBEDDER_CACHE[resolved] = embedder
        return embedder


def preload_embedding_model(embedding_model_name: str) -> str:
    """提前把模型加载进缓存，返回解析后的路径/名字。"""
    resolved = _resolve_embedding_model_name_or_path(embedding_model_name)
    get_shared_embedder(embedding_model_name)
    return resolved


class VectorStore:
    """对 ChromaDB 的一层简单封装。

    这个类负责：
    - 初始化持久化向量库
    - 把 chunk 写进去
    - 根据 query 查回相似 chunk
    - 按 chunk 顺序扩展前后邻居
    """

    def __init__(self, persist_dir: str, embedding_model_name: str, embedder: Optional[Any] = None) -> None:
        # `Path(...)` 比直接拼字符串路径更安全、也更容易跨平台。
        self.persist_dir = Path(persist_dir)
        # `parents=True`：父目录不存在时一并创建
        # `exist_ok=True`：目录已存在也不报错
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(name="video_chunks")
        # 如果外部已经传入 embedder 就直接用；否则自己加载。
        self.embedder = embedder if embedder is not None else self._load_embedder(embedding_model_name)

        # 这两个成员是为了“找相邻 chunk”准备的索引结构：
        # - `chunk_ids_sorted`：按 chunk 编号排序后的 chunk_id 列表
        # - `chunk_id_to_pos`：chunk_id -> 它在列表中的位置
        self.chunk_ids_sorted: List[str] = []
        self.chunk_id_to_pos: Dict[str, int] = {}
        # 如果持久化目录里已经有旧数据，启动时就把索引恢复出来。
        self.hydrate_chunk_index()

    def reset(self) -> None:
        """清空当前 collection。"""
        try:
            self.client.delete_collection("video_chunks")
        except Exception:
            # collection 不存在时忽略错误，保持 reset 的幂等性。
            pass
        self.collection = self.client.get_or_create_collection(name="video_chunks")
        self.chunk_ids_sorted = []
        self.chunk_id_to_pos = {}

    def hydrate_chunk_index(self) -> None:
        """从向量库已有数据里恢复 chunk 顺序索引。"""
        try:
            got = self.collection.get(include=["metadatas"])
        except Exception:
            return

        # `dict.get("metadatas", [])`：拿不到就给默认值 `[]`
        # `or []`：即便拿到的是 `None`，也再次兜底为空列表
        metas = got.get("metadatas", []) or []
        uniq_chunk_ids = sorted(
            {
                # 这是一个集合推导式(set comprehension)：
                # 作用是“抽取出所有非空 chunk_id，并自动去重”。
                str(m.get("chunk_id", "")).strip()
                for m in metas
                if isinstance(m, dict) and str(m.get("chunk_id", "")).strip()
            },
            # `key=self._chunk_num` 表示排序时，不按字符串字典序，
            # 而是按 chunk_id 末尾的数字大小排序。
            key=self._chunk_num,
        )
        self.chunk_ids_sorted = uniq_chunk_ids
        # 这是一个字典推导式(dict comprehension)。
        # 例如 `["chunk_0", "chunk_1"]` 会变成
        # `{"chunk_0": 0, "chunk_1": 1}`
        self.chunk_id_to_pos = {cid: i for i, cid in enumerate(uniq_chunk_ids)}

    @staticmethod
    def _load_embedder(embedding_model_name: str):
        # `@staticmethod` 表示这是“静态方法”：
        # 它属于这个类，但不依赖 `self`（实例）本身。
        return get_shared_embedder(embedding_model_name)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """把多条文本转成普通 Python 列表格式的向量。"""
        vectors = self.embedder.encode(texts, normalize_embeddings=True)
        rows: List[List[float]] = []
        for v in vectors:
            # 有些 embedding 库返回的是 numpy 数组 / tensor，
            # 它们通常带 `.tolist()` 方法；这里统一转成纯 Python 列表。
            if hasattr(v, "tolist"):
                rows.append(v.tolist())
            else:
                rows.append([float(x) for x in v])
        return rows

    @staticmethod
    def _chunk_num(chunk_id: str) -> int:
        """提取 chunk_id 末尾的数字，用于排序。

        例如：
        - `chunk_7` -> 7
        - `part_12` -> 12
        - 如果末尾没有数字 -> -1
        """
        m = re.search(r"(\d+)$", chunk_id)
        # `X if 条件 else Y` 仍然是条件表达式。
        return int(m.group(1)) if m else -1

    def add_chunk_cards(self, video_id: str, chunk_cards: List[ChunkCard]) -> None:
        """把每个 chunk 的摘要和原始文本一起写入向量库。"""
        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict] = []

        # 每个 `ChunkCard` 会被拆成两条记录：
        # 1. `chunk_card`：结构化摘要，更适合概览检索
        # 2. `raw_text`：ASR/OCR 原始文本，更适合细节回查
        for c in chunk_cards:
            card_text = (
                f"[{c.t_start}-{c.t_end}] {c.one_liner}\n"
                f"bullets: {c.bullets}\n"
                f"entities: {c.entities}\n"
                f"visual_facts: {c.visual_facts}\n"
                f"tags: {c.tags}"
            )
            raw_text = f"ASR: {compact_text(c.asr_text, 2000)}\nOCR: {compact_text(c.ocr_text, 1500)}"

            # `extend([...])` 是一次加入多个元素。
            ids.extend([f"{c.chunk_id}__card", f"{c.chunk_id}__raw"])
            docs.extend([card_text, raw_text])
            metas.extend(
                [
                    {
                        "video_id": video_id,
                        "chunk_id": c.chunk_id,
                        "section_id": c.section_id,
                        "t_start": c.t_start,
                        "t_end": c.t_end,
                        "importance": c.importance,
                        "modality": "chunk_card",
                    },
                    {
                        "video_id": video_id,
                        "chunk_id": c.chunk_id,
                        "section_id": c.section_id,
                        "t_start": c.t_start,
                        "t_end": c.t_end,
                        "importance": c.importance,
                        "modality": "raw_text",
                    },
                ]
            )

        # 文本和元数据准备好后，统一做 embedding，再批量写入 collection。
        embeds = self._embed(docs)
        self.collection.add(ids=ids, documents=docs, embeddings=embeds, metadatas=metas)

        # 这里再次构建 chunk 顺序索引，后面扩展相邻 chunk 时会用到。
        uniq_chunk_ids = sorted({c.chunk_id for c in chunk_cards}, key=self._chunk_num)
        self.chunk_ids_sorted = uniq_chunk_ids
        self.chunk_id_to_pos = {cid: i for i, cid in enumerate(uniq_chunk_ids)}

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """按 query 检索最相似的若干条记录。"""
        if not self.chunk_ids_sorted:
            return []

        # `_embed()` 接收的是“文本列表”，所以这里要写成 `[query_text]`。
        # 因为只查一条 query，所以最后用 `[0]` 取出第一条向量。
        q_emb = self._embed([query_text])[0]
        result = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        rows: List[Dict] = []
        # Chroma 的返回值是“批量查询格式”，即使只查 1 条 query，
        # `documents` / `metadatas` / `distances` 也通常是二维列表，
        # 所以这里统一取第 0 项。
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        # `zip(a, b, c)` 会把三组列表按位置对齐后一起迭代。
        for d, m, dist in zip(docs, metas, dists):
            rows.append(
                {
                    "document": d,
                    "metadata": m,
                    "distance": dist,
                }
            )
        return rows

    def fetch_chunk_rows(self, chunk_id: str) -> List[Dict]:
        """取回某个 chunk_id 对应的所有记录。"""
        got = self.collection.get(
            where={"chunk_id": chunk_id},
            include=["documents", "metadatas"],
        )
        docs = got.get("documents", [])
        metas = got.get("metadatas", [])

        rows: List[Dict] = []
        for d, m in zip(docs, metas):
            # 这里不是相似度检索出来的结果，所以 `distance` 设为 `None`。
            rows.append({"document": d, "metadata": m, "distance": None})
        return rows

    def expand_hits_with_neighbors(self, hits: List[Dict], neighbor_k: int) -> List[Dict]:
        """把命中的 chunk 向左右扩展若干个相邻 chunk。"""
        selected_chunk_ids: List[str] = []
        seen_chunk_ids = set()

        for h in hits:
            # `h.get("metadata", {}) or {}` 的意思是：
            # - 优先拿 `metadata`
            # - 如果没有这个键，就用空字典
            # - 如果拿到的是 `None`，也继续退回空字典
            m = h.get("metadata", {}) or {}
            cid = m.get("chunk_id", "")
            if cid and cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                selected_chunk_ids.append(cid)

        if neighbor_k > 0 and self.chunk_ids_sorted:
            extra = []
            # `list(selected_chunk_ids)` 是拷贝一份列表。
            # 这样即使后面 `selected_chunk_ids.extend(extra)`，当前循环也不会受影响。
            for cid in list(selected_chunk_ids):
                pos = self.chunk_id_to_pos.get(cid)
                if pos is None:
                    continue
                left = max(0, pos - neighbor_k)
                right = min(len(self.chunk_ids_sorted) - 1, pos + neighbor_k)
                for i in range(left, right + 1):
                    ncid = self.chunk_ids_sorted[i]
                    if ncid not in seen_chunk_ids:
                        seen_chunk_ids.add(ncid)
                        extra.append(ncid)
            selected_chunk_ids.extend(extra)

        rows: List[Dict] = []
        for cid in selected_chunk_ids:
            rows.extend(self.fetch_chunk_rows(cid))
        return rows


def build_claim_evidence_pack(
    store: VectorStore,
    claims: List[str],
    top_k: int,
    neighbor_k: int,
) -> str:
    """把多条 claim 的检索证据拼成一段文本。"""
    lines: List[str] = []
    # `enumerate(..., start=1)` 会在遍历时同时给出编号，并从 1 开始计数。
    for i, claim in enumerate(claims, start=1):
        lines.append(f"## Claim {i}: {claim}")
        hits = store.query(claim, top_k=top_k)
        expanded = store.expand_hits_with_neighbors(hits, neighbor_k=neighbor_k)

        # 切片 `[:N]` 表示“只取前 N 项”。
        for h in expanded[: top_k * 2 + max(0, neighbor_k) * 2]:
            m = h.get("metadata", {}) or {}
            lines.append(
                "- "
                + json.dumps(
                    {
                        "t_start": m.get("t_start", ""),
                        "t_end": m.get("t_end", ""),
                        "section_id": m.get("section_id", ""),
                        "modality": m.get("modality", ""),
                        # `compact_text(..., 500)` 用来截断证据长度，避免文本太长。
                        "evidence": compact_text(h.get("document", ""), 500),
                    },
                    # `ensure_ascii=False` 能让中文按原样输出，不会被转成 `\uXXXX`。
                    ensure_ascii=False,
                )
            )
        lines.append("")

    return "\n".join(lines)


def build_question_evidence_pack(
    store: VectorStore,
    question: str,
    top_k: int,
    neighbor_k: int,
    max_rows_per_chunk: int = 2,
) -> Tuple[str, List[Dict[str, str]]]:
    """针对单个向量库构造“问答证据包”。

    返回值是一个二元组(tuple)：
    - 第 1 项：拼好的文本
    - 第 2 项：结构化证据列表
    """
    query_text = str(question or "").strip()
    if not query_text:
        return "", []

    # 这里把外部传入的参数“修正”为安全范围：
    # - `top_k` 至少是 1
    # - `neighbor_k` 至少是 0
    # - `max_rows_per_chunk` 至少是 1
    effective_top_k = max(1, int(top_k))
    effective_neighbor_k = max(0, int(neighbor_k))
    effective_rows_per_chunk = max(1, int(max_rows_per_chunk))

    # 先多取一点候选结果，因为后面要按 chunk 去重。
    hits = store.query(query_text=query_text, top_k=max(effective_top_k * 2, effective_top_k))
    if not hits:
        return "", []

    selected_chunk_ids: List[str] = []
    seen_chunk_ids = set()
    # 记录“每个 chunk 当前看到的最好距离（距离越小越相似）”。
    best_distance_by_chunk: Dict[str, float] = {}

    for hit in hits:
        meta = hit.get("metadata", {}) or {}
        chunk_id = str(meta.get("chunk_id", "")).strip()
        if not chunk_id:
            continue
        if chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            selected_chunk_ids.append(chunk_id)
        dist = hit.get("distance")
        if chunk_id not in best_distance_by_chunk or float(dist or 0.0) < best_distance_by_chunk[chunk_id]:
            best_distance_by_chunk[chunk_id] = float(dist or 0.0)
        # 这里的停止条件是“已经收集到足够多的唯一 chunk”，
        # 而不是“已经遍历完所有命中行”。
        if len(selected_chunk_ids) >= effective_top_k:
            break

    if effective_neighbor_k > 0 and store.chunk_ids_sorted:
        extra_chunk_ids: List[str] = []
        for chunk_id in list(selected_chunk_ids):
            pos = store.chunk_id_to_pos.get(chunk_id)
            if pos is None:
                continue
            left = max(0, pos - effective_neighbor_k)
            right = min(len(store.chunk_ids_sorted) - 1, pos + effective_neighbor_k)
            for idx in range(left, right + 1):
                neighbor_chunk_id = store.chunk_ids_sorted[idx]
                if neighbor_chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(neighbor_chunk_id)
                extra_chunk_ids.append(neighbor_chunk_id)
        selected_chunk_ids.extend(extra_chunk_ids)

    evidence_rows: List[Dict[str, str]] = []
    # 最终总行数上限 = chunk 数量 * 每个 chunk 最多保留的行数
    max_total_rows = max(effective_top_k, len(selected_chunk_ids)) * effective_rows_per_chunk

    for chunk_id in selected_chunk_ids:
        chunk_rows = store.fetch_chunk_rows(chunk_id)
        chunk_rows.sort(
            # `sort(key=...)` 表示按 key 计算出来的值排序。
            # 这里 key 是一个二元组：
            # 1. 先让 `chunk_card` 排在前面
            # 2. 如果同一优先级，再按 `modality` 字符串排序
            key=lambda row: (
                0 if str((row.get("metadata", {}) or {}).get("modality", "")) == "chunk_card" else 1,
                str((row.get("metadata", {}) or {}).get("modality", "")),
            )
        )

        # 控制“同一个 chunk 同一种 modality 只保留一条”。
        added_modalities = set()
        for row in chunk_rows:
            meta = row.get("metadata", {}) or {}
            modality = str(meta.get("modality", "")).strip() or "unknown"
            if modality in added_modalities:
                continue

            evidence_text = compact_text(
                str(row.get("document", "")).strip(),
                # 摘要信息通常更短，所以 `chunk_card` 给更小的长度上限。
                260 if modality == "chunk_card" else 380,
            )
            if not evidence_text:
                continue

            evidence_row = {
                "chunk_id": chunk_id,
                "t_start": str(meta.get("t_start", "")).strip(),
                "t_end": str(meta.get("t_end", "")).strip(),
                "section_id": str(meta.get("section_id", "")).strip(),
                "modality": modality,
                "distance": f"{best_distance_by_chunk[chunk_id]:.4f}" if chunk_id in best_distance_by_chunk else "",
                "evidence": evidence_text,
            }
            evidence_rows.append(evidence_row)
            added_modalities.add(modality)

            if len(added_modalities) >= effective_rows_per_chunk or len(evidence_rows) >= max_total_rows:
                break

        if len(evidence_rows) >= max_total_rows:
            break

    if not evidence_rows:
        return "", []

    lines = ["## Question Evidence"]
    for row in evidence_rows:
        lines.append("- " + json.dumps(row, ensure_ascii=False))
    return "\n".join(lines), evidence_rows


def build_multi_store_question_evidence_pack(
    sources: List[Dict[str, Any]],
    question: str,
    top_k: int,
    neighbor_k: int,
    max_rows_per_chunk: int = 2,
) -> Tuple[str, List[Dict[str, str]]]:
    """同时从多个向量库中检索证据，并合并成统一结果。"""
    query_text = str(question or "").strip()
    if not query_text:
        return "", []

    effective_top_k = max(1, int(top_k))
    effective_neighbor_k = max(0, int(neighbor_k))
    effective_rows_per_chunk = max(1, int(max_rows_per_chunk))

    # `candidate_hits` 先收集“各个 source 的候选命中”，
    # 后面再统一排序和截断。
    candidate_hits: List[Dict[str, Any]] = []
    for source in sources:
        store = source.get("store")
        if not isinstance(store, VectorStore) or not store.chunk_ids_sorted:
            continue

        try:
            hits = store.query(
                query_text=query_text,
                top_k=max(effective_top_k * 2, effective_top_k),
            )
        except Exception:
            # 某个 source 查询失败时，不影响其他 source。
            continue

        best_distance_by_chunk: Dict[str, float] = {}
        first_rank_by_chunk: Dict[str, int] = {}
        for rank, hit in enumerate(hits, start=1):
            meta = hit.get("metadata", {}) or {}
            chunk_id = str(meta.get("chunk_id", "")).strip()
            if not chunk_id:
                continue

            # `setdefault(key, value)`：
            # 如果 key 不存在，就设置为 value；
            # 如果已经存在，就保持原值不变。
            # 这里等于“只记录这个 chunk 第一次出现时的排名”。
            first_rank_by_chunk.setdefault(chunk_id, rank)
            dist = float(hit.get("distance") or 0.0)
            if chunk_id not in best_distance_by_chunk or dist < best_distance_by_chunk[chunk_id]:
                best_distance_by_chunk[chunk_id] = dist

        for chunk_id, rank in first_rank_by_chunk.items():
            candidate_hits.append(
                {
                    "source": source,
                    "chunk_id": chunk_id,
                    "rank": rank,
                    "distance": best_distance_by_chunk.get(chunk_id, float("inf")),
                }
            )

    candidate_hits.sort(
        # Python 排序支持“按元组逐项比较”，所以这里会依次按：
        # 1. distance（越小越好）
        # 2. rank（越靠前越好）
        # 3. source_job_id（用于稳定排序）
        # 4. chunk 编号（用于稳定排序）
        key=lambda item: (
            float(item.get("distance", float("inf"))),
            int(item.get("rank", 999999)),
            str((item.get("source", {}) or {}).get("source_job_id", "")),
            VectorStore._chunk_num(str(item.get("chunk_id", ""))),
        )
    )

    selected_hits: List[Dict[str, Any]] = []
    selected_keys = set()
    for hit in candidate_hits:
        source = hit.get("source", {}) or {}
        # 这里用 `(source_job_id, chunk_id)` 作为联合主键，避免不同视频的同名 chunk 混淆。
        key = (
            str(source.get("source_job_id", "")).strip(),
            str(hit.get("chunk_id", "")).strip(),
        )
        if not key[0] or not key[1] or key in selected_keys:
            continue

        selected_keys.add(key)
        selected_hits.append(hit)
        if len(selected_hits) >= effective_top_k:
            break

    if effective_neighbor_k > 0:
        extra_hits: List[Dict[str, Any]] = []
        # 对每个已选中的命中，再在它所属的 store 里补相邻 chunk。
        for hit in list(selected_hits):
            source = hit.get("source", {}) or {}
            store = source.get("store")
            if not isinstance(store, VectorStore) or not store.chunk_ids_sorted:
                continue

            source_job_id = str(source.get("source_job_id", "")).strip()
            chunk_id = str(hit.get("chunk_id", "")).strip()
            pos = store.chunk_id_to_pos.get(chunk_id)
            if pos is None:
                continue

            left = max(0, pos - effective_neighbor_k)
            right = min(len(store.chunk_ids_sorted) - 1, pos + effective_neighbor_k)
            for idx in range(left, right + 1):
                neighbor_chunk_id = store.chunk_ids_sorted[idx]
                key = (source_job_id, neighbor_chunk_id)
                if key in selected_keys:
                    continue
                selected_keys.add(key)
                extra_hits.append(
                    {
                        "source": source,
                        "chunk_id": neighbor_chunk_id,
                        "rank": hit.get("rank", 999999),
                        "distance": hit.get("distance", float("inf")),
                    }
                )
        selected_hits.extend(extra_hits)

    evidence_rows: List[Dict[str, str]] = []
    max_total_rows = max(effective_top_k, len(selected_hits)) * effective_rows_per_chunk

    for hit in selected_hits:
        source = hit.get("source", {}) or {}
        store = source.get("store")
        if not isinstance(store, VectorStore):
            continue

        chunk_id = str(hit.get("chunk_id", "")).strip()
        chunk_rows = store.fetch_chunk_rows(chunk_id)
        chunk_rows.sort(
            key=lambda row: (
                0 if str((row.get("metadata", {}) or {}).get("modality", "")) == "chunk_card" else 1,
                str((row.get("metadata", {}) or {}).get("modality", "")),
            )
        )

        added_modalities = set()
        for row in chunk_rows:
            meta = row.get("metadata", {}) or {}
            modality = str(meta.get("modality", "")).strip() or "unknown"
            if modality in added_modalities:
                continue

            evidence_text = compact_text(
                str(row.get("document", "")).strip(),
                260 if modality == "chunk_card" else 380,
            )
            if not evidence_text:
                continue

            evidence_row = {
                "chunk_id": chunk_id,
                "t_start": str(meta.get("t_start", "")).strip(),
                "t_end": str(meta.get("t_end", "")).strip(),
                "section_id": str(meta.get("section_id", "")).strip(),
                "modality": modality,
                "distance": f"{float(hit.get('distance', 0.0)):.4f}",
                "evidence": evidence_text,
                "source_job_id": str(source.get("source_job_id", "")).strip(),
                "source_video_name": str(source.get("source_video_name", "")).strip(),
                "source_video_path": str(source.get("source_video_path", "")).strip(),
            }
            evidence_rows.append(evidence_row)
            added_modalities.add(modality)

            if len(added_modalities) >= effective_rows_per_chunk or len(evidence_rows) >= max_total_rows:
                break

        if len(evidence_rows) >= max_total_rows:
            break

    if not evidence_rows:
        return "", []

    lines = ["## Question Evidence"]
    for row in evidence_rows:
        lines.append("- " + json.dumps(row, ensure_ascii=False))
    return "\n".join(lines), evidence_rows
