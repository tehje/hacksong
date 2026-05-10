"""和视频切片、抽帧相关的工具函数。

这份文件主要做 4 件事：
1. 通过 `ffprobe` 获取视频时长。
2. 通过 `ffmpeg` 按时间范围裁剪视频片段。
3. 按规则给长视频切分 chunk（小片段）。
4. 在指定时间点抽取关键帧图片。

给新手的阅读提示：
- 以 `_` 开头的函数，通常表示“模块内部使用的辅助函数”。
- `List[str]`、`Tuple[List[str], List[float]]` 这类写法叫“类型标注”，
  用来说明参数和返回值的大致类型，方便阅读和静态检查。
- `Path` 是 `pathlib` 里的路径对象，比直接拼字符串路径更安全、清晰。
"""

from __future__ import annotations

# 这行的作用是：让类型标注延后解析。
# 对新手来说，可以简单理解为“让 `List[ChunkMeta]` 这类标注更稳妥一些”。

import json
import subprocess
from pathlib import Path
from typing import List, Sequence, Tuple

from .schemas import ChunkMeta


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    """运行一条子进程命令，并在失败时抛出更好读的错误信息。

    `cmd` 是一个字符串列表，例如：
    `["ffmpeg", "-y", "-i", "a.mp4", "out.mp4"]`

    这里不用一整条字符串，而是用列表，是因为 `subprocess.run(...)`
    更容易正确处理带空格的参数，也更安全。
    """
    try:
        # `capture_output=True` 表示捕获标准输出和标准错误输出。
        # `text=True` 表示把输出按文本字符串处理，而不是 bytes。
        # `check=True` 表示只要返回码不是 0，就自动抛出异常。
        return subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        # `or ""` 的意思是：如果 e.stderr / e.stdout 是 None，就改用空字符串。
        # `.strip()` 会去掉两端空白字符，方便日志更紧凑。
        stderr = (e.stderr or "").strip()
        stdout = (e.stdout or "").strip()
        # `" ".join(cmd)` 会把列表里的命令参数拼成一行，便于输出。
        cmd_text = " ".join(cmd)

        # 如果日志太长，就截断，避免异常信息刷屏。
        if len(stderr) > 1200:
            stderr = stderr[:1200] + "..."
        if len(stdout) > 400:
            stdout = stdout[:400] + "..."

        # `raise ... from e` 表示“保留原始异常链”，调试时更有帮助。
        raise RuntimeError(
            f"subprocess failed: {cmd_text}\n"
            f"returncode={e.returncode}\n"
            f"stderr={stderr}\n"
            f"stdout={stdout}"
        ) from e


def _safe_seek_timestamp(ts: float, chunk_start: float, chunk_end: float, end_margin_sec: float = 0.5) -> float:
    """把抽帧时间限制在 chunk 的合法范围内。

    这样做是为了避免抽帧时间跑到片段外面，或者太贴近结尾导致 ffmpeg 抽帧失败。
    """
    start = float(chunk_start)
    end = float(chunk_end)

    # 如果收到异常区间（结束时间不大于开始时间），兜底给 1 秒长度。
    if end <= start:
        end = start + 1.0

    # 给结尾留一点“安全边距”，避免过于接近尾帧。
    safe_max = end - max(0.0, float(end_margin_sec))
    if safe_max < start:
        safe_max = start

    # `max(start, min(ts, safe_max))` 是经典的“夹紧(clamp)”写法：
    # 小于 start 就取 start，大于 safe_max 就取 safe_max。
    return max(start, min(float(ts), safe_max))


def _extract_keyframe_once(video_path: str, ts: float, out_path: Path, max_width: int = 0) -> None:
    """在单个时间点尝试抽取 1 张关键帧。"""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{ts:.3f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-q:v",
        "2",
    ]

    # 如果传入了最大宽度，就给 ffmpeg 加缩放滤镜。
    # `cmd.extend([...])` 的作用是一次性往列表末尾追加多个元素。
    if int(max_width) > 0:
        # `scale=min(max_width, iw):-2` 的意思大致是：
        # 宽度不超过 max_width，高度按比例自动缩放。
        # 这里 `\\,` 是为了把逗号正确传给 ffmpeg 过滤器。
        cmd.extend(["-vf", f"scale=min({int(max_width)}\\,iw):-2"])
    cmd.append(str(out_path))
    _run(cmd)

    # ffmpeg 即使执行过，也可能没真正产出有效文件，所以这里再做一次兜底检查。
    if (not out_path.exists()) or out_path.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg produced empty keyframe: path={out_path}, ts={ts:.3f}")


def _extract_keyframe_with_fallback(
    video_path: str,
    chunk: ChunkMeta,
    base_ts: float,
    out_path: Path,
    max_width: int = 0,
) -> float:
    """抽帧失败时，自动尝试几个更早的时间点作为回退方案。

    返回值是“最终实际成功抽帧的时间点”。
    """
    # 先试原始时间点，再逐步往前偏移。
    fallback_offsets = [0.0, 0.3, 0.7, 1.2, 2.0, 3.0]
    # 用来记录每次失败的摘要，便于最后拼成一条更清楚的报错。
    tried_msgs: List[str] = []
    # 有些回退时间经过 `_safe_seek_timestamp(...)` 夹紧后会变成同一个值，
    # 所以这里用集合去重，避免重复尝试。
    visited = set()

    for off in fallback_offsets:
        ts = _safe_seek_timestamp(
            ts=float(base_ts) - float(off),
            chunk_start=chunk.t_start,
            chunk_end=chunk.t_end,
        )
        # 保留 3 位小数，既够用，也能减少浮点数精度噪声。
        key = round(ts, 3)
        if key in visited:
            continue
        visited.add(key)

        try:
            _extract_keyframe_once(video_path=video_path, ts=ts, out_path=out_path, max_width=max_width)
            return float(key)
        except Exception as e:
            # 把异常压成单行短文本，方便最后汇总。
            msg = str(e).replace("\n", " ").strip()
            if len(msg) > 220:
                msg = msg[:220] + "..."
            tried_msgs.append(f"{ts:.3f}s => {msg}")

    raise RuntimeError(
        f"extract keyframe failed for {chunk.chunk_id}, base_ts={base_ts:.3f}. "
        f"tried: {' | '.join(tried_msgs)}"
    )


def probe_video_duration(video_path: str) -> float:
    """用 ffprobe 读取视频总时长（秒）。"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        video_path,
    ]
    p = _run(cmd)
    # ffprobe 上面要求输出 json，所以这里直接把文本解析成字典。
    data = json.loads(p.stdout)
    return float(data["format"]["duration"])


def extract_video_clip(
    video_path: str,
    t_start: float,
    t_end: float,
    out_path: Path,
) -> Path:
    """按起止时间裁出一个视频片段，并写到 `out_path`。"""
    # 起始时间不能小于 0。
    start = max(0.0, float(t_start))
    # 结束时间至少要比开始时间大 0.5 秒，避免得到一个几乎空的片段。
    end = max(start + 0.5, float(t_end))
    duration = max(0.5, end - start)

    # `parents=True` 表示需要时连父目录一起创建。
    # `exist_ok=True` 表示目录已存在也不要报错。
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        video_path,
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    _run(cmd)

    # 再次确认输出文件确实存在且非空。
    if (not out_path.exists()) or out_path.stat().st_size <= 0:
        raise RuntimeError(
            f"ffmpeg produced empty clip: path={out_path}, "
            f"start={start:.3f}, end={end:.3f}"
        )
    return out_path


def make_chunks(
    video_path: str,
    chunk_seconds: int,
    overlap_seconds: int,
    section_minutes: int,
) -> List[ChunkMeta]:
    """按固定时长切分视频，生成一组 `ChunkMeta`。

    这是一个“朴素兜底方案”：
    不做复杂语义分段，只按固定秒数切。
    """
    duration = probe_video_duration(video_path)
    chunks: List[ChunkMeta] = []

    # 如果 chunk=30 秒、overlap=5 秒，那么 step=25 秒。
    # 也就是下一个 chunk 从 25 秒处开始，从而和上一个 chunk 重叠 5 秒。
    step = max(1, chunk_seconds - overlap_seconds)
    idx = 0
    t = 0.0
    # 每个 section 至少按 60 秒计算。
    section_len = max(60, section_minutes * 60)

    while t < duration:
        t_start = max(0.0, t)
        t_end = min(duration, t_start + chunk_seconds)
        # `//` 是“整除”，这里用来判断当前 chunk 落在哪个 section。
        section_idx = int(t_start // section_len)
        chunk_id = f"chunk_{idx:05d}"
        section_id = f"section_{section_idx:03d}"
        chunks.append(
            ChunkMeta(
                chunk_id=chunk_id,
                index=idx,
                t_start=t_start,
                t_end=t_end,
                section_id=section_id,
            )
        )
        idx += 1
        t += step

    return chunks


def normalize_keyframe_timestamps(
    chunk_start: float,
    chunk_end: float,
    candidate_timestamps: Sequence[float],
    desired_count: int,
) -> List[float]:
    """把候选抽帧时间整理成“合法、去重、数量合适”的时间点列表。

    处理步骤：
    1. 过滤掉超出 chunk 范围的时间点。
    2. 去重并排序。
    3. 如果太多，就尽量均匀地挑出 `desired_count` 个。
    4. 如果太少，就用均匀分布的时间点补齐。
    """
    # 最少也要保留 1 张关键帧。
    desired = max(1, int(desired_count))
    start = float(chunk_start)
    end = float(chunk_end)
    if end <= start:
        end = start + 1.0

    filtered: List[float] = []
    seen = set()
    for ts in candidate_timestamps:
        try:
            # 如果传进来的时间点是字符串之类的，也尽量转成 float。
            t = float(ts)
        except Exception:
            continue
        if t < start or t > end:
            continue
        key = round(t, 3)
        if key in seen:
            continue
        seen.add(key)
        filtered.append(key)

    filtered.sort()

    selected: List[float] = []
    if filtered:
        if len(filtered) <= desired:
            selected = list(filtered)
        elif desired == 1:
            # 只需要 1 张时，直接取中间位置附近的时间点。
            selected = [filtered[len(filtered) // 2]]
        else:
            n = len(filtered)
            # 这里的思路是：按“分位点”去选，尽量选得均匀一些，
            # 同时避免总是拿到第一个/最后一个时间点。
            for i in range(desired):
                # `i` 从 0 到 desired-1。
                # 这个公式会把这些位置映射到 filtered 列表内部较均匀的索引上。
                pos = (i + 0.5) * n / desired - 0.5
                idx = int(round(pos))
                # 再做一次边界保护，避免索引越界。
                idx = max(0, min(n - 1, idx))
                selected.append(filtered[idx])
            # 用集合再次去重，再排序。
            selected = sorted(list({round(x, 3) for x in selected}))

    if len(selected) < desired:
        duration = max(1.0, end - start)
        # 列表推导式：
        # 会在区间内部按均匀间隔生成 desired 个时间点。
        uniform = [round(start + duration * ((i + 1) / (desired + 1)), 3) for i in range(desired)]
        picked = list(selected)
        seen2 = {round(x, 3) for x in picked}
        for t in uniform:
            if len(picked) >= desired:
                break
            if round(t, 3) in seen2:
                continue
            seen2.add(round(t, 3))
            picked.append(t)
        selected = sorted(picked)

    if len(selected) > desired:
        selected = selected[:desired]

    # 最后统一转成 float，保证返回值类型稳定。
    return [float(x) for x in selected]


def extract_keyframes_at_timestamps(
    video_path: str,
    chunk: ChunkMeta,
    timestamps: Sequence[float],
    out_dir: Path,
    keyframes_per_chunk: int,
    keyframe_max_width: int = 0,
) -> Tuple[List[str], List[float]]:
    """按给定时间点抽取关键帧。

    返回两个列表：
    1. 关键帧图片路径列表
    2. 实际成功抽帧的时间点列表
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_ts = normalize_keyframe_timestamps(
        chunk_start=chunk.t_start,
        chunk_end=chunk.t_end,
        candidate_timestamps=list(timestamps),
        desired_count=keyframes_per_chunk,
    )

    paths: List[str] = []
    actual_ts: List[float] = []
    # `enumerate(..., start=1)` 表示循环时同时拿到“序号”和“元素”，
    # 并且序号从 1 开始，而不是默认的 0。
    for i, ts in enumerate(selected_ts, start=1):
        out_path = out_dir / f"{chunk.chunk_id}_kf{i}.jpg"
        used_ts = _extract_keyframe_with_fallback(
            video_path=video_path,
            chunk=chunk,
            base_ts=ts,
            out_path=out_path,
            max_width=keyframe_max_width,
        )
        paths.append(str(out_path))
        actual_ts.append(used_ts)

    return paths, actual_ts


def extract_keyframes_for_chunk(
    video_path: str,
    chunk: ChunkMeta,
    keyframes_per_chunk: int,
    out_dir: Path,
    keyframe_max_width: int = 0,
) -> List[str]:
    """对单个 chunk 做均匀抽帧。

    这是兜底逻辑：如果没有更复杂的时间点策略，
    就在 chunk 内均匀取若干个点来抽帧。
    """
    duration = max(1.0, chunk.t_end - chunk.t_start)
    # 列表推导式会生成多个基础时间点。
    # 例如需要 3 张图时，大致会取 1/4、2/4、3/4 处。
    base_ts = [
        chunk.t_start + duration * ((i + 1) / (max(1, keyframes_per_chunk) + 1))
        for i in range(max(1, keyframes_per_chunk))
    ]
    paths, selected_ts = extract_keyframes_at_timestamps(
        video_path=video_path,
        chunk=chunk,
        timestamps=base_ts,
        out_dir=out_dir,
        keyframes_per_chunk=keyframes_per_chunk,
        keyframe_max_width=keyframe_max_width,
    )
    # 把最终采用的时间点回写到 chunk 对象里，方便后续流程继续使用。
    chunk.keyframe_timestamps = selected_ts
    return paths


def split_sections(chunks: List[ChunkMeta]) -> List[List[ChunkMeta]]:
    """按 `section_id` 把 chunk 列表分组。"""
    by_section = {}
    for c in chunks:
        # `setdefault(key, default)` 的意思是：
        # 如果 key 不存在，就先放一个默认值；然后返回这个值。
        # 所以这里等价于“先确保该 section 对应一个列表，再 append”。
        by_section.setdefault(c.section_id, []).append(c)
    keys = sorted(by_section.keys())
    # 最后按 section_id 排序后返回二维列表。
    return [by_section[k] for k in keys]
