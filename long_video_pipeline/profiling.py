from __future__ import annotations

"""运行过程耗时统计工具。

这个文件的目标很简单：把程序里每个阶段的耗时记下来，方便后面分析“时间都花到哪里了”。

它会在指定目录下维护两个文件：
1. `timing_events.jsonl`
   每一行都是一个 JSON 对象，表示一次具体事件。
   `jsonl` = JSON Lines，适合“不断追加写入”的场景。
2. `timing_summary.json`
   把所有事件做一个汇总，方便直接查看总耗时、平均耗时、最慢步骤等。

典型用法：

```python
profiler = RunProfiler(output_dir)

with profiler.stage("prepare.asr", category="prepare"):
    do_something()
```

上面的 `with ...:` 代码块执行完后，无论成功还是报错，都会自动记录一条耗时事件。
"""

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterator, Optional


def _utc_now() -> str:
    """返回当前 UTC 时间，格式是 ISO 字符串。

    例子：`2026-04-04T08:30:12+00:00`

    这里故意统一用 UTC，是为了避免不同时区下的时间难以比较。
    """

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunProfiler:
    """记录事件耗时，并定期生成汇总结果。

    这个类可以理解成一个“小型计时器 + 统计器”：
    - `stage(...)` 适合包住一段代码，自动测量执行时长
    - `record(...)` 适合手动写入一条事件
    - `flush_summary()` 用于强制把汇总文件刷新到磁盘
    """

    # 每累计多少条新事件，就尝试写一次 summary。
    SUMMARY_WRITE_EVERY = 24
    # 即使新事件不够 24 条，只要距离上次写 summary 超过 2 秒，也会尝试写。
    SUMMARY_WRITE_INTERVAL_SEC = 2.0

    def __init__(self, root_dir: Path) -> None:
        """初始化 profiler，并尝试读取历史事件。

        参数：
        - `root_dir`：保存统计文件的目录
        """

        # `Path(root_dir)` 可以把传入值统一转换成 `pathlib.Path` 对象，
        # 后面做路径拼接会更直观，例如 `self.root_dir / "xxx.json"`。
        self.root_dir = Path(root_dir)
        # `parents=True` 表示父目录不存在时一起创建；
        # `exist_ok=True` 表示目录已存在也不报错。
        self.root_dir.mkdir(parents=True, exist_ok=True)

        # 事件明细文件：一行一个 JSON。
        self.events_path = self.root_dir / "timing_events.jsonl"
        # 汇总文件：包含总计、平均值、最慢事件等。
        self.summary_path = self.root_dir / "timing_summary.json"

        # `Lock()` 是线程锁。
        # 如果多个线程同时写事件，不加锁就可能出现数据竞争或文件写乱。
        self._lock = Lock()

        # `_events` 保存在内存中的所有事件。
        # 类型标注 `list[Dict[str, Any]]` 的意思是：
        # - 外层是列表 `list`
        # - 列表里的每一项是字典 `Dict[str, Any]`
        # - 键通常是字符串，值可以是任意类型
        self._events: list[Dict[str, Any]] = []

        # `_dirty=True` 表示“内存里的统计数据变了，还没同步到 summary 文件里”。
        self._dirty = False
        # 距离上次写 summary 之后，又新增了多少条事件。
        self._pending_events_since_summary = 0
        # `perf_counter()` 适合做性能计时，精度高，而且不受系统时间调整影响。
        self._last_summary_write_perf = time.perf_counter()

        # 如果历史事件文件已经存在，启动时先读回来。
        # 这样即使程序中途重启，也不会丢掉之前已经记录的耗时。
        if self.events_path.exists():
            try:
                with self.events_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        # `strip()` 会去掉行首尾空白和换行符。
                        line = line.strip()
                        if line:
                            # `json.loads(...)` 会把 JSON 字符串转成 Python 对象。
                            self._events.append(json.loads(line))
            except Exception:
                # 这里选择“读失败就清空”，是一种容错策略：
                # 宁可从空数据重新开始，也不要让初始化直接崩掉。
                self._events = []

        # 启动时先写一份 summary，让目录里的汇总文件始终存在且与历史事件对齐。
        self._write_summary()
        self._dirty = False
        self._pending_events_since_summary = 0
        self._last_summary_write_perf = time.perf_counter()

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        category: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Iterator[None]:
        """把一段代码包成“自动计时”的上下文管理器。

        用法：

        ```python
        with profiler.stage("stage.prepare", category="stage"):
            ...
        ```

        语法说明：
        - `@contextmanager` 会把这个函数变成可用于 `with` 的对象。
        - 函数中间的 `yield`，可以理解成“把控制权暂时交给 with 代码块”。
        - `*` 的意思是：`category` 和 `meta` 必须写成关键字参数，
          也就是要写成 `category="stage"`，不能只靠位置传参。
        - 返回类型 `Iterator[None]` 是因为这个函数内部用了 `yield`，
          但并不会向 `with ... as x` 提供额外值，所以是 `None`。
        """

        # 记录“真实世界时间”，适合写日志、看开始/结束时刻。
        started_at = _utc_now()
        # 记录“高精度计时起点”，适合拿来计算耗时。
        started_perf = time.perf_counter()
        status = "ok"
        err_msg = ""

        try:
            # `yield` 这一句执行时，程序会进入 `with profiler.stage(...):`
            # 的代码块内部，真正去运行用户想计时的逻辑。
            yield
        except Exception as exc:
            # 如果 `with` 代码块里抛出异常，就把状态记成 error。
            status = "error"
            err_msg = str(exc)
            # `raise` 表示把原异常继续抛出去，不吞掉错误。
            # 也就是说：这个 profiler 负责记录错误，但不会偷偷忽略错误。
            raise
        finally:
            # `finally` 的特点是：无论成功还是异常，都会执行。
            # 所以很适合放“收尾逻辑”，例如记录耗时。
            self.record(
                name=name,
                category=category,
                duration_sec=time.perf_counter() - started_perf,
                status=status,
                meta=meta,
                started_at=started_at,
                ended_at=_utc_now(),
                error=err_msg,
            )

    def record(
        self,
        *,
        name: str,
        category: str,
        duration_sec: float,
        status: str = "ok",
        meta: Optional[Dict[str, Any]] = None,
        started_at: Optional[str] = None,
        ended_at: Optional[str] = None,
        error: str = "",
    ) -> None:
        """手动写入一条事件记录。

        和 `stage(...)` 的区别：
        - `stage(...)`：自动计时，适合包住一整段代码
        - `record(...)`：手动传入耗时，适合你已经自己算好了时间的场景
        """

        # 这里构造一条标准化事件。
        # 注意：这只是一个普通 Python 字典，后面会被写成 JSON。
        event = {
            "name": str(name),
            "category": str(category),
            "status": str(status),
            # `max(0.0, ...)` 是为了避免出现负数耗时。
            # `round(..., 6)` 保留 6 位小数，兼顾精度和可读性。
            "duration_sec": round(max(0.0, float(duration_sec)), 6),
            # `a or b` 的意思是：如果 `a` 有值就用 `a`，否则用 `b`。
            "started_at": started_at or _utc_now(),
            "ended_at": ended_at or _utc_now(),
            # `meta` 可以放一些附加信息，例如 chunk 数量、video_id 等。
            "meta": meta or {},
        }
        if error:
            # 只有在确实有错误信息时，才额外写入 `error` 字段。
            event["error"] = error

        # 这里上锁，保证“改内存 + 写文件 + 刷 summary 状态”是一组原子操作。
        with self._lock:
            self._events.append(event)

            # 以追加模式 `"a"` 打开文件，把这条事件追加到 jsonl 末尾。
            with self.events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

            self._dirty = True
            self._pending_events_since_summary += 1
            self._maybe_write_summary_locked(force=False)

    def flush_summary(self) -> None:
        """强制刷新 summary 文件。

        有些时候程序马上就要退出，这时不想等“定期刷新”触发，
        就可以主动调用这个方法。
        """

        with self._lock:
            self._maybe_write_summary_locked(force=True)

    def _maybe_write_summary_locked(self, *, force: bool) -> None:
        """在“已经持有锁”的前提下，判断是否要写 summary。

        方法名里的 `_locked` 是一种命名提醒：
        调用这个方法之前，外部应该已经进入 `with self._lock:`。
        这不是 Python 强制规则，但属于常见工程约定。
        """

        if not self._dirty:
            # 没有新变化，就没必要重写 summary。
            return

        now = time.perf_counter()

        # 满足任一条件就写：
        # 1. 外部强制要求写
        # 2. 新事件条数达到阈值
        should_write = force or self._pending_events_since_summary >= self.SUMMARY_WRITE_EVERY

        # 如果上面还不满足，再看是否超过时间阈值。
        if not should_write and (now - self._last_summary_write_perf) >= self.SUMMARY_WRITE_INTERVAL_SEC:
            should_write = True
        if not should_write:
            return

        self._write_summary()
        self._dirty = False
        self._pending_events_since_summary = 0
        self._last_summary_write_perf = now

    def _write_summary(self) -> None:
        """把当前所有事件聚合后，写入 summary 文件。"""

        # `aggregates` 按事件名字聚合，例如：
        # "prepare.asr" -> 这个步骤总共跑了几次、总耗时多少、最长一次多长等。
        aggregates: Dict[str, Dict[str, Any]] = {}
        # `category_totals` 按类别累计总耗时，例如：
        # "prepare" -> 总共花了多少秒。
        category_totals: Dict[str, float] = {}

        for event in self._events:
            # `dict.get("key", default)` 的意思是：
            # 如果键存在就取值，否则返回默认值。
            name = str(event.get("name", ""))
            category = str(event.get("category", ""))
            duration = float(event.get("duration_sec", 0.0))
            status = str(event.get("status", ""))

            # `setdefault(key, default)` 的意思是：
            # - 如果 `key` 已经存在，就返回原值
            # - 如果不存在，就先放入 `default`，再返回它
            #
            # 所以这里的效果是：第一次见到某个 `name` 时，
            # 自动创建一份统计骨架；后面再见到同名事件时，复用这份骨架继续累加。
            agg = aggregates.setdefault(
                name,
                {
                    "name": name,
                    "category": category,
                    "count": 0,
                    "error_count": 0,
                    "total_duration_sec": 0.0,
                    "avg_duration_sec": 0.0,
                    "max_duration_sec": 0.0,
                },
            )

            # 下面这些都是“累计统计”。
            agg["count"] += 1
            agg["total_duration_sec"] += duration
            # 取当前最大耗时。
            agg["max_duration_sec"] = max(float(agg["max_duration_sec"]), duration)
            if status != "ok":
                agg["error_count"] += 1

            # 这里按 category 累加总耗时。
            category_totals[category] = category_totals.get(category, 0.0) + duration

        # 第二轮循环，把聚合结果中的数字整理成最终展示格式。
        for agg in aggregates.values():
            # `max(1, ...)` 是为了避免除以 0。
            count = max(1, int(agg["count"]))
            agg["total_duration_sec"] = round(float(agg["total_duration_sec"]), 6)
            agg["avg_duration_sec"] = round(float(agg["total_duration_sec"]) / count, 6)
            agg["max_duration_sec"] = round(float(agg["max_duration_sec"]), 6)

        # 最终 summary 也是一个普通字典，后面会整体写入 JSON 文件。
        summary = {
            "updated_at": _utc_now(),
            "event_count": len(self._events),
            "events_path": str(self.events_path),

            # `sorted(..., key=...)` 表示“按指定规则排序”。
            # 这里的 key 是 `(-item[1], item[0])`，意思是：
            # 1. 先按耗时倒序（因为前面加了负号）
            # 2. 如果耗时一样，再按分类名字正序
            "totals_by_category_sec": {
                k: round(v, 6) for k, v in sorted(category_totals.items(), key=lambda item: (-item[1], item[0]))
            },

            # 这里把每个 name 的聚合结果按总耗时从大到小排序。
            "totals_by_name": sorted(
                aggregates.values(),
                key=lambda item: (-float(item["total_duration_sec"]), item["name"]),
            ),

            # 找出最慢的 50 条原始事件。
            # `[:50]` 是 Python 切片语法，表示“只取前 50 个元素”。
            "slowest_events": sorted(
                self._events,
                key=lambda item: (-float(item.get("duration_sec", 0.0)), str(item.get("name", ""))),
            )[:50],
        }

        # `indent=2` 让 JSON 更易读；`ensure_ascii=False` 允许直接写中文。
        self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
