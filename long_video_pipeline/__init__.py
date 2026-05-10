from importlib import import_module
from typing import Any

__all__ = ["LongVideoSummaryPipeline", "PipelineConfig"]


def __getattr__(name: str) -> Any:
    if name == "LongVideoSummaryPipeline":
        return import_module(".pipeline", __name__).LongVideoSummaryPipeline
    if name == "PipelineConfig":
        return import_module(".settings", __name__).PipelineConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
