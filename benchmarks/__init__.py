"""Benchmarks: defines benchmark configurations and runners."""

from .swebench import SWEbenchBenchmark
from .swebench_lite import SWEBenchLite

__all__ = ["SWEbenchBenchmark", "SWEBenchLite"]
