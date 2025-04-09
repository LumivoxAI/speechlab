import json
from time import perf_counter_ns
from typing import Any, Self, Callable
from pathlib import Path

import numpy as np


class STTTestResult:
    def __init__(
        self,
        name: str,
        data: list[tuple[str, str, int]],
        gpu_bytes: int | None = None,
    ) -> None:
        self.name = name
        self.data = data
        self.gpu_bytes = gpu_bytes

    def compare(self, other: Self) -> dict[str, tuple[str, str]]:
        results = {}
        for i, v1 in enumerate(self.data):
            v2 = other.data[i]
            if v1[0] != v2[0]:
                raise ValueError(f"File names do not match: {v1[0]} != {v2[0]}")
            if v1[1] != v2[1]:
                results[v1[0]] = (v1[1], v2[1])

        return results

    def print_stats(self) -> None:
        print(f"{self.name}:")
        dts = np.array([dt for _, _, dt in self.data])
        print(f"Mean: {dts.mean() / 1e6:.2f} ms")
        print(f"Median: {np.median(dts) / 1e6:.2f} ms")
        print(f"Max: {dts.max() / 1e6:.2f} ms")
        print(f"Min: {dts.min() / 1e6:.2f} ms")
        print(f"Std: {dts.std() / 1e6:.2f} ms")
        print(f"Total: {dts.sum() / 1e9:.2f} s")
        print(f"Count: {len(dts)}")
        print(f"Throughput: {len(dts) / (dts.sum() / 1e9):.2f} qps")
        if self.gpu_bytes is not None:
            print(f"Peak usage: {self.gpu_bytes / 1024**2:.2f} MB")


class STTTestUtils:
    def __init__(self, result_dir: Path) -> None:
        result_dir.mkdir(exist_ok=True, parents=True)
        self._result_dir = result_dir

    def test_batch(
        self,
        name: str,
        audio_data: list[tuple[str, Any]],
        stt_func: Callable,
    ) -> STTTestResult:
        results = []
        for filename, data in audio_data:
            start = perf_counter_ns()
            text = stt_func(data)
            dt = perf_counter_ns() - start
            results.append((filename, text, dt))

        return STTTestResult(
            name=name,
            data=results,
        )

    def save(self, result: STTTestResult) -> None:
        write_data = {
            "data": result.data,
            "gpu_bytes": result.gpu_bytes,
        }

        with open(self._result_dir / f"{result.name}.json", "w") as f:
            json.dump(write_data, f, ensure_ascii=False, indent=2, sort_keys=True)

    def load(self, name: str) -> STTTestResult:
        with open(self._result_dir / f"{name}.json", "r") as f:
            data = json.load(f)

        return STTTestResult(
            name=name,
            data=data["data"],
            gpu_bytes=data["gpu_bytes"],
        )

    def compare(self, name1: str, name2: str) -> dict[str, tuple[str, str]]:
        result1 = self.load(name1)
        result2 = self.load(name2)

        return result1.compare(result2)

    def print_stats(self, names: list[str]) -> None:
        for name in names:
            result = self.load(name)
            result.print_stats()
            print()
