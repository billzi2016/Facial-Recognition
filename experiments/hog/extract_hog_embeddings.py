#!/usr/bin/env python3
"""使用 HOG/dlib 路线提取 128 维人脸 embedding。

脚本意图：
- 作为传统 CPU 对照组，使用 `face_recognition` 的 HOG 检测和 dlib 128 维编码。
- 读取 `data/manifests/images.csv`，默认处理全量有效图片。
- 使用 `CPU 总核心数 - 2` 做多进程并行，固定给系统和桌面环境留出两个核心。
- 所有长任务使用 `tqdm` 展示进度、速度和粗略 ETA。
- embedding 必须使用 `h5py` 存储为 HDF5，并在写入 dataset 时直接启用
  `compression="gzip", compression_opts=1`，不允许先写未压缩 H5 再外部 gzip。
- metadata、detections、failures 和 benchmark 单独写盘，方便后续 FAISS 和实验报告复用。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm


EMBEDDING_DIM = 128
DEFAULT_MANIFEST_PATH = Path("data/manifests/images.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/hog")


@dataclass(frozen=True)
class ManifestRecord:
    """输入 manifest 中一张图片的最小必要信息。"""

    row_index: int
    image_id: str
    path: Path
    person_id: str
    split: str


@dataclass(frozen=True)
class HogResult:
    """单张图片的 HOG 处理结果。

    embedding 为 `None` 表示该图片处理失败，失败原因写入 `failure_reason`。
    多进程 worker 返回 dataclass 而不是直接写文件，是为了让主进程统一排序和落盘，
    确保 HDF5 中的 embedding 行顺序与 metadata CSV 一一对应。
    """

    row_index: int
    image_id: str
    path: str
    person_id: str
    split: str
    face_count: int
    selected_box_top: int | None
    selected_box_right: int | None
    selected_box_bottom: int | None
    selected_box_left: int | None
    detect_latency_ms: float
    encode_latency_ms: float
    embedding: list[float] | None
    failure_reason: str


def parse_bool(value: str) -> bool:
    """解析 manifest 中的布尔字符串。

    CSV 中常见值可能是 `true/false`、`1/0`、空字符串。这里用宽松解析，
    避免因为大小写或空值导致全量脚本中断。
    """

    return value.strip().lower() in {"1", "true", "yes", "y"}


def parse_split_filter(raw: str) -> set[str] | None:
    """解析 `--splits` 参数。

    返回 `None` 表示不过滤 split；否则只保留用户指定的 split 名称。
    """

    if raw.strip().lower() in {"", "all", "*"}:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def read_manifest(manifest_path: Path, split_filter: set[str] | None, limit: int | None) -> list[ManifestRecord]:
    """读取图片 manifest，并按 split、excluded 和 limit 过滤。

    这里不检查图片是否真的能解码，因为 HOG worker 会实际读取图片并记录失败。
    这样主进程读取 manifest 的速度足够快，错误也能落到统一的 failures.csv。
    """

    records: list[ManifestRecord] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for row_index, row in enumerate(tqdm(reader, desc="read manifest", unit="row")):
            split = row.get("split", "")
            if split_filter is not None and split not in split_filter:
                continue
            if parse_bool(row.get("excluded", "false")):
                continue

            image_path = row.get("path", "")
            image_id = row.get("image_id", "")
            if not image_path or not image_id:
                continue

            records.append(
                ManifestRecord(
                    row_index=row_index,
                    image_id=image_id,
                    path=Path(image_path),
                    person_id=row.get("person_id", ""),
                    split=split,
                )
            )

            if limit is not None and len(records) >= limit:
                break
    return records


def largest_face_box(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """从多个 HOG 检测框中选择面积最大的人脸框。

    `face_recognition` 的 box 顺序是 `(top, right, bottom, left)`。
    多脸图片如果不固定选择策略，会导致同一张图在不同运行中可能编码到不同人脸。
    """

    return max(boxes, key=lambda box: max(0, box[2] - box[0]) * max(0, box[1] - box[3]))


def process_one(record: ManifestRecord) -> HogResult:
    """处理单张图片：读取、HOG 检测、选择最大脸、提取 128 维编码。

    该函数会在 worker 进程中执行。`face_recognition` 放在函数内部导入，
    是为了让每个子进程自己初始化 dlib 相关状态，减少主进程被序列化时的额外负担。
    """

    start_detect = time.perf_counter()
    try:
        import face_recognition

        image = face_recognition.load_image_file(record.path)
        boxes = face_recognition.face_locations(image, model="hog")
        detect_latency_ms = (time.perf_counter() - start_detect) * 1000.0

        if not boxes:
            return HogResult(
                row_index=record.row_index,
                image_id=record.image_id,
                path=str(record.path),
                person_id=record.person_id,
                split=record.split,
                face_count=0,
                selected_box_top=None,
                selected_box_right=None,
                selected_box_bottom=None,
                selected_box_left=None,
                detect_latency_ms=detect_latency_ms,
                encode_latency_ms=0.0,
                embedding=None,
                failure_reason="no_face_detected",
            )

        selected_box = largest_face_box(boxes)
        start_encode = time.perf_counter()
        encodings = face_recognition.face_encodings(image, known_face_locations=[selected_box])
        encode_latency_ms = (time.perf_counter() - start_encode) * 1000.0

        if not encodings:
            return HogResult(
                row_index=record.row_index,
                image_id=record.image_id,
                path=str(record.path),
                person_id=record.person_id,
                split=record.split,
                face_count=len(boxes),
                selected_box_top=selected_box[0],
                selected_box_right=selected_box[1],
                selected_box_bottom=selected_box[2],
                selected_box_left=selected_box[3],
                detect_latency_ms=detect_latency_ms,
                encode_latency_ms=encode_latency_ms,
                embedding=None,
                failure_reason="encoding_failed",
            )

        embedding = np.asarray(encodings[0], dtype=np.float32)
        if embedding.shape != (EMBEDDING_DIM,):
            return HogResult(
                row_index=record.row_index,
                image_id=record.image_id,
                path=str(record.path),
                person_id=record.person_id,
                split=record.split,
                face_count=len(boxes),
                selected_box_top=selected_box[0],
                selected_box_right=selected_box[1],
                selected_box_bottom=selected_box[2],
                selected_box_left=selected_box[3],
                detect_latency_ms=detect_latency_ms,
                encode_latency_ms=encode_latency_ms,
                embedding=None,
                failure_reason=f"unexpected_embedding_shape:{embedding.shape}",
            )

        return HogResult(
            row_index=record.row_index,
            image_id=record.image_id,
            path=str(record.path),
            person_id=record.person_id,
            split=record.split,
            face_count=len(boxes),
            selected_box_top=selected_box[0],
            selected_box_right=selected_box[1],
            selected_box_bottom=selected_box[2],
            selected_box_left=selected_box[3],
            detect_latency_ms=detect_latency_ms,
            encode_latency_ms=encode_latency_ms,
            embedding=embedding.tolist(),
            failure_reason="",
        )
    except Exception as exc:  # noqa: BLE001 - 全量实验需要记录坏样本而不是中断整个任务。
        detect_latency_ms = (time.perf_counter() - start_detect) * 1000.0
        return HogResult(
            row_index=record.row_index,
            image_id=record.image_id,
            path=str(record.path),
            person_id=record.person_id,
            split=record.split,
            face_count=0,
            selected_box_top=None,
            selected_box_right=None,
            selected_box_bottom=None,
            selected_box_left=None,
            detect_latency_ms=detect_latency_ms,
            encode_latency_ms=0.0,
            embedding=None,
            failure_reason=f"exception:{type(exc).__name__}:{exc}",
        )


def run_workers(records: list[ManifestRecord], worker_count: int) -> list[HogResult]:
    """使用半数 CPU worker 并行处理图片。

    `as_completed` 可以让 tqdm 按实际完成速度推进，而不是被慢图片阻塞。
    后续写盘前会按 `row_index` 排序，保证结果顺序稳定。
    """

    results: list[HogResult] = []
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(process_one, record) for record in records]
        for future in tqdm(as_completed(futures), total=len(futures), desc="hog encode", unit="image"):
            results.append(future.result())
    return sorted(results, key=lambda result: result.row_index)


def percentile(values: list[float], pct: float) -> float:
    """计算简单百分位数；空列表返回 0，避免 benchmark 写盘失败。"""

    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, round((pct / 100.0) * (len(sorted_values) - 1))))
    return float(sorted_values[index])


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """写 CSV 文件，并用 tqdm 展示写入进度。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in tqdm(rows, desc=f"write {path.name}", unit="row"):
            writer.writerow(row)


def write_outputs(results: list[HogResult], output_dir: Path, benchmark_extra: dict[str, Any]) -> None:
    """写 HDF5 embedding、metadata、detections、failures 和 benchmark。

    embedding 使用 HDF5 的 dataset 内置 gzip 压缩：
    `create_dataset(..., compression="gzip", compression_opts=1)`。
    这和把 `.h5` 文件写完后再外部 gzip 是两回事；后者会让下游无法直接随机读取 HDF5 dataset。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    successes = [result for result in results if result.embedding is not None]
    failures = [result for result in results if result.embedding is None]

    embeddings = np.asarray([result.embedding for result in successes], dtype=np.float32)
    h5_path = output_dir / "embeddings.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("embeddings", data=embeddings, compression="gzip", compression_opts=1)
        h5.attrs["embedding_dim"] = EMBEDDING_DIM
        h5.attrs["model"] = "face_recognition_hog_dlib"
        h5.attrs["compression"] = "gzip"
        h5.attrs["compression_opts"] = 1
        for key, value in benchmark_extra.items():
            h5.attrs[key] = value

    metadata_rows = [
        {
            "embedding_index": index,
            "image_id": result.image_id,
            "path": result.path,
            "person_id": result.person_id,
            "split": result.split,
            "row_index": result.row_index,
        }
        for index, result in enumerate(successes)
    ]
    write_csv(
        output_dir / "embedding_metadata.csv",
        ["embedding_index", "image_id", "path", "person_id", "split", "row_index"],
        metadata_rows,
    )

    detection_rows = [
        {
            "image_id": result.image_id,
            "path": result.path,
            "person_id": result.person_id,
            "split": result.split,
            "face_count": result.face_count,
            "selected_box_top": result.selected_box_top,
            "selected_box_right": result.selected_box_right,
            "selected_box_bottom": result.selected_box_bottom,
            "selected_box_left": result.selected_box_left,
            "detect_latency_ms": f"{result.detect_latency_ms:.4f}",
            "encode_latency_ms": f"{result.encode_latency_ms:.4f}",
            "success": str(result.embedding is not None).lower(),
        }
        for result in results
    ]
    write_csv(
        output_dir / "detections.csv",
        [
            "image_id",
            "path",
            "person_id",
            "split",
            "face_count",
            "selected_box_top",
            "selected_box_right",
            "selected_box_bottom",
            "selected_box_left",
            "detect_latency_ms",
            "encode_latency_ms",
            "success",
        ],
        detection_rows,
    )

    failure_rows = [
        {
            "image_id": result.image_id,
            "path": result.path,
            "person_id": result.person_id,
            "split": result.split,
            "failure_reason": result.failure_reason,
            "detect_latency_ms": f"{result.detect_latency_ms:.4f}",
            "encode_latency_ms": f"{result.encode_latency_ms:.4f}",
        }
        for result in failures
    ]
    write_csv(
        output_dir / "failures.csv",
        ["image_id", "path", "person_id", "split", "failure_reason", "detect_latency_ms", "encode_latency_ms"],
        failure_rows,
    )

    detect_latencies = [result.detect_latency_ms for result in results]
    encode_latencies = [result.encode_latency_ms for result in successes]
    face_counts = [result.face_count for result in results]
    benchmark = {
        **benchmark_extra,
        "input_images": len(results),
        "successful_embeddings": len(successes),
        "failed_images": len(failures),
        "embedding_dim": EMBEDDING_DIM,
        "embedding_file": str(h5_path),
        "h5_dataset": "embeddings",
        "h5_compression": "gzip",
        "h5_compression_opts": 1,
        "faces_detected_rate": len(successes) / len(results) if results else 0.0,
        "no_face_rate": sum(1 for result in results if result.face_count == 0) / len(results) if results else 0.0,
        "multi_face_rate": sum(1 for count in face_counts if count > 1) / len(results) if results else 0.0,
        "detect_latency_ms_p50": percentile(detect_latencies, 50),
        "detect_latency_ms_p95": percentile(detect_latencies, 95),
        "encode_latency_ms_p50": percentile(encode_latencies, 50),
        "encode_latency_ms_p95": percentile(encode_latencies, 95),
        "detect_latency_ms_mean": statistics.fmean(detect_latencies) if detect_latencies else 0.0,
        "encode_latency_ms_mean": statistics.fmean(encode_latencies) if encode_latencies else 0.0,
    }
    with (output_dir / "benchmark.json").open("w", encoding="utf-8") as fout:
        json.dump(benchmark, fout, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Extract HOG/dlib 128D face embeddings.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", default="all", help="Comma-separated splits, or 'all'.")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit. Full run is default.")
    parser.add_argument("--workers", type=int, default=None, help="Override worker count. Default is half CPU cores.")
    return parser.parse_args()


def main() -> int:
    """脚本入口：读取 manifest、并行提取 HOG embedding、写出实验产物。"""

    args = parse_args()
    cpu_count = os.cpu_count() or 2
    worker_count = args.workers if args.workers is not None else max(1, cpu_count - 2)
    if worker_count < 1:
        raise ValueError("--workers must be >= 1")

    split_filter = parse_split_filter(args.splits)
    records = read_manifest(args.manifest, split_filter, args.limit)
    if not records:
        raise RuntimeError("No input images matched the manifest/filter settings.")

    start_time = time.perf_counter()
    results = run_workers(records, worker_count)
    elapsed_seconds = time.perf_counter() - start_time

    benchmark_extra = {
        "cpu_count": cpu_count,
        "worker_count": worker_count,
        "worker_policy": "cpu_count_minus_2" if args.workers is None else "manual_override",
        "manifest": str(args.manifest),
        "splits": args.splits,
        "limit": args.limit if args.limit is not None else "none",
        "elapsed_seconds": elapsed_seconds,
    }
    write_outputs(results, args.output_dir, benchmark_extra)
    print(f"HOG outputs written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
