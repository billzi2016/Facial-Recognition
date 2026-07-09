#!/usr/bin/env python3
"""使用 InsightFace 预训练模型提取 ArcFace 512 维 embedding。

脚本意图：
- 这是当前项目主线：输入图片已经是 CelebA 对齐人脸图，目标是提取 ArcFace 512 维身份向量。
- 使用 InsightFace 标准 Python 包和 ONNX Runtime provider，不训练模型。
- Mac Apple Silicon 优先使用 `CoreMLExecutionProvider`，不可用时 fallback 到 `CPUExecutionProvider`。
- InsightFace `FaceAnalysis.get(img)` 是单图 API；本脚本不把它误写成 PyTorch DataLoader。
- HDF5 embedding 必须使用 `h5py` 存储，并在写入 dataset 时直接启用
  `compression="gzip", compression_opts=1`，不允许先写未压缩 H5 再外部 gzip。
- 所有全量处理和写盘动作使用 `tqdm` 展示进度、速度和粗略 ETA。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import onnxruntime as ort
from tqdm import tqdm


EMBEDDING_DIM = 512
DEFAULT_MANIFEST_PATH = Path("data/manifests/images.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/insightface")
DEFAULT_MODEL_NAME = "buffalo_l"
PREFERRED_PROVIDERS = ["CoreMLExecutionProvider", "CPUExecutionProvider"]


@dataclass(frozen=True)
class ManifestRecord:
    """输入 manifest 中一张图片的最小必要信息。"""

    row_index: int
    image_id: str
    path: Path
    person_id: str
    split: str


@dataclass(frozen=True)
class ArcFaceResult:
    """单张图片的 ArcFace 处理结果。

    embedding 为 `None` 表示该图片读取、检测或编码失败；失败原因写入 failures.csv。
    """

    row_index: int
    image_id: str
    path: str
    person_id: str
    split: str
    face_count: int
    selected_score: float | None
    selected_box_x1: float | None
    selected_box_y1: float | None
    selected_box_x2: float | None
    selected_box_y2: float | None
    latency_ms: float
    embedding: list[float] | None
    failure_reason: str


def parse_bool(value: str) -> bool:
    """解析 manifest 中的布尔字符串。"""

    return value.strip().lower() in {"1", "true", "yes", "y"}


def parse_split_filter(raw: str) -> set[str] | None:
    """解析 `--splits` 参数；`all` 或 `*` 表示不过滤。"""

    if raw.strip().lower() in {"", "all", "*"}:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def read_manifest(manifest_path: Path, split_filter: set[str] | None, limit: int | None) -> list[ManifestRecord]:
    """读取图片 manifest，并过滤 excluded、split 和可选 limit。"""

    records: list[ManifestRecord] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for row_index, row in enumerate(tqdm(reader, desc="read manifest", unit="row")):
            split = row.get("split", "")
            if split_filter is not None and split not in split_filter:
                continue
            if parse_bool(row.get("excluded", "false")):
                continue

            image_id = row.get("image_id", "")
            image_path = row.get("path", "")
            if not image_id or not image_path:
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


def choose_providers(preferred: list[str]) -> list[str]:
    """根据当前 ONNX Runtime 可用 provider 选择实际 provider 顺序。

    InsightFace 最终会把这个 provider 列表传给 ONNX Runtime。Mac 上如果存在
    `CoreMLExecutionProvider` 就优先使用；否则至少保留 CPU provider，保证脚本可运行。
    """

    available = ort.get_available_providers()
    selected = [provider for provider in preferred if provider in available]
    if "CPUExecutionProvider" not in selected:
        selected.append("CPUExecutionProvider")
    return selected


def build_app(model_name: str, providers: list[str], det_size: tuple[int, int]):
    """初始化 InsightFace FaceAnalysis。

    导入放在函数内，是为了让 `--help` 这类轻量命令不触发模型初始化，也方便错误信息集中在运行阶段。
    `ctx_id=0` 是 InsightFace 的历史接口要求；实际执行设备由 ONNX Runtime providers 决定。
    """

    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name=model_name, providers=providers)
    app.prepare(ctx_id=0, det_size=det_size)
    return app


def largest_face(faces: list[Any]) -> Any:
    """从 InsightFace 检测结果中选择 bbox 面积最大的人脸。"""

    return max(faces, key=lambda face: max(0.0, float(face.bbox[2] - face.bbox[0])) * max(0.0, float(face.bbox[3] - face.bbox[1])))


def process_one(app: Any, record: ManifestRecord) -> ArcFaceResult:
    """读取单张图片并提取 ArcFace embedding。

    CelebA 是已对齐人脸，但 InsightFace 标准 API 仍会先检测再返回 embedding。
    对检测失败的图片保留 failure 记录，而不是中断全量任务。
    """

    start = time.perf_counter()
    try:
        image = cv2.imread(str(record.path))
        if image is None:
            return ArcFaceResult(
                row_index=record.row_index,
                image_id=record.image_id,
                path=str(record.path),
                person_id=record.person_id,
                split=record.split,
                face_count=0,
                selected_score=None,
                selected_box_x1=None,
                selected_box_y1=None,
                selected_box_x2=None,
                selected_box_y2=None,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                embedding=None,
                failure_reason="image_read_failed",
            )

        faces = app.get(image)
        latency_ms = (time.perf_counter() - start) * 1000.0
        if not faces:
            return ArcFaceResult(
                row_index=record.row_index,
                image_id=record.image_id,
                path=str(record.path),
                person_id=record.person_id,
                split=record.split,
                face_count=0,
                selected_score=None,
                selected_box_x1=None,
                selected_box_y1=None,
                selected_box_x2=None,
                selected_box_y2=None,
                latency_ms=latency_ms,
                embedding=None,
                failure_reason="no_face_detected",
            )

        face = largest_face(faces)
        embedding = np.asarray(face.embedding, dtype=np.float32)
        if embedding.shape != (EMBEDDING_DIM,):
            return ArcFaceResult(
                row_index=record.row_index,
                image_id=record.image_id,
                path=str(record.path),
                person_id=record.person_id,
                split=record.split,
                face_count=len(faces),
                selected_score=float(getattr(face, "det_score", 0.0)),
                selected_box_x1=float(face.bbox[0]),
                selected_box_y1=float(face.bbox[1]),
                selected_box_x2=float(face.bbox[2]),
                selected_box_y2=float(face.bbox[3]),
                latency_ms=latency_ms,
                embedding=None,
                failure_reason=f"unexpected_embedding_shape:{embedding.shape}",
            )

        return ArcFaceResult(
            row_index=record.row_index,
            image_id=record.image_id,
            path=str(record.path),
            person_id=record.person_id,
            split=record.split,
            face_count=len(faces),
            selected_score=float(getattr(face, "det_score", 0.0)),
            selected_box_x1=float(face.bbox[0]),
            selected_box_y1=float(face.bbox[1]),
            selected_box_x2=float(face.bbox[2]),
            selected_box_y2=float(face.bbox[3]),
            latency_ms=latency_ms,
            embedding=embedding.tolist(),
            failure_reason="",
        )
    except Exception as exc:  # noqa: BLE001 - 全量实验记录坏样本，不因单张图中断。
        return ArcFaceResult(
            row_index=record.row_index,
            image_id=record.image_id,
            path=str(record.path),
            person_id=record.person_id,
            split=record.split,
            face_count=0,
            selected_score=None,
            selected_box_x1=None,
            selected_box_y1=None,
            selected_box_x2=None,
            selected_box_y2=None,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            embedding=None,
            failure_reason=f"exception:{type(exc).__name__}:{exc}",
        )


def percentile(values: list[float], pct: float) -> float:
    """计算简单百分位数；空列表返回 0。"""

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


def write_outputs(results: list[ArcFaceResult], output_dir: Path, benchmark_extra: dict[str, Any]) -> None:
    """写 HDF5 embedding、metadata、detections、failures 和 benchmark。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    successes = [result for result in results if result.embedding is not None]
    failures = [result for result in results if result.embedding is None]

    embeddings = np.asarray([result.embedding for result in successes], dtype=np.float32)
    h5_path = output_dir / "embeddings.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("embeddings", data=embeddings, compression="gzip", compression_opts=1)
        h5.attrs["embedding_dim"] = EMBEDDING_DIM
        h5.attrs["model"] = benchmark_extra["model_name"]
        h5.attrs["embedding_model"] = "ArcFace"
        h5.attrs["compression"] = "gzip"
        h5.attrs["compression_opts"] = 1
        for key, value in benchmark_extra.items():
            if isinstance(value, (str, int, float, bool)):
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
            "selected_score": result.selected_score,
            "selected_box_x1": result.selected_box_x1,
            "selected_box_y1": result.selected_box_y1,
            "selected_box_x2": result.selected_box_x2,
            "selected_box_y2": result.selected_box_y2,
            "latency_ms": f"{result.latency_ms:.4f}",
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
            "selected_score",
            "selected_box_x1",
            "selected_box_y1",
            "selected_box_x2",
            "selected_box_y2",
            "latency_ms",
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
            "latency_ms": f"{result.latency_ms:.4f}",
        }
        for result in failures
    ]
    write_csv(output_dir / "failures.csv", ["image_id", "path", "person_id", "split", "failure_reason", "latency_ms"], failure_rows)

    latencies = [result.latency_ms for result in results]
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
        "multi_face_rate": sum(1 for result in results if result.face_count > 1) / len(results) if results else 0.0,
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
        "latency_ms_mean": statistics.fmean(latencies) if latencies else 0.0,
    }
    with (output_dir / "benchmark.json").open("w", encoding="utf-8") as fout:
        json.dump(benchmark, fout, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Extract InsightFace ArcFace 512D embeddings.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--splits", default="all", help="Comma-separated splits, or 'all'.")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit. Full run is default.")
    parser.add_argument("--det-size", default="640,640", help="Detection input size, formatted as width,height.")
    return parser.parse_args()


def parse_det_size(raw: str) -> tuple[int, int]:
    """解析 InsightFace det_size 参数。"""

    parts = [int(item.strip()) for item in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("--det-size must be formatted as width,height")
    return parts[0], parts[1]


def main() -> int:
    """脚本入口：初始化 InsightFace，顺序提取 ArcFace embedding，并写实验产物。"""

    args = parse_args()
    det_size = parse_det_size(args.det_size)
    split_filter = parse_split_filter(args.splits)
    records = read_manifest(args.manifest, split_filter, args.limit)
    if not records:
        raise RuntimeError("No input images matched the manifest/filter settings.")

    available_providers = ort.get_available_providers()
    providers = choose_providers(PREFERRED_PROVIDERS)
    app = build_app(args.model_name, providers, det_size)

    start_time = time.perf_counter()
    results = [process_one(app, record) for record in tqdm(records, desc="arcface encode", unit="image")]
    elapsed_seconds = time.perf_counter() - start_time

    benchmark_extra = {
        "model_name": args.model_name,
        "onnxruntime_available_providers": ",".join(available_providers),
        "onnxruntime_selected_providers": ",".join(providers),
        "provider_used": providers[0],
        "coreml_available": "CoreMLExecutionProvider" in available_providers,
        "manifest": str(args.manifest),
        "splits": args.splits,
        "limit": args.limit if args.limit is not None else "none",
        "det_size": f"{det_size[0]},{det_size[1]}",
        "elapsed_seconds": elapsed_seconds,
    }
    write_outputs(results, args.output_dir, benchmark_extra)
    print(f"InsightFace outputs written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
