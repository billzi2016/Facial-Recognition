#!/usr/bin/env python3
"""运行 FAISS 1:N 检索实验。

脚本意图：
- FAISS 实验回答的是“给一张 query 人脸，能不能在 gallery 中找到同一个人”。
- 同一个脚本同时支持 ArcFace 主路线和 HOG baseline，不把实验逻辑写死到某一种 embedding。
- 输入 embedding 必须来自 HDF5，dataset 名称默认是 `embeddings`。
- metadata CSV 用来确定 gallery、query_known、query_unknown 三类 split。
- 输出 index、query 结果、阈值扫描结果和 benchmark，方便综合报告直接引用。

实验边界：
- 本脚本只做检索和拒识，不做无监督聚类。
- DBSCAN 是另一个实验，虽然它复用同一份 embedding，但回答的问题不同。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import h5py
import numpy as np
from tqdm import tqdm


DEFAULT_ROUTES = {
    "insightface": {
        "embedding_file": "outputs/insightface/embeddings.h5",
        "metadata_file": "outputs/insightface/embedding_metadata.csv",
        "output_dir": "outputs/faiss/insightface",
        "threshold_min": 0.20,
        "threshold_max": 0.80,
        "threshold_steps": 61,
    },
    "hog": {
        "embedding_file": "outputs/hog/embeddings.h5",
        "metadata_file": "outputs/hog/embedding_metadata.csv",
        "output_dir": "outputs/faiss/hog",
        "threshold_min": 0.20,
        "threshold_max": 0.80,
        "threshold_steps": 61,
    },
}


@dataclass(frozen=True)
class MetadataRow:
    """一条 embedding metadata 记录。

    embedding_index 必须与 HDF5 dataset 的行号对应。FAISS 查询结果只返回向量行号，
    因此后续所有身份、图片路径和 split 都从这里回查。
    """

    embedding_index: int
    image_id: str
    path: str
    person_id: str
    split: str
    row_index: str


def read_embeddings(path: Path, dataset_name: str) -> np.ndarray:
    """从 HDF5 读取 embedding 矩阵。"""

    with h5py.File(path, "r") as h5:
        embeddings = np.asarray(h5[dataset_name], dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    return embeddings


def read_metadata(path: Path) -> list[MetadataRow]:
    """读取 embedding metadata CSV。"""

    rows: list[MetadataRow] = []
    with path.open("r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for row in tqdm(reader, desc=f"read {path.name}", unit="row"):
            rows.append(
                MetadataRow(
                    embedding_index=int(row["embedding_index"]),
                    image_id=row["image_id"],
                    path=row["path"],
                    person_id=row.get("person_id", ""),
                    split=row.get("split", ""),
                    row_index=row.get("row_index", ""),
                )
            )
    return rows


def normalize_l2(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embedding，用 inner product 表示 cosine similarity。

    FAISS 的 `IndexFlatIP` 对归一化向量做内积时，结果就是余弦相似度。
    这样 HOG 和 ArcFace 都可以使用同一套阈值扫描逻辑，分数越高越相似。
    """

    normalized = np.ascontiguousarray(embeddings.astype(np.float32, copy=True))
    faiss.normalize_L2(normalized)
    return normalized


def select_rows(metadata: list[MetadataRow], split: str, limit: int | None) -> list[MetadataRow]:
    """按 split 选择 metadata 行，并支持可选 limit。"""

    selected = [row for row in metadata if row.split == split and row.person_id]
    if limit is not None:
        return selected[:limit]
    return selected


def build_index(gallery_vectors: np.ndarray) -> faiss.Index:
    """构建精确 FAISS inner product index。"""

    index = faiss.IndexFlatIP(gallery_vectors.shape[1])
    index.add(gallery_vectors)
    return index


def search_queries(
    index: faiss.Index,
    query_vectors: np.ndarray,
    query_rows: list[MetadataRow],
    gallery_rows: list[MetadataRow],
    top_k: int,
    desc: str,
) -> list[dict[str, Any]]:
    """执行 Top-K 检索并整理成 CSV 行。

    FAISS 一次 search 可以处理一批 query。这里仍按 chunk 分批，是为了让 tqdm 能显示进度，
    同时避免一次性把超大结果矩阵塞进内存。
    """

    rows: list[dict[str, Any]] = []
    chunk_size = 4096
    for start in tqdm(range(0, len(query_rows), chunk_size), desc=desc, unit="chunk"):
        end = min(start + chunk_size, len(query_rows))
        scores, indices = index.search(query_vectors[start:end], top_k)
        for local_index, query_row in enumerate(query_rows[start:end]):
            top_index = int(indices[local_index][0])
            top_gallery = gallery_rows[top_index]
            top_score = float(scores[local_index][0])
            top_person_ids = [gallery_rows[int(idx)].person_id for idx in indices[local_index] if int(idx) >= 0]
            rows.append(
                {
                    "query_image_id": query_row.image_id,
                    "query_path": query_row.path,
                    "query_person_id": query_row.person_id,
                    "query_split": query_row.split,
                    "top1_image_id": top_gallery.image_id,
                    "top1_path": top_gallery.path,
                    "top1_person_id": top_gallery.person_id,
                    "top1_score": top_score,
                    "top1_correct": str(top_gallery.person_id == query_row.person_id).lower(),
                    "topk_person_ids": "|".join(top_person_ids),
                    "topk_hit": str(query_row.person_id in top_person_ids).lower(),
                }
            )
    return rows


def scan_thresholds(known_rows: list[dict[str, Any]], unknown_rows: list[dict[str, Any]], thresholds: np.ndarray) -> list[dict[str, Any]]:
    """扫描拒识阈值。

    对 known query，分数低于阈值表示被错误拒识。
    对 unknown query，分数低于阈值表示成功拒识；分数高于阈值表示误接收。
    """

    report: list[dict[str, Any]] = []
    known_scores = np.asarray([float(row["top1_score"]) for row in known_rows], dtype=np.float32)
    known_correct = np.asarray([row["top1_correct"] == "true" for row in known_rows], dtype=bool)
    unknown_scores = np.asarray([float(row["top1_score"]) for row in unknown_rows], dtype=np.float32)

    for threshold in tqdm(thresholds, desc="scan thresholds", unit="threshold"):
        known_accepted = known_scores >= threshold
        unknown_rejected = unknown_scores < threshold
        accepted_known_correct = known_correct & known_accepted
        report.append(
            {
                "threshold": f"{float(threshold):.6f}",
                "known_accept_rate": float(known_accepted.mean()) if known_accepted.size else 0.0,
                "known_top1_accuracy_after_threshold": float(accepted_known_correct.mean()) if known_correct.size else 0.0,
                "false_reject_rate": float((~known_accepted).mean()) if known_accepted.size else 0.0,
                "unknown_reject_rate": float(unknown_rejected.mean()) if unknown_rejected.size else 0.0,
                "false_accept_rate": float((~unknown_rejected).mean()) if unknown_rejected.size else 0.0,
            }
        )
    return report


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写 CSV。空结果也会写一个只有表头的文件，方便下游判断。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in tqdm(rows, desc=f"write {path.name}", unit="row"):
            writer.writerow(row)


def save_index(index: faiss.Index, path: Path) -> None:
    """保存 FAISS index。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def summarize(known_rows: list[dict[str, Any]], unknown_rows: list[dict[str, Any]], elapsed: float, extra: dict[str, Any]) -> dict[str, Any]:
    """生成 benchmark 摘要。"""

    known_scores = [float(row["top1_score"]) for row in known_rows]
    unknown_scores = [float(row["top1_score"]) for row in unknown_rows]
    top1_accuracy = statistics.fmean(row["top1_correct"] == "true" for row in known_rows) if known_rows else 0.0
    topk_accuracy = statistics.fmean(row["topk_hit"] == "true" for row in known_rows) if known_rows else 0.0
    return {
        **extra,
        "known_queries": len(known_rows),
        "unknown_queries": len(unknown_rows),
        "top1_accuracy": top1_accuracy,
        "topk_accuracy": topk_accuracy,
        "known_score_mean": statistics.fmean(known_scores) if known_scores else 0.0,
        "unknown_score_mean": statistics.fmean(unknown_scores) if unknown_scores else 0.0,
        "elapsed_seconds": elapsed,
    }


def run_route(args: argparse.Namespace) -> None:
    """运行单条 embedding 路线的 FAISS 实验。"""

    embeddings = normalize_l2(read_embeddings(args.embedding_file, args.dataset))
    metadata = read_metadata(args.metadata_file)
    if len(metadata) != embeddings.shape[0]:
        raise ValueError(f"Metadata rows ({len(metadata)}) do not match embeddings ({embeddings.shape[0]})")

    gallery_rows = select_rows(metadata, "gallery", None)
    known_rows = select_rows(metadata, "query_known", args.max_known)
    unknown_rows = select_rows(metadata, "query_unknown", args.max_unknown)
    if not gallery_rows:
        raise RuntimeError("No gallery rows found. Check metadata split labels.")

    gallery_indices = [row.embedding_index for row in gallery_rows]
    known_indices = [row.embedding_index for row in known_rows]
    unknown_indices = [row.embedding_index for row in unknown_rows]

    start = time.perf_counter()
    index = build_index(embeddings[gallery_indices])
    known_results = search_queries(index, embeddings[known_indices], known_rows, gallery_rows, args.top_k, "search known")
    unknown_results = search_queries(index, embeddings[unknown_indices], unknown_rows, gallery_rows, args.top_k, "search unknown")
    elapsed = time.perf_counter() - start

    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps, dtype=np.float32)
    threshold_rows = scan_thresholds(known_results, unknown_results, thresholds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_index(index, args.output_dir / "index.faiss")
    write_csv(args.output_dir / "known_search_results.csv", known_results)
    write_csv(args.output_dir / "unknown_search_results.csv", unknown_results)
    write_csv(args.output_dir / "threshold_report.csv", threshold_rows)

    benchmark = summarize(
        known_results,
        unknown_results,
        elapsed,
        {
            "route": args.route,
            "embedding_file": str(args.embedding_file),
            "metadata_file": str(args.metadata_file),
            "embedding_dim": int(embeddings.shape[1]),
            "index_type": "IndexFlatIP",
            "score": "cosine_similarity_after_l2_normalize",
            "top_k": args.top_k,
        },
    )
    with (args.output_dir / "benchmark.json").open("w", encoding="utf-8") as fout:
        json.dump(benchmark, fout, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Run FAISS lookup for one embedding route.")
    parser.add_argument("--route", choices=sorted(DEFAULT_ROUTES), default="insightface")
    parser.add_argument("--embedding-file", type=Path, default=None)
    parser.add_argument("--metadata-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset", default="embeddings")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-known", type=int, default=None)
    parser.add_argument("--max-unknown", type=int, default=None)
    parser.add_argument("--threshold-min", type=float, default=None)
    parser.add_argument("--threshold-max", type=float, default=None)
    parser.add_argument("--threshold-steps", type=int, default=None)
    args = parser.parse_args()

    defaults = DEFAULT_ROUTES[args.route]
    args.embedding_file = args.embedding_file or Path(defaults["embedding_file"])
    args.metadata_file = args.metadata_file or Path(defaults["metadata_file"])
    args.output_dir = args.output_dir or Path(defaults["output_dir"])
    args.threshold_min = args.threshold_min if args.threshold_min is not None else float(defaults["threshold_min"])
    args.threshold_max = args.threshold_max if args.threshold_max is not None else float(defaults["threshold_max"])
    args.threshold_steps = args.threshold_steps if args.threshold_steps is not None else int(defaults["threshold_steps"])
    return args


def main() -> int:
    """脚本入口。"""

    run_route(parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
