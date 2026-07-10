#!/usr/bin/env python3
"""扫描 DBSCAN 的 eps 参数。

脚本意图：
- DBSCAN 对 `eps` 非常敏感，同一组 embedding 换一个 eps 就可能从“全是噪声”
  变成“一个大簇”。
- 本脚本专门做参数扫描，不生成 PCA/t-SNE/UMAP 图，避免每个 eps 都重复做慢速绘图。
- 输出 CSV 和 JSON，方便先找到可用区间，再用 `run_dbscan_experiment.py` 跑正式图。

维护说明：
- 默认 route 是 HOG，默认 metric 是 euclidean，因为 HOG + cosine 会把样本连成一个大簇。
- ArcFace 也可以复用这个脚本，只需要传 `--route insightface`。
- embedding 仍然从 HDF5 读取，不改动上游结果文件。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
from tqdm import tqdm


DEFAULT_ROUTES = {
    "insightface": {
        "embedding_file": "outputs/insightface/embeddings.h5",
        "metadata_file": "outputs/insightface/embedding_metadata.csv",
        "output_dir": "outputs/dbscan/insightface_cosine_eps_sweep",
        "eps_values": "0.30,0.34,0.38,0.42,0.45,0.48,0.52,0.56,0.60,0.64,0.68",
        "metric": "cosine",
    },
    "hog": {
        "embedding_file": "outputs/hog/embeddings.h5",
        "metadata_file": "outputs/hog/embedding_metadata.csv",
        "output_dir": "outputs/dbscan/hog_euclidean_eps_sweep",
        "eps_values": "0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00",
        "metric": "euclidean",
    },
}


@dataclass(frozen=True)
class MetadataRow:
    """一条 embedding metadata 记录。

    person_id 用来评估聚类纯度，embedding_index 用来从 HDF5 中取对应行。
    """

    embedding_index: int
    image_id: str
    person_id: str


def parse_eps_values(raw: str) -> list[float]:
    """解析逗号分隔的 eps 列表。"""

    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("eps list is empty")
    return values


def read_embeddings(path: Path, dataset_name: str) -> np.ndarray:
    """从 HDF5 读取 embedding 矩阵。"""

    with h5py.File(path, "r") as h5:
        embeddings = np.asarray(h5[dataset_name], dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    return embeddings


def read_metadata(path: Path) -> list[MetadataRow]:
    """读取 metadata CSV。

    这里只取扫描需要的字段，避免把无关列传来传去。
    """

    rows: list[MetadataRow] = []
    with path.open("r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for row in tqdm(reader, desc=f"read {path.name}", unit="row"):
            rows.append(
                MetadataRow(
                    embedding_index=int(row["embedding_index"]),
                    image_id=row["image_id"],
                    person_id=row.get("person_id", ""),
                )
            )
    return rows


def sample_rows(rows: list[MetadataRow], max_samples: int, seed: int) -> list[MetadataRow]:
    """按固定随机种子抽样，保证 sweep 和正式 DBSCAN 可复现。"""

    if max_samples <= 0 or len(rows) <= max_samples:
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), max_samples))
    return [rows[index] for index in indices]


def summarize_labels(labels: np.ndarray, rows: list[MetadataRow], elapsed: float, extra: dict[str, Any]) -> dict[str, Any]:
    """根据 DBSCAN labels 生成一行 sweep 指标。

    ARI/NMI 只在非噪声点上计算，因为噪声点的 `-1` 不是身份簇。
    如果某个 eps 把所有点都标为噪声，指标会写 0，避免误读。
    """

    person_ids = np.asarray([row.person_id for row in rows], dtype=object)
    known_mask = np.asarray([bool(row.person_id) for row in rows], dtype=bool)
    non_noise = labels != -1
    comparable = known_mask & non_noise
    cluster_count = int(len(set(labels.tolist()) - {-1}))
    noise_count = int((labels == -1).sum())
    largest_cluster = 0
    if cluster_count:
        counts = np.bincount(labels[labels >= 0])
        largest_cluster = int(counts.max()) if counts.size else 0

    return {
        **extra,
        "sample_count": int(len(labels)),
        "cluster_count_excluding_noise": cluster_count,
        "noise_count": noise_count,
        "noise_rate": float(noise_count / len(labels)) if len(labels) else 0.0,
        "largest_cluster_size": largest_cluster,
        "largest_cluster_rate": float(largest_cluster / len(labels)) if len(labels) else 0.0,
        "adjusted_rand_index_non_noise": float(adjusted_rand_score(person_ids[comparable], labels[comparable])) if comparable.any() else 0.0,
        "normalized_mutual_info_non_noise": float(normalized_mutual_info_score(person_ids[comparable], labels[comparable])) if comparable.any() else 0.0,
        "elapsed_seconds": elapsed,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出 sweep CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Sweep DBSCAN eps values for one embedding route.")
    parser.add_argument("--route", choices=sorted(DEFAULT_ROUTES), default="hog")
    parser.add_argument("--embedding-file", type=Path, default=None)
    parser.add_argument("--metadata-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset", default="embeddings")
    parser.add_argument("--eps-values", default=None)
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--metric", default=None)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    defaults = DEFAULT_ROUTES[args.route]
    args.embedding_file = args.embedding_file or Path(defaults["embedding_file"])
    args.metadata_file = args.metadata_file or Path(defaults["metadata_file"])
    args.output_dir = args.output_dir or Path(defaults["output_dir"])
    args.eps_values = parse_eps_values(args.eps_values or defaults["eps_values"])
    args.metric = args.metric if args.metric is not None else str(defaults["metric"])
    return args


def main() -> int:
    """脚本入口。"""

    args = parse_args()
    embeddings_all = read_embeddings(args.embedding_file, args.dataset)
    metadata_all = read_metadata(args.metadata_file)
    if len(metadata_all) != embeddings_all.shape[0]:
        raise ValueError(f"Metadata rows ({len(metadata_all)}) do not match embeddings ({embeddings_all.shape[0]})")

    metadata = sample_rows(metadata_all, args.max_samples, args.seed)
    indices = [row.embedding_index for row in metadata]
    embeddings = embeddings_all[indices]
    if args.metric == "cosine":
        embeddings = normalize(embeddings)

    rows: list[dict[str, Any]] = []
    for eps in tqdm(args.eps_values, desc=f"sweep {args.route}", unit="eps"):
        start = time.perf_counter()
        labels = DBSCAN(eps=eps, min_samples=args.min_samples, metric=args.metric, n_jobs=-1).fit_predict(embeddings)
        elapsed = time.perf_counter() - start
        rows.append(
            summarize_labels(
                labels,
                metadata,
                elapsed,
                {
                    "route": args.route,
                    "eps": eps,
                    "min_samples": args.min_samples,
                    "metric": args.metric,
                    "max_samples": args.max_samples,
                    "embedding_dim": int(embeddings.shape[1]),
                },
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "eps_sweep.csv", rows)
    with (args.output_dir / "eps_sweep.json").open("w", encoding="utf-8") as fout:
        json.dump(rows, fout, ensure_ascii=False, indent=2)
    print(f"DBSCAN eps sweep written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
