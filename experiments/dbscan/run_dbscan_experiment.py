#!/usr/bin/env python3
"""运行 DBSCAN 聚类和 2D 可视化实验。

脚本意图：
- DBSCAN 实验回答的是“不给标签时，embedding 能不能自然分成相册式分组”。
- 它和 FAISS 是两个实验。FAISS 做 1:N 检索，DBSCAN 做无监督聚类。
- 同一个脚本同时支持 ArcFace 主路线和 HOG baseline。
- 聚类标签、PCA/t-SNE/UMAP 坐标、JPG 图和 report 都按路线分别输出。

全量提醒：
- DBSCAN 和 t-SNE/UMAP 在 20 万张图片上可能非常慢，也可能占用大量内存。
- 脚本默认提供 `--max-samples` 和 `--plot-sample-size`，用于先做可控规模实验。
- 如果要跑全量，把 `--max-samples` 设为 0；这会进入真正的全量聚类。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
from tqdm import tqdm


DEFAULT_ROUTES = {
    "insightface": {
        "embedding_file": "outputs/insightface/embeddings.h5",
        "metadata_file": "outputs/insightface/embedding_metadata.csv",
        "output_dir": "outputs/dbscan/insightface",
        "eps": 0.56,
        "metric": "cosine",
    },
    "hog": {
        "embedding_file": "outputs/hog/embeddings.h5",
        "metadata_file": "outputs/hog/embedding_metadata.csv",
        "output_dir": "outputs/dbscan/hog",
        "eps": 0.40,
        "metric": "euclidean",
    },
}


@dataclass(frozen=True)
class MetadataRow:
    """一条 embedding metadata 记录。"""

    embedding_index: int
    image_id: str
    path: str
    person_id: str
    split: str
    row_index: str


def configure_runtime_cache(output_dir: Path) -> None:
    """给 matplotlib 和 UMAP/numba 指定可写缓存目录。

    某些环境导入 `matplotlib` 或 `umap` 时会因为默认缓存位置不可用而报错。这里在导入
    这些库之前设置缓存目录，让缓存落在被 git 忽略的实验输出目录中。
    """

    cache_dir = output_dir / "_numba_cache"
    mpl_dir = output_dir / "_matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def read_embeddings(path: Path, dataset_name: str) -> np.ndarray:
    """从 HDF5 读取 embedding。"""

    with h5py.File(path, "r") as h5:
        embeddings = np.asarray(h5[dataset_name], dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    return embeddings


def read_metadata(path: Path) -> list[MetadataRow]:
    """读取 metadata CSV。"""

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


def sample_rows(rows: list[MetadataRow], max_samples: int, seed: int) -> list[MetadataRow]:
    """按需抽样 metadata 行。

    `max_samples <= 0` 表示不抽样，直接使用全部成功 embedding。
    """

    if max_samples <= 0 or len(rows) <= max_samples:
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), max_samples))
    return [rows[index] for index in indices]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in tqdm(rows, desc=f"write {path.name}", unit="row"):
            writer.writerow(row)


def run_dbscan(embeddings: np.ndarray, eps: float, min_samples: int, metric: str) -> np.ndarray:
    """运行 DBSCAN。"""

    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    return model.fit_predict(embeddings)


def compute_2d(embeddings: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """计算 PCA、t-SNE 和 UMAP 的 2D 坐标。"""

    outputs: dict[str, np.ndarray] = {}
    for name in tqdm(["pca", "tsne", "umap"], desc="compute 2d", unit="method"):
        if name == "pca":
            outputs[name] = PCA(n_components=2, random_state=seed).fit_transform(embeddings)
        elif name == "tsne":
            perplexity = max(5, min(30, (len(embeddings) - 1) // 3))
            outputs[name] = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(embeddings)
        else:
            import umap

            outputs[name] = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1).fit_transform(embeddings)
    return outputs


def save_plot(points: np.ndarray, labels: np.ndarray, path: Path, title: str) -> None:
    """保存 JPG scatter 图。"""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7), dpi=140)
    unique_labels = np.unique(labels)
    for label in tqdm(unique_labels, desc=f"plot {path.name}", unit="cluster", leave=False):
        mask = labels == label
        if label == -1:
            plt.scatter(points[mask, 0], points[mask, 1], s=4, c="#888888", alpha=0.35, label="-1 noise")
        else:
            plt.scatter(points[mask, 0], points[mask, 1], s=4, alpha=0.70, label=str(label))
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    if len(unique_labels) <= 15:
        plt.legend(markerscale=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, format="jpg")
    plt.close()

    # 不同 matplotlib 版本对 JPG quality 参数支持不一致。
    # 先确保 JPG 一定写出，再用 Pillow 尝试按 quality=90 重新保存。
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.save(path, format="JPEG", quality=90, optimize=True)
    except Exception:
        pass


def summarize(labels: np.ndarray, metadata: list[MetadataRow], elapsed: float, extra: dict[str, Any]) -> dict[str, Any]:
    """生成聚类摘要。"""

    known_mask = np.asarray([bool(row.person_id) for row in metadata], dtype=bool)
    person_ids = np.asarray([row.person_id for row in metadata], dtype=object)
    non_noise = labels != -1
    comparable = known_mask & non_noise
    return {
        **extra,
        "sample_count": int(len(labels)),
        "cluster_count_excluding_noise": int(len(set(labels.tolist()) - {-1})),
        "noise_count": int((labels == -1).sum()),
        "noise_rate": float((labels == -1).mean()) if len(labels) else 0.0,
        "elapsed_seconds": elapsed,
        "adjusted_rand_index_non_noise": float(adjusted_rand_score(person_ids[comparable], labels[comparable])) if comparable.any() else 0.0,
        "normalized_mutual_info_non_noise": float(normalized_mutual_info_score(person_ids[comparable], labels[comparable])) if comparable.any() else 0.0,
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Run DBSCAN clustering and 2D visualization for one embedding route.")
    parser.add_argument("--route", choices=sorted(DEFAULT_ROUTES), default="insightface")
    parser.add_argument("--embedding-file", type=Path, default=None)
    parser.add_argument("--metadata-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset", default="embeddings")
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--metric", default=None)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--plot-sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    defaults = DEFAULT_ROUTES[args.route]
    args.embedding_file = args.embedding_file or Path(defaults["embedding_file"])
    args.metadata_file = args.metadata_file or Path(defaults["metadata_file"])
    args.output_dir = args.output_dir or Path(defaults["output_dir"])
    args.eps = args.eps if args.eps is not None else float(defaults["eps"])
    args.metric = args.metric if args.metric is not None else str(defaults["metric"])
    return args


def main() -> int:
    """脚本入口。"""

    args = parse_args()
    configure_runtime_cache(args.output_dir)

    embeddings_all = read_embeddings(args.embedding_file, args.dataset)
    metadata_all = read_metadata(args.metadata_file)
    if len(metadata_all) != embeddings_all.shape[0]:
        raise ValueError(f"Metadata rows ({len(metadata_all)}) do not match embeddings ({embeddings_all.shape[0]})")

    metadata = sample_rows(metadata_all, args.max_samples, args.seed)
    indices = [row.embedding_index for row in metadata]
    embeddings = embeddings_all[indices]
    if args.metric == "cosine":
        embeddings = normalize(embeddings)

    start = time.perf_counter()
    labels = run_dbscan(embeddings, args.eps, args.min_samples, args.metric)
    elapsed = time.perf_counter() - start

    label_rows = [
        {
            "image_id": row.image_id,
            "path": row.path,
            "person_id": row.person_id,
            "split": row.split,
            "embedding_index": row.embedding_index,
            "cluster_label": int(label),
        }
        for row, label in zip(metadata, labels, strict=True)
    ]
    write_csv(args.output_dir / "cluster_labels.csv", label_rows)

    plot_metadata = sample_rows(metadata, args.plot_sample_size, args.seed)
    plot_lookup = {row.embedding_index: i for i, row in enumerate(metadata)}
    plot_indices = [plot_lookup[row.embedding_index] for row in plot_metadata]
    plot_embeddings = embeddings[plot_indices]
    plot_labels = labels[plot_indices]
    points_by_method = compute_2d(plot_embeddings, plot_labels, args.seed)
    for method, points in points_by_method.items():
        save_plot(points, plot_labels, args.output_dir / f"{method}_2d.jpg", f"{args.route} {method.upper()} DBSCAN eps={args.eps} n={len(plot_labels)}")

    report = summarize(
        labels,
        metadata,
        elapsed,
        {
            "route": args.route,
            "embedding_file": str(args.embedding_file),
            "metadata_file": str(args.metadata_file),
            "embedding_dim": int(embeddings.shape[1]),
            "eps": args.eps,
            "min_samples": args.min_samples,
            "metric": args.metric,
            "max_samples": args.max_samples,
            "plot_sample_size": args.plot_sample_size,
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "cluster_report.json").open("w", encoding="utf-8") as fout:
        json.dump(report, fout, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
