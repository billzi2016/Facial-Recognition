#!/usr/bin/env python3
"""生成实验报告用的对比图。

脚本意图：
- 把 HOG baseline 和 ArcFace 主路线的关键结果画进图里，避免报告只放没有上下文的散点图。
- 图像统一输出 JPG，放在 `outputs/report/` 下。
- 当前项目还没有决定最终哪些图进入仓库，所以本脚本只生成本地报告资产，不修改 `.gitignore`。
- 图中直接写入样本数、簇数量、噪声率、Top-1/Top-K(k=5) 等数字，让读者不打开 JSON 也能看懂结果。

图的含义：
- celeba_identity_examples 展示同一身份的多张图和不同身份的图，说明 gallery、known query
  和 unknown query 的实验来源。
- hog_feature_grid 展示 HOG 实际依赖的局部梯度方向，而不是 embedding PCA 空间里的方向。
- quality_comparison 展示两条路线提取 embedding 的成功率和失败数。
- faiss_comparison 展示两条路线的检索准确率和已知/陌生人分数间隔。
- dbscan_comparison 展示两条路线的聚类数量、噪声率和标签一致性指标。
- vector_field 图使用 PCA 2D 坐标，并画出每个区域指向其簇中心的平均方向。
  这不是物理向量场，而是 embedding 空间中的“局部归拢方向”，用来观察聚类结构是否清晰。
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm


OUTPUT_DIR = Path("outputs/report")
ROUTES = {
    "ArcFace": {
        "key": "insightface",
        "embedding_file": Path("outputs/insightface/embeddings.h5"),
        "metadata_file": Path("outputs/insightface/embedding_metadata.csv"),
        "embedding_benchmark": Path("outputs/insightface/benchmark.json"),
        "faiss_benchmark": Path("outputs/faiss/insightface/benchmark.json"),
        "dbscan_report": Path("outputs/dbscan/insightface/cluster_report.json"),
        "cluster_labels": Path("outputs/dbscan/insightface/cluster_labels.csv"),
        "color": "#2f6fed",
    },
    "HOG": {
        "key": "hog",
        "embedding_file": Path("outputs/hog/embeddings.h5"),
        "metadata_file": Path("outputs/hog/embedding_metadata.csv"),
        "embedding_benchmark": Path("outputs/hog/benchmark.json"),
        "faiss_benchmark": Path("outputs/faiss/hog/benchmark.json"),
        "dbscan_report": Path("outputs/dbscan/hog/cluster_report.json"),
        "cluster_labels": Path("outputs/dbscan/hog/cluster_labels.csv"),
        "color": "#d55e00",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""

    with path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def read_cluster_labels(path: Path) -> list[dict[str, str]]:
    """读取 DBSCAN cluster label CSV。"""

    with path.open("r", newline="", encoding="utf-8") as fin:
        return list(csv.DictReader(fin))


def read_embeddings(path: Path, indices: list[int]) -> np.ndarray:
    """按 embedding_index 读取 HDF5 embedding。"""

    with h5py.File(path, "r") as h5:
        dataset = h5["embeddings"]
        return np.asarray(dataset[indices], dtype=np.float32)


def save_bar_labels(ax: Any, bars: Any, fmt: str = "{:.2f}") -> None:
    """给柱状图添加数值标签。"""

    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value, fmt.format(value), ha="center", va="bottom", fontsize=9)


def save_figure(path: Path) -> None:
    """保存 JPG，并用 Pillow 尝试设置质量参数。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, format="jpg")
    plt.close()
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.save(path, format="JPEG", quality=90, optimize=True)
    except Exception:
        pass


def read_metadata_dicts(path: Path) -> list[dict[str, str]]:
    """读取 embedding metadata，并保留原始字段名。

    报告里的样例图需要图片路径、person_id 和 split。这里不复用 DBSCAN 的
    cluster_labels.csv，是因为 cluster 文件只覆盖抽样后的聚类样本，而 CelebA
    数据集示例应该从完整 metadata 中选择。
    """

    with path.open("r", newline="", encoding="utf-8") as fin:
        return list(csv.DictReader(fin))


def load_rgb_image(path: str) -> np.ndarray:
    """读取一张 RGB 图片。

    Matplotlib 使用 RGB，而 OpenCV 默认返回 BGR。为了避免颜色通道错误，这里使用
    Pillow 统一读取，后面的 HOG 梯度计算再按需要转成灰度图。
    """

    return np.asarray(Image.open(path).convert("RGB"))


def choose_example_rows(metadata: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """选择 CelebA 报告样例。

    第一行需要同一个 person_id 的三张图，用来说明 gallery 与 query_known 的关系：
    gallery 是数据库里的已知照片，query_known 是同一身份的另一张照片。

    第二行需要三个不同 person_id 的图，用来说明不同身份和 unknown query。优先选
    query_unknown，是为了让读者看到陌生人拒识实验里用到的样本来源。
    """

    by_person: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metadata:
        if row.get("person_id") and Path(row["path"]).exists():
            by_person[row["person_id"]].append(row)

    same_person: list[dict[str, str]] = []
    for rows in by_person.values():
        splits = {row["split"] for row in rows}
        if len(rows) >= 3 and {"gallery", "query_known"} <= splits:
            same_person = sorted(rows, key=lambda item: (item["split"] != "gallery", item["image_id"]))[:3]
            break
    if not same_person:
        same_person = next(rows[:3] for rows in by_person.values() if len(rows) >= 3)

    different_people: list[dict[str, str]] = []
    seen: set[str] = set()
    preferred = [row for row in metadata if row.get("split") == "query_unknown"]
    fallback = metadata
    for row in preferred + fallback:
        person_id = row.get("person_id", "")
        if not person_id or person_id in seen or not Path(row["path"]).exists():
            continue
        different_people.append(row)
        seen.add(person_id)
        if len(different_people) == 3:
            break

    return same_person[:3], different_people


def plot_celeba_identity_examples(metadata: list[dict[str, str]]) -> None:
    """画 CelebA 同人和异人样例图。

    这张图不参与模型评估，只负责解释数据集结构：同一身份会出现在 gallery 和
    query_known 中，不在 gallery 的身份会进入 query_unknown，用来测试陌生人拒识。
    """

    same_person, different_people = choose_example_rows(metadata)
    rows = [same_person, different_people]
    row_titles = ["same person: gallery and known queries", "different people: unknown/retrieval contrast"]

    fig, axes = plt.subplots(2, 3, figsize=(10, 7), dpi=150)
    for row_index, row_group in enumerate(rows):
        for col_index, item in enumerate(row_group):
            ax = axes[row_index][col_index]
            ax.imshow(load_rgb_image(item["path"]))
            ax.set_title(f"id={item['person_id']} | {item['split']}\n{item['image_id']}", fontsize=9)
            ax.axis("off")
        axes[row_index][0].set_ylabel(row_titles[row_index], fontsize=10)

    fig.suptitle("CelebA aligned face examples used by the gallery, known query, and unknown query splits")
    save_figure(OUTPUT_DIR / "celeba_identity_examples.jpg")


def compute_hog_cell_vectors(image: np.ndarray, cell_size: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算 HOG 风格的局部梯度方向。

    HOG 的核心不是直接记住像素，而是在小网格里统计边缘朝向。这里为了报告可读性，
    对每个 cell 计算一个“平均方向”：
    - Sobel 算子得到每个像素的 x/y 梯度。
    - 梯度幅值越大，对方向平均的贡献越大。
    - 方向使用 unsigned orientation，也就是 0 到 180 度；一条边向左或向右都表示同一条边。

    返回值直接给 Matplotlib quiver 使用。
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (144, 180), interpolation=cv2.INTER_AREA)
    gray_float = gray.astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    angle = np.mod(np.arctan2(grad_y, grad_x), np.pi)

    xs: list[float] = []
    ys: list[float] = []
    us: list[float] = []
    vs: list[float] = []
    strengths: list[float] = []

    for y in range(0, gray.shape[0] - cell_size + 1, cell_size):
        for x in range(0, gray.shape[1] - cell_size + 1, cell_size):
            cell_mag = magnitude[y : y + cell_size, x : x + cell_size]
            if float(cell_mag.sum()) <= 0.02:
                continue
            cell_angle = angle[y : y + cell_size, x : x + cell_size]
            # unsigned orientation 要按 2 * angle 做圆周平均，否则 1 度和 179 度会被错误抵消。
            mean_cos = float((np.cos(2 * cell_angle) * cell_mag).sum())
            mean_sin = float((np.sin(2 * cell_angle) * cell_mag).sum())
            mean_angle = 0.5 * np.arctan2(mean_sin, mean_cos)
            strength = float(cell_mag.mean())
            xs.append(x + cell_size / 2)
            ys.append(y + cell_size / 2)
            us.append(np.cos(mean_angle) * strength)
            vs.append(np.sin(mean_angle) * strength)
            strengths.append(strength)

    return gray, np.asarray(xs), np.asarray(ys), np.asarray(us), np.asarray(vs)


def plot_hog_feature_grid(metadata: list[dict[str, str]]) -> None:
    """画 6 张 CelebA 图片的 HOG 局部梯度方向。

    这张图专门解释 HOG baseline 的“特征”是什么。它不是最终 128 维 dlib embedding，
    而是 HOG 检测阶段常用的局部方向统计直觉：脸部轮廓、眼睛、鼻梁和嘴部会形成
    一组边缘方向分布，传统方法靠这些方向模式找脸。
    """

    same_person, different_people = choose_example_rows(metadata)
    examples = (same_person + different_people)[:6]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=150)
    for ax, item in zip(axes.flat, examples, strict=True):
        image = load_rgb_image(item["path"])
        gray, xs, ys, us, vs = compute_hog_cell_vectors(image)
        ax.imshow(gray, cmap="gray")
        ax.quiver(xs, ys, us, -vs, color="#f2c14e", angles="xy", scale_units="xy", scale=0.08, width=0.004)
        ax.set_title(f"id={item['person_id']} | {item['split']}\n{item['image_id']}", fontsize=9)
        ax.axis("off")

    fig.suptitle("HOG local gradient directions, 6 CelebA faces, 3 columns x 2 rows")
    save_figure(OUTPUT_DIR / "hog_feature_grid.jpg")


def plot_quality_comparison(metrics: dict[str, dict[str, Any]]) -> None:
    """画 embedding 提取质量对比图。"""

    names = list(metrics)
    success_rates = [metrics[name]["embedding"]["faces_detected_rate"] * 100 for name in names]
    failed = [metrics[name]["embedding"]["failed_images"] for name in names]
    colors = [ROUTES[name]["color"] for name in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=140)
    bars = axes[0].bar(names, success_rates, color=colors)
    axes[0].set_title("Embedding success rate")
    axes[0].set_ylabel("success rate (%)")
    axes[0].set_ylim(0, 105)
    save_bar_labels(axes[0], bars, "{:.1f}")

    bars = axes[1].bar(names, failed, color=colors)
    axes[1].set_title("Failed images")
    axes[1].set_ylabel("count")
    save_bar_labels(axes[1], bars, "{:.0f}")

    fig.suptitle("Embedding extraction: ArcFace keeps almost all aligned faces, HOG misses more")
    save_figure(OUTPUT_DIR / "embedding_quality_comparison.jpg")


def plot_faiss_comparison(metrics: dict[str, dict[str, Any]]) -> None:
    """画 FAISS 检索指标对比图。"""

    names = list(metrics)
    colors = [ROUTES[name]["color"] for name in names]
    top1 = [metrics[name]["faiss"]["top1_accuracy"] * 100 for name in names]
    topk = [metrics[name]["faiss"]["topk_accuracy"] * 100 for name in names]
    score_gap = [
        metrics[name]["faiss"]["known_score_mean"] - metrics[name]["faiss"]["unknown_score_mean"]
        for name in names
    ]

    x = np.arange(len(names))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=140)
    bars1 = axes[0].bar(x - width / 2, top1, width, label="Top-1", color=colors)
    bars2 = axes[0].bar(x + width / 2, topk, width, label="Top-K(k=5)", color=["#7aa6ff", "#f0a36a"])
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("accuracy (%)")
    axes[0].set_title("Known identity retrieval")
    axes[0].legend()
    save_bar_labels(axes[0], bars1, "{:.1f}")
    save_bar_labels(axes[0], bars2, "{:.1f}")

    bars = axes[1].bar(names, score_gap, color=colors)
    axes[1].set_title("Mean score gap: known minus unknown")
    axes[1].set_ylabel("cosine score gap")
    save_bar_labels(axes[1], bars, "{:.3f}")

    fig.suptitle("FAISS search: ArcFace separates known and unknown identities more clearly")
    save_figure(OUTPUT_DIR / "faiss_comparison.jpg")


def plot_dbscan_comparison(metrics: dict[str, dict[str, Any]]) -> None:
    """画 DBSCAN 聚类指标对比图。"""

    names = list(metrics)
    colors = [ROUTES[name]["color"] for name in names]
    clusters = [metrics[name]["dbscan"]["cluster_count_excluding_noise"] for name in names]
    noise_rates = [metrics[name]["dbscan"]["noise_rate"] * 100 for name in names]
    nmi = [metrics[name]["dbscan"]["normalized_mutual_info_non_noise"] for name in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=140)
    bars = axes[0].bar(names, clusters, color=colors)
    axes[0].set_title("Non-noise clusters")
    axes[0].set_ylabel("cluster count")
    save_bar_labels(axes[0], bars, "{:.0f}")

    bars = axes[1].bar(names, noise_rates, color=colors)
    axes[1].set_title("Noise rate")
    axes[1].set_ylabel("noise (%)")
    axes[1].set_ylim(0, 100)
    save_bar_labels(axes[1], bars, "{:.1f}")

    bars = axes[2].bar(names, nmi, color=colors)
    axes[2].set_title("NMI on non-noise points")
    axes[2].set_ylabel("NMI")
    axes[2].set_ylim(0, 1.05)
    save_bar_labels(axes[2], bars, "{:.3f}")

    fig.suptitle("DBSCAN: ArcFace covers more identities; HOG forms fewer high-purity clusters")
    save_figure(OUTPUT_DIR / "dbscan_comparison.jpg")


def sample_cluster_rows(rows: list[dict[str, str]], sample_size: int, seed: int) -> list[dict[str, str]]:
    """抽样用于向量场图的 cluster rows。"""

    if len(rows) <= sample_size:
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), sample_size))
    return [rows[index] for index in indices]


def plot_cluster_distribution(route_name: str, rows: list[dict[str, str]]) -> None:
    """画主要 cluster 大小分布。"""

    counts = Counter(int(row["cluster_label"]) for row in rows)
    noise = counts.pop(-1, 0)
    top = counts.most_common(20)
    labels = [str(label) for label, _ in top]
    values = [count for _, count in top]

    plt.figure(figsize=(12, 5), dpi=140)
    bars = plt.bar(labels, values, color=ROUTES[route_name]["color"])
    plt.title(f"{route_name} DBSCAN cluster size distribution, top 20 clusters, noise={noise}")
    plt.xlabel("cluster label")
    plt.ylabel("images")
    save_bar_labels(plt.gca(), bars, "{:.0f}")
    save_figure(OUTPUT_DIR / f"{ROUTES[route_name]['key']}_cluster_distribution.jpg")


def plot_vector_field(route_name: str, rows: list[dict[str, str]], route: dict[str, Any], sample_size: int = 6000) -> None:
    """画 embedding 空间中的局部归拢方向图。

    做法：
    1. 从 DBSCAN 结果里抽样。
    2. 用 PCA 把 embedding 压到 2D。
    3. 对每个非噪声簇计算 2D 中心。
    4. 对每个网格区域，平均该区域内样本指向各自簇中心的方向。

    如果一种 embedding 的聚类结构塌成一个大簇，向量场会显示大量点都被拉向同一个中心。
    如果聚类结构更清楚，方向会分布在多个局部区域。
    """

    sampled = sample_cluster_rows(rows, sample_size, seed=42)
    indices = [int(row["embedding_index"]) for row in sampled]
    labels = np.asarray([int(row["cluster_label"]) for row in sampled], dtype=np.int32)
    embeddings = normalize(read_embeddings(route["embedding_file"], indices))
    points = PCA(n_components=2, random_state=42).fit_transform(embeddings)

    centers: dict[int, np.ndarray] = {}
    for label in sorted(set(labels.tolist())):
        if label == -1:
            continue
        mask = labels == label
        if mask.sum() > 0:
            centers[label] = points[mask].mean(axis=0)

    grid_bins = 18
    x_edges = np.linspace(points[:, 0].min(), points[:, 0].max(), grid_bins + 1)
    y_edges = np.linspace(points[:, 1].min(), points[:, 1].max(), grid_bins + 1)
    buckets: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    origins: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)

    for point, label in zip(points, labels, strict=True):
        if label == -1 or label not in centers:
            continue
        x_bin = np.searchsorted(x_edges, point[0], side="right") - 1
        y_bin = np.searchsorted(y_edges, point[1], side="right") - 1
        if x_bin < 0 or x_bin >= grid_bins or y_bin < 0 or y_bin >= grid_bins:
            continue
        vector = centers[label] - point
        norm = np.linalg.norm(vector)
        if norm > 0:
            buckets[(x_bin, y_bin)].append(vector / norm)
            origins[(x_bin, y_bin)].append(point)

    arrow_x: list[float] = []
    arrow_y: list[float] = []
    arrow_u: list[float] = []
    arrow_v: list[float] = []
    for key, vectors in buckets.items():
        if len(vectors) < 3:
            continue
        origin = np.mean(origins[key], axis=0)
        vector = np.mean(vectors, axis=0)
        arrow_x.append(float(origin[0]))
        arrow_y.append(float(origin[1]))
        arrow_u.append(float(vector[0]))
        arrow_v.append(float(vector[1]))

    plt.figure(figsize=(10, 8), dpi=140)
    noise_mask = labels == -1
    plt.scatter(points[noise_mask, 0], points[noise_mask, 1], s=4, c="#999999", alpha=0.25, label="noise")
    plt.scatter(points[~noise_mask, 0], points[~noise_mask, 1], s=4, c=labels[~noise_mask], cmap="tab20", alpha=0.65, label="clustered")
    if arrow_x:
        plt.quiver(arrow_x, arrow_y, arrow_u, arrow_v, color="black", alpha=0.75, width=0.003, scale=22)

    unique_clusters = len(set(labels.tolist()) - {-1})
    noise_rate = float(noise_mask.mean()) if len(labels) else 0.0
    plt.title(f"{route_name} embedding direction field, sample={len(labels)}, clusters={unique_clusters}, noise={noise_rate:.1%}")
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="best", fontsize=8)
    save_figure(OUTPUT_DIR / f"{route['key']}_vector_field.jpg")


def main() -> int:
    """脚本入口。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, Any]] = {}
    cluster_rows: dict[str, list[dict[str, str]]] = {}
    example_metadata = read_metadata_dicts(ROUTES["ArcFace"]["metadata_file"])

    for name, route in tqdm(ROUTES.items(), desc="load metrics", unit="route"):
        metrics[name] = {
            "embedding": load_json(route["embedding_benchmark"]),
            "faiss": load_json(route["faiss_benchmark"]),
            "dbscan": load_json(route["dbscan_report"]),
        }
        cluster_rows[name] = read_cluster_labels(route["cluster_labels"])

    plot_celeba_identity_examples(example_metadata)
    plot_hog_feature_grid(example_metadata)
    plot_quality_comparison(metrics)
    plot_faiss_comparison(metrics)
    plot_dbscan_comparison(metrics)
    for name, route in ROUTES.items():
        plot_cluster_distribution(name, cluster_rows[name])
        plot_vector_field(name, cluster_rows[name], route)

    print(f"Report figures written to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
