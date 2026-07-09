#!/usr/bin/env python3
"""生成 CelebA 全量数据集 manifest。

脚本意图：
- 扫描 Kaggle/CelebA 解压后的全量对齐人脸图片。
- 读取身份标注文件，并生成 images、identities、splits、quality_tags 四类 CSV。
- 主实验必须使用全量数据；debug subset 只能从这些 manifest 派生，不能替代主实验。
- 所有全量扫描和写盘动作都使用 `tqdm`，方便观察大致速度和进度。
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Kaggle 的 CelebA 包解压后通常是 data/raw/celeba/img_align_celeba/img_align_celeba/*.jpg。
# 如果用户使用其他镜像，可以通过 --image-dir 覆盖。
DEFAULT_IMAGE_DIR = Path("data/raw/celeba/img_align_celeba/img_align_celeba")
DEFAULT_MANIFEST_DIR = Path("data/manifests")
DEFAULT_IDENTITY_CANDIDATES = [
    Path("data/raw/celeba/list_identity_celeba.csv"),
    Path("data/raw/celeba/identity_CelebA.txt"),
]


@dataclass(frozen=True)
class ImageRecord:
    """manifest 中一张图片的标准记录。

    excluded/exclude_reason 先保留字段，是为了后续加入坏图、缺标签、无法解码等排除逻辑时，
    不破坏 CSV schema。
    """
    image_id: str
    path: Path
    person_id: str | None
    split: str
    excluded: bool
    exclude_reason: str


def resolve_identity_path(identity_path: Path | None) -> Path | None:
    """确定实际使用的身份标注文件。

    用户显式传入路径时优先使用；否则按常见 CelebA/Kaggle 文件名自动探测。
    找不到身份文件也允许继续，因为聚类实验可以无标签运行，但检索和评估指标会受限。
    """
    if identity_path is None:
        for candidate in DEFAULT_IDENTITY_CANDIDATES:
            if candidate.exists():
                return candidate
        return None
    return identity_path if identity_path.exists() else None


def read_identity_map(identity_path: Path | None) -> dict[str, str]:
    """读取 `image_id -> person_id` 映射。

    同时兼容两类格式：
    - 空格分隔的官方 txt：`000001.jpg 2880`
    - Kaggle CSV：`image_id,identity`
    """
    if identity_path is None:
        print("Identity file not found; manifests will be generated without person_id labels.")
        return {}

    mapping: dict[str, str] = {}
    with identity_path.open("r", encoding="utf-8") as fin:
        first_line = fin.readline()
        fin.seek(0)

        if "," in first_line and "image" in first_line.lower():
            reader = csv.DictReader(fin)
            for row in tqdm(reader, desc="read identities", unit="row"):
                image_id = row.get("image_id") or row.get("image") or row.get("filename")
                person_id = row.get("identity") or row.get("person_id")
                if image_id and person_id:
                    mapping[image_id] = person_id
            return mapping

        for line in tqdm(fin, desc="read identities", unit="line"):
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 2 or parts[0].lower() in {"image_id", "image"}:
                continue
            mapping[parts[0]] = parts[1]
    return mapping


def scan_images(image_dir: Path) -> list[Path]:
    """递归扫描图片目录，返回按文件名排序的图片路径。

    注意：这里不做图像解码，只按扩展名筛选。解码校验会在后续编码脚本里做，
    这样 manifest 生成可以保持轻量。
    """
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    files = [path for path in image_dir.rglob("*") if path.is_file()]
    images: list[Path] = []
    for path in tqdm(files, desc="filter images", unit="file"):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return sorted(images)


def choose_unknown_people(person_to_images: dict[str, list[str]], ratio: float, seed: int) -> set[str]:
    """从全量身份中抽出一部分作为陌生人测试身份。

    这些身份只会进入 `query_unknown`，不会进入 gallery，从而保证拒识实验成立。
    """
    people = sorted(person_to_images)
    if not people or ratio <= 0:
        return set()
    rng = random.Random(seed)
    count = max(1, int(len(people) * ratio))
    count = min(count, len(people))
    return set(rng.sample(people, count))


def assign_splits(
    image_paths: list[Path],
    identity_map: dict[str, str],
    unknown_ratio: float,
    seed: int,
) -> list[ImageRecord]:
    """给全量图片分配实验 split。

    规则保持简单且可复现：
    - 未找到身份标签的图片进入 `cluster_mix`，可用于无监督聚类。
    - 被抽为 unknown 的身份进入 `query_unknown`。
    - 其余身份第一张进入 `gallery`，后续图片进入 `query_known`。

    这样全量图片都有去向，不会出现一大块数据完全不用。
    """
    person_to_images: dict[str, list[str]] = defaultdict(list)
    for path in image_paths:
        person_id = identity_map.get(path.name)
        if person_id is not None:
            person_to_images[person_id].append(path.name)

    unknown_people = choose_unknown_people(person_to_images, unknown_ratio, seed)
    per_person_seen: Counter[str] = Counter()
    records: list[ImageRecord] = []

    for path in tqdm(image_paths, desc="assign splits", unit="image"):
        image_id = path.name
        person_id = identity_map.get(image_id)
        excluded = False
        exclude_reason = ""

        if person_id is None:
            split = "cluster_mix"
        elif person_id in unknown_people:
            split = "query_unknown"
        else:
            per_person_seen[person_id] += 1
            split = "gallery" if per_person_seen[person_id] == 1 else "query_known"

        records.append(
            ImageRecord(
                image_id=image_id,
                path=path,
                person_id=person_id,
                split=split,
                excluded=excluded,
                exclude_reason=exclude_reason,
            )
        )

    return records


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    """写 CSV 并展示写入进度。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in tqdm(rows, desc=f"write {path.name}", unit="row"):
            writer.writerow(row)


def write_manifests(records: list[ImageRecord], manifest_dir: Path) -> None:
    """把内存中的全量记录落盘为四个 manifest 文件。"""
    image_rows = [
        {
            "image_id": record.image_id,
            "path": str(record.path),
            "person_id": record.person_id or "",
            "split": record.split,
            "excluded": str(record.excluded).lower(),
            "exclude_reason": record.exclude_reason,
        }
        for record in records
    ]
    write_csv(
        manifest_dir / "images.csv",
        ["image_id", "path", "person_id", "split", "excluded", "exclude_reason"],
        image_rows,
    )

    identity_counts: Counter[str] = Counter(record.person_id for record in records if record.person_id)
    identity_rows = [
        {"person_id": person_id, "image_count": count}
        for person_id, count in sorted(identity_counts.items(), key=lambda item: int(item[0]))
    ]
    write_csv(manifest_dir / "identities.csv", ["person_id", "image_count"], identity_rows)

    split_counts: Counter[str] = Counter(record.split for record in records)
    split_rows = [{"split": split, "image_count": count} for split, count in sorted(split_counts.items())]
    write_csv(manifest_dir / "splits.csv", ["split", "image_count"], split_rows)

    quality_rows = [
        {
            "image_id": record.image_id,
            "quality_tag": "unchecked",
            "note": "",
        }
        for record in records
    ]
    write_csv(manifest_dir / "quality_tags.csv", ["image_id", "quality_tag", "note"], quality_rows)


def parse_args() -> argparse.Namespace:
    """解析 manifest 生成参数。"""
    parser = argparse.ArgumentParser(description="Generate CelebA full-dataset manifests.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--identity-path", type=Path, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--unknown-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    """脚本入口：读取身份、扫描图片、分配 split、写 manifest。"""
    args = parse_args()
    identity_path = resolve_identity_path(args.identity_path)
    identity_map = read_identity_map(identity_path)
    image_paths = scan_images(args.image_dir)
    records = assign_splits(image_paths, identity_map, args.unknown_ratio, args.seed)
    write_manifests(records, args.manifest_dir)

    print(f"Images: {len(records)}")
    print(f"Manifest directory: {args.manifest_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
