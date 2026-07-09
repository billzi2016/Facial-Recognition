#!/usr/bin/env python3
"""下载并整理 CelebA 身份标注文件。

脚本意图：
- 当前 Kaggle 主数据集 `jessicali9530/celeba-dataset` 只包含图片、属性、bbox、landmarks 和 split，
  不包含身份标注，因此只能生成无 `person_id` 的 manifest。
- 本脚本专门补充 `identity_CelebA.txt`，让后续 ArcFace + FAISS 能做熟人检索和陌生人拒识评估。
- 默认从 Kaggle 数据集 `kymo9890/identity-celeba` 下载小型身份标注包。
- 下载后自动解压、查找身份文件、规范化保存到 `data/raw/celeba/identity_CelebA.txt`。
- 默认删除下载 zip，避免仓库目录里长期保留无用压缩包。

安全约束：
- 不读取、不打印、不复制用户的 Kaggle token。
- 认证完全交给本机 Kaggle CLI，例如 `~/.kaggle/access_token`。
- 真实下载文件位于 `data/raw/celeba/`，由 `.gitignore` 忽略，不进入 git。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

from tqdm import tqdm


DEFAULT_DATASET = "kymo9890/identity-celeba"
DEFAULT_OUTPUT_DIR = Path("data/raw/celeba")
DEFAULT_TARGET_NAME = "identity_CelebA.txt"
IDENTITY_NAME_HINTS = ("identity", "celeba")


def run_kaggle_download(dataset: str, output_dir: Path, force: bool) -> Path:
    """调用 Kaggle CLI 下载身份标注包，并返回最新 zip 路径。

    这里不直接操作 token。只调用 `kaggle datasets download`，让 Kaggle CLI 自己读取本机认证配置。
    使用最新 zip 的原因是不同镜像数据集的 zip 文件名可能不同，不能硬编码文件名。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    command = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(output_dir)]
    if force:
        command.append("--force")

    print("Run Kaggle identity download:")
    print(" ".join(command))
    subprocess.run(command, check=True)

    zip_files = sorted(output_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not zip_files:
        raise FileNotFoundError(f"Kaggle download finished but no zip file was found in {output_dir}")
    return zip_files[0]


def extract_zip(zip_path: Path, extract_dir: Path, force: bool) -> None:
    """解压身份标注 zip。

    身份标注包很小，但仍然用 `tqdm` 展示进度，保持所有数据准备脚本的行为一致。
    如果目标文件已经存在且没有 `--force`，仍允许解压跳过，由后续规范化步骤决定是否覆盖。
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        members = archive.infolist()
        for member in tqdm(members, desc=f"extract {zip_path.name}", unit="file"):
            target = extract_dir / member.filename
            if target.exists() and not force:
                continue
            archive.extract(member, extract_dir)


def looks_like_identity_file(path: Path) -> bool:
    """判断一个文件是否可能是 CelebA 身份标注。

    先用文件名做快速过滤，再读取少量内容验证格式。典型格式是：
    `000001.jpg 2880`
    """
    name = path.name.lower()
    if not all(hint in name for hint in IDENTITY_NAME_HINTS):
        return False

    try:
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                parts = line.strip().replace(",", " ").split()
                if len(parts) >= 2 and parts[0].lower().endswith(".jpg"):
                    return True
    except UnicodeDecodeError:
        return False
    return False


def find_identity_file(search_dir: Path) -> Path:
    """在解压目录中查找身份标注文件。

    不假设 Kaggle 镜像内部路径固定，只要文件名和内容看起来像身份标注，就可以识别。
    如果找不到，明确报错，让用户知道需要换数据集 slug 或手动提供文件。
    """
    candidates = sorted(path for path in search_dir.rglob("*") if path.is_file())
    for candidate in candidates:
        if looks_like_identity_file(candidate):
            return candidate

    raise FileNotFoundError(
        f"No CelebA identity file was found under {search_dir}. "
        "Try another Kaggle dataset slug or provide identity_CelebA.txt manually."
    )


def normalize_identity_file(src: Path, dst: Path, force: bool) -> Path:
    """把找到的身份文件复制为项目约定文件名。

    后续 `prepare_celeba_manifests.py` 会自动查找 `data/raw/celeba/identity_CelebA.txt`。
    因此这里统一落到该文件名，避免每个下游脚本都处理各种镜像命名差异。
    """
    if dst.exists() and not force:
        print(f"Found existing identity file, skip overwrite: {dst}")
        return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def remove_zip(zip_path: Path, keep_zip: bool) -> None:
    """按项目约定删除下载 zip。

    用户之前明确希望 zip 包存在就删掉；保留 `--keep-zip` 只是给排错场景留出口。
    """
    if keep_zip:
        print(f"Keep zip file: {zip_path}")
        return
    zip_path.unlink(missing_ok=True)
    print(f"Removed zip file: {zip_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Download and normalize CelebA identity annotation.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help=f"Kaggle dataset slug. Default: {DEFAULT_DATASET}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-name", default=DEFAULT_TARGET_NAME)
    parser.add_argument("--force", action="store_true", help="Redownload and overwrite existing identity file.")
    parser.add_argument("--keep-zip", action="store_true", help="Keep downloaded zip for debugging.")
    return parser.parse_args()


def main() -> int:
    """脚本入口：下载、解压、定位身份文件、规范化文件名、删除 zip。"""
    args = parse_args()
    target_path = args.output_dir / args.target_name

    zip_path = run_kaggle_download(args.dataset, args.output_dir, args.force)
    extract_zip(zip_path, args.output_dir, args.force)
    identity_path = find_identity_file(args.output_dir)
    normalized_path = normalize_identity_file(identity_path, target_path, args.force)
    remove_zip(zip_path, args.keep_zip)

    print(f"Identity file ready: {normalized_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
