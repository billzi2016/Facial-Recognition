#!/usr/bin/env python3
"""下载、复制并可选解压 CelebA 数据集文件。

脚本意图：
- 把数据集准备动作固化成可复用脚本，避免临时 `python -c` 命令导致流程不可追踪。
- 支持三种来源：本地 zip、普通 URL、Kaggle CLI 数据集。
- 真实数据写入 `data/raw/celeba/`，由 `.gitignore` 忽略，不进入 git。
- 所有长耗时复制、下载、解压过程都通过 `tqdm` 显示大致进度。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from zipfile import ZipFile

from tqdm import tqdm


# 复制和普通 URL 下载都按块处理，避免一次性把 1GB+ 文件读入内存。
CHUNK_SIZE = 1024 * 1024
DEFAULT_OUTPUT_DIR = Path("data/raw/celeba")
DEFAULT_KAGGLE_DATASET = "jessicali9530/celeba-dataset"


def copy_with_progress(src: Path, dst: Path, force: bool) -> Path:
    """把本地 zip 复制到项目数据目录，并显示复制进度。

    这个函数用于用户已经手动下载好 CelebA zip 的情况。复制而不是移动，
    是为了不破坏用户原始下载文件的位置。
    """
    if not src.exists():
        raise FileNotFoundError(f"Source zip does not exist: {src}")
    if dst.exists() and not force:
        print(f"Found existing file, skip copy: {dst}")
        return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    total = src.stat().st_size
    with src.open("rb") as fin, dst.open("wb") as fout:
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"copy {src.name}") as bar:
            while True:
                chunk = fin.read(CHUNK_SIZE)
                if not chunk:
                    break
                fout.write(chunk)
                bar.update(len(chunk))
    return dst


def download_with_progress(url: str, dst: Path, force: bool) -> Path:
    """从普通 HTTP/HTTPS URL 下载 zip，并显示字节级进度。

    这里使用 Python 标准库 `urllib`，避免为了一个下载动作引入 requests 依赖。
    如果服务端不返回 Content-Length，tqdm 仍会显示已下载字节数，只是没有准确百分比。
    """
    if dst.exists() and not force:
        print(f"Found existing file, skip download: {dst}")
        return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        total = int(response.headers.get("Content-Length") or 0)
        with dst.open("wb") as fout:
            with tqdm(total=total, unit="B", unit_scale=True, desc=f"download {dst.name}") as bar:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    fout.write(chunk)
                    bar.update(len(chunk))
    return dst


def download_from_kaggle(dataset: str, output_dir: Path, force: bool) -> Path:
    """通过 Kaggle CLI 下载数据集，并返回最新下载到的 zip 路径。

    认证策略：
    - 不读取、不打印、不复制用户 token。
    - 直接调用本机 `kaggle datasets download`。
    - 由 Kaggle CLI 自己从 `~/.kaggle` 读取 `access_token` 或其他本地认证配置。

    这样可以避免项目代码接触凭据，也避免把任何 secret 写入仓库。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    command = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(output_dir)]
    if force:
        command.append("--force")

    print("Run Kaggle dataset download:")
    print(" ".join(command))
    print("Kaggle credentials are read by the Kaggle CLI from your local Kaggle config.")
    subprocess.run(command, check=True)

    # Kaggle CLI 的输出文件名通常由数据集 slug 决定，例如 celeba-dataset.zip。
    # 这里取目录下最新的 zip，兼容用户传入不同 Kaggle 数据集 slug 的情况。
    zip_files = sorted(output_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not zip_files:
        raise FileNotFoundError(f"Kaggle download finished but no zip file was found in {output_dir}")
    return zip_files[0]


def extract_zip(zip_path: Path, output_dir: Path, force: bool) -> None:
    """解压 zip，并用 marker 文件避免重复解压。

    CelebA 图片数量超过 20 万，重复解压会浪费时间和磁盘写入。
    `.extract_complete` 只表示当前脚本曾经完成过一次解压；如果需要重跑，
    使用 `--force` 明确覆盖这个保护。
    """
    marker = output_dir / ".extract_complete"
    if marker.exists() and not force:
        print(f"Found extraction marker, skip extract: {marker}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        members = archive.infolist()
        for member in tqdm(members, desc=f"extract {zip_path.name}", unit="file"):
            archive.extract(member, output_dir)
    marker.write_text(f"{zip_path.name}\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    三种数据来源互斥，防止同一次运行既下载又复制，导致输出文件难以判断。
    """
    parser = argparse.ArgumentParser(description="Prepare CelebA raw zip files.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zip-path", type=Path, help="Local CelebA zip file, for example img_align_celeba.zip.")
    source.add_argument("--download-url", help="Direct URL to download a CelebA zip file.")
    source.add_argument(
        "--kaggle-dataset",
        nargs="?",
        const=DEFAULT_KAGGLE_DATASET,
        help=f"Download with Kaggle CLI. Defaults to {DEFAULT_KAGGLE_DATASET}.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--filename", default="img_align_celeba.zip")
    parser.add_argument("--extract", action="store_true", help="Extract the zip after it is available.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing zip or rerun extraction.")
    return parser.parse_args()


def main() -> int:
    """脚本入口：准备 zip，按需解压，并打印最终文件位置。"""
    args = parse_args()
    destination = args.output_dir / args.filename

    # source group 保证三选一；这里按来源分发到对应实现。
    if args.zip_path:
        zip_path = copy_with_progress(args.zip_path.expanduser(), destination, args.force)
    elif args.kaggle_dataset:
        zip_path = download_from_kaggle(args.kaggle_dataset, args.output_dir, args.force)
    else:
        zip_path = download_with_progress(args.download_url, destination, args.force)

    if args.extract:
        extract_zip(zip_path, args.output_dir, args.force)

    print(f"Dataset file ready: {zip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
