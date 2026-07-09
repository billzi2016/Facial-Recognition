# Facial Recognition

## 项目定位

本项目当前只做 **ArcFace 人脸特征提取、FAISS 检索、DBSCAN 聚类与可视化分析**。

当前前提是：输入数据已经完成了人脸检测、裁剪或对齐，因此主流程不再重复做人脸检测。

也就是说，当前主线不是：

```text
原图 -> RetinaFace 检测 -> ArcFace 编码
```

而是：

```text
已检测/已对齐人脸图
  -> ArcFace 提取 512 维 embedding
  -> FAISS 1:N 检索
  -> DBSCAN 无监督聚类
  -> PCA / t-SNE / UMAP 2D JPG 可视化
```

## 为什么当前只用 ArcFace

RetinaFace 的作用是找脸，ArcFace 的作用是认人。

本项目现在已经有可用的人脸输入，所以当前阶段只需要 ArcFace：

- 提取 512 维身份向量。
- 建立全量人脸向量库。
- 使用 FAISS 做熟人检索与陌生人拒识。
- 使用 DBSCAN 做智能相册式无监督聚类。
- 使用 PCA、t-SNE、UMAP 输出 2D 染色图。

## 实验模块

详细 PRD 在 [spec/](spec/) 目录。

当前核心模块：

- 数据集准备：全量数据进入实验 split，不保留大块未使用数据。
- ArcFace 编码：使用预训练模型做 inference，不训练模型。
- FAISS 检索：只使用 FAISS，不使用 pgvector。
- DBSCAN 聚类：对 ArcFace embedding 做无监督聚类。
- 2D 可视化：PCA、t-SNE、UMAP 三类图统一保存为 JPG。

HOG/dlib 传统路线可以作为实验对照组保留在 spec 中，但不是当前主线。

RetinaFace 或 SCRFD 检测路线只在后续需要处理未检测原图时再启用。

## HOG 对照实验

HOG 对照组脚本位于：

```bash
python3 experiments/hog/extract_hog_embeddings.py
```

该脚本使用 CPU 半数 worker 并行处理全量 manifest，输出到 `outputs/hog/`。HOG embedding 使用 `h5py` 存储为 HDF5，并通过 `h5py.create_dataset(..., compression="gzip", compression_opts=1)` 在写入 dataset 时直接启用内置 gzip 压缩。

## 数据集准备

如果本机已经配置 Kaggle token，例如 `~/.kaggle/access_token`，可以先用 Kaggle CLI 下载 CelebA：

```bash
python3 data/scripts/download_celeba.py --kaggle-dataset --extract
```

也可以指定本地 zip：

```bash
python3 data/scripts/download_celeba.py --zip-path /path/to/img_align_celeba.zip --extract
```

生成全量 manifest：

```bash
python3 data/scripts/prepare_celeba_manifests.py
```

如果当前 Kaggle 主数据集缺少身份标注，可以补充 `identity_CelebA.txt`：

```bash
python3 data/scripts/download_celeba_identity.py
python3 data/scripts/prepare_celeba_manifests.py
```

## 工程约束

- 下载、准备、编码、检索、聚类、绘图流程都要写成脚本文件，不能依赖 `python -c`。
- 所有长任务必须使用 `tqdm` 展示进度、速度和粗略 ETA。
- 绘图产物统一保存为 JPG，不使用 PNG。
- 主实验使用全量数据集；debug subset 只能用于快速验证。
- FAISS 是唯一向量检索方案。

## 输出目标

最终实验应产出：

- ArcFace 512 维 embedding。
- FAISS index。
- 熟人检索与陌生人拒识结果。
- DBSCAN 聚类标签。
- PCA / t-SNE / UMAP 三张 2D JPG 可视化图。
- benchmark 报告。
