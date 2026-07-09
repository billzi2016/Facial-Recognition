# PRD: 数据集下载与样本准备

## 背景

本实验需要一个可复现的人脸数据集，用于对比 HOG/dlib 与 InsightFace 预训练推理路线在人脸检测、向量编码、检索和聚类任务中的表现。首选 CelebA 已对齐图片集，也可保留少量未对齐或复杂姿态图片用于检测鲁棒性测试。

## 目标

- 下载或接入 CelebA 数据集。
- 建立统一的数据目录结构。
- 将全量 CelebA 图片纳入主实验流程。
- 允许额外生成小规模 debug subset，但 debug subset 只能用于快速验证，不能替代主实验。
- 生成身份标签、图片清单与实验 split。
- 保留正常正脸、侧脸、歪头、强光、暗光、遮挡等样本类型。
- 确保全量数据都有明确去向，不保留一大块完全未使用图片。

## 非目标

- 不在本 PRD 中训练模型。
- 不在本 PRD 中做向量检索。
- 不在本 PRD 中做聚类或可视化。

## 输入

- `img_align_celeba.zip`
- CelebA identity annotation 文件，例如 `identity_CelebA.txt`
- 可选属性文件，例如 `list_attr_celeba.txt`
- 可选 bbox 文件，例如 `list_bbox_celeba.txt`

## 输出

- `data/raw/celeba/`
- `data/processed/celeba_subset/`
- `data/processed/celeba_full/`
- `data/manifests/images.csv`
- `data/manifests/identities.csv`
- `data/manifests/splits.csv`
- `data/manifests/quality_tags.csv`
- `scripts/download_celeba.py`
- `scripts/prepare_celeba_manifests.py`

## 目录建议

```text
data/
  raw/
    celeba/
      img_align_celeba/
      identity_CelebA.txt
  processed/
    celeba_full/
      gallery/
      query_known/
      query_unknown/
      cluster_mix/
    celeba_subset/
      debug/
  manifests/
    images.csv
    identities.csv
    splits.csv
    quality_tags.csv
scripts/
  download_celeba.py
  prepare_celeba_manifests.py
```

## 脚本要求

下载与准备流程必须写成可复用脚本文件：

- `scripts/download_celeba.py`
- `scripts/prepare_celeba_manifests.py`

禁止把核心流程写成：

```bash
python -c "..."
```

原因：

- `python -c` 不利于复现。
- 不方便记录参数和日志。
- 不方便后续扩展断点续传、校验、重跑与错误恢复。

## 核心流程

1. 校验数据文件是否存在。
2. 解压图片集到 `data/raw/celeba/img_align_celeba/`。
3. 读取 identity annotation，建立 `image_id -> person_id` 映射。
4. 按 person_id 统计每个人图片数量。
5. 对全量图片做合法性检查。
6. 将全量图片划分到 gallery、query_known、query_unknown、cluster_mix 等 split。
7. 可选额外生成 debug subset，用于脚本 smoke test。
8. 生成 manifest CSV。
9. 对异常图片、损坏图片、非 RGB 图片做记录。
10. 输出 split 统计，确认没有大规模未使用数据。

所有扫描、校验、解压后索引和 manifest 生成步骤必须使用 `tqdm` 展示进度。

## Split 设计

- `gallery`: 用于建立基准向量库。
- `query_known`: 数据库中已有身份的新图片，用于熟人检索。
- `query_unknown`: 数据库外身份图片，用于陌生人拒识。
- `cluster_mix`: 混合图片集合，用于 DBSCAN 无监督聚类。
- `debug`: 小规模快速验证集合，只能用于开发调试和 smoke test。

## 全量数据使用原则

- 主实验以全量 CelebA 为准。
- 全量图片必须被分配到某个主 split，或者被记录为明确排除原因。
- 排除原因必须写入 manifest，例如损坏、非图片、缺失标签、无法读取。
- debug subset 可以从全量 split 中派生，但不能让全量主实验缺席。
- benchmark 报告必须明确标注结果来自 full dataset 还是 debug subset。

## 验收标准

- 可以通过 manifest 找到每一张图片的本地路径。
- 每张图片有稳定的 `image_id`。
- 每张已知身份图片有 `person_id`。
- `query_unknown` 中的身份不出现在 `gallery`。
- split 脚本固定随机种子，保证可复现。
- 全量图片要么进入主实验 split，要么有明确排除原因。
- 下载和 manifest 生成逻辑以脚本文件形式存在，不依赖 `python -c`。

## 风险

- CelebA 下载地址可能需要手动授权或镜像。
- 图片数量过大时，解压和扫描耗时较长。
- 未对齐原图可能缺少统一 bbox，需要额外检测流程。
- 全量实验耗时明显高于 debug subset，需要支持断点续跑与进度日志。
