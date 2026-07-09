# PRD: 数据集下载与样本准备

## 背景

本实验需要一个可复现的人脸数据集，用于对比 HOG/dlib 与 InsightFace/MPS 两条路线在人脸检测、向量编码、检索和聚类任务中的表现。首选 CelebA 已对齐图片集，也可保留少量未对齐或复杂姿态图片用于检测鲁棒性测试。

## 目标

- 下载或接入 CelebA 数据集。
- 建立统一的数据目录结构。
- 抽样 1000 到 10000 人规模的数据子集。
- 生成身份标签、图片清单与实验 split。
- 保留正常正脸、侧脸、歪头、强光、暗光、遮挡等样本类型。

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
- `data/manifests/images.csv`
- `data/manifests/identities.csv`
- `data/manifests/splits.csv`
- `data/manifests/quality_tags.csv`

## 目录建议

```text
data/
  raw/
    celeba/
      img_align_celeba/
      identity_CelebA.txt
  processed/
    celeba_subset/
      train/
      gallery/
      query_known/
      query_unknown/
      cluster_mix/
  manifests/
    images.csv
    identities.csv
    splits.csv
    quality_tags.csv
```

## 核心流程

1. 校验数据文件是否存在。
2. 解压图片集到 `data/raw/celeba/img_align_celeba/`。
3. 读取 identity annotation，建立 `image_id -> person_id` 映射。
4. 按 person_id 统计每个人图片数量。
5. 过滤图片数量不足的身份。
6. 抽样目标身份数，例如 1000、5000、10000。
7. 划分 gallery、query_known、query_unknown、cluster_mix。
8. 生成 manifest CSV。
9. 对异常图片、损坏图片、非 RGB 图片做记录。

## Split 设计

- `gallery`: 用于建立基准向量库。
- `query_known`: 数据库中已有身份的新图片，用于熟人检索。
- `query_unknown`: 数据库外身份图片，用于陌生人拒识。
- `cluster_mix`: 混合图片集合，用于 DBSCAN 无监督聚类。

## 验收标准

- 可以通过 manifest 找到每一张图片的本地路径。
- 每张图片有稳定的 `image_id`。
- 每张已知身份图片有 `person_id`。
- `query_unknown` 中的身份不出现在 `gallery`。
- 抽样脚本固定随机种子，保证可复现。

## 风险

- CelebA 下载地址可能需要手动授权或镜像。
- 图片数量过大时，解压和扫描耗时较长。
- 未对齐原图可能缺少统一 bbox，需要额外检测流程。
