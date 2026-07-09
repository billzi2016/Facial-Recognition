# PRD: DBSCAN 聚类与 2D 可视化实验

## 背景

智能相册场景中，用户通常不会提前提供身份标签。DBSCAN 可以基于人脸 embedding 做无监督聚类，并把无法归类的照片标记为噪声点 `-1`。本实验对比 HOG 128 维向量与 ArcFace 512 维向量的聚类纯度，并通过 PCA、t-SNE、UMAP 三种 2D 降维图观察聚类结构。

## 目标

- 对混合照片 embedding 执行 DBSCAN。
- 分别评估 HOG 128 维和 ArcFace 512 维聚类效果。
- 输出 PCA、t-SNE、UMAP 三张 2D 染色图。
- 分析 DBSCAN 标签为 `-1` 的噪声点。

## 非目标

- 不训练监督分类器。
- 不使用人工标签参与 DBSCAN 聚类过程。
- 不使用 pgvector。

## 输入

- `outputs/hog/embeddings.npy`
- `outputs/hog/embedding_metadata.csv`
- `outputs/insightface/embeddings.npy`
- `outputs/insightface/embedding_metadata.csv`
- `data/manifests/splits.csv`

## 输出

- `outputs/dbscan/hog_cluster_labels.csv`
- `outputs/dbscan/mps_cluster_labels.csv`
- `outputs/dbscan/hog_pca_2d.jpg`
- `outputs/dbscan/hog_tsne_2d.jpg`
- `outputs/dbscan/hog_umap_2d.jpg`
- `outputs/dbscan/mps_pca_2d.jpg`
- `outputs/dbscan/mps_tsne_2d.jpg`
- `outputs/dbscan/mps_umap_2d.jpg`
- `outputs/dbscan/cluster_report.json`

## 核心实现

```python
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

labels = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(embeddings)
points_pca = PCA(n_components=2, random_state=seed).fit_transform(embeddings)
points_tsne = TSNE(n_components=2, random_state=seed).fit_transform(embeddings)
points_umap = umap.UMAP(n_components=2, random_state=seed).fit_transform(embeddings)
```

## DBSCAN 参数建议

HOG 128 维：

- `eps`: `0.48` 到 `0.55`
- `min_samples`: `3` 到 `5`

ArcFace 512 维：

- `eps`: `0.40` 到 `0.48`
- `min_samples`: `3` 到 `5`

参数必须做扫描，不能只跑单点。

## 可视化要求

每条路线输出三张图：

1. PCA 2D scatter
2. t-SNE 2D scatter
3. UMAP 2D scatter

图像格式：

- 统一输出 JPG。
- 不输出 PNG，避免全量实验图像文件过大。
- JPG 保存时应记录质量参数，例如 `quality=90` 或项目约定值。

染色规则：

- 点颜色使用 DBSCAN cluster label。
- label `-1` 使用灰色或低透明度。
- 可选使用不同 marker 标记 known、unknown、blurred、landscape 等质量标签。
- 图标题必须包含路线、降维方法、eps、min_samples、样本数。

## 指标

如果 cluster_mix 有隐藏真实身份标签，可计算：

- `adjusted_rand_index`
- `normalized_mutual_info`
- `homogeneity`
- `completeness`
- `v_measure`
- `noise_rate`
- `cluster_count`

如果不使用真实标签，则至少输出：

- 聚类数量
- 噪声点数量
- 最大簇大小
- 单样本簇数量
- 每个簇的样例图片路径

## 验收标准

- HOG 和 ArcFace 两条路线都有独立聚类结果。
- 每条路线都有 PCA、t-SNE、UMAP 三张 2D 图。
- 图中的颜色来自 DBSCAN label，而不是人工身份标签。
- 噪声点 `-1` 被单独统计和抽样检查。

## 风险

- t-SNE 对随机种子和 perplexity 敏感，需要固定参数。
- UMAP 需要额外安装 `umap-learn`。
- 高维距离分布可能导致 DBSCAN eps 难以迁移，必须通过扫描选择。
