# DBSCAN 聚类与 2D 可视化实验

DBSCAN 实验回答的是另一个问题：不给身份标签，只给一批 embedding，算法能不能把相似人脸分到同一组。无法稳定归类的图片会被标记为 `-1`，也就是噪声点。

这个实验和 FAISS 不同。FAISS 是给 query 找 gallery 中的身份；DBSCAN 是把一堆图片自己分组。两者共享 embedding，但实验目的和指标不同。

## 输入

ArcFace 主路线：

```text
outputs/insightface/embeddings.h5
outputs/insightface/embedding_metadata.csv
```

HOG baseline：

```text
outputs/hog/embeddings.h5
outputs/hog/embedding_metadata.csv
```

两条路线都必须跑。报告需要比较 ArcFace 和 HOG 在聚类数量、噪声率和可视化结构上的差异。

## 运行

ArcFace：

```bash
python3 experiments/dbscan/run_dbscan_experiment.py --route insightface
```

HOG：

```bash
python3 experiments/dbscan/run_dbscan_experiment.py --route hog
```

DBSCAN 和 t-SNE/UMAP 在全量数据上可能很慢。脚本默认使用可控样本规模。需要全量聚类时，显式设置：

```bash
python3 experiments/dbscan/run_dbscan_experiment.py \
  --route insightface \
  --max-samples 0
```

小样本验证：

```bash
python3 experiments/dbscan/run_dbscan_experiment.py \
  --route insightface \
  --max-samples 500 \
  --plot-sample-size 200 \
  --output-dir outputs/dbscan_smoke/insightface
```

smoke 输出验证完要删除。

## 输出

每条路线会输出：

```text
outputs/dbscan/<route>/cluster_labels.csv
outputs/dbscan/<route>/pca_2d.jpg
outputs/dbscan/<route>/tsne_2d.jpg
outputs/dbscan/<route>/umap_2d.jpg
outputs/dbscan/<route>/cluster_report.json
```

图像统一是 JPG，不输出 PNG。

## 结果判断

正常结果应该满足：

- `cluster_labels.csv` 中每张图都有一个 cluster label。
- 噪声点用 `-1` 表示。
- PCA、t-SNE、UMAP 三张图都存在。
- report 记录样本数、聚类数量、噪声率和聚类指标。
- ArcFace 和 HOG 都有独立输出目录。
