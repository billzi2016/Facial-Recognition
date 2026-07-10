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

ArcFace 正式聚类和 2D 图，采用 cosine：

```bash
python3 experiments/dbscan/run_dbscan_experiment.py \
  --route insightface \
  --metric cosine \
  --eps 0.56
```

HOG 正式聚类和 2D 图，采用 euclidean：

```bash
python3 experiments/dbscan/run_dbscan_experiment.py \
  --route hog \
  --metric euclidean \
  --eps 0.40
```

ArcFace 使用 cosine 距离，HOG 使用 euclidean 距离。两条路线的 embedding 分布不同，DBSCAN 参数不能混用。

ArcFace + cosine 参数扫描，用于选择 ArcFace 正式参数：

```bash
python3 experiments/dbscan/sweep_dbscan_eps.py \
  --route insightface \
  --metric cosine \
  --eps-values 0.30,0.34,0.38,0.42,0.45,0.48,0.52,0.56,0.60,0.64,0.68 \
  --output-dir outputs/dbscan/insightface_cosine_eps_sweep
```

ArcFace + euclidean 参数扫描。这个组合也要记录，但当前结果全是噪声，最终不采用：

```bash
python3 experiments/dbscan/sweep_dbscan_eps.py \
  --route insightface \
  --metric euclidean \
  --eps-values 0.40,0.60,0.80,1.00,1.20,1.40,1.60 \
  --output-dir outputs/dbscan/insightface_euclidean_eps_sweep
```

HOG + cosine 参数扫描。这个组合也要记录，但当前结果会连成一个大簇，最终不采用：

```bash
python3 experiments/dbscan/sweep_dbscan_eps.py \
  --route hog \
  --metric cosine \
  --eps-values 0.12,0.16,0.20,0.24,0.28,0.32,0.36,0.40,0.44,0.48,0.52 \
  --output-dir outputs/dbscan/hog_cosine_eps_sweep
```

HOG + euclidean 参数扫描，用于选择 HOG 正式参数：

```bash
python3 experiments/dbscan/sweep_dbscan_eps.py \
  --route hog \
  --metric euclidean \
  --eps-values 0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00 \
  --output-dir outputs/dbscan/hog_euclidean_eps_sweep
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
