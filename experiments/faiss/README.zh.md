# FAISS 检索实验

FAISS 实验回答的是一个明确的问题：给一张 query 人脸，系统能不能在 gallery 向量库里找到同一个人。如果 query 是陌生人，最近的结果也不应该被直接当成已知身份，而是要通过阈值触发拒识。

这个实验和 DBSCAN 不同。FAISS 是有目标的 1:N 检索，使用 `gallery`、`query_known` 和 `query_unknown`。DBSCAN 是无监督聚类，不使用 gallery 查询结构。

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

两条路线都必须跑。ArcFace 是主路线，HOG 是 baseline。报告需要把两者放在一起比较。

## 运行

ArcFace：

```bash
python3 experiments/faiss/run_faiss_experiment.py --route insightface
```

HOG：

```bash
python3 experiments/faiss/run_faiss_experiment.py --route hog
```

小样本验证可以限制 query 数量：

```bash
python3 experiments/faiss/run_faiss_experiment.py \
  --route insightface \
  --max-known 100 \
  --max-unknown 100 \
  --output-dir outputs/faiss_smoke/insightface
```

smoke 输出验证完要删除，不要长期保留测试目录。

## 输出

每条路线会输出：

```text
outputs/faiss/<route>/index.faiss
outputs/faiss/<route>/known_search_results.csv
outputs/faiss/<route>/unknown_search_results.csv
outputs/faiss/<route>/threshold_report.csv
outputs/faiss/<route>/benchmark.json
```

`known_search_results.csv` 用来计算 Top-1 和 Top-K 命中率。`unknown_search_results.csv` 和 `threshold_report.csv` 用来分析陌生人拒识。

## 结果判断

正常结果应该满足：

- gallery 数量大于 0。
- known query 有 Top-1 和 Top-K 结果。
- unknown query 单独统计，不混进 known accuracy。
- benchmark 记录 embedding 维度、index 类型、score 方式和耗时。
- ArcFace 和 HOG 都有独立输出目录。
