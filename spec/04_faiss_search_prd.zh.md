# PRD: FAISS 1:N 向量检索实验

## 背景

人脸识别中的 1:N 检索需要在向量库中快速查找最相近身份。本实验只使用 FAISS 作为向量检索方案。FAISS 必须同时跑 InsightFace ArcFace 512 维主路线和 HOG/dlib 128 维 baseline，对比两种 embedding 在熟人检索和陌生人拒识中的表现。

## 目标

- 分别构建 HOG 128 维 FAISS 索引和 ArcFace 512 维 FAISS 索引。
- 完成熟人 query 的 Top-K 检索。
- 完成陌生人 query 的阈值拒识测试。
- 输出检索延迟、Top-1 准确率、Top-K 命中率和拒识率。
- 分别输出 ArcFace 主路线 report 和 HOG baseline report，并在综合报告中做对比。

## 非目标

- 不使用 pgvector。
- 不引入数据库服务。
- 不训练 embedding 模型。

## 输入

- `outputs/hog/embeddings.h5`
- `outputs/hog/embedding_metadata.csv`
- `outputs/insightface/embeddings.h5`
- `outputs/insightface/embedding_metadata.csv`
- `data/manifests/splits.csv`

## 输出

- `outputs/faiss/hog_128.index`
- `outputs/faiss/insightface_512.index`
- `outputs/faiss/hog_search_results.csv`
- `outputs/faiss/insightface_search_results.csv`
- `outputs/faiss/threshold_report.csv`
- `outputs/faiss/benchmark.json`

## 索引策略

全量主实验的基线索引优先使用精确检索：

```text
IndexFlatL2
```

如果向量已做 L2 normalize，也可以使用：

```text
IndexFlatIP
```

索引选择必须记录在 benchmark 中。

## 实验步骤

1. 读取 gallery split 的 embedding。
2. 构建 HOG 128 维索引。
3. 构建 ArcFace 512 维索引。
4. 读取 `query_known` 执行熟人检索。
5. 读取 `query_unknown` 执行陌生人拒识。
6. 对不同阈值进行扫描。
7. 输出 Top-K 命中率、错误匹配样本和拒识结果。

索引构建、query 检索和阈值扫描都必须使用 `tqdm` 展示进度。

## 阈值建议

- HOG 128 维路线初始阈值：`0.50`
- ArcFace 512 维路线初始阈值：`0.45`

最终阈值应通过实验扫描确定，不应只依赖经验值。

## 指标

- `top1_accuracy`
- `top5_accuracy`
- `false_accept_rate`
- `false_reject_rate`
- `unknown_reject_rate`
- `search_latency_ms_p50`
- `search_latency_ms_p95`
- `index_build_time_ms`
- `index_size_mb`

## 验收标准

- 两条路线分别生成独立 FAISS index 文件。
- query 结果能追溯到原图、预测身份、真实身份、距离或相似度。
- 陌生人样本必须单独统计。
- 阈值扫描结果可用于实验报告画曲线。
- ArcFace 和 HOG 两条路线都必须完成，不允许只跑主路线后省略 baseline。

## 风险

- L2 距离和余弦相似度不能混用，需要明确 normalize 策略。
- gallery 中同一身份多张照片时，需要定义身份聚合策略。
- query_known 不能直接复用 gallery 的同一张图片，否则结果虚高。
