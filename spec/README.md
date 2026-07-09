# Facial Recognition Lab PRD Index

## 目标

本目录用于沉淀 Mac Apple Silicon 环境下的人脸识别与聚类实验 PRD。实验对比传统 CPU 路线 HOG/dlib 与现代 MPS 加速路线 InsightFace，并使用 FAISS 完成 1:N 向量检索，使用 DBSCAN 与 2D 降维图完成智能相册聚类分析。

## PRD 文件列表

1. [01_dataset_download_prd.md](01_dataset_download_prd.md)
   - CelebA 数据集下载、抽样、目录结构、身份标签整理与数据清洗。

2. [02_hog_face_recognition_prd.md](02_hog_face_recognition_prd.md)
   - HOG CPU 人脸检测、dlib 128 维人脸编码、性能与漏检率评估。

3. [03_insightface_mps_prd.md](03_insightface_mps_prd.md)
   - InsightFace RetinaFace 检测、ArcFace 512 维特征、MPS 加速路线评估。

4. [04_faiss_search_prd.md](04_faiss_search_prd.md)
   - 基于 FAISS 的 1:N 检索实验，不使用 pgvector。

5. [05_dbscan_2d_visualization_prd.md](05_dbscan_2d_visualization_prd.md)
   - DBSCAN 聚类实验，以及 PCA、t-SNE、UMAP 三种 2D 可视化染色图。

6. [06_benchmark_report_prd.md](06_benchmark_report_prd.md)
   - 综合评测报告，包括速度、准确率、鲁棒性、聚类纯度和实验结论。

## 推荐执行顺序

1. 先完成数据集下载和样本目录标准化。
2. 分别跑通 HOG 路线和 InsightFace MPS 路线。
3. 将两条路线的人脸向量写入独立 FAISS 索引。
4. 做熟人检索、陌生人拒识和阈值分析。
5. 做 DBSCAN 聚类，并输出三类 2D 降维图。
6. 汇总 benchmark 报告。

## 统一约束

- 检索方案只使用 FAISS，不使用 pgvector。
- HOG 路线作为传统 CPU 对照组。
- InsightFace 路线优先验证 MPS 可用性，不能使用 MPS 时需要记录 fallback 原因。
- 所有实验产物必须保留可复现实验参数。
- 主实验必须使用全量数据集。允许建立 debug subset 做快速验证，但最终报告不能只基于 subset。
- 全量数据可以被划分到不同 split，但不能长期保留一大块完全未使用数据。
- 下载、解压、manifest 生成等流程必须落地为脚本文件，不能依赖 `python -c` 一次性命令。
- 所有绘图产物统一保存为 JPG，避免 PNG 文件过大。
