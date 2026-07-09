# PRD: 综合 Benchmark 报告

## 背景

本实验最终需要形成一份可以复现实验过程、解释实验结果并支撑技术结论的 benchmark 报告。报告核心是对比传统 HOG CPU 路线与 InsightFace MPS 路线在人脸检测、向量编码、FAISS 检索和 DBSCAN 聚类中的差异。

## 目标

- 汇总所有实验输出。
- 对比 HOG 与 InsightFace/MPS 的速度、准确率、鲁棒性和聚类质量。
- 基于全量 CelebA 主实验给出结论，并明确记录总人数、总图片数和各 split 覆盖率。
- 产出图表和可复现实验参数。

## 非目标

- 不在报告阶段新增算法。
- 不修改前面实验的原始结果。
- 不根据结论反向调整数据集。

## 输入

- `outputs/hog/benchmark.json`
- `outputs/insightface/benchmark.json`
- `outputs/faiss/benchmark.json`
- `outputs/faiss/threshold_report.csv`
- `outputs/dbscan/cluster_report.json`
- `outputs/dbscan/*.jpg`

## 输出

- `reports/facial_recognition_lab_report.md`
- `reports/figures/`
- `reports/tables/`

## 报告结构

1. 实验摘要
2. 硬件与环境
3. 数据集与 split
4. HOG CPU 路线结果
5. InsightFace MPS 路线结果
6. FAISS 1:N 检索结果
7. DBSCAN 聚类与 2D 可视化结果
8. 错误案例分析
9. 结论与建议

## 必须包含的图表

- HOG vs InsightFace 检测延迟对比。
- HOG vs InsightFace 编码延迟对比。
- FAISS Top-1、Top-5 检索准确率对比。
- 陌生人拒识阈值扫描曲线。
- HOG PCA/t-SNE/UMAP 聚类图。
- ArcFace PCA/t-SNE/UMAP 聚类图。
- DBSCAN 噪声点样例图。

所有报告图像统一引用 JPG 文件，不使用 PNG。

## 必须包含的表格

- 数据集 split 统计。
- 全量数据覆盖统计，包括总图片数、总身份数、有效图片数、排除图片数和排除原因。
- 硬件与软件版本。
- 检测成功率与失败类型。
- embedding 维度与向量数量。
- FAISS index 类型、大小和构建耗时。
- 聚类数量、噪声率和聚类指标。

## 核心结论模板

报告最终应回答：

1. HOG 在全量主实验的正脸场景中是否足够快。
2. HOG 在侧脸、歪头、复杂光照下的漏检是否明显。
3. InsightFace/MPS 是否在复杂场景中更鲁棒。
4. 128 维 dlib embedding 和 512 维 ArcFace embedding 在 FAISS 检索中谁更稳定。
5. DBSCAN 对 128 维和 512 维向量的聚类纯度差异。
6. Apple Silicon MPS 是否值得作为本项目的主路线。

## 验收标准

- 报告中的每个数字都能追溯到实验输出文件。
- 图表路径真实存在。
- 环境版本和关键参数完整记录。
- 报告必须说明 full dataset 与 debug subset 的区别，主结论不能只来自 debug subset。
- 全量图片要么进入主实验 split，要么有明确排除原因和数量统计。
- 结论区分事实、观察和推断。

## 风险

- 如果 MPS fallback 到 CPU，报告必须如实说明。
- 如果全量数据 split 不均衡，聚类指标可能偏高或偏低。
- 如果阈值未扫描，拒识率结论不可靠。
