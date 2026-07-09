# PRD: InsightFace MPS 人脸检测与 512 维编码实验

## 背景

InsightFace 提供现代人脸检测与识别能力。实验路线使用 RetinaFace 完成人脸检测，使用 ArcFace 提取 512 维向量，并优先在 Apple Silicon 的 MPS 后端上运行，以评估现代深度学习路线在 Mac 上的性能和鲁棒性。

## 目标

- 使用 InsightFace 完成人脸检测。
- 使用 ArcFace 提取 512 维人脸向量。
- 优先验证并使用 PyTorch MPS 设备。
- 记录复杂姿态、侧脸、强弱光场景下的检测成功率。
- 输出可被 FAISS 使用的 512 维向量文件。

## 非目标

- 不训练新模型。
- 不使用 pgvector。
- 不在本 PRD 中完成 DBSCAN 聚类。

## 输入

- `data/manifests/images.csv`
- `data/manifests/splits.csv`
- `data/processed/celeba_subset/`

## 输出

- `outputs/insightface/detections.csv`
- `outputs/insightface/embeddings.npy`
- `outputs/insightface/embedding_metadata.csv`
- `outputs/insightface/failures.csv`
- `outputs/insightface/benchmark.json`

## 设备策略

优先级：

1. `mps`
2. `cpu`

如果无法使用 MPS，必须记录原因，例如依赖不支持、算子 fallback、模型运行错误或环境未启用。

## 核心流程

1. 检查 PyTorch MPS 是否可用。
2. 初始化 InsightFace 模型。
3. 对每张图片执行人脸检测。
4. 对检测到的人脸提取 512 维 ArcFace embedding。
5. 对多脸图片默认保留面积最大的人脸。
6. 对检测失败或编码失败样本写入 failure 文件。
7. 保存向量矩阵和 metadata。
8. 输出 benchmark。

## 指标

- `mps_available`
- `device_used`
- `detect_latency_ms_p50`
- `detect_latency_ms_p95`
- `encode_latency_ms_p50`
- `encode_latency_ms_p95`
- `faces_detected_rate`
- `no_face_rate`
- `multi_face_rate`
- `embedding_dim = 512`

## 对比重点

- 与 HOG 相比，侧脸和歪头场景是否更稳定。
- MPS 是否带来更好的批处理吞吐。
- 512 维 ArcFace embedding 是否在检索和聚类中更稳。

## 验收标准

- 能输出 512 维 embedding。
- 每条 embedding 都能追溯原图、person_id、检测框和设备信息。
- MPS 使用状态被明确记录。
- 与 HOG 路线输出格式保持一致，方便后续 FAISS 和 DBSCAN 共用。

## 风险

- InsightFace 部分运行时默认依赖 ONNX Runtime，MPS 支持路径需要实际验证。
- PyTorch MPS 对部分算子可能存在 fallback 或性能不稳定。
- 模型下载可能受网络影响，需要支持本地模型缓存。
