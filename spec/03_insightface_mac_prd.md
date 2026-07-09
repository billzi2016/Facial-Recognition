# PRD: InsightFace Mac 推理与 512 维编码实验

## 背景

InsightFace 提供现代人脸检测与识别能力。标准 InsightFace Python 包的推理后端是 ONNX Runtime，加载的是预训练 ONNX 模型，不是本实验里重新训练 PyTorch 模型。Mac Apple Silicon 上优先验证 ONNX Runtime 的 `CoreMLExecutionProvider`，不可用时 fallback 到 `CPUExecutionProvider`。实验路线使用预训练检测模型完成人脸检测，使用 ArcFace 提取 512 维向量，以评估现代深度学习推理路线在 Mac 上的性能和鲁棒性。

## 目标

- 使用 InsightFace 标准 API 对已对齐人脸图执行检测与 ArcFace 编码。
- 使用 ArcFace 提取 512 维人脸向量。
- 使用 InsightFace 提供的预训练模型，只做 inference，不做训练。
- 优先验证并使用 ONNX Runtime `CoreMLExecutionProvider`。
- 推理端使用单个受控 InsightFace 实例顺序处理图片，避免多个进程同时持有 ONNX 模型。
- 记录复杂姿态、侧脸、强弱光场景下的检测成功率。
- 输出可被 FAISS 使用的 512 维向量文件。

## 非目标

- 不训练新模型。
- 不把 InsightFace 标准 Python 包误写成 PyTorch DataLoader 推理管线。
- 不使用 pgvector。
- 不在本 PRD 中完成 DBSCAN 聚类。

## 输入

- `data/manifests/images.csv`
- `data/manifests/splits.csv`
- `data/processed/celeba_full/`

## 输出

- `outputs/insightface/detections.csv`
- `outputs/insightface/embeddings.h5`
- `outputs/insightface/embedding_metadata.csv`
- `outputs/insightface/failures.csv`
- `outputs/insightface/benchmark.json`

## 设备策略

优先级：

1. `CoreMLExecutionProvider`
2. `CPUExecutionProvider`

如果无法使用 CoreML provider，必须记录原因，例如 onnxruntime 构建不包含该 provider、模型算子不支持、模型运行错误或环境未启用。

如果后续单独引入 PyTorch 版本 ArcFace/SCRFD，才可以评估 `mps`。该路线必须作为独立实现记录，不能和标准 InsightFace ONNX Runtime 路线混写。

## 并发与喂入策略

- InsightFace `FaceAnalysis.get(img)` 是单图 API，不是原生 PyTorch `DataLoader` API。
- 全量脚本可以使用 `torch.utils.data.DataLoader` 或自定义 producer-consumer 队列做图片读取和预处理。
- 当前实现使用单个受控 InsightFace 实例顺序推理。
- 后续如需并行，loader worker 只做图片路径读取、图片解码、颜色格式转换、基础校验和 batch 组装。
- 模型推理端不能无节制多进程同时持有模型。
- `tqdm` 包裹全量图片处理循环，展示已处理图片数、处理速度和粗略 ETA。

## 核心流程

1. 检查 ONNX Runtime 可用 providers。
2. 优先使用 `CoreMLExecutionProvider` 初始化 InsightFace 模型，否则使用 `CPUExecutionProvider`。
3. 对每张图片执行人脸检测。
4. 对检测到的人脸提取 512 维 ArcFace embedding。
5. 对多脸图片默认保留面积最大的人脸。
6. 对检测失败或编码失败样本写入 failure 文件。
7. 保存向量矩阵和 metadata。
8. 输出 benchmark。

embedding 必须使用 `h5py` 存储为 HDF5，并且必须用
`h5py.create_dataset(..., compression="gzip", compression_opts=1)` 在写入 dataset 时直接启用内置 gzip 压缩，不允许先生成未压缩 `.h5` 再用外部 gzip 压缩文件。

## 指标

- `onnxruntime_available_providers`
- `coreml_available`
- `device_used`
- `embedding_dim = 512`
- `h5_compression = gzip`
- `h5_compression_opts = 1`
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
- CoreML provider 是否带来更好的批处理吞吐。
- 512 维 ArcFace embedding 是否在检索和聚类中更稳。

## 验收标准

- 能输出 512 维 embedding。
- embedding 文件格式为 `outputs/insightface/embeddings.h5`，dataset 名称为 `embeddings`。
- HDF5 dataset 使用 `compression="gzip"` 和 `compression_opts=1`。
- 每条 embedding 都能追溯原图、person_id、检测框和设备信息。
- ONNX Runtime provider 使用状态被明确记录。
- 明确记录使用的是预训练模型，不涉及训练。
- 全量脚本必须有 `tqdm` 进度条。
- 与 HOG 路线输出格式保持一致，方便后续 FAISS 和 DBSCAN 共用。

## 风险

- ONNX Runtime CoreML provider 对部分模型或算子可能 fallback，需记录实际 provider 和耗时。
- 如果只安装普通 `onnxruntime`，可能只能使用 CPU provider。
- 模型下载可能受网络影响，需要支持本地模型缓存。
