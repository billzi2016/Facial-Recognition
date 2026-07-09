# PRD: HOG CPU 人脸检测与 128 维编码实验

## 背景

HOG 是传统计算机视觉中的经典人脸检测方法。本实验将其作为 CPU 对照组，用于评估传统路线在正脸、轻微姿态变化、侧脸和复杂光照场景中的速度与鲁棒性。

## 目标

- 使用 `face_recognition.face_locations(image, model="hog")` 完成人脸检测。
- 使用 dlib 预训练模型提取 128 维人脸向量。
- 使用本机 CPU 总核心数的一半做多进程并行。
- 记录单图延迟、批量吞吐、检测成功率和漏检类型。
- 输出可被 FAISS 使用的 128 维 HDF5 向量文件。

## 非目标

- 不使用 CNN 检测模型。
- 不使用 MPS 或 GPU。
- 不在本 PRD 中完成 FAISS 检索逻辑。

## 输入

- `data/manifests/images.csv`
- `data/manifests/splits.csv`
- `data/processed/celeba_full/`

## 输出

- `outputs/hog/detections.csv`
- `outputs/hog/embeddings.h5`
- `outputs/hog/embedding_metadata.csv`
- `outputs/hog/failures.csv`
- `outputs/hog/benchmark.json`

## 核心实现

```python
import os
from concurrent.futures import ProcessPoolExecutor

import face_recognition
import h5py

image = face_recognition.load_image_file(image_path)
boxes = face_recognition.face_locations(image, model="hog")
encodings = face_recognition.face_encodings(image, known_face_locations=boxes)

workers = max(1, (os.cpu_count() or 2) // 2)
with ProcessPoolExecutor(max_workers=workers) as pool:
    ...

with h5py.File("outputs/hog/embeddings.h5", "w") as h5:
    h5.create_dataset("embeddings", data=embeddings, compression="gzip", compression_opts=1)
```

## 并行策略

- `face_recognition.face_locations(..., model="hog")` 单次调用不负责全量任务的多进程调度。
- HOG 全量脚本必须在脚本层使用 `ProcessPoolExecutor` 或等价多进程方案。
- worker 数量固定为 `max(1, os.cpu_count() // 2)`。
- 每个 worker 独立读取图片、检测人脸、提取 128 维编码。
- 主进程负责 `tqdm` 进度条、结果汇总、失败记录和最终写盘。
- 不使用全部 CPU，避免 Mac 桌面卡顿，并给系统、监控和其他实验保留资源。

## 实验步骤

1. 读取 manifest。
2. 对每张图片执行 HOG 人脸检测。
3. 记录检测框数量、检测耗时、图片尺寸。
4. 对检测到的人脸提取 128 维编码。
5. 对多脸图片记录规则：默认保留面积最大的人脸。
6. 对无脸图片写入 `failures.csv`。
7. 保存向量矩阵和 metadata。
8. 汇总 benchmark。

所有全量步骤必须使用 `tqdm` 展示进度和处理速度。

embedding 必须使用 `h5py` 存储为 HDF5，并且必须用
`h5py.create_dataset(..., compression="gzip", compression_opts=1)` 在写入 dataset 时直接启用内置 gzip 压缩，不允许先生成未压缩 `.h5` 再用外部 gzip 压缩文件。

## 指标

- `cpu_count`
- `worker_count`
- `worker_policy = half_cpu`
- `detect_latency_ms_p50`
- `detect_latency_ms_p95`
- `encode_latency_ms_p50`
- `encode_latency_ms_p95`
- `faces_detected_rate`
- `no_face_rate`
- `multi_face_rate`
- `embedding_dim = 128`

## 重点观察

- 正脸场景下 HOG 是否足够快。
- 歪头超过一定角度后是否明显漏检。
- 侧脸、遮挡、强光、暗光下的失败比例。
- CPU 占用与并发处理能力。

## 验收标准

- 能稳定输出 128 维 embedding。
- embedding 文件格式为 `outputs/hog/embeddings.h5`，dataset 名称为 `embeddings`。
- HDF5 dataset 使用 `compression="gzip"` 和 `compression_opts=1`。
- `embedding_metadata.csv` 可追溯到原图和 person_id。
- 无脸、多脸、编码失败都被记录。
- benchmark 文件包含硬件环境、参数和耗时统计。

## 风险

- `face_recognition` 和 `dlib` 在 Apple Silicon 上安装可能需要额外编译依赖。
- HOG 对侧脸和大角度旋转不鲁棒。
- 多脸图片会影响身份标签，需要明确选择策略。
