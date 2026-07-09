# PRD: HOG CPU 人脸检测与 128 维编码实验

## 背景

HOG 是传统计算机视觉中的经典人脸检测方法。本实验将其作为 CPU 对照组，用于评估传统路线在正脸、轻微姿态变化、侧脸和复杂光照场景中的速度与鲁棒性。

## 目标

- 使用 `face_recognition.face_locations(image, model="hog")` 完成人脸检测。
- 使用 dlib 预训练模型提取 128 维人脸向量。
- 记录单图延迟、批量吞吐、检测成功率和漏检类型。
- 输出可被 FAISS 使用的 128 维向量文件。

## 非目标

- 不使用 CNN 检测模型。
- 不使用 MPS 或 GPU。
- 不在本 PRD 中完成 FAISS 检索逻辑。

## 输入

- `data/manifests/images.csv`
- `data/manifests/splits.csv`
- `data/processed/celeba_subset/`

## 输出

- `outputs/hog/detections.csv`
- `outputs/hog/embeddings.npy`
- `outputs/hog/embedding_metadata.csv`
- `outputs/hog/failures.csv`
- `outputs/hog/benchmark.json`

## 核心实现

```python
import face_recognition

image = face_recognition.load_image_file(image_path)
boxes = face_recognition.face_locations(image, model="hog")
encodings = face_recognition.face_encodings(image, known_face_locations=boxes)
```

## 实验步骤

1. 读取 manifest。
2. 对每张图片执行 HOG 人脸检测。
3. 记录检测框数量、检测耗时、图片尺寸。
4. 对检测到的人脸提取 128 维编码。
5. 对多脸图片记录规则：默认保留面积最大的人脸。
6. 对无脸图片写入 `failures.csv`。
7. 保存向量矩阵和 metadata。
8. 汇总 benchmark。

## 指标

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
- `embedding_metadata.csv` 可追溯到原图和 person_id。
- 无脸、多脸、编码失败都被记录。
- benchmark 文件包含硬件环境、参数和耗时统计。

## 风险

- `face_recognition` 和 `dlib` 在 Apple Silicon 上安装可能需要额外编译依赖。
- HOG 对侧脸和大角度旋转不鲁棒。
- 多脸图片会影响身份标签，需要明确选择策略。
