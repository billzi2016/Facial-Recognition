# PRD: InsightFace ArcFace embeddings

## Purpose

The ArcFace route is the main embedding route. The project starts from aligned CelebA face images, so the goal is to turn each face into a 512 dimensional identity vector that can be searched with FAISS and grouped with DBSCAN.

The implementation uses the standard InsightFace Python package. That package loads ONNX models and runs them through ONNX Runtime. On Apple Silicon, the preferred provider is `CoreMLExecutionProvider`; if it is not available, the script falls back to `CPUExecutionProvider`.

## Goals

- Use pretrained InsightFace models. No training happens in this lab.
- Extract 512 dimensional ArcFace embeddings.
- Prefer ONNX Runtime CoreML provider on Mac.
- Write embeddings to HDF5 with `h5py`.
- Produce metadata, detection rows, failure rows, and benchmark statistics.

## Inputs

- `data/manifests/images.csv`
- aligned CelebA images referenced by the manifest

## Outputs

- `outputs/insightface/embeddings.h5`
- `outputs/insightface/embedding_metadata.csv`
- `outputs/insightface/detections.csv`
- `outputs/insightface/failures.csv`
- `outputs/insightface/benchmark.json`

The HDF5 file contains a dataset named `embeddings`.

## Runtime behavior

`FaceAnalysis.get(img)` is a single image API. It is not a PyTorch DataLoader path and it does not use `model.to("mps")`. ONNX Runtime decides how to execute the ONNX models through the selected providers.

On this project, the common provider order is:

```text
CoreMLExecutionProvider
CPUExecutionProvider
```

CoreML can dispatch work to CPU, GPU, or the Apple Neural Engine. Seeing CoreML in the provider list means the script is not limited to plain CPU, but it does not prove that every operation ran on the Neural Engine.

## Storage rule

Embedding storage uses HDF5 dataset compression directly:

```python
h5.create_dataset("embeddings", data=embeddings, compression="gzip", compression_opts=1)
```

The script must not write an uncompressed `.h5` file and then compress the whole file externally.

## Acceptance criteria

- The script can process a small limit for smoke testing.
- The HDF5 dataset has shape `(N, 512)`.
- The HDF5 dataset reports `compression == "gzip"` and `compression_opts == 1`.
- The benchmark records selected ONNX Runtime providers.
- Failure cases are written to `failures.csv`.
- The output can feed both FAISS and DBSCAN.
