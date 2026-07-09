# PRD: HOG baseline embeddings

## Purpose

The HOG route is the traditional CPU baseline. It gives the lab a reference point: what happens if the project does not use the ArcFace embedding model and instead uses the older dlib based face recognition stack.

The baseline is not skipped after embedding extraction. It goes through the same FAISS and DBSCAN experiments as ArcFace so the comparison is based on the same dataset and the same downstream tasks.

## Goals

- Detect faces with `face_recognition.face_locations(..., model="hog")`.
- Extract 128 dimensional dlib embeddings.
- Use `CPU count - 2` worker processes by default.
- Write embeddings to HDF5 with `h5py`.
- Produce metadata, detection rows, failure rows, and benchmark statistics.

## Inputs

- `data/manifests/images.csv`
- aligned CelebA images referenced by the manifest

## Outputs

- `outputs/hog/embeddings.h5`
- `outputs/hog/embedding_metadata.csv`
- `outputs/hog/detections.csv`
- `outputs/hog/failures.csv`
- `outputs/hog/benchmark.json`

The HDF5 file contains a dataset named `embeddings`.

## Storage rule

Embedding storage uses HDF5 dataset compression directly:

```python
h5.create_dataset("embeddings", data=embeddings, compression="gzip", compression_opts=1)
```

The script must not write an uncompressed `.h5` file and then compress the whole file with an external gzip command. Dataset level compression keeps the file readable as HDF5 and allows later tools to load the dataset normally.

## Worker policy

The default worker count is:

```python
max(1, os.cpu_count() - 2)
```

This keeps two CPU cores free for the operating system and other work. A manual `--workers` option can override it for smoke tests or controlled benchmarks.

## Acceptance criteria

- The script can process a small limit for smoke testing.
- The HDF5 dataset has shape `(N, 128)`.
- The HDF5 dataset reports `compression == "gzip"` and `compression_opts == 1`.
- Failure cases are written to `failures.csv` instead of stopping the full run.
- The benchmark records CPU count, worker count, latency, success rate, and failure rate.
