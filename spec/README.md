# Facial recognition lab PRD index

This directory describes the experiments behind the face recognition lab. The project starts from aligned CelebA face images, extracts face embeddings, and compares two embedding routes through the same search and clustering steps.

ArcFace is the main route. It produces 512 dimensional embeddings with a pretrained InsightFace model. HOG and dlib form the CPU baseline. They produce 128 dimensional embeddings and make it possible to measure how much the modern embedding model changes retrieval and clustering results.

The downstream experiments are separate:

- FAISS tests identity lookup. It answers whether a query face can find the right person in the gallery and whether an unknown person can be rejected.
- DBSCAN tests grouping. It answers whether unlabeled face embeddings can be grouped into albums and which images should be treated as noise.

Both experiments run on both embedding sources:

```text
outputs/insightface/embeddings.h5
outputs/hog/embeddings.h5
```

## PRD files

- [Dataset preparation](01_dataset_download_prd.md)
- [HOG baseline](02_hog_face_recognition_prd.md)
- [InsightFace ArcFace](03_insightface_mac_prd.md)
- [FAISS search](04_faiss_search_prd.md)
- [DBSCAN and 2D visualization](05_dbscan_2d_visualization_prd.md)
- [Benchmark report](06_benchmark_report_prd.md)

## Shared rules

- The full dataset is the main experiment input. A debug subset can be used for quick checks, but it cannot replace full results.
- Embeddings are stored in HDF5 with `h5py`.
- HDF5 datasets use `compression="gzip"` and `compression_opts=1` at dataset creation time.
- Plot files are JPG, not PNG.
- Long scripts show progress with `tqdm`.
- FAISS is the only vector search engine used in this lab.
