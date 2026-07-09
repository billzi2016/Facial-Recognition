# Facial Recognition

Documentation site: https://billzi2016.github.io/Facial-Recognition/

This project builds a face recognition lab around aligned CelebA face images. The main path extracts ArcFace embeddings, stores them in HDF5, searches them with FAISS, and groups them with DBSCAN. The HOG/dlib path runs as a baseline through the same FAISS and DBSCAN steps so the report can compare both routes.

The input images are already aligned faces, so the main workflow does not need a separate face detector. RetinaFace or SCRFD can be added later if the project starts from raw, undetected images.

## What the project does

The project turns each aligned face image into a numeric identity vector. Those vectors are then used for lookup and grouping:

- ArcFace creates 512 dimensional embeddings for the main workflow.
- FAISS searches the embedding database for known and unknown identities.
- DBSCAN groups unlabeled face photos into clusters.
- PCA, t-SNE, and UMAP provide two dimensional views of the clusters.
- HOG and dlib run as a CPU baseline. They use the same downstream FAISS and DBSCAN experiments, which makes the comparison measurable rather than anecdotal.

## Dataset

The dataset scripts live under `data/scripts/`.

Download and extract CelebA from Kaggle:

```bash
python3 data/scripts/download_celeba.py --kaggle-dataset --extract
```

Add identity labels when the main Kaggle package does not include them:

```bash
python3 data/scripts/download_celeba_identity.py
python3 data/scripts/prepare_celeba_manifests.py
```

The manifest files describe which images belong to gallery, known query, and unknown query splits. Generated data and manifests are ignored by git.

## ArcFace workflow

Run the main embedding extractor:

```bash
python3 experiments/insightface/extract_arcface_embeddings.py
```

The script uses InsightFace with ONNX Runtime. On Apple Silicon it prefers `CoreMLExecutionProvider`, then falls back to CPU if needed. Embeddings are stored with `h5py` in HDF5 format using dataset level gzip compression:

```python
h5.create_dataset("embeddings", data=embeddings, compression="gzip", compression_opts=1)
```

## HOG baseline

Run the CPU baseline:

```bash
python3 experiments/hog/extract_hog_embeddings.py
```

The HOG script leaves two CPU cores free by default and writes 128 dimensional dlib embeddings to HDF5 with the same gzip setting.

## Shared FAISS and DBSCAN experiments

FAISS search and DBSCAN clustering must run for both embedding sources:

```text
outputs/insightface/embeddings.h5
outputs/hog/embeddings.h5
```

ArcFace is the main route because it is the model intended for the project. HOG is the baseline. Both routes produce their own search results, clustering results, plots, and report sections so the final benchmark can show what changes when the embedding model changes.

FAISS search:

```bash
python3 experiments/faiss/run_faiss_experiment.py --route insightface
python3 experiments/faiss/run_faiss_experiment.py --route hog
```

DBSCAN clustering and plots:

```bash
python3 experiments/dbscan/run_dbscan_experiment.py --route insightface
python3 experiments/dbscan/run_dbscan_experiment.py --route hog
```

Experiment notes:

- [experiments/faiss/README.md](https://github.com/billzi2016/Facial-Recognition/blob/main/experiments/faiss/README.md)
- [experiments/dbscan/README.md](https://github.com/billzi2016/Facial-Recognition/blob/main/experiments/dbscan/README.md)

## More documentation

The full documentation source is in `docs-site/`. Chinese source documents use `.zh.md`; English files use the plain `.md` name.

PRD files live in [spec/README.md](https://github.com/billzi2016/Facial-Recognition/blob/main/spec/README.md).
