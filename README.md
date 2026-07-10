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
python3 experiments/dbscan/run_dbscan_experiment.py --route insightface --metric cosine --eps 0.56
python3 experiments/dbscan/run_dbscan_experiment.py --route hog --metric euclidean --eps 0.40
```

DBSCAN sweeps cover both routes and both distance metrics. The final runs use ArcFace + cosine and HOG + euclidean. ArcFace + euclidean and HOG + cosine are still recorded as controls so the report can explain why they are not used.

```bash
python3 experiments/dbscan/sweep_dbscan_eps.py \
  --route insightface \
  --metric cosine \
  --eps-values 0.30,0.34,0.38,0.42,0.45,0.48,0.52,0.56,0.60,0.64,0.68 \
  --output-dir outputs/dbscan/insightface_cosine_eps_sweep

python3 experiments/dbscan/sweep_dbscan_eps.py \
  --route insightface \
  --metric euclidean \
  --eps-values 0.40,0.60,0.80,1.00,1.20,1.40,1.60 \
  --output-dir outputs/dbscan/insightface_euclidean_eps_sweep

python3 experiments/dbscan/sweep_dbscan_eps.py \
  --route hog \
  --metric cosine \
  --eps-values 0.12,0.16,0.20,0.24,0.28,0.32,0.36,0.40,0.44,0.48,0.52 \
  --output-dir outputs/dbscan/hog_cosine_eps_sweep

python3 experiments/dbscan/sweep_dbscan_eps.py \
  --route hog \
  --metric euclidean \
  --eps-values 0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00 \
  --output-dir outputs/dbscan/hog_euclidean_eps_sweep
```

Experiment notes:

- [experiments/faiss/README.md](https://github.com/billzi2016/Facial-Recognition/blob/main/experiments/faiss/README.md)
- [experiments/dbscan/README.md](https://github.com/billzi2016/Facial-Recognition/blob/main/experiments/dbscan/README.md)

## More documentation

The full documentation source is in `docs-site/`. Chinese source documents use `.zh.md`; English files use the plain `.md` name.

PRD files live in [spec/README.md](https://github.com/billzi2016/Facial-Recognition/blob/main/spec/README.md).
