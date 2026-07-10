# DBSCAN clustering and 2D visualization

The DBSCAN experiment asks a different question: without using identity labels, can the embeddings form useful groups of similar faces? Images that do not fit a stable group are labeled `-1`, which means noise.

This is not the same experiment as FAISS. FAISS searches a gallery for a query identity. DBSCAN groups a set of images by embedding similarity. They use the same embeddings, but they measure different behavior.

## Inputs

ArcFace route:

```text
outputs/insightface/embeddings.h5
outputs/insightface/embedding_metadata.csv
```

HOG baseline:

```text
outputs/hog/embeddings.h5
outputs/hog/embedding_metadata.csv
```

Both routes must run. The report compares cluster count, noise rate, and plot structure between ArcFace and HOG.

## Run

ArcFace final clustering and 2D plots, using cosine:

```bash
python3 experiments/dbscan/run_dbscan_experiment.py \
  --route insightface \
  --metric cosine \
  --eps 0.56
```

HOG final clustering and 2D plots, using euclidean:

```bash
python3 experiments/dbscan/run_dbscan_experiment.py \
  --route hog \
  --metric euclidean \
  --eps 0.40
```

ArcFace uses cosine distance. HOG uses euclidean distance. Their embedding distributions are different, so DBSCAN parameters should not be shared blindly.

ArcFace + cosine sweep. This is the sweep used to choose the ArcFace final setting:

```bash
python3 experiments/dbscan/sweep_dbscan_eps.py \
  --route insightface \
  --metric cosine \
  --eps-values 0.30,0.34,0.38,0.42,0.45,0.48,0.52,0.56,0.60,0.64,0.68 \
  --output-dir outputs/dbscan/insightface_cosine_eps_sweep
```

ArcFace + euclidean sweep. This is recorded as a control, but it is not used because the tested range produced all noise:

```bash
python3 experiments/dbscan/sweep_dbscan_eps.py \
  --route insightface \
  --metric euclidean \
  --eps-values 0.40,0.60,0.80,1.00,1.20,1.40,1.60 \
  --output-dir outputs/dbscan/insightface_euclidean_eps_sweep
```

HOG + cosine sweep. This is recorded as a control, but it is not used because the tested range collapses into one large cluster:

```bash
python3 experiments/dbscan/sweep_dbscan_eps.py \
  --route hog \
  --metric cosine \
  --eps-values 0.12,0.16,0.20,0.24,0.28,0.32,0.36,0.40,0.44,0.48,0.52 \
  --output-dir outputs/dbscan/hog_cosine_eps_sweep
```

HOG + euclidean sweep. This is the sweep used to choose the HOG final setting:

```bash
python3 experiments/dbscan/sweep_dbscan_eps.py \
  --route hog \
  --metric euclidean \
  --eps-values 0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00 \
  --output-dir outputs/dbscan/hog_euclidean_eps_sweep
```

DBSCAN and t-SNE/UMAP can be slow on the full dataset. The script defaults to a controlled sample size. For full clustering, set:

```bash
python3 experiments/dbscan/run_dbscan_experiment.py \
  --route insightface \
  --max-samples 0
```

For a small check:

```bash
python3 experiments/dbscan/run_dbscan_experiment.py \
  --route insightface \
  --max-samples 500 \
  --plot-sample-size 200 \
  --output-dir outputs/dbscan_smoke/insightface
```

Remove smoke output after checking it.

## Outputs

Each route writes:

```text
outputs/dbscan/<route>/cluster_labels.csv
outputs/dbscan/<route>/pca_2d.jpg
outputs/dbscan/<route>/tsne_2d.jpg
outputs/dbscan/<route>/umap_2d.jpg
outputs/dbscan/<route>/cluster_report.json
```

Plots are JPG files, not PNG files.

## How to read the result

A valid run should have:

- one cluster label for each processed image
- noise points marked as `-1`
- PCA, t-SNE, and UMAP plots
- a report with sample count, cluster count, noise rate, and clustering metrics
- separate output directories for ArcFace and HOG
