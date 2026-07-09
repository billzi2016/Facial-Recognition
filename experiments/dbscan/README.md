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

ArcFace:

```bash
python3 experiments/dbscan/run_dbscan_experiment.py --route insightface
```

HOG:

```bash
python3 experiments/dbscan/run_dbscan_experiment.py --route hog
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
