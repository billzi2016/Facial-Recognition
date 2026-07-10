# PRD: DBSCAN clustering and 2D visualization

## Purpose

DBSCAN is the grouping experiment. It does not ask for a query identity and it does not build a gallery lookup service. It takes a set of embeddings and tries to group similar faces together without using labels during clustering.

This experiment is separate from FAISS. FAISS is lookup. DBSCAN is grouping. They share embeddings, but they answer different questions.

DBSCAN must run twice:

- once for ArcFace embeddings from `outputs/insightface/embeddings.h5`
- once for HOG baseline embeddings from `outputs/hog/embeddings.h5`

## Goals

- Run DBSCAN on ArcFace embeddings.
- Run DBSCAN on HOG embeddings.
- Save cluster labels for both routes.
- Produce PCA, t-SNE, and UMAP two dimensional views for both routes.
- Save all plots as JPG.
- Compare the noise rate, cluster count, and label quality between routes.

## Inputs

- `outputs/insightface/embeddings.h5`
- `outputs/insightface/embedding_metadata.csv`
- `outputs/hog/embeddings.h5`
- `outputs/hog/embedding_metadata.csv`

## Outputs

- `outputs/dbscan/insightface/cluster_labels.csv`
- `outputs/dbscan/insightface/cluster_report.json`
- `outputs/dbscan/hog/cluster_labels.csv`
- `outputs/dbscan/hog/cluster_report.json`
- `outputs/dbscan/insightface/*_2d.jpg`
- `outputs/dbscan/hog/*_2d.jpg`
- `outputs/dbscan/insightface_cosine_eps_sweep/eps_sweep.csv`
- `outputs/dbscan/insightface_euclidean_eps_sweep/eps_sweep.csv`
- `outputs/dbscan/hog_cosine_eps_sweep/eps_sweep.csv`
- `outputs/dbscan/hog_euclidean_eps_sweep/eps_sweep.csv`
- a comparison report for the benchmark

## Parameter sweep

The final run uses:

- ArcFace: `metric=cosine`, `eps=0.56`, `min_samples=4`
- HOG: `metric=euclidean`, `eps=0.40`, `min_samples=4`

The sweep must cover four combinations:

- ArcFace + cosine
- ArcFace + euclidean
- HOG + cosine
- HOG + euclidean

The selection cannot rely on NMI alone. It must consider non-noise cluster count, noise rate, largest cluster share, ARI, and NMI. Largest cluster share catches collapse into one oversized cluster. ARI and NMI are used only after clustering to compare against identity labels; DBSCAN itself does not use those labels.

## Plot rule

Each route produces three plots:

- PCA 2D scatter
- t-SNE 2D scatter
- UMAP 2D scatter

The color comes from the DBSCAN cluster label. Noise points with label `-1` should be visually distinct, usually gray or low opacity.

## Acceptance criteria

- ArcFace and HOG both have independent DBSCAN results.
- Each route has PCA, t-SNE, and UMAP JPG plots.
- The plotted color represents DBSCAN output, not manual identity labels.
- Noise points are counted and sampled for review.
- The final report treats DBSCAN as a separate experiment from FAISS.
