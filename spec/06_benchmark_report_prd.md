# PRD: benchmark report

## Purpose

The benchmark report explains what happened across the full experiment. It should not read like a list of tool names. It should show how the dataset moved through the pipeline, what each route produced, and what changed when the embedding model changed.

ArcFace is the main route. HOG is the baseline. FAISS and DBSCAN both run on both routes.

## Required sections

1. Dataset and split summary
2. HOG baseline embedding results
3. ArcFace embedding results
4. FAISS identity lookup
5. DBSCAN grouping and 2D views
6. Error cases and failure patterns
7. Comparison between ArcFace and HOG
8. Final conclusions

## What the report must explain

The report must make the difference between the experiments clear:

- FAISS tests lookup. It asks whether a query image finds the right identity in a gallery and whether an unknown person is rejected.
- DBSCAN tests grouping. It asks whether unlabeled embeddings form useful clusters and which images become noise.

Both experiments must include ArcFace and HOG. The report cannot present ArcFace only and call the baseline optional.

## Required artifacts

- Embedding benchmark tables for both routes
- FAISS Top-1 and Top-K(k=5) tables for both routes
- Unknown rejection threshold results for both routes
- DBSCAN cluster summary for both routes
- PCA, t-SNE, and UMAP JPG plots for both routes
- A comparison table covering latency, failure rate, search accuracy, and clustering quality

## Acceptance criteria

- Every number in the report links back to an output file.
- The report states whether it used the full dataset or a debug subset.
- ArcFace and HOG results are both present in FAISS and DBSCAN sections.
- The conclusion separates measured facts from interpretation.
- Plot paths point to JPG files, not PNG files.
