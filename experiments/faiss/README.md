# FAISS search experiment

The FAISS experiment answers one question: given a query face, can the system find the same identity in the gallery? If the query is an unknown person, the nearest result should not be accepted automatically. A threshold decides whether the system rejects it.

This is different from DBSCAN. FAISS is targeted 1:N lookup with `gallery`, `query_known`, and `query_unknown`. DBSCAN is unlabeled grouping.

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

Both routes must run. ArcFace is the main route. HOG is the baseline used for comparison.

## Run

ArcFace:

```bash
python3 experiments/faiss/run_faiss_experiment.py --route insightface
```

HOG:

```bash
python3 experiments/faiss/run_faiss_experiment.py --route hog
```

For a small check, limit the query counts:

```bash
python3 experiments/faiss/run_faiss_experiment.py \
  --route insightface \
  --max-known 100 \
  --max-unknown 100 \
  --output-dir outputs/faiss_smoke/insightface
```

Remove smoke output after checking it.

## Outputs

Each route writes:

```text
outputs/faiss/<route>/index.faiss
outputs/faiss/<route>/known_search_results.csv
outputs/faiss/<route>/unknown_search_results.csv
outputs/faiss/<route>/threshold_report.csv
outputs/faiss/<route>/benchmark.json
```

`known_search_results.csv` measures Top-1 and Top-K hits. `unknown_search_results.csv` and `threshold_report.csv` measure unknown rejection.

## How to read the result

A valid run should have:

- at least one gallery vector
- known query rows with Top-1 and Top-K results
- unknown query rows reported separately
- benchmark fields for embedding dimension, index type, score type, and elapsed time
- separate output directories for ArcFace and HOG
