# PRD: FAISS search

## Purpose

FAISS is the identity lookup experiment. It answers a different question from clustering: given one query face, can the system find the matching identity in the gallery, and can it reject a person who is not in the gallery?

This experiment must run twice:

- once for ArcFace embeddings from `outputs/insightface/embeddings.h5`
- once for HOG baseline embeddings from `outputs/hog/embeddings.h5`

ArcFace is the main route. HOG is the baseline. Both routes must produce search results and threshold reports.

## Goals

- Build a FAISS index for the gallery split.
- Search `query_known` and measure Top-K accuracy.
- Search `query_unknown` and measure unknown rejection.
- Scan thresholds instead of relying on a single guessed value.
- Write route specific outputs and a comparison summary.

## Inputs

- `outputs/insightface/embeddings.h5`
- `outputs/insightface/embedding_metadata.csv`
- `outputs/hog/embeddings.h5`
- `outputs/hog/embedding_metadata.csv`
- `data/manifests/splits.csv`

## Outputs

- `outputs/faiss/insightface/`
- `outputs/faiss/hog/`
- route specific FAISS index files
- route specific search result CSV files
- route specific threshold reports
- a comparison report that puts ArcFace and HOG side by side

## Index choice

The first full run should use exact search. `IndexFlatL2` is the baseline. If embeddings are normalized and cosine style scoring is desired, `IndexFlatIP` can be used, but the normalization rule must be written into the benchmark.

## Acceptance criteria

- ArcFace and HOG both have independent FAISS results.
- Known query results can trace the query image, predicted identity, true identity, and distance or score.
- Unknown query results are reported separately.
- Threshold scanning is available for both routes.
- The final report does not omit the HOG baseline.
