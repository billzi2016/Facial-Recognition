# Face recognition experiment report

This report uses the full aligned CelebA face dataset. The experiment ran two embedding routes:

- ArcFace: the main route, producing 512 dimensional embeddings.
- HOG/dlib: the CPU baseline, producing 128 dimensional embeddings.

Both routes go through the same downstream experiments. FAISS tests 1:N lookup and unknown rejection. DBSCAN tests unlabeled grouping and 2D visualization. This keeps the comparison focused on the embedding model rather than on different downstream logic.

## Dataset structure

![CelebA examples](outputs/report/celeba_identity_examples.jpg)

CelebA is an aligned face image dataset with identity labels. This experiment uses those labels to build three splits: `gallery` contains known photos stored in the vector index, `query_known` contains other photos of identities that already exist in the gallery, and `query_unknown` contains identities that are not in the gallery. That gives the experiment two checks at once: whether the same person can be retrieved, and whether an unknown person is rejected.

The first row in the figure shows multiple photos of the same identity, so the known query is not just a duplicate of the gallery image. The second row shows different identities, which is the basic contrast behind lookup and unknown rejection.

## Embedding extraction

![Embedding quality](outputs/report/embedding_quality_comparison.jpg)

ArcFace produced 202414 embeddings and failed on 185 images, for a success rate of about 99.91%. HOG produced 196990 embeddings and failed on 5609 images, for a success rate of about 97.23%.

Even with aligned face images, HOG misses more samples. ArcFace takes longer overall, but it preserves more usable images for the later experiments.

| Route | Input images | Successful embeddings | Failed images | Success rate |
| --- | ---: | ---: | ---: | ---: |
| ArcFace | 202599 | 202414 | 185 | 99.91% |
| HOG | 202599 | 196990 | 5609 | 97.23% |

## FAISS search

![FAISS comparison](outputs/report/faiss_comparison.jpg)

The FAISS experiment builds a gallery index, then searches `query_known` and `query_unknown`. `query_known` measures whether the same identity can be retrieved. `query_unknown` measures how easy it is to reject people who are not in the gallery.

ArcFace reached 93.04% Top-1 accuracy and 95.22% Top-K(k=5) accuracy. HOG reached 71.42% Top-1 accuracy and 82.11% Top-K(k=5) accuracy.

The score gap matters even more. ArcFace has a mean known score of 0.595 and a mean unknown score of 0.288, a gap of about 0.307. HOG has a mean known score of 0.953 and a mean unknown score of 0.936, a gap of only about 0.017. That makes threshold selection much harder for HOG.

| Route | Known query | Unknown query | Top-1 | Top-K(k=5) | Known mean | Unknown mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ArcFace | 172592 | 20669 | 93.04% | 95.22% | 0.595 | 0.288 |
| HOG | 168034 | 20105 | 71.42% | 82.11% | 0.953 | 0.936 |

Here `K=5` means each query asks FAISS for the five nearest gallery candidates. Top-1 only checks the first candidate. Top-K(k=5) checks whether the correct identity appears anywhere in those five candidates.

## DBSCAN clustering

![DBSCAN comparison](outputs/report/dbscan_comparison.jpg)

DBSCAN is different from FAISS. FAISS searches for a query identity. DBSCAN groups embeddings without using labels. The current DBSCAN run uses 20000 samples for clustering and visualization.

The DBSCAN setting was selected by sweeping route and distance metric combinations. The selection used non-noise cluster count, noise rate, largest cluster share, ARI, and NMI. Largest cluster share catches collapse into one oversized cluster. NMI and ARI use identity labels only for after-the-fact evaluation; they are not used by DBSCAN itself.

The final ArcFace setting is `metric=cosine, eps=0.56`. It produced 1607 non-noise clusters with a 62.19% noise rate and a non-noise NMI of 0.996. The final HOG setting is `metric=euclidean, eps=0.40`. It produced 339 non-noise clusters with a 90.61% noise rate and a non-noise NMI of 0.921. This means HOG is not unable to cluster at all. It can find some pure small groups, but it covers far fewer images.

| Route | Samples | Non-noise clusters | Noise points | Noise rate | Non-noise NMI |
| --- | ---: | ---: | ---: | ---: | ---: |
| ArcFace cosine eps=0.56 | 20000 | 1607 | 12437 | 62.19% | 0.996 |
| HOG euclidean eps=0.40 | 20000 | 339 | 18122 | 90.61% | 0.921 |

| Sweep combination | Observation | Used |
| --- | --- | --- |
| ArcFace + cosine | At `eps=0.56`, noise decreases, the largest cluster remains small, and NMI stays around 0.996 | yes |
| ArcFace + euclidean | The tested range produced all noise | no |
| HOG + cosine | The tested range connected everything into one large cluster | no |
| HOG + euclidean | `eps=0.40` forms pure small groups; `eps=0.50` starts to show oversized-cluster risk | yes |

## Cluster distribution and vector fields

![ArcFace cluster distribution](outputs/report/insightface_cluster_distribution.jpg)

![HOG cluster distribution](outputs/report/hog_cluster_distribution.jpg)

The distribution charts show the largest DBSCAN clusters directly. ArcFace creates many smaller identity groups. HOG also forms some small groups under the euclidean setting, but with a much higher noise rate.

The vector field plots first project embeddings to PCA 2D space. Each grid arrow shows the average direction from local samples toward their assigned cluster centers. This is not a physical vector field. It is a way to see whether the embedding space has local grouping structure.

![ArcFace vector field](outputs/report/insightface_vector_field.jpg)

![HOG vector field](outputs/report/hog_vector_field.jpg)

ArcFace shows local directions across several regions, which matches the many-cluster DBSCAN result. HOG has sparser local structure. It can form some small groups, but its coverage is much lower than ArcFace.

## HOG feature intuition

![HOG feature grid](outputs/report/hog_feature_grid.jpg)

This figure shows what the HOG baseline looks at during detection. The yellow marks are not arrows with a forward direction. They are unsigned orientation-bin line segments: HOG cares about edge orientation, not whether the edge points left or right. When several bins in one cell have strong gradient responses, the cell can look like a small star of overlapping line segments. Face outlines, eyes, nose bridges, and mouths create dense local orientation histograms that a traditional detector can match. This is different from ArcFace embeddings: ArcFace learns identity-oriented deep features, while HOG describes local edge structure.

## Conclusion

ArcFace is the better main route in this experiment. It keeps more images, retrieves identities more accurately, and leaves a wider score gap between known and unknown people. That score gap matters because unknown rejection depends on a usable threshold.

HOG remains useful as a baseline. It proves that the same FAISS and DBSCAN pipeline can run on traditional embeddings. With euclidean DBSCAN it can form some pure small groups, but its coverage is much lower than ArcFace. The main project should continue around ArcFace embeddings, with HOG kept as the comparison route.
