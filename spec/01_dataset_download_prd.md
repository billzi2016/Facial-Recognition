# PRD: dataset download and manifests

## Purpose

The dataset step prepares the aligned CelebA face images and the metadata needed by every later experiment. The output is not just a folder of images. It is a set of manifest files that tell the other scripts where each image is, which identity it belongs to, and which experiment split should use it.

## Goals

- Download or reuse the full CelebA aligned image dataset.
- Add the CelebA identity annotation when the image package does not include it.
- Generate full dataset manifests for images, identities, splits, and quality tags.
- Make sure every usable image has a clear destination.
- Keep debug subsets separate from the full experiment.

## Inputs

- `data/raw/celeba/img_align_celeba/img_align_celeba/`
- `data/raw/celeba/identity_CelebA.txt`
- optional CelebA files such as attributes, bounding boxes, landmarks, and original split files

## Outputs

- `data/manifests/images.csv`
- `data/manifests/identities.csv`
- `data/manifests/splits.csv`
- `data/manifests/quality_tags.csv`

The generated CSV files are ignored by git because they are local experiment outputs.

## Scripts

The dataset scripts live under `data/scripts/`:

- `download_celeba.py` downloads or copies the image package and can extract it.
- `download_celeba_identity.py` downloads the identity annotation and normalizes it to `identity_CelebA.txt`.
- `prepare_celeba_manifests.py` scans the full dataset and writes the manifest CSV files.

The scripts are files on disk. They are not one line `python -c` commands, so the dataset setup can be reviewed and repeated.

## Split meaning

- `gallery` contains the reference images used to build the identity database.
- `query_known` contains other images of identities already present in the gallery.
- `query_unknown` contains identities that do not appear in the gallery.
- `cluster_mix` is available for grouping experiments when labels are not used.

The identity file is what makes `gallery`, `query_known`, and `query_unknown` possible. Without it, the project can still cluster embeddings, but it cannot measure known identity lookup or unknown rejection correctly.

## Acceptance criteria

- The manifest contains the full aligned image set.
- The split summary reports gallery, known query, and unknown query counts when identity labels are available.
- The zip files used for download are removed after extraction unless debugging requires keeping them.
- Kaggle credentials never enter the repository.
- The output can be used by HOG, ArcFace, FAISS, DBSCAN, and the final report.
