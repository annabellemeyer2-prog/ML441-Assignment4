# ML 441 Assignment 4: Ensemble Learning

Comparison of a tuned random forest with k nearest neighbours (kNN) and a classification tree on a modified version of the UNSW-NB15 network intrusion dataset (257 673 instances, 43 descriptive features, 10 classes).

## Contents

- `27129160RW441assignment4.ipynb` – the full notebook: data checks, preprocessing, tuning, evaluation, statistical tests and figures.

## Method summary

- 10-fold stratified cross validation with one saved fold assignment (seed 42), shared by all models.
- Control parameters tuned inside each training fold on a stratified subsample (21 000 tuning / 9 000 validation instances), selected on validation macro F1.
- Models:
  - kNN: one hot encoding, correlated features removed, min max scaling, k tuned.
  - Classification tree: entropy, ordinal encoding, maximum depth and minimum leaf size tuned.
  - Random forest: 150 entropy trees grown until pure, features per split (m) and bag size tuned.
- Performance measures: accuracy, macro F1, macro recall, Cohen's kappa and the Matthews correlation coefficient.
- Models compared with paired Wilcoxon signed rank tests and the Holm correction.

## How to run

1. Open the notebook in Google Colab.
2. Upload the dataset to Google Drive and set `OUT_PATH` in the first cells to your folder.
3. Run all cells.

Slow steps save their results to Drive and reload them on later runs. Two switches control this:

- `RERUN_BASELINES = True` retrains kNN and the classification tree.
- `RERUN_RF = True` retrains the random forest (about 1.5 hours over 10 folds).

Leave both as `False` to reuse the saved results.

## Requirements

Python 3 with scikit-learn 1.6.1, pandas, NumPy, SciPy, imbalanced-learn and matplotlib (all available on Colab).

## Data

The dataset is not included in this repository. It was provided as part of the course.
