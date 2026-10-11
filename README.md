# Comparison of Random ViT and MAE-Pretrained ViT for Cat-Dog Classification

## 1. Project Overview

This project compares two Vision Transformer (ViT) classification approaches
on a cat-versus-dog image classification task:

- Experiment A: ViT with random initialization, trained for 200 epochs.
- Experiment B: Masked Autoencoder (MAE) self-supervised pretraining,
  followed by ViT classification fine-tuning for 200 epochs.

The purpose is to investigate whether MAE pretraining improves downstream
image classification performance under the current experimental setup.

## 2. Experimental Pipeline

### Experiment A: Random ViT

A ViT-B/16 model is initialized without pretrained weights and trained
for binary cat-dog classification.

### Experiment B: MAE -> ViT

1. Pretrain the MAE model using image reconstruction with masked patches.
2. Load the pretrained encoder weights into the ViT classification model.
3. Fine-tune the classifier for cat-dog classification.

MAE pretraining: 200 epochs.
Classification fine-tuning: 200 epochs.

## 3. Dataset

The processed dataset is organized into training and validation subsets.

The current experiment logs indicate 20,000 training images and 3,628
validation images. The validation set is imbalanced, so its composition
should be considered when interpreting the results.

The dataset is not included in this repository. Users must prepare the
dataset locally and update paths in the training scripts if necessary.

## 4. Best Validation Results

| Metric | Random ViT | MAE -> ViT |
|---|---:|---:|
| Best validation accuracy | 81.31% | 83.46% |
| Best validation F1 | 70.63% | 73.39% |
| Best validation AUC | 87.44% | 89.53% |

Best metrics are selected independently for each metric from the recorded
validation logs. They may correspond to different epochs.

These are results from the current experiment and validation split.
They do not establish statistical significance or guarantee performance
on other datasets.

## 5. Result Files

- `final_report/figures/`: training and validation comparison plots.
- `final_report/tables/metrics_comparison.csv`: summary of final and best
  validation metrics.
- `experiment_A_random_vit/logs/train.csv`: Experiment A training log.
- `experiment_B_mae_vit/logs/pretrain.csv`: MAE pretraining log.
- `experiment_B_mae_vit/logs/finetune.csv`: Experiment B fine-tuning log.

## 6. Repository Contents

- `experiment_A_random_vit/`: Random ViT training code and logs.
- `experiment_B_mae_vit/`: MAE pretraining and ViT fine-tuning code.
- `final_report/`: comparison plots and metric summaries.

Large datasets and model checkpoints are intentionally excluded from Git.

## 7. Reproducibility Notes

The training scripts, hyperparameters, dataset paths, and software
environment should be checked before reproducing the experiments.

The recorded validation metrics are reported as obtained from the
existing CSV logs. Further evaluation using saved best checkpoints,
confusion matrices, and ROC curves is recommended before drawing
strong conclusions about model generalization.
