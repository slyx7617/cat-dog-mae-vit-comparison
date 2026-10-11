import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = "/home/luxiaoya/cat_dog_cfnew"

TRAIN_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset_full",
    "processed",
    "train"
)

VAL_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset_full",
    "processed",
    "val"
)

MAE_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "experiment_B_mae_vit",
    "checkpoints",
    "mae_best.pth"
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "experiment_B_mae_vit",
    "checkpoints"
)

LOG_DIR = os.path.join(
    PROJECT_ROOT,
    "experiment_B_mae_vit",
    "logs"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "experiment_B_mae_vit",
    "results"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================
# Hyperparameters
# ============================================================

SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 200

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05

NUM_WORKERS = 4
NUM_CLASSES = 2

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(SEED)

# ============================================================
# Import MAE model
# ============================================================

sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiment_B_mae_vit"))

from mae_model import mae_vit_base_patch16_dec512d8b


# ============================================================
# GPU selection
# ============================================================

from common.gpu_utils import get_free_gpu

if torch.cuda.is_available():
    gpu_id = get_free_gpu()
    DEVICE = torch.device(f"cuda:{gpu_id}")
else:
    DEVICE = torch.device("cpu")


# ============================================================
# Dataset
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=VAL_DIR,
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ============================================================
# MAE -> Classification model
# ============================================================

class MAEViTClassifier(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        # Create the same MAE architecture used during pretraining
        self.mae = mae_vit_base_patch16_dec512d8b()

        # Classification head
        self.head = nn.Linear(
            768,
            num_classes
        )

    def forward(self, x):

        # Use the pretrained MAE encoder.
        #
        # mask_ratio=0 means that during classification
        # we do NOT mask image patches.
        #
        # The MAE encoder returns:
        # latent, mask, ids_restore
        #
        # latent shape:
        # [B, 197, 768]
        #
        # 197 = 1 CLS token + 196 image patches

        try:
            latent, _, _ = self.mae.forward_encoder(
                x,
                mask_ratio=0.0
            )
        except TypeError:
            latent, _, _ = self.mae.forward_encoder(
                x,
                0.0
            )

        # CLS token
        cls_feature = latent[:, 0]

        # Classification
        logits = self.head(cls_feature)

        return logits


# ============================================================
# Load MAE pretrained checkpoint
# ============================================================

def load_mae_checkpoint(model, checkpoint_path):

    print()
    print("=" * 60)
    print("Loading MAE pretrained checkpoint")
    print("=" * 60)

    print("Checkpoint:")
    print(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"MAE checkpoint not found:\n{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    # Different training scripts may save checkpoints
    # in different formats.
    if isinstance(checkpoint, dict):

        if "model" in checkpoint:
            state_dict = checkpoint["model"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Remove possible "module." prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        cleaned_state_dict[new_key] = value

    # Only load MAE weights into the MAE part.
    mae_state_dict = {}

    for key, value in cleaned_state_dict.items():

        if key.startswith("mae."):
            mae_state_dict[key[len("mae."):]] = value

        else:
            mae_state_dict[key] = value

    missing_keys, unexpected_keys = model.mae.load_state_dict(
        mae_state_dict,
        strict=False
    )

    print()
    print("MAE checkpoint loaded.")

    if len(missing_keys) > 0:
        print("Missing keys:")
        for key in missing_keys[:20]:
            print("  ", key)

    if len(unexpected_keys) > 0:
        print("Unexpected keys:")
        for key in unexpected_keys[:20]:
            print("  ", key)

    print("=" * 60)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(labels, predictions, probabilities):

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    try:
        auc = roc_auc_score(
            labels,
            probabilities
        )
    except ValueError:
        auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    total = 0

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        running_loss += (
            loss.item() * images.size(0)
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[:, 1]

        predictions = outputs.argmax(
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        total += labels.size(0)

    val_loss = running_loss / total

    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )

    return (
        val_loss,
        metrics,
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 60)
    print("MAE -> ViT Classification")
    print("=" * 60)

    print()
    print("Dataset:")
    print(TRAIN_DIR)

    print()
    print("Classes:")
    print(train_dataset.classes)

    print()
    print("Class to index:")
    print(train_dataset.class_to_idx)

    print()
    print("Train samples:")
    print(len(train_dataset))

    print("Validation samples:")
    print(len(val_dataset))

    print()
    print("Image size:")
    print(f"{IMAGE_SIZE} x {IMAGE_SIZE}")

    print()
    print("Batch size:")
    print(BATCH_SIZE)

    print()
    print("Epochs:")
    print(EPOCHS)

    print()
    print("Learning rate:")
    print(LEARNING_RATE)

    print()
    print("Weight decay:")
    print(WEIGHT_DECAY)

    print()
    print("Device:")
    print(DEVICE)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = MAEViTClassifier(
        num_classes=NUM_CLASSES
    )

    # --------------------------------------------------------
    # Load pretrained MAE
    # --------------------------------------------------------

    load_mae_checkpoint(
        model,
        MAE_CHECKPOINT
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Classification loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Cosine learning-rate scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # --------------------------------------------------------
    # CSV log
    # --------------------------------------------------------

    log_path = os.path.join(
        LOG_DIR,
        "finetune.csv"
    )

    with open(
        log_path,
        "w"
    ) as f:

        f.write(
            "epoch,"
            "train_loss,"
            "train_accuracy,"
            "val_loss,"
            "val_accuracy,"
            "val_precision,"
            "val_recall,"
            "val_f1,"
            "val_auc,"
            "learning_rate\n"
        )

    best_f1 = -1.0
    best_epoch = 0

    best_metrics = None

    final_labels = None
    final_predictions = None
    final_probabilities = None

    train_losses = []
    val_losses = []

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        (
            val_loss,
            metrics,
            labels,
            predictions,
            probabilities
        ) = validate(
            model,
            val_loader,
            criterion
        )

        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(
            train_loss
        )

        val_losses.append(
            val_loss
        )

        # ----------------------------------------------------
        # Save CSV
        # ----------------------------------------------------

        with open(
            log_path,
            "a"
        ) as f:

            f.write(
                f"{epoch},"
                f"{train_loss},"
                f"{train_accuracy},"
                f"{val_loss},"
                f"{metrics['accuracy']},"
                f"{metrics['precision']},"
                f"{metrics['recall']},"
                f"{metrics['f1']},"
                f"{metrics['auc']},"
                f"{current_lr}\n"
            )

        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Val Loss: {val_loss:.6f} "
            f"Val Acc: {metrics['accuracy']:.4f} "
            f"Val F1: {metrics['f1']:.4f} "
            f"Val AUC: {metrics['auc']:.4f} "
            f"LR: {current_lr:.8f}"
        )

        # ----------------------------------------------------
        # Save best checkpoint by validation F1
        # ----------------------------------------------------

        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]
            best_epoch = epoch
            best_metrics = metrics.copy()

            final_labels = labels.copy()
            final_predictions = predictions.copy()
            final_probabilities = probabilities.copy()

            best_checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                "mae_vit_best.pth"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "metrics": best_metrics
                },
                best_checkpoint_path
            )

            print(
                f"Best MAE-ViT model saved: "
                f"{best_checkpoint_path}"
            )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step()

    # ========================================================
    # Save latest model
    # ========================================================

    latest_checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        "mae_vit_latest.pth"
    )

    torch.save(
        {
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_f1": best_f1
        },
        latest_checkpoint_path
    )

    # ========================================================
    # Save best metrics
    # ========================================================

    metrics_path = os.path.join(
        RESULT_DIR,
        "mae_vit_best_metrics.json"
    )

    metrics_to_save = {
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "accuracy": best_metrics["accuracy"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1": best_metrics["f1"],
        "auc": best_metrics["auc"]
    }

    with open(
        metrics_path,
        "w"
    ) as f:

        json.dump(
            metrics_to_save,
            f,
            indent=4
        )

    # ========================================================
    # Confusion matrix
    # ========================================================

    cm = confusion_matrix(
        final_labels,
        final_predictions
    )

    cm_path = os.path.join(
        RESULT_DIR,
        "mae_vit_confusion_matrix.png"
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(cm)

    plt.title(
        "MAE-ViT Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.xticks(
        [0, 1],
        ["cats", "dogs"]
    )

    plt.yticks(
        [0, 1],
        ["cats", "dogs"]
    )

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        cm_path,
        dpi=300
    )

    plt.close()

    # ========================================================
    # Loss curve
    # ========================================================

    loss_curve_path = os.path.join(
        RESULT_DIR,
        "mae_vit_loss_curve.png"
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        range(1, EPOCHS + 1),
        train_losses,
        label="Train Loss"
    )

    plt.plot(
        range(1, EPOCHS + 1),
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MAE-ViT Classification Loss")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        loss_curve_path,
        dpi=300
    )

    plt.close()

    # ========================================================
    # Final output
    # ========================================================

    print()
    print("=" * 60)
    print("MAE-ViT Classification Finished")
    print("=" * 60)

    print()
    print(f"Best epoch: {best_epoch}")

    print(f"Best validation F1: {best_f1:.6f}")

    print()
    print("Best metrics:")

    print(
        f"Accuracy : {best_metrics['accuracy']:.6f}"
    )

    print(
        f"Precision: {best_metrics['precision']:.6f}"
    )

    print(
        f"Recall   : {best_metrics['recall']:.6f}"
    )

    print(
        f"F1       : {best_metrics['f1']:.6f}"
    )

    print(
        f"AUC      : {best_metrics['auc']:.6f}"
    )

    print()
    print("Best checkpoint:")
    print(best_checkpoint_path)

    print()
    print("Latest checkpoint:")
    print(latest_checkpoint_path)

    print()
    print("Training log:")
    print(log_path)

    print()
    print("Metrics:")
    print(metrics_path)

    print()
    print("Confusion matrix:")
    print(cm_path)

    print()
    print("Loss curve:")
    print(loss_curve_path)

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
