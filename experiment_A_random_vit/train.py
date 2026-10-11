import os
import sys
import csv
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================
# 添加项目根目录
# ============================================================

PROJECT_ROOT = "/home/luxiaoya/cat_dog_cfnew"

sys.path.append(PROJECT_ROOT)

from common.dataset import get_dataloaders
from common.gpu_utils import get_free_gpu

from models import RandomInitViT


# ============================================================
# 实验配置
# ============================================================

SEED = 42

BATCH_SIZE = 64

EPOCHS = 200

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 0.05

NUM_WORKERS = 4

NUM_CLASSES = 2

IMAGE_SIZE = 224


# ============================================================
# 路径
# ============================================================

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT_DIR = os.path.join(
    EXPERIMENT_DIR,
    "checkpoints"
)

LOG_DIR = os.path.join(
    EXPERIMENT_DIR,
    "logs"
)

RESULT_DIR = os.path.join(
    EXPERIMENT_DIR,
    "results"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# 固定随机种子
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    # 为了实验可复现
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# 选择 GPU
# ============================================================

def get_device():

    if not torch.cuda.is_available():

        print("CUDA is not available.")

        return torch.device("cpu")

    gpu_id = get_free_gpu()

    print(f"Automatically selected GPU: {gpu_id}")

    return torch.device(f"cuda:{gpu_id}")


# ============================================================
# 单个 epoch：训练
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    all_predictions = []

    all_labels = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 分类损失
        loss = criterion(
            outputs,
            labels
        )

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 记录
        running_loss += (
            loss.item() *
            images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# 单个 epoch：验证
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    all_labels = []

    all_predictions = []

    all_probabilities = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        running_loss += (
            loss.item() *
            images.size(0)
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        # 狗类概率
        all_probabilities.extend(
            probabilities[:, 1].cpu().numpy()
        )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    try:

        auc = roc_auc_score(
            all_labels,
            all_probabilities
        )

    except ValueError:

        auc = float("nan")

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    return {
        "loss": epoch_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm.tolist()
    }


# ============================================================
# 保存最终结果
# ============================================================

def save_final_results(
    model,
    loader,
    criterion,
    device
):

    metrics = validate(
        model,
        loader,
        criterion,
        device
    )

    result_path = os.path.join(
        RESULT_DIR,
        "final_metrics.json"
    )

    with open(
        result_path,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print()
    print("=" * 60)
    print("Final Validation Results")
    print("=" * 60)

    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"F1       : {metrics['f1']:.4f}"
    )

    print(
        f"AUC      : {metrics['auc']:.4f}"
    )

    print()
    print("Confusion Matrix:")

    print(
        np.array(
            metrics["confusion_matrix"]
        )
    )

    print()
    print(
        f"Saved to: {result_path}"
    )

    return metrics


# ============================================================
# 主函数
# ============================================================

def main():

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print()
    print("=" * 60)
    print("Random Initialization ViT Experiment")
    print("=" * 60)

    print(
        f"Device: {device}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Image size: {IMAGE_SIZE}"
    )

    print(
        "Model: ViT-B/16"
    )

    print(
        "Initialization: RANDOM"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_loader, val_loader, train_dataset, val_dataset = \
        get_dataloaders(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )

    print()
    print("Classes:")
    print(
        train_dataset.classes
    )

    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = RandomInitViT(
        num_classes=NUM_CLASSES
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Learning rate scheduler
    # --------------------------------------------------------

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        LOG_DIR,
        "train.csv"
    )

    csv_file = open(
        csv_path,
        "w",
        newline=""
    )

    csv_writer = csv.writer(
        csv_file
    )

    csv_writer.writerow([
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_auc",
        "learning_rate"
    ])

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_f1 = -1.0

    best_epoch = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss, train_accuracy = \
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )

        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Val Loss: {val_metrics['loss']:.4f} "
            f"Val Acc: {val_metrics['accuracy']:.4f} "
            f"Val F1: {val_metrics['f1']:.4f} "
            f"Val AUC: {val_metrics['auc']:.4f}"
        )

        csv_writer.writerow([
            epoch,
            train_loss,
            train_accuracy,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["f1"],
            val_metrics["auc"],
            current_lr
        ])

        csv_file.flush()

        # ----------------------------------------------------
        # 保存最佳模型
        #
        # 使用 F1 作为选择标准
        # 因为 validation set 不平衡
        # ----------------------------------------------------

        if val_metrics["f1"] > best_f1:

            best_f1 = val_metrics["f1"]

            best_epoch = epoch

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                "best_model.pth"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "classes": train_dataset.classes
                },
                checkpoint_path
            )

            print(
                f"  -> Best model saved "
                f"(F1={best_f1:.4f})"
            )

        scheduler.step()

    csv_file.close()

    # --------------------------------------------------------
    # 加载最佳模型
    # --------------------------------------------------------

    best_model_path = os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth"
    )

    checkpoint = torch.load(
        best_model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print()
    print("=" * 60)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation F1: {best_f1:.4f}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    save_final_results(
        model,
        val_loader,
        criterion,
        device
    )

    print()
    print("Training finished.")


if __name__ == "__main__":

    main()


