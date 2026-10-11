import os
import csv
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mae_model import mae_vit_base_patch16_dec512d8b
from common.gpu_utils import get_free_gpu


# ============================================================
# 1. Basic Settings
# ============================================================

SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 64

EPOCHS = 200

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05

NUM_WORKERS = 4

MASK_RATIO = 0.75


# ============================================================
# 2. Paths
# ============================================================

PROJECT_ROOT = "/home/luxiaoya/cat_dog_cfnew"

TRAIN_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset_full",
    "processed",
    "train"
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
# 3. Random Seed
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    # 为了实验可重复
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# 4. Automatically Select GPU
# ============================================================

if torch.cuda.is_available():

    gpu_id = get_free_gpu()

    DEVICE = torch.device(
        f"cuda:{gpu_id}"
    )

    print(
        f"Automatically selected GPU: {gpu_id}"
    )

else:

    DEVICE = torch.device("cpu")

    print("CUDA is not available. Using CPU.")


# ============================================================
# 5. Dataset Transform
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# 6. Dataset
# ============================================================

train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=transform
)


# ============================================================
# 7. DataLoader
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=True,

    drop_last=True
)


# ============================================================
# 8. Print Dataset Information
# ============================================================

print()
print("=" * 60)
print("MAE Pretraining Configuration")
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
print("Training images:")
print(len(train_dataset))

print()
print("Image size:")
print(
    f"{IMAGE_SIZE} x {IMAGE_SIZE}"
)

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
print("Mask ratio:")
print(MASK_RATIO)

print()
print("Number of patches:")
print(
    (IMAGE_SIZE // 16) ** 2
)

print()
print("Device:")
print(DEVICE)

print()
print("=" * 60)


# ============================================================
# 9. Create MAE Model
# ============================================================

model = mae_vit_base_patch16_dec512d8b()

model = model.to(DEVICE)


# ============================================================
# 10. Loss
# ============================================================

# MAE 自己内部已经计算 reconstruction loss
#
# 所以这里不需要额外的 CrossEntropyLoss。
#
# model(images) 会返回：
#
#     loss
#     prediction
#     mask

# ============================================================
# 11. Optimizer
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 12. Learning Rate Scheduler
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

    optimizer,

    T_max=EPOCHS
)


# ============================================================
# 13. CSV Log
# ============================================================

log_file = os.path.join(
    LOG_DIR,
    "pretrain.csv"
)


with open(
    log_file,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "epoch",
        "train_loss",
        "learning_rate"
    ])


# ============================================================
# 14. Training
# ============================================================

best_loss = float("inf")


for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0

    total_samples = 0


    for batch_idx, (images, _) in enumerate(train_loader):

        # ----------------------------------------------------
        # Move images to GPU
        # ----------------------------------------------------

        images = images.to(
            DEVICE,
            non_blocking=True
        )


        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # MAE Forward
        # ----------------------------------------------------

        loss, pred, mask = model(
            images,
            mask_ratio=MASK_RATIO
        )


        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = images.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size


        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        if (
            batch_idx + 1
        ) % 100 == 0:

            current_loss = (
                running_loss /
                total_samples
            )

            print(
                f"Epoch [{epoch}/{EPOCHS}] "
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {current_loss:.6f}"
            )


    # ========================================================
    # Epoch Loss
    # ========================================================

    epoch_loss = (
        running_loss /
        total_samples
    )


    # ========================================================
    # Scheduler
    # ========================================================

    scheduler.step()


    current_lr = optimizer.param_groups[0]["lr"]


    # ========================================================
    # Save Log
    # ========================================================

    with open(
        log_file,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            epoch,
            epoch_loss,
            current_lr
        ])


    # ========================================================
    # Print Epoch Result
    # ========================================================

    print()

    print(
        f"Epoch [{epoch}/{EPOCHS}] "
        f"Train Loss: {epoch_loss:.6f} "
        f"LR: {current_lr:.8f}"
    )


    # ========================================================
    # Save Best Encoder
    # ========================================================

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        best_checkpoint = os.path.join(
            CHECKPOINT_DIR,
            "mae_best.pth"
        )

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "loss":
                    epoch_loss,

                "mask_ratio":
                    MASK_RATIO
            },
            best_checkpoint
        )

        print(
            f"Best MAE model saved: "
            f"{best_checkpoint}"
        )


    # ========================================================
    # Save Latest Checkpoint
    # ========================================================

    latest_checkpoint = os.path.join(
        CHECKPOINT_DIR,
        "mae_latest.pth"
    )

    torch.save(
        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "loss":
                epoch_loss,

            "mask_ratio":
                MASK_RATIO
        },
        latest_checkpoint
    )


# ============================================================
# 15. Training Finished
# ============================================================

print()
print("=" * 60)
print("MAE Pretraining Finished")
print("=" * 60)

print()
print(f"Best MAE loss: {best_loss:.6f}")

print()
print(
    f"Best checkpoint:"
    f"\n{CHECKPOINT_DIR}/mae_best.pth"
)

print()
print(
    f"Latest checkpoint:"
    f"\n{CHECKPOINT_DIR}/mae_latest.pth"
)

print()
print(
    f"Training log:"
    f"\n{log_file}"
)

print()
print("=" * 60)

