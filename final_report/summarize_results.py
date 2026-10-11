import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

A_CSV = ROOT / "experiment_A_random_vit/logs/train.csv"
B_CSV = ROOT / "experiment_B_mae_vit/logs/finetune.csv"

OUT = ROOT / "final_report"
FIG = OUT / "figures"
TAB = OUT / "tables"

FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"找不到日志文件：{path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"日志文件没有数据：{path}")

    for row in rows:
        for key, value in row.items():
            if key == "epoch":
                row[key] = int(value)
            elif value not in (None, ""):
                try:
                    row[key] = float(value)
                except (ValueError, TypeError):
                    pass

    return rows


A = read_csv(A_CSV)
B = read_csv(B_CSV)

experiments = {
    "Random ViT": A,
    "MAE→ViT": B,
}

metrics = [
    "train_loss",
    "val_loss",
    "train_accuracy",
    "val_accuracy",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_auc",
]


def save_plot(metric, title, ylabel, filename):
    plt.figure(figsize=(9, 5))

    for name, rows in experiments.items():
        if metric not in rows[0]:
            print(f"跳过 {name} 的 {metric}：日志中不存在该字段")
            continue

        epochs = [r["epoch"] for r in rows]
        values = [r[metric] for r in rows]
        plt.plot(epochs, values, label=name, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = FIG / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"已生成：{path}")


save_plot(
    "train_loss", "Training Loss Comparison",
    "Training Loss", "training_loss_comparison.png"
)
save_plot(
    "val_loss", "Validation Loss Comparison",
    "Validation Loss", "validation_loss_comparison.png"
)
save_plot(
    "train_accuracy", "Training Accuracy Comparison",
    "Training Accuracy", "training_accuracy_comparison.png"
)
save_plot(
    "val_accuracy", "Validation Accuracy Comparison",
    "Validation Accuracy", "validation_accuracy_comparison.png"
)
save_plot(
    "val_f1", "Validation F1 Comparison",
    "F1 Score", "validation_f1_comparison.png"
)
save_plot(
    "val_auc", "Validation AUC Comparison",
    "AUC", "validation_auc_comparison.png"
)

# 每个实验：最后一轮结果 + 各项指标达到最佳时的 epoch
summary = []

for name, rows in experiments.items():
    last = rows[-1]
    item = {
        "experiment": name,
        "epochs_recorded": len(rows),
        "final_epoch": last["epoch"],
    }

    for metric in [
        "train_loss", "train_accuracy",
        "val_loss", "val_accuracy",
        "val_precision", "val_recall", "val_f1", "val_auc"
    ]:
        if metric in last:
            item[f"final_{metric}"] = last[metric]

    for metric in [
        "val_accuracy", "val_f1", "val_auc"
    ]:
        if metric in rows[0]:
            best = max(rows, key=lambda r: r[metric])
            item[f"best_{metric}"] = best[metric]
            item[f"best_{metric}_epoch"] = best["epoch"]

    summary.append(item)

summary_path = TAB / "metrics_comparison.csv"
fieldnames = list(dict.fromkeys(
    key for item in summary for key in item.keys()
))

with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary)

json_path = TAB / "metrics_comparison.json"
with json_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# 复制已有的 MAE→ViT 混淆矩阵，不伪造 Random ViT 的混淆矩阵
existing_cm = ROOT / "experiment_B_mae_vit/results/mae_vit_confusion_matrix.png"
if existing_cm.exists():
    destination = FIG / "mae_vit_confusion_matrix.png"
    shutil.copy2(existing_cm, destination)
    print(f"已复制已有混淆矩阵：{destination}")
else:
    print("未找到已有 MAE→ViT 混淆矩阵，暂不复制。")

print("\n========== 结果汇总 ==========")
for item in summary:
    print(f"\n{item['experiment']}")
    print(f"记录 epoch 数：{item['epochs_recorded']}")
    print(f"最后 epoch：{item['final_epoch']}")

    for metric in ["val_accuracy", "val_f1", "val_auc"]:
        key = f"final_{metric}"
        if key in item:
            print(f"最后一轮 {metric}: {item[key]:.4f}")

        best_key = f"best_{metric}"
        epoch_key = f"best_{metric}_epoch"
        if best_key in item:
            print(
                f"最佳 {metric}: {item[best_key]:.4f} "
                f"(epoch {item[epoch_key]})"
            )

print(f"\n汇总表：{summary_path}")
print(f"汇总 JSON：{json_path}")
print(f"图像目录：{FIG}")
print("\n完成。")
