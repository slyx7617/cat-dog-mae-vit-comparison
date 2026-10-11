import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


DATA_ROOT = "/home/luxiaoya/cat_dog_cfnew/dataset_full/processed"


def get_transforms():
    """
    返回训练集和验证集的数据增强。
    两个实验必须使用完全相同的数据处理。
    """

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    return train_transform, val_transform


def get_datasets():
    """
    返回：
        train_dataset
        val_dataset
    """

    train_transform, val_transform = get_transforms()

    train_dir = os.path.join(DATA_ROOT, "train")
    val_dir = os.path.join(DATA_ROOT, "val")

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=val_transform
    )

    return train_dataset, val_dataset


def get_dataloaders(
    batch_size=64,
    num_workers=4
):
    """
    创建训练和验证 DataLoader。
    """

    train_dataset, val_dataset = get_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return (
        train_loader,
        val_loader,
        train_dataset,
        val_dataset
    )


if __name__ == "__main__":

    train_dataset, val_dataset = get_datasets()

    print("Dataset root:")
    print(DATA_ROOT)

    print()

    print("Classes:")
    print(train_dataset.classes)

    print("Class to index:")
    print(train_dataset.class_to_idx)

    print()

    print("Train samples:")
    print(len(train_dataset))

    print("Validation samples:")
    print(len(val_dataset))

