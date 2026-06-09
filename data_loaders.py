import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import BATCH_SIZE


def build_transforms(dataset_name):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if dataset_name == "Brain Tumor":
        train_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.85, 1.0),
                    ratio=(0.95, 1.05),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.10, contrast=0.10),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                normalize,
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                normalize,
            ]
        )
        return train_transform, test_transform

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return transform, transform


def build_dataloaders(train_dir, test_dir, dataset_name):
    train_transform, test_transform = build_transforms(dataset_name)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_dataset, test_dataset, train_loader, test_loader


def partition_data(dataset, n_clients=2, beta=0.8):
    labels = np.array([label for _, label in dataset])
    indices = np.arange(len(labels))
    client_indices = {client_id: [] for client_id in range(n_clients)}

    for class_id in np.unique(labels):
        class_indices = indices[labels == class_id]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet(np.ones(n_clients) * beta)
        split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        splits = np.split(class_indices, split_points)

        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    print("\n" + "=" * 80)
    print("DATA DISTRIBUTION ACROSS CLIENTS")
    print("=" * 80)
    for client_id, idxs in client_indices.items():
        client_labels = labels[idxs]
        dist = {
            dataset.classes[class_id]: int(np.sum(client_labels == class_id))
            for class_id in range(len(dataset.classes))
        }
        print(f"Client {client_id}: Total={len(idxs)}, Distribution={dist}")
    print("=" * 80)
    return client_indices
