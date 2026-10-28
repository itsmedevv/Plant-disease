"""
dataset.py - Dataset discovery, stratified splitting, and PyTorch DataLoader pipeline.

This module:
1. Automatically discovers the 'color' directory inside 'data/' (handling nested folders).
2. Filters classes by crop (e.g. Tomato) and caps samples per class if requested.
3. Splits images 80/10/10 into train/validation/test sets deterministically with stratification.
4. Provides a custom PyTorch Dataset that loads images on-demand in batches.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def find_color_dir(base_dir: str = "data") -> Path:
    """
    Search recursively for the 'color' directory that contains PlantVillage class folders.
    Ignores 'grayscale' and 'segmented' folders.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Base data directory '{base_dir}' does not exist.")

    # Check direct path first
    direct_color = base_path / "color"
    if direct_color.is_dir() and any(direct_color.iterdir()):
        return direct_color

    # Search recursively for a directory named 'color'
    for candidate in base_path.rglob("color"):
        if candidate.is_dir():
            # Check if candidate has subdirectories resembling class names (e.g. contains '___')
            subdirs = [p.name for p in candidate.iterdir() if p.is_dir()]
            if any("___" in name for name in subdirs):
                return candidate

    raise FileNotFoundError(
        "Could not automatically locate the 'color' dataset directory inside 'data/'. "
        "Please ensure the PlantVillage color dataset is present in 'data/'."
    )


def collect_image_records(
    color_dir: Path,
    crop: Optional[str] = None,
    max_per_class: Optional[int] = None,
    seed: int = 42
) -> Tuple[List[Tuple[str, int]], List[str]]:
    """
    Scans the color directory, applies optional crop and max_per_class filters,
    and returns a list of (image_path, class_idx) tuples along with the class name list.
    """
    # Valid image extensions
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    # List all class directories
    all_class_dirs = sorted([d for d in color_dir.iterdir() if d.is_dir()])

    # Filter by crop if requested (e.g., crop='Tomato' matches 'Tomato___Early_blight')
    if crop:
        crop_clean = crop.strip().lower()
        selected_dirs = [
            d for d in all_class_dirs
            if d.name.lower().startswith(crop_clean + "___") or d.name.lower().startswith(crop_clean)
        ]
    else:
        selected_dirs = all_class_dirs

    if not selected_dirs:
        raise ValueError(
            f"No class folders found matching crop filter '{crop}'. "
            f"Available folders include: {[d.name for d in all_class_dirs[:5]]}..."
        )

    class_names = [d.name for d in selected_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    rng = np.random.RandomState(seed)
    records: List[Tuple[str, int]] = []

    for class_dir in selected_dirs:
        class_name = class_dir.name
        class_idx = class_to_idx[class_name]

        # Gather all valid image file paths in this class folder
        image_files = [
            str(f) for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in valid_exts
        ]
        image_files.sort()  # Deterministic order before random sampling

        # Cap images per class if requested
        if max_per_class and len(image_files) > max_per_class:
            chosen_indices = rng.choice(len(image_files), size=max_per_class, replace=False)
            image_files = [image_files[i] for i in chosen_indices]

        for img_path in image_files:
            records.append((img_path, class_idx))

    return records, class_names


def split_records_stratified(
    records: List[Tuple[str, int]],
    seed: int = 42
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Splits records into 80% train, 10% validation, and 10% test sets,
    stratified by class index with a fixed seed.
    """
    paths = [r[0] for r in records]
    labels = [r[1] for r in records]

    # First split: 80% train, 20% temp (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        test_size=0.20,
        random_state=seed,
        stratify=labels
    )

    # Second split: Split the 20% temp equally into 10% val and 10% test
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=seed,
        stratify=temp_labels
    )

    train_records = list(zip(train_paths, train_labels))
    val_records = list(zip(val_paths, val_labels))
    test_records = list(zip(test_paths, test_labels))

    return train_records, val_records, test_records


class PlantDiseaseDataset(Dataset):
    """
    PyTorch Dataset that loads images lazily from disk.
    Ensures memory efficiency for large datasets by only loading batch items as needed.
    """

    def __init__(self, records: List[Tuple[str, int]], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.records[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    """
    Standard transforms for MobileNetV2:
    - Resize to 224x224.
    - Standard ImageNet normalization.
    - Training gets light augmentation: random flip, small rotation, brightness jitter.
    - Val/Test has no augmentation.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, eval_transform


def build_dataloaders(
    train_records: List[Tuple[str, int]],
    val_records: List[Tuple[str, int]],
    test_records: List[Tuple[str, int]],
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds PyTorch DataLoaders for train, val, and test splits.
    Default num_workers=0 avoids Windows multiprocessing spawn issues.
    """
    train_tf, eval_tf = get_transforms()

    train_dataset = PlantDiseaseDataset(train_records, transform=train_tf)
    val_dataset = PlantDiseaseDataset(val_records, transform=eval_tf)
    test_dataset = PlantDiseaseDataset(test_records, transform=eval_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader

