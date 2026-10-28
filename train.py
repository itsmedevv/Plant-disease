"""
train.py - MobileNetV2 transfer learning for Plant Disease Detection.

Key features:
1. Flexible CLI arguments: --crop, --max_per_class, --epochs, --batch_size, --lr.
2. Auto-detects data directory and loads images lazily in mini-batches.
3. 80/10/10 stratified split by class with a fixed reproducible seed.
4. MobileNetV2 transfer learning with ImageNet pretrained weights.
5. Light augmentation on training set only.
6. Saves the best model checkpoint to 'models/model.pt' and class mapping to 'models/classes.json'.
"""

import argparse
import json
import os
from pathlib import Path
import time
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from dataset import (
    find_color_dir,
    collect_image_records,
    split_records_stratified,
    build_dataloaders
)


def build_model(num_classes: int) -> nn.Module:
    """
    Initializes MobileNetV2 with ImageNet pretrained weights and modifies
    the final classification head for the target number of plant disease classes.
    """
    # Load pretrained MobileNetV2
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    # Replace final linear layer in model.classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> tuple[float, float]:
    """
    Runs a single training epoch through the DataLoader.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate_val(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """
    Evaluates model performance on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 for Plant Disease Detection")
    parser.add_argument("--crop", type=str, default=None, help="Filter for a specific crop (e.g. 'Tomato')")
    parser.add_argument("--max_per_class", type=int, default=None, help="Maximum number of images per class")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoaders (default: 32)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 0.0003)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save model artifacts")
    args = parser.parse_args()

    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f" Plant Disease Detection - Training Pipeline")
    print(f" Device in use: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Parameters: crop={args.crop}, max_per_class={args.max_per_class}, epochs={args.epochs}, lr={args.lr}")
    print(f"==================================================")

    # 2. Locate dataset
    color_dir = find_color_dir(args.data_dir)
    print(f"[Dataset] Using color directory: {color_dir}")

    # 3. Collect image records and class names
    records, class_names = collect_image_records(
        color_dir=color_dir,
        crop=args.crop,
        max_per_class=args.max_per_class,
        seed=args.seed
    )
    num_classes = len(class_names)
    print(f"[Dataset] Found {len(records)} total images across {num_classes} classes.")
    for idx, cname in enumerate(class_names):
        count = sum(1 for _, c_idx in records if c_idx == idx)
        print(f"  - [{idx:02d}] {cname}: {count} images")

    # 4. Stratified 80/10/10 split
    train_records, val_records, test_records = split_records_stratified(records, seed=args.seed)
    print(f"[Split] Train: {len(train_records)} | Validation: {len(val_records)} | Test: {len(test_records)}")

    # 5. Build DataLoaders (images loaded lazily in batches)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        batch_size=args.batch_size,
        num_workers=0
    )

    # 6. Initialize Model, Loss, Optimizer, and Scheduler
    model = build_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Prepare destination directory
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save class names and test records metadata immediately
    classes_file = save_path / "classes.json"
    with open(classes_file, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"[Artifacts] Saved class names to {classes_file}")

    test_meta_file = save_path / "test_records.json"
    with open(test_meta_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_records": test_records,
            "classes": class_names,
            "seed": args.seed,
            "crop": args.crop,
            "max_per_class": args.max_per_class
        }, f, indent=2)
    print(f"[Artifacts] Saved test set records to {test_meta_file}")

    # 7. Training Loop
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_file = save_path / "model.pt"

    start_time = time.time()
    print("\nStarting Training...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_val(model, val_loader, criterion, device)
        scheduler.step()

        epoch_duration = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Time: {epoch_duration:.1f}s"
        )

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_file)
            print(f"  --> Saved new best model to {best_model_file} (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc*100:.2f}%)")

    total_time = time.time() - start_time
    print(f"\n==================================================")
    print(f" Training Complete in {total_time/60:.2f} minutes!")
    print(f" Best Validation Loss: {best_val_loss:.4f} | Best Validation Acc: {best_val_acc*100:.2f}%")
    print(f" Saved Best Model: {best_model_file}")
    print(f"==================================================")


if __name__ == "__main__":
    main()

