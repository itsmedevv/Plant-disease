"""
evaluate.py - Test evaluation, per-class metrics, confusion matrix, and report generation.

Evaluates the trained MobileNetV2 model on the held-out test split (10%).
Outputs:
1. Overall Test Accuracy
2. Per-class Precision, Recall, and F1-score
3. Confusion matrix heatmap saved to 'outputs/confusion_matrix.png'
4. Full classification report saved to 'outputs/classification_report.txt'
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
from torchvision import models

from dataset import (
    find_color_dir,
    collect_image_records,
    split_records_stratified,
    PlantDiseaseDataset,
    get_transforms
)


def load_trained_model(model_path: str, num_classes: int, device: torch.device) -> nn.Module:
    """
    Instantiates MobileNetV2 with the target head and loads trained weights.
    """
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def get_test_records_and_classes(
    data_dir: str,
    models_dir: str
) -> Tuple[List[Tuple[str, int]], List[str]]:
    """
    Loads test records from 'models/test_records.json' if available,
    or reconstructs the exact split using 'models/classes.json' and dataset discovery.
    """
    models_p = Path(models_dir)
    test_meta_file = models_p / "test_records.json"
    classes_file = models_p / "classes.json"

    if not classes_file.exists():
        raise FileNotFoundError(f"Classes file not found at '{classes_file}'. Please train the model first.")

    with open(classes_file, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    if test_meta_file.exists():
        with open(test_meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            test_records = [(r[0], r[1]) for r in meta["test_records"]]
            return test_records, class_names

    # Fallback reconstruction
    color_dir = find_color_dir(data_dir)
    records, discovered_classes = collect_image_records(color_dir=color_dir, seed=42)
    _, _, test_records = split_records_stratified(records, seed=42)
    return test_records, class_names


def main():
    parser = argparse.ArgumentParser(description="Evaluate Plant Disease Detection model on Test Split")
    parser.add_argument("--model", type=str, default="models/model.pt", help="Path to trained model checkpoint")
    parser.add_argument("--classes", type=str, default="models/classes.json", help="Path to classes.json")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory containing model artifacts")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save evaluation outputs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print(" Plant Disease Detection - Evaluation Pipeline")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Model checkpoint: {args.model}")
    print("==================================================")

    # 1. Load test records and classes
    test_records, class_names = get_test_records_and_classes(args.data_dir, args.models_dir)
    num_classes = len(class_names)
    print(f"[Evaluation] Found {len(test_records)} test samples across {num_classes} classes.")

    # 2. Build test DataLoader
    _, eval_tf = get_transforms()
    test_dataset = PlantDiseaseDataset(test_records, transform=eval_tf)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # 3. Load Model
    model = load_trained_model(args.model, num_classes, device)

    # 4. Run Test Inference
    y_true = []
    y_pred = []

    print("[Evaluation] Running inference on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 5. Compute Metrics
    accuracy = accuracy_score(y_true, y_pred)
    target_names = [c.replace("___", " / ").replace("_", " ") for c in class_names]

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    print("\n--------------------------------------------------")
    print(f" Overall Test Accuracy: {accuracy*100:.2f}%")
    print("--------------------------------------------------")
    print("Classification Report:")
    print(report)

    # 6. Save outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Plant Disease Detection - Test Set Evaluation\n")
        f.write(f"Model Checkpoint: {args.model}\n")
        f.write(f"Test Samples: {len(y_true)}\n")
        f.write(f"Overall Accuracy: {accuracy*100:.2f}%\n\n")
        f.write("Per-Class Metrics:\n")
        f.write(report)
    print(f"[Saved] Classification report saved to: {report_path}")

    # 7. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    fig_size = max(8, int(num_classes * 0.8))
    plt.figure(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=True
    )
    plt.title(f"Confusion Matrix (Test Accuracy: {accuracy*100:.2f}%)", fontsize=14, pad=15)
    plt.xlabel("Predicted Label", fontsize=12, labelpad=10)
    plt.ylabel("True Label", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Saved] Confusion matrix plot saved to: {cm_path}")
    print("==================================================")


if __name__ == "__main__":
    main()

