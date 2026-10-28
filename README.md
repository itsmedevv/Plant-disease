# 🌿 Plant Disease Detection for Farmers

A deep-learning prototype designed to assist farmers and agricultural students in diagnosing crop diseases early from smartphone leaf photographs, understanding model decisions via Grad-CAM visual heatmaps, and receiving cautious, practical agronomic guidance.

---

## 📌 Problem & Overview

Crop diseases cause substantial yield loss worldwide. Smallholder farmers often struggle with early disease detection and lack immediate access to certified agronomists or plant pathology clinics. 

This project provides:
1. **Accurate Disease Classification**: Lightweight transfer learning with **MobileNetV2** pretrained on ImageNet.
2. **Visual Explainability (Grad-CAM)**: Heatmaps highlighting the exact lesions and spots that drove the diagnosis.
3. **Safety Guardrail**: Suppresses uncertain predictions if model confidence is below **60%**, advising the farmer to retake a clear photo.
4. **Actionable Agronomic Guidance**: Practical cultural and management advice from an advisory database (`advice.json`) paired with clear disclaimers to consult local agricultural officers.

---

## 🏗️ Architecture & Technical Highlights

```
Leaf Photo (RGB) ──> [Resize 224x224 & Normalize] ──> MobileNetV2 Backbone ──> Class Logits / Probabilities
                                                              │
                                                  [Grad-CAM Hook Layer]
                                                  (features[-1] Conv2D)
                                                              │
                                                              ▼
                                                   Attention Heatmap Overlay
```

- **Model Backbone**: `MobileNetV2` with ImageNet weights. The depthwise-separable convolution design ensures lightweight inference, making it suitable for edge devices and mobile apps.
- **Dataset**: [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) (`data/color`). The pipeline auto-detects nested paths (e.g. `data/archive/plantvillage dataset/color`), ignoring grayscale and segmented images.
- **DataLoader & Memory Safety**: Images are indexed as file paths and loaded on-the-fly in mini-batches. No bulk dataset loading into RAM.
- **Deterministic Stratification**: 80% train / 10% validation / 10% test split stratified by class with a fixed random seed (`seed=42`).
- **Data Augmentation**: Training set applies light augmentation (random horizontal flip, small rotation ±15°, brightness jitter ±20%). Validation and test sets use clean resizing and normalization.

---

## 📂 Project Structure

```text
plantdisease/
├── .gitignore               # Excludes data/, venv/, models/*.pt, outputs/
├── requirements.txt         # Package dependencies (PyTorch, Streamlit, etc.)
├── dataset.py               # Auto-detection, lazy loading, stratified splits, DataLoaders
├── train.py                 # Training script with CLI arguments (--crop, --max_per_class, --epochs)
├── evaluate.py              # Test split metrics, classification report, confusion matrix plot
├── gradcam.py               # Grad-CAM implementation for MobileNetV2
├── advice.json              # Curated agricultural advice for diseases
├── app.py                   # Streamlit interactive web application
└── README.md                # Documentation & usage guide
```

---

## 🚀 Setup & Installation

### 1. Create Virtual Environment & Activate

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

With CUDA (recommended if NVIDIA GPU is present):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Or standard CPU installation:
```bash
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

### Step 1: Quick Test Run (Smoke Test)
Train on **Tomato** classes with a maximum of 300 images per class for 2 epochs:

```bash
python train.py --crop Tomato --max_per_class 300 --epochs 2
```
*Outputs saved: `models/model.pt` (best checkpoint), `models/classes.json` (class names), `models/test_records.json` (test split metadata).*

### Step 2: Full Dataset Training (All 38 Classes)
To train on all 38 PlantVillage classes across all crops:

```bash
python train.py --epochs 5 --batch_size 32 --lr 0.0003
```

Optional flags:
- `--crop <CropName>`: e.g. `--crop Potato`, `--crop Apple`, `--crop Grape`.
- `--max_per_class <N>`: caps the number of samples per class for faster experimentation.
- `--epochs <N>`: number of training epochs (default: `5`).
- `--batch_size <N>`: batch size (default: `32`).

---

## 📈 Evaluation & Results

Evaluate the model against the held-out 10% test split:

```bash
python evaluate.py
```

This generates:
- **Terminal output**: Overall accuracy and per-class precision, recall, and F1-score.
- **`outputs/classification_report.txt`**: Complete per-class breakdown.
- **`outputs/confusion_matrix.png`**: High-resolution heatmap of actual vs. predicted labels.

---

## 🔍 Standalone Grad-CAM Heatmap

You can generate a Grad-CAM explanation directly from the command line:

```bash
python gradcam.py --image "data/archive/plantvillage dataset/color/Tomato___Early_blight/sample.JPG" --output "outputs/gradcam_sample.png"
```

---

## 🌐 Running the Streamlit Web Application

Launch the farmer web interface:

```bash
streamlit run app.py
```

### App Features:
1. **Leaf Upload**: Upload any `.jpg` or `.png` leaf photograph.
2. **Top Predictions**: Displays the primary predicted diagnosis along with top-3 probabilities.
3. **Safety Guardrail**: If the top prediction is below **60%**, a warning banner instructs the user: *"Not sure, please retake a clear photo of a single leaf"*.
4. **Side-by-Side Grad-CAM**: Visual proof showing which leaf regions influenced the decision.
5. **Practical Advisory**: Pathogen cause, symptoms, cultural sanitation steps, and reminders to consult local agricultural extension officers.

---

## ⚠️ Limitations

> **Lab Conditions vs. Real-World Fields**:
> - The **PlantVillage** dataset consists of single leaves harvested and photographed against uniform gray or black backgrounds under controlled laboratory lighting.
> - In real agricultural field conditions, leaves are photographed attached to plants with complex backgrounds (soil, overlapping foliage, weeds, hands), varying sunlight, deep shadows, and potential physical damage (dust, insect bites, multiple simultaneous diseases).
> - As a result, model accuracy on field photos will be noticeably lower than on clean benchmark images. Users should treat this tool as a preliminary screening aid and always verify findings with a certified agricultural extension officer or local plant clinic.

