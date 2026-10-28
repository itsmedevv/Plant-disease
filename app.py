"""
app.py - Streamlit Web Application: Plant Disease Detection for Farmers.

Organized, farmer-first diagnostic dashboard featuring:
1. Flexible Input Methods: Upload Image, Camera Capture, or 1-Click Preloaded Samples.
2. Clear Diagnostic Status: Health status badges, certainty levels, and Top-3 probabilities.
3. Reliability Guardrail: < 60% confidence prompts a retake warning with photo tips.
4. Visual Explainability (Grad-CAM): Side-by-side comparison with color legend.
5. Structured Agronomic Guidance: Symptoms, Cultural Practices, Organic Solutions, and Official Disclaimers.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from gradcam import GradCAM


# --- Page Configuration ---
st.set_page_config(
    page_title="Plant Disease Detection for Farmers",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional, Clean Agricultural Theme CSS ---
st.markdown("""
<style>
    /* Global layout & typography */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    
    /* Header & Title Banner */
    .app-header {
        background: linear-gradient(135deg, #1e3a1e 0%, #2e7d32 100%);
        color: white;
        padding: 1.6rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
    }
    .app-header h1 {
        color: #ffffff;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: #e8f5e9;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
    }
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }
    .status-badge {
        background-color: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.35);
        color: #ffffff;
        padding: 3px 10px;
        border-radius: 16px;
        font-size: 0.82rem;
        font-weight: 500;
    }

    /* Section Cards */
    .section-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.4rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Diagnosis Banner Card */
    .diagnosis-card-healthy {
        background: #f0fdf4;
        border: 2px solid #86efac;
        border-radius: 10px;
        padding: 1.3rem;
        margin-bottom: 1rem;
    }
    .diagnosis-card-disease {
        background: #fff7ed;
        border: 2px solid #fdba74;
        border-radius: 10px;
        padding: 1.3rem;
        margin-bottom: 1rem;
    }
    .diag-label {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .diag-label-healthy { color: #166534; }
    .diag-label-disease { color: #c2410c; }
    .diag-name {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
        margin: 0.3rem 0;
    }

    /* Guardrail Warning Card */
    .warning-box {
        background-color: #fffbeb;
        border-left: 6px solid #f59e0b;
        border-radius: 8px;
        padding: 1.4rem;
        margin: 1.2rem 0;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.08);
    }
    .warning-box h3 {
        color: #92400e;
        margin-top: 0;
        font-size: 1.25rem;
    }

    /* Legend & Explanations */
    .cam-legend {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        margin-top: 0.6rem;
    }

    /* Advisory Box */
    .disclaimer-banner {
        background-color: #eff6ff;
        border-left: 5px solid #3b82f6;
        border-radius: 6px;
        padding: 1rem;
        color: #1e3a8a;
        font-size: 0.92rem;
        margin-top: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Cache Model & Resources ---
@st.cache_resource
def load_model_and_classes(model_path: str = "models/model.pt", classes_path: str = "models/classes.json"):
    """
    Caches model loading and classes list across user sessions.
    """
    if not Path(classes_path).exists():
        return None, None, f"Classes file not found at '{classes_path}'."
    if not Path(model_path).exists():
        return None, None, f"Trained model checkpoint not found at '{model_path}'."

    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(classes))

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model, classes, None


@st.cache_data
def load_advice(advice_path: str = "advice.json") -> Dict[str, Any]:
    """
    Loads agricultural disease guidance from advice.json.
    """
    if not Path(advice_path).exists():
        return {}
    with open(advice_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def get_sample_images() -> Dict[str, str]:
    """
    Finds sample images from the 'samples/' directory or 'data/' folder for quick 1-click testing.
    """
    samples = {}

    # Check dedicated lightweight samples directory first (works in cloud deployment)
    samples_dir = Path("samples")
    if samples_dir.exists():
        for file in sorted(samples_dir.glob("*.jpg")):
            class_name = file.stem
            display_name = format_class_name(class_name)
            samples[display_name] = str(file)
        if samples:
            return samples

    base_data = Path("data")
    if not base_data.exists():
        return samples

    # Search for color folder
    color_dirs = list(base_data.rglob("color"))
    if not color_dirs:
        return samples
    color_dir = color_dirs[0]

    # Pre-select representative tomato classes
    target_samples = [
        ("Tomato - Early Blight", "Tomato___Early_blight"),
        ("Tomato - Healthy Leaf", "Tomato___healthy"),
        ("Tomato - Late Blight", "Tomato___Late_blight"),
        ("Tomato - Bacterial Spot", "Tomato___Bacterial_spot"),
        ("Tomato - Leaf Mold", "Tomato___Leaf_Mold"),
        ("Tomato - Spider Mites", "Tomato___Spider_mites Two-spotted_spider_mite"),
        ("Tomato - Yellow Leaf Curl Virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"),
    ]

    for label, folder_name in target_samples:
        folder = color_dir / folder_name
        if folder.is_dir():
            files = list(folder.glob("*.JPG")) + list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
            if files:
                samples[label] = str(files[0])

    return samples


def format_class_name(raw_name: str) -> str:
    """
    Formats raw PlantVillage class names into human-readable strings.
    """
    if "___" in raw_name:
        crop, disease = raw_name.split("___", 1)
        crop_clean = crop.replace("_", " ").title()
        disease_clean = disease.replace("_", " ").title()
        return f"{crop_clean} - {disease_clean}"
    return raw_name.replace("_", " ").title()


def get_confidence_rating(prob: float) -> Tuple[str, str, str]:
    """
    Returns certainty label, badge color, and description.
    """
    if prob >= 0.85:
        return "High Certainty", "#15803d", "The model exhibits strong visual pattern alignment."
    elif prob >= 0.60:
        return "Moderate Certainty", "#d97706", "The model matched key features, but verify in field."
    else:
        return "Low Certainty (Unreliable)", "#dc2626", "Visual features are ambiguous or noisy."


def main():
    # --- Sidebar ---
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/plant-under-sun.png", width=64)
        st.title("Agronomy Assistant")
        st.write("Diagnostic decision-support tool for farmers and agriculture researchers.")

        st.markdown("---")
        st.subheader("⚙️ System Status")
        model, class_names, error_msg = load_model_and_classes()
        if error_msg:
            st.error("Model: Offline")
            st.caption(error_msg)
        else:
            device_name = "NVIDIA CUDA GPU" if torch.cuda.is_available() else "CPU"
            st.success(f"Model: Online ({len(class_names)} Classes)")
            st.caption(f"**Hardware:** {device_name}")
            st.caption(f"**Architecture:** MobileNetV2 (Pretrained)")

        st.markdown("---")
        st.subheader("📷 Photo Best Practices")
        st.markdown("""
        - 🌿 **Single Leaf**: Frame exactly one leaf.
        - ☀️ **Even Lighting**: Avoid strong direct glare or dark shadows.
        - 🎯 **Sharp Focus**: Keep disease lesions clear.
        - 🧹 **Clean Background**: Neutral ground or plain backdrop.
        """)

        st.markdown("---")
        with st.expander("📋 View Supported Disease Classes"):
            if class_names:
                for idx, c in enumerate(class_names, 1):
                    st.write(f"{idx}. {format_class_name(c)}")
            else:
                st.write("No model loaded.")

        st.markdown("---")
        with st.expander("⚠️ Dataset Limitations"):
            st.caption(
                "Trained on PlantVillage dataset images captured in controlled lab settings. "
                "Natural field conditions (direct sunlight, overlapping foliage, dust) "
                "may lower prediction accuracy. Always verify with local agricultural officers."
            )

    # --- Header Banner ---
    st.markdown("""
    <div class="app-header">
        <h1>🌿 Plant Disease Detection for Farmers</h1>
        <p>Instant crop leaf diagnosis, visual AI attention heatmaps (Grad-CAM), and actionable management guidelines.</p>
        <div class="badge-container">
            <span class="status-badge">⚡ Real-Time Inference</span>
            <span class="status-badge">🔍 Grad-CAM Explainability</span>
            <span class="status-badge">🛡️ 60% Safety Guardrail</span>
            <span class="status-badge">📋 Advisory Guidelines</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Check model readiness
    if error_msg:
        st.warning(f"⚠️ {error_msg}")
        st.info("👉 Please train the model first by running:\n`python train.py --crop Tomato --max_per_class 300 --epochs 2`")
        return

    advice_db = load_advice()
    device = next(model.parameters()).device

    # --- Step 1: Input Selection Card ---
    st.markdown('<div class="section-title">📥 Step 1: Provide a Leaf Photograph</div>', unsafe_allow_html=True)
    
    input_tab1, input_tab2, input_tab3 = st.tabs([
        "📂 Upload File",
        "📸 Take Photo with Camera",
        "🧪 Test with Sample Leaf"
    ])

    pil_image: Optional[Image.Image] = None
    image_source_label = ""

    with input_tab1:
        uploaded_file = st.file_uploader(
            "Upload an image of a plant leaf (.jpg, .jpeg, .png)",
            type=["jpg", "jpeg", "png"],
            key="file_uploader"
        )
        if uploaded_file is not None:
            try:
                pil_image = Image.open(uploaded_file).convert("RGB")
                image_source_label = f"Uploaded File: {uploaded_file.name}"
            except Exception as e:
                st.error(f"Error opening image: {e}")

    with input_tab2:
        camera_file = st.camera_input("Point camera at a single leaf and capture")
        if camera_file is not None:
            try:
                pil_image = Image.open(camera_file).convert("RGB")
                image_source_label = "Camera Capture"
            except Exception as e:
                st.error(f"Error opening camera image: {e}")

    with input_tab3:
        sample_dict = get_sample_images()
        if sample_dict:
            selected_sample = st.selectbox(
                "Choose a pre-loaded sample from the dataset to test immediately:",
                options=list(sample_dict.keys()),
                key="sample_selector"
            )
            if st.button("🔍 Load Selected Sample"):
                sample_path = sample_dict[selected_sample]
                try:
                    pil_image = Image.open(sample_path).convert("RGB")
                    image_source_label = f"Dataset Sample: {selected_sample}"
                    st.session_state["loaded_sample_img"] = pil_image
                    st.session_state["loaded_sample_label"] = image_source_label
                except Exception as e:
                    st.error(f"Error loading sample: {e}")

            if "loaded_sample_img" in st.session_state and pil_image is None:
                pil_image = st.session_state["loaded_sample_img"]
                image_source_label = st.session_state.get("loaded_sample_label", "Sample Leaf")
        else:
            st.info("No sample images found in data folder. Use the Upload tab to provide an image.")

    # Guard: If no image selected yet, show onboarding instructions
    if pil_image is None:
        st.markdown("---")
        st.info("💡 **Getting Started:** Select a method above—upload a leaf photo, take a picture with your camera, or click **Test with Sample Leaf** for an instant demo.")
        return

    # --- Image Preprocessing & Inference ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Compute Grad-CAM and Probabilities
    gradcam = GradCAM(model)
    heatmap, pred_idx, confidence = gradcam.generate_heatmap(input_tensor)
    overlay, colored_heatmap = gradcam.overlay_heatmap(pil_image, heatmap)

    predicted_raw_class = class_names[pred_idx]
    predicted_display_name = format_class_name(predicted_raw_class)
    is_healthy = "healthy" in predicted_raw_class.lower()

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)[0].cpu().numpy()

    top_k_indices = np.argsort(probabilities)[::-1][:min(3, len(class_names))]
    top_3 = [(class_names[idx], float(probabilities[idx])) for idx in top_k_indices]

    st.markdown("---")

    # --- Step 2: Diagnostic Results & Confidence Guardrail ---
    st.markdown('<div class="section-title">📊 Step 2: Diagnostic Evaluation</div>', unsafe_allow_html=True)

    CONFIDENCE_THRESHOLD = 0.60

    # Low Confidence Branch (< 60%)
    if confidence < CONFIDENCE_THRESHOLD:
        st.markdown(
            f"""
            <div class="warning-box">
                <h3>⚠️ Not sure, please retake a clear photo of a single leaf</h3>
                <p>The highest model confidence for this image is only <b>{confidence*100:.1f}%</b>, which is below our 60% reliability safety threshold.</p>
                <p><b>Why this happens:</b></p>
                <ul>
                    <li>The leaf might be blurry, poorly lit, or out of focus.</li>
                    <li>There may be distracting elements in the background (fingers, soil, multiple overlapping leaves).</li>
                    <li>The symptoms may be early or atypical.</li>
                </ul>
                <p><b>Recommended Action:</b> Please retake a well-lit photo focusing squarely on a single leaf against a plain background.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("🔍 View Tentative AI Analysis (Low Confidence Inspection)"):
            col1, col2 = st.columns(2)
            with col1:
                st.image(pil_image, caption=f"Analyzed Image ({image_source_label})", width="stretch")
            with col2:
                st.image(overlay, caption=f"Grad-CAM Heatmap (Tentative: {predicted_display_name})", width="stretch")

            st.write("#### Probabilities breakdown:")
            for c_name, prob in top_3:
                st.write(f"- **{format_class_name(c_name)}**: `{prob*100:.1f}%`")
                st.progress(prob)

        return

    # High Confidence Branch (>= 60%)
    cert_label, cert_color, cert_desc = get_confidence_rating(confidence)
    card_class = "diagnosis-card-healthy" if is_healthy else "diagnosis-card-disease"
    label_class = "diag-label-healthy" if is_healthy else "diag-label-disease"
    status_text = "HEALTHY CROP DETECTED" if is_healthy else "CROP DISEASE IDENTIFIED"

    diag_col, prob_col = st.columns([1.2, 1], gap="large")

    with diag_col:
        st.markdown(
            f"""
            <div class="{card_class}">
                <div class="diag-label {label_class}">{status_text}</div>
                <div class="diag-name">{predicted_display_name}</div>
                <div style="margin-top: 0.5rem; display: flex; align-items: center; gap: 0.8rem;">
                    <span style="font-size: 1.25rem; font-weight: 700; color: #1f2937;">{confidence*100:.1f}% Confidence</span>
                    <span style="background-color: {cert_color}; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.82rem; font-weight: 600;">{cert_label}</span>
                </div>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.88rem; color: #4b5563;">{cert_desc}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with prob_col:
        st.write("#### Top-3 Candidate Diseases")
        for rank, (c_name, prob) in enumerate(top_3, 1):
            disp = format_class_name(c_name)
            is_top = (rank == 1)
            prefix = "🥇" if rank == 1 else ("🥈" if rank == 2 else "🥉")
            st.write(f"{prefix} **{disp}** — `{prob*100:.1f}%`")
            st.progress(min(prob, 1.0))

    st.markdown("---")

    # --- Step 3: Visual Inspection & Explainability (Grad-CAM) ---
    st.markdown('<div class="section-title">🔍 Step 3: Visual AI Inspection (Grad-CAM Heatmap)</div>', unsafe_allow_html=True)
    st.write(
        "Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the exact parts of the leaf "
        "that caused the neural network to make this diagnosis."
    )

    cam_col1, cam_col2 = st.columns(2, gap="medium")
    with cam_col1:
        st.image(pil_image, caption=f"Original Leaf Photo ({image_source_label})", width="stretch")
    with cam_col2:
        st.image(overlay, caption=f"Grad-CAM Attention Overlay ({predicted_display_name})", width="stretch")

    st.markdown("""
    <div class="cam-legend">
        <b>Color Legend:</b>
        <span style="color:#dc2626; font-weight:600; margin-left:10px;">🔴 Red / Warm Colors:</span> Primary visual features (active disease lesions, spots, yellowing margins).
        <span style="color:#2563eb; font-weight:600; margin-left:15px;">🔵 Blue / Cool Colors:</span> Background leaf tissue ignored by the model.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Step 4: Agronomic Management Guide ---
    st.markdown('<div class="section-title">📋 Step 4: Farmer Advisory & Treatment Plan</div>', unsafe_allow_html=True)

    advice_info = advice_db.get(predicted_raw_class, advice_db.get("default", {}))

    adv_tab1, adv_tab2, adv_tab3 = st.tabs([
        "🔍 Pathogen & Symptoms",
        "🛡️ Cultural & Field Management",
        "📞 Official Advice & Precautions"
    ])

    with adv_tab1:
        st.markdown(f"**Identified Cause:** `{advice_info.get('cause', 'Biological pathogen or environmental factor')}`")
        st.write("#### Key Symptoms to Confirm in Field:")
        st.info(advice_info.get('symptoms', 'Inspect foliage carefully for discoloration, spotting, or leaf drop.'))

    with adv_tab2:
        st.write("#### Recommended Practical Steps:")
        tips = advice_info.get("advice", ["Inspect adjacent plants and maintain sanitation."])
        for tip in tips:
            st.markdown(f"- {tip}")

    with adv_tab3:
        disclaimer = advice_info.get(
            "disclaimer",
            "Always consult a local agricultural extension officer or certified agronomist before applying chemical treatments."
        )
        st.markdown(
            f"""
            <div class="disclaimer-banner">
                <h4>⚠️ Official Agronomic Protocol</h4>
                <p>{disclaimer}</p>
                <p><b>Recommended Local Extension Actions:</b></p>
                <ul>
                    <li>Take a physical leaf sample sealed in a clean plastic bag to your nearest district agriculture office or university extension center.</li>
                    <li>Confirm whether local quarantine alerts or community-wide treatment advisories are active in your area.</li>
                    <li>Never apply unverified chemical fungicides or pesticides without certified dosage guidance.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
