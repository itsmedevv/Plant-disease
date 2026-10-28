"""
gradcam.py - Gradient-weighted Class Activation Mapping (Grad-CAM) for MobileNetV2.

Grad-CAM highlights the discriminative regions of an input image that led the neural network
to its prediction. For leaf disease classification, this shows the farmer/student exactly
which leaf lesions or spots the model focused on.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


class GradCAM:
    """
    Grad-CAM implementation designed for MobileNetV2 architectures.
    Hooks into the final convolutional feature layer to capture activations and gradients.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        # Default target layer is the last layer in MobileNetV2 features
        if target_layer is None:
            self.target_layer = self.model.features[-1]
        else:
            self.target_layer = target_layer

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out[0] contains the gradient of the score with respect to the feature map
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates a 2D normalized Grad-CAM heatmap for a single input tensor (1, C, H, W).

        Returns:
            heatmap: 2D numpy array of shape (H, W) in [0, 1]
            predicted_class: integer index of the predicted class
            confidence: softmax probability of the predicted class
        """
        # Ensure gradients are enabled for the backward pass
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        probabilities = F.softmax(logits, dim=1)

        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

        if target_class is None:
            target_class = predicted_class

        # Score for target class
        score = logits[0, target_class]
        score.backward()

        # Gradients: (1, Channels, H, W), Activations: (1, Channels, H, W)
        grads = self.gradients[0]  # (Channels, H, W)
        acts = self.activations[0]  # (Channels, H, W)

        # Global average pooling of gradients gives the importance weight of each channel
        weights = torch.mean(grads, dim=(1, 2))  # (Channels,)

        # Weighted combination of activation maps
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        # Apply ReLU to focus only on features that have a positive influence on the target class
        cam = F.relu(cam)

        # Convert to numpy and normalize to [0, 1]
        cam_np = cam.cpu().numpy()
        cam_max = cam_np.max()
        if cam_max > 0:
            cam_np = cam_np / cam_max
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, predicted_class, confidence

    def overlay_heatmap(
        self,
        pil_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resizes the heatmap to the original image dimensions and creates a visual overlay.

        Returns:
            overlay: RGB uint8 numpy image of original blended with heatmap
            colored_heatmap: RGB uint8 numpy image of colored heatmap alone
        """
        # Convert PIL image to RGB numpy array
        img_np = np.array(pil_image.convert("RGB"))
        h, w = img_np.shape[:2]

        # Resize heatmap to match original image size
        resized_cam = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        # Convert to 0-255 uint8 and apply colormap
        heatmap_uint8 = np.uint8(255 * resized_cam)
        colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

        # Blend original image and colored heatmap
        overlay = np.uint8(alpha * colored_rgb + (1 - alpha) * img_np)

        return overlay, colored_rgb


def load_model_for_inference(model_path: str, num_classes: int, device: torch.device) -> nn.Module:
    """
    Builds a MobileNetV2 architecture with the specified number of classes and loads saved weights.
    """
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def run_gradcam_on_image(
    image_path: str,
    model_path: str = "models/model.pt",
    classes_path: str = "models/classes.json",
    output_path: Optional[str] = "outputs/gradcam_sample.png"
):
    """
    Helper function to run Grad-CAM on a given image path and save the visual output.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load classes
    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    # Load model
    model = load_model_for_inference(model_path, len(class_names), device)

    # Load image
    pil_image = Image.open(image_path).convert("RGB")

    # Transform image for model input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Compute Grad-CAM
    gradcam = GradCAM(model)
    heatmap, pred_idx, conf = gradcam.generate_heatmap(input_tensor)
    overlay, colored_heatmap = gradcam.overlay_heatmap(pil_image, heatmap)

    predicted_class = class_names[pred_idx]
    print(f"Prediction: {predicted_class} (Confidence: {conf*100:.2f}%)")

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Save side-by-side comparison: [Original | Heatmap | Overlay]
        orig_np = np.array(pil_image)
        # Resize to match height if needed
        h = 400
        w = int(pil_image.width * (h / pil_image.height))
        orig_resized = cv2.resize(orig_np, (w, h))
        overlay_resized = cv2.resize(overlay, (w, h))
        side_by_side = np.hstack([orig_resized, overlay_resized])

        Image.fromarray(side_by_side).save(out_p)
        print(f"Grad-CAM explanation saved to: {out_p}")

    return predicted_class, conf, overlay


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Grad-CAM on a single leaf image.")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image (jpg/png)")
    parser.add_argument("--model", type=str, default="models/model.pt", help="Path to trained model")
    parser.add_argument("--classes", type=str, default="models/classes.json", help="Path to classes.json")
    parser.add_argument("--output", type=str, default="outputs/gradcam_sample.png", help="Path to save overlay image")
    args = parser.parse_args()

    run_gradcam_on_image(args.image, args.model, args.classes, args.output)

