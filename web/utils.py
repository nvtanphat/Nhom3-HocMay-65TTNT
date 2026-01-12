"""
Utility functions for image processing and model inference
Supports both TensorFlow/Keras and PyTorch models
"""
import os
import sys
import numpy as np
import cv2

# Add parent directory to path for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import tensorflow as tf
import torch

from src.preprocessing import (
    resize_with_padding, 
    preprocess_image, 
    preprocess_image_resnet50,
    preprocess_image_pytorch,
)
from src.gradcam import generate_gradcam as gradcam_generate, apply_heatmap
from config import MODEL_DIR, MODELS, CLASS_NAMES
from multitask_model import load_multitask_model

# Mapping preprocessing functions
PREPROCESS_FUNCS = {
    'default': preprocess_image,
    'resnet50': preprocess_image_resnet50,
    'pytorch': preprocess_image_pytorch,
}


def get_preprocess_func(model_type):
    """Lấy hàm tiền xử lý phù hợp dựa trên tên model"""
    preprocess_type = MODELS[model_type]['preprocess']
    return PREPROCESS_FUNCS.get(preprocess_type, preprocess_image)


def load_model(model_type):
    """Hàm tải model ,nhận diện framework"""
    config = MODELS[model_type]
    model_path = os.path.join(MODEL_DIR, config['file'])
    
    if not os.path.exists(model_path):
        return None
    
    # Check if PyTorch model
    if config.get('is_pytorch', False):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return load_multitask_model(model_path, device=device)
    else:
        # Load Keras model
        return tf.keras.models.load_model(model_path)


def predict(model, image, model_type):
    """
    Hàm dự đoán cốt lõi (Core Inference Logic)
    """
    config = MODELS[model_type]
    
    # Check if multi-task model
    if config.get('is_multitask', False):
        return predict_multitask(model, image, config)
    
    # Standard classification prediction
    preprocess_func = get_preprocess_func(model_type)
    
    # Preprocess
    img_resized = resize_with_padding(image, config['img_size'])
    img_batch = np.expand_dims(preprocess_func(img_resized), axis=0)
    
    # Predict
    predictions = model.predict(img_batch, verbose=0)[0]
    pred_idx = np.argmax(predictions)
    
    return {
        'class': CLASS_NAMES[pred_idx],
        'confidence': float(predictions[pred_idx] * 100),
        'probabilities': dict(zip(CLASS_NAMES, (predictions * 100).tolist())),
        'processed_image': img_resized,
        'batch': img_batch,
        'is_multitask': False
    }


def predict_multitask(model, image, config):
    """
    Predict both classification and segmentation using multi-task model
    """
    # Preprocess image
    img_resized = resize_with_padding(image, config['img_size'])
    img_preprocessed = preprocess_image_pytorch(img_resized)
    
    # Convert to PyTorch tensor [B, C, H, W]
    img_tensor = torch.from_numpy(img_preprocessed).permute(2, 0, 1).unsqueeze(0).float()
    
    # Get device from model
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        seg_mask, cls_logits = model(img_tensor)
    
    # Classification results
    probs = torch.softmax(cls_logits, dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    
    # Segmentation mask (apply sigmoid for binary mask)
    seg_mask_np = torch.sigmoid(seg_mask[0, 0]).cpu().numpy()
    
    return {
        'class': CLASS_NAMES[pred_idx],
        'confidence': float(probs[pred_idx] * 100),
        'probabilities': dict(zip(CLASS_NAMES, (probs * 100).tolist())),
        'processed_image': img_resized,
        'segmentation_mask': seg_mask_np,
        'is_multitask': True,
        'batch': None  # Not used for PyTorch models
    }


def create_mask_overlay(image, mask, alpha=0.5, color=(255, 0, 0)):
    """
        Tạo lớp phủ màu đỏ lên vùng u não (Segmentation Overlay)
    """
    # Resize mask to match image if needed
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # Threshold mask
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Create colored mask image
    colored_mask = np.zeros_like(image)
    colored_mask[:, :] = color
    
    # Create result by blending
    result = image.copy()
    
    # Only overlay where mask is 1
    mask_3d = np.stack([binary_mask] * 3, axis=-1)
    blended = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    result = np.where(mask_3d == 1, blended, image)
    
    return result


def generate_gradcam(model, img_batch):
    """Generate Grad-CAM heatmap (for Keras models only)"""
    try:
        return gradcam_generate(model, img_batch)
    except Exception:
        return None


def apply_heatmap_overlay(image, heatmap, alpha=0.4):
    """Apply heatmap overlay to image"""
    return apply_heatmap(image, heatmap, threshold=0.0, alpha=alpha)


def read_image(file_bytes):
    """Read image from bytes"""
    image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_sample_images(data_dir, class_name, count=5):
    """Get sample images from dataset"""
    class_path = os.path.join(data_dir, 'Testing', class_name)
    if not os.path.exists(class_path):
        return []
    
    samples = []
    for f in os.listdir(class_path)[:count]:
        img = cv2.imread(os.path.join(class_path, f))
        if img is not None:
            samples.append({
                'path': os.path.join(class_path, f),
                'name': f,
                'image': cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            })
    return samples
