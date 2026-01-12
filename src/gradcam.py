"""
Module Trực quan hóa Grad-CAM
"""
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

# Ensure src directory is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import resize_with_padding, preprocess_image, IMG_SIZE


# ============================================================================
# MODEL DETECTION
# ============================================================================

def _detect_model_type(model):
    """ Hàm phát hiện loại kiến trúc mô hình dựa trên Layer names"""
    for layer in model.layers:
        name = layer.name.lower()
        if 'resnet' in name:
            return 'resnet50', layer
        if 'xception' in name:
            return 'xception', layer
        if 'efficientnet' in name:
            return 'efficientnet', layer
    return 'cnn', None


def _find_conv_layer(model, layer_name=None):
    """Tìm lớp Convolution (Tích chập) cuối cùng trong mô hình."""
    model_type, base_model = _detect_model_type(model)
    
    if layer_name and base_model:
        try:
            return base_model, base_model.get_layer(layer_name)
        except ValueError:
            pass
    
    # Auto-detect last conv layer
    if base_model:
        for layer in reversed(base_model.layers):
            if 'conv' in layer.name.lower() and hasattr(layer, 'output'):
                return base_model, layer
    
    # Fallback: search main model
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower() and hasattr(layer, 'output'):
            return None, layer
    
    raise ValueError("No conv layer found")


# ============================================================================
# GRAD-CAM GENERATION
# ============================================================================

def generate_gradcam(model, img_array, layer_name=None):
    """
    Hàm tạo Grad-CAM tổng quát cho mọi loại model.

    """
    model_type, base_model = _detect_model_type(model)
    
    if model_type == 'resnet50':
        return _generate_gradcam_transfer(model, base_model, img_array, 
                                          layer_name or 'conv5_block3_out')
    elif model_type == 'xception':
        return _generate_gradcam_transfer(model, base_model, img_array,
                                          layer_name or 'block14_sepconv2_act')
    elif model_type == 'efficientnet':
        return _generate_gradcam_transfer(model, base_model, img_array,
                                          layer_name or 'top_conv')
    else:
        return _generate_gradcam_cnn(model, img_array, layer_name)


def _generate_gradcam_transfer(model, base_model, img_array, layer_name):
    """Xử lý Grad-CAM cho các mô hình Transfer Learning."""
    conv_layer = base_model.get_layer(layer_name)
    conv_model = tf.keras.Model(base_model.input, conv_layer.output)
    
    img_tensor = tf.cast(img_array, tf.float32)
    
    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_tensor)
        tape.watch(conv_outputs)
        
        # Apply remaining layers manually
        x = tf.reduce_mean(conv_outputs, axis=[1, 2])  # GlobalAveragePooling2D
        
        # Find base model index and apply Dense layers
        base_idx = next(i for i, l in enumerate(model.layers) if l == base_model)
        for layer in model.layers[base_idx + 1:]:
            if 'pooling' not in layer.name.lower():
                x = layer(x, training=False)
        
        target_score = x[:, tf.argmax(x[0])]
    
    grads = tape.gradient(target_score, conv_outputs)
    if grads is None:
        return np.zeros((7, 7), dtype=np.float32)
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    
    max_val = tf.reduce_max(heatmap)
    return (heatmap / (max_val + 1e-10)).numpy()


def _generate_gradcam_cnn(model, img_array, layer_name=None):
    """Xử lý Grad-CAM cho CNN from-scratch"""
    _, conv_layer = _find_conv_layer(model, layer_name)
    
    grad_model = tf.keras.Model(model.input, [conv_layer.output, model.output])
    img_tensor = tf.cast(img_array, tf.float32)
    
    with tf.GradientTape() as tape:
        outputs = grad_model(img_tensor)
        conv_out = outputs[0]
        preds = outputs[1]
        # Handle case where preds might be a list
        if isinstance(preds, list):
            preds = preds[0]
        preds = tf.convert_to_tensor(preds)
        tape.watch(conv_out)
        pred_idx = tf.argmax(preds[0])
        loss = preds[0, pred_idx]
    
    grads = tape.gradient(loss, conv_out)
    if grads is None:
        return np.zeros((7, 7), dtype=np.float32)
    
    weights = tf.reduce_mean(grads, axis=(1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * weights[0], axis=-1)
    heatmap = tf.nn.relu(heatmap)
    
    max_val = tf.reduce_max(heatmap)
    return (heatmap / (max_val + 1e-10)).numpy()


# ============================================================================
# HEATMAP OVERLAY
# ============================================================================

def apply_heatmap(img, heatmap, threshold=0.3, alpha=0.4):
    """Apply heatmap overlay to image"""
    heatmap = np.where(heatmap < threshold, 0, heatmap)
    hm_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    jet_heatmap = (plt.cm.jet(hm_resized)[:, :, :3] * 255).astype(np.uint8)
    return cv2.addWeighted(img, 1-alpha, jet_heatmap, alpha, 0)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _save_and_show(output_dir, filename):
    """Save figure and show"""
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_path}')


def _create_grid(n_items, cols=4):
    """Create subplot grid"""
    rows = max(1, (n_items + cols - 1) // cols)
    cols_actual = min(n_items, cols)
    fig, axes = plt.subplots(rows, cols_actual, figsize=(4*cols_actual, 4*rows))
    if n_items == 1:
        return fig, [axes]
    if rows == 1 or cols_actual == 1:
        return fig, list(np.atleast_1d(axes))
    return fig, list(axes.flatten())


def _hide_empty_axes(axes, n_display):
    """Hide unused axes"""
    for j in range(n_display, len(axes)):
        axes[j].axis('off')


# ============================================================================
# GRAD-CAM VISUALIZATION FUNCTIONS
# ============================================================================

def _process_gradcam_sample(model, row, class_names, img_size, preprocess_func, layer_name, threshold):
    """Process single sample for Grad-CAM visualization"""
    img = cv2.cvtColor(cv2.imread(row['filepath']), cv2.COLOR_BGR2RGB)
    img_resized = resize_with_padding(img, img_size)
    x = preprocess_func(img_resized)[np.newaxis, ...]
    
    preds = model.predict(x, verbose=0)[0]
    pred_idx = np.argmax(preds)
    pred_label = class_names[pred_idx]
    true_label = row['label']
    is_correct = pred_label == true_label
    
    heatmap = generate_gradcam(model, x, layer_name)
    overlay = apply_heatmap(img_resized, heatmap, threshold)
    
    return {
        'overlay': overlay,
        'pred_label': pred_label,
        'true_label': true_label,
        'confidence': preds[pred_idx],
        'is_correct': is_correct
    }


def show_cam_samples(model, test_df, class_names, cam_type='Grad-CAM', 
                     output_dir='results', n_samples=12, threshold=0.3,
                     layer_name=None, img_size=None, preprocess_func=None):
    """Display Grad-CAM visualization samples"""
    img_size = img_size or IMG_SIZE
    preprocess_func = preprocess_func or preprocess_image
    
    samples = test_df.sample(n=min(n_samples, len(test_df)), random_state=42)
    fig, axes = _create_grid(len(samples))
    
    correct = 0
    for idx, (_, row) in enumerate(samples.iterrows()):
        result = _process_gradcam_sample(model, row, class_names, img_size, 
                                         preprocess_func, layer_name, threshold)
        correct += result['is_correct']
        
        axes[idx].imshow(result['overlay'])
        axes[idx].axis('off')
        status, color = ('✓', 'green') if result['is_correct'] else ('✗', 'red')
        axes[idx].set_title(f"{status} True: {result['true_label']}\n"
                           f"Pred: {result['pred_label']} ({result['confidence']*100:.1f}%)",
                           fontsize=10, color=color, fontweight='bold')
    
    _hide_empty_axes(axes, len(samples))
    accuracy = correct / len(samples) * 100
    plt.suptitle(f'Grad-CAM | Accuracy: {correct}/{len(samples)} ({accuracy:.1f}%)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    _save_and_show(output_dir, 'gradcam_samples.png')
    print(f'Sample Accuracy: {correct}/{len(samples)} ({accuracy:.1f}%)')


def save_cam_outputs(model, test_df, class_names, cam_type='Grad-CAM', threshold=0.3,
                     layer_name=None, img_size=None, preprocess_func=None, 
                     output_dir=None, model_name=None):
    """
    Lưu toàn bộ ảnh kết quả Grad-CAM ra thư mục.
    """
    img_size = img_size or IMG_SIZE
    preprocess_func = preprocess_func or preprocess_image
    
    # Create folder name with model name
    if model_name:
        cam_folder = f"gradcam_{model_name}"
    else:
        cam_folder = "gradcam_output"
    
    # Use output_dir if provided, otherwise save in current directory
    if output_dir:
        out_dir = os.path.join(output_dir, cam_folder)
    else:
        out_dir = cam_folder
    
    for cls in class_names:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)
    
    correct = 0
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Generating Grad-CAM'):
        result = _process_gradcam_sample(model, row, class_names, img_size,
                                         preprocess_func, layer_name, threshold)
        correct += result['is_correct']
        
        status = 'correct' if result['is_correct'] else 'wrong'
        fname = os.path.basename(row['filepath']).split('.')[0]
        save_path = os.path.join(out_dir, result['true_label'], 
                                 f"{fname}_{status}_{result['pred_label']}_{result['confidence']*100:.0f}.png")
        cv2.imwrite(save_path, cv2.cvtColor(result['overlay'], cv2.COLOR_RGB2BGR))
    
    accuracy = correct / len(test_df) * 100
    print(f"\nDone! Saved {len(test_df)} images to '{out_dir}/'")
    print(f"Test Accuracy: {correct}/{len(test_df)} ({accuracy:.2f}%)")
