"""
Module Đánh giá Mô hình phân loại U não
Bao gồm: Các chỉ số đo lường, Ma trận nhầm lẫn (Confusion Matrix), Đường cong ROC, và hiển thị các dự đoán sai.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

# Re-export Grad-CAM functions from gradcam module
from gradcam import (
    generate_gradcam, apply_heatmap,
    show_cam_samples, save_cam_outputs
)


# ============================================================================
# CORE EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(model, test_generator, test_df, class_names):
    """Hàm đánh giá tổng quát: chạy dự đoán trên toàn bộ tập test và in báo cáo."""
    all_preds, all_labels = [], []
    
    print("Evaluating model...")
    for i in range(len(test_generator)):
        X_batch, y_batch = test_generator[i]
        all_preds.extend(model.predict(X_batch, verbose=0))
        all_labels.extend(y_batch)
    
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    y_pred, y_true = np.argmax(all_preds, axis=1), np.argmax(all_labels, axis=1)
    
    print(f'\nTest Accuracy: {np.mean(y_pred == y_true)*100:.2f}%')
    _print_roc_auc(all_labels, all_preds, class_names)
    
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, target_names=class_names))
    return y_true.tolist(), y_pred.tolist(), all_preds


def _print_roc_auc(all_labels, all_preds, class_names):
    """Hàm phụ trợ: Tính và in điểm ROC-AUC"""
    try:
        if len(class_names) == 2:
            print(f'ROC-AUC Score: {roc_auc_score(all_labels[:, 1], all_preds[:, 1]):.4f}')
        else:
            roc_auc = roc_auc_score(all_labels, all_preds, multi_class='ovr', average='macro')
            print(f'ROC-AUC Score (macro): {roc_auc:.4f}')
            roc_auc_per_class = roc_auc_score(all_labels, all_preds, multi_class='ovr', average=None)
            print('Per-class ROC-AUC:')
            for i, cls in enumerate(class_names):
                print(f'  {cls}: {roc_auc_per_class[i]:.4f}')
    except Exception as e:
        print(f'Could not calculate ROC-AUC: {e}')

# ===========================================================================

# PLOTTING FUNCTIONS


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir='output'):
    """Vẽ Ma trận nhầm lẫn (Confusion Matrix)"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    _save_and_show(output_dir, 'confusion_matrix.png')


def plot_roc_curve(y_true_labels, predictions, class_names, output_dir='output'):
    """Vẽ đường cong ROC cho từng lớp (One-vs-Rest)"""
    n_classes = len(class_names)
    y_true_labels = np.array(y_true_labels)
    y_true_onehot = np.eye(n_classes)[y_true_labels] if y_true_labels.ndim == 1 else y_true_labels
    
    predictions = np.array(predictions)
    fpr, tpr, roc_auc_dict = {}, {}, {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], predictions[:, i])
        roc_auc_dict[i] = auc(fpr[i], tpr[i])
    
    fpr['micro'], tpr['micro'], _ = roc_curve(y_true_onehot.ravel(), predictions.ravel())
    roc_auc_dict['micro'] = auc(fpr['micro'], tpr['micro'])
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc_dict[i]:.4f})')
    
    plt.plot(fpr['micro'], tpr['micro'], color='navy', linestyle='--', lw=2,
             label=f'Micro-average (AUC = {roc_auc_dict["micro"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_and_show(output_dir, 'roc_curve.png')
    return roc_auc_dict


def plot_training_history(history, output_dir='output'):
    """Vẽ biểu đồ quá trình huấn luyện (Accuracy và Loss theo Epoch)"""
    acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
    loss, val_loss = history.history['loss'], history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (train, val, title) in zip(axes, [(acc, val_acc, 'Accuracy'), (loss, val_loss, 'Loss')]):
        ax.plot(epochs, train, 'b-', label='Train')
        ax.plot(epochs, val, 'r-', label='Valid')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _save_and_show(output_dir, 'training_history.png')
    print(f'\nBest Train Accuracy: {max(acc)*100:.2f}%')
    print(f'Best Valid Accuracy: {max(val_acc)*100:.2f}%')


def show_wrong_predictions(model, test_generator, test_df, class_names, output_dir='output'):
    """Hiển thị các ảnh dự đoán sai"""
    all_preds, all_labels, all_filepaths = [], [], []
    
    for i in range(len(test_generator)):
        X_batch, y_batch = test_generator[i]
        all_preds.extend(model.predict(X_batch, verbose=0))
        all_labels.extend(np.argmax(y_batch, axis=1))
        start_idx = i * test_generator.batch_size
        all_filepaths.extend(test_df.iloc[start_idx:start_idx + len(y_batch)]['filepath'].values)
    
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    y_pred = np.argmax(all_preds, axis=1)
    wrong_indices = np.where(y_pred != all_labels)[0]
    
    print(f"Total wrong predictions: {len(wrong_indices)} / {len(all_labels)}")
    
    if len(wrong_indices) > 0:
        n_display = min(len(wrong_indices), 12)
        fig, axes = _create_grid(n_display)
        
        for i, idx in enumerate(wrong_indices[:n_display]):
            img = cv2.cvtColor(cv2.imread(all_filepaths[idx]), cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
            axes[i].set_title(
                f'True: {class_names[all_labels[idx]]}\n'
                f'Pred: {class_names[y_pred[idx]]} ({all_preds[idx][y_pred[idx]]*100:.1f}%)',
                color='red', fontweight='bold'
            )
            axes[i].axis('off')
        
        _hide_empty_axes(axes, n_display)
        plt.suptitle(f'Wrong Predictions ({len(wrong_indices)} images)', 
                     fontsize=14, fontweight='bold', color='red')
        plt.tight_layout()
        _save_and_show(output_dir, 'wrong_predictions.png')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _save_and_show(output_dir, filename):
    """Lưu biểu đồ vào thư mục và hiển thị lên màn hình"""
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_path}')


def _create_grid(n_items, cols=4):
    """Tạo lưới subplot động dựa trên số lượng item"""
    rows = max(1, (n_items + cols - 1) // cols)
    cols_actual = min(n_items, cols)
    fig, axes = plt.subplots(rows, cols_actual, figsize=(4*cols_actual, 4*rows))
    if n_items == 1:
        return fig, [axes]
    if rows == 1 or cols_actual == 1:
        return fig, list(np.atleast_1d(axes))
    return fig, list(axes.flatten())


def _hide_empty_axes(axes, n_display):
    """Ẩn các khung hình thừa nếu số lượng ảnh không lấp đầy lưới"""
    for j in range(n_display, len(axes)):
        axes[j].axis('off')



