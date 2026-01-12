"""
Script chính cho bài toán Phân loại U não trên ảnh MRI
Cách sử dụng: 
    python main.py                    # Chạy mặc định: mô hình CNN
    python main.py --model xception   # Chạy mô hình Xception
"""
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# Windows multiprocessing fix
import multiprocessing
multiprocessing.freeze_support()

import tensorflow as tf
from sklearn.model_selection import train_test_split

from preprocessing import create_dataframe, CustomDataGenerator, BATCH_SIZE, preprocess_image_resnet50
from eda import analyze_images, plot_distribution, show_sample_images, visualize_augmentation
from evaluation import (
    evaluate_model, plot_confusion_matrix, plot_training_history,
    show_wrong_predictions, plot_roc_curve
)
from gradcam import show_cam_samples, save_cam_outputs

# ============================================================================
# Cấu hình cho các loại mô hình
MODEL_CONFIGS = {
    'cnn': {
        'name': 'model_nguyenvantanphat',
        'img_size': (224, 224),
    },
    'xception': {
        'name': 'model1_nguyenvantanphat', 
        'img_size': (299, 299),
    },
    'resnet50': {
        'name': 'model_phamthanhdoanh',
        'img_size': (224, 224),
    },

}


def parse_args():
    """Hàm xử lý tham số dòng lệnh (Command Line Arguments)"""
    parser = argparse.ArgumentParser(description='Brain Tumor MRI Classification')
    parser.add_argument('--model', type=str, default='cnn', 
                        choices=['cnn', 'xception', 'resnet50'],
                        help='Model type: cnn, xception, or resnet50')
    return parser.parse_args()



def main():
    args = parse_args() # Lấy tham số dòng lệnh
    config = MODEL_CONFIGS[args.model] 
    
    # Thiết lập đường dẫn thư mục
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_NAME = config['name']
    IMG_SIZE = config['img_size']
    MODEL_PATH = os.path.join(BASE_DIR, 'model', f'{MODEL_NAME}.keras')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results', MODEL_NAME)
    
    print(f'TensorFlow version: {tf.__version__}')
    print(f'GPU available: {len(tf.config.list_physical_devices("GPU")) > 0}')
    print(f'Model: {args.model.upper()} | Image size: {IMG_SIZE}')
    
    # Data paths
    train_dir = os.path.join(BASE_DIR, 'data', 'brainmri', 'Training')
    test_dir = os.path.join(BASE_DIR, 'data', 'brainmri', 'Testing')
    
    # Load dữ liệu vào DataFrame
    train_df = create_dataframe(train_dir)
    test_df = create_dataframe(test_dir)
    print(f'Training: {len(train_df)}, Testing: {len(test_df)}')
    
    # Split validation/test
    valid_df, test_df = train_test_split(
        test_df, train_size=0.5, random_state=42, stratify=test_df['label']
    )
    print(f'Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}')
    
    # Create directories
    os.makedirs(os.path.join(BASE_DIR, 'model'), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # EDA
    analyze_images(train_df, 'Training')
    analyze_images(test_df, 'Testing')
    plot_distribution(train_df, valid_df, test_df, RESULTS_DIR)
    show_sample_images(train_df, RESULTS_DIR, n_samples=4)
    
    # Create data generators
    class_names = sorted(train_df['label'].unique())
    
    # Lấy hàm tiền xử lý và thang độ xám dựa trên từng loại mô hình
    grayscale = config.get('grayscale', False)
    if args.model == 'resnet50':
        preprocess_func = preprocess_image_resnet50
    else:
        preprocess_func = None
    
    train_gen = CustomDataGenerator(train_df, BATCH_SIZE, IMG_SIZE, augment=True, 
                                    shuffle=True, class_names=class_names,
                                    preprocess_func=preprocess_func, grayscale=grayscale)
    valid_gen = CustomDataGenerator(valid_df, BATCH_SIZE, IMG_SIZE, augment=False,
                                    shuffle=False, class_names=class_names,
                                    preprocess_func=preprocess_func, grayscale=grayscale)
    test_gen = CustomDataGenerator(test_df, BATCH_SIZE, IMG_SIZE, augment=False,
                                   shuffle=False, class_names=class_names,
                                   preprocess_func=preprocess_func, grayscale=grayscale)
    # Trực quan hóa xem ảnh sau Augment
    visualize_augmentation(train_df, class_names, RESULTS_DIR, n_samples=4)

    
   # Logic Huấn luyện (Training Logic)
   # Kiểm tra: Nếu đã có file model (.keras) thì load lên dùng luôn, KHÔNG train lại.
    if os.path.exists(MODEL_PATH):
        print(f'\n[INFO] Found existing model at {MODEL_PATH}')
        print('[INFO] Skipping training, loading model...')
        
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        print('\n[INFO] No existing model found. Starting training...')
        # Chiến lược cho từng loại model
        if args.model == 'cnn':
            from model_nguyenvantanphat import build_cnn_model, compile_model, train_model
            model = build_cnn_model(input_shape=(*IMG_SIZE, 3), num_classes=len(class_names))
            model = compile_model(model, learning_rate=0.001)
            model.summary()
            # Train 1 giai đoạn
            history = train_model(model, train_gen, valid_gen, epochs=70, model_path=MODEL_PATH)
            plot_training_history(history, RESULTS_DIR)
            
        elif args.model == 'xception':
            from model1_nguyenvantanphat import build_xception_model, train_xception
            model, base_model = build_xception_model(input_shape=(*IMG_SIZE, 3), 
                                                      num_classes=len(class_names))
            model.summary()
            # Train 2 giai đoạn (Transfer Learning):
            # 1. Train top layers (head)
            # 2. Fine-tune toàn bộ model với learning rate nhỏ
            history1, history2 = train_xception(model, base_model, train_gen, valid_gen, 
                                                 model_path=MODEL_PATH)
            plot_training_history_combined(history1, history2, RESULTS_DIR)
        
        elif args.model == 'resnet50':
            from model_phamthanhdoanh import build_resnet50_model, train_resnet50
            model, base_model = build_resnet50_model(input_shape=(*IMG_SIZE, 3), 
                                                      num_classes=len(class_names))
            model.summary()
            # Train 2 giai đoạn (Transfer Learning):
            # 1. Train top layers (head)
            # 2. Fine-tune toàn bộ model với learning rate nhỏ
            history1, history2 = train_resnet50(model, base_model, train_gen, valid_gen, 
                                                 model_path=MODEL_PATH)
            plot_training_history_combined(history1, history2, RESULTS_DIR)
        

        # Load lại model tốt nhất đã lưu trong quá trình train (Checkpoint)
        model = tf.keras.models.load_model(MODEL_PATH)
    
    # Evaluate
    y_true, y_pred, predictions = evaluate_model(model, test_gen, test_df, class_names)
    plot_confusion_matrix(y_true, y_pred, class_names, RESULTS_DIR)
    plot_roc_curve(y_true, predictions, class_names, RESULTS_DIR)
    show_wrong_predictions(model, test_gen, test_df, class_names, RESULTS_DIR)
    
    # CAM visualizations (Unified for all models)
    print(f'\n[INFO] Generating CAM visualizations for {args.model.upper()}...')
    
    # Grad-CAM samples
    show_cam_samples(model, test_df, class_names, cam_type='Grad-CAM',
                     output_dir=RESULTS_DIR, n_samples=12, img_size=IMG_SIZE,
                     preprocess_func=preprocess_func)
    

   
    # Save Grad-CAM for ALL test images
    print('\n[INFO] Generating Grad-CAM for full test set...')
    save_cam_outputs(model, test_df, class_names, cam_type='Grad-CAM',
                     img_size=IMG_SIZE, preprocess_func=preprocess_func,
                     output_dir=RESULTS_DIR, model_name=MODEL_NAME)
    
    print(f'\nResults saved to: {RESULTS_DIR}')
    print(f'Model saved at: {MODEL_PATH}')
    print(f'Total params: {model.count_params():,}')


def plot_training_history_combined(history1, history2, output_dir):
    """Vẽ biểu đồ training history cho 2 giai đoạn"""
    import matplotlib.pyplot as plt
    
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    epochs = range(1, len(acc) + 1)
    phase1_end = len(history1.history['accuracy'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(epochs, acc, 'b-', label='Train')
    axes[0].plot(epochs, val_acc, 'r-', label='Valid')
    axes[0].axvline(x=phase1_end, color='g', linestyle='--', label='Fine-tuning')
    axes[0].set_title('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, loss, 'b-', label='Train')
    axes[1].plot(epochs, val_loss, 'r-', label='Valid')
    axes[1].axvline(x=phase1_end, color='g', linestyle='--', label='Fine-tuning')
    axes[1].set_title('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150)
    plt.show()
    print(f'Saved: {output_dir}/training_history.png')


if __name__ == '__main__':
    main()
