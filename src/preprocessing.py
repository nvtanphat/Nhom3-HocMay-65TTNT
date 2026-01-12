"""
Module Tiền xử lý dữ liệu cho bài toán Phân loại U não trên ảnh MRI
"""
import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

# Default settings (can be overridden)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def create_dataframe(data_dir):
    """
    Tạo DataFrame từ cấu trúc thư mục.
    """
    filepaths, labels = [], []
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                filepaths.append(os.path.join(class_path, img_name))
                labels.append(class_name)
    return pd.DataFrame({'filepath': filepaths, 'label': labels})


def resize_with_padding(image, target_size=(224, 224)):
    """
    Thay đổi kích thước ảnh nhưng giữ nguyên tỷ lệ khung hình (thêm viền đen).
    Giúp ảnh không bị méo khi đưa về hình vuông.
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas


def apply_clahe(image):
    """
    Áp dụng cân bằng histogram thích ứng (CLAHE) để tăng độ tương phản.
    Kỹ thuật này rất hiệu quả với ảnh y tế (X-quang, MRI) để làm rõ các chi tiết xương/mô.
    """
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def preprocess_image(image):
    """
    Hàm tiền xử lý cho mô hình CNN tự xây dựng và Xception.
    """
    enhanced = apply_clahe(image)
    return enhanced.astype(np.float32) / 127.5 - 1.0


def preprocess_image_resnet50(image):
    """
    Hàm tiền xử lý dành cho ResNet50
    """
    enhanced = apply_clahe(image).astype(np.float32)
    # Convert RGB to BGR
    enhanced = enhanced[..., ::-1]
    # Zero-center by ImageNet mean (BGR order)
    enhanced[..., 0] -= 103.939  # Blue
    enhanced[..., 1] -= 116.779  # Green
    enhanced[..., 2] -= 123.68   # Red
    return enhanced


def preprocess_image_pytorch(image):
    """
    Hàm tiền xử lý cho PyTorch model MTL.
    """
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    # ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # Apply normalization
    image = (image - mean) / std
    return image.astype(np.float32)





class CustomDataGenerator(tf.keras.utils.Sequence):
    """Generator dữ liệu load ảnh theo từng batch và tăng cường dữ liệu (Augmentation)."""
    
    def __init__(self, dataframe, batch_size=32, img_size=(224, 224), 
                 augment=False, shuffle=True, class_names=None, preprocess_func=None, grayscale=False):
        self.df = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle
        self.class_names = class_names or sorted(dataframe['label'].unique())
        self.n_classes = len(self.class_names)
        self.indexes = np.arange(len(self.df))
        self.preprocess_func = preprocess_func or preprocess_image
        self.grayscale = grayscale
        self.n_channels = 1 if grayscale else 3
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        # Trả về số lượng batch trong một epoch
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        # Lấy một batch dữ liệu
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_data = self.df.iloc[batch_indexes]
        
        X = np.zeros((len(batch_data), *self.img_size, self.n_channels), dtype=np.float32)
        y = np.zeros((len(batch_data), self.n_classes), dtype=np.float32)
        
        for i, (_, row) in enumerate(batch_data.iterrows()):
            img = cv2.imread(row['filepath'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = resize_with_padding(img, self.img_size)
            
            if self.augment:
                img = self._augment(img)
            
            X[i] = self.preprocess_func(img)
            label_idx = self.class_names.index(row['label'])
            y[i, label_idx] = 1.0
        return X, y
    
    def _augment(self, img):
        """Hàm thực hiện tăng cường dữ liệu ngẫu nhiên"""
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)
        
        angle = np.random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h), borderValue=0)
        
        factor = np.random.uniform(0.9, 1.1)
        img = np.clip(img * factor, 0, 255).astype(np.uint8)
        return img
    
    def on_epoch_end(self):
        """Được gọi sau khi kết thúc mỗi epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
