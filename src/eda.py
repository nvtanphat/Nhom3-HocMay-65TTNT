"""
Module Phân tích Khám phá Dữ liệu (EDA) cho bộ dữ liệu MRI U não
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from preprocessing import resize_with_padding, IMG_SIZE


def analyze_images(df, name='Dataset'):
    """
    Phân tích các thuộc tính kỹ thuật của ảnh (định dạng file, kích thước, số kênh màu)
    """
    sizes = []
    extensions = []
    channels = []
    
    for filepath in df['filepath'].values:
        ext = filepath.split('.')[-1].lower()
        extensions.append(ext)
        
        img = cv2.imread(filepath)
        if img is not None:
            h, w = img.shape[:2]
            sizes.append((w, h))
            channels.append(img.shape[2] if len(img.shape) == 3 else 1)
    
    size_counter = Counter(sizes)
    ext_counter = Counter(extensions)
    channel_counter = Counter(channels)
    
    print(f'{name} Image Analysis')
    print(f'Total images: {len(df)}')
    print(f'\nImage Extensions:')
    for ext, count in ext_counter.most_common():
        print(f'  .{ext}: {count} ({count/len(df)*100:.1f}%)')
    print(f'\nImage Sizes (WxH):')
    for size, count in size_counter.most_common(5):
        print(f'  {size[0]}x{size[1]}: {count} ({count/len(df)*100:.1f}%)')
    if len(size_counter) > 5:
        print(f'  ... and {len(size_counter)-5} more sizes')
    print(f'\nImage Channels:')
    for ch, count in channel_counter.most_common():
        ch_name = 'RGB' if ch == 3 else ('Grayscale' if ch == 1 else f'{ch} channels')
        print(f'  {ch_name}: {count} ({count/len(df)*100:.1f}%)')
    print()


def plot_distribution(train_df, valid_df, test_df, output_dir='results'):
    """
    Vẽ biểu đồ phân bố số lượng ảnh của từng lớp trong các tập Train/Valid/Test
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    for ax, (name, df) in zip(axes, [('Train', train_df), ('Valid', valid_df), ('Test', test_df)]):
        df['label'].value_counts().plot(kind='bar', ax=ax, color=colors)
        ax.set_title(f'{name} Distribution')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_path}')


def show_sample_images(df, output_dir='results', n_samples=3):
    """
    Hiển thị ngẫu nhiên một số ảnh mẫu từ mỗi lớp
    """
    classes = df['label'].unique()
    fig, axes = plt.subplots(len(classes), n_samples, figsize=(4*n_samples, 4*len(classes)))
    
    for i, cls in enumerate(classes):
        class_df = df[df['label'] == cls].sample(n_samples)
        for j, (_, row) in enumerate(class_df.iterrows()):
            img = cv2.imread(row['filepath'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax = axes[i, j] if len(classes) > 1 else axes[j]
            ax.imshow(img)
            ax.set_title(f'{cls}\n{img.shape[1]}x{img.shape[0]}', fontsize=10)
            ax.axis('off')
    plt.suptitle('Sample Images from Each Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'sample_images.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_path}')


def visualize_augmentation(df, class_names, output_dir='results', n_samples=4):
    """Trực quan hóa quá trình tăng cường dữ liệu (Data Augmentation)"""
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    samples = df.sample(n_samples)
    
    for idx, (_, row) in enumerate(samples.iterrows()):
        img = cv2.imread(row['filepath'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original = resize_with_padding(img, IMG_SIZE)
        
        axes[idx, 0].imshow(original)
        axes[idx, 0].set_title(f'Original\n{row["label"]}', fontsize=10)
        axes[idx, 0].axis('off')
        
        for j in range(3):
            aug_img = original.copy()
            if np.random.random() > 0.5:
                aug_img = cv2.flip(aug_img, 1)
            angle = np.random.uniform(-15, 15)
            h, w = aug_img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            aug_img = cv2.warpAffine(aug_img, M, (w, h), borderValue=0)
            
            axes[idx, j+1].imshow(aug_img)
            axes[idx, j+1].set_title(f'Augmented {j+1}', fontsize=10)
            axes[idx, j+1].axis('off')
    
    plt.suptitle('Original vs Augmented Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'augmentation_samples.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_path}')
