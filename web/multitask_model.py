"""
Multi-Task Model Architecture for Brain Tumor Classification + Segmentation
Kiến trúc mô hình Đa nhiệm: Phân loại + Phân đoạn U não
Sử dụng EfficientNet-B3 làm Encoder, kết hợp U-Net Decoder và cơ chế chú ý CBAM
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ChannelAttention(nn.Module):
    """
    Chú ý Kênh (Channel Attention)
    Mục tiêu: Xác định xem 'Kênh' (feature map) nào chứa thông tin quan trọng.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False), 
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """
    Chú ý Không gian (Spatial Attention)
    Mục tiêu: Xác định 'Vị trí' (pixel) nào trong ảnh là quan trọng (ví dụ: vị trí khối u).
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Kết hợp tuần tự: Channel Attention trước -> Spatial Attention sau
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))


class MultiTaskModel(nn.Module):
    """
    Mô hình chính xử lý đồng thời 2 nhiệm vụ:
    1. Segmentation (Phân đoạn): Tìm vị trí khối u.
    2. Classification (Phân loại): Xác định loại u.
    """
    def __init__(self, num_classes=4, encoder_name="efficientnet-b3"):
        super().__init__()

        # Khởi tạo U-Net with EfficientNet-B3 encoder
        self.seg_model = smp.Unet(
            encoder_name=encoder_name, 
            encoder_weights=None,  # Weights loaded from checkpoint
            in_channels=3, 
            classes=1              # Output là 1 kênh (Binary Mask: Có u hoặc không)
        )
        
        # Decoder attention modules
        decoder_channels = [256, 128, 64, 32, 16] # Kênh của các lớp decoder
        self.decoder_attentions = nn.ModuleList([CBAM(ch) for ch in decoder_channels]) # Attention cho phần Decoder (Segmentation)
        
        # Classification branch
        encoder_channels = self.seg_model.encoder.out_channels[-1]
        self.cls_attention = CBAM(encoder_channels) # Attention cho phần Classification
        # Classification Head (Đầu ra phân loại)
        self.pool = nn.AdaptiveAvgPool2d(1)          # Global Average Pooling
        self.classifier = nn.Sequential(
            nn.Flatten(),           
            nn.BatchNorm1d(encoder_channels), 
            nn.Dropout(0.5),        
            nn.Linear(encoder_channels, 512), 
            nn.SiLU(), 
            nn.BatchNorm1d(512), 
            nn.Dropout(0.4),
            nn.Linear(512, 256), 
            nn.SiLU(), 
            nn.Dropout(0.3), 
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
            Luồng dữ liệu đi qua mạng (Forward Pass)
        """
        # Lấy các đặc trưng từ encoder
        features = self.seg_model.encoder(x)

         # --- SEGMENTATION BRANCH ---
        decoder_out = self.seg_model.decoder(features)         # Đưa các đặc trưng vào decoder của U-Net
        decoder_out = self.decoder_attentions[-1](decoder_out) # Áp dụng CBAM Attention vào đầu ra của decoder
        seg_mask = self.seg_model.segmentation_head(decoder_out) # Đưa qua segmentation head
        # ---CLASSIFICATION BRANCH ---
        cls_features = self.cls_attention(features[-1])          # Áp dụng CBAM Attention vào đặc trưng cuối cùng của encoder
        cls_logits = self.classifier(self.pool(cls_features))   #  Pooling và đưa vào Classifier
        return seg_mask, cls_logits
    
    def freeze_encoder(self):
        """ Đóng băng encoder (không cập nhật trọng số) """
        for p in self.seg_model.encoder.parameters():
            p.requires_grad = False
    
    def unfreeze_encoder(self):
        """ Mở đóng băng encoder weights cho fine-tuning"""
        for p in self.seg_model.encoder.parameters():
            p.requires_grad = True


def load_multitask_model(model_path, device='cpu'):
    """
    Load multi-task model from checkpoint
    """
    model = MultiTaskModel(num_classes=4, encoder_name="efficientnet-b3")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model
