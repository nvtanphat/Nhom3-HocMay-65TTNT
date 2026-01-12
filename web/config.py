"""
Configuration settings for the web application
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'brainmri')

# Class names
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

DISEASE_INFO = {
    'glioma': {
        'name': 'U Thần Kinh Đệm (Glioma)',
        'description': (
            'Glioma là loại u não phổ biến nhất, xuất phát từ các tế bào thần kinh đệm '
            '(glial cells) – các tế bào có vai trò hỗ trợ và bảo vệ neuron trong não.'
        ),
        'severity': 'Nghiêm trọng',
        'symptoms': [
            'Đau đầu kéo dài, tăng dần theo thời gian',
            'Co giật, động kinh',
            'Buồn nôn, nôn mửa',
            'Thay đổi tính cách hoặc hành vi',
            'Suy giảm trí nhớ và khả năng tập trung',
            'Yếu hoặc tê liệt một bên cơ thể'
        ],
        'recommendation': (
            'Cần gặp bác sĩ chuyên khoa Thần kinh càng sớm càng tốt. '
            'Nên chụp MRI bổ sung và thực hiện sinh thiết để xác định chính xác loại glioma.'
        )
    },

    'meningioma': {
        'name': 'U Màng Não (Meningioma)',
        'description': (
            'Meningioma là khối u phát triển từ màng não (meninges) – lớp màng bao quanh '
            'não và tủy sống. Phần lớn là u lành tính và phát triển chậm.'
        ),
        'severity': 'Trung bình – Thường lành tính',
        'symptoms': [
            'Đau đầu',
            'Giảm thị lực hoặc nhìn đôi',
            'Giảm thính lực hoặc ù tai',
            'Suy giảm trí nhớ',
            'Yếu tay hoặc chân',
            'Co giật'
        ],
        'recommendation': (
            'Cần theo dõi định kỳ với bác sĩ chuyên khoa. '
            'Nếu khối u nhỏ và không gây triệu chứng, có thể chỉ cần theo dõi. '
            'Trường hợp u lớn hoặc gây biến chứng có thể cần phẫu thuật.'
        )
    },

    'notumor': {
        'name': 'Không Phát Hiện U',
        'description': (
            'Không phát hiện dấu hiệu của khối u trong hình ảnh MRI não.'
        ),
        'severity': 'Bình thường',
        'symptoms': [],
        'recommendation': (
            'Kết quả cho thấy không có khối u. '
            'Tuy nhiên, nếu vẫn xuất hiện các triệu chứng bất thường, '
            'bạn nên thăm khám bác sĩ để được tư vấn thêm.'
        )
    },

    'pituitary': {
        'name': 'U Tuyến Yên (Pituitary Tumor)',
        'description': (
            'U tuyến yên phát triển tại tuyến yên – một tuyến nội tiết quan trọng nằm '
            'ở đáy não. Phần lớn là u lành tính (adenoma) nhưng có thể ảnh hưởng đến '
            'việc sản xuất hormone.'
        ),
        'severity': 'Trung bình',
        'symptoms': [
            'Đau đầu',
            'Rối loạn thị lực (thu hẹp tầm nhìn)',
            'Mệt mỏi kéo dài không rõ nguyên nhân',
            'Thay đổi cân nặng bất thường',
            'Rối loạn kinh nguyệt (ở nữ)',
            'Giảm ham muốn tình dục'
        ],
        'recommendation': (
            'Cần xét nghiệm hormone và theo dõi với bác sĩ chuyên khoa Nội tiết. '
            'Tùy vào kích thước và loại u, phương pháp điều trị có thể là dùng thuốc '
            'hoặc phẫu thuật.'
        )
    }
}

# Model configurations
MODELS = {
    'CNN': {
        'file': 'model_nguyenvantanphat.keras',
        'img_size': (224, 224),
        'preprocess': 'default',
        'results_folder': 'model_nguyenvantanphat',
        'description': (
            'Mô hình CNN tự thiết kế với 3 khối Convolution, '
            'Batch Normalization và Dropout. '
            'Được tối ưu cho bài toán phân loại ảnh MRI não.'
        ),
        'developer': 'Nguyễn Văn Tấn Phát',
    },

    'Xception': {
        'file': 'model1_nguyenvantanphat.keras',
        'img_size': (299, 299),
        'preprocess': 'default',
        'results_folder': 'model1_nguyenvantanphat',
        'description': (
            'Mô hình Transfer Learning từ Xception (pretrained trên ImageNet). '
            'Xception sử dụng depthwise separable convolutions, '
            'phù hợp và hiệu quả với dữ liệu ảnh y tế.'
        ),
        'developer': 'Nguyễn Văn Tấn Phát',
    },

    'ResNet50': {
        'file': 'model_phamthanhdoanh.keras',
        'img_size': (224, 224),
        'preprocess': 'resnet50',
        'results_folder': 'model_phamthanhdoanh',
        'description': (
            'Mô hình Transfer Learning từ ResNet50 với skip connections. '
            'Kiến trúc 50 lớp giúp mô hình học được các đặc trưng phức tạp của khối u não.'
        ),
        'developer': 'Phạm Thành Doanh',
    },

    'MultiTask-EfficientNetB3': {
        'file': 'best_final.pth',
        'img_size': (320, 320),
        'preprocess': 'pytorch',
        'results_folder': 'multitask_efficientnetb3',
        'description': (
            'Mô hình đa nhiệm: EfficientNet-B3 làm encoder kết hợp U-Net decoder '
            'và cơ chế CBAM Attention. '
            'Hỗ trợ đồng thời Classification và Segmentation.'
        ),
        'developer': 'Multi-task Team',
        'is_pytorch': True,
        'is_multitask': True,
    }
}

