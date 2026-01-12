
## Giới thiệu
Dự án này tập trung vào việc áp dụng các kỹ thuật Học sâu (Deep Learning) để tự động phân loại và phân đoạn các loại u não từ ảnh MRI. Hệ thống hỗ trợ chẩn đoán 4 loại tình trạng:
- **Glioma** (U thần kinh đệm)
- **Meningioma** (U màng não)
- **Pituitary** (U tuyến yên)
- **No Tumor** (Không có u)

Dự án cung cấp giao diện web trực quan để người dùng có thể tải ảnh lên và nhận kết quả chẩn đoán cũng như hình ảnh phân đoạn vùng u (segmentation) hoặc bản đồ nhiệt (Grad-CAM).

Link dataset (Brain Tumor MRI Dataset): https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Link dataset (🧠 BRISC 2025): https://www.kaggle.com/datasets/briscdataset/brisc2025/

## Cài đặt

### Yêu cầu hệ thống
- Python >= 3.8
- Khuyên dùng môi trường ảo (Virtual Environment)

### Cài đặt thư viện
Chạy lệnh sau để cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## Hướng dẫn Sử dụng

### 1. Huấn luyện Mô hình
Sử dụng script `src/main.py` để huấn luyện các mô hình.

- **Chạy mặc định (CNN)**:
  ```bash
  python src/main.py
  ```

- **Chạy mô hình Xception**:
  ```bash
  python src/main.py --model xception
  ```

- **Chạy mô hình ResNet50**:
  ```bash
  python src/main.py --model resnet50
  ```

### 2. Chạy Ứng dụng Web
Sử dụng Streamlit để khởi chạy giao diện web:

```bash
streamlit run web/app.py
```

Sau khi chạy lệnh, truy cập vào đường dẫn được hiển thị trên terminal (thường là `http://localhost:8501`) để sử dụng ứng dụng.

## Cấu trúc Thư mục

- `data/`: Chứa dữ liệu ảnh MRI (Training và Testing).
- `model/`: Chứa các file trọng số mô hình đã huấn luyện (.keras, .pth).
- `notebook/`: Chứa các Jupyter Notebook dùng để thử nghiệm và phân tích.
  - `01-cnnpro99-nguyenvantanphat.ipynb`
  - `03-xception-nguyenvantanphat.ipynb`
  - `03-xception-tanphatxhoangloc.ipynb`
  - `05-mutiltaskxception-nguyenvantanphat.ipynb`
  - `06-edaandevalueclassicationbrics2025-nguyenvantanphat.ipynb`
- `results/`: Chứa kết quả đánh giá, biểu đồ training, và ảnh visualizations.
- `src/`: Mã nguồn chính cho việc huấn luyện và đánh giá mô hình.
  - `main.py`: Script huấn luyện chính.
  - `eda.py`: Phân tích khám phá dữ liệu.
  - `model_*.py`: Định nghĩa kiến trúc các mô hình.
  - `preprocessing.py`: Các hàm tiền xử lý ảnh.
  - `gradcam.py`: Tạo bản đồ nhiệt Grad-CAM.
- `web/`: Mã nguồn cho ứng dụng web Streamlit.
  - `app.py`: File chính của ứng dụng web.
  - `config.py`: Cấu hình hệ thống.

## Tác giả
- Nguyễn Văn Tấn Phát - 2351267275
- Phạm Thành Doanh 
- Nguyễn Hoàng Lộc 
