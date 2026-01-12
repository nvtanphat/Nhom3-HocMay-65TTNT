"""
Ứng dụng Web phân loại U não trên ảnh MRI
Sử dụng thư viện Streamlit để tạo giao diện.
"""
import os
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import cv2

from config import CLASS_NAMES, MODELS, RESULTS_DIR, DATA_DIR, DISEASE_INFO
from utils import (
    load_model, predict, generate_gradcam, 
    apply_heatmap_overlay, read_image, get_sample_images,
    create_mask_overlay
)


# Page config
st.set_page_config(
    page_title="Phân Loại U Não",       # Tiêu đề trên tab trình duyệt
    layout="wide",                      # Dùng toàn màn hình thay vì cột giữa hẹp
    initial_sidebar_state="expanded"    # Dùng toàn màn hình thay vì cột giữa hẹp
)


def main():
    #----THANH BÊN (SIDEBAR)-----
    with st.sidebar:
        st.header("Cấu Hình Model")

        # Hộp chọn (Selectbox) để người dùng chọn Model muốn sử dụng
        st.write("Chọn Model:")
        model_type = st.selectbox("model_select", list(MODELS.keys()), label_visibility="collapsed")
        
        # Hiển thị thông tin Framework của model (PyTorch hoặc TensorFlow)
        config = MODELS[model_type]
        if config.get('is_pytorch', False):
            st.info("Framework: PyTorch")
            if config.get('is_multitask', False):
                st.success("Multi-task: Classification + Segmentation")
        else:
            st.info("Framework: TensorFlow/Keras")
        # Checkbox để bật chế độ so sánh tất cả các model cùng lúc
        compare_all = st.checkbox("So sánh tất cả Models", value=False)
        # Phần hiển thị thông tin về các loại bệnh
        st.write("")
        st.write("**Các class phân loại:**")
        for cls in CLASS_NAMES:
            st.write(f"- {cls.capitalize()}")
    
    # Load model
    model = load_cached_model(model_type)
    if model is None:
        st.error(f"Không tìm thấy model: {MODELS[model_type]['file']}")
        return
    
    # ---- GIAO DIỆN CHÍNH (MAIN AREA)-----

    # Chia giao diện thành 2 cột: Cột trái (Upload/Ảnh) - Cột phải (Kết quả
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.subheader("Chọn hình ảnh để phân tích")
        
        uploaded = st.file_uploader(
            "Kéo thả file vào đây",
            type=['jpg', 'jpeg', 'png'],
            help="Giới hạn 200MB - JPG, JPEG, PNG"
        )
        
        if uploaded is not None:
            image = read_image(uploaded.read())
            st.image(image, caption="Ảnh đã upload", use_column_width=True)
        
        # Model info in left column
        st.markdown("---")
        display_model_info_sidebar(model_type)
    
    with col_right:
        if uploaded is None:
            st.info("Upload ảnh để bắt đầu phân tích")
        else:
            # Predict button
            if st.button("Predict", type="primary", use_container_width=True):
                with st.spinner("Đang phân tích..."):
                    if compare_all:
                        results = compare_all_models(image)
                        display_comparison_results(results)
                    else:
                        result = predict(model, image, model_type)
                        display_single_result(result, model_type)


@st.cache_resource
def load_cached_model(model_type):
    """
    Wrapper hàm load_model với tính năng Cache của Streamlit.
    Giúp không phải load lại model nặng nề mỗi khi người dùng tương tác với giao diện.
    """
    return load_model(model_type)


def compare_all_models(image):
    """Hàm chạy vòng lặp qua tất cả model để so sánh"""
    results = []
    for model_type in MODELS.keys():
        model = load_cached_model(model_type)
        if model is not None:
            start_time = time.time()
            result = predict(model, image, model_type)
            elapsed = (time.time() - start_time) * 1000
            results.append({
                'Model': model_type,
                'Prediction': result['class'].capitalize(),
                'Confidence (%)': round(result['confidence'], 2),
                'Time (ms)': round(elapsed, 2),
                'probs': result['probabilities']
            })
    return results


def display_comparison_results(results):
    """"" Hàm chạy vòng lặp qua tất cả model để so sánh """""
    st.subheader("So sánh tất cả Models")
    
    # Create DataFrame for table
    df = pd.DataFrame([{
        'Model': r['Model'],
        'Prediction': r['Prediction'],
        'Confidence (%)': r['Confidence (%)'],
        'Time (ms)': r['Time (ms)']
    } for r in results])
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Chart
    st.subheader("Biểu đồ Confidence")
    plot_comparison_chart(results)


def display_single_result(result, model_type):
    """Hiển thị kết quả của một model"""
    st.subheader("Kết quả phân tích")
    
    # Result table
    df = pd.DataFrame([{
        'Model': model_type,
        'Prediction': result['class'].upper(),
        'Confidence (%)': round(result['confidence'], 2)
    }])
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.subheader("Biểu đồ Confidence")
    plot_probability_chart(result['probabilities'])
    
    # Segmentation for multi-task models
    if result.get('is_multitask', False):
        display_segmentation(result)
    else:
        # Grad-CAM section (for Keras models only)
        st.subheader("Grad-CAM Visualization")
        display_gradcam(result, model_type)
    
    # Disease information
    display_disease_info(result['class'])


def display_segmentation(result):
    """ Hiển thị mặt nạ phân đoạn (dành cho model Multi-task) """
    st.subheader("Phân đoạn vùng u (Segmentation)")
    
    mask = result.get('segmentation_mask')
    if mask is None:
        st.warning("Không có dữ liệu segmentation")
        return
    
    original_image = result['processed_image']
    
   # Tạo lớp phủ màu (overlay) lên ảnh gốc
    overlay = create_mask_overlay(original_image, mask, alpha=0.5, color=(255, 0, 0))
    
    # Tạo lớp phủ màu (overlay) lên ảnh gốc
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Ảnh gốc**")
        st.image(original_image, use_column_width=True)
    
    with col2:
        st.write("**Segmentation Mask**")
        # Tô màu cho mask
        mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
        mask_colored = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
        mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
        st.image(mask_colored, use_column_width=True)
    
    with col3:
        st.write("**Overlay**")
        st.image(overlay, use_column_width=True)
    
    # Tô màu cho mask
    tumor_ratio = np.mean(mask > 0.5) * 100
    if tumor_ratio > 1:
        st.info(f"Vùng u được phát hiện chiếm khoảng {tumor_ratio:.1f}% diện tích ảnh")
    else:
        st.success("Không phát hiện vùng u đang có trong ảnh")
    
    st.caption("Vùng màu đỏ là vùng model dự đoán có u não")

def display_disease_info(predicted_class):
    """Hiển thị thông tin y khoa dựa trên lớp dự đoán"""
    info = DISEASE_INFO.get(predicted_class)
    if info is None:
        return

    st.subheader("Thông tin bệnh")

    # Hiển thị mức độ nghiêm trọng với màu sắc tương ứng
    severity = info['severity']
    if 'Nghiem trong' in severity:
        st.error(f"**Mức độ:** {severity}")
    elif 'Binh thuong' in severity:
        st.success(f"**Mức độ:** {severity}")
    else:
        st.warning(f"**Mức độ:** {severity}")
    
    # Tên bệnh và mô tả
    st.markdown(f"### {info['name']}")
    st.write(info['description'])
    
    # Liệt kê triệu chứng
    if info['symptoms']:
        st.markdown("**Triệu chứng thường gặp:**")
        for symptom in info['symptoms']:
            st.write(f"• {symptom}")
    
    # Recommendation
    st.info(f"**Khuyến nghị:** {info['recommendation']}")
    
    # Cảnh báo miễn trừ trách nhiệm 
    st.caption("⚠️ Đây chỉ là kết quả dự đoán của AI. Vui lòng tham khảo bác sĩ để có chẩn đoán chính xác.")


def display_model_info(model_type):
    """ hiển thị thông tin chi tiết model"""
    config = MODELS.get(model_type)
    if config is None:
        return

    st.subheader("Thông tin Model")

    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Độ chính xác (Accuracy)", f"{config.get('accuracy', 'N/A')}%")
        st.write(f"**Kích thước ảnh:** {config['img_size'][0]}x{config['img_size'][1]}")
    
    with col2:
        st.write(f"**Người phát triển:** {config.get('developer', 'N/A')}")
        st.write(f"**Framework:** TensorFlow/Keras")

    st.write(f"**Mô tả:** {config.get('description', 'Không có mô tả')}")

    # Dataset info
    st.markdown("---")
    st.write("**Dữ liệu huấn luyện:**")
    st.write("• Dataset: Brain Tumor MRI Dataset")
    st.write("• Số lượng class: 4 (Glioma, Meningioma, No Tumor, Pituitary)")
    st.write("• Tổng số ảnh: ~7000 ảnh MRI")
def display_model_info_sidebar(model_type):
    """"hiển thị thông tin chi tiết model"""
    config = MODELS.get(model_type)
    if config is None:
        return
    
    st.subheader(f"Model: {model_type}")
    
    # Model details
    st.write(f"**Kích thước:** {config['img_size'][0]}x{config['img_size'][1]}")
    st.write(f"**Developer:** {config.get('developer', 'N/A')}")
    
    # Description
    with st.expander("Mô tả model"):
        st.write(config.get('description', 'Không có mô tả'))
    
    # Dataset info
    with st.expander("Thông tin Dataset"):
        st.write("• Brain Tumor MRI Dataset")
        st.write("• 4 classes: Glioma, Meningioma, No Tumor, Pituitary")
        st.write("• ~7000 ảnh MRI")


def display_gradcam(result, model_type):
    """Hiển thị trực quan hóa Grad-CAM (Bản đồ nhiệt)"""
    model = load_cached_model(model_type)
    if model is None:
        st.warning("Kh không thể tải model cho Grad-CAM")
        return
    
    with st.spinner("Đang tạo Grad-CAM..."):
        heatmap = generate_gradcam(model, result['batch'])
    
    if heatmap is None:
        st.warning("không thể tạo Grad-CAM cho model này")
        return
    
   # Tạo Grad-CAM
    overlay = apply_heatmap_overlay(result['processed_image'], heatmap, alpha=0.4)
    
    # Display in 3 columns
    col1, col2, col3 = st.columns(3)

    # Hiển thị 3 cột: Ảnh gốc, Heatmap màu, và Ảnh chồng lớp
    with col1:
        st.write("**Ảnh gốc**")
        st.image(result['processed_image'], use_column_width=True)
    
    with col2:
        st.write("**Grad-CAM Heatmap**")
        # Chuyển heatmap xám sang màu (JET colormap: Xanh->Đỏ)
        heatmap_resized = cv2.resize(heatmap, (result['processed_image'].shape[1], result['processed_image'].shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        st.image(heatmap_colored, use_column_width=True)
    
    with col3:
        st.write("**Overlay**")
        st.image(overlay, use_column_width=True)
    
    st.caption("Vùng màu đỏ/vàng là vùng model tập trung để đưa ra dự đoán")


def plot_comparison_chart(results):
    """Vẽ biểu đồ cột so sánh độ tin cậy của các model"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    models = [r['Model'] for r in results]
    confidences = [r['Confidence (%)'] for r in results]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(models)]
    bars = ax.bar(models, confidences, color=colors, width=0.6)
    
    ax.set_ylabel('Confidence (%)')
    ax.set_ylim(0, 110)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, conf in zip(bars, confidences):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{conf:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def plot_probability_chart(probabilities):
    """ Vẽ biểu đồ xác suất chi tiết cho 1 model"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    classes = [c.capitalize() for c in probabilities.keys()]
    probs = list(probabilities.values())
    
    colors = ['#3498db'] * len(classes)
    max_idx = probs.index(max(probs))
    colors[max_idx] = '#e74c3c'
    
    bars = ax.bar(classes, probs, color=colors, width=0.6)
    
    ax.set_ylabel('Confidence (%)')
    ax.set_ylim(0, 110)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{prob:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


if __name__ == "__main__":
    main()
