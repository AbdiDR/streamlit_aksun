import streamlit as st
import os
import pickle
import cv2
import numpy as np
from scipy.spatial.distance import cdist

# Path ke model
BOW_FILE_PICKLE = "model/sift_bow_dictionary.pkl"
SCALER_FILE_PICKLE = "model/sift_SCALER_WS_FILE_PICKLE_SIFT_full.pkl"
SVM_FILE_PICKLE = "model/sift_svm_with_sift_model_claude_full.pkl"

# Konfigurasi folder
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Fungsi-fungsi classifier
def load_file_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def center_handwritten_image(image, canvas_size=(192, 192)):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    cropped = gray[y:y+h, x:x+w]
    canvas = np.ones(canvas_size, dtype=np.uint8) * 255
    start_x = max((canvas_size[1] - w) // 2, 0)
    start_y = max((canvas_size[0] - h) // 2, 0)
    canvas[start_y:start_y+h, start_x:start_x+w] = cropped
    return canvas

def preprocess_image(filepath):
    image = cv2.imread(filepath)
    img = cv2.resize(image, (192, 192))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    centered = center_handwritten_image(img)
    equalized = cv2.equalizeHist(centered)
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(equalized, None)
    img_with_keypoints = cv2.drawKeypoints(equalized, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    original_name = os.path.splitext(os.path.basename(filepath))[0]
    preprocessed_path = os.path.join("static/results", f"processed_image_{original_name}.jpg")
    cv2.imwrite(preprocessed_path, img_with_keypoints)
    return preprocessed_path, equalized

def create_feature_bow(image_descriptor, bow, num_cluster):
    features = np.zeros(num_cluster, dtype=float)
    if image_descriptor is not None:
        distance = cdist(image_descriptor, bow)
        argmin = np.argmin(distance, axis=1)
        for j in argmin:
            features[j] += 1.0
    return features

def extract_features(image):
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(image, None)
    bow = load_file_pickle(BOW_FILE_PICKLE)
    num_clusters = bow.shape[0]
    if descriptors is not None:
        return create_feature_bow(descriptors, bow, num_clusters)
    return np.zeros(num_clusters)

def predict(filepath):
    preprocessed_path, preprocessed_image = preprocess_image(filepath)
    features = extract_features(preprocessed_image)
    scaler = load_file_pickle(SCALER_FILE_PICKLE)
    scaled_features = scaler.transform([features])
    svm_model = load_file_pickle(SVM_FILE_PICKLE)
    probabilities = svm_model.predict_proba(scaled_features)[0]
    label = svm_model.classes_[np.argmax(probabilities)]
    confidence = round(np.max(probabilities) * 100, 2)
    return preprocessed_path, label, confidence

# Konfigurasi Streamlit
st.set_page_config(
    page_title="Klasifikasi Aksara Sunda",
    page_icon="✍️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        background-color: #f3f4f6;
    }
    .upload-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        max-width: 32rem;
        margin: auto;
    }
    .prediction-text {
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #0d6efd;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
    }
    .stButton > button:hover {
        background-color: #0b5ed7;
    }
    .image-container {
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        padding: 0.5rem;
        margin-bottom: 1rem;
        background-color: white;
    }
    .upload-text {
        color: #6b7280;
        text-align: center;
        padding: 2rem;
    }
    .header-text {
        color: #111827;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subheader-text {
        color: #374151;
        text-align: center;
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Judul aplikasi
st.markdown("<h1 class='header-text'>Klasifikasi Aksara Sunda</h1>", unsafe_allow_html=True)

# Container utama
with st.container():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        # st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader("", type=['jpg'], key="file_uploader")
        
        if uploaded_file is not None:
            # Simpan file
            filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Proses prediksi
                preprocessed_path, prediction, confidence = predict(filepath)
                
                # Tampilkan gambar
                img_col1, img_col2 = st.columns(2)
                
                with img_col1:
                    st.markdown("<p class='subheader-text'>Gambar Asli</p>", unsafe_allow_html=True)
                    st.image(uploaded_file, use_container_width=True)
                
                with img_col2:
                    st.markdown("<p class='subheader-text'>Hasil Keypoint</p>", unsafe_allow_html=True)
                    st.image(preprocessed_path, use_container_width=True)
                
                # Hasil prediksi
                st.markdown(f"""
                    <div class='prediction-text'>
                        <p>Karakter: {prediction}</p>
                        <p>Confidence Level: {confidence}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Bersihkan file
                if os.path.exists(filepath):
                    os.remove(filepath)
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan dalam pemrosesan gambar: {str(e)}")
        
        else:
            # Tampilan default
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.markdown("<p class='subheader-text'>Gambar Asli</p>", unsafe_allow_html=True)
                st.markdown("<div class='image-container'><p class='upload-text'>Masukkan Gambar</p></div>", unsafe_allow_html=True)
            
            with img_col2:
                st.markdown("<p class='subheader-text'>Hasil Keypoint</p>", unsafe_allow_html=True)
                st.markdown("<div class='image-container'><p class='upload-text'>Belum Ada Hasil</p></div>", unsafe_allow_html=True)
            
            st.markdown("""
                <div class='prediction-text'>
                    <p>Karakter: -</p>
                    <p>Confidence Level: -</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# Footer informasi
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #4b5563; font-size: 0.875rem;'>
    <p>Upload gambar dengan format .jpg untuk klasifikasi</p>
    <p>Model menggunakan SIFT dan SVM untuk klasifikasi Aksara Sunda</p>
</div>
""", unsafe_allow_html=True)