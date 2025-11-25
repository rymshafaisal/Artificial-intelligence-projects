# import required libraries
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

CLASS_NAMES= ['Audi', 'Hyundai Creta', 'Mahindra Scorpio', 'Rolls Royce', 'Swift', 'Tata Safari', 'Toyota Innova'] 

@st.cache_resource
def load_trained_model():
    return load_model("CarModel.h5")

def preprocess_image(image):
    image = image.resize((128,128))
    img_array = img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_car_class(model,image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    idx = np.argmax(predictions[0])
    return CLASS_NAMES[idx], float(predictions[0][idx])*100, predictions[0], idx

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Car Classifier",
    page_icon="🚗",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            color: #f5f5f5;
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3, h4 {
            color: #FFD700;
            text-align: center;
            font-weight: bold;
        }
        .card {
            background: rgba(255,255,255,0.05);
            padding: 25px;
            border-radius: 15px;
            border: 1px solid #FFD700;
            box-shadow: 0 4px 25px rgba(0,0,0,0.4);
            margin: 10px;
        }
        .stButton>button {
            background: #FFD700;
            color: black;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 24px;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background: #ff4b4b;
            color: white;
            transform: scale(1.05);
        }
        .footer {
            text-align: center;
            padding: 15px;
            margin-top: 40px;
            font-size: 14px;
            color: #bbb;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<h1>🚘 Car Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h4>Upload a car image and let AI recognize the brand & model instantly</h4>", unsafe_allow_html=True)
st.markdown("---")

# ---------- LOAD MODEL ----------
with st.spinner("🔄 Loading AI Model..."):
    model = load_trained_model()
if model is None:
    st.error("⚠️ Failed to load the model. Ensure CarModel.h5 exists in directory.")
    st.stop()
st.success("✅ Model Loaded Successfully")

# ---------- LAYOUT: TWO COLUMNS ----------
col1, col2 = st.columns([1,1])

# ----- LEFT BOX: Upload & Classify -----
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📂 Upload & Classify")
    uploaded_file = st.file_uploader("Upload a Car Image", type=['jpg','jpeg','png'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🚀 Classify Car"):
            with st.spinner("🔍 Analyzing Image..."):
                predicted_class, confidence, all_preds, pred_idx = predict_car_class(model, image)
            st.session_state.results = (predicted_class, confidence, all_preds, pred_idx, image)
    st.markdown("</div>", unsafe_allow_html=True)

# ----- RIGHT BOX: Results -----
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏆 Classification Results")

    if "results" in st.session_state:
        predicted_class, confidence, all_preds, pred_idx, image = st.session_state.results

        st.success(f"**Predicted Car:** {predicted_class}")
        st.metric("Confidence", f"{confidence:.2f}%")
        st.progress(confidence/100)

        # Bar Chart
        st.subheader("📊 Prediction Probabilities")
        fig, ax = plt.subplots(figsize=(8,5))
        bars = ax.bar(CLASS_NAMES, all_preds*100, color='skyblue')
        bars[pred_idx].set_color('red')
        ax.set_ylabel("Confidence (%)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Top 3
        st.subheader("🥇 Top 3 Predictions")
        sorted_idx = np.argsort(all_preds)[::-1]
        for i in range(3):
            idx = sorted_idx[i]
            st.markdown(f"{i+1}. **{CLASS_NAMES[idx]}** — {all_preds[idx]*100:.2f}%")
    else:
        st.info("⚡ Results will appear here after classification.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
    <div class="footer">
        🚀 Built with <b style="color:#FFD700;">Streamlit</b> & <b style="color:#FF4B4B;">TensorFlow</b>  
        by <b style="color:#4CAF50;">Dileep Kumar</b>
    </div>
""", unsafe_allow_html=True)
