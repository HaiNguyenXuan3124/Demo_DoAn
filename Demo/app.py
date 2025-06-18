import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load mô hình và các thành phần liên quan
model = load_model("Saved_model_LSTM500000.h5")
tokenizer = joblib.load("tokenizer_500000.pkl")
label_encoder = joblib.load("label_encoder_500000.pkl")
max_len = joblib.load("max_len_500000.pkl")

# Giao diện người dùng
st.set_page_config(page_title="Dự đoán cảm xúc", layout="centered")
st.title("📢 Phân tích cảm xúc đánh giá khách sạn (LSTM)")

text_input = st.text_area("✍️ Nhập đánh giá (tiếng Anh):", height=150)

if st.button("Dự đoán"):
    if not text_input.strip():
        st.warning("Vui lòng nhập nội dung.")
    else:
        # Xử lý đầu vào
        seq = tokenizer.texts_to_sequences([text_input])
        pad = pad_sequences(seq, maxlen=max_len)

        # Dự đoán
        pred_proba = model.predict(pad)[0]
        pred_class = np.argmax(pred_proba)
        label = label_encoder.inverse_transform([pred_class])[0]

        st.success(f"Kết quả dự đoán: **{label}**")
        st.write(f"🔍 Xác suất: {pred_proba}")
