import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Klasifikasi Bangun Datar", page_icon="🔷")

# Load model & labels
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

with open("labels.txt","r") as f:
    labels=[l.strip() for l in f.readlines()]

st.markdown("""
# 🔷 Klasifikasi Bangun Datar
Gunakan kamera untuk mendeteksi jenis bangun datar.
""")

camera_file = st.camera_input("Ambil gambar dengan kamera")

if camera_file:
    img = Image.open(camera_file).convert("RGB")
    st.image(img, caption="Gambar dari kamera", use_column_width=True)

    img = img.resize((224,224))
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr,0)

    pred = model.predict(arr)
    idx = np.argmax(pred)
    conf = float(pred[0][idx])

    st.subheader("🎯 Hasil Prediksi")
    st.success(f"**{labels[idx]}**")
    st.progress(conf)
    st.write(f"Confidence: {conf:.2f}")
