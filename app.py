import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load model and class names
model = tf.keras.models.load_model("model.h5")
class_names = open("class_names.txt").read().splitlines()

st.title("🎨 AI Drawing Guessing Game")
st.markdown("Draw something below, and the AI will guess what it is!")

# Drawing canvas
canvas = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Prediction
if canvas.image_data is not None:
    img = Image.fromarray((canvas.image_data[:, :, :3]).astype('uint8'))
    img = ImageOps.grayscale(img).resize((28, 28))
    img = np.array(img).reshape(1, 28, 28, 1) / 255.0

    preds = model.predict(img)[0]
    top_3 = np.argsort(preds)[-3:][::-1]

    st.subheader("🤖 AI's Best Guesses:")
    for i in top_3:
        st.write(f"**{class_names[i].replace('_', ' ').title()}** - {preds[i] * 100:.2f}%")
