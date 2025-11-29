# app.py
# streamlit UI
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from core import load_model, image_to_text, line_segmentation

st.title(":rainbow[texty]: convert your handwritten notes into text!")

st.markdown('Choose a file to upload:')
uploaded_img = st.file_uploader("Choose a file to upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

st.markdown(":orange[*Note: at this time, texty only accepts .jpg, .jpeg and .png images, "
            "and identify notes that are written relatively clearly and well-lit.]")

# loads model
processor, model = load_model()

if uploaded_img is not None:
    image = Image.open(uploaded_img) # convert to PIL
    image = np.array(image.convert('RGB')) # to RGB -> np array
    line_images = line_segmentation(image)

    display_texts = []

    for i in range(len(line_images)):
        # convert bgr to rgb
        line_rgb = cv2.cvtColor(line_images[i], cv2.COLOR_BGR2RGB)
        # convert rgb to pil for the next function
        line_pil = Image.fromarray(line_rgb)
        
        generated_text = image_to_text(line_pil, processor, model)

        display_texts.append(generated_text)
    
    with st.container(border=True):
        col1, col2 = st.columns([0.5, 0.5])

        with col1:
            st.markdown('Uploaded image:')
            st.image(image, width=300)  
        with col2:
            full_text = "\n".join(display_texts)
            st.markdown('Text extracted:')
            st.text_area("Text extracted", value=full_text, label_visibility="collapsed")

