import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the MobileNetV2 model
@st.cache_resource
def load_model():
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    return model

model = load_model()

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

def decode_predictions(preds):
    return tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]

st.title("Image Identifier App")

uploaded_file = st.file_uploader("Choose a file", type=None)

if uploaded_file is not None:
    file_type = uploaded_file.type
    file_name = uploaded_file.name
    
    st.write(f"**Filename:** {file_name}")
    
    # Check if the file is a JPG image
    if file_type == "image/jpeg" or file_name.lower().endswith(('.jpg', '.jpeg')):
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")
        
        try:
            image = Image.open(uploaded_file)
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
            decoded_preds = decode_predictions(predictions)
            
            st.write("### Predictions:")
            for i, (imagenet_id, label, score) in enumerate(decoded_preds):
                st.write(f"{i+1}. **{label}**: {score:.2f}")
        except Exception as e:
            st.error(f"Error processing image: {e}")
            
    else:
        st.write(f"**File Type:** {file_type if file_type else 'Unknown'}")
        st.info("This is not identified as a JPG image, so no classification was performed.")
