import cv2 # openCV library for image processing
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import ( 
    # tensorflow and keras has pretrained ML models for image classification
    # mobilenet_v2 is a very lightweight model suitable for mobile and web applications
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

from PIL import Image # Pillow library for image handling

def load_model(): # MobileNetV2 = convolutional NN
    model = MobileNetV2(weights='imagenet') # Load the MobileNetV2 model with pre-trained weights on ImageNet dataset
    return model 

def preprocess_image(image): # Turn image into format suitable for MobileNetV2
    image = np.array(image) # Convert the image to a numpy array
    image = cv2.resize(image, (224, 224)) # Resize the image to 224x224 pixels to fit MobileNetV2 input size
    image = preprocess_input(image) # Preprocess the image for MobileNetV2
    image = np.expand_dims(image, axis=0) # Add a batch dimension. Model expect multiple images => add a dimension at axis 0 to make it look like a batch    
    return image

def classify_image(model, image): # Classify the image using the model
    try:
        processed_image    = preprocess_image(image) # Preprocess the image
        predictions        = model.predict(processed_image) # Predict the class of the image
        decoded_predictions = decode_predictions(predictions, top=3)[0] # Decode the predictions to get human-readable labels. 
                                                                       # top=3 means we want the top 3 predictions i.e. the 3 most likely classes (highest prob.)
                                                                       # [0] to get the first (and only) batch item's predictions
        return decoded_predictions # Return the top 3 predictions       
    except Exception as e:
        st.error(f"Error during classification: {str(e)}")
        return None
    
def main():
    st.set_page_config(page_title="AI-powered Image Classifier by Anh Duc Nguyen", page_icon = "", layout="centered")
    st.title("AI-powered Image Classifier by Anh Duc Nguyen")
    st.write("Upload an image and let the AI classify it for you!")

    @st.cache_resource # Cache the model loading to avoid reloading it on every interaction
    def load_cached_model():
        return load_model()
    
    model = load_cached_model() # Load the model

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"]) # File uploader widget

    if uploaded_file is not None:
        image = st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True) # Display the uploaded image
        btn = st.button("Classify Image")

        if btn:
            with st.spinner("Classifying..."): # Show a spinner while classifying
                image = Image.open(uploaded_file)
                predictions = classify_image(model, image)

                if predictions:
                    st.subheader("Top 3 Predictions:")
                    for _, label, score in predictions: #_ = anonymous variable as the 1st one we don't care, only label and score
                        st.write(f"**{label}**: {score * 100:.2f}%")  # show label in bold and score as percentage  

if __name__ == "__main__":
    main()