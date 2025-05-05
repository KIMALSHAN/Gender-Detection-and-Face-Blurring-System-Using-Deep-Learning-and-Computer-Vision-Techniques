import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load the gender classification model
model_path = "models/gender_model.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found.")
model = load_model(model_path)

# Function to preprocess an image for the gender model
def preprocess_image(image, img_size=(64, 64)):
    image_resized = cv2.resize(image, img_size)
    image_normalized = image_resized / 255.0
    return np.expand_dims(image_normalized, axis=0)

# Function to detect faces and blur specific ones
def detect_and_blur_faces(image_path, model, blur_gender="female"):
    # Load the Haar Cascade for face detection
    haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)

    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from '{image_path}'.")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract face region
        face = image[y:y + h, x:x + w]

        # Preprocess the face for gender prediction
        preprocessed_face = preprocess_image(face)

        # Predict gender
        predictions = model.predict(preprocessed_face)
        predicted_gender = np.argmax(predictions)

        # Map predictions to gender labels
        gender_label = "male" if predicted_gender == 0 else "female"

        # Blur the face if it matches the target gender
        if gender_label == blur_gender:
            face_blurred = cv2.GaussianBlur(face, (99, 99), 30)
            image[y:y + h, x:x + w] = face_blurred

    return image

# Streamlit App
st.title("Gender Detection and Face Blurring System")

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save the uploaded file temporarily
    temp_file_path = "uploaded_image.jpg"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the original image
    st.image(temp_file_path, caption="Uploaded Image", use_column_width=True)

    # Perform face detection and blurring
    try:
        blurred_image = detect_and_blur_faces(temp_file_path, model)

        # Save and display the output
        output_file_path = "blurred_output.jpg"
        cv2.imwrite(output_file_path, blurred_image)
        st.image(output_file_path, caption="Blurred Output Image", use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Clean up the temporary file
    os.remove(temp_file_path)
