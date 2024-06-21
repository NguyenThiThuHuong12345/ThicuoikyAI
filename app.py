import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt

# Load the model
model = load_model('model_cnn.h5')
output_class = ["battery", "glass", "metal", "organic", "paper", "plastic"]

# Function to preprocess the input image
def preprocessing_input(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)  # VGG16 preprocess_input
    return img

def plot_images(original, preprocessed):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(original)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Remove the batch dimension for display
    preprocessed = np.squeeze(preprocessed, axis=0)

    axs[1].imshow(preprocessed.astype('uint8'))
    axs[1].set_title('Preprocessed Image')
    axs[1].axis('off')

    st.pyplot(fig)

def predict_user(img_path):
    img = preprocessing_input(img_path)
    plot_images(Image.open(img_path), img)
    predicted_array = model.predict(img)
    predicted_value = output_class[np.argmax(predicted_array)]
    predicted_accuracy = round(np.max(predicted_array) * 100, 2)
    return predicted_value, predicted_accuracy

# Streamlit app
def main():
    st.title("Waste Material Classification")

    uploaded_file = st.file_uploader("Upload an image of waste material", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        # Save the uploaded file to disk
        img_path = f"temp_{uploaded_file.name}"
        img.save(img_path)

        # Make prediction
        predicted_value, predicted_accuracy = predict_user(img_path)
        
        st.write(f"Your waste material is **{predicted_value}** with **{predicted_accuracy}%** accuracy.")

if __name__ == "__main__":
    main()
