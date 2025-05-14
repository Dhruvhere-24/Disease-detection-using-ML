import pickle as pk
import numpy as np
import cv2
# import cnn_model as cnmod

# Load the saved model
with open(r"C:\Users\DHRUV\Downloads\Disease detection using ML\svm\svm_model.pkl", "rb") as file:
    loaded_model = pk.load(file)

def preprocess_image(image_path):
    image = cv2.imread(image_path)  # Read image
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    image = cv2.resize(image, (224, 224))  # Resize to match training size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = image / 255.0  # Normalize (0-1 scale)
    image = image.flatten()  # Flatten image (for SVM input)
    
    return np.array(image).reshape(1, -1)


def use_svm_model(image):
    # Preprocess the image
    image = preprocess_image(image)
    prediction = loaded_model.predict(image)
    return prediction[0]