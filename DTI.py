import os
import cv2
import numpy as np

# Define paths to the directories containing healthy and diseased crop images
healthy_dir = r"C:\Users\DHRUV\Downloads\Telegram Desktop\split_data\diseases free"
diseased_dir = r"C:\Users\DHRUV\Downloads\Telegram Desktop\split_data\diseases present"

# Parameters
image_size = (128, 128)  # Resize images to 128x128 pixels

# Initialize empty lists to store images and labels
images = []
labels = []

# Load healthy crop images and assign label 0
for filename in os.listdir(healthy_dir):
    img_path = os.path.join(healthy_dir, filename)
    img = cv2.imread(img_path)
    
    if img is not None:
        img = cv2.resize(img, image_size)  # Resize image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = img.flatten() / 255.0  # Normalize pixel values to the range [0, 1]
        
        images.append(img)  # Add image to the list
        labels.append(0)  # Label 0 for healthy crops

# Load diseased crop images and assign label 1
for filename in os.listdir(diseased_dir):
    img_path = os.path.join(diseased_dir, filename)
    img = cv2.imread(img_path)
    
    if img is not None:
        img = cv2.resize(img, image_size)  # Resize image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = img.flatten() / 255.0  # Normalize pixel values to the range [0, 1]
        
        images.append(img)  # Add image to the list
        labels.append(1)  # Label 1 for diseased crops

# Convert images and labels lists to numpy arrays for easier processing
X = np.array(images)
y = np.array(labels)

print(f"Loaded {len(X)} images.")
print("programe is done")