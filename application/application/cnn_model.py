import torch
import torch.nn as nn
import cv2
import numpy as np

#  Define the same model architecture
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.model(x)

#  Prediction function
def predict_image(image_path, model_path=r"C:\Users\DHRUV\Downloads\Disease detection using ML\cnn\cnn_disease_model.pth"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = MyCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or unreadable: {image_path}")

    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).item()

    return prediction  # 0 = healthy, 1 = diseased

#  Example usage
if __name__ == "__main__":
    test_image = "sample DF.JPG"
    result = predict_image(test_image)
    print(result)
    print("Prediction:", result, "(Diseased)" if result == 1 else "(Healthy)")