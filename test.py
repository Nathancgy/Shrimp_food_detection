import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, padding=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_transformed = transform(image)
    image_batch = image_transformed.unsqueeze(0)
    
    return image_batch

def predict_mask(model, image_batch):
    model.eval()
    with torch.no_grad():
        prediction = model(image_batch)
        predicted_mask = prediction > 0.5
    return predicted_mask.squeeze().cpu().numpy()

def highlight_circle(original_image, predicted_mask):
    predicted_mask_resized = cv2.resize(
        predicted_mask.astype(np.float32), 
        (original_image.shape[1], original_image.shape[0])
    )
    _, mask_uint8 = cv2.threshold(predicted_mask_resized, 0.5, 255, cv2.THRESH_BINARY)
    mask_uint8 = mask_uint8.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    highlighted_image = original_image.copy()
    cv2.drawContours(highlighted_image, contours, -1, (0, 255, 0), 3)
    return highlighted_image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_path = 'imageset/pond_before/test4.png'

model = CNN() 
model.load_state_dict(torch.load('segmentation_model.pth'))

image_batch = process_image(image_path)
predicted_mask = predict_mask(model, image_batch)

original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

highlighted_image = highlight_circle(original_image, predicted_mask)

plt.imshow(highlighted_image)
plt.axis('off')
plt.show()
