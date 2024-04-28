from flask import Flask, request, render_template, Response, jsonify, send_from_directory
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from torch.nn import Module, Linear, Conv2d, ReLU, MaxPool2d, ConvTranspose2d, Sigmoid
import torch.nn as nn
import time
from PIL import Image
import matplotlib.pyplot as plt
import warnings

def non_maximum_suppression(image, scores, threshold, bound=10):
    """
    Performs non-maximum suppression on corner detection scores to ensure only the
    strongest corner within a specified bound (default 10x10 pixels) is kept.
    This process reduces redundancy among detected corners.
    
    Parameters:
    - image: The image being processed (not directly used but contextual).
    - scores: A 2D array of Harris corner detection scores.
    - threshold: The minimum score a corner must have to be considered significant.
    - bound: The size of the neighborhood in pixels to search for the strongest corner.
    
    Returns:
    - final_corners: A list of coordinates for the remaining corners after suppression.
    """
    final_corners = []
    for i in range(0, image.shape[0], bound):
        for j in range(0, image.shape[1], bound):
            window = scores[i:i+bound, j:j+bound]
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(window)
            if maxVal > threshold:
                globalMaxLoc = (j + maxLoc[0], i + maxLoc[1])
                final_corners.append(globalMaxLoc)
    return final_corners

def DataVisualize(corners_count, frame_rate):
    """
    Visualizes the trend and variability of corner detections across video frames.
    It filters the data to include counts from every 5th frame, fits a polynomial
    curve to this filtered data to show the trend, and calculates plus visualizes
    the standard deviation as a shaded area around the curve to indicate variability.

    It plots the number of corners and shrimp detections detected in each frame.
    
    Parameters:
    - corners_count: A list containing the number of corners detected in each frame.
    - shrimp_detections: A list containing binary flags for shrimp detections (1 for detected, 0 for not).
    
    The function creates and displays a matplotlib plot illustrating these aspects.
    """
    filtered_corners_count = [corners_count[i] for i in range(len(corners_count)) if i % 5 == 0]
    frame_numbers = [i for i in range(len(corners_count)) if i % 5 == 0]

    degree_of_polynomial = 4
    coefficients = np.polyfit(frame_numbers, filtered_corners_count, degree_of_polynomial)
    polynomial = np.poly1d(coefficients)

    x_curve = np.linspace(min(frame_numbers), max(frame_numbers), 500)
    y_curve = polynomial(x_curve)

    std_dev = np.std([y - polynomial(x) for x, y in zip(frame_numbers, filtered_corners_count)])
    y_upper = y_curve + std_dev
    y_lower = y_curve - std_dev

    plt.figure(figsize=(12, 7))
    plt.scatter(frame_numbers, filtered_corners_count, color='blue', s=10, label='Detected Corners')
    plt.plot(x_curve, y_curve, color='red', linewidth=2, label='Best-fit Curve')
    plt.fill_between(x_curve, y_lower, y_upper, color='red', alpha=0.1, label='Standard Deviation')
    plt.title('Number of Corners Detected in Each Frame (Filtered) with Best-fit Curve', fontsize=16)
    plt.xlabel('Frame Number', fontsize=14)
    plt.ylabel('Corners Detected', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    corners_count = []

    detections, time_seconds = zip(*shrimp_detections) 

    plt.figure(figsize=(12, 6))
    plt.plot(time_seconds, detections, linestyle='-', marker='o', color='deepskyblue', markersize=5)

    for i, detection in enumerate(detections):
        if detection:
            plt.annotate('Detected', (time_seconds[i], detections[i]), textcoords="offset points", xytext=(0,10), ha='center', color='green')

    plt.title('Shrimp Detections Over Video Time', fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Detection (1=Detected, 0=Not Detected)', fontsize=14)
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ['Not Detected', 'Detected'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.legend()
    plt.show()
    return

# Prepare image batch for CNN processing.
def process_image(image):
    image_transformed = transform(image)
    image_batch = image_transformed.unsqueeze(0)
    return image_batch

# Predict mask from image batch using a model.
def predict_mask(model, image_batch):
    model.eval()
    with torch.no_grad():
        prediction = model(image_batch)
        predicted_mask = prediction > 0.5
    return predicted_mask.squeeze().cpu().numpy()

# Class for pond detection and processing.
class Pond():

    def __init__(self, frame):
        self.frame = frame
    
    def extract_pond(self, original_image, predicted_mask):
        predicted_mask_resized = cv2.resize(predicted_mask.astype(np.float32), 
                                            (original_image.shape[1], original_image.shape[0]))
        mask_uint8 = (predicted_mask_resized > 0.5).astype(np.uint8) * 255
        extracted_pond = cv2.bitwise_and(original_image, original_image, mask=mask_uint8)
        return extracted_pond

    # Apply the mask on the pond using a CNN model.
    def mask_pond(self):
        pond_model = CNN()
        pond_model.load_state_dict(torch.load('model_state/segmentation.pth'))
        pil_image = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)).convert('RGB')
        image_batch = process_image(pil_image)
        predicted_mask = predict_mask(pond_model, image_batch)
        self.frame = self.extract_pond(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB), predicted_mask)
        return

    # Fit a circular mask to the largest pond contour. 
    # This will eliminate rough edges that will affect corner detection.
    def fit_circle(self):
        mask = self.frame[:,:,0]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)
        circular_mask = np.zeros_like(mask)
        cv2.circle(circular_mask, center, radius, color=255, thickness=cv2.FILLED)
        self.frame = np.stack((circular_mask,)*3, axis=-1)

    def __repr__(self):
        return "Extracted pond from the video using trained CNN"

# Crop the borders of an image (e.g., 5 pixels away if border_size = 5).
def crop_borders(image, border_size = 5):
    height, width = image.shape[:2]
    cropped_image = image[border_size:height - border_size, border_size:width - border_size]
    return cropped_image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Pond processing module
class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.encoder = nn.Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            ReLU(inplace=True),
            Conv2d(32, 64, kernel_size=4, padding=2),
            Conv2d(64, 128, kernel_size=3, padding=1),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            ConvTranspose2d(128, 1, kernel_size=2, stride=2),
            Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
# Fetch the video file
cap = cv2.VideoCapture('videos/testvideo3.mp4')
frame_rate = cap.get(cv2.CAP_PROP_FPS)
success, frame = cap.read()
corners_count = []

# ResNet18 model
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft.load_state_dict(torch.load('model_state/model_state_dict.pth'))
model_ft.eval()

# Variables
shrimp_detections = []
detection_interval = 2
last_detection_time = 0
start_time = time.time()

# Frame preprocess
pond = Pond(frame)
pond.mask_pond()
pond.fit_circle()

# Real time detection
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = crop_borders(cv2.bitwise_and(frame, pond.frame))

    operatedImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    operatedImage = np.float32(operatedImage)
    dest = cv2.cornerHarris(operatedImage, 3, 5, 0.08)
    dest = cv2.dilate(dest, None)
    threshold = 0.01 * dest.max()
    corners = non_maximum_suppression(frame, dest, threshold)
    for corner in corners:
        cv2.circle(frame, corner, radius=3, color=(0, 0, 255), thickness=-1)

    corners_count.append(len(corners))
    
    cv2.imshow('Frame with Corners', frame)

    current_time = time.time() - start_time
    if current_time - last_detection_time >= detection_interval:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
        transformed_image = transform(pil_image).unsqueeze(0)

        with torch.no_grad():
            outputs = model_ft(transformed_image)
            _, preds = torch.max(outputs, 1)

        # Record detection with current time
        shrimp_detections.append((int(preds[0] == 1), current_time))
        
        last_detection_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
DataVisualize(corners_count, frame_rate=frame_rate)