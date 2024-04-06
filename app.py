import cv2
import numpy as np
import torch
from torchvision import models
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import time
from torchvision import transforms

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

# Fetch the video file
cap = cv2.VideoCapture('videos/testvideo3.mp4')
frame_rate = cap.get(cv2.CAP_PROP_FPS)
print(frame_rate)
corners_count = []

# ResNet18 model
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

# Load the saved model state dictionary
model_ft.load_state_dict(torch.load('model_state/model_state_dict.pth'))
model_ft.eval()

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Variables for tracking shrimp detections
shrimp_detections = []
detection_interval = 2
last_detection_time = 0

start_time = time.time()

# Real time detection
while True:
    success, frame = cap.read()
    if not success:
        break
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
        transformed_image = data_transforms(pil_image).unsqueeze(0)

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