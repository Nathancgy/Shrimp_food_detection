# Shrimp Food Detection

## Overview
The *shrimp_food_detection* project aims to optimize feeding practices in shrimp farming by automating the detection and visualization of leftover food in shrimp ponds using advanced computer vision techniques. Additionally, it incorporates a Convolutional Neural Network (CNN) model to detect the presence of shrimp, enabling shrimp farmers to adjust feeding schedules more precisely and monitor shrimp activity effectively. This project presents a novel approach to enhancing sustainability and profitability in shrimp farming operations.

## Features
- **Food Detection**: Utilizes the Harris corner detection algorithm enhanced with non-maximum suppression to identify potential leftover food particles.

<img src = 'https://github.com/Nathancgy/Shrimp_food_detection/blob/main/img/curve.png?raw=true' width = '600'>

- **Shrimp Detection**: Employs a CNN model, specifically the ResNet architecture, to detect the presence of shrimp within the pond environment.

<img src = 'https://github.com/Nathancgy/Shrimp_food_detection/blob/main/img/detect.png?raw=true' width = '600'>

- **Real-time Visualization**: Offers real-time graphs to visualize the quantity of leftover food and shrimp activity, aiding in the optimization of feeding schedules.

<img src = 'https://github.com/Nathancgy/Shrimp_food_detection/blob/main/img/realtime.png?raw=true' width = '600'>

- **Pond Detection**: Employs a 3 layer encoder and a 1 layer decoder CNN model to detect the pond, accurate against different shapes.
  
<img src = 'https://github.com/Nathancgy/Shrimp_food_detection/blob/main/img/pond.png?raw=true' width = '600'>

## Installation

To set up the *shrimp_food_detection* project, follow these steps:

1. Clone the repository:
```
git clone https://github.com/yourrepository/shrimp_food_detection.git
```
2. Navigate to the project directory:
```
cd shrimp_food_detection
```
3. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage
1. To run the application, execute the following command in the terminal:
```
python app.py
```
2. For training the shrimp detection model, run:
```
jupyter notebook train.ipynb
```
3. For pond identification, run:
```
jupyter notebook pond_identify.ipynb
```

## Contributing
Contributions to shrimp_food_detection are welcome!

## License
Distributed under the MIT License. See LICENSE for more information.