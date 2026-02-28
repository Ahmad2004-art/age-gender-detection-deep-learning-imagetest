🎯 Age & Gender Detection from Images
📌 Overview

This project is a Deep Learning-based computer vision system that predicts age and gender from facial images.

The model is built using a Convolutional Neural Network (CNN) implemented with TensorFlow/Keras and is designed for static image inference. The system performs face detection, image preprocessing, and prediction in a structured pipeline.

The goal of this project is to explore multi-task learning by combining:

Gender classification (Binary Classification)

Age prediction (Regression)

🧠 Problem Statement

Automatic age and gender prediction has applications in:

Smart surveillance systems

Demographic analytics

Retail and marketing insights

Human-computer interaction

This project builds a CNN model capable of learning facial features and predicting both tasks simultaneously.

🏗 Approach

The system follows these steps:

Face detection using OpenCV

Image preprocessing (resize, normalization)

CNN-based feature extraction

Multi-output prediction:

Gender (classification)

Age (regression)

🧬 Model Architecture

Convolutional Neural Network (CNN)

Multiple Conv + Pooling layers

Fully connected layers

Two output heads:

Gender classification (Softmax / Sigmoid)

Age regression (Linear output)

This multi-task setup allows the network to share facial feature representations while optimizing two objectives.

⚙ Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy

Pillow

📊 Features

✔ Gender classification (Male / Female)
✔ Age prediction (Regression output)
✔ Face detection on static images
✔ Image preprocessing pipeline
✔ Deep Learning-based inference

🚀 How to Run

Clone the repository:

git clone https://github.com/yourusername/age-gender-detection.git
cd age-gender-detection

Install dependencies:

pip install -r requirements.txt

Run inference:

python test.py
🔮 Future Improvements

Improve dataset diversity for better generalization

Use Transfer Learning (ResNet / MobileNet)

Add real-time webcam support

Deploy model as REST API
