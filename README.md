Handwritten Digit Recognition System
A complete deep learning solution that accurately identifies handwritten digits (0-9) using a Convolutional Neural Network. Features a user-friendly drawing interface for real-time predictions.

Features
High accuracy digit recognition (>98%)
Interactive drawing GUI
Real-time prediction with confidence scores
Clean, efficient preprocessing pipeline
Optimized CNN architecture with dropout

Technical Stack
Python
TensorFlow & Keras (Model)
OpenCV (Image Processing)
Tkinter (GUI)
NumPy

Project Structure
train_digit_recognizer.py - Model training script
digit_predictor.py - GUI application
mnist.h5 - Trained model file
requirements.txt - Dependencies

Installation
Clone the repository
Install dependencies: pip install -r requirements.txt
Run training: python train_digit_recognizer.py
Launch GUI: python digit_predictor.py

Usage
Draw a digit (0-9) on the canvas
Click 'Recognise' for prediction
View results with confidence percentage
Use 'Clear' to start over

Model Architecture
Input: 28x28 grayscale images
convolutional Layers with ReLU activation
MaxPooling layers
Fully connected layers with dropout
Output: 10-unit softmax layer

Performance
Training Accuracy: >99%
Validation Accuracy: >98%
Optimizer: Adam (learning rate: 0.001)
Loss Function: Categorical Crossentropy
