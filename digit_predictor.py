from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
import os
import cv2
from PIL import ImageGrab, Image
import numpy as np

# Load the trained model
model = load_model('mnist.h5')

def preprocessing_image(img_path='test.jpg'):
    """function to preprocess the image"""
    image = cv2.imread(img_path)
    
    # Debug info
    print(f"Loaded image shape: {image.shape if image is not None else 'None'}")
    
    if image is None:
        print("Error: Could not load image")
        return np.zeros((28, 28))
    
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    preprocessed_digit = np.zeros((28, 28))
    
    if contours:
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            
            # Only process reasonably sized contours
            if w > 5 and h > 5:
                cv2.rectangle(image, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)
                digit = thresh[y:y+h, x:x+w]
                
                # Calculate aspect ratio and resize appropriately
                if w > h:
                    new_w = 20
                    new_h = int(20 * h / w)
                else:
                    new_h = 20
                    new_w = int(20 * w / h)
                
                resized_digit = cv2.resize(digit, (new_w, new_h))
                
                # Center the digit in a 28x28 image
                pad_x = (28 - new_w) // 2
                pad_y = (28 - new_h) // 2
                
                preprocessed_digit = np.zeros((28, 28))
                preprocessed_digit[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_digit
                break  # Process only the first valid contour
    
    # Debug info
    print(f"Preprocessed image range: {np.min(preprocessed_digit)} to {np.max(preprocessed_digit)}")
    
    return preprocessed_digit

def predict_digit(img):
    """function to predict the digit"""
    img.save('test.jpg')
    preprocessed_image = preprocessing_image()
    
    # Debug info
    print(f"Final preprocessed image shape: {preprocessed_image.shape}")
    print(f"Unique values: {np.unique(preprocessed_image)}")
    
    # Normalize like training data
    img_normalized = preprocessed_image.reshape(1, 28, 28, 1).astype('float32') / 255.0
    
    # Predict
    result = model.predict(img_normalized, verbose=0)[0]
    predicted_digit = np.argmax(result)
    confidence = np.max(result)
    
    print(f"Raw predictions: {result}")
    print(f"Predicted: {predicted_digit}, Confidence: {confidence:.4f}")
    
    os.remove('test.jpg')
    return predicted_digit, confidence

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        self.title("Digit Recognizer")
        
        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Draw a digit", font=("Helvetica", 24))
        self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting) 
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)
        
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, padx=10)
        self.label.grid(row=0, column=1, pady=2, padx=10)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=10)
        self.button_clear.grid(row=1, column=0, pady=2, padx=10)
        
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        
    def clear_all(self):
        self.canvas.delete("all")
        self.label.configure(text="Draw a digit")
        
    def reset(self, event):
        self.x, self.y = None, None
        
    def classify_handwriting(self):
        # Get canvas coordinates
        x = self.winfo_rootx() + self.canvas.winfo_x()
        y = self.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        
        # Capture the canvas area
        im = ImageGrab.grab((x, y, x1, y1))
        
        # Predict
        digit, acc = predict_digit(im)
        self.label.configure(text=f"{digit}, {int(acc*100)}%")
        
    def draw_lines(self, event):
        if self.x and self.y:
            self.canvas.create_line(self.x, self.y, event.x, event.y, width=8, fill='black', capstyle=ROUND, smooth=TRUE)
        self.x = event.x
        self.y = event.y

# Create and run the application
if __name__ == "__main__":
    app = App()
    app.mainloop()