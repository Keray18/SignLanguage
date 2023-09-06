from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import time
import numpy as np
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__)

# Define the directory where uploaded files will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the 'uploads' directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

cap = None
detector = None
imgSize = 300
counter = 0
current_image_path = None

# Initialize the camera and hand detector
def initialize_camera_and_detector():
    global cap, detector
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)

# Capture a frame from the camera and process it
def capture_and_process_frame():
    global cap, detector, counter, current_image_path
    succ, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white canvas
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Calculate aspect ratio
        aspectRatio = h / w

        # Calculate target dimensions for resizing
        if aspectRatio > 1:
            target_height = imgSize
            target_width = int(target_height / aspectRatio)
        else:
            target_width = imgSize
            target_height = int(target_width * aspectRatio)

        # Resize the cropped hand region to the target dimensions
        imgResize = cv2.resize(img[y - 20:y + h + 20, x - 20:x + w + 20], (target_width, target_height))

        # Calculate the position to paste the resized image on the white canvas
        x_offset = (imgSize - target_width) // 2
        y_offset = (imgSize - target_height) // 2

        # Paste the resized image on the white canvas
        imgWhite[y_offset:y_offset + target_height, x_offset:x_offset + target_width] = imgResize

        cv2.imshow("ImageCrop", img[y - 20:y + h + 20, x - 20:x + w + 20])
        cv2.imshow("ImageWhite", imgWhite)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['GET', 'POST'])
def start_camera():
    if request.method == 'POST':
        initialize_camera_and_detector()
        return redirect(url_for('capture_image'))
    return render_template('start_camera.html')

@app.route('/capture_image', methods=['GET', 'POST'])
def capture_image():
    if request.method == 'POST':
        global current_image_path
        current_image_path = capture_and_save_image()
        return redirect(url_for('show_prediction'))
    return render_template('capture_image.html')

@app.route('/show_prediction')
def show_prediction():
    if current_image_path:
        # Implement prediction logic here using the saved image (current_image_path)
        # Replace the following placeholder code
        prediction = "Placeholder Prediction"
        return render_template('show_prediction.html', prediction=prediction)
    return redirect(url_for('capture_image'))

if __name__ == '__main__':
    app.run(debug=True)
