import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector

# Define the size for the square canvas and other constants
imgSize = 300  # Size of the square canvas to which the hand image will be resized
offset = 20    # Offset used for cropping around the detected hand
folder = "Data/A"  # Directory where the images will be saved
counter = 0    # Counter for the number of images saved

# Initialize the hand detector with a maximum of one hand
detector = HandDetector(maxHands=1)

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

def resize_to_square(img, target_size):
    # Calculate new dimensions, keeping aspect ratio
    height, width, _ = img.shape
    scale = target_size / max(height, width)
    new_width, new_height = int(width * scale), int(height * scale)
    resized_img = cv2.resize(img, (new_width, new_height))

    # Calculate padding to make the image square
    delta_w = target_size - new_width
    delta_h = target_size - new_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # Add padding to the resized image and return
    return cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

# Main loop for hand detection and image processing
while True:
    success, img = cap.read()
    if success:
        # Detect hands in the current frame
        hands, img = detector.findHands(img)
        if hands:
            # If a hand is detected, get its bounding box
            hand = hands[0]
            x, y, w, h = hand['bbox']
            # Crop the image around the hand
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            # Resize the cropped hand image to a square shape
            imgWhite = resize_to_square(imgCrop, imgSize)
            # Display the cropped and resized images
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        # Display the original image
        cv2.imshow("Image", img)
        # Check if the 's' key is pressed to save the image
        key = cv2.waitKey(1)
        if key == ord("s"):
            counter += 1
            # Save the resized image with a timestamp
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            # Print the count of saved images
            print(counter)
