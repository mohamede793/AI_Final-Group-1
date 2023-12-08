import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize video capture from webcam, hand detector for one hand, and classifier using a pre-trained model.
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Define constants for image processing
offset = 20  # Offset used in cropping the image around the detected hand
imgSize = 300  # Size of the square canvas for the processed image
labels = ["Hello", "I", "A", "M", "O", "B", "C"]  # Classification labels

def prepare_image_for_classification(img, bbox, img_size, offset):
    # Extract bounding box coordinates
    x, y, w, h = bbox
    # Crop the image around the hand
    img_crop = img[y - offset:y + h + offset, x - offset:x + w + offset]
    # Calculate the aspect ratio of the hand
    aspect_ratio = h / w

    # Resize and pad the image to make it square while maintaining the aspect ratio
    if aspect_ratio > 1:
        # For tall images
        w_cal = math.ceil((img_size / h) * w)
        img_resize = cv2.resize(img_crop, (w_cal, img_size))
        w_gap = math.ceil((img_size - w_cal) / 2)
        img_white = np.ones((img_size, img_size, 3), np.uint8) * 255
        img_white[:, w_gap:w_cal + w_gap] = img_resize
    else:
        # For wide images
        h_cal = math.ceil((img_size / w) * h)
        img_resize = cv2.resize(img_crop, (img_size, h_cal))
        h_gap = math.ceil((img_size - h_cal) / 2)
        img_white = np.ones((img_size, img_size, 3), np.uint8) * 255
        img_white[h_gap:h_cal + h_gap, :] = img_resize

    return img_white

while True:
    # Capture each frame from the webcam
    success, img = cap.read()
    if success:
        img_output = img.copy()
        # Detect hands in the captured image
        hands, _ = detector.findHands(img)

        if hands:
            # Process the first detected hand
            hand = hands[0]
            img_for_classification = prepare_image_for_classification(img, hand['bbox'], imgSize, offset)
            # Classify the processed image
            prediction, index = classifier.getPrediction(img_for_classification, draw=False)

            # Display the classification label on the image
            x, y, w, h = hand['bbox']
            # Draw a rectangle and label around the detected hand
            cv2.rectangle(img_output, (x - offset, y - offset - 50), (x - offset + 180, y - offset), (255, 0, 255), cv2.FILLED)
            cv2.putText(img_output, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(img_output, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        # Show the image with the classification result in a window
        cv2.imshow("Image", img_output)
        # Wait for a key press for 1 millisecond
        cv2.waitKey(1)
