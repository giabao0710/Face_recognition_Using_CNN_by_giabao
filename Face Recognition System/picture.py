import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras._tf_keras.keras.models import load_model
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Initialize face detector
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load pre-trained model
model = load_model('keras_model.h5')

# Function to get class name based on predicted class index
def get_className(classNo):
    if classNo == 0:
        return "Bao"
    elif classNo == 1:
        return "Tan"
    else:
        return "Unknown"

# Set a confidence threshold
threshold = 0.99

# Create file dialog for selecting input image file
Tk().withdraw()  # Hide the root tkinter window
image_path = askopenfilename(title="Select Input Image File", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
if not image_path:
    print("No input image selected. Exiting.")
    exit()

# Read the input image
img_original = cv2.imread(image_path)
if img_original is None:
    print("Error: Cannot read the image file.")
    exit()

# Detect faces in the image
faces = facedetect.detectMultiScale(img_original, 1.3, 5)

font = cv2.FONT_HERSHEY_COMPLEX

for x, y, w, h in faces:
    # Crop the image to the region of interest (the detected face)
    crop_img = img_original[y:y+h, x:x+w]

    # Resize the cropped image to 224x224 to match the model input size
    img = cv2.resize(crop_img, (224, 224))

    # Convert the image to grayscale (assuming the model was trained on grayscale images)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert the image to float32 and normalize it
    img = img.astype(np.float32)
    img = img / 255.0

    # Reshape the image to match the input shape of the model (1, 224, 224, 1)
    img = img.reshape(1, 224, 224, 1)

    # Get model predictions
    prediction = model.predict(img)

    # For multi-class models, we use argmax to get the predicted class
    classIndex = np.argmax(prediction, axis=-1)
    probabilityValue = np.amax(prediction)

    # If neither class has high enough probability, classify as "Unknown"
    if probabilityValue < threshold:
        classIndex = -1  # Classify as "Unknown"
        probabilityValue = 0

    # Draw bounding box and label based on predicted class
    if classIndex == 0:  # Class 0 ("Bao")
        cv2.rectangle(img_original, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_original, get_className(classIndex), (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    elif classIndex == 1:  # Class 1 ("Tan")
        cv2.rectangle(img_original, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_original, get_className(classIndex), (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    elif classIndex == -1:  # Unknown class
        cv2.rectangle(img_original, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box for unknown
        cv2.putText(img_original, "Unknown", (x, y - 10), font, 0.75, (255, 0, 0), 1, cv2.LINE_AA)

    # Show the prediction probability value on the screen
    cv2.putText(img_original, str(round(probabilityValue * 100, 2)) + "%", (x, y - 40), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

# Display the processed image
cv2.imshow('Output Image', img_original)
cv2.waitKey(0)
cv2.destroyAllWindows()
