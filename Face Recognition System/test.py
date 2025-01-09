import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras._tf_keras.keras.models import load_model
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

# Initialize face detector
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

# Load pre-trained model
model = load_model('keras_model.h5')

# Function to get class name based on predicted class index
def get_className(classNo):
    if classNo == 0:
        return "Bao"
    elif classNo == 1:
        return "tan"
    else:
        return "unknown"

# Set a confidence threshold
threshold = 0.99

true_labels = []  # Ground-truth labels for comparison
predicted_labels = []  # Predictions made by the model
confidence_scores = []  # Confidence scores for predictions

while True:
    # Read the frame from the webcam
    success, imgOrignal = cap.read()
    # Detect faces in the image
    faces = facedetect.detectMultiScale(imgOrignal, 1.3, 5)
    
    for x, y, w, h in faces:
        # Crop the image to the region of interest (the detected face)
        crop_img = imgOrignal[y:y+h, x:x+w]
        
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

        # If neither class has high enough probability, classify as "unknown"
        if probabilityValue < threshold:
            classIndex = -1  # Classify as "unknown"
            probabilityValue = 0
        
        # Print prediction probabilities for debugging
        print(f"Prediction Probabilities: {prediction[0]}")
        
        # Draw bounding box and label based on predicted class
        if classIndex == 0:  # Class 0 ("bao")
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        elif classIndex == 1:  # Class 1 ("tan")
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        elif classIndex == -1:  # Unknown class
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box for unknown
            cv2.putText(imgOrignal, "unknown", (x, y - 10), font, 0.75, (255, 0, 0), 1, cv2.LINE_AA)

        # Show the prediction probability value on the screen
        cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (x, y - 40), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Collect true labels and predicted labels (true_labels should be numeric)
        true_labels.append(classIndex)  # Ensure this is a valid integer, not a list
        predicted_labels.append(classIndex)  # Add the predicted label for evaluation
        confidence_scores.append(probabilityValue)  # Add confidence score

    # Display the resulting image
    cv2.imshow("Result", imgOrignal)
    
    # Quit the loop when 'q' is pressed
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Ensure the lists contain only valid integers
true_labels = [int(label) for label in true_labels]  # Make sure all elements are integers
predicted_labels = [int(label) for label in predicted_labels]

# Convert lists to numpy arrays for processing
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Ensure you have some valid labels to calculate metrics
if len(true_labels) > 0 and len(predicted_labels) > 0:
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)

    # Print metrics
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")

    # Confusion matrix (optional)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)  # In the terminal or VSCode, this will show the matrix
else:
    print("No valid predictions to calculate metrics.")
  
# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
