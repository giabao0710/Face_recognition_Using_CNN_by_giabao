import cv2
import os
import numpy as np
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras._tf_keras.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Directory where images are stored
data_dir = 'images'

# Initialize face detector
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize lists to hold image data and labels
faces_data = []
labels = []

# Loop through each person's folder
for person_name in os.listdir(data_dir):
    person_path = os.path.join(data_dir, person_name)
    if os.path.isdir(person_path):
        for image_name in os.listdir(person_path):
            img_path = os.path.join(person_path, image_name)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (224, 224))  # Resize to 224x224 for input to the model
                faces_data.append(face)
                labels.append(person_name)

# Convert faces_data to numpy array and normalize
faces_data = np.array(faces_data)
faces_data = faces_data / 255.0  # Normalize pixel values to range [0, 1]
faces_data = faces_data.reshape(-1, 224, 224, 1)  # Reshape to match model input

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(faces_data, labels, test_size=0.2, random_state=42)

# Create a CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
#model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(np.unique(labels)), activation='softmax'))  # Output layer with number of classes

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save('keras_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")