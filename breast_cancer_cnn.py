import zipfile
import os

zip_path = "images.zip"
extract_path = "/content/images"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Files extracted to:", extract_path)

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

labels_df = pd.read_excel("labels.xlsx")
labels_df["label"] = labels_df["label"].map({"PB": 0, "PM": 1})

image_dir = "/content/images/images.zip"

X= []
y=[]
image_size = (224, 224)

for index, row in labels_df.iterrows():
    image_path = os.path.join(image_dir, row["Image"])

    if not os.path.exists(image_path):
        print(f"Missing file: {image_path}")
        continue

    img = cv2.imread(image_path)

    if img is None:
        print(f"Corrupt or unsupported format: {image_path}")
        continue

    img = cv2.resize(img, image_size)
    img = img / 255.0

    X.append(img)
    y.append(row["label"])


X = np.array(X)
y = np.array(y)
y = to_categorical(y, num_classes=2)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=16)

model.save("breast_cancer_classifier.h5")

val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc:.2f}")

import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model("breast_cancer_classifier.h5")

# Define image size (same as during training)
image_size = (224, 224)

# Define class labels
class_labels = {0: "PB", 1: "PM"}  # Modify as per your dataset labels

def predict_image(image_path):
    # Load image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Preprocess image
    img = cv2.resize(img, image_size)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get class with highest probability
    confidence = np.max(predictions) * 100  # Convert to percentage

    # Print result
    print(f"Prediction: {class_labels[predicted_class]} ({confidence:.2f}% confidence)")

# Example usage: Provide the image path correctly
predict_image("/content/IIR0001.jpg")
