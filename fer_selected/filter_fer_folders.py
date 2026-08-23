import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Set the directories for your dataset
train_dir = "fer_selected/train"
test_dir = "fer_selected/test"

img_size = (64, 64)

# Initialize variables to hold the features and labels
features = []
labels = []

# Function to load images and extract features
def load_images_from_directory(directory):
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                # Load the image
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, img_size)
                img = img.astype('float32') / 255.0  # Normalize
                features.append(img.flatten())  # Flatten to 1D vector
                labels.append(label)

# Load train and test images
load_images_from_directory(train_dir)
load_images_from_directory(test_dir)

# Convert the labels to numerical values using LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(labels)

# Convert features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the model (using RandomForestClassifier as an example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "emotion_model.pkl")
print("✅ Model trained and saved as emotion_model.pkl")
