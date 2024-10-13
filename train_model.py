import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from keras_facenet import FaceNet
import pickle

print("Starting the script...")

# Initialize the FaceNet embedder
embedder = FaceNet()

print("FaceNet initialized.")

dataset_path = 'C:\\Users\\HP\\Desktop\\Attendance_folder\\Students_Dataset'

print(f"Dataset path set to: {dataset_path}")

# Prepare lists to store features and labels
X = []
y = []

print("Loading data...")

# Iterate through each student's folder and images
for student_name in os.listdir(dataset_path):
    student_folder = os.path.join(dataset_path, student_name)
    for img_name in os.listdir(student_folder):
        img_path = os.path.join(student_folder, img_name)
        
        # Detect and extract the embedding for the face in the image
        img_embedding = embedder.extract(img_path, threshold=0.95)
        
        if len(img_embedding) > 0:  
            X.append(img_embedding[0]['embedding'])  
            y.append(student_name)  

print(f"Data loading completed. Number of samples: {len(X)}")

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

print("Data conversion to numpy arrays completed.")

# Encode the labels (student names)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Labels encoded.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("Data split into training and testing sets.")

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

print("Training the KNN model...")

# Train the KNN model on the training data
knn.fit(X_train, y_train)

print("Model training completed.")

# Evaluate the model on the test data
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {accuracy * 100:.2f}%")

# Save the trained model to a file
with open('knn_model.pkl', 'wb') as model_file:
    pickle.dump(knn, model_file)

print("Model saved to knn_model.pkl.")

# Save the label encoder as well
with open('label_encoder.pkl', 'wb') as label_file:
    pickle.dump(label_encoder, label_file)

print("Label encoder saved to label_encoder.pkl.")

# Plot learning curves to analyze overfitting and underfitting
print("Generating learning curves...")

# Initialize lists to store results
train_sizes = np.arange(10, len(X_train), int(len(X_train) / 10))  # Vary the training sizes
train_accuracies = []
test_accuracies = []

for size in train_sizes:
    # Train with a subset of the training data
    X_subset, _, y_subset, _ = train_test_split(X_train, y_train, train_size=size, random_state=42)
    knn.fit(X_subset, y_subset)
    
    # Calculate accuracies
    train_pred = knn.predict(X_subset)
    test_pred = knn.predict(X_test)
    
    train_accuracy = accuracy_score(y_subset, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(train_sizes, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(train_sizes, test_accuracies, label='Validation Accuracy', marker='o')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curves for KNN')
plt.legend()
plt.grid(True)
plt.show()

print("Script completed.")

