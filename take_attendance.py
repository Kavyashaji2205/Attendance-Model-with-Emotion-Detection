import numpy as np
import cv2
from keras.models import load_model
from keras_facenet import FaceNet
from datetime import datetime
from collections import defaultdict
import pickle
import os

# Load the trained KNN model and Label Encoder
with open('C:\\Users\\HP\\Desktop\\Attendance_folder\\models\\knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)
with open('C:\\Users\\HP\\Desktop\\Attendance_folder\\models\\label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the emotion detection model
emotion_model = load_model('C:\\Users\\HP\\Desktop\\Attendance_folder\\models\\attendance_emotion_detection_model.h5')

# Print the model summary to check the input shape
emotion_model.summary()

# Define the emotion labels 
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize FaceNet
embedder = FaceNet()

# Detect the face using OpenCV's Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the allowed attendance time range
start_time = datetime.strptime("09:30", "%H:%M").time()
end_time = datetime.strptime("10:00", "%H:%M").time()

# Dictionary to track attendance
attendance_dict = {}

# Dictionary to track consecutive detections
consecutive_detections = defaultdict(int)

# Threshold for consecutive frames
CONSECUTIVE_THRESHOLD = 3

# Define the path for the attendance file
attendance_file = 'attendance.csv'

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)  

while True:
    current_time = datetime.now().time()

    # Check if the current time is within the allowed attendance time
    if start_time <= current_time <= end_time:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]

                # For Face Recognition
                face_resized = cv2.resize(face, (160, 160))  
                face_resized = np.expand_dims(face_resized, axis=0)
                face_embedding = embedder.embeddings(face_resized)

                # Predict the student's identity using the KNN model
                prediction = knn_model.predict(face_embedding)
                predicted_student = label_encoder.inverse_transform(prediction)[0]

                # Increment consecutive detection count
                consecutive_detections[predicted_student] += 1

                # Only mark attendance if the student has been detected in multiple consecutive frames
                if consecutive_detections[predicted_student] >= CONSECUTIVE_THRESHOLD:
                    if predicted_student not in attendance_dict:
                        # For Emotion Detection
                        face_resized_emotion = cv2.resize(face, (28, 28))  
                        face_resized_emotion = cv2.cvtColor(face_resized_emotion, cv2.COLOR_BGR2GRAY) 
                        face_resized_emotion = np.expand_dims(face_resized_emotion, axis=-1)  
                        face_resized_emotion = np.expand_dims(face_resized_emotion, axis=0)  
                        face_resized_emotion = face_resized_emotion / 255.0  

                        # Flatten the image to match the input shape (784,)
                        face_resized_emotion = np.reshape(face_resized_emotion, (1, 784))  
                        # Predict emotion
                        try:
                            emotion_prediction = emotion_model.predict(face_resized_emotion)
                            emotion_index = np.argmax(emotion_prediction)

                            # Ensure the index is within the range of emotion_labels
                            if 0 <= emotion_index < len(emotion_labels):
                                emotion_label = emotion_labels[emotion_index]
                            else:
                                emotion_label = "Unknown"
                        except Exception as e:
                            print(f"Error during emotion prediction: {e}")
                            emotion_label = "Error"

                        # Mark attendance and save to file
                        with open(attendance_file, 'a') as f:
                            f.write(f"{predicted_student}, Present, {datetime.now()}, {emotion_label}\n")

                        # Mark as present in memory
                        attendance_dict[predicted_student] = True

                        # Print the result
                        print(f"Predicted Student: {predicted_student} - Marked as Present with Emotion: {emotion_label}")

                        # Reset consecutive detection count
                        consecutive_detections[predicted_student] = 0

        else:
            print("No face detected in the frame.")

        # Show the video frame with the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Attendance and Emotion Detection System - Press "q" to quit', frame)
    else:
        print(f"Current time {current_time} is outside the allowed attendance time.")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows() 
