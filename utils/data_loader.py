import os
import numpy as np
import cv2

def load_dataset(directory, label, frame_skip=10):
    data = []
    labels = []
    for filename in os.listdir(directory):
        video_path = os.path.join(directory, filename)
        if os.path.isfile(video_path):
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_skip == 0:
                    frame = cv2.resize(frame, (128, 128))  # Resize to input size
                    data.append(frame)
                    labels.append(label)
                frame_count += 1
            cap.release()
    return np.array(data), np.array(labels)
