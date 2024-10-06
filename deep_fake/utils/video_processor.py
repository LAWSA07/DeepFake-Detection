import cv2
import numpy as np

def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every frame
        frame = cv2.resize(frame, (224, 224))  # Resize to (224, 224) for the model input
        frame = frame / 255.0  # Normalize the pixel values to the range [0, 1]
        frames.append(frame)

        frame_count += 1
        if frame_count >= 16:  # Process 16 frames at a time
            break

    cap.release()

    if not frames:
        raise ValueError("No frames could be extracted from the video.")

    frames = np.array(frames)

    # Ensure we have exactly 16 frames
    if frames.shape[0] < 16:
        # If we have fewer than 16 frames, repeat the last frame
        frames = np.pad(frames, ((0, 16 - frames.shape[0]), (0, 0), (0, 0), (0, 0)), mode='edge')
    elif frames.shape[0] > 16:
        # If we have more than 16 frames, take the first 16
        frames = frames[:16]

    # Reshape the input to match the model's expected input
    frames = frames.reshape(1, 16, 224, 224, 3)

    # Run prediction using the model
    predictions = model.predict(frames)

    # The model output is (None, 2), so we need to interpret it
    is_deepfake = np.argmax(predictions[0]) == 1
    confidence = predictions[0][1] if is_deepfake else predictions[0][0]

    return is_deepfake, float(confidence)