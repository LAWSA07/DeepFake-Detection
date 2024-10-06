from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import uuid
import datetime
import logging
from logging.handlers import RotatingFileHandler
from pymongo import MongoClient
import tensorflow as tf
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from collections import Counter

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configuration
class Config:
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb+srv://aswalh0707:jjKko2LYesBDx8Cb@cluster0.f04xs.mongodb.net/deepfake-detection?retryWrites=true&w=majority&appName=Cluster0')
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app.config.from_object(Config)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Setup logging
if not app.debug:
    file_handler = RotatingFileHandler('app.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Deepfake detection startup')

# MongoDB setup
try:
    client = MongoClient(app.config['MONGO_URI'])
    db = client['deepfake-detection']
    results_collection = db['detection_results']
    client.server_info()  # will throw an exception if connection fails
    app.logger.info('MongoDB connected successfully')
except Exception as e:
    app.logger.error(f'MongoDB connection failed: {str(e)}')
    raise

# Load the pre-trained deepfake detection model
try:
    model_path = r'C:\Users\aswal\deep_fake\DeepFake_Model_3.h5'
    model = load_model(model_path)
    app.logger.info('Model loaded successfully')
except OSError as e:
    app.logger.error(f"Error loading the model: {str(e)}")
    raise

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Preprocess each frame to fit the model input
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # Resize frame to the input size expected by the model
    frame_normalized = frame_resized / 255.0       # Normalize pixel values
    frame_reshaped = np.expand_dims(frame_normalized, axis=0)  # Reshape to fit model input shape
    return frame_reshaped

# Predict if a frame is real or fake
def predict_frame(frame):
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)
    app.logger.info(f"Frame prediction: {prediction}")
    return 'real' if prediction[0][0] > 0.5 else 'fake'

# Process video to classify frames
def classify_video(video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    results = []

    while cap.isOpened():
        ret, frame = cap.read()  # Read one frame
        if not ret:
            break

        frame_count += 1
        # Only process every 'frame_skip' frame
        if frame_count % frame_skip == 0:
            result = predict_frame(frame)
            results.append(result)

    cap.release()

    # Count the majority result ('real' or 'fake')
    result_count = Counter(results)
    video_result = 'fake' if result_count['fake'] > result_count['real'] else 'real'
    app.logger.info(f"Fake frames: {result_count['fake']}, Real frames: {result_count['real']}")

    return video_result, result_count['fake'] / (result_count['fake'] + result_count['real'])

@app.route('/')
def index():
    return render_template('index.html', message='Welcome to the Deepfake Detection API')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        if file:
            filename = str(uuid.uuid4()) + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                video_result, confidence = classify_video(filepath)
                
                # Store result in MongoDB
                result = {
                    'filename': filename,
                    'is_deepfake': video_result == 'fake',
                    'confidence': float(confidence),
                    'timestamp': datetime.datetime.utcnow()
                }
                results_collection.insert_one(result)
                
                return jsonify({
                    'is_deepfake': video_result == 'fake',
                    'confidence': float(confidence)
                })
            except Exception as e:
                app.logger.error(f'Error processing video: {str(e)}')
                return jsonify({'error': 'There was an error processing the video.'}), 500
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
    except Exception as e:
        app.logger.error(f'Unexpected error: {str(e)}')
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=False)  # Changed to False for production use