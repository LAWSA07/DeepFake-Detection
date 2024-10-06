from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
import datetime
import logging
from logging.handlers import RotatingFileHandler
from pymongo import MongoClient
import tensorflow as tf
from dotenv import load_dotenv
from utils.video_processor import process_video
from keras.models import load_model

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configuration
class Config:
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb+srv://aswalh0707:l9h4BplwLxHIywm9@cluster0.0pfhp.mongodb.net/deepfake-detection?retryWrites=true&w=majority&appName=Cluster0')
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv'}

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

# Load the model
try:
    model = load_model('DeepFake_Model_new3.h5')
    app.logger.info('Model loaded successfully')
except Exception as e:
    app.logger.error(f'Failed to load model: {str(e)}')
    raise

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

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
                is_deepfake, confidence = process_video(filepath, model)
                
                # Store result in MongoDB
                result = {
                    'filename': filename,
                    'is_deepfake': bool(is_deepfake),
                    'confidence': float(confidence),
                    'timestamp': datetime.datetime.utcnow()
                }
                results_collection.insert_one(result)
                
                return jsonify({
                    'is_deepfake': bool(is_deepfake),
                    'confidence': float(confidence)
                })
            except Exception as e:
                app.logger.error('Error processing video:', error)
                return jsonify({'error': 'Error processing video'}), 500
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
    except Exception as e:
        app.logger.error(f'Unexpected error: {str(e)}')
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=False)