# Deep Fake Detection System

This project is a web application that detects deep fake videos using machine learning.

## Setup Instructions

1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

3. Ensure MongoDB is installed and running

4. Place your trained model file (model.h5) in the project root directory

5. Run the application:
   ```
   python app.py
   ```

6. Open a web browser and go to http://localhost:5000

## Project Structure

```
deep_fake/
│   app.py
│   requirements.txt
│   model.h5
│   README.md
│
├── static/
│   └── css/
│       └── style.css
│   └── js/
│       └── script.js
│
├── templates/
│   └── index.html
│
├── uploads/
│
└── utils/
    └── video_processor.py
```

## Usage

1. Click "Choose Video" to select a video file
2. Click "Analyze Video" to process the video
3. Wait for the result to appear

## Notes

- The system processes every 5th frame of the video for efficiency
- Results are stored in MongoDB for future reference
- Uploaded videos are automatically deleted after processing
```