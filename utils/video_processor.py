import cv2
import os

def extract_frames(video_path, output_dir, frame_interval=1):
    os.makedirs(output_dir, exist_ok=True)
    
    video = cv2.VideoCapture(video_path)
    
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_num in range(0, frame_count, frame_interval):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = video.read()
        if ret:
            output_path = os.path.join(output_dir, f"frame_{frame_num:04d}.jpg")
            cv2.imwrite(output_path, frame)
    
    video.release()

def process_videos(input_dir, output_dir, frame_interval=1):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                video_output_dir = os.path.join(output_dir, relative_path, os.path.splitext(file)[0])
                extract_frames(video_path, video_output_dir, frame_interval)

# Usage example:
# process_videos('data/videos', 'data/frames', frame_interval=30)