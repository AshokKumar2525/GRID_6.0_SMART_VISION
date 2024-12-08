from flask import Flask, render_template, request
from freshness_detection import process_image, process_video
from brand_products import detect_brands_and_count_image, detect_brands_and_count_video
import cv2
import base64
import numpy as np
import os

app = Flask(__name__)

# Directory to save uploaded and processed videos
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/task2', methods=['GET', 'POST'])
def task2():
    processed_image = None
    processed_video = None
    error = None

    if request.method == 'POST':
        # Handle image upload
        if 'image' in request.files and request.files['image'].filename:
            uploaded_image = request.files['image']
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            try:
                # Process the image for freshness detection
                processed_img = process_image(image)

                # Convert the processed image to base64 for rendering in HTML
                _, buffer = cv2.imencode('.jpg', processed_img)
                processed_image = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                error = f"Error processing image: {str(e)}"

        # Handle video upload
        if 'video' in request.files and request.files['video'].filename:
            uploaded_video = request.files['video']
            video_path = os.path.join(UPLOAD_FOLDER, uploaded_video.filename)
            processed_video_path = os.path.join(PROCESSED_FOLDER, "processed_" + uploaded_video.filename)
            uploaded_video.save(video_path)

            try:
                # Process the video for freshness detection
                process_video(video_path, processed_video_path)

                # Assign the processed video path for rendering in HTML
                processed_video = processed_video_path
            except Exception as e:
                error = f"Error processing video: {str(e)}"

    return render_template(
        'task2.html',
        processed_image=processed_image,
        processed_video=processed_video,
        error=error
    )

@app.route('/task1', methods=['GET', 'POST'])
def task1():
    processed_image = None
    processed_video = None
    brand_counts = None
    expiry_result = {}  # Initialize as an empty dictionary
    error = None

    if request.method == 'POST':
        # Handle image upload
        if 'image' in request.files and request.files['image'].filename:
            uploaded_image = request.files['image']
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            try:
                # Perform brand recognition and counting on the image
                processed_img, brand_counts = detect_brands_and_count_image(image)

                # Perform expiry detection on the image
                # Make sure you assign a value to expiry_result here, for example:
                expiry_result = {"brand": "expiry_date"}

                # Convert processed image to base64 for displaying in HTML
                _, buffer = cv2.imencode('.jpg', processed_img)
                processed_image = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                error = f"Error processing image: {str(e)}"

        # Handle video upload
        if 'video' in request.files and request.files['video'].filename:
            uploaded_video = request.files['video']
            video_path = os.path.join(UPLOAD_FOLDER, uploaded_video.filename)
            processed_video_path = os.path.join(PROCESSED_FOLDER, "processed_" + uploaded_video.filename)
            uploaded_video.save(video_path)

            try:
                # Perform brand recognition and counting on the video
                brand_counts = detect_brands_and_count_video(video_path, processed_video_path)
                processed_video = processed_video_path
            except Exception as e:
                error = f"Error processing video: {str(e)}"

    return render_template(
        'task1.html',
        processed_image=processed_image,
        processed_video=processed_video,
        brand_counts=brand_counts,
        expiry_result=expiry_result,  # Ensure expiry_result is passed
        error=error
    )

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

