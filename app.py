from flask import Flask, render_template, request
from freshness_detection import process_image, process_video
import cv2
import base64
import numpy as np
import os
from database import insert_brand_data_batch
from view import fetch_data_from_database
from BrandAndExpiry import process_brand_video,extract_dates_from_video,extract_and_clean_expiry_dates,map_brands_to_expiry_dates,process_brand_image,extract_dates_from_image
from collections import defaultdict


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
    freshness_results=None
    error = None

    if request.method == 'POST':
        # Handle image upload
        if 'image' in request.files and request.files['image'].filename:
            uploaded_image = request.files['image']
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            try:
                # Process the image for freshness detection
                processed_img,freshness_results = process_image(image)

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
                freshness_results=process_video(video_path, processed_video_path)

                # Assign the processed video path for rendering in HTML
                processed_video = processed_video_path
            except Exception as e:
                error = f"Error processing video: {str(e)}"

    return render_template(
        'task2.html',
        processed_image=processed_image,
        processed_video=processed_video,
        freshness_results=freshness_results,
        error=error
    )


@app.route('/task1', methods=['GET', 'POST'])
def task1():
    processed_image_brand = None
    processed_image_expiry = None
    processed_video_brand = None
    mapped_results = None
    error = None

    if request.method == 'POST':
        # Handle image uploads (brand and expiry images)
        if 'brand_image' in request.files and request.files['brand_image'].filename and \
           'expiry_image' in request.files and request.files['expiry_image'].filename:
            try:
                # Read and process brand and expiry images from the uploaded files
                brand_image = request.files['brand_image']
                expiry_image = request.files['expiry_image']

                brand_imgae_path = os.path.join(UPLOAD_FOLDER, brand_image.filename)
                expiry_image_path = os.path.join(UPLOAD_FOLDER, expiry_image.filename)
                expiry_image.save(expiry_image_path)

                # Read image data into numpy array
                brand_image_bytes = np.asarray(bytearray(brand_image.read()), dtype=np.uint8)

                # Decode images using OpenCV
                brand_image = cv2.imdecode(brand_image_bytes, 1)
                previous_predictions = []  
                brand_counts = defaultdict(int)

                # Process Image
                image, new_predictions, brand_counts= process_brand_image(brand_image,previous_predictions,brand_counts)
                print("Brand Counts:", new_predictions)

                # Extract expiry dates from image
                expiry_dates = extract_dates_from_image(expiry_image_path)
                print("Expiry Dates:", expiry_dates)

                detected_dates = {i + 1: dates for i, dates in enumerate(expiry_dates)}

                # Clean expiry dates
                consolidated_expiry_dates = extract_and_clean_expiry_dates(detected_dates)
                print(consolidated_expiry_dates)
                cleaned_expiry_dates = [expiry for frame, expiry in consolidated_expiry_dates.items()]
                print("Cleaned Expiry Dates:", cleaned_expiry_dates)


                # Map brands to expiry dates
                mapped_results = map_brands_to_expiry_dates(brand_counts, cleaned_expiry_dates)
                print("Mapped Results:", mapped_results)


                insert_brand_data_batch(mapped_results)
            except Exception as e:
                error = f"Error processing images: {str(e)}"

        # Handle video uploads (brand and expiry videos) [This part remains unchanged]
        if 'brand_video' in request.files and request.files['brand_video'].filename and \
           'expiry_video' in request.files and request.files['expiry_video'].filename:
            try:
                # Save uploaded videos and process them as needed (same logic as before)
                brand_video = request.files['brand_video']
                expiry_video = request.files['expiry_video']

                brand_video_path = os.path.join(UPLOAD_FOLDER, brand_video.filename)
                expiry_video_path = os.path.join(UPLOAD_FOLDER, expiry_video.filename)

                processed_brand_video_path = os.path.join(PROCESSED_FOLDER, "processed_" + brand_video.filename)

                brand_video.save(brand_video_path)
                expiry_video.save(expiry_video_path)

                # Process Brand Video
                brand_counts = process_brand_video(brand_video_path,processed_brand_video_path)
                print("Brand Counts:", brand_counts)

                # Process Expiry Date Video
                expiry_dates = extract_dates_from_video(expiry_video_path)
                print("Expiry Dates:", expiry_dates)

                detected_dates = {i + 1: dates for i, dates in enumerate(expiry_dates)}

                # Clean expiry dates
                consolidated_expiry_dates = extract_and_clean_expiry_dates(detected_dates)
                cleaned_expiry_dates = [expiry for frame, expiry in consolidated_expiry_dates.items()]
                print("Cleaned Expiry Dates:", cleaned_expiry_dates)

                # Map brands to expiry dates
                mapped_results = map_brands_to_expiry_dates(brand_counts, cleaned_expiry_dates)
                print("Mapped Results:", mapped_results)

                processed_video_brand = processed_brand_video_path
                insert_brand_data_batch(mapped_results)

            except Exception as e:
                error = f"Error processing videos: {str(e)}"

    return render_template(
        'task1.html',
        processed_image_brand=processed_image_brand,
        processed_image_expiry=processed_image_expiry,
        processed_video_brand=processed_video_brand,
        mapped_results=mapped_results,
        error=error
    )

@app.route('/database')
def database():
    # Fetch data from database
    brand_data, freshness_data = fetch_data_from_database()

    # Render the database.html template and pass the data
    return render_template('database.html', brand_data=brand_data, freshness_data=freshness_data)



@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True,host="0.0.0.0", port=port)
