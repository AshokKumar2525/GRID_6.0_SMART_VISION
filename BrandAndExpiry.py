import cv2
import numpy as np
import base64
import pytesseract
import re
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
from inference_sdk import InferenceHTTPClient
from collections import defaultdict
from PIL import Image
import random

# Initialize inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="xSDpa81O2Q9cacyvS4Wl"
)

# Function to clean and refine the expiry date
def clean_date(date):
    """
    Clean and refine the date to handle various formats like DD/MM/YYYY, MM/YYYY, or malformed years.
    """
    current_year = datetime.now().year
    current_year_short = int(str(current_year)[-2:])
    max_year = current_year + 2  # Allow up to 2 years in the future
    max_year_short = int(str(max_year)[-2:])

    # Split the date by delimiters
    date_parts = re.split(r'[-/]', date)

    if len(date_parts) == 2:
        # Handle MM/YYYY or MM/YY formats
        month, year = date_parts
        if len(year) == 2:
            year = int(year)
            if year > max_year_short:
                return None
            year = f"{current_year // 100}{year}"
        day = random.randint(1, 28)  # Assign a random day
        return f"{day}/{month}/{year}"

    if len(date_parts) == 3:
        # Handle DD/MM/YYYY or malformed years
        day, month, year = date_parts
        if len(year) == 4:
            year = int(year)
            if year > max_year:
                year = str(year)[:2]  # Truncate malformed year
        elif len(year) == 2:
            year = int(year)
            if year > max_year_short:
                return None
            year = f"{current_year // 100}{year}"
        return f"{day}/{month}/{year}"

    return None

def format_year(date):
    """
    Ensures the year in the date is in four-digit format.

    Args:
    - date (str): A date string in the format 'DD/MM/YYYY' or similar.

    Returns:
    - str: The formatted date string with a four-digit year, or "Invalid Date" if the format is incorrect.
    """
    if not date:  # Check for empty or None input
        return "Invalid Date"

    parts = date.split('/')
    
    # Validate the date format
    if len(parts) != 3:
        print(f"Warning: Invalid date format '{date}'. Skipping.")
        return "Invalid Date"
    
    # Format year to four digits if needed
    if len(parts[2]) == 2:  # If year has two digits
        parts[2] = '20' + parts[2]
    return '/'.join(parts)

def clean_expiry_dates(expiry_dates, num_brands):
    """
    Cleans the expiry dates by removing adjacent dates differing by less than 30 days 
    and adjusts the number of dates to match the number of brands.

    Parameters:
    expiry_dates (list): A list of lists containing expiry dates as strings.
    num_brands (int): The number of brands to match the dates to.

    Returns:
    list: A list of cleaned expiry dates.
    """
    # Flatten the list to extract all dates
    all_dates = [date for sublist in expiry_dates for date in sublist if date]

    # Parse dates into datetime objects with support for both %Y and %y formats
    parsed_dates = []
    for date in all_dates:
        try:
            parsed_dates.append(datetime.strptime(date, "%d/%m/%Y"))
        except ValueError:
            parsed_dates.append(datetime.strptime(date, "%d/%m/%y"))

    # Sort dates in ascending order
    parsed_dates.sort()

    # Filter dates by ensuring at least 30 days difference between consecutive dates
    filtered_dates = []
    for date in parsed_dates:
        if not filtered_dates or (date - filtered_dates[-1]).days >= 30:
            filtered_dates.append(date)

    # Adjust the number of dates to match the number of brands
    if len(filtered_dates) > num_brands:
        filtered_dates = filtered_dates[:num_brands]
    elif len(filtered_dates) < num_brands:
        # Pad with 'NA' if there are not enough dates
        filtered_dates = [None] * (num_brands - len(filtered_dates)) + filtered_dates

    # Convert dates back to string format
    cleaned_dates = [date.strftime("%d/%m/%Y") if date else 'NA' for date in filtered_dates]

    return cleaned_dates

# Extract dates from the image using OCR
def extract_dates_from_image(image):
    """
    Extract and refine dates from an image.
    """
    try:
        text = pytesseract.image_to_string(image)
        date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}[-/]\d{2,4})\b'
        dates = re.findall(date_pattern, text)

        refined_dates = set()
        for date in dates:
            cleaned_date = clean_date(date)
            if cleaned_date:
                refined_dates.add(cleaned_date)

        return list(refined_dates)
    except Exception as e:
        print(f"Error processing the image: {e}")
        return []

# Extract dates from the video using OCR
def extract_dates_from_video(video_path, skip_frames=5):
    """
    Extract and refine dates from a video.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        dates_per_frame = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(pil_image)

            # Print OCR text for debugging
            print(f"Text from frame {frame_count}: {text}")

            date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}[-/]\d{2,4})\b'
            dates = re.findall(date_pattern, text)

            refined_dates = set()
            for date in dates:
                cleaned_date = clean_date(date)
                if cleaned_date:
                    refined_dates.add(cleaned_date)

            dates_per_frame.append(list(refined_dates))

        cap.release()
        return dates_per_frame
    except Exception as e:
        print(f"Error processing the video: {e}")
        return []

# Function to process a single image and return brand counts and expiry dates
def process_brand_image(image):
    try:
        _, buffer = cv2.imencode('.jpg', image)
        image_b64 = base64.b64encode(buffer).decode('utf-8')

        result = CLIENT.infer(image_b64, model_id="grocery-dataset-q9fj2/5")
        predictions = result.get('predictions', [])

        brand_counts = defaultdict(int)

        for pred in predictions:
            brand_name = pred['class']
            brand_counts[brand_name] += 1

        return brand_counts
    except Exception as e:
        raise ValueError(f"Error during image processing: {str(e)}")

# Function to process a video and return brand counts
def process_brand_video(video_path, output_video_path, skip_frames=90, resize_dim=(640, 480)):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Define video writer for the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        fps = cap.get(cv2.CAP_PROP_FPS) / skip_frames  # Adjust FPS based on skip_frames
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        overall_brand_counts = defaultdict(int)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames == 0:
                try:
                    resized_frame = cv2.resize(frame, resize_dim)
                    frame_brand_counts = process_brand_image(resized_frame)

                    for brand, count in frame_brand_counts.items():
                        overall_brand_counts[brand] += count

                    # Write the processed frame to the output video
                    out.write(resized_frame)
                except Exception as e:
                    print(f"Error processing frame {frame_count + 1}: {str(e)}")
            frame_count += 1

        cap.release()
        out.release()  # Release the video writer
        return overall_brand_counts
    except Exception as e:
        raise ValueError(f"Error during video processing: {str(e)}")

# Function to process the expiry date video
def process_expiry_date_video(video_path, skip_frames=5):
    try:
        expiry_dates = extract_dates_from_video(video_path, skip_frames)
        return expiry_dates
    except Exception as e:
        raise ValueError(f"Error during expiry date video processing: {str(e)}")

def map_brands_to_expiry_dates(brand_counts, expiry_dates):
    """
    Maps brand names and their counts to expiry dates.

    Args:
    - brand_counts (dict): Dictionary with brand names as keys and their counts as values.
    - expiry_dates (list): List of expiry dates corresponding to the brands.

    Returns:
    - list: List of tuples containing brand name, count, and expiry date.
    """
    # Initialize the result list
    mapped_results = []

    # Get the list of brand names and counts
    brand_items = list(brand_counts.items())

    # Iterate over brand_items and map with expiry_dates safely
    for i, (brand_name, count) in enumerate(brand_items):
        # Check if the expiry date index exists
        expiry_date = expiry_dates[i] if i < len(expiry_dates) else "No Expiry Date Available"
        mapped_results.append((brand_name, count, expiry_date))

    return mapped_results

# Flask Blueprint
brand_blueprint = Blueprint('brand_blueprint', __name__)

@brand_blueprint.route('/detect-brand-expiry', methods=['POST'])
def detect_brand_expiry():
    try:
        if 'brand_video' in request.files and 'expiry_video' in request.files:
            brand_video = request.files['brand_video']
            expiry_video = request.files['expiry_video']

            brand_video_path = "static/uploads/brand_video.mp4"
            expiry_video_path = "static/uploads/expiry_video.mp4"
            brand_video.save(brand_video_path)
            expiry_video.save(expiry_video_path)

            # Process Brand Video
            brand_counts = process_brand_video(brand_video_path)

            # Process Expiry Date Video
            expiry_dates = process_expiry_date_video(expiry_video_path)

            # Clean expiry dates
            cleaned_expiry_dates = clean_expiry_dates(expiry_dates)

            # Map brands to expiry dates
            mapped_results = map_brands_to_expiry_dates(brand_counts, cleaned_expiry_dates)

            return jsonify({
                "message": "Brand and expiry date videos processed successfully",
                "brand_counts": brand_counts,
                "expiry_dates": mapped_results
            })

        return jsonify({"error": "No valid files provided"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Script entry point
if __name__ == "__main__":
    print("Select mode: \n1. Image\n2. Video")
    mode = input("Enter 1 for image or 2 for video: ")
    brand_output_path = "static/processed/brand_video.mp4"

    if mode == "2":
        print("Please provide two separate videos for processing:")
        brand_video_path = input("Enter the path to the brand detection video: ")
        expiry_video_path = input("Enter the path to the expiry date detection video: ")

        # Process Brand Video
        brand_counts = process_brand_video(brand_video_path,brand_output_path)
        print("Brand Counts:", brand_counts)

        # Process Expiry Date Video
        expiry_dates = process_expiry_date_video(expiry_video_path)
        print("Expiry Dates:", expiry_dates)

        # Clean expiry dates
        cleaned_expiry_dates = clean_expiry_dates(expiry_dates,len(brand_counts))
        print("Cleaned Expiry Dates:", cleaned_expiry_dates)

        # Map brands to expiry dates
        mapped_results = map_brands_to_expiry_dates(brand_counts, cleaned_expiry_dates)
        print("Mapped Results:", mapped_results)

    elif mode == "1":
        brand_image_path = input("Enter the path to the brand detection image: ")
        expiry_image_path = input("Enter the path to the expiry date detection image: ")

        brand_image = cv2.imread(brand_image_path)
        expiry_image = cv2.imread(expiry_image_path)

        # Process Image
        brand_counts = process_brand_image(brand_image)
        print("Brand Counts:", brand_counts)

        # Extract expiry dates from image
        expiry_dates = extract_dates_from_image(expiry_image)
        print("Expiry Dates:", expiry_dates)
        if len(expiry_dates)==1:
            expiry_dates=[expiry_dates]
        # Clean expiry dates
        cleaned_expiry_dates = clean_expiry_dates(expiry_dates, len(brand_counts))

        print("Cleaned Expiry Dates:", cleaned_expiry_dates)

        # Map brands to expiry dates
        mapped_results = map_brands_to_expiry_dates(brand_counts, cleaned_expiry_dates)
        print("Mapped Results:", mapped_results)
    else:
        print("Enter only 1 or 2")