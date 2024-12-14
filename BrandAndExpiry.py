import cv2
import numpy as np
import base64
import pytesseract
import re
from datetime import datetime, timedelta
from inference_sdk import InferenceHTTPClient
from collections import defaultdict
from PIL import Image
import random

# Initialize inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="xSDpa81O2Q9cacyvS4Wl"
)

# Function to process a single image and return brand counts
def process_brand_image(image, previous_predictions, brand_counts):
    try:
        _, buffer = cv2.imencode('.jpg', image)
        image_b64 = base64.b64encode(buffer).decode('utf-8')

        result = CLIENT.infer(image_b64, model_id="grocery-dataset-q9fj2/5")
        predictions = result.get('predictions', [])

        # Clean predictions based on overlap and confidence score
        cleaned_predictions = clean_predictions(predictions)

        # Check for each cleaned prediction if it is already counted
        new_predictions = []
        for pred in cleaned_predictions:
            x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
            brand_name = pred['class']
            should_add = True

            # Compare with previous predictions
            for prev_pred in previous_predictions:
                prev_x, prev_y, prev_w, prev_h, prev_name = prev_pred
                if prev_name == brand_name and (x < prev_x * 0.8):  # Assuming a 20% leftward difference
                    should_add = False
                    # Update previous prediction with current x
                    previous_predictions[previous_predictions.index(prev_pred)] = (x, y, width, height, brand_name)
                    break

            if should_add:
                new_predictions.append((x, y, width, height, brand_name))
                if brand_name in brand_counts:
                    brand_counts[brand_name] += 1
                else:
                    brand_counts[brand_name] = 1

        # Draw bounding boxes and labels for valid predictions
        for pred in new_predictions:
            x, y, width, height, brand_name = pred
            top_left = (int(x - width / 2), int(y - height / 2))
            bottom_right = (int(x + width / 2), int(y + height / 2))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            label = f"Brand: {brand_name}"
            cv2.putText(image, label, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image, new_predictions, brand_counts
    except Exception as e:
        raise ValueError(f"Error during image processing: {str(e)}")

# Function to clean predictions by removing overlapping bounding boxes
def clean_predictions(predictions):
    cleaned_predictions = []
    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)

    for pred in predictions:
        x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
        overlap = False
        for prev_pred in cleaned_predictions:
            prev_x, prev_y, prev_width, prev_height = prev_pred['x'], prev_pred['y'], prev_pred['width'], prev_pred['height']
            if is_overlap(x, y, width, height, prev_x, prev_y, prev_width, prev_height):
                overlap = True
                break
        if not overlap:
            cleaned_predictions.append(pred)

    return cleaned_predictions

# Function to check if two bounding boxes overlap
def is_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    x_overlap = max(0, min(x1_max, x2_max) - max(x1, x2))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1, y2))

    overlap_area = x_overlap * y_overlap
    box1_area = w1 * h1
    box2_area = w2 * h2

    overlap_ratio = overlap_area / float(min(box1_area, box2_area))

    return overlap_ratio > 0.5

# Function to process a video and return brand counts
def process_brand_video(video_path, output_path, skip_frames=90, resize_dim=(640, 480)):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        previous_predictions = []  
        brand_counts = defaultdict(int)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames == 0:
                try:
                    resized_frame = cv2.resize(frame, resize_dim)
                    processed_frame, new_predictions, brand_counts = process_brand_image(resized_frame, previous_predictions, brand_counts)
                    processed_frame = cv2.resize(processed_frame, (width, height))

                    previous_predictions = new_predictions

                    for _ in range(int(0.5 * fps)):
                        out.write(processed_frame)
                except Exception as e:
                    print(f"Error processing frame {frame_count + 1}: {str(e)}")
                    out.write(frame)
            else:
                out.write(frame)

            frame_count += 1

        cap.release()
        out.release()

        print("Brand counts:")
        for brand, count in brand_counts.items():
            print(f"{brand}: {count}")

        print("Video processing complete.")
        return brand_counts
    except Exception as e:
        raise ValueError(f"Error during video processing: {str(e)}")


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
        expiry_date = expiry_dates[i] if i < len(expiry_dates) else "NA"
        mapped_results.append((brand_name, count, expiry_date))

    return mapped_results

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


def extract_dates_from_image(image_path):
    """
    Extract and refine dates from an image.
    """
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        print("Extracted text from image:", text)

        date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}[-/]\d{2,4}|(?:0?[1-9]|1[0-2])[-/]\d{2,4})\b'
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


def extract_dates_from_video(video_path, skip_frames=10):
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
            print(f"Extracted text from frame {frame_count}: {text}")

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


def extract_and_clean_expiry_dates(detected_dates, range_days=30):
    """
    Extract expiry dates from detected dates in video frames, validate them,
    and merge expiry dates that differ by a specified number of days.

    Parameters:
        detected_dates (dict): A dictionary with frame numbers as keys and lists of detected dates as values.
        range_days (int): The maximum allowed difference in days to consider dates as the same item.

    Returns:
        dict: A dictionary with consolidated expiry dates, preserving the original frame order.
    """
    # Step 1: Extract expiry dates
    expiry_dates = {}
    current_date = datetime.now()

    # Store all dates across frames for checking later
    all_dates = {}

    for frame, dates in detected_dates.items():
        if not dates:
            continue

        # Add to all_dates to check for duplicates across frames
        all_dates[frame] = dates

        if len(dates) == 1:
            # Single date: Check conditions for expiry date
            try:
                detected_date = datetime.strptime(dates[0], "%d/%m/%Y")

                # If the detected date is in the current year, add 1 year
                if detected_date.year == current_date.year:
                    expiry_date = detected_date + timedelta(days=365)
                    expiry_dates[frame] = expiry_date.strftime("%d/%m/%Y")
                elif detected_date.year > current_date.year and detected_date.year <= current_date.year + 2:
                    # If the detected date is in the future (within 1 or 2 years)
                    expiry_dates[frame] = detected_date.strftime("%d/%m/%Y")
            except ValueError:
                continue

        elif len(dates) > 1:
            # Multiple dates: Validate and find the latest one (expiry date)
            valid_dates = []
            for date_str in dates:
                try:
                    valid_dates.append(datetime.strptime(date_str, "%d/%m/%Y"))
                except ValueError:
                    continue

            if valid_dates:
                # Take the latest (highest) date as the expiry date
                expiry_date = max(valid_dates)
                expiry_dates[frame] = expiry_date.strftime("%d/%m/%Y")

    # Step 2: Merge expiry dates
    frame_date_list = [(frame, datetime.strptime(date, "%d/%m/%Y")) for frame, date in expiry_dates.items()]
    
    merged_dates = {}  # To store merged results
    if frame_date_list:
        current_group = [frame_date_list[0]]  # Start with the first frame-date pair

        for i in range(1, len(frame_date_list)):
            current_frame, current_date = frame_date_list[i]
            last_frame, last_date = current_group[-1]

            # Check if the current date is within the range of the last date in the group
            if abs((current_date - last_date).days) <= range_days:
                current_group.append((current_frame, current_date))
            else:
                best_frame, best_date = current_group[0]  # Take the first in frame order
                merged_dates[best_frame] = best_date.strftime("%d/%m/%Y")
                current_group = [(current_frame, current_date)]

        if current_group:
            best_frame, best_date = current_group[0]  # Take the first in frame order
            merged_dates[best_frame] = best_date.strftime("%d/%m/%Y")

    # Step 3: Check for single dates appearing as manufacture date
    for frame, dates in all_dates.items():
        if len(dates) == 1:
            for other_frame, other_dates in all_dates.items():
                if len(other_dates) == 2:
                    # If the single date appears in any frame with two dates, we exclude it as manufacture date
                    if dates[0] in other_dates:
                        print(f"Excluding {dates[0]} in frame {frame} as it appears as manufacture date in frame {other_frame}.")
                        continue

    return merged_dates


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

    elif mode == "1":
        brand_image_path = input("Enter the path to the brand detection image: ")
        expiry_image_path = input("Enter the path to the expiry date detection image: ")

        brand_image = cv2.imread(brand_image_path)
        expiry_image = cv2.imread(expiry_image_path)
        previous_predictions = []  
        brand_counts = defaultdict(int)
        # Process Image
        image, new_predictions, brand_counts= process_brand_image(brand_image,previous_predictions,brand_counts)
        print("Brand Counts:", brand_counts)

        # Extract expiry dates from image
        expiry_dates = extract_dates_from_image(expiry_image_path)
        print("Expiry Dates:", expiry_dates)

        detected_dates = {i + 1: dates for i, dates in enumerate(expiry_dates)}

        # Clean expiry dates
        consolidated_expiry_dates = extract_and_clean_expiry_dates(detected_dates)
        cleaned_expiry_dates = [expiry for frame, expiry in consolidated_expiry_dates.items()]
        print("Cleaned Expiry Dates:", cleaned_expiry_dates)


        # Map brands to expiry dates
        mapped_results = map_brands_to_expiry_dates(brand_counts, cleaned_expiry_dates)
        print("Mapped Results:", mapped_results)
    else:
        print("Enter only 1 or 2")
