import cv2
from PIL import Image
import pytesseract
import re
from datetime import datetime
import random


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


# Example usage
if __name__ == "__main__":
    # For Image
    image_path = "last2.png"
    dates_image = extract_dates_from_image(image_path)
    if dates_image:
        print("Dates found in the image:", dates_image)
    else:
        print("No dates found in the image.")

    # For Video
    # video_path = "output3.mp4"
    # dates_video = extract_dates_from_video(video_path)
    # if any(dates_video):
    #     print("Dates found in the video frames:")
    #     for i, frame_dates in enumerate(dates_video):
    #         print(f"Frame {i + 1}: {frame_dates}")
    # else:
    #     print("No dates found in the video.")

