import cv2
import os
import pytesseract
from inference_sdk import InferenceHTTPClient
from collections import Counter
import base64
import numpy as np

# Initialize Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="VDWdiDCrMynoYfeMyEeC"  # Replace with your actual API key if needed
)

# Directory to store processed files
PROCESSED_FOLDER = "processed"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def encode_image_to_base64(image_bytes):
    """
    Encode image byte array to base64 string for API inference.
    """
    base64_bytes = base64.b64encode(image_bytes).decode('utf-8')
    return base64_bytes

def process_image(image_path):
    """
    Process image to detect expiry date using Tesseract OCR and RoboFlow inference.
    """
    # Read the image as byte array
    with open(image_path, 'rb') as image_file:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)

    # Read the image using OpenCV
    image = cv2.imdecode(file_bytes, 1)

    # Encode the image to base64
    base64_image = encode_image_to_base64(file_bytes)

    # Perform inference using the RoboFlow API
    result = CLIENT.infer(base64_image, model_id="expiry-date-detection-ssxnm/1")

    # Extract predictions from the result
    predictions = result.get('predictions', [])

    if predictions:
        # Create a list to store extracted dates
        extracted_dates = []

        # Loop through all predictions and extract text from each ROI
        for prediction in predictions:
            x, y, w, h = int(prediction['x']), int(prediction['y']), int(prediction['width']), int(prediction['height'])

            # Extract the Region of Interest (ROI) based on the bounding box
            roi = image[y - h // 2:y + h // 2, x - w // 2:x + w // 2]

            # Use pytesseract to perform OCR on the ROI
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update path to Tesseract
            extracted_text = pytesseract.image_to_string(roi, config='--psm 6')  # Adjust config as needed

            # Append the extracted text to the list
            extracted_dates.append(extracted_text.strip())

        # Determine the manufacturing and expiry dates
        if len(extracted_dates) == 2:
            mfg_date = extracted_dates[0]
            expiry_date = extracted_dates[1]

            # Check if the last two characters of both dates are digits
            if mfg_date[-2:].isdigit() and expiry_date[-2:].isdigit():
                # Swap dates if necessary
                if int(mfg_date[-2:]) > int(expiry_date[-2:]):
                    mfg_date, expiry_date = expiry_date, mfg_date
            else:
                mfg_date = "Not Detected"
                expiry_date = "Not Detected"

        elif len(extracted_dates) == 1:
            mfg_date = extracted_dates[0]
            expiry_date = "Not Detected"
        else:
            mfg_date = "Not Detected"
            expiry_date = "Not Detected"

        # Print the manufacturing and expiry dates
        print("\nExtracted Dates:")
        print(f"Manufacturing Date: {mfg_date}")
        print(f"Expiry Date: {expiry_date}")
    else:
        print("No date predictions found.")


def detect_brands_and_count_image(image, save_path=None):
    """
    Detect brands in an image and count the items.
    """
    # Save the image temporarily for inference
    temp_image_path = os.path.join(PROCESSED_FOLDER, "temp_brand_image.png")
    cv2.imwrite(temp_image_path, image)

    # Perform inference
    try:
        result = CLIENT.infer(temp_image_path, model_id="grocery-dataset-q9fj2/5")

        print(result)
        if result.get('predictions') is None:
            raise ValueError("No predictions found in the response.")
    except Exception as e:
        print(f"Inference failed: {str(e)}")
        raise

    # Initialize counters and create a copy of the image for drawing
    brand_counts = Counter()
    predictions = result.get('predictions', [])
    processed_image = image.copy()

    # Process predictions
    for pred in predictions:
        try:
            # Use .get() to handle missing keys
            x = int(pred.get('x', 0) - pred.get('width', 0) / 2)
            y = int(pred.get('y', 0) - pred.get('height', 0) / 2)
            width = int(pred.get('width', 0))
            height = int(pred.get('height', 0))
            class_name = pred.get('class', 'Unknown')

            # Update the brand count
            brand_counts[class_name] += 1

            # Draw bounding boxes and labels on the processed image
            cv2.rectangle(processed_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(
                processed_image,
                class_name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        except Exception as e:
            print(f"Error processing prediction: {str(e)}")
            continue

    # Save the processed image if a path is provided
    if save_path:
        cv2.imwrite(save_path, processed_image)

    return processed_image, dict(brand_counts)


def detect_brands_and_count_video(video_path, output_path):
    """
    Detect brands in a video and count items frame-by-frame.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create a video writer for saving processed video
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize a counter for brand counts
    brand_counts = Counter()

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect brands and count items in the current frame
        try:
            processed_frame, frame_counts = detect_brands_and_count_image(frame)
            brand_counts.update(frame_counts)
            out.write(processed_frame)
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            continue

    # Release resources
    cap.release()
    out.release()

    return dict(brand_counts)


if __name__ == "__main__":
    # Test functionality independently
    print("Select mode: \n1. Image\n2. Video")
    mode = input("Enter 1 for image or 2 for video: ")

    if mode == "1":
        # Test with an image
        image_path = input("Enter the path to the test image: ")
        image = cv2.imread(image_path)

        if image is None:
            print("Error: Could not load image. Please check the file path.")
        else:
            try:
                # Process the image for both brand detection and expiry date detection
                processed_path = os.path.join(PROCESSED_FOLDER, "processed_image.png")
                processed_image, brand_counts = detect_brands_and_count_image(image, save_path=processed_path)

                # Display results
                print("Brand Counts:")
                for brand, count in brand_counts.items():
                    print(f"{brand}: {count}")

                # Show the processed image
                cv2.imshow("Processed Image", processed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # Perform expiry date detection
                process_image(image_path)

            except Exception as e:
                print(f"Error during processing: {str(e)}")

    elif mode == "2":
        # Test with a video
        video_path = input("Enter the path to the test video: ")
        output_path = os.path.join(PROCESSED_FOLDER, "processed_video.mp4")

        try:
            # Process the video for brand detection
            brand_counts = detect_brands_and_count_video(video_path, output_path)

            # Display results
            print("Brand Counts (aggregated across all frames):")
            for brand, count in brand_counts.items():
                print(f"{brand}: {count}")

            print(f"Processed video saved at: {output_path}")
        except Exception as e:
            print(f"Error during processing: {str(e)}")

    else:
        print("Invalid input. Please enter 1 for image or 2 for video.")
