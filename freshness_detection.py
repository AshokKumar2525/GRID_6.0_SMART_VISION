import cv2
import numpy as np
import base64
from flask import Blueprint, request, jsonify
from inference_sdk import InferenceHTTPClient
from database import insert_freshness_data_batch

# Initialize client for inference
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="xSDpa81O2Q9cacyvS4Wl"
)

# Define freshness index mappings
FRESHNESS_INDEX_MAP = {
    "Fresh": 100,
    "Semifresh": 75,
    "Semirotten": 50,
    "Rotten": 25
}
# Function to process a single image
def process_image_with_details(image):
    """
    Process an image to detect freshness, draw bounding boxes, and return detection details.
    """
    try:
        # Convert the image to a base64-encoded string
        _, buffer = cv2.imencode('.jpg', image)
        image_b64 = base64.b64encode(buffer).decode('utf-8')

        # Perform inference using the RoboFlow API
        result = CLIENT.infer(image_b64, model_id="freshness-nnryh/1")
        predictions = result.get('predictions', [])
        frame_details = []  # List to collect details for this frame

        # Parse predictions and draw bounding boxes
        for pred in predictions:
            x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
            class_name = pred['class'].replace(" ", "_")

            # Parse class name and map to freshness index
            freshness_class, fruit_name = parse_class_name(class_name)
            freshness_index = FRESHNESS_INDEX_MAP.get(freshness_class, 0)
            
            # Collect details for this produce
            frame_details.append((fruit_name, freshness_index))  # Adjust as per database schema

            # Draw bounding box and label
            top_left = (int(x - width / 2), int(y - height / 2))
            bottom_right = (int(x + width / 2), int(y + height / 2))
            cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
            label = f"{fruit_name}: {freshness_class} ({freshness_index})"
            cv2.putText(image, label, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return image, frame_details
    except Exception as e:
        raise ValueError(f"Error during image processing: {str(e)}")




def process_image(image):
    """
    Process an image to detect freshness, draw bounding boxes, and return the result.
    """
    try:
        # Convert the image to a base64-encoded string
        _, buffer = cv2.imencode('.jpg', image)
        image_b64 = base64.b64encode(buffer).decode('utf-8')

        # Perform inference using the RoboFlow API
        result = CLIENT.infer(image_b64, model_id="freshness-nnryh/1")
        predictions = result.get('predictions', [])

        details = []  # List to hold detection details

        # Parse predictions and draw bounding boxes
        for pred in predictions:
            x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
            class_name = pred['class'].replace(" ", "_")
            
            # Parse class name and map to freshness index
            freshness_class, fruit_name = parse_class_name(class_name)
            freshness_index = FRESHNESS_INDEX_MAP.get(freshness_class, 0)
            details.append((fruit_name, freshness_index))
            insert_freshness_data_batch([(fruit_name, freshness_index)])

            # Draw bounding box and label
            top_left = (int(x - width / 2), int(y - height / 2))
            bottom_right = (int(x + width / 2), int(y + height / 2))
            cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
            label = f"{fruit_name}: {freshness_class} ({freshness_index})"
            cv2.putText(image, label, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        print("Detection details for the image:", details)  # Print detection details
        return image, details
    except Exception as e:
        raise ValueError(f"Error during image processing: {str(e)}")


def parse_class_name(class_name):
    """Parse class name and extract freshness class and fruit name."""
    if "_" in class_name:
        cls_values = class_name.split("_", 1)
        if cls_values[0] in FRESHNESS_INDEX_MAP:
            return cls_values[0], cls_values[1]
        return cls_values[1], cls_values[0]
    return "Unknown", "Unknown"

# Function to process a video
def process_video(video_path, output_path, skip_frames=25):
    """
    Process a video to detect freshness for every nth frame, collect details of produce,
    and send them as a list to the database after processing.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Get video properties
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define VideoWriter object
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        processed_count = 0
        produce_details = []  # List to collect details of all detected produce

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every nth frame
            if frame_count % skip_frames == 0:
                try:
                    print(f"Processing frame {frame_count + 1}/{total_frames}...")
                    # Process the frame and collect detection details
                    processed_frame, frame_details = process_image_with_details(frame)
                    processed_count += 1
                    produce_details.extend(frame_details)  # Append current frame details to the list
                    out.write(processed_frame)
                except Exception as e:
                    print(f"Error processing frame {frame_count + 1}: {str(e)}")
                    out.write(frame)  # Write original frame if processing fails
            else:
                out.write(frame)  # Write unprocessed frames

            frame_count += 1

        # Send collected details to the database
        if produce_details:
            insert_freshness_data_batch(produce_details)

        print(f"Processing completed. Processed {processed_count} frames out of {total_frames}.")
        print("Detection details for the video:", produce_details)  # Print detection details for the video
        cap.release()
        out.release()
        return produce_details
    except Exception as e:
        raise ValueError(f"Error during video processing: {str(e)}")
    
# Flask Blueprint
freshness_blueprint = Blueprint('freshness_blueprint', __name__)

@freshness_blueprint.route('/detect-freshness', methods=['POST'])
def detect_freshness():
    """Handle API requests for image and video freshness detection."""
    try:
        if 'image' in request.files:
            uploaded_file = request.files['image']
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            processed_image = process_image(image)
            _, buffer = cv2.imencode('.jpg', processed_image)
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            return jsonify({"message": "Image processed successfully", "image": image_b64})

        elif 'video' in request.files:
            uploaded_file = request.files['video']
            input_path = "static/uploads/input_video.mp4"
            output_path = "static/processed/output_video.mp4"
            uploaded_file.save(input_path)
            process_video(input_path, output_path)
            return jsonify({"message": "Video processed successfully", "output_path": output_path})

        return jsonify({"error": "No valid file provided"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Test the script independently
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
                # Process the image and get details
                processed_image, data_list = process_image(image)

                # Save and display the processed image
                processed_path = "processed_image.jpg"
                cv2.imwrite(processed_path, processed_image)
                print(f"Processed image saved at: {processed_path}")
                print("Detected Details (Fruit Name and Freshness Index):")
                for fruit_name, freshness_index in data_list:
                    print(f"Fruit: {fruit_name}, Freshness Index: {freshness_index}")

                # Display the processed image
                cv2.imshow("Processed Image", processed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Error during image processing: {str(e)}")

    elif mode == "2":
        # Test with a video
        video_path = input("Enter the path to the test video: ")
        output_path = "processed_video.mp4"

        try:
            # Process the video and get details
            data = process_video(video_path, output_path)
            print(f"Processed video saved at: {output_path}")
            print("Detected Details (Fruit Name and Freshness Index):")
            for fruit_name, freshness_index in data:
                print(f"Fruit: {fruit_name}, Freshness Index: {freshness_index}")
        except Exception as e:
            print(f"Error during video processing: {str(e)}")

    else:
        print("Invalid input. Please enter 1 for image or 2 for video.")
