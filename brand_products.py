import cv2
import numpy as np
import base64
from flask import Blueprint, request, jsonify
from inference_sdk import InferenceHTTPClient
from collections import defaultdict

# Initialize inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="xSDpa81O2Q9cacyvS4Wl"
)

# Function to process a single image and return brand counts
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

            x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
            top_left = (int(x - width / 2), int(y - height / 2))
            bottom_right = (int(x + width / 2), int(y + height / 2))

            # Draw bounding boxes and labels
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            label = f"Brand: {brand_name}"
            cv2.putText(image, label, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image, dict(brand_counts)
    except Exception as e:
        raise ValueError(f"Error during image processing: {str(e)}")

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
        overall_brand_counts = defaultdict(int)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames == 0:
                try:
                    resized_frame = cv2.resize(frame, resize_dim)
                    processed_frame, frame_brand_counts = process_brand_image(resized_frame)
                    processed_frame = cv2.resize(processed_frame, (width, height))
                    for brand, count in frame_brand_counts.items():
                        overall_brand_counts[brand] += count

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

        print("Video processing complete.")
        return overall_brand_counts
    except Exception as e:
        raise ValueError(f"Error during video processing: {str(e)}")

# Script entry point
if __name__ == "__main__":
    print("Select mode: \n1. Image\n2. Video")
    mode = input("Enter 1 for image or 2 for video: ")

    if mode == "1":
        image_path = input("Enter the path to the test image: ")
        image = cv2.imread(image_path)

        if image is None:
            print("Error: Could not load image. Please check the file path.")
        else:
            try:
                processed_image, brand_counts = process_brand_image(image)
                processed_path = "processed_image.jpg"
                cv2.imwrite(processed_path, processed_image)
                print(f"Processed image saved at: {processed_path}")
                print(f"Brand counts: {brand_counts}")

                cv2.imshow("Processed Image", processed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Error during image processing: {str(e)}")

    elif mode == "2":
        video_path = input("Enter the path to the test video: ")
        output_path = "processed_video.mp4"

        try:
            brand_counts = process_brand_video(video_path, output_path)
            print(f"Processed video saved at: {output_path}")
            print(f"Brand counts: {brand_counts}")
        except Exception as e:
            print(f"Error during video processing: {str(e)}")

    else:
        print("Invalid input. Please enter 1 for image or 2 for video.")
