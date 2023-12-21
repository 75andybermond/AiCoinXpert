"""Run the Yolo models with webcam setup and object drawing and saving"""
import argparse
import datetime
import logging
import time
import tkinter as tk

import cv2
import numpy.typing as npt
from ultralytics import YOLO

# pylint: disable=no-member
# pylint: disable=invalid-name
MODEL_PATH = "/workspaces/AiCoinXpert/src/backend/video/best.pt"

# Configure the logging settings with the custom formatter
logging.basicConfig(
    filename="/workspaces/AiCoinXpert/src/backend/video/tmp/detection_log.txt",
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y%m%d_%H%M%S",  # Define log format
)


class YOLODetector:
    """Yolo object detector."""

    def __init__(self, camera_index: int, model_path: str, threshold: float):
        self.model = YOLO(model_path)
        self.threshold = threshold
        self.cap = cv2.VideoCapture(camera_index)
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.is_running = False
        self.start_time = time.time()
        self.frame_count = 0

    def detect_objects(self):
        """Function to detect objects in a video stream using YOLO model"""
        try:
            self.is_running = True
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    self.on_close()
                    return

                # Perform object detection on the frame using the YOLO model
                results = self.model(frame)[0]

                # Draw bounding boxes and class labels on the frame
                self.draw_objects(frame, results, confidence_threshold=65)

                # Display the frame with bounding boxes and class labels using OpenCV
                cv2.imshow("YOLO Object Detection", frame)

                # Handle keyboard events
                self.handle_events()

        except cv2.error as error_cv2:
            print(f"OpenCV error: {error_cv2}")
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Stopping YOLO object detection.")

    def draw_objects(
        self, frame: npt.ArrayLike, results: npt.ArrayLike, confidence_threshold: int
    ) -> None:
        """
        Draw bounding boxes and save cropped images of detected objects.

        Args:
            frame (npt.ArrayLike): The input image frame.
            results (npt.ArrayLike): The object detection results.
            confidence_threshold (int): The confidence threshold for object detection.

        This function iterates through the detection results, extracts relevant information
        about the detected objects (class name, confidence score, bounding box coordinates),
        and performs the following actions for objects with confidence scores above the
        specified threshold:
        - Saves cropped images of the detected objects.
        - Draws bounding boxes around the detected objects on the input frame.
        - Logs information about the detected objects.

        The saved images are named with a timestamp and stored in a specific directory.

        Example:
            draw_objects(frame, results, 80)  # Draw objects with confidence > 80%
        """
        for r in results:
            class_name = int(r.boxes.cls)
            class_name = results.names[class_name]

            confidence = r.boxes.conf
            confidence = int(confidence * 100)

            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                if confidence > confidence_threshold:
                    cropped_image = frame[y1:y2, x1:x2]
                    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_filename = f"/workspaces/AiCoinXpert/src/backend/video/tmp/images/{current_time},{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}.jpg"
                    cv2.imwrite(
                        image_filename, cropped_image, [cv2.IMWRITE_JPEG_QUALITY, 100]
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    logging.info(
                        f"Class: {class_name}, Degree Of Certainty: {confidence}, Region Of Interest:[x1: {x1:.3f},y1: {y1:.3f}, x2: {x2:.3f}, y2: {y2:.3f}]"
                    )

    def extract_frames(self):
        """Continuously extract frames from a video capture and perform object detection.

        Yields:
            bytes: Frames encoded in bytes
        """
        try:
            self.is_running = True
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    self.on_close()
                    return

                # Process your frame (e.g., object detection)
                results = self.model(frame)[0]
                self.draw_objects(frame, results, confidence_threshold=65)

                # Resize the frame
                frame = cv2.resize(frame, (0, 0), fx=2.5, fy=2.5)

                # Encode the frame to JPG format and yield it
                success, encoded_frame = cv2.imencode(".jpg", frame)
                if success:
                    yield encoded_frame.tobytes()

        except cv2.error as error_cv2:
            print(f"OpenCV error: {error_cv2}")
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Stopping YOLO object detection.")

    def handle_events(self):
        """Perform stopping webcam by pressing button "q"."""
        key = cv2.waitKeyEx(1)
        if key == ord("q"):
            self.on_close()

    def on_close(self):
        """Close the loop running."""
        self.is_running = False
        self.root.quit()

    def start(self):
        """Start the video frame."""
        self.root.mainloop()


# if __name__ == "__main__":
# # parser = argparse.ArgumentParser()
# # parser.add_argument("--camera", type=int, default=2, help="camera index")
# # parser.add_argument(
# #     "--model",
# #     type=str,
# #     default=MODEL_PATH,
# #     help="path to YOLO model",
# # )
# # parser.add_argument(
# #     "--threshold", type=float, default=1.0, help="detection threshold"
# # )
# # args = parser.parse_args()

# # detector = YOLODetector(args.camera, args.model, args.threshold)
# # try:
# #     detector.detect_objects()
# # except KeyboardInterrupt:
# #     print("Keyboard interrupt detected. Stopping YOLO object detection.")

# # Create the YOLO detector
# # detector = YOLODetector(args.camera, args.model, args.threshold)
#     yolo = YOLODetector(2, MODEL_PATH, 1.0)
#     yolo.detect_objects()
