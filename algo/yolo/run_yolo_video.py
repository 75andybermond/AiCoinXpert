"""Module to run YOLO object detection on a video stream using OpenCV"""
import argparse
import threading
import tkinter as tk

import cv2
from ultralytics import YOLO

# pylint: disable=no-member


class YOLODetector:
    """Class to run YOLO object detection on a video stream using OpenCV"""

    def __init__(self, camera_index: int, model_path: str, threshold: float):
        """Initializes the YOLODetector object

        Args:
            camera_index (int): Webcam index to use for video capture
            model_path (str): Path to the YOLO model saved in a .pt file
            threshold (float): Detection threshold set the sensitivity of the detector
        """
        self.model = YOLO(model_path)

        # Set the detection threshold
        self.threshold = threshold

        # Open the webcam
        self.cap = cv2.VideoCapture(camera_index)

        # Create a tkinter root window
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Flag to indicate when to stop the detection thread
        self.is_running = False

        # Start the detection thread
        self.detector_thread = threading.Thread(target=self.detect_objects)
        self.detector_thread.daemon = True
        self.detector_thread.start()

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
                self.draw_objects(frame, results)

                # Display the frame with bounding boxes and class labels using OpenCV
                cv2.imshow("YOLO Object Detection", frame)

                # Handle keyboard events
                self.handle_events()

        except cv2.error as error_cv2:
            print(f"OpenCV error: {error_cv2}")
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Stopping YOLO object detection.")

    def draw_objects(self, frame, results):
        """Function to draw bounding boxes and class labels on the detected objects

        Args:
            frame (numpy.ndarray): Frame from the video stream
            results (YOLO): YOLO object detection results
        """
        for result in results.boxes.data.tolist():
            x1, y2, x2, y1, score, class_id = result

            if score > self.threshold:
                cv2.rectangle(
                    frame, (int(x1), int(y2)), (int(x2), int(y1)), (0, 255, 0), 4
                )
                class_name = results.names[int(class_id)].upper()
                text = f"{class_name} ({int(x1)}, {int(y2)})"
                cv2.putText(
                    frame,
                    text,
                    # results.names[int(class_id)].upper(),
                    (int(x1), int(y2 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA,
                )

    def handle_events(self):
        """Function to handle keyboard events"""
        key = cv2.waitKeyEx(1)
        if key == ord("q"):
            self.on_close()

    def on_close(self):
        """Function to stop the detection thread and close the tkinter window"""
        self.is_running = False
        self.root.quit()

    def start(self):
        """Function to start the YOLO object detection"""
        # Start the tkinter event loop
        self.root.mainloop()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument(
        "--model",
        type=str,
        default="/workspaces/AICoinXpert/runs/detect/train38/weights/best.pt",
        help="path to YOLO model",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.1, help="detection threshold"
    )
    args = parser.parse_args()

    # Create the YOLO detector
    detector = YOLODetector(args.camera, args.model, args.threshold)

    # Start the YOLO object detection
    try:
        detector.start()
    finally:
        # Release the OpenCV video capture object and destroy all OpenCV windows
        detector.cap.release()
        cv2.destroyAllWindows()
