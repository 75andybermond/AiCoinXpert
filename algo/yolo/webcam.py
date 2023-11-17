import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Is possible that you need to install:

# sudo apt-get update
# sudo apt-get install libglib2.0-dev


class Webcam:
    def __init__(self, camera_index=2):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        self.fig, self.ax = plt.subplots()
        self.vid = None

    def update_frame(self, *args):
        ret, frame = self.cap.read()
        if ret:
            if self.vid is None:
                self.vid = self.ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                self.vid.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def start(self):
        ani = FuncAnimation(self.fig, self.update_frame, interval=50)
        plt.show()

        self.cap.release()


if __name__ == "__main__":
    webcam = Webcam(camera_index=2)
    webcam.start()
