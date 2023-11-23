"""Generate fake camera for codespaces testing."""
import cv2


class FakeCamera:
    """Fake camera class. It reads a single image and returns it as a frame."""

    def __init__(
        self,
        image_path="/workspaces/AiCoinXpert/src/backend/camera/istockphoto-641624656-612x612.jpg",
    ):
        self.frame = cv2.imread(image_path)
        self.running = True

    def read(self):
        """Read the frame and return it."""
        if self.running:
            return True, self.frame
        return False, None

    def release(self):
        """Release the FakeCamera object."""
        self.running = False
