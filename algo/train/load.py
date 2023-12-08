from typing import Tuple
from torchvision.transforms import transforms
from PIL import Image
import torch
from torchvision.models import densenet201
from pathlib import Path
from torchvision import datasets

TRAIN_DIR = Path("/workspaces/AiCoinXpert/algo/webscraping/data/data_for_inference")

class Model:  # replace with your actual model class
    def __init__(self, model_path: str):
        self.model = densenet201()
        #self.model = torch.load(model_path, map_location=torch.device("cpu"))
        self.train_dir = TRAIN_DIR
        self.model.eval()
        self.setup_dataset()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def setup_dataset(self, dataset_dir: str = None) -> None:
        """Setup the dataset used in training to get the classes and the number of classes.

        Args:
            dataset_dir (str, optional): Location path of the dataset.
        """
        if dataset_dir is not None:
            self.train_dir = Path(dataset_dir)
        self.train_dataset = datasets.ImageFolder(self.train_dir)
        
    def predict(self, image_path: str, apply_transform: bool = False) -> Tuple[str, float]:
        # Load the image
        image = Image.open(image_path)

        # Apply the transformations if needed
        if apply_transform:
            transform = transforms.Compose([
                # Add your transformations here
                transforms.ToTensor()  # Convert the image to a Tensor
            ])
            image = transform(image)

        # Add an extra dimension because the model expects batches
        image = image.unsqueeze(0)

        with torch.inference_mode():
            transformed_image = image.to(self.device)
            target_image_pred = self.model(transformed_image)
            target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
            target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
            # Get predicted label and probability
            predicted_class = self.train_dataset.classes[target_image_pred_label.item()]
            probability = target_image_pred_probs[0, target_image_pred_label].item()
            return predicted_class, probability

model_path = "/workspaces/AiCoinXpert/train_12_06_2023_15_59.pth"
model = Model(model_path)

predicted_class, probability = model.predict("/workspaces/AiCoinXpert/algo/webscraping/data/unseen_coins/Monaco-20-Cent-2013-3032500-155539840594140_augmented_61.jpg", apply_transform=True)