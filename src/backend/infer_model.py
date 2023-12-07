"""Classify images using a trained model."""
import random
from io import BytesIO
from pathlib import Path
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import datasets, transforms
from torchvision.models import densenet201

# pylint: disable=no-member


TRAIN_DIR = Path("/workspaces/AiCoinXpert/algo/webscraping/data/data_for_inference")
MODEL_PATH = Path("/workspaces/AiCoinXpert/train_12_06_2023_15_59.pth")


class ImageClassifier:
    """Classify images using a trained model. Transfoms images to tensors and normalizes them."""

    def __init__(self) -> None:
        self.train_dir = TRAIN_DIR
        self.model_save_path = MODEL_PATH
        self.output_classes_number = 198
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_dataset()
        self.load_model()

    def setup_dataset(self, dataset_dir: str = None) -> None:
        """Setup the dataset used in training to get the classes and the number of classes.

        Args:
            dataset_dir (str, optional): Location path of the dataset.
        """
        if dataset_dir is not None:
            self.train_dir = Path(dataset_dir)
        self.train_dataset = datasets.ImageFolder(self.train_dir)

    def load_model(self, model_save_path: str = None) -> None:
        """Load and prepare the model.

        Args:
            model_save_path (str, optional): Model path.
        """
        if model_save_path is not None:
            self.model_save_path = Path(model_save_path)
        self.model = densenet201()
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(
                in_features=1920, out_features=self.output_classes_number, bias=True
            ),
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(self.model_save_path, map_location=torch.device("cpu"))
        )

    async def predict_image(
        self,
        probability_threshold: float,
        image_data: bytes,
        image_name: str,
        image_size: Tuple[int, int] = (224, 224),
        transform: transforms = None,
    ):
        """Apply tranformations to the image and predict the class of the image.

        Args:
            image_data (bytes): Image in bytes format.
            image_size (Tuple[int, int], optional): Size of the image. Defaults to (224, 224).
            transform (transforms, optional): Transformations to apply to the image.
            probability_threshold (float, optional): Minimum probability
            threshold for the prediction. Defaults to 0.5.

        Returns:
            json: JSON object with the class and the probability of the prediction.
        """

        img = Image.open(BytesIO(image_data))

        if transform is not None:
            image_transform = transform
        else:
            image_transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        self.model.to(self.device)
        self.model.eval()
        with torch.inference_mode():
            transformed_image = image_transform(img).unsqueeze(dim=0)
            target_image_pred = self.model(transformed_image.to(self.device))
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
        # print(target_image_pred_probs.max().item())
        if target_image_pred_probs.max().item() < probability_threshold:
            return {"class": "Unknown", "prob": 0.0}

        result = {
            "class": self.train_dataset.classes[target_image_pred_label],
            "prob": target_image_pred_probs.max().item(),
            "picture_name": image_name,
        }

        print(result)
        return result

    def plot_images(
        self,
        image_path: bytes,
        image_size: Tuple[int, int] = (224, 224),
        transform: transforms = None,
    ):
        """Plot de predictions of the images.

        Args:
            image_path (str): Path of the images to predict.
            image_size (Tuple[int, int], optional): Size of the image. Defaults to (224, 224).
            transform (transforms, optional): Transformations to apply to the image.

        Returns:
            str: Predicted class.
            float: Probability of the prediction.
        """
        prediction = self.predict_image(image_path, image_size, transform)
        if isinstance(prediction, dict):
            pred_label = prediction["class"]
            pred_prob = prediction["prob"]
        else:
            pred_label, pred_prob = prediction
        img = Image.open(BytesIO(image_path))
        plt.imshow(img)
        plt.title(f"Pred: {pred_label} | Prob: {pred_prob:.3f}")
        plt.axis(False)
        plt.show()
        return pred_label, pred_prob

    @classmethod
    def get_random_images(cls, num_images_to_get: int, image_dir: str) -> Callable:
        """Select random images from a directory.

        Args:
            num_images_to_get (int): The number of images to get.
            image_dir (str): Path of the directory to get the images from.

        Returns:
            object: A random selection of images in the directory selected.
        """
        image_dir_path = Path(image_dir)
        unseen = list(image_dir_path.glob("**/*"))
        return random.sample(population=unseen, k=num_images_to_get)
