import os
from dataclasses import dataclass
from typing import List, Tuple
from warnings import filterwarnings

import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
from torchvision import transforms

from models_name import Device, ModelName
from train import CustomDataset, Model

filterwarnings("ignore")


@dataclass
class Predict:
    """
    A class for making predictions on images using a trained PyTorch model.
    """

    model: torch.nn.Module
    class_names: List[str]
    image_size: Tuple[int, int] = (224, 224)
    transform: torchvision.transforms = None
    device: torch.device = Device.CPU.value

    def __post_init__(self):
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def predict_image(
        self, image_path: str, apply_transform: bool = False
    ) -> Tuple[str, float]:
        """
        Predicts the class of an image and returns the predicted class and its probability.

        Args:
            image_path (str): The path to the image file.
            apply_transform (bool): Whether to apply the transformation to the image.

        Returns:
            Tuple[str, float]: The predicted class and its probability.
        """
        # Open image
        img = Image.open(image_path)

        # Transform image and send it to the target device
        with torch.no_grad():
            if apply_transform:
                transformed_image = self.transform(img).unsqueeze(dim=0).to(self.device)
            else:
                transformed_image = (
                    transforms.ToTensor()(img).unsqueeze(dim=0).to(self.device)
                )

            # Make a prediction on the image
            target_image_pred = self.model(transformed_image)

        # Get predicted label and probability
        predicted_class = self.class_names[target_image_pred.argmax(dim=1)]
        probability = torch.softmax(target_image_pred, dim=1).max().item()

        return predicted_class, probability


def load_model_and_predict(
    image_path: str, model_path: str, apply_transform: bool = False
) -> Tuple[str, float]:
    """
    Load the trained model and predict the class of an image.

    Args:
        image_path (str): The path to the image file.
        model_path (str): The path to the saved model state dictionary.
        apply_transform (bool): Whether to apply the transformation to the image.

    Returns:
        Tuple[str, float]: The predicted class and its probability.
    """
    # Define the class names for the model
    dataset = CustomDataset(
        train_dir="/home/abermond/Desktop/workspaces/AICoinXpert/algo/webscraping/data/selected_80_percent_filtered/train/",
        test_dir="/home/abermond/Desktop/workspaces/AICoinXpert/algo/webscraping/data/selected_80_percent_filtered/test/",
        val_dir="/home/abermond/Desktop/workspaces/AICoinXpert/algo/webscraping/data/selected_80_percent_filtered/eval/",
        batch_size=250,
    )
    class_names = dataset.class_names

    # Load the trained model
    model = Model(ModelName.DENSENET201.value, dataset)
    model.model.load_state_dict(torch.load(model_path))

    # Create a Predict object
    predictor = Predict(model=model.model, class_names=class_names)

    # Predict the image
    predicted_class, probability = predictor.predict_image(
        image_path, apply_transform=apply_transform
    )
    return predicted_class, probability


def plot_image(self, image_path: str, apply_transform: bool = False):
    """
    Plots the image along with its predicted class and probability.

    Args:
        image_path (str): The path to the image file.
        apply_transform (bool): Whether to apply the transformation to the image.
    """
    # Get the predicted class and probability
    predicted_class, probability = self.predict_image(
        image_path, apply_transform=apply_transform
    )

    # Open the image
    img = Image.open(image_path)

    # Plot the image
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} | Probability: {probability:.3f}")
    plt.axis("off")
    plt.show()


def main(folder_path: str, apply_transform: bool = False, plot_images: bool = False):
    """
    The main function for making predictions on images using a trained PyTorch model.

    Args:
        folder_path (str): The path to the folder containing the image files.
        apply_transform (bool): Whether to apply the transformation to the images.
    """
    # Define the path to the saved model state dictionary
    model_path = "/home/abermond/Desktop/workspaces/AICoinXpert/best_model_2023-08-02_19-27-40.pth"

    # Loop through all the images in the folder
    for filename in os.listdir(folder_path):
        if (
            filename.endswith(".jpg")
            or filename.endswith(".jpeg")
            or filename.endswith(".png")
            or filename.endswith(".webp")
        ):
            # Get the prediction
            image_path = os.path.join(folder_path, filename)

            # Check if the file exists before attempting to open it
            if not os.path.exists(image_path):
                print(f"File not found: {image_path}")
                continue

            # Predict the image
            predicted_class, probability = load_model_and_predict(
                image_path, model_path, apply_transform=apply_transform
            )

            # Print the prediction in the log
            filename = os.path.basename(image_path)
            print(
                f"Image: {filename} | Pred: {predicted_class} | Prob: {probability:.3f}"
            )

            # Plot the image along with its predicted class and probability if requested
            if plot_images:
                img = Image.open(image_path)
                plt.imshow(img)
                plt.title(
                    f"Predicted: {predicted_class} | Probability: {probability:.3f}"
                )
                plt.axis("off")
                plt.show()
                print(
                    f"Image: {filename} | Pred: {predicted_class} | Prob: {probability:.3f}"
                )


if __name__ == "__main__":
    FOLDER_PATH = "/home/abermond/Desktop/unseen_coins/"
    main(FOLDER_PATH, apply_transform=False, plot_images=False)
