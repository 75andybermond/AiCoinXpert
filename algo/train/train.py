from __future__ import division, print_function

import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Define the ModelName enum
class ModelName:
    """ModelName enum class that contains the available pre-trained models by their names."""

    DENSENET201 = "densenet201"
    RESNET50 = "resnet50"
    VGG16 = "vgg16"


@dataclass
class CustomDataset:
    """Initializes the CustomDataset object that will contain the train, test, and val datasets.

    Args:
        train_dir (str): Train dataset directory
        test_dir (str): Test dataset directory
        val_dir (str): Validation dataset directory
        batch_size (int): Batch size is the number of training examples utilized in one iteration
        shuffle (bool): Whether to shuffle the data or not. Default is True.
    """

    train_dir: str
    test_dir: str
    val_dir: str
    batch_size: int
    shuffle: bool = True
    train_dataloader: Optional[DataLoader] = None
    test_dataloader: Optional[DataLoader] = None
    val_dataloader: Optional[DataLoader] = None

    def __post_init__(self):
        # Define manual transformations
        self.manual_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        # Load the dataset
        self._create_folder()
        self._load_dataset(self.shuffle)

    def _create_folder(self):
        """
        Load the data and apply the manual transformations
        """
        self.train_dataset = ImageFolder(
            root=self.train_dir, transform=self.manual_transforms
        )
        self.test_dataset = ImageFolder(
            root=self.test_dir, transform=self.manual_transforms
        )
        self.val_dataset = ImageFolder(
            root=self.val_dir, transform=self.manual_transforms
        )
        self.class_names = self.train_dataset.classes
        self.len_class_names = len(self.class_names)

    def _load_dataset(self, shuffle):
        """
        Load the data into train, test, and validation dataloaders

        Args:
            shuffle (bool): Apply shuffling to the data or not
        """
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=shuffle
        )
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )


class Model:
    """Model class that will contain the pre-trained model, loss function, and optimizer."""

    AVAILABLE_MODELS = [ModelName.DENSENET201, ModelName.RESNET50, ModelName.VGG16]

    def __init__(self, model_name: ModelName, custom_dataset: CustomDataset):
        """Initializes the Model name choosen and the training dataset.

        Args:
            model_name (ModelName): Model name to be used for training
            custom_dataset (CustomDataset): CustomDataset object that contains the train/test/val
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model name '{model_name}'. Available models: {', '.join(self.AVAILABLE_MODELS)}"
            )

        # Load the specified pre-trained model
        model_dict = {
            ModelName.DENSENET201: models.densenet201(pretrained=True),
            ModelName.RESNET50: models.resnet50(pretrained=True),
            ModelName.VGG16: models.vgg16(pretrained=True),
        }

        if model_name not in model_dict:
            raise ValueError(
                f"Invalid model name '{model_name}'. Available models: {', '.join(model_dict.keys())}"
            )

        # Get the pre-trained model
        model = model_dict[model_name]

        # Freeze model parameters
        for param in model.parameters():
            param.requires_grad = False

        # Set the manual seeds
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # Recreate the classifier layer and seed it to the target device
        classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(
                in_features=model.classifier.in_features,
                out_features=custom_dataset.len_class_names,  # same number of output units as our number of classes
                bias=True,
            ),
        ).to(DEVICE)

        # Replace the classifier in the model
        model.classifier = classifier

        # Move the model to the target device
        model.to(DEVICE)

        # Define Optimizer and Loss Function
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Move the loss function to the target device
        loss_func.to(DEVICE)

        # Store the model and related components as attributes of the class
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.custom_dataset = custom_dataset


class Train:
    """Train class that will contain the model, train dataloader, epochs, project name, and tag."""

    def __init__(
        self,
        model: Model,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        epochs: int,
        project: str,
        tag: str,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.epochs = epochs
        self.project = project
        self.tag = tag

    @staticmethod
    def calculate_precision(predicted, targets) -> float:
        """Calculate the precision of the model.

        Args:
            predicted (Tensor): Predicted values
            targets (Tensor): Target values

        Returns:
            float: Precision of the model
        """
        true_positives = torch.sum((predicted == 1) & (targets == 1)).item()
        false_positives = torch.sum((predicted == 1) & (targets == 0)).item()

        if (true_positives + false_positives) == 0:
            return 0.0

        precision = true_positives / (true_positives + false_positives)
        return precision

    def evaluate(self, model, dataloader, loss_func, device):
        """Evaluate the model.

        Args:
            model (Model): Model to be evaluated
            dataloader (DataLoader): DataLoader to be used for evaluation
            loss_func (LossFunction): Loss function to be used for evaluation
            device (str): Device to be used for evaluation

        Returns:
            average_loss (float): Average loss of the model
            accuracy (float): Accuracy of the model
        """
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = loss_func(outputs, targets)

                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)

        average_loss = total_loss / len(dataloader)
        accuracy = 100.0 * total_correct / total_samples

        return average_loss, accuracy

    def train_model(self, patience=2):
        """Train the model.

        Args:
            patience (int, optional): Number of epochs to wait if the loss is not improving
        """
        # Initialize Wandb run
        wandb.init(
            project=self.project,
            tags=[self.tag],
            config={
                "epochs": self.epochs,
                "batch_size": self.train_dataloader.batch_size,
                "lr": 0.001,
                "dropout": 0.2,
            },
        )

        start_time = time.time()
        best_loss = float("inf")
        counter = 0

        for epoch in range(self.epochs):
            # Training code here
            self.model.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_loss_total = 0.0

            for batch_idx, (train_inputs, train_targets) in enumerate(
                self.train_dataloader
            ):
                train_inputs = train_inputs.to(DEVICE)
                train_targets = train_targets.to(DEVICE)

                self.model.optimizer.zero_grad()

                train_outputs = self.model.model(train_inputs)
                train_loss = self.model.loss_func(train_outputs, train_targets)
                train_loss.backward()
                self.model.optimizer.step()

                train_loss_total += train_loss.item()

                _, train_predicted = train_outputs.max(1)
                train_total += train_targets.size(0)
                train_correct += train_predicted.eq(train_targets).sum().item()

            train_accuracy = 100.0 * train_correct / train_total
            train_loss = train_loss_total / len(self.train_dataloader)

            train_precision = self.calculate_precision(train_predicted, train_targets)

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "train_precision": train_precision,
                }
            )

            # Test code here
            test_loss, test_accuracy = self.evaluate(
                self.model.model,
                self.test_dataloader,
                self.model.loss_func,
                DEVICE,
            )

            eval_loss, eval_accuracy = self.evaluate(
                self.model.model, self.test_dataloader, self.model.loss_func, DEVICE
            )

            wandb.log(
                {
                    "epochs": epoch,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "eval_loss": eval_loss,
                    "eval_accuracy": eval_accuracy,
                },
            )

            self.log_confusion_matrix(self.model.model, self.test_dataloader)

            if test_loss < best_loss:
                best_loss = test_loss
                counter = 0
                # Save the best model
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                torch.save(self.model.model.state_dict(), f"best_model_{timestamp}.pth")
            else:
                counter += 1
                if counter >= patience:
                    break

        # End the timer and print out how long it took
        end_time = time.time()
        print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")

        # Finish wandb run
        wandb.finish()

    def log_confusion_matrix(self, model, dataloader):
        """Calculate and log the confusion matrix.

        Args:
            model (Model): Model to be evaluated
            dataloader (DataLoader): DataLoader to be used for evaluation
        """
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                outputs = model(inputs)

                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        cm = confusion_matrix(all_targets, all_predictions)
        class_names = (
            self.model.custom_dataset.class_names
        )  # Access the class names from the CustomDataset through the Model
        wandb.log(
            {
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_targets,
                    preds=all_predictions,
                    class_names=class_names,
                )
            }
        )


def main():
    # Initialize the CustomDataset object
    data = CustomDataset(
        train_dir="/home/abermond/Desktop/workspaces/AICoinXpert/algo/webscraping/data/selected_80_percent_filtered/train/",
        test_dir="/home/abermond/Desktop/workspaces/AICoinXpert/algo/webscraping/data/selected_80_percent_filtered/test/",
        val_dir="/home/abermond/Desktop/workspaces/AICoinXpert/algo/webscraping/data/selected_80_percent_filtered/eval/",
        batch_size=250,
    )
    # Instantiate the model with the dataset prepared
    model = Model(ModelName.RESNET50, data)
    # Train the model
    training_step = Train(
        model,
        data.train_dataloader,
        data.test_dataloader,
        epochs=10,
        project="resnet50",
        tag="pool_split_80_percent_filtered",
    )
    training_step.train_model()


if __name__ == "__main__":
    main()
