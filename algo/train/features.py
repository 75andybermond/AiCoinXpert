import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_features(model, data_loader):
    feature_vectors = []
    labels = []
    with torch.no_grad(), tqdm(total=len(data_loader)) as progress_bar:
        for images, targets in data_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            features = model(images)
            feature_vectors.append(features.cpu().numpy())
            labels.append(targets.cpu().numpy())
            progress_bar.update(1)
    feature_vectors = np.concatenate(feature_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)
    return feature_vectors, labels


def main():
    # Load the datasets
    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset_dir = "/home/abermond/Desktop/workspaces/AICoinXpert/algo/webscraping/data/selected_80_percent_filtered/train/"
    test_dataset_dir = "/home/abermond/Desktop/workspaces/AICoinXpert/algo/webscraping/data/selected_80_percent_filtered/test/"
    eval_dataset_dir = "/home/abermond/Desktop/workspaces/AICoinXpert/algo/webscraping/data/selected_80_percent_filtered/eval/"

    train_dataset = ImageFolder(root=train_dataset_dir, transform=data_transform)
    test_dataset = ImageFolder(root=test_dataset_dir, transform=data_transform)
    eval_dataset = ImageFolder(root=eval_dataset_dir, transform=data_transform)

    # Define the model and feature extraction layer
    model_name = "densenet201"  # Change this to the model of your choice
    model = models.__dict__[model_name](pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(DEVICE)
    feature_extractor.eval()

    # Extract features for each dataset
    train_features, train_labels = extract_features(
        feature_extractor, DataLoader(train_dataset, batch_size=12, shuffle=True)
    )
    test_features, test_labels = extract_features(
        feature_extractor, DataLoader(test_dataset, batch_size=12, shuffle=True)
    )
    eval_features, eval_labels = extract_features(
        feature_extractor, DataLoader(eval_dataset, batch_size=12, shuffle=False)
    )

    # Save the features to disk
    np.save("train_features.npy", train_features)
    np.save("train_targets.npy", train_labels)
    np.save("test_features.npy", test_features)
    np.save("test_targets.npy", test_labels)
    np.save("eval_features.npy", eval_features)
    np.save("eval_targets.npy", eval_labels)

    # Train the classifier
    num_epochs = 32
    classifier = train_classifier(train_features, train_labels, num_epochs=num_epochs)

    # Evaluate the classifier
    train_accuracy = evaluate_classifier(classifier, train_features, train_labels)
    test_accuracy = evaluate_classifier(classifier, test_features, test_labels)
    eval_accuracy = evaluate_classifier(classifier, eval_features, eval_labels)

    print(f"Train accuracy: {train_accuracy:.2f}")
    print(f"Test accuracy: {test_accuracy:.2f}")
    print(f"Evaluation accuracy: {eval_accuracy:.2f}")


if __name__ == "__main__":
    main()
