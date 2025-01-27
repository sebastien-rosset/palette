#!/usr/bin/env python3

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from sklearn.model_selection import train_test_split
import ast


class ArtDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["Path"]

        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Prepare labels
        labels = {
            "image_type": (
                1 if row["type"] == "artwork" else 0 if row["type"] == "regular" else -1
            ),
            "has_signature": (
                1
                if row["has_signature"] == "yes"
                else 0 if row["has_signature"] == "no" else -1
            ),
            "angle": (
                1
                if row["angle"] == "straight"
                else 0 if row["angle"] == "angled" else -1
            ),
            "cropping": (
                1
                if row["cropping"] == "well_cropped"
                else 0 if row["cropping"] == "includes_surroundings" else -1
            ),
        }

        return image, labels


class ArtClassifier(nn.Module):
    def __init__(self, num_classes_dict):
        super(ArtClassifier, self).__init__()

        # Load pre-trained ResNet model
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Replace final layer with multiple heads
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Create separate heads for each task
        self.heads = nn.ModuleDict(
            {
                "image_type": nn.Linear(feature_dim, num_classes_dict["image_type"]),
                "has_signature": nn.Linear(
                    feature_dim, num_classes_dict["has_signature"]
                ),
                "angle": nn.Linear(feature_dim, num_classes_dict["angle"]),
                "cropping": nn.Linear(feature_dim, num_classes_dict["cropping"]),
            }
        )

    def forward(self, x):
        features = self.backbone(x)
        return {task: head(features) for task, head in self.heads.items()}


def train_model(train_loader, val_loader, model, device, num_epochs=10):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            batch_labels = {k: v.to(device) for k, v in labels.items()}

            optimizer.zero_grad()
            outputs = model(images)

            loss = sum(criterion(outputs[k], batch_labels[k]) for k in outputs.keys())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                batch_labels = {k: v.to(device) for k, v in labels.items()}

                outputs = model(images)
                loss = sum(
                    criterion(outputs[k], batch_labels[k]) for k in outputs.keys()
                )
                val_loss += loss.item()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        logging.info(f"Epoch {epoch+1}/{num_epochs}:")
        logging.info(f"Training Loss: {train_loss/len(train_loader):.4f}")
        logging.info(f"Validation Loss: {val_loss/len(val_loader):.4f}")

    return best_model_state


def main():
    parser = argparse.ArgumentParser(description="Train art classifier model")
    parser.add_argument(
        "--labels",
        type=str,
        default="art_catalog_labels.csv",
        help="Path to labeled data CSV file",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default="art_classifier_model.pth",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train"
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load labeled data
    df = pd.read_csv(args.labels)

    # Split data into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Set up data transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    train_dataset = ArtDataset(train_df, transform=transform)
    val_dataset = ArtDataset(val_df, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Set up model
    num_classes_dict = {"image_type": 2, "has_signature": 2, "angle": 2, "cropping": 2}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ArtClassifier(num_classes_dict).to(device)

    # Train model
    logging.info("Starting training...")
    best_model_state = train_model(train_loader, val_loader, model, device, args.epochs)

    # Save model
    torch.save(
        {"model_state_dict": best_model_state, "num_classes_dict": num_classes_dict},
        args.model_output,
    )
    logging.info(f"Model saved to {args.model_output}")


if __name__ == "__main__":
    main()
