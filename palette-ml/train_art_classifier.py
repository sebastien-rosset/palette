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
import torchvision.ops as ops


def custom_collate(batch):
    images = torch.stack([item[0] for item in batch])

    # Handle labels dictionary
    labels = {}
    for key in batch[0][1].keys():
        if key == "signature_boxes":
            # For signature boxes, we'll pad to max number of boxes in the batch
            max_boxes = max(item[1]["signature_boxes"].size(0) for item in batch)
            if max_boxes == 0:
                # If no boxes in batch, create empty tensor
                labels["signature_boxes"] = torch.zeros(
                    (len(batch), 0, 4), dtype=torch.float32
                )
            else:
                # Pad all tensors to max_boxes
                padded_boxes = []
                for item in batch:
                    boxes = item[1]["signature_boxes"]
                    if boxes.size(0) < max_boxes:
                        # Pad with zeros
                        padding = torch.zeros(
                            (max_boxes - boxes.size(0), 4), dtype=torch.float32
                        )
                        boxes = torch.cat([boxes, padding], dim=0)
                    padded_boxes.append(boxes)
                labels["signature_boxes"] = torch.stack(padded_boxes)
        else:
            # For other labels, just stack them
            labels[key] = torch.tensor([item[1][key] for item in batch])

    return images, labels


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
        original_size = image.size  # Store original size for bbox scaling

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

        # Handle signature bounding boxes
        if "signature_boxes" in row and row["signature_boxes"] != "[]":
            try:
                # Parse bounding boxes from string format
                boxes = ast.literal_eval(row["signature_boxes"])
                if boxes and isinstance(boxes, list):
                    # Scale boxes to normalized coordinates (0-1)
                    scaled_boxes = []
                    for box in boxes:
                        # Handle dictionary format of boxes
                        x1 = float(box["x1"])
                        y1 = float(box["y1"])
                        x2 = float(box["x2"])
                        y2 = float(box["y2"])
                        scaled_boxes.append([x1, y1, x2, y2])  # Already normalized
                    labels["signature_boxes"] = torch.tensor(
                        scaled_boxes, dtype=torch.float32
                    )
                    labels["num_signatures"] = len(scaled_boxes)
                else:
                    labels["signature_boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                    labels["num_signatures"] = 0
            except (ValueError, SyntaxError, KeyError) as e:
                print(f"Error parsing signature boxes for {image_path}: {e}")
                labels["signature_boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                labels["num_signatures"] = 0
        else:
            labels["signature_boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            labels["num_signatures"] = 0

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

        # Add signature detection head
        self.signature_detector = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4),  # x1, y1, x2, y2
            nn.Sigmoid(),  # Normalize coordinates to [0, 1]
        )

    def forward(self, x):
        features = self.backbone(x)
        outputs = {task: head(features) for task, head in self.heads.items()}

        # Add signature detection output
        outputs["signature_boxes"] = self.signature_detector(features)

        return outputs


def compute_box_loss(pred_boxes, true_boxes):
    """Compute loss for bounding box prediction"""
    if true_boxes.size(1) == 0:  # No boxes in batch
        return torch.tensor(0.0).to(pred_boxes.device)

    # Compute loss only for valid boxes (non-zero)
    valid_mask = (true_boxes.sum(dim=-1) != 0).float()  # [batch_size, max_boxes]

    # Reshape predictions to match true boxes shape
    pred_boxes = pred_boxes.unsqueeze(1).expand(-1, true_boxes.size(1), -1)

    # Compute GIoU loss
    loss = ops.generalized_box_iou_loss(
        pred_boxes.view(-1, 4), true_boxes.view(-1, 4), reduction="none"
    )

    # Apply valid mask and average
    loss = (loss.view(valid_mask.shape) * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    return loss


def train_model(train_loader, val_loader, model, device, num_epochs=15):
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

            # Compute classification losses
            class_loss = sum(
                criterion(outputs[k], batch_labels[k])
                for k in ["image_type", "has_signature", "angle", "cropping"]
            )

            # Compute box loss only for images with signatures
            box_loss = compute_box_loss(
                outputs["signature_boxes"], batch_labels["signature_boxes"]
            )

            # Combined loss
            loss = class_loss + box_loss
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

                # Compute losses
                class_loss = sum(
                    criterion(outputs[k], batch_labels[k])
                    for k in ["image_type", "has_signature", "angle", "cropping"]
                )
                box_loss = compute_box_loss(
                    outputs["signature_boxes"], batch_labels["signature_boxes"]
                )
                loss = class_loss + box_loss
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
        "--epochs", type=int, default=15, help="Number of epochs to train"
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, collate_fn=custom_collate
    )

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
