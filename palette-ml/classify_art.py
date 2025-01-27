#!/usr/bin/env python3

import argparse
import logging
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
import csv
from train_art_classifier import ArtClassifier
from pathlib import Path


def load_model(model_path, device):
    # Load saved model state
    checkpoint = torch.load(model_path, map_location=device)

    # Create model and load state
    model = ArtClassifier(checkpoint["num_classes_dict"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def predict_image(model, image_path, transform, device):
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(image_tensor)
        predictions = {
            task: torch.softmax(output, dim=1).cpu().numpy()[0]
            for task, output in outputs.items()
        }

    # Convert predictions to labels
    labels = {
        "type": "artwork" if predictions["image_type"][1] > 0.5 else "regular",
        "has_signature": "yes" if predictions["has_signature"][1] > 0.5 else "no",
        "angle": "straight" if predictions["angle"][1] > 0.5 else "angled",
        "cropping": (
            "well_cropped"
            if predictions["cropping"][1] > 0.5
            else "includes_surroundings"
        ),
    }

    # Add confidence scores
    confidences = {
        "type": float(max(predictions["image_type"])),
        "has_signature": float(max(predictions["has_signature"])),
        "angle": float(max(predictions["angle"])),
        "cropping": float(max(predictions["cropping"])),
    }

    return labels, confidences


def main():
    parser = argparse.ArgumentParser(
        description="Classify art images using trained model"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="art_catalog.csv",
        help="Path to input CSV file with image paths",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="art_classifier_model.pth",
        help="Path to trained model file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="art_catalog_predictions.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        help="Minimum confidence threshold for predictions",
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.model, device)

    # Set up image transformation
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load input CSV
    df = pd.read_csv(args.input_csv)

    # Prepare output CSV
    fieldnames = [
        "Path",
        "type",
        "material",
        "has_signature",
        "angle",
        "cropping",
        "signature_boxes",
        "confidence",
    ]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Process each image
        for idx, row in df.iterrows():
            image_path = row["Path"]
            logging.info(f"Processing image {idx+1}/{len(df)}: {image_path}")

            try:
                # Get predictions
                labels, confidences = predict_image(
                    model, image_path, transform, device
                )

                # Calculate overall confidence
                overall_confidence = sum(confidences.values()) / len(confidences)

                # Only write predictions if confidence is above threshold
                if overall_confidence >= args.confidence_threshold:
                    writer.writerow(
                        {
                            "Path": image_path,
                            "type": labels["type"],
                            "material": "",  # Material classification not implemented yet
                            "has_signature": labels["has_signature"],
                            "angle": labels["angle"],
                            "cropping": labels["cropping"],
                            "signature_boxes": "[]",  # Signature box detection not implemented yet
                            "confidence": overall_confidence,
                        }
                    )

            except Exception as e:
                logging.error(f"Error processing {image_path}: {str(e)}")
                continue

    logging.info(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
