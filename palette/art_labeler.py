#!/usr/bin/env python3

import argparse
import logging
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import csv
import os
import pandas as pd
from enum import Enum


class PhotoType(Enum):
    REGULAR = "regular"
    ARTWORK = "artwork"
    UNKNOWN = "unknown"


class PhotoAngle(Enum):
    STRAIGHT = "straight"
    ANGLED = "angled"
    UNKNOWN = "unknown"


class CroppingQuality(Enum):
    WELL_CROPPED = "well_cropped"
    INCLUDES_SURROUNDINGS = "includes_surroundings"
    UNKNOWN = "unknown"


class ImageLabeler:
    def __init__(self, input_csv, output_csv):
        self.root = tk.Tk()
        self.root.title("Artwork Labeler")

        # Load input CSV data
        self.df = pd.read_csv(input_csv)
        self.current_index = 0
        self.output_csv = output_csv

        # Load or create output CSV
        self.processed_files = set()
        if os.path.exists(output_csv):
            with open(output_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.processed_files.add(row["Path"])
        else:
            # Create output CSV with headers
            self.write_csv_headers()

        # Skip to first unprocessed image
        while (
            self.current_index < len(self.df)
            and self.df.iloc[self.current_index]["Path"] in self.processed_files
        ):
            self.current_index += 1

        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Create left panel for image
        self.canvas = tk.Canvas(self.main_container, bg="gray")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create right panel for controls
        self.controls = ttk.Frame(self.main_container)
        self.controls.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Variables for signature box drawing
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.current_boxes = []

        self.setup_controls()
        self.setup_bindings()
        self.load_current_image()

        # Start the main loop
        self.root.mainloop()

    def write_csv_headers(self):
        headers = [
            "Path",
            "type",
            "material",
            "has_signature",
            "angle",
            "cropping",
            "signature_boxes",
        ]
        with open(self.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    def setup_controls(self):
        # Image type selection
        ttk.Label(self.controls, text="Image Type (1-regular, 2-artwork):").pack(
            anchor="w"
        )
        self.image_type_var = tk.StringVar(value=PhotoType.UNKNOWN.value)
        ttk.Label(self.controls, textvariable=self.image_type_var).pack(anchor="w")

        # Material selection
        ttk.Label(self.controls, text="Material:").pack(anchor="w")
        self.material_var = tk.StringVar()
        materials = ["", "Huile", "Pastel", "Technique mixte", "Lavis", "Acrylique"]
        self.material_combo = ttk.Combobox(
            self.controls, textvariable=self.material_var, values=materials
        )
        self.material_combo.pack(anchor="w")

        # Signature presence
        ttk.Label(self.controls, text="Signature (y/n):").pack(anchor="w")
        self.signature_var = tk.StringVar(value="unknown")
        ttk.Label(self.controls, textvariable=self.signature_var).pack(anchor="w")

        # Photo angle
        ttk.Label(self.controls, text="Photo Angle (3-straight, 4-angled):").pack(
            anchor="w"
        )
        self.angle_var = tk.StringVar(value=PhotoAngle.UNKNOWN.value)
        ttk.Label(self.controls, textvariable=self.angle_var).pack(anchor="w")

        # Cropping quality
        ttk.Label(
            self.controls, text="Cropping (5-well, 6-includes surroundings):"
        ).pack(anchor="w")
        self.cropping_var = tk.StringVar(value=CroppingQuality.UNKNOWN.value)
        ttk.Label(self.controls, textvariable=self.cropping_var).pack(anchor="w")

        # Navigation buttons
        ttk.Button(self.controls, text="Previous (←)", command=self.prev_image).pack(
            pady=5
        )
        ttk.Button(self.controls, text="NEXT (→)", command=self.next_image).pack(pady=5)

        # Progress indicator
        self.progress_var = tk.StringVar()
        self.update_progress_indicator()
        ttk.Label(self.controls, textvariable=self.progress_var).pack(pady=5)

        # Instructions
        instructions = """
        Instructions:
        1/2 - Set image type
        y/n - Toggle signature
        3/4 - Set photo angle
        5/6 - Set cropping quality
        Click and drag - Draw signature box
        r - Reset signature boxes
        ←/→ - Navigate images
        """
        ttk.Label(self.controls, text=instructions).pack(anchor="w", pady=20)

    def update_progress_indicator(self):
        total = len(self.df)
        processed = len(self.processed_files)
        self.progress_var.set(f"Progress: {processed}/{total} images processed")

    def setup_bindings(self):
        self.root.bind("<Key>", self.handle_key)
        self.canvas.bind("<ButtonPress-1>", self.start_rect)
        self.canvas.bind("<B1-Motion>", self.draw_rect)
        self.canvas.bind("<ButtonRelease-1>", self.end_rect)

    def handle_key(self, event):
        key = event.char.lower()
        if key == "1":
            self.image_type_var.set(PhotoType.REGULAR.value)
        elif key == "2":
            self.image_type_var.set(PhotoType.ARTWORK.value)
        elif key == "y":
            self.signature_var.set("yes")
        elif key == "n":
            self.signature_var.set("no")
        elif key == "3":
            self.angle_var.set(PhotoAngle.STRAIGHT.value)
        elif key == "4":
            self.angle_var.set(PhotoAngle.ANGLED.value)
        elif key == "5":
            self.cropping_var.set(CroppingQuality.WELL_CROPPED.value)
        elif key == "6":
            self.cropping_var.set(CroppingQuality.INCLUDES_SURROUNDINGS.value)
        elif key == "r":
            self.reset_signature_boxes()
        elif event.keysym == "Left":
            self.prev_image()
        elif event.keysym == "Right":
            self.next_image()

    def load_current_image(self):
        if 0 <= self.current_index < len(self.df):
            row = self.df.iloc[self.current_index]
            image_path = row["Path"]

            # Check if image was already processed
            if image_path in self.processed_files:
                self.next_image()
                return

            # Load and display image
            try:
                image = Image.open(image_path)
                # Resize image to fit canvas while maintaining aspect ratio
                display_size = (800, 600)
                image.thumbnail(display_size, Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(image)
                self.canvas.config(width=image.width, height=image.height)
                self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

                # Store image dimensions for signature box calculations
                self.image_width = image.width
                self.image_height = image.height

                # Reset current labels
                self.reset_labels()

            except Exception as e:
                logging.error(f"Error loading image {image_path}: {e}")
                self.next_image()

    def reset_labels(self):
        self.image_type_var.set(PhotoType.UNKNOWN.value)
        self.material_var.set("")
        self.signature_var.set("unknown")
        self.angle_var.set(PhotoAngle.UNKNOWN.value)
        self.cropping_var.set(CroppingQuality.UNKNOWN.value)
        self.reset_signature_boxes()

    def save_current_labels(self):
        if 0 <= self.current_index < len(self.df):
            image_path = self.df.iloc[self.current_index]["Path"]

            # Skip if already processed
            if image_path in self.processed_files:
                return

            # Prepare row data
            row_data = {
                "Path": image_path,
                "type": self.image_type_var.get(),
                "material": self.material_var.get(),
                "has_signature": self.signature_var.get(),
                "angle": self.angle_var.get(),
                "cropping": self.cropping_var.get(),
                "signature_boxes": str(self.current_boxes),
            }

            # Append to CSV
            with open(self.output_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row_data.keys())
                writer.writerow(row_data)

            # Mark as processed
            self.processed_files.add(image_path)
            self.update_progress_indicator()

    def start_rect(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

    def draw_rect(self, event):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, x, y, outline="red", width=2
        )

    def end_rect(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        # Store normalized coordinates
        box = {
            "x1": min(self.start_x, x) / self.image_width,
            "y1": min(self.start_y, y) / self.image_height,
            "x2": max(self.start_x, x) / self.image_width,
            "y2": max(self.start_y, y) / self.image_height,
        }
        self.current_boxes.append(box)
        self.rect_id = None

    def reset_signature_boxes(self):
        self.current_boxes = []
        self.canvas.delete("signature_box")

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

    def next_image(self):
        if self.current_index < len(self.df) - 1:
            # Save current labels before moving to next image
            self.save_current_labels()

            # Move to next unprocessed image
            self.current_index += 1
            while (
                self.current_index < len(self.df)
                and self.df.iloc[self.current_index]["Path"] in self.processed_files
            ):
                self.current_index += 1

            self.load_current_image()


def setup_logging(log_level=logging.INFO, log_file=None):
    """Configure logging with optional file output"""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)


def main():
    parser = argparse.ArgumentParser(description="Catalog art images in a directory")
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to input CSV file with image data",
        default="art_catalog.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="art_catalog_labels.csv",
        help="Output CSV file path (default: art_catalog_labels.csv)",
    )
    parser.add_argument("-l", "--log", type=str, help="Log file path (optional)")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level=log_level, log_file=args.log)

    labeler = ImageLabeler(args.csv, args.output)


if __name__ == "__main__":
    main()
