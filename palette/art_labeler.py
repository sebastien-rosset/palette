#!/usr/bin/env python3

import argparse
import logging
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import csv
import json
import os
import pandas as pd
from enum import Enum, auto


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
    def __init__(self, csv_path):
        self.root = tk.Tk()
        self.root.title("Artwork Labeler")

        # Load CSV data
        self.df = pd.read_csv(csv_path)
        self.current_index = 0
        self.labels = {}

        # Load existing labels if any
        self.labels_file = "image_labels.json"
        if os.path.exists(self.labels_file):
            with open(self.labels_file, "r") as f:
                self.labels = json.load(f)

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
        ttk.Button(self.controls, text="Next (→)", command=self.next_image).pack(pady=5)
        ttk.Button(self.controls, text="Save (s)", command=self.save_labels).pack(
            pady=5
        )

        # Instructions
        instructions = """
        Instructions:
        1/2 - Set image type
        y/n - Toggle signature
        3/4 - Set photo angle
        5/6 - Set cropping quality
        Click and drag - Draw signature box
        r - Reset signature boxes
        s - Save labels
        ←/→ - Navigate images
        """
        ttk.Label(self.controls, text=instructions).pack(anchor="w", pady=20)

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
        elif key == "s":
            self.save_labels()
        elif event.keysym == "Left":
            self.prev_image()
        elif event.keysym == "Right":
            self.next_image()

    def load_current_image(self):
        if 0 <= self.current_index < len(self.df):
            row = self.df.iloc[self.current_index]
            image_path = row["Path"]

            # Load and display image
            try:
                image = Image.open(image_path)
                # Resize image to fit canvas while maintaining aspect ratio
                display_size = (800, 600)  # Adjust as needed
                image.thumbnail(display_size, Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(image)
                self.canvas.config(width=image.width, height=image.height)
                self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

                # Store image dimensions for signature box calculations
                self.image_width = image.width
                self.image_height = image.height

                # Load existing labels if any
                self.load_current_labels()

            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

    def load_current_labels(self):
        image_path = self.df.iloc[self.current_index]["Path"]
        if image_path in self.labels:
            label_data = self.labels[image_path]
            self.image_type_var.set(label_data.get("type", PhotoType.UNKNOWN.value))
            self.material_var.set(label_data.get("material", ""))
            self.signature_var.set(label_data.get("has_signature", "unknown"))
            self.angle_var.set(label_data.get("angle", PhotoAngle.UNKNOWN.value))
            self.cropping_var.set(
                label_data.get("cropping", CroppingQuality.UNKNOWN.value)
            )

            # Restore signature boxes
            self.current_boxes = label_data.get("signature_boxes", [])
            self.redraw_signature_boxes()

    def save_labels(self):
        image_path = self.df.iloc[self.current_index]["Path"]
        self.labels[image_path] = {
            "type": self.image_type_var.get(),
            "material": self.material_var.get(),
            "has_signature": self.signature_var.get(),
            "angle": self.angle_var.get(),
            "cropping": self.cropping_var.get(),
            "signature_boxes": self.current_boxes,
        }

        with open(self.labels_file, "w") as f:
            json.dump(self.labels, f, indent=2)
        print(f"Labels saved for {image_path}")

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

    def redraw_signature_boxes(self):
        self.canvas.delete("signature_box")
        for box in self.current_boxes:
            self.canvas.create_rectangle(
                box["x1"] * self.image_width,
                box["y1"] * self.image_height,
                box["x2"] * self.image_width,
                box["y2"] * self.image_height,
                outline="red",
                width=2,
                tags="signature_box",
            )

    def reset_signature_boxes(self):
        self.current_boxes = []
        self.canvas.delete("signature_box")

    def prev_image(self):
        if self.current_index > 0:
            self.save_labels()
            self.current_index -= 1
            self.load_current_image()

    def next_image(self):
        if self.current_index < len(self.df) - 1:
            self.save_labels()
            self.current_index += 1
            self.load_current_image()


def setup_logging(log_level=logging.INFO, log_file=None):
    """Configure logging with optional file output"""
    # Configure base logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],  # Console output
    )

    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Catalog art images in a directory")
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file with image data",
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
    # Parse arguments
    args = parser.parse_args()
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level=log_level, log_file=args.log)

    labeler = ImageLabeler(args.csv)


if __name__ == "__main__":
    main()
