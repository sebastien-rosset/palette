#!/usr/bin/env python3

import argparse
import logging
import re
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
    def __init__(self, input_csv, output_csv, filename_regex):
        self.root = tk.Tk()
        self.root.title("Artwork Labeler")
        self.filename_regex = filename_regex
        if self.filename_regex:
            self.filename_regex = re.compile(self.filename_regex)
        logging.info(f"Filename regex: {self.filename_regex}")
        self.current_image_modified = False
        self.catalog_number_var = tk.StringVar()
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

        self.filename_var = tk.StringVar()
        self.title_var = tk.StringVar()
        self.dimensions_var = tk.StringVar()

        # Variables for signature box drawing
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.current_boxes = []

        self.setup_controls()
        self.load_current_image()

        # Start the main loop
        self.root.mainloop()

    def mark_as_modified(self, *args):
        """Callback for when any control value changes"""
        self.current_image_modified = True

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
        # Image metadata display
        metadata_frame = ttk.LabelFrame(
            self.controls, text="Image Information", padding="5 5 5 5"
        )
        metadata_frame.pack(fill="x", padx=5, pady=5)

        # Catalog Number
        ttk.Label(metadata_frame, text="Catalog Number:").pack(anchor="w")
        ttk.Label(
            metadata_frame, textvariable=self.catalog_number_var, style="Info.TLabel"
        ).pack(anchor="w", padx=10)

        # Filename
        ttk.Label(metadata_frame, text="Filename:").pack(anchor="w")
        ttk.Label(
            metadata_frame, textvariable=self.filename_var, style="Info.TLabel"
        ).pack(anchor="w", padx=10)

        # Title
        ttk.Label(metadata_frame, text="Title:").pack(anchor="w")
        ttk.Label(
            metadata_frame, textvariable=self.title_var, style="Info.TLabel"
        ).pack(anchor="w", padx=10)

        # Dimensions
        ttk.Label(metadata_frame, text="Dimensions:").pack(anchor="w")
        ttk.Label(
            metadata_frame, textvariable=self.dimensions_var, style="Info.TLabel"
        ).pack(anchor="w", padx=10)

        # Create a frame for each group of controls
        image_type_frame = ttk.LabelFrame(
            self.controls, text="Image Type", padding="5 5 5 5"
        )
        image_type_frame.pack(fill="x", padx=5, pady=5)

        # Image type radio buttons
        self.image_type_var = tk.StringVar(value="")
        ttk.Radiobutton(
            image_type_frame,
            text="Regular",
            value="regular",
            variable=self.image_type_var,
        ).pack(anchor="w")
        ttk.Radiobutton(
            image_type_frame,
            text="Artwork",
            value="artwork",
            variable=self.image_type_var,
        ).pack(anchor="w")
        ttk.Button(
            image_type_frame, text="Reset", command=lambda: self.image_type_var.set("")
        ).pack(anchor="w")

        # Material selection
        material_frame = ttk.LabelFrame(
            self.controls, text="Material", padding="5 5 5 5"
        )
        material_frame.pack(fill="x", padx=5, pady=5)

        self.material_var = tk.StringVar()
        materials = ["Huile", "Pastel", "Technique mixte", "Lavis", "Acrylique"]
        for material in materials:
            ttk.Radiobutton(
                material_frame,
                text=material,
                value=material,
                variable=self.material_var,
            ).pack(anchor="w")
        ttk.Button(
            material_frame, text="Reset", command=lambda: self.material_var.set("")
        ).pack(anchor="w")

        # Signature presence
        signature_frame = ttk.LabelFrame(
            self.controls, text="Signature", padding="5 5 5 5"
        )
        signature_frame.pack(fill="x", padx=5, pady=5)

        self.signature_var = tk.StringVar(value="")
        ttk.Radiobutton(
            signature_frame, text="Yes", value="yes", variable=self.signature_var
        ).pack(anchor="w")
        ttk.Radiobutton(
            signature_frame, text="No", value="no", variable=self.signature_var
        ).pack(anchor="w")
        ttk.Button(
            signature_frame, text="Reset", command=lambda: self.signature_var.set("")
        ).pack(anchor="w")

        # Photo angle
        angle_frame = ttk.LabelFrame(
            self.controls, text="Photo Angle", padding="5 5 5 5"
        )
        angle_frame.pack(fill="x", padx=5, pady=5)

        self.angle_var = tk.StringVar(value="")
        ttk.Radiobutton(
            angle_frame, text="Straight", value="straight", variable=self.angle_var
        ).pack(anchor="w")
        ttk.Radiobutton(
            angle_frame, text="Angled", value="angled", variable=self.angle_var
        ).pack(anchor="w")
        ttk.Button(
            angle_frame, text="Reset", command=lambda: self.angle_var.set("")
        ).pack(anchor="w")

        # Cropping quality
        cropping_frame = ttk.LabelFrame(
            self.controls, text="Cropping", padding="5 5 5 5"
        )
        cropping_frame.pack(fill="x", padx=5, pady=5)

        self.cropping_var = tk.StringVar(value="")
        ttk.Radiobutton(
            cropping_frame,
            text="Well Cropped",
            value="well_cropped",
            variable=self.cropping_var,
        ).pack(anchor="w")
        ttk.Radiobutton(
            cropping_frame,
            text="Includes Surroundings",
            value="includes_surroundings",
            variable=self.cropping_var,
        ).pack(anchor="w")
        ttk.Button(
            cropping_frame, text="Reset", command=lambda: self.cropping_var.set("")
        ).pack(anchor="w")

        self.image_type_var.trace_add("write", self.mark_as_modified)
        self.material_var.trace_add("write", self.mark_as_modified)
        self.signature_var.trace_add("write", self.mark_as_modified)
        self.angle_var.trace_add("write", self.mark_as_modified)
        self.cropping_var.trace_add("write", self.mark_as_modified)

        # Signature box controls
        signature_box_frame = ttk.LabelFrame(
            self.controls, text="Signature Box", padding="5 5 5 5"
        )
        signature_box_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(
            signature_box_frame, text="Click and drag on image to draw signature box"
        ).pack(anchor="w")
        ttk.Button(
            signature_box_frame, text="Reset Boxes", command=self.reset_signature_boxes
        ).pack(anchor="w")

        # Navigation buttons
        nav_frame = ttk.Frame(self.controls)
        nav_frame.pack(fill="x", padx=5, pady=15)

        ttk.Button(nav_frame, text="← Previous", command=self.prev_image).pack(
            side="left", padx=5
        )
        ttk.Button(nav_frame, text="Next →", command=self.next_image).pack(
            side="right", padx=5
        )

        # Progress indicator
        self.progress_var = tk.StringVar()
        self.update_progress_indicator()
        ttk.Label(self.controls, textvariable=self.progress_var).pack(pady=5)

        # Canvas bindings for signature box
        self.canvas.bind("<ButtonPress-1>", self.start_rect)
        self.canvas.bind("<B1-Motion>", self.draw_rect)
        self.canvas.bind("<ButtonRelease-1>", self.end_rect)

        # Style for metadata labels
        style = ttk.Style()
        style.configure("Info.TLabel", font=("TkDefaultFont", 9, "bold"))

    def update_metadata_display(self):
        if 0 <= self.current_index < len(self.df):
            row = self.df.iloc[self.current_index]

            # Update metadata display - safely get values with fallback to empty string
            self.catalog_number_var.set(str(row.get("catalog_number", "")))
            self.filename_var.set(os.path.basename(row["Path"]))
            self.title_var.set(str(row.get("title", "")))
            self.dimensions_var.set(str(row.get("dimensions", "")))
            logging.info(f"Catalog number: {self.catalog_number_var.get()}")
        else:
            logging.info("All images processed")

    def update_progress_indicator(self):
        total = len(self.df)
        processed = len(self.processed_files)
        self.progress_var.set(f"Progress: {processed}/{total} images processed")

    def load_current_image(self):
        if 0 <= self.current_index < len(self.df):
            row = self.df.iloc[self.current_index]
            image_path = row["Path"]
            filename = os.path.basename(image_path)

            # Check if image was already processed
            if image_path in self.processed_files:
                self.next_image()
                logging.info(f"Skipping image {image_path}")
                return

            # Check if filename matches regex pattern
            if self.filename_regex and not self.filename_regex.search(filename):
                logging.info(f"Skipping file {filename} - does not match regex pattern")
                self.next_image()
                return

            # Load and display image
            try:
                logging.info(f"Loading image {image_path}")
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

                self.update_metadata_display()

                # Reset current labels
                self.reset_labels()
                self.current_image_modified = False

            except Exception as e:
                logging.error(f"Error loading image {image_path}: {e}")
                self.next_image()
        else:
            logging.info("All images processed")

    def reset_labels(self):
        self.image_type_var.set("")
        self.material_var.set("")
        self.signature_var.set("")
        self.angle_var.set("")
        self.cropping_var.set("")
        self.reset_signature_boxes()
        self.current_image_modified = False

    def has_labels_to_save(self):
        """Check if any attributes have been set and the image has been modified"""
        return self.current_image_modified and any(
            [
                self.image_type_var.get() != "",
                self.material_var.get() != "",
                self.signature_var.get() != "",
                self.angle_var.get() != "",
                self.cropping_var.get() != "",
                len(self.current_boxes) > 0,
            ]
        )

    def save_current_labels(self):
        if 0 <= self.current_index < len(self.df):
            image_path = self.df.iloc[self.current_index]["Path"]

            # Skip if already processed
            if image_path in self.processed_files:
                return

            # Skip if no attributes were set
            if not self.has_labels_to_save():
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

    def update_progress_indicator(self):
        total = len(self.df)
        processed = len(self.processed_files)
        current_dir = os.path.dirname(self.df.iloc[self.current_index]["Path"])
        self.progress_var.set(
            f"Directory: {current_dir}\nProgress: {processed}/{total} images processed"
        )

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
        self.current_image_modified = True

    def reset_signature_boxes(self):
        self.current_boxes = []
        self.canvas.delete("signature_box")

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

    def next_image(self):
        """Navigate to next image using directory-based approach"""
        if self.current_index >= len(self.df) - 1:
            return

        # Try to save current labels if they exist
        if self.current_image_modified:
            self.save_current_labels()

        current_dir = os.path.dirname(self.df.iloc[self.current_index]["Path"])

        # First try to find next unprocessed image in current directory
        next_idx = self.find_next_unprocessed_in_directory(current_dir)

        logging.info(
            f"Processing image {self.current_index+1}/{len(self.df)}. Current directory: {current_dir}. Next idx: {next_idx}"
        )

        if next_idx is None:
            # If no more unprocessed images in current directory, find next directory
            next_dir = self.find_next_unprocessed_directory()
            if next_dir is not None:
                next_idx = self.find_next_unprocessed_in_directory(next_dir)

        if next_idx is not None:
            self.current_index = next_idx
            self.load_current_image()

    def get_directory_groups(self):
        """Group files by their parent directory"""
        dir_groups = {}
        for idx, row in self.df.iterrows():
            dir_path = os.path.dirname(row["Path"])
            if dir_path not in dir_groups:
                dir_groups[dir_path] = []
            dir_groups[dir_path].append(idx)
        return dir_groups

    def find_next_unprocessed_directory(self):
        """Find next directory with unprocessed images"""
        dir_groups = self.get_directory_groups()
        current_dir = os.path.dirname(self.df.iloc[self.current_index]["Path"])

        # Get list of directories
        dirs = list(dir_groups.keys())
        if not dirs:
            return None

        # Find current directory index
        try:
            current_dir_idx = dirs.index(current_dir)
        except ValueError:
            current_dir_idx = -1

        # Check each directory starting from the next one
        for i in range(len(dirs)):
            check_idx = (current_dir_idx + 1 + i) % len(dirs)
            dir_path = dirs[check_idx]

            # Check if directory has any unprocessed images
            for idx in dir_groups[dir_path]:
                filepath = self.df.iloc[idx]["Path"]
                if filepath not in self.processed_files:
                    return dir_path

        return None

    def find_next_unprocessed_in_directory(self, dir_path):
        """Find next unprocessed image in specified directory"""
        dir_groups = self.get_directory_groups()
        if dir_path not in dir_groups:
            return None

        # Get all indices in this directory
        dir_indices = dir_groups[dir_path]

        # Find where we are in this directory's list
        try:
            current_position = dir_indices.index(self.current_index)
            # Start searching from the next position
            dir_indices = dir_indices[current_position + 1 :]
        except ValueError:
            # If current index not found, search all indices
            pass

        # Look for next unprocessed image
        for idx in dir_indices:
            if self.df.iloc[idx]["Path"] not in self.processed_files:
                return idx

        return None


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
    parser.add_argument(
        "--regex",
        type=str,
        help="Regex matching filename",
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

    labeler = ImageLabeler(args.csv, args.output, args.regex)


if __name__ == "__main__":
    main()
