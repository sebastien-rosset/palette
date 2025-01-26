#!/usr/bin/env python3

import os
import re
import csv
import argparse
import logging
from typing import Dict, List, Optional, Tuple
from PIL import Image
import imagehash


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


def calculate_french_canvas_size(
    size_number: int, format_type: str
) -> tuple[float, float]:
    """
    Calculate French standard canvas size based on size number and format.

    Args:
        size_number: The size number (0, 1, 2, etc.)
        format_type: 'F' for Figure, 'M' for Marine, 'P' for Paysage (Landscape)

    Returns:
        Tuple of (width, height) in centimeters
    """
    # Base size (Size 0) in centimeters
    BASE_SIZES = {
        "F": (12, 16),  # Figure (portrait)
        "M": (15, 22),  # Marine (seascape)
        "P": (15, 22),  # Paysage (landscape)
    }

    if format_type not in BASE_SIZES:
        raise ValueError(f"Unknown format type: {format_type}")

    if size_number < 0:
        raise ValueError("Size number cannot be negative")

    base_width, base_height = BASE_SIZES[format_type]

    if size_number == 0:
        return (base_width, base_height)

    # The French system approximately follows a geometric progression
    # Each size up multiplies the area by roughly 2
    # To maintain proportions, width and height each increase by √2
    scale_factor = pow(2, size_number / 2)  # sqrt(2) for each size number

    width = round(base_width * scale_factor, 1)
    height = round(base_height * scale_factor, 1)

    # For Paysage (landscape) format, swap dimensions
    if format_type == "P":
        width, height = height, width

    return (width, height)


def generate_french_canvas_sizes(max_size: int = 50) -> dict:
    """
    Generate a dictionary of all French canvas sizes up to the specified maximum size.

    Args:
        max_size: Maximum size number to generate (default 12)

    Returns:
        Dictionary of canvas sizes in the same format as FRENCH_CANVAS_SIZES
    """
    sizes = {"F": {}, "M": {}, "P": {}}

    for format_type in sizes:
        for size in range(max_size + 1):
            sizes[format_type][str(size)] = calculate_french_canvas_size(
                size, format_type
            )

    return sizes


# Generate the full size dictionary
FRENCH_CANVAS_SIZES = generate_french_canvas_sizes()


def get_standard_size(standard: str) -> Optional[Tuple[int, int]]:
    """
    Convert French canvas standard size to dimensions

    :param standard: Size standard (e.g. '12F', '2P')
    :return: Tuple of (width, height) in cm, or None if not found
    """
    match = re.match(r"(\d+)([FMPfmp])", standard)
    if not match:
        return None

    size, type_ = match.groups()
    type_ = type_.upper()

    try:
        return FRENCH_CANVAS_SIZES.get(type_, {}).get(size)
    except (TypeError, KeyError):
        raise ValueError(f"Invalid standard size: {standard}")


MATERIAL_INDICATORS = {
    "H°": "Huile (horizontal)",
    "H": "Huile",
    "Dcr": "Dessin à la craie",
    "Pg": "Peinture à gouache",
    "Lch": "Lavis à l'encre de chine",
    "L/s": "Lavis sur",
    "Ps": "Pastel",
    "Lch/s": "Lavis à l'encre de chine sur",
    "L": "Lavis",
    "Ac": "Aquarelle",
    "Aps": "Aquarelle et pastel",
    "LTM": "Lavis technique mixte",
    "DL": "Dessin lavis",
    "D": "Dessin",
    "Ac/p": "Aquarelle et peinture",
    "Tm": "Technique mixte",
}


class ArtCatalog:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.catalog: Dict[str, Dict] = {}
        self.image_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".tiff",
            ".bmp",
            ".gif",
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_contextual_year(self, filepath: str) -> Optional[int]:
        """
        Extract year from directory path or filename
        Prioritizes directory naming conventions, falls back to filename
        """
        # Try extracting year from directory path
        path_components = filepath.split(os.path.sep)
        for component in path_components:
            # First check for CATALOGUE-YYYY or CATALOGUE-YY format
            year_match = re.search(r"(?:CATALOGUE[-_])?(\d{4}|(\d{2}))", component)
            if year_match:
                if year_match.group(1):
                    return int(year_match.group(1))
                elif year_match.group(2):
                    year_digit = int(year_match.group(2))
                    return 1900 + year_digit if year_digit <= 99 else 2000 + year_digit

        # If no year found in path, try extracting from filename
        filename = os.path.basename(filepath)
        filename_year_match = re.search(r"(\d{4}|\d{2})", filename)
        if filename_year_match:
            year = int(filename_year_match.group(1))
            if 1900 <= year <= 2100:
                return year

        return None

    def parse_filename(
        self, filepath: str, filename: str, default_year: Optional[int] = None
    ) -> Dict:
        """Extract detailed information from filename

        The filename generally follows this pattern:
        reference_numbers - title - material - format

        Where:
        - reference_numbers: can contain digits, dashes, and single letters (e.g., "9061-d", "21205-4")
        - title: the artwork title (e.g., "Famille bernache", "Coucher de soleil")
        - material: technique indicator (e.g., "Lch", "Ps")
        - format: dimensions or standard size (e.g., "50X50", "15F")

        Elements may be missing and separators may vary.
        """
        info = {
            "original_filename": filename,
            "year": default_year,
            "catalog_number": "",
            "title": "",
            "material": "",
            "orientation": "",
            "size_standard": "",
            "width": "",
            "height": "",
            "size_type": "",
        }

        # Remove file extension
        name_without_ext = os.path.splitext(filename)[0]

        # First check for trailing copy number (e.g., "(1)", "(2)")
        copy_match = re.search(r"\(\d+\)$", name_without_ext)
        if copy_match:
            name_without_ext = name_without_ext[: copy_match.start()].strip("-")

        # First extract the reference numbers from the start
        parts = name_without_ext.split("-")

        # Extract catalog number and sequence numbers
        ref_parts = []
        current_idx = 0
        for part in parts:
            if part.isdigit() or (len(part) == 1 and part.isalpha()):
                ref_parts.append(part)
                current_idx += 1
            else:
                break

        if ref_parts:
            info["catalog_number"] = "-".join(ref_parts)
            # Get the rest of the string, preserving the first character of the title
            name_without_ext = "-".join(parts[current_idx:])
        else:
            name_without_ext = name_without_ext

        # Look for dimensions from the end
        dim_match = re.search(
            r"[-\s]*(\d+[.,]?\d*)[\s]*[xX][\s]*(\d+[.,]?\d*)[-\s]*$", name_without_ext
        )
        if dim_match:
            width = float(dim_match.group(1).replace(",", "."))
            height = float(dim_match.group(2).replace(",", "."))
            info["width"] = width
            info["height"] = height

            # Set orientation based on dimensions
            if width > height:
                info["orientation"] = "horizontal"
            elif height > width:
                info["orientation"] = "vertical"

            name_without_ext = name_without_ext[: dim_match.start()].strip("-")
        else:
            # Look for standard size format (e.g., 15F)
            size_match = re.search(r"[-\s]*(\d+)([FMPfmp])[-\s]*$", name_without_ext)
            if size_match:
                info["size_standard"] = size_match.group(1) + size_match.group(2)
                info["size_type"] = size_match.group(2).upper()

                # Try to get standard dimensions
                std_dims = get_standard_size(info["size_standard"])
                if std_dims:
                    info["width"], info["height"] = std_dims
                else:
                    raise ValueError(f"Invalid standard size: {info['size_standard']}")
                name_without_ext = name_without_ext[: size_match.start()].strip("-")

        # Look for material indicators from the end
        material_indicators = sorted(MATERIAL_INDICATORS.keys(), key=len, reverse=True)
        for ind in material_indicators:
            if name_without_ext.endswith(f"-{ind}") or name_without_ext.endswith(ind):
                info["material"] = MATERIAL_INDICATORS[ind]
                name_without_ext = name_without_ext[: -(len(ind))].strip("-")
                break

        # Extract orientation from markers if not set by dimensions
        if not info["orientation"]:
            if "H°" in name_without_ext or re.search(r"[Hh](?![\w])", name_without_ext):
                info["orientation"] = "horizontal"
            elif re.search(r"[Vv](?![\w])", name_without_ext):
                info["orientation"] = "vertical"

        # Clean up any remaining technical markers from the title
        info["title"] = name_without_ext.strip("-").strip()

        if not info["title"]:
            logging.error(f"Could not extract title from filename: {filepath}")
            info["title"] = filename

        logging.debug(f"Title: {info['title']}, Material: {info['material']}")
        return info

    def generate_image_hash(self, filepath: str) -> str:
        """Generate a perceptual hash of the image"""
        try:
            with Image.open(filepath) as img:
                return str(imagehash.phash(img))
        except Exception as e:
            print(f"Error hashing {filepath}: {e}")
            return ""

    def find_and_catalog_images(self):
        """Recursively find and catalog images from the base path"""
        for root, dir, files in os.walk(self.base_path):
            for filename in files:
                # Check if file is an image
                if not filename.lower().endswith(self.image_extensions):
                    continue
                filepath = os.path.join(root, filename)

                # Extract contextual year
                year = self.extract_contextual_year(filepath)

                # Parse filename
                file_info = self.parse_filename(filepath, filename, year)

                # Generate image hash
                image_hash = self.generate_image_hash(filepath)

                # Store in catalog
                if image_hash not in self.catalog:
                    logging.info(f"Cataloging {filepath}: {file_info}")
                    self.catalog[image_hash] = {"files": [filepath], "info": file_info}
                else:
                    # If hash exists, add this file to the list of duplicate/similar images
                    self.catalog[image_hash]["files"].append(filepath)

    def generate_csv_report(self, output_file="art_catalog_report.csv"):
        """Generate a comprehensive CSV report of the cataloged images"""
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            csv_writer = csv.writer(f)

            # Write header
            headers = [
                "Image Hash",
                "Duplicate Count",
                "Title",
                "Year",
                "Catalog Number",
                "Orientation",
                "Width",
                "Height",
                "File Paths",
            ]
            csv_writer.writerow(headers)

            # Write data rows
            for image_hash, image_data in self.catalog.items():
                info = image_data["info"]
                csv_writer.writerow(
                    [
                        image_hash,
                        len(image_data["files"]),
                        info.get("title", ""),
                        info.get("year", ""),
                        info.get("catalog_number", ""),
                        info.get("orientation", ""),
                        info.get("width", ""),
                        info.get("height", ""),
                        "; ".join(image_data["files"]),
                    ]
                )


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Catalog art images in a directory")
    parser.add_argument("--path", type=str, help="Directory containing art images")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="art_catalog_report.csv",
        help="Output CSV file path (default: art_catalog_report.csv)",
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

    # Create and run catalog
    catalog = ArtCatalog(args.path)
    catalog.find_and_catalog_images()
    catalog.generate_csv_report(args.output)
    print(f"Catalog report generated: {args.output}")


if __name__ == "__main__":
    main()
