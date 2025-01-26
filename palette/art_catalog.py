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
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Console output
        ]
    )

    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)

FRENCH_CANVAS_SIZES = {
    # Figure (portrait) sizes
    'F': {
        '0': (12, 16),
        '1': (15, 22),
        '2': (22, 33),
        '3': (24, 35),
        '4': (33, 41),
        '5': (41, 50),
        # Add more as needed
    },
    # Marine (seascape) sizes
    'M': {
        '0': (15, 22),
        '1': (22, 33),
        '2': (33, 41),
        # Add more as needed
    },
    # Landscape sizes
    'P': {
        '0': (15, 22),
        '1': (22, 33),
        '2': (33, 41),
        # Add more as needed
    }
}

def get_standard_size(standard: str) -> Optional[Tuple[int, int]]:
    """
    Convert French canvas standard size to dimensions
    
    :param standard: Size standard (e.g. '12F', '2P')
    :return: Tuple of (width, height) in cm, or None if not found
    """
    match = re.match(r'(\d+)([FMPfmp])', standard)
    if not match:
        return None
    
    size, type_ = match.groups()
    type_ = type_.upper()
    
    try:
        return FRENCH_CANVAS_SIZES.get(type_, {}).get(size)
    except (TypeError, KeyError):
        return None

MATERIAL_INDICATORS = {
    'H°': 'Huile (horizontal)',
    'H': 'Huile',
    'Dcr': 'Dessin à la craie',
    'Pg': 'Peinture à gouache',
    'Lch': "Lavis à l'encre de chine",
    'L/s': 'Lavis sur',
    'Ps': 'Pastel',
    'Lch/s': "Lavis à l'encre de chine sur",
    'L': 'Lavis',
    'Ac': 'Aquarelle',
    'Aps': 'Aquarelle et pastel',
    'LTM': 'Lavis technique mixte',
    'DL': 'Dessin lavis',
    'D': 'Dessin',
    'Ac/p': 'Aquarelle et peinture',
    'Tm': 'Technique mixte'
}

class ArtCatalog:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.catalog: Dict[str, Dict] = {}
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif')
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
            year_match = re.search(r'(?:CATALOGUE[-_])?(\d{4}|(\d{2}))', component)
            if year_match:
                if year_match.group(1):
                    return int(year_match.group(1))
                elif year_match.group(2):
                    year_digit = int(year_match.group(2))
                    return 1900 + year_digit if year_digit <= 99 else 2000 + year_digit
        
        # If no year found in path, try extracting from filename
        filename = os.path.basename(filepath)
        filename_year_match = re.search(r'(\d{4}|\d{2})', filename)
        if filename_year_match:
            year = int(filename_year_match.group(1))
            if 1900 <= year <= 2100:
                return year
        
        return None

    def parse_filename(self, filename: str, default_year: Optional[int] = None) -> Dict:
        """Extract detailed information from filename"""
        info = {
            'original_filename': filename,
            'year': default_year,
            'catalog_number': '',
            'title': '',
            'material': '',
            'orientation': '',
            'size_standard': '',
            'width': '',
            'height': '',
            'size_type': ''
        }
        
        # Remove file extension
        name_without_ext = os.path.splitext(filename)[0]
        
        # Split by '-'
        parts = name_without_ext.split('-')
        
        # First part might be a number (catalog/sequence number)
        if parts[0].isdigit():
            info['catalog_number'] = parts[0]
            parts = parts[1:]
        
        # Detect material/technique
        material_match = next((ind for ind in MATERIAL_INDICATORS if ind in parts), None)
        if material_match:
            info['material'] = MATERIAL_INDICATORS[material_match]
        
        # Detect standard size and type (F, M, P, etc.)
        size_std_match = re.search(r'(\d+)([FMPfmp])', name_without_ext)
        if size_std_match:
            info['size_standard'] = size_std_match.group(0)
            info['size_type'] = size_std_match.group(2).upper()
            
            # Attempt to get standard dimensions
            std_dims = get_standard_size(info['size_standard'])
            if std_dims:
                info['width'], info['height'] = std_dims
        
        # Detect numeric dimensions with dot or comma as decimal separator
        dim_match = re.search(r'(\d+[.,]?\d*)[\s]*[xX][\s]*(\d+[.,]?\d*)', name_without_ext)
        if dim_match:
            width = float(dim_match.group(1).replace(',', '.'))
            height = float(dim_match.group(2).replace(',', '.'))
            info['width'] = width
            info['height'] = height
        
        # Check for orientation
        if re.search(r'[Hh]°?', name_without_ext):
            info['orientation'] = 'horizontal'
        elif re.search(r'[Vv]', name_without_ext):
            info['orientation'] = 'vertical'
        
        # Prepare title removal markers
        title_markers = [
            info['catalog_number'],
            *[ind for ind in MATERIAL_INDICATORS if ind in parts],
            r'\d+[FMPfmp]',  # Size standards
            r'\d+[xX]\d+',   # Dimensions
            r'[Hh]°?',       # Horizontal
            r'[Vv]',         # Vertical
        ]
        
        # Construct title by removing these markers
        title_parts = []
        for part in parts:
            # Check if part matches any marker
            if not any(re.search(marker, part) for marker in title_markers):
                title_parts.append(part)
        
        info['title'] = ' '.join(title_parts).strip()
        
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
        for root, _, files in os.walk(self.base_path):
            for filename in files:
                # Check if file is an image
                if not filename.lower().endswith(self.image_extensions):
                    continue
                filepath = os.path.join(root, filename)
                
                # Extract contextual year
                year = self.extract_contextual_year(filepath)
                
                # Parse filename
                file_info = self.parse_filename(filename, year)
                
                # Generate image hash
                image_hash = self.generate_image_hash(filepath)
                
                # Store in catalog
                if image_hash not in self.catalog:
                    logging.info(f"Cataloging {filepath}: {file_info}")
                    self.catalog[image_hash] = {
                        'files': [filepath],
                        'info': file_info
                    }
                else:
                    # If hash exists, add this file to the list of duplicate/similar images
                    self.catalog[image_hash]['files'].append(filepath)
    
    def generate_csv_report(self, output_file='art_catalog_report.csv'):
        """Generate a comprehensive CSV report of the cataloged images"""
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            
            # Write header
            headers = [
                'Image Hash', 'Duplicate Count', 'Title', 'Year', 
                'Catalog Number', 'Orientation', 'Width', 'Height', 
                'File Paths'
            ]
            csv_writer.writerow(headers)
            
            # Write data rows
            for image_hash, image_data in self.catalog.items():
                info = image_data['info']
                csv_writer.writerow([
                    image_hash,
                    len(image_data['files']),
                    info.get('title', ''),
                    info.get('year', ''),
                    info.get('catalog_number', ''),
                    info.get('orientation', ''),
                    info.get('width', ''),
                    info.get('height', ''),
                    '; '.join(image_data['files'])
                ])

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Catalog art images in a directory')
    parser.add_argument('--path', type=str, help='Directory containing art images')
    parser.add_argument('-o', '--output', type=str, default='art_catalog_report.csv', 
                        help='Output CSV file path (default: art_catalog_report.csv)')
    parser.add_argument('-l', '--log', type=str, 
                        help='Log file path (optional)')
    parser.add_argument('--log-level', type=str, 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        default='INFO',
                        help='Logging level (default: INFO)')
    # Parse arguments
    args = parser.parse_args()
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level=log_level, log_file=args.log)
    
    # Create and run catalog
    catalog = ArtCatalog(args.path)
    catalog.find_and_catalog_images()
    catalog.generate_csv_report(args.output)
    print(f"Catalog report generated: {args.output}")

if __name__ == '__main__':
    main()