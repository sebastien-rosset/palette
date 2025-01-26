#!/usr/bin/env python3

import os
import re
import csv
import argparse
import logging
from typing import Dict, List, Optional, Tuple
from PIL import Image
import imagehash

class ArtCatalog:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.catalog: Dict[str, Dict] = {}
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif')
    
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
        """Extract information from filename"""
        info = {
            'original_filename': filename,
            'year': default_year,
            'catalog_number': '',
            'title': '',
            'orientation': '',
            'width': '',
            'height': ''
        }
        
        # Remove file extension
        name_without_ext = os.path.splitext(filename)[0]
        
        # Split by '-'
        parts = name_without_ext.split('-')
        
        # First part might be a number (catalog/sequence number)
        if parts[0].isdigit():
            info['catalog_number'] = parts[0]
            parts = parts[1:]
        
        # Last part might contain size information
        size_match = re.search(r'(\d+)[xX](\d+)', parts[-1])
        if size_match:
            info['width'] = int(size_match.group(1))
            info['height'] = int(size_match.group(2))
            parts = parts[:-1]
        
        # Check for orientation
        orientation_match = re.search(r'^[Hh]°?$', parts[-1])
        if orientation_match:
            info['orientation'] = 'horizontal'
            parts = parts[:-1]
        elif len(parts) > 1 and re.search(r'^[Vv]$', parts[-1]):
            info['orientation'] = 'vertical'
            parts = parts[:-1]
        
        # Remaining parts form the title
        info['title'] = ' '.join(parts)
        
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
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create and run catalog
    catalog = ArtCatalog(args.path)
    catalog.find_and_catalog_images()
    catalog.generate_csv_report(args.output)
    print(f"Catalog report generated: {args.output}")

if __name__ == '__main__':
    main()