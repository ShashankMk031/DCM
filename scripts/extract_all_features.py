"""
Script to batch extract features from all audio files in the DATA directory.
"""

import argparse
import logging
import os
from pathlib import Path
import time
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_album(album_path, output_dir, force=False):
    """Process all audio files in a single album directory."""
    try:
        # Create a safe directory name for output
        album_name = os.path.basename(album_path.rstrip('/'))
        safe_album_name = "".join(c if c.isalnum() or c in ' ._-' else '_' for c in album_name)
        output_file = os.path.join(output_dir, f"{safe_album_name}_features.csv")
        
        # Skip if output file exists and force is False
        if not force and os.path.exists(output_file):
            logger.info(f"Skipping {album_name} - features already extracted")
            return True
            
        logger.info(f"Processing album: {album_name}")
        
        # Run the extract command
        cmd = f"dcm extract \"{album_path}\" -o \"{output_file}\""
        if force:
            cmd += " --force"
        
        return_code = os.system(cmd)
        if return_code != 0:
            logger.error(f"Error processing {album_name}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error processing {album_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Batch extract features from music library')
    parser.add_argument('data_dir', default='DATA', nargs='?', 
                       help='Path to the DATA directory containing music files')
    parser.add_argument('-o', '--output-dir', default='features',
                       help='Directory to save feature files (default: features/)')
    parser.add_argument('-f', '--force', action='store_true',
                       help='Force re-extraction even if features exist')
    parser.add_argument('--max-workers', type=int, default=1,
                       help='Number of parallel workers (not implemented yet)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all album directories
    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Get all album directories (one level deep from data_dir)
    album_dirs = []
    for entry in os.scandir(data_dir):
        if entry.is_dir():
            album_dirs.append(entry.path)
    
    if not album_dirs:
        logger.warning(f"No album directories found in {data_dir}")
        return
    
    logger.info(f"Found {len(album_dirs)} album directories")
    
    # Process each album
    success_count = 0
    start_time = time.time()
    
    for album_path in tqdm(album_dirs, desc="Processing albums"):
        try:
            if process_album(album_path, args.output_dir, args.force):
                success_count += 1
        except Exception as e:
            logger.error(f"Unexpected error processing {album_path}: {str(e)}")
    
    # Print summary
    elapsed = time.time() - start_time
    logger.info(f"Processed {success_count}/{len(album_dirs)} albums successfully in {elapsed:.2f} seconds")
    logger.info(f"Feature files saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
