"""
Script to batch extract features from all audio files in the DATA directory.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcm.core.extract_features import process_directory

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

def main():
    parser = argparse.ArgumentParser(description='Batch extract features from music library')
    parser.add_argument('data_dir', default='DATA', nargs='?', 
                       help='Path to the DATA directory containing music files')
    parser.add_argument('-o', '--output', default='features/all_features.csv',
                       help='path to output file (default: features/all_features.csv)')
    parser.add_argument('-f', '--force', action='store_true',
                       help='Force re-extraction even if features exist')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: all CPUs)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Directory not found: {data_dir}")
        return
        
    logger.info(f"Starting parallel feature extraction on {data_dir}")
    if args.workers:
        logger.info(f"Using {args.workers} worker processes")
    else:
        logger.info(f"Using default (max) worker processes")
        
    # Call the optimized parallel processor
    # This handles recursion, finding files, output saving/loading, and multiprocessing internally
    df = process_directory(
        input_dir=data_dir,
        output_file=args.output,
        force=args.force,
        n_workers=args.workers
    )
    
    if not df.empty:
        logger.info(f"Successfully processed library. Total features: {len(df)}")
        logger.info(f"Saved to: {os.path.abspath(args.output)}")
    else:
        logger.warning("No features extracted.")

if __name__ == "__main__":
    main()
