"""
Feature extraction module for DCM (Desktop/Mobile Music Assistant).
Handles audio feature extraction from music files using librosa.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supported audio file extensions
SUPPORTED_FORMATS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}

# Feature extraction parameters optimized for Indian music
SAMPLE_RATE = 44100  # Higher sample rate for better audio quality
N_FFT = 4096         # Larger FFT for better frequency resolution
HOP_LENGTH = 1024    # Larger hop for efficiency
N_MFCC = 20          # More MFCCs for better feature representation
N_CHROMA = 24        # More chroma bins for Indian classical music

def get_audio_files(directory: Union[str, Path]) -> List[Path]:
    """
    Recursively find all supported audio files in a directory.
    
    Args:
        directory: Path to the directory containing audio files
        
    Returns:
        List of Path objects to audio files
    """
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")
    
    audio_files = []
    for ext in SUPPORTED_FORMATS:
        audio_files.extend(directory.rglob(f'*{ext}'))
    
    return sorted(audio_files)

def extract_features(audio_path: Union[str, Path]) -> Dict[str, Union[float, list]]:
    """
    Extract audio features from a single audio file with optimizations for Indian music.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Dictionary containing extracted features
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    features = {
        'file_path': str(audio_path),
        'file_name': audio_path.name,
        'file_extension': audio_path.suffix.lower(),
        'file_size_mb': os.path.getsize(audio_path) / (1024 * 1024)  # Convert to MB
    }
    
    try:
        # Try loading with different backends if needed
        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            logger.warning(f"PySoundFile failed. Trying audioread instead.")
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, res_type='kaiser_fast')
        
        # Extract features
        features = {}
        
        # Basic features
        features['duration'] = librosa.get_duration(y=y, sr=sr)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        features['spectral_centroid_std'] = float(np.std(spectral_centroid))
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate_mean'] = float(np.mean(zcr))
        features['zero_crossing_rate_std'] = float(np.std(zcr))
        
        # Root Mean Square (Energy)
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        
        # Enhanced MFCCs for Indian music
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=128,  # More mel bands for better frequency resolution
            fmin=27.5,   # Lower frequency bound (A0)
            fmax=16000,  # Upper frequency bound for Indian instruments
            htk=True     # Use HTK formula for mel scale
        )
        for i in range(N_MFCC):
            features[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc[i]))
            features[f'mfcc_{i+1}_std'] = float(np.std(mfcc[i]))
        
        # Enhanced Chroma features for Indian classical music
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            n_chroma=N_CHROMA,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            tuning=0.0,  # Standard tuning
            norm=2      # Normalize each chroma band
        )
        for i in range(chroma.shape[0]):
            features[f'chroma_{i+1}_mean'] = float(np.mean(chroma[i, :]))
            features[f'chroma_{i+1}_std'] = float(np.std(chroma[i, :]))
        
        # Tempo with Indian music optimization
        tempo, _ = librosa.beat.beat_track(
            y=y,
            sr=sr,
            trim=False,
            start_bpm=80,  # Common starting BPM for Indian music
            tightness=100   # Tighter tracking for Indian rhythms
        )
        features['tempo'] = float(tempo)
        
        # Indian music specific features
        try:
            # Harmonic-percussive source separation
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Tonic detection (approximate for Indian music)
            tonic_freq = librosa.estimate_tuning(y=y_harmonic, sr=sr)
            features['tonic_deviation'] = float(tonic_freq)  # Deviation from A4=440Hz
            
            # Rhythm features
            onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)
            pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
            features['rhythm_regularity'] = float(np.mean(pulse))  # Higher = more regular rhythm
            
            # Detect if the music has a drone (common in Indian classical)
            spectral_flatness = librosa.feature.spectral_flatness(y=y_harmonic)
            features['drone_likelihood'] = float(np.mean(1 - spectral_flatness))  # Lower = more drone-like
            
        except Exception as e:
            logger.warning(f"Could not extract Indian music features: {str(e)}")
            features.update({
                'tonic_deviation': 0.0,
                'rhythm_regularity': 0.0,
                'drone_likelihood': 0.0
            })
        
        # Add file metadata
        features['file_path'] = str(audio_path.resolve())
        features['file_name'] = audio_path.name
        features['file_extension'] = audio_path.suffix.lower()
        features['file_size_mb'] = os.path.getsize(audio_path) / (1024 * 1024)
        
        return features
        
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {str(e)}")
        return {}

def process_directory(
    input_dir: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    force: bool = False
) -> pd.DataFrame:
    """
    Process all audio files in a directory and extract features.
    
    Args:
        input_dir: Directory containing audio files
        output_file: Optional path to save features as CSV/JSON
        force: If True, overwrite existing output file
        
    Returns:
        DataFrame containing extracted features
    """
    input_dir = Path(input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")
    
    # Find all audio files
    audio_files = get_audio_files(input_dir)
    if not audio_files:
        logger.warning(f"No supported audio files found in {input_dir}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(audio_files)} audio files in {input_dir}")
    
    # Check if output file exists and load it if force is False
    features_list = []
    existing_files = set()
    
    if output_file and Path(output_file).exists() and not force:
        try:
            if str(output_file).endswith('.json'):
                with open(output_file, 'r') as f:
                    features_list = json.load(f)
            else:  # CSV
                df = pd.read_csv(output_file)
                features_list = df.to_dict('records')
            
            existing_files = {f['file_path'] for f in features_list}
            logger.info(f"Loaded {len(features_list)} existing features from {output_file}")
        except Exception as e:
            logger.warning(f"Error loading existing features: {e}")
            features_list = []
    
    # Process audio files
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for audio_file in tqdm(audio_files, desc="Extracting features"):
        file_path = str(audio_file.resolve())
        
        # Skip if already processed
        if file_path in existing_files:
            skipped_count += 1
            continue
        
        # Extract features
        features = extract_features(audio_file)
        
        if features:
            features_list.append(features)
            processed_count += 1
            
            # Save periodically (every 10 files)
            if processed_count % 10 == 0 and output_file:
                save_features(features_list, output_file)
        else:
            error_count += 1
    
    logger.info(
        f"Feature extraction complete. "
        f"Processed: {processed_count}, "
        f"Skipped: {skipped_count}, "
        f"Errors: {error_count}"
    )
    
    # Save the final results
    if output_file and features_list:
        save_features(features_list, output_file)
    
    return pd.DataFrame(features_list)

def save_features(
    features_list: List[Dict],
    output_file: Union[str, Path],
    format: str = 'auto'
) -> None:
    """
    Save features to a file.
    
    Args:
        features_list: List of feature dictionaries
        output_file: Path to save the features
        format: Output format ('csv', 'json', or 'auto' based on file extension)
    """
    if not features_list:
        logger.warning("No features to save")
        return
    
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine format from file extension if auto
    if format == 'auto':
        if str(output_file).endswith('.json'):
            format = 'json'
        else:
            format = 'csv'  # Default to CSV
    
    try:
        if format == 'json':
            with open(output_file, 'w') as f:
                json.dump(features_list, f, indent=2)
        else:  # CSV
            df = pd.DataFrame(features_list)
            df.to_csv(output_file, index=False)
        
        logger.info(f"Saved {len(features_list)} features to {output_file}")
    except Exception as e:
        logger.error(f"Error saving features to {output_file}: {e}")

def load_features(input_file: Union[str, Path]) -> pd.DataFrame:
    """
    Load features from a file.
    
    Args:
        input_file: Path to the features file (CSV or JSON)
        
    Returns:
        DataFrame containing the loaded features
    """
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Features file not found: {input_file}")
    
    try:
        if str(input_file).endswith('.json'):
            with open(input_file, 'r') as f:
                features = json.load(f)
            return pd.DataFrame(features)
        else:  # CSV
            return pd.read_csv(input_file)
    except Exception as e:
        raise ValueError(f"Error loading features from {input_file}: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract audio features from music files')
    parser.add_argument('input_dir', help='Directory containing audio files')
    parser.add_argument('-o', '--output', help='Output file (CSV or JSON)', default='audio_features.csv')
    parser.add_argument('-f', '--force', action='store_true', help='Overwrite existing output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Process the directory
    df = process_directory(args.input_dir, args.output, args.force)
    
    if not df.empty:
        print(f"\nExtracted features for {len(df)} audio files.")
        print("Available features:", ", ".join(df.columns.tolist()))
    else:
        print("No features were extracted. Check the log for errors.")
        exit(1)
