"""
Song recommendation engine for DCM Player.
Uses cosine similarity and KNN to find musically similar songs.
"""

import os
import re
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import joblib
import logging
from tqdm import tqdm

# For rich text formatting
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.panel import Panel
from rich.text import Text
from rich import box

# Initialize console for rich output
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SongRecommender:
    """Song recommendation engine using audio feature similarity."""
    
    def __init__(self, features_path: str = None, model_path: str = None, feature_weights: Dict[str, float] = None):
        """
        Initialize the recommender with features and/or a pre-trained model.
        
        Args:
            features_path: Path to the combined features CSV file
            model_path: Path to a pre-trained model
        """
        self.features_df = None
        self.scaler = StandardScaler()
        self.pca = None
        self.model = None
        self.feature_columns = None
        # Default weights: Boost Timbre (MFCC), reduce Key (Chroma) slightly
        self.feature_weights = feature_weights or {
            'mfcc': 1.2,
            'chroma': 0.8,
            'tempo': 1.0,
            'spectral': 1.0,
            'default': 1.0
        }
        
        # Debug info
        logger.debug(f"Initializing SongRecommender with features_path={features_path}, model_path={model_path}")
        
        try:
            if features_path:
                logger.debug(f"Loading features from: {features_path}")
                self.load_features(features_path)
                logger.debug(f"Features loaded. Shape: {self.features_df.shape if self.features_df is not None else 'None'}")
                if self.features_df is not None and not self.features_df.empty:
                    logger.debug(f"Columns: {self.features_df.columns.tolist()}")
                    logger.debug(f"First file_path: {self.features_df['file_path'].iloc[0] if 'file_path' in self.features_df.columns else 'No file_path column'}")
            
            if model_path:
                logger.debug(f"Loading model from: {model_path}")
                self.load_model(model_path)
                logger.debug(f"Model loaded. Feature columns: {self.feature_columns}")
            
            # Initialize with empty DataFrame if no features loaded
            if self.features_df is None:
                logger.debug("No features loaded, initializing empty DataFrame")
                self.features_df = pd.DataFrame()
            
            logger.debug("SongRecommender initialization complete")

        except Exception as e:
            logger.error(f"Error initializing SongRecommender: {str(e)}")
            logger.debug(f"Features DataFrame: {self.features_df}")
            if hasattr(self, 'features_df') and self.features_df is not None:
                logger.debug(f"DataFrame columns: {self.features_df.columns.tolist()}")
            raise
    
    def load_features(self, filepath: str) -> None:
        """
        Load and preprocess song features from a CSV file.
        
        Args:
            filepath: Path to the CSV file containing song features
        """
        logger.info(f"Loading features from {filepath}")
        
        try:
            # Read the CSV file with explicit handling for file paths
            self.features_df = pd.read_csv(filepath, dtype={'file_path': str, 'file_name': str})
            
            # Debug: Print the columns we found
            logger.debug(f"Columns in features file: {self.features_df.columns.tolist()}")
            
            # Ensure required columns exist
            required_columns = ['file_path']
            for col in required_columns:
                if col not in self.features_df.columns:
                    # Try to find case-insensitive match
                    col_lower = col.lower()
                    matching_cols = [c for c in self.features_df.columns if c.lower() == col_lower]
                    if matching_cols:
                        # Rename the column to our expected name
                        self.features_df.rename(columns={matching_cols[0]: col}, inplace=True)
                        logger.warning(f"Renamed column '{matching_cols[0]}' to '{col}'")
                    else:
                        raise ValueError(f"Missing required column in features file: {col}")
            
            # Ensure file_path is treated as string and clean it
            self.features_df['file_path'] = self.features_df['file_path'].astype(str).str.strip()
            
            # Set the feature columns (all numeric columns except metadata)
            metadata_columns = ['file_path', 'file_name', 'file_extension', 'file_size_mb']
            self.feature_columns = [col for col in self.features_df.columns 
                                  if col not in metadata_columns 
                                  and pd.api.types.is_numeric_dtype(self.features_df[col])]
            
            # Log some debug info
            logger.info(f"Loaded {len(self.features_df)} songs with {len(self.feature_columns)} features each")
            logger.debug(f"First file path: {self.features_df['file_path'].iloc[0] if not self.features_df.empty else 'No data'}")
            
        except Exception as e:
            logger.error(f"Error loading features file: {str(e)}")
            if 'features_df' in locals() and self.features_df is not None:
                logger.error(f"Columns in DataFrame: {self.features_df.columns.tolist()}")
                if not self.features_df.empty:
                    logger.error(f"First row data: {self.features_df.iloc[0].to_dict()}")
            raise
    
    def preprocess_features(self, n_components: float = 0.95) -> np.ndarray:
        """
        Preprocess features with scaling and optional PCA.
        
        Args:
            n_components: Variance ratio to preserve with PCA (0.95 = 95%)
                         Set to None to skip PCA
        """
        if self.features_df is None:
            raise ValueError("No features loaded. Call load_features() first.")
        
        # Scale features first
        logger.info("Scaling features...")
        X = self.scaler.fit_transform(self.features_df[self.feature_columns])
        
        # Apply feature weights to the scaled features
        logger.info("Applying feature weights...")
        
        # We need to map column indices to names since X is a numpy array
        # The order of columns in X matches self.feature_columns
        for i, col in enumerate(self.feature_columns):
            weight = self.feature_weights['default']
            if 'mfcc' in col.lower():
                weight = self.feature_weights['mfcc']
            elif 'chroma' in col.lower():
                weight = self.feature_weights['chroma']
            elif 'tempo' in col.lower():
                weight = self.feature_weights['tempo']
            elif 'spectral' in col.lower() or 'centroid' in col.lower() or 'bandwidth' in col.lower():
                weight = self.feature_weights['spectral']
            
            if weight != 1.0:
                X[:, i] *= weight
        
        # Apply PCA if requested
        if n_components is not None:
            self.pca = PCA(n_components=n_components, random_state=42)
            X = self.pca.fit_transform(X)
            logger.info(f"Reduced to {X.shape[1]} principal components "
                      f"(explains {100 * np.sum(self.pca.explained_variance_ratio_):.1f}% of variance)")
        
        return X
    
    def train_model(self, n_neighbors: int = 5, metric: str = 'cosine', n_components: float = 0.95) -> None:
        """Train the KNN model on the preprocessed features."""
        if self.features_df is None:
            raise ValueError("No features loaded. Call load_features() first.")
        
        X = self.preprocess_features(n_components=n_components)
        
        logger.info(f"Training KNN model with {n_neighbors} neighbors...")
        self.model = NearestNeighbors(
            n_neighbors=min(n_neighbors + 1, len(self.features_df)),  # +1 to exclude self
            metric=metric,
            n_jobs=-1  # Use all available cores
        )
        self.model.fit(X)
        logger.info("Model training complete")
    
    def find_similar_songs(self, song_path: str, n_songs: int = 5) -> pd.DataFrame:
        """
        Find songs similar to the given song.
        
        Args:
            song_path: Path to the query song (can be full path or just filename)
            n_songs: Number of similar songs to return
            
        Returns:
            DataFrame containing similar songs and their similarity scores
        """
        if self.features_df is None or self.model is None:
            raise ValueError("Features and model must be loaded before finding similar songs")
        
        # Ensure file_path is treated as string
        self.features_df['file_path'] = self.features_df['file_path'].astype(str)
        
        # Extract just the filename if a full path was provided
        song_filename = os.path.basename(song_path)
        
        # Try to find the song in our features using different matching strategies
        song_match = None
        
        # 1. Try exact match on full path
        song_match = self.features_df[self.features_df['file_path'] == song_path]
        
        # 2. Try matching just the filename
        if song_match.empty:
            song_match = self.features_df[self.features_df['file_path'].str.endswith(song_filename)]
        
        # 3. Try partial match if filename contains ' - ' (common in music files)
        if song_match.empty and ' - ' in song_filename:
            song_part = song_filename.split(' - ')[0].strip()
            song_match = self.features_df[self.features_df['file_path'].str.contains(song_part, regex=False)]
        
        # 4. Try with just the base filename (without extension)
        if song_match.empty:
            base_name = os.path.splitext(song_filename)[0]
            song_match = self.features_df[self.features_df['file_path'].str.contains(base_name, regex=False)]
        
        if song_match.empty:
            # For debugging: print available paths if we can't find a match
            logger.error(f"Could not find song: {song_path}")
            logger.error(f"Available paths start with: {self.features_df['file_path'].head(3).tolist()}")
            raise ValueError(f"Song not found in database: {song_path}")
        
        song_idx = song_match.index[0]
        
        try:
            # Get the song's features - reuse preprocess_features to ensure consistent weighting/scaling
            # We assume n_components is consistent with how the model was trained (stored in self.pca)
            features = self.preprocess_features(n_components=self.pca.n_components if self.pca else None)
            
            distances, indices = self.model.kneighbors(
                features[song_idx].reshape(1, -1),
                n_neighbors=min(n_songs + 1, len(self.features_df))  # Ensure we don't ask for more than we have
            )
            
            # Get the similar songs (skip the first one as it's the query song itself)
            similar_indices = indices[0][1:]
            similar_distances = 1 - distances[0][1:]  # Convert distance to similarity
            
            # Create result DataFrame
            result = self.features_df.iloc[similar_indices].copy()
            result['similarity_score'] = similar_distances
            
            # Sort by similarity (descending)
            result = result.sort_values('similarity_score', ascending=False)
            
            return result
            
        except Exception as e:
            logger.error(f"Error finding similar songs: {str(e)}")
            logger.error(f"Features shape: {features.shape if 'features' in locals() else 'N/A'}")
            logger.error(f"Song index: {song_idx}")
            logger.error(f"Feature columns: {self.feature_columns}")
            raise
    
    def save_model(self, output_path: str) -> None:
        """Save the trained model and preprocessing objects."""
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
        model_data = {
            'scaler': self.scaler,
            'pca': self.pca,
            'model': self.model,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, output_path)
        logger.info(f"Saved model to {output_path}")
    
    def load_model(self, model_path: str):
        """Load a pre-trained model from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model_data = joblib.load(model_path)
        self.model = model_data.get('model')
        self.scaler = model_data.get('scaler', StandardScaler())
        self.pca = model_data.get('pca')
        self.feature_columns = model_data.get('feature_columns', [])
        
        if self.model is None or self.pca is None:
            raise ValueError("Invalid model file: missing required components")
            
        logger.info(f"Loaded model from {model_path}")
        return self

def parse_metadata(file_path: str) -> dict:
    """Extract metadata from file path."""
    # Try to extract artist and title from filename
    filename = os.path.basename(file_path)
    name = os.path.splitext(filename)[0]
    
    # Common patterns in filenames
    patterns = [
        r'(?P<artist>.*?)\s*-\s*(?P<title>.*?)(?:\s*\(|\s*\[|\s*-\s*\d|$)',
        r'(?P<track_num>\d+)[\s.-]*(?P<artist>.*?)\s*-\s*(?P<title>.*?)(?:\s*\(|\s*\[|$)',
    ]
    
    metadata = {
        'filename': filename,
        'title': name,
        'artist': 'Unknown Artist',
        'duration': '--:--',
        'similarity': 0.0
    }
    
    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            metadata.update({k: v.strip() for k, v in match.groupdict().items() if v})
            break
    
    return metadata

def format_duration(seconds: float) -> str:
    """Convert duration in seconds to MM:SS format."""
    if pd.isna(seconds):
        return '--:--'
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}:{seconds:02d}"

def print_similar_songs(query_song: str, similar_songs: pd.DataFrame, features_df: pd.DataFrame):
    """Print similar songs in a formatted table.
    
    Args:
        query_song: Path to the query song
        similar_songs: DataFrame containing similar songs and their similarity scores
        features_df: DataFrame containing all song features
    """
    if features_df is None or features_df.empty:
        console.print("[red]Error:[/] No song features loaded")
        return
        
    if similar_songs is None or similar_songs.empty:
        console.print("[yellow]Warning:[/] No similar songs found")
        return
        
    # Get query song info
    query_info = parse_metadata(query_song)
    
    # Create table
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("#", style="dim", width=4)
    table.add_column("Title")
    table.add_column("Artist")
    table.add_column("Duration", justify="right")
    table.add_column("Similarity", justify="right")
    
    try:
        # Debug: Print available columns
        console.print(f"[dim]Available columns: {features_df.columns.tolist()}")
        
        # Find query song in features
        query_filename = os.path.basename(query_song)
        
        # Check if file_path column exists and has string values
        if 'file_path' not in features_df.columns:
            raise KeyError("'file_path' column not found in features data")
            
        # Ensure we're working with strings
        features_df['file_path'] = features_df['file_path'].astype(str)
        
        # Try exact match first
        query_match = features_df[features_df['file_path'].str.endswith(query_filename, na=False)]
        
        # If no exact match, try partial match
        if query_match.empty and ' - ' in query_filename:
            song_part = query_filename.split(' - ')[0].strip()
            query_match = features_df[features_df['file_path'].str.contains(song_part, na=False, regex=False)]
        
        # If still no match, try with just the base filename
        if query_match.empty:
            base_name = os.path.splitext(query_filename)[0]
            query_match = features_df[features_df['file_path'].str.contains(base_name, na=False, regex=False)]
        
        # Get duration if we found a match
        if not query_match.empty:
            query_duration = query_match['duration'].iloc[0]
            console.print(f"[green]✓[/] Found query song in features")
        else:
            query_duration = 0
            console.print(f"[yellow]Warning:[/] Could not find query song in features")
            console.print(f"[dim]Searched for: {query_filename}")
            
            # Print first few file paths for debugging
            console.print("\n[dim]First few file paths in features:")
            for path in features_df['file_path'].head(3):
                console.print(f"- {path}")
        
        # Add query song
        table.add_row(
            "🔍",
            Text(query_info['title'], style="bold cyan"),
            query_info['artist'],
            format_duration(query_duration),
            "100%",
            style="on dark_blue"
        )
        
        # Add similar songs
        for i, (_, row) in enumerate(similar_songs.iterrows(), 1):
            try:
                song_path = row.get('file_path', '')
                if pd.isna(song_path) or not song_path:
                    continue
                    
                song_info = parse_metadata(song_path)
                similarity = float(row.get('similarity_score', 0))
                
                # Color based on similarity
                if similarity > 0.7:
                    style = "green"
                elif similarity > 0.5:
                    style = "yellow"
                else:
                    style = "red"
                
                # Get duration if available
                duration = row.get('duration', 0)
                if pd.isna(duration):
                    duration = 0
                
                table.add_row(
                    str(i),
                    song_info.get('title', 'Unknown Title')[:50],
                    song_info.get('artist', 'Unknown Artist')[:30],
                    format_duration(duration),
                    f"{similarity*100:.0f}%",
                    style=style
                )
            except Exception as e:
                console.print(f"[yellow]Warning:[/] Could not process song {i}: {str(e)}")
                continue
        
        # Print the results
        console.print("\n" + "🎵" * 20 + "\n", justify="center")
        console.print(Panel.fit("DCM Music Recommender", style="bold blue"))
        console.print(f"[dim]Found {len(similar_songs)} similar songs\n")
        console.print(table)
        console.print("\n[dim]Based on audio analysis of rhythm, melody, and tone")
        console.print("🎵" * 20 + "\n")
        
    except Exception as e:
        console.print(f"[red]Error:[/] Could not display results: {str(e)}")
        console.print("\n[dim]Debug Info:")
        console.print(f"- Query song: {query_song}")
        console.print(f"- Features columns: {features_df.columns.tolist() if not features_df.empty else 'Empty'}")
        console.print(f"- Similar songs columns: {similar_songs.columns.tolist() if not similar_songs.empty else 'Empty'}")

def main():
    """Main entry point for the recommendation system."""
    import argparse
    
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description="🎵 DCM Music Recommender - Find similar songs based on audio features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "features_file",
        help="Path to the combined features CSV file"
    )
    parser.add_argument(
        "query_song",
        help="Path to the query song (must be in the features file)"
    )
    
    # Optional arguments
    parser.add_argument(
        "-n", "--num-songs",
        type=int,
        default=5,
        help="Number of similar songs to return"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file for results (CSV)"
    )
    parser.add_argument(
        "--save-model",
        help="Path to save the trained model"
    )
    parser.add_argument(
        "--load-model",
        help="Path to load a pre-trained model"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True  # Override any existing handlers
        )
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Initialize recommender with progress
    with console.status("[bold green]Initializing...") as status:
        try:
            # Always load features file, even with a pre-trained model
            recommender = SongRecommender(
                features_path=args.features_file,  # Always load features
                model_path=args.load_model
            )
            
            # If we have a model but no features, that's an error
            if args.load_model and recommender.features_df is None:
                raise ValueError("Features file is required even with a pre-trained model")
                
        except Exception as e:
            logger.error(f"Failed to initialize SongRecommender: {str(e)}")
            if args.debug:
                logger.exception("Detailed traceback:")
            raise
    
    # Train or load model with progress
    if args.load_model:
        console.print(f"[green]✓[/] Using pre-trained model from {args.load_model}")
    else:
        with console.status("[bold green]Training recommendation model..."):
            recommender.train_model(n_neighbors=args.num_songs)
            if args.save_model:
                recommender.save_model(args.save_model)
                console.print(f"[green]✓[/] Model saved to {args.save_model}")
    
    # Find similar songs
    try:
        with console.status("[bold green]Finding similar songs..."):
            similar_songs = recommender.find_similar_songs(
                song_path=args.query_song,
                n_songs=args.num_songs
            )
        
        # Print results
        print_similar_songs(args.query_song, similar_songs, recommender.features_df)
        
        # Save to file if requested
        if args.output:
            similar_songs.to_csv(args.output, index=False)
            console.print(f"[green]✓[/] Results saved to {args.output}")
            
    except Exception as e:
        console.print(f"[red]Error:[/] {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
