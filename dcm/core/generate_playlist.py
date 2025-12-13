"""DCM Playlist Generator - Fixed Version"""

import os
import logging
import random
from pathlib import Path
from typing import List, Optional, Dict, Union, Callable

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PlaylistGenerator:
    """Generates playlists based on audio features and similarity."""
    
    def __init__(self, features_file: str):
        """Initialize with path to features CSV file."""
        self.features_df = None
        self.load_features(features_file)
    
    def load_features(self, features_file: str) -> None:
        """Load and validate features from CSV file."""
        try:
            if not os.path.exists(features_file):
                raise FileNotFoundError(f"Features file not found: {features_file}")
            
            self.features_df = pd.read_csv(features_file)
            
            if self.features_df.empty:
                raise ValueError("Features file is empty")
                
            required_columns = ['file_path', 'tempo', 'energy', 'danceability']
            for col in required_columns:
                if col not in self.features_df.columns:
                    raise ValueError(f"Missing required column: {col}")
                    
            logger.info(f"Successfully loaded {len(self.features_df)} songs")
            
        except Exception as e:
            logger.error(f"Error loading features: {str(e)}")
            raise
    
    def generate_playlist(
        self,
        song_paths: List[str],
        output_file: str,
        playlist_name: str = "My Playlist",
        mood: Optional[str] = None,
        max_songs: int = 20,
        shuffle: bool = True,
        progress_callback: Optional[Callable[[float, str], bool]] = None,
        dynamic: bool = True,
        genre: Optional[str] = None
    ) -> bool:
        """Generate a playlist based on input songs and optional filters."""
        def update_progress(progress: float, status: str) -> bool:
            if progress_callback and not progress_callback(progress, status):
                logger.info("Playlist generation cancelled by user")
                return False
            return True
        
        try:
            # Validate input
            if not song_paths:
                logger.error("No input songs provided")
                return False
                
            if not update_progress(0.0, "Starting playlist generation..."):
                return False
            
            # Convert to absolute paths
            abs_paths = [str(Path(p).resolve()) for p in song_paths]
            
            # Filter valid songs
            valid_songs = [p for p in abs_paths 
                         if p in self.features_df['file_path'].values]
            
            if not valid_songs:
                logger.error("No valid songs found in features")
                update_progress(-1, "Error: No valid songs found")
                return False
                
            playlist = valid_songs.copy()
            
            # Add similar songs if needed
            if dynamic and len(playlist) < max_songs:
                if not update_progress(0.3, "Finding similar songs..."):
                    return False
                    
                # Use the last song as reference
                reference_song = playlist[-1]
                remaining = max_songs - len(playlist)
                
                # Find similar songs
                similar = self.find_similar_songs(
                    song_path=reference_song,
                    n=remaining * 2,  # Get extra in case some are already in playlist
                    genre_filter=genre,
                    mood_filter=mood,
                    exclude_paths=playlist
                )
                
                # Add new songs to playlist
                for song in similar:
                    if len(playlist) >= max_songs:
                        break
                    if song not in playlist:
                        playlist.append(song)
            
            # Shuffle if requested
            if shuffle and len(playlist) > 1:
                if not update_progress(0.9, "Shuffling playlist..."):
                    return False
                first = playlist[0]
                rest = playlist[1:]
                random.shuffle(rest)
                playlist = [first] + rest
            
            # Save playlist
            if not update_progress(0.95, "Saving playlist..."):
                return False
                
            self.save_playlist(playlist, output_file, playlist_name)
            update_progress(1.0, "Playlist generated successfully!")
            return True
            
        except Exception as e:
            error_msg = f"Error generating playlist: {str(e)}"
            logger.error(error_msg, exc_info=True)
            update_progress(-1, f"Error: {error_msg[:50]}...")
            return False
    
    def find_similar_songs(
        self,
        song_path: str,
        n: int = 10,
        genre_filter: Optional[str] = None,
        mood_filter: Optional[str] = None,
        exclude_paths: Optional[List[str]] = None
    ) -> List[str]:
        """Find songs similar to the given song."""
        try:
            # Get feature vectors
            features = self.get_feature_vectors()
            
            # Find index of reference song
            ref_idx = self.features_df.index[self.features_df['file_path'] == song_path].tolist()
            if not ref_idx:
                logger.warning(f"Song not found in features: {song_path}")
                return []
                
            ref_idx = ref_idx[0]
            ref_features = features[ref_idx].reshape(1, -1)
            
            # Calculate similarities
            similarities = cosine_similarity(ref_features, features).flatten()
            
            # Create results dataframe
            results = self.features_df.copy()
            results['similarity'] = similarities
            
            # Apply filters
            if genre_filter:
                results = results[results['genre'].str.lower() == genre_filter.lower()]
                
            if mood_filter:
                results = results[results['mood'].str.lower() == mood_filter.lower()]
                
            if exclude_paths:
                results = results[~results['file_path'].isin(exclude_paths)]
            
            # Sort by similarity and get top N
            results = results.sort_values('similarity', ascending=False).head(n)
            
            return results['file_path'].tolist()
            
        except Exception as e:
            logger.error(f"Error finding similar songs: {str(e)}", exc_info=True)
            return []
    
    def get_feature_vectors(self) -> np.ndarray:
        """Get normalized feature vectors."""
        feature_cols = [c for c in self.features_df.columns 
                       if c not in ['file_path', 'genre', 'mood'] 
                       and pd.api.types.is_numeric_dtype(self.features_df[c])]
        
        features = self.features_df[feature_cols].values
        return StandardScaler().fit_transform(features)
    
    def save_playlist(
        self, 
        song_paths: List[str], 
        output_file: str, 
        playlist_name: str = "My Playlist"
    ) -> bool:
        """Save playlist to file in M3U format."""
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"#EXTM3U\n")
                f.write(f"#PLAYLIST:{playlist_name}\n")
                
                for path in song_paths:
                    f.write(f"#EXTINF:0,{os.path.basename(path)}\n")
                    f.write(f"{path}\n")
            
            logger.info(f"Playlist saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving playlist: {str(e)}")
            return False

def main():
    """Command-line interface for playlist generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate playlists based on audio features')
    parser.add_argument('songs', nargs='+', help='Input song paths')
    parser.add_argument('-o', '--output', required=True, help='Output playlist file')
    parser.add_argument('--name', default='My Playlist', help='Playlist name')
    parser.add_argument('--mood', help='Filter by mood')
    parser.add_argument('--genre', help='Filter by genre')
    parser.add_argument('--max-songs', type=int, default=20, help='Maximum songs in playlist')
    parser.add_argument('--no-shuffle', action='store_false', dest='shuffle', 
                       help='Disable shuffling')
    parser.add_argument('--features', required=True, help='Path to features CSV file')
    
    args = parser.parse_args()
    
    try:
        generator = PlaylistGenerator(args.features)
        success = generator.generate_playlist(
            song_paths=args.songs,
            output_file=args.output,
            playlist_name=args.name,
            mood=args.mood,
            genre=args.genre,
            max_songs=args.max_songs,
            shuffle=args.shuffle
        )
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
