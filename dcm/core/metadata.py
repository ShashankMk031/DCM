
import os
import hashlib
from pathlib import Path
from typing import Optional, Tuple
import logging

from mutagen import File
from mutagen.id3 import ID3, APIC
from mutagen.flac import Picture
from mutagen.mp4 import MP4

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / '.dcm' / 'cache' / 'album_art'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_album_art(file_path: str) -> Optional[str]:
    """
    Extract album art from an audio file and save it to a cache directory.
    return: Path to the cached image file, or None if no art found.
    """
    if not os.path.exists(file_path):
        return None

    try:
        # Generate a unique filename for the cached image based on file path
        # Using hash of file path to ensure unique but consistent mapping
        file_hash = hashlib.md5(file_path.encode('utf-8')).hexdigest()
        cache_path = CACHE_DIR / f"{file_hash}.jpg"
        
        # Return cached path if it exists
        if cache_path.exists():
            return str(cache_path)

        audio = File(file_path)
        if not audio:
            return None

        art_data = None
        
        # Handle MP3 (ID3)
        if isinstance(audio.tags, ID3):
            for tag in audio.tags.values():
                if isinstance(tag, APIC):
                    art_data = tag.data
                    break
        
        # Handle FLAC
        elif hasattr(audio, 'pictures'):
            if audio.pictures:
                art_data = audio.pictures[0].data
                
        # Handle MP4/M4A
        elif isinstance(audio, MP4):
            if 'covr' in audio:
                art_data = audio['covr'][0]

        if art_data:
            with open(cache_path, 'wb') as img_f:
                img_f.write(art_data)
            return str(cache_path)
            
    except Exception as e:
        logger.warning(f"Failed to extract album art from {file_path}: {e}")
        
    return None

def get_metadata(file_path: str) -> dict:
    """Extract basic metadata (Artist, Title, Album)"""
    meta = {'title': 'Unknown Title', 'artist': 'Unknown Artist', 'album': 'Unknown Album'}
    
    try:
        if not os.path.exists(file_path):
            return meta
            
        audio = File(file_path, easy=True)
        if audio:
            meta['title'] = audio.get('title', [os.path.basename(file_path)])[0]
            meta['artist'] = audio.get('artist', ['Unknown Artist'])[0]
            meta['album'] = audio.get('album', ['Unknown Album'])[0]
    except Exception as e:
        logger.warning(f"Failed to extract metadata from {file_path}: {e}")
        
    return meta
