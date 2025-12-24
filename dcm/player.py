
import os
import pygame
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicPlayer:
    """A music player that handles audio playback using Pygame."""
    
    def __init__(self):
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # State properties
        self.current_song = None
        self.is_playing = False
        self.volume = 1.0
        self.playlist = []
        self.current_index = -1
        self._duration = 0.0 # Cached duration
        
    @property
    def duration(self):
        """Get the duration of the current song in seconds."""
        # Pygame doesn't give duration directly easily for streams, 
        # so we rely on what was set during load, or 0.
        return self._duration

    @property
    def current_position(self):
        """Get current playback position in seconds."""
        if not self.is_playing and not pygame.mixer.music.get_busy():
            return 0.0
        # get_pos returns milliseconds since play started
        # Note: This resets on pause/unpause in some versions, but usually accurate enough for simple UI
        try:
            pos_ms = pygame.mixer.music.get_pos()
            if pos_ms >= 0:
                return pos_ms / 1000.0
            return 0.0
        except:
            return 0.0
    
    def seek(self, position_seconds):
        """
        Seek to a specific position in the song.
        
        Args:
            position_seconds: Position to seek to in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self.current_song:
            return False
        
        try:
            # Clamp position to valid range
            position_seconds = max(0, min(position_seconds, self._duration))
            
            # For WAV files (our M4A transcodes), we need to rewind first then set_pos
            was_playing = self.is_playing
            
            # Stop and reload the song
            pygame.mixer.music.stop()
            pygame.mixer.music.rewind()
            
            # Set position (for WAV this works after rewind)
            if position_seconds > 0:
                pygame.mixer.music.set_pos(position_seconds)
            
            # Resume playing if it was playing before
            if was_playing:
                pygame.mixer.music.play()
                self.is_playing = True
            
            logger.info(f"Seeked to {position_seconds:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"Error seeking: {e}")
            # Try to resume playback if it was playing
            if was_playing:
                try:
                    pygame.mixer.music.play()
                    self.is_playing = True
                except:
                    pass
            return False

    def load_song(self, file_path):
        """Load a song from file path."""
        if not os.path.exists(file_path):
            logger.error(f"MusicPlayer: File not found: {file_path}")
            return False
        
        # For M4A/AAC files, transcode to WAV first (pygame has issues with these)
        actual_file = file_path
        if file_path.lower().endswith(('.m4a', '.aac', '.mp4')):
            actual_file = self._transcode_to_wav(file_path)
            if actual_file is None:
                logger.error(f"MusicPlayer: Failed to transcode {file_path}")
                return False
            
        try:
            pygame.mixer.music.load(actual_file)
            self.current_song = file_path  # Store original path for display
            
            # Try to get duration using Mutagen as Pygame doesn't provide it reliably
            try:
                from mutagen import File
                audio = File(file_path)
                if audio and audio.info:
                    self._duration = audio.info.length
                else:
                    self._duration = 0.0
            except:
                self._duration = 0.0
                
            self.is_playing = False
            return True
            
        except Exception as e:
            logger.error(f"MusicPlayer: Error loading song: {str(e)}")
            return False
    
    def _transcode_to_wav(self, file_path):
        """Transcode M4A/AAC to temporary WAV file using ffmpeg."""
        import subprocess
        import tempfile
        import hashlib
        
        # Create a deterministic temp file name based on the source path
        path_hash = hashlib.md5(file_path.encode()).hexdigest()[:12]
        temp_dir = tempfile.gettempdir()
        wav_path = os.path.join(temp_dir, f"dcm_player_{path_hash}.wav")
        
        # If already transcoded, reuse it
        if os.path.exists(wav_path):
            logger.debug(f"Using cached WAV: {wav_path}")
            return wav_path
        
        logger.info(f"Transcoding to WAV: {os.path.basename(file_path)}")
        
        try:
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', file_path,
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '2',  # Stereo
                '-v', 'quiet',
                wav_path
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(wav_path):
                return wav_path
            else:
                logger.error(f"ffmpeg failed: {result.stderr.decode()}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timeout")
            return None
        except Exception as e:
            logger.error(f"Transcoding error: {e}")
            return None
    
    def play(self, file_path=None):
        """Play the current or specified song."""
        if file_path and file_path != self.current_song:
            if not self.load_song(file_path):
                return False
        
        if not self.current_song:
            return False
            
        try:
            pygame.mixer.music.play()
            self.is_playing = True
            return True
        except Exception as e:
            logger.error(f"MusicPlayer: Error playing song: {str(e)}")
            return False
    
    def pause(self):
        """Pause the current song."""
        if self.is_playing:
            pygame.mixer.music.pause()
            self.is_playing = False
            return True
        return False

    def unpause(self):
        """Resume the current song."""
        if not self.is_playing and self.current_song:
            pygame.mixer.music.unpause()
            self.is_playing = True
            return True
        return False
    
    def stop(self):
        """Stop playback."""
        pygame.mixer.music.stop()
        self.is_playing = False
        return True
    
    def seek(self, position):
        """Seek to a specific position in the song."""
        try:
            pygame.mixer.music.set_pos(position)
            return True
        except Exception as e:
            logger.error(f"MusicPlayer: Error seeking: {str(e)}")
            return False
    
    # Playlist management
    def next(self):
        """Play the next song in the playlist."""
        if not self.playlist:
            return False
            
        next_index = (self.current_index + 1) % len(self.playlist)
        self.current_index = next_index
        return self.play(self.playlist[self.current_index])
    
    def previous(self):
        """Play the previous song in the playlist."""
        if not self.playlist:
            return False
            
        prev_index = (self.current_index - 1) % len(self.playlist)
        self.current_index = prev_index
        return self.play(self.playlist[self.current_index])
    
    def cleanup(self):
        """Clean up resources."""
        self.stop()
        pygame.mixer.quit()

# Global player instance
player = MusicPlayer()

# Clean up on exit
import atexit
atexit.register(player.cleanup)
