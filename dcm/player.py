
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
        if not self.is_playing:
            return 0.0
        # get_pos returns milliseconds since play started
        # Note: This resets on pause/unpause in some versions, but usually accurate enough for simple UI
        try:
            return pygame.mixer.music.get_pos() / 1000.0
        except:
            return 0.0

    def load_song(self, file_path):
        """Load a song from file path."""
        if not os.path.exists(file_path):
            logger.error(f"MusicPlayer: File not found: {file_path}")
            return False
            
        try:
            pygame.mixer.music.load(file_path)
            self.current_song = file_path
            
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
