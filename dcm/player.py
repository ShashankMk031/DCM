import os
from kivy.core.audio import SoundLoader
from kivy.clock import Clock
from kivy.properties import (
    ObjectProperty, NumericProperty, StringProperty, BooleanProperty, ListProperty
)
from kivy.event import EventDispatcher
from kivy.logger import Logger

class MusicPlayer(EventDispatcher):
    """A music player that handles audio playback."""
    
    # Properties
    current_song = ObjectProperty(None, allownone=True)
    current_position = NumericProperty(0.0)
    duration = NumericProperty(0.0)
    volume = NumericProperty(1.0)
    is_playing = BooleanProperty(False)
    playlist = ListProperty([])
    current_index = NumericProperty(-1)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sound = None
        self._update_ev = None
        self.on_song_end_callback = None  # Callback for when a song ends
        self.bind(volume=self._on_volume)
    
    def _on_volume(self, instance, value):
        """Update volume when the volume property changes."""
        if self.sound:
            self.sound.volume = value
    
    def _update(self, dt):
        """Update the current position of the song."""
        if self.sound and self.is_playing:
            self.current_position = self.sound.get_pos()
            
            # Check if song has ended
            if self.sound.state == 'stop' and self.current_position > 0 and \
               abs(self.duration - self.current_position) < 0.1:
                # Store callback before calling next() as it might trigger a new song load
                callback = self.on_song_end_callback
                self.next()
                # If we have a callback and the song ended naturally, call it
                if callback and self.current_position > 0:
                    callback()
    
    def load_song(self, file_path):
        """Load a song from file path."""
        if not os.path.exists(file_path):
            Logger.error(f"MusicPlayer: File not found: {file_path}")
            return False
            
        # Stop current song if playing
        self.stop()
        
        try:
            self.sound = SoundLoader.load(file_path)
            if not self.sound:
                Logger.error(f"MusicPlayer: Could not load file: {file_path}")
                return False
                
            self.current_song = file_path
            self.duration = self.sound.length
            self.current_position = 0.0
            self.is_playing = False
            
            # Set volume
            self.sound.volume = self.volume
            
            # Start update clock if not already running
            if not self._update_ev:
                self._update_ev = Clock.schedule_interval(self._update, 0.1)
                
            return True
            
        except Exception as e:
            Logger.error(f"MusicPlayer: Error loading song: {str(e)}")
            return False
    
    def play(self, file_path=None):
        """Play the current or specified song."""
        if file_path and file_path != self.current_song:
            if not self.load_song(file_path):
                return False
        
        if not self.sound:
            return False
            
        if self.sound.state == 'play':
            return True
            
        try:
            self.sound.play()
            self.is_playing = True
            return True
        except Exception as e:
            Logger.error(f"MusicPlayer: Error playing song: {str(e)}")
            return False
    
    def pause(self):
        """Pause the current song."""
        if self.sound and self.sound.state == 'play':
            self.sound.stop()
            self.is_playing = False
            return True
        return False
    
    def stop(self):
        """Stop playback and reset position."""
        if self.sound:
            self.sound.stop()
            self.sound.unload()
            self.sound = None
            
        self.current_position = 0.0
        self.duration = 0.0
        self.is_playing = False
        
        # Don't unschedule the update event here, as we might load a new song
        
        return True
    
    def seek(self, position):
        """Seek to a specific position in the song."""
        if self.sound and 0 <= position <= self.duration:
            try:
                self.sound.seek(position)
                self.current_position = position
                return True
            except Exception as e:
                Logger.error(f"MusicPlayer: Error seeking: {str(e)}")
        return False
    
    # Playlist management
    def set_playlist(self, song_paths, start_index=0):
        """Set the playlist and optionally start playing from start_index."""
        if not song_paths:
            return False
            
        self.playlist = song_paths
        self.current_index = max(0, min(start_index, len(song_paths) - 1))
        
        # Load the first song
        return self.load_song(self.playlist[self.current_index])
    
    def next(self):
        """Play the next song in the playlist."""
        if not self.playlist:
            return False
            
        next_index = (self.current_index + 1) % len(self.playlist)
        if next_index == self.current_index:
            return False  # Only one song in playlist
            
        self.current_index = next_index
        return self.load_song(self.playlist[self.current_index]) and self.play()
    
    def previous(self):
        """Play the previous song in the playlist."""
        if not self.playlist:
            return False
            
        prev_index = (self.current_index - 1) % len(self.playlist)
        if prev_index == self.current_index:
            return False  # Only one song in playlist
            
        self.current_index = prev_index
        return self.load_song(self.playlist[self.current_index]) and self.play()
    
    def clear_playlist(self):
        """Clear the current playlist."""
        self.stop()
        self.playlist = []
        self.current_index = -1
    
    def cleanup(self):
        """Clean up resources."""
        self.stop()
        if self._update_ev:
            self._update_ev.cancel()
            self._update_ev = None

# Global player instance
player = MusicPlayer()

# Clean up on exit
import atexit
atexit.register(player.cleanup)
