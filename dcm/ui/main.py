import os
import sys
import platform
import random
from pathlib import Path
from functools import partial

# Kivy imports
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import (
    ListProperty, StringProperty, ObjectProperty, 
    NumericProperty, BooleanProperty, DictProperty
)
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout

# KivyMD imports
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.screen import MDScreen
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.uix.button import MDRaisedButton, MDFloatingActionButton, MDIconButton, MDFlatButton
from kivymd.uix.label import MDLabel


class MainScreen(MDScreen):
    """Main application screen that contains the screen manager and navigation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'main_screen'
from kivymd.uix.list import (
    MDList, OneLineIconListItem, TwoLineListItem, 
    ThreeLineListItem, IconLeftWidget, ImageLeftWidget
)
from kivymd.uix.dialog import MDDialog
from kivymd.uix.progressbar import MDProgressBar
from kivy.uix.filechooser import FileChooserIconView, FileChooserListLayout, FileChooserController, FileChooserLayout, FileChooserListView
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivymd.uix.selectioncontrol import MDCheckbox
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.label import MDLabel
from kivymd.uix.filemanager import MDFileManager

# Import our modules
from dcm.database import db
from dcm.player import player as music_player

# Add the project root to the path to import the core models 
# Add the project root to the path to impor thte core models 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) 
if project_root not in sys.path:
    sys.path.append(project_root) 

class FileChoosePopup(MDBoxLayout):
    """Popup for file selection using KivyMD components."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint = (0.9, 0.9)
        self.spacing = '12dp'
        self.padding = '12dp'
        
class DCMApp(MDApp):
    selected_files = ListProperty([]) 
    playlist_name = StringProperty('')
    
    # Player state properties
    current_song = StringProperty('No song selected')
    current_artist = StringProperty('')
    current_album = StringProperty('')
    is_playing = BooleanProperty(False)
    current_time = NumericProperty(0)
    total_time = NumericProperty(0)
    volume = NumericProperty(0.8)  # Default volume (0.0 to 1.0)
    
    def __init__(self, **kwargs):
        # Initialize the app first
        super(DCMApp, self).__init__(**kwargs)
        
        # Theme configuration for KivyMD 1.1.1
        # Set theme style - 'Light' or 'Dark'
        self.theme_cls.theme_style = "Light"
        
        # Set primary color palette - choose from:
        # 'Red', 'Pink', 'Purple', 'DeepPurple', 'Indigo', 'Blue', 'LightBlue',
        # 'Cyan', 'Teal', 'Green', 'LightGreen', 'Lime', 'Yellow', 'Amber',
        # 'Orange', 'DeepOrange', 'Brown', 'Gray', 'BlueGray'
        self.theme_cls.primary_palette = "Blue"
        
        # Set primary color hue - '50' to '900' or 'A100', 'A200', 'A400', 'A700'
        self.theme_cls.primary_hue = "500"  # Standard blue
        
        # Set accent color palette (for buttons, sliders, etc.)
        self.theme_cls.accent_palette = "Blue"
        
        # Set accent color hue
        self.theme_cls.accent_hue = "A200"  # Brighter blue for accents
        
        # Disable theme switching animation to prevent potential issues
        self.theme_cls.theme_style_switch_animation = False
        
        # Initialize player state
        self.selected_song = None
        self.current_playlist = []
        self.current_index = -1
        
        # Setup player callbacks
        music_player.bind(
            current_position=self.update_time_display,
            duration=self.update_duration,
            is_playing=self.update_play_button_state
        )
        # Set up song end callback
        music_player.on_song_end_callback = self.on_song_ended
        
        # Set initial volume
        music_player.volume = self.volume

    def build(self):
        """Build the application"""
        # Set window background color
        from kivy.utils import get_color_from_hex
        Window.clearcolor = get_color_from_hex('#f5f5f5')  # Light gray background
        
        # Load the KV file
        from kivy.lang import Builder
        kv_path = os.path.join(os.path.dirname(__file__), 'main.kv')
        if os.path.exists(kv_path):
            Builder.load_file(kv_path)
        else:
            print(f"Error: KV file not found at {kv_path}")
            
        # Create and return the main screen
        return MainScreen()

    def exit_manager(self, *args):
        """Called when the file manager is closed"""
        self.file_manager.close()

    def select_path(self, path):
        """Handle the selected file path"""
        self.exit_manager()
        if path:
            if isinstance(path, list):
                self.selected_files.extend(path)
            else:
                self.selected_files.append(path)
            self.show_snackbar(f"Selected {len(self.selected_files)} files")

    def switch_screen(self, screen_name):
        """
        Switch between different screens and update the active button state
        
        Args:
            screen_name (str): Name of the screen to switch to
            
        Returns:
            bool: True if screen switch was successful, False otherwise
        """
        try:
            print(f"Attempting to switch to screen: {screen_name}")
            
            # Get the screen manager and switch to the requested screen
            if hasattr(self, 'root') and hasattr(self.root, 'ids'):
                screen_manager = self.root.ids.get('screen_manager')
                if screen_manager:
                    print(f"Found screen manager. Current screen: {screen_manager.current}")
                    
                    # Check if the screen exists, if not create a basic one
                    if not screen_manager.has_screen(screen_name):
                        print(f"Screen '{screen_name}' not found, creating a new one")
                        from kivymd.uix.screen import MDScreen
                        screen = MDScreen(name=screen_name)
                        screen_manager.add_widget(screen)
                    
                    # Set the current screen
                    screen_manager.current = screen_name
                    print(f"Switched to screen: {screen_name}")
                    
                    # Update navigation button states
                    self.update_nav_buttons(screen_name)
                    
                    # Additional screen-specific logic
                    try:
                        if screen_name == 'playlists_screen' and hasattr(self, 'refresh_playlists'):
                            self.refresh_playlists()
                        elif screen_name == 'library_screen' and hasattr(self, 'load_library'):
                            self.load_library()
                        elif screen_name == 'recommend_screen' and hasattr(self, 'prepare_recommendations'):
                            self.prepare_recommendations()
                        elif screen_name == 'generate_screen' and hasattr(self, 'prepare_generation'):
                            self.prepare_generation()
                        elif screen_name == 'settings_screen' and hasattr(self, 'load_settings'):
                            self.load_settings()
                    except Exception as e:
                        print(f"Error in screen-specific logic for '{screen_name}': {str(e)}")
                    
                    return True
                else:
                    print("Screen manager not found in root.ids")
            else:
                print("Root widget or ids not available")
                
            return False
        except Exception as e:
            print(f"Error switching to screen '{screen_name}': {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    def on_start(self):
        """Called when the application is starting up"""
        print("DCM Playlist Generator started")
        
        def set_initial_screen(dt):
            """Set the initial screen after the UI is fully loaded"""
            try:
                # Access the screen manager through the root widget
                if hasattr(self, 'root') and hasattr(self.root, 'ids'):
                    if 'screen_manager' in self.root.ids:
                        screen_manager = self.root.ids.screen_manager
                        if not screen_manager.has_screen('recommend_screen'):
                            # If recommend_screen doesn't exist, create and add it
                            from kivymd.uix.screen import MDScreen
                            screen = MDScreen(name='recommend_screen')
                            screen_manager.add_widget(screen)
                        
                        screen_manager.current = 'recommend_screen'
                        # Update navigation button states
                        self.update_nav_buttons('recommend_screen')
                        return
                
                print("Warning: Could not find or access screen manager")
                
            except Exception as e:
                print(f"Error in set_initial_screen: {str(e)}")
        
        # Schedule the screen change to happen after the UI is fully loaded
        Clock.schedule_once(set_initial_screen, 0.5)  # Slightly longer delay to ensure UI is ready
        
    def update_nav_buttons(self, active_screen):
        """
        Update the navigation buttons to show which screen is active
        
        Args:
            active_screen (str): Name of the currently active screen
        """
        try:
            print(f"Updating navigation buttons for screen: {active_screen}")
            if not hasattr(self, 'root') or not hasattr(self.root, 'ids'):
                print("Cannot update buttons: root or ids not available")
                return
                
            # List of all navigation buttons
            nav_buttons = {
                'playlists_btn': 'playlists_screen',
                'library_btn': 'library_screen',
                'recommend_btn': 'recommend_screen',
                'generate_btn': 'generate_screen'
            }
            
            # Debug: Print available IDs
            print(f"Available IDs in root: {list(self.root.ids.keys())}")
            
            # Update button states
            for btn_id, screen_name in nav_buttons.items():
                if btn_id in self.root.ids:
                    btn = self.root.ids[btn_id]
                    if screen_name == active_screen:
                        # Active button
                        print(f"Setting {btn_id} as active")
                        btn.md_bg_color = self.theme_cls.primary_light
                        btn.text_color = (1, 1, 1, 1)
                    else:
                        # Inactive button
                        print(f"Setting {btn_id} as inactive")
                        btn.md_bg_color = self.theme_cls.primary_color
                        btn.text_color = (1, 1, 1, 0.8)
                else:
                    print(f"Button {btn_id} not found in root.ids")
                        
        except Exception as e:
            print(f"Error updating navigation buttons: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def generate_playlist(self):
        """Handle the generate playlist button click
        
        This method handles both song-based recommendations and genre/mood-based
        playlist generation using the PlaylistGenerator class, with enhanced
        error handling and user feedback.
        """
        from dcm.ui import LoadingDialog
        from kivy.clock import Clock
        import os
        from datetime import datetime
        
        print("Generate playlist button clicked")
        
        def update_loading(message: str = None, progress: float = None, max_steps: float = None):
            """Helper to update the loading dialog"""
            if hasattr(self, 'loading_dialog') and self.loading_dialog:
                if message is not None:
                    self.loading_dialog.status_text = message
                if progress is not None:
                    self.loading_dialog.progress = progress
                if max_steps is not None:
                    self.loading_dialog.max_progress = max_steps
        
        def show_error(title: str, message: str, dismiss_loading: bool = True):
            """Helper to show an error dialog and log the error"""
            print(f"Error: {title} - {message}")
            if dismiss_loading and hasattr(self, 'loading_dialog') and self.loading_dialog:
                self.loading_dialog.dismiss()
            self.show_error_dialog(title, message)
        
        def cancel_operation():
            """Handle cancellation of the current operation"""
            self.operation_cancelled = True
            if hasattr(self, 'loading_dialog') and self.loading_dialog:
                self.loading_dialog.dismiss()
            self.show_snackbar("Operation cancelled")
        
        # Show loading dialog with progress bar
        self.loading_dialog = LoadingDialog(
            title="Generating Playlist",
            status_text="Initializing...",
            progress=0,
            max_progress=10,  # Will be updated based on operation
            cancel_button_text="CANCEL"
        )
        self.loading_dialog.set_cancel_callback(cancel_operation)
        self.loading_dialog.open()
        
        # Track operation state
        self.current_operation = "playlist_generation"
        self.operation_cancelled = False
        
        # Use Clock to run the generation asynchronously
        def generate_async(dt):
            try:
                # Check if operation was cancelled
                if self.operation_cancelled:
                    return
                    
                # Check if UI is properly initialized
                if not hasattr(self, 'root') or 'screen_manager' not in self.root.ids:
                    show_error("UI Error", "Application UI is not properly initialized.")
                    return
                
                current_screen = self.root.ids.screen_manager.current
                
                if current_screen == 'recommend_screen':
                    # Song-based recommendations (Spotify-like)
                    update_loading("Analyzing selected song...", 1, 5)
                    
                    if not self.selected_song:
                        show_error("No Song Selected", "Please select a song to get recommendations.")
                        return
                    
                    # Initialize PlaylistGenerator with features file
                    features_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'song_features.csv')
                    if not os.path.exists(features_file):
                        show_error("Missing Data", "Could not find song features database. Please ensure the application is properly installed.")
                        return
                    
                    try:
                        update_loading("Loading song database...", 2)
                        generator = PlaylistGenerator(features_file)
                        
                        update_loading("Finding similar songs...", 3)
                        recommended_songs = generator.find_similar_songs(
                            song_path=self.selected_song,
                            n=5,
                            mood_filter=self.get_selected_mood()
                        )
                        
                        if recommended_songs:
                            # Convert file paths to song dictionaries
                            self.recommended_songs = []
                            for i, song in enumerate(recommended_songs, 1):
                                if self.operation_cancelled:
                                    return
                                try:
                                    self.recommended_songs.append({
                                        'title': os.path.splitext(os.path.basename(song))[0],
                                        'file_path': song,
                                        'artist': "Unknown Artist",
                                        'album': "Unknown Album"
                                    })
                                    update_loading(f"Processing recommendation {i}/{len(recommended_songs)}...", 3 + (i/len(recommended_songs)))
                                except Exception as e:
                                    print(f"Warning: Could not process song {song}: {str(e)}")
                            
                            if self.recommended_songs:
                                update_loading("Updating recommendations...", 4)
                                self.update_recommendations_ui()
                                self.switch_screen('recommend_screen')
                                self.show_snackbar(f"Found {len(self.recommended_songs)} similar songs")
                            else:
                                show_error("No Valid Songs", "Could not process any of the recommended songs.")
                        else:
                            update_loading("No similar songs found, generating alternatives...", 3)
                            self.simulate_recommendations()
                            
                    except Exception as e:
                        error_msg = f"Error generating recommendations: {str(e)}"
                        print(error_msg)
                        show_error("Generation Error", "Failed to generate song recommendations. Please try again.")
                    
                elif current_screen == 'generate_screen':
                    # Genre/mood-based playlist generation
                    update_loading("Preparing playlist generation...", 1, 6)
                    mood = self.get_selected_mood()
                    
                    # Validate selected files
                    if not self.selected_files:
                        show_error("No Files Selected", "Please select at least one music file.")
                        return
                    
                    # Check if files exist
                    missing_files = [f for f in self.selected_files if not os.path.exists(f)]
                    if missing_files:
                        show_error("Missing Files", f"Could not find the following files:\n{', '.join(missing_files[:3])}{'...' if len(missing_files) > 3 else ''}")
                        return
                    
                    # Initialize PlaylistGenerator with features file
                    features_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'song_features.csv')
                    if not os.path.exists(features_file):
                        show_error("Missing Data", "Could not find song features database. Please ensure the application is properly installed.")
                        return
                    
                    try:
                        update_loading("Loading song database...", 2)
                        generator = PlaylistGenerator(features_file)
                        
                        # Create output directory if it doesn't exist
                        output_dir = os.path.join(os.path.expanduser('~'), 'Music', 'DCM_Playlists')
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Generate a unique filename based on timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_file = os.path.join(output_dir, f"playlist_{timestamp}.m3u")
                        
                        playlist_name = f"{mood.capitalize()} Mix - {datetime.now().strftime('%b %d, %Y')}" if mood else f"My Playlist - {datetime.now().strftime('%b %d, %Y')}"
                        
                        update_loading("Analyzing songs...", 3)
                        
                        # Generate playlist with progress updates
                        def progress_callback(progress, status):
                            if self.operation_cancelled:
                                return False
                            update_loading(status, 3 + (progress * 2))  # Scale progress to 3-5 range
                            return True
                        
                        update_loading("Generating playlist...", 3)
                        success = generator.generate_playlist(
                            song_paths=self.selected_files,
                            output_file=output_file,
                            playlist_name=playlist_name,
                            mood=mood,
                            max_songs=20,
                            shuffle=True,
                            progress_callback=progress_callback
                        )
                        
                        if not success or not os.path.exists(output_file):
                            raise Exception("Failed to generate playlist file")
                            
                        try:
                            # Load the generated playlist
                            with open(output_file, 'r', encoding='utf-8') as f:
                                playlist_songs = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                            
                            if not playlist_songs:
                                raise ValueError("Generated playlist is empty")
                            
                            # Update UI with the generated playlist
                            update_loading("Finalizing playlist...", 5)
                            self.recommended_songs = []
                            
                            for i, song in enumerate(playlist_songs, 1):
                                if self.operation_cancelled:
                                    return
                                try:
                                    self.recommended_songs.append({
                                        'title': os.path.splitext(os.path.basename(song))[0],
                                        'file_path': song,
                                        'artist': "Unknown Artist",
                                        'album': "Unknown Album"
                                    })
                                    # Update progress based on processing songs
                                    update_loading(
                                        f"Processing {i}/{len(playlist_songs)} songs...",
                                        5 + (i / len(playlist_songs))
                                    )
                                except Exception as e:
                                    print(f"Warning: Could not process song {song}: {str(e)}")
                            
                            if self.recommended_songs:
                                self.update_recommendations_ui()
                                self.switch_screen('recommend_screen')
                                self.show_snackbar(f"Playlist generated with {len(self.recommended_songs)} songs")
                            else:
                                raise ValueError("No valid songs found in the generated playlist")
                                
                        except Exception as e:
                            error_msg = f"Error processing generated playlist: {str(e)}"
                            print(error_msg)
                            show_error("Playlist Error", "The playlist was generated but could not be loaded. Please try again.")
                            
                    except Exception as e:
                        error_msg = f"Error generating playlist: {str(e)}"
                        print(error_msg)
                        show_error("Generation Error", "Failed to generate playlist. Please try again with different settings.")
                
                else:
                    show_error("Invalid Screen", "Cannot generate playlist from the current screen.")
                
            except Exception as e:
                if not self.operation_cancelled:  # Don't show error if operation was cancelled
                    error_msg = f"Unexpected error: {str(e)}"
                    print(error_msg)
                    show_error("Unexpected Error", "An unexpected error occurred. Please restart the application and try again.")
                
            finally:
                # Close loading dialog if not already closed
                if hasattr(self, 'loading_dialog') and self.loading_dialog:
                    self.loading_dialog.dismiss()
        
        # Schedule the async generation with a small delay to allow UI to update
        Clock.schedule_once(generate_async, 0.1)
        
    def on_playlist_generated(self):
        """Callback when playlist generation is complete
        
        This method is called after a playlist has been successfully generated.
        It updates the UI to show the generated playlist and prepares it for playback.
        """
        try:
            if hasattr(self, 'recommended_songs') and self.recommended_songs:
                # Update the current playlist for playback
                self.current_playlist = [song.get('file_path', '') for song in self.recommended_songs]
                self.current_index = 0
                
                # Update the UI to show the new playlist
                self.update_recommendations_ui()
                
                # Show success message
                self.show_snackbar(f"Generated playlist with {len(self.recommended_songs)} songs")
                
                # Auto-play the first song if there are any
                if self.current_playlist and self.current_playlist[0]:
                    self.play_song(0)
            else:
                self.show_error_dialog("Error", "No songs were added to the playlist.")
                
        except Exception as e:
            error_msg = f"Error finalizing playlist: {str(e)}"
            print(error_msg)
            self.show_error_dialog("Error", "Failed to finalize the playlist.")

    def file_manager_open(self):
        # Create a popup for file selection
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)
        
        # Add instruction label
        instruction = MDLabel(
            text="Select a music file to add to your library",
            size_hint_y=None,
            height='40dp',
            theme_text_color='Primary',
            halign='center',
            valign='middle'
        )
        content.add_widget(instruction)
        
        # Set a default directory that should exist
        default_dir = os.path.expanduser('~')
        music_dir = os.path.join(default_dir, 'Music')
        
        # Use Music directory if it exists, otherwise use home directory
        start_path = music_dir if os.path.exists(music_dir) else default_dir
        
        # Create file chooser
        file_chooser = FileChooserListView(
            path=start_path,
            filters=['*.mp3', '*.wav', '*.ogg', '*.m4a', '*.flac'],
            size_hint=(1, 1)
        )
        content.add_widget(file_chooser)
        
        # Create buttons container
        btn_box = BoxLayout(size_hint_y=None, height='50dp', spacing=10)
        
        # Create buttons
        btn_select = MDRaisedButton(
            text='SELECT',
            size_hint=(0.5, 1),
            md_bg_color=self.theme_cls.primary_color
        )
        
        btn_cancel = MDFlatButton(
            text='CANCEL',
            size_hint=(0.5, 1),
            theme_text_color='Custom',
            text_color=self.theme_cls.primary_color
        )
        
        btn_box.add_widget(btn_select)
        btn_box.add_widget(btn_cancel)
        content.add_widget(btn_box)
        
        # Create and configure popup
        popup = Popup(
            title='Select Music File',
            content=content,
            size_hint=(0.9, 0.9)
        )
        
        def select_file(instance):
            if file_chooser.selection:
                self.selected_song = file_chooser.selection[0]
                song_name = os.path.basename(self.selected_song)
                self.show_snackbar(f"Selected: {song_name}")
            popup.dismiss()
        
        btn_select.bind(on_release=select_file)
        btn_cancel.bind(on_release=popup.dismiss)
        
        # Open the popup
        popup.open()
    
    def simulate_recommendations(self):
        """Simulate getting recommendations for the selected song.
        
        Handles both string (file path) and dictionary (song object) types for self.selected_song.
        Generates diverse recommendations instead of variations of the same song.
        """
        try:
            # Show loading dialog
            loading_dialog = self.show_loading_dialog("Analyzing your song and finding similar tracks...")
            
            # Use Clock to simulate async processing
            def process_recommendations(dt):
                try:
                    # Determine the base song name and path
                    if isinstance(self.selected_song, dict):
                        # Handle case where selected_song is a dictionary
                        song_title = self.selected_song.get('title', 'Unknown Song')
                        song_artist = self.selected_song.get('artist', 'Unknown Artist')
                        song_album = self.selected_song.get('album', 'Unknown Album')
                        song_path = self.selected_song.get('file_path', '')
                    else:
                        # Handle case where selected_song is a file path string
                        song_path = self.selected_song
                        song_title = os.path.splitext(os.path.basename(song_path))[0]
                        song_artist = "Unknown Artist"
                        song_album = "Unknown Album"
                    
                    # Generate diverse recommendations with different artists and albums
                    recommended_songs = []
                    
                    # Sample data for realistic recommendations
                    artists = ["The Beatles", "Taylor Swift", "Ed Sheeran", "Adele", "Coldplay", 
                              "BTS", "Billie Eilish", "Drake", "Ariana Grande", "The Weeknd"]
                    
                    albums = ["Midnight Memories", "Red (Taylor's Version)", "÷ (Divide)", "30", 
                             "Music of the Spheres", "Map of the Soul: 7", "Happier Than Ever",
                             "Certified Lover Boy", "Positions", "After Hours"]
                    
                    genres = ["Pop", "Rock", "Hip Hop", "R&B", "Electronic", "Jazz", "Classical", "Country"]
                    
                    moods = ["Happy", "Energetic", "Chill", "Romantic", "Melancholic", "Upbeat", "Relaxed", "Party"]
                    
                    # Generate 5 unique recommendations
                    used_indices = set()
                    for i in range(5):
                        # Ensure unique artist/album combinations
                        while True:
                            artist_idx = random.randint(0, len(artists) - 1)
                            album_idx = random.randint(0, len(albums) - 1)
                            if (artist_idx, album_idx) not in used_indices:
                                used_indices.add((artist_idx, album_idx))
                                break
                        
                        # Create a realistic song title based on the original song's mood
                        mood = random.choice(moods)
                        song_theme = f"{mood} {random.choice(['Nights', 'Days', 'Vibes', 'Memories', 'Moments'])}"
                        
                        # Create a recommendation
                        rec = {
                            'title': song_theme,
                            'artist': artists[artist_idx],
                            'album': albums[album_idx],
                            'genre': random.choice(genres),
                            'mood': mood,
                            'duration': random.randint(180, 300),  # 3-5 minutes
                            'file_path': f"/path/to/music/{artists[artist_idx].lower().replace(' ', '_')}_{song_theme.lower().replace(' ', '_')}.mp3"
                        }
                        recommended_songs.append(rec)
                    
                    # Update the UI with recommendations
                    self.recommended_songs = recommended_songs
                    
                    # Close loading dialog
                    if hasattr(self, 'loading_dialog') and self.loading_dialog:
                        self.loading_dialog.dismiss()
                        
                    # Cancel progress updates
                    if hasattr(self, 'progress_ev'):
                        self.progress_ev.cancel()
                    
                    # Update UI and switch screens
                    self.update_recommendations_ui()
                    self.switch_screen('recommend_screen')
                    
                except Exception as e:
                    # Close loading dialog on error
                    if hasattr(self, 'loading_dialog') and self.loading_dialog:
                        self.loading_dialog.dismiss()
                    
                    # Cancel progress updates on error
                    if hasattr(self, 'progress_ev'):
                        self.progress_ev.cancel()
                    
                    error_msg = f"Error in recommendation process: {str(e)}"
                    print(error_msg)
                    self.show_error_dialog("Error", "Could not generate recommendations. Please try again.")
            
            # Schedule the processing to happen asynchronously
            Clock.schedule_once(process_recommendations, 0.5)
            
            return True
            
        except Exception as e:
            # Close loading dialog if it's open
            if hasattr(self, 'loading_dialog') and self.loading_dialog:
                self.loading_dialog.dismiss()
                
            # Cancel progress updates if they exist
            if hasattr(self, 'progress_ev'):
                self.progress_ev.cancel()
                
            error_msg = f"Error generating recommendations: {str(e)}"
            print(error_msg)
            self.show_error_dialog("Error", "Could not generate recommendations. Please try again.")
            return False

    def show_snackbar(self, text, duration=2.0):
        """Show a notification popup with the given text"""
        try:
            # Create a popup with the message
            content = BoxLayout(orientation='vertical', padding='8dp', spacing='8dp')
            content.add_widget(Label(
                text=text, 
                size_hint_y=None, 
                height='40dp',
                color=(0, 0, 0, 1)  # Black text
            ))
            
            # Create a close button
            btn_close = Button(
                text='OK',
                size_hint=(None, None),
                size=('100dp', '40dp'),
                pos_hint={'center_x': 0.5},
                background_color=(0.13, 0.59, 0.95, 1),  # Blue button
                color=(1, 1, 1, 1)  # White text
            )
            
            # Create the popup
            popup = Popup(
                title='Notification',
                content=content,
                size_hint=(0.8, None),
                height='150dp',
                auto_dismiss=True,
                background_color=(0.9, 0.9, 0.9, 1)  # Light gray background
            )
            
            # Bind close button
            btn_close.bind(on_release=popup.dismiss)
            content.add_widget(btn_close)
            
            # Show the popup
            popup.open()
            
            # Auto-dismiss after duration if specified
            if duration > 0:
                Clock.schedule_once(lambda dt: popup.dismiss(), duration)
                
        except Exception as e:
            print(f"Error showing snackbar: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def get_selected_mood(self):
        """Get the currently selected mood from checkboxes (for genre-based playlists)
        
        Returns:
            str: The selected mood, or None if in song-based recommendation mode
        """
        # Check if we're in song-based recommendation mode
        if hasattr(self, 'root') and hasattr(self.root, 'ids') and 'screen_manager' in self.root.ids:
            if self.root.ids.screen_manager.current == 'recommend_screen':
                # In song-based mode, we don't need mood selection
                return 'auto'  # Indicate automatic mood detection from song
        
        # For genre/mood-based playlist generation
        try:
            # List of possible mood checkboxes with their display names
            mood_mapping = {
                'happy_check': 'happy',
                'sad_check': 'sad',
                'energetic_check': 'energetic',
                'calm_check': 'calm'
            }
            
            # Safely get the screen manager and current screen
            if not hasattr(self, 'root') or not hasattr(self.root, 'ids'):
                print("Warning: Root or root.ids not available")
                return None
                
            # Try to get the screen manager from root.ids
            screen_manager = None
            if 'screen_manager' in self.root.ids:
                screen_manager = self.root.ids.screen_manager
            
            if not screen_manager:
                # Try an alternative approach to find the screen manager
                for widget in self.root.walk():
                    if hasattr(widget, 'id') and widget.id == 'screen_manager':
                        screen_manager = widget
                        break
            
            if not screen_manager:
                print("Warning: Could not find screen manager in the widget tree")
                return None
            
            # Get the current screen
            current_screen = None
            if hasattr(screen_manager, 'current_screen'):
                current_screen = screen_manager.current_screen
            elif hasattr(screen_manager, 'current'):
                # Try to get the screen by name if current_screen is not available
                current_screen = screen_manager.get_screen(screen_manager.current) if hasattr(screen_manager, 'get_screen') else None
            
            if not current_screen:
                print("Warning: Could not get current screen")
                return None
            
            # Find all checkboxes in the current screen
            for widget in current_screen.walk(restrict=True):
                if hasattr(widget, 'id') and widget.id in mood_mapping:
                    if hasattr(widget, 'active') and widget.active:
                        print(f"Found active mood: {mood_mapping[widget.id]}")
                        return mood_mapping[widget.id]
            
            print("No mood selected")
            return None
            
        except Exception as e:
            print(f"Error in get_selected_mood: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def update_recommendations_ui(self):
        """Update the UI with the recommended songs"""
        if not hasattr(self, 'root') or not hasattr(self.root, 'ids'):
            return
            
        # Get the recommended songs list widget
        recommended_list = self.root.ids.get('recommended_songs_list')
        if not recommended_list:
            return
            
        # Clear the current list
        recommended_list.clear_widgets()
        
        if not hasattr(self, 'recommended_songs') or not self.recommended_songs:
            # Show a message if no recommendations are available
            label = MDLabel(
                text='No recommendations available',
                halign='center',
                theme_text_color='Secondary',
                size_hint_y=None,
                height='40dp'
            )
            recommended_list.add_widget(label)
            return
            
        try:
            # Add each recommended song to the list
            for song in self.recommended_songs:
                # Ensure we have a dictionary with the required keys
                if not isinstance(song, dict):
                    song = {'title': str(song), 'artist': 'Unknown Artist', 'album': 'Unknown Album'}
                
                # Create a container for the song info
                content = BoxLayout(orientation='vertical', padding=["16dp", "8dp", "8dp", "8dp"])
                
                # Add song title
                title_label = MDLabel(
                    text=str(song.get('title', 'Unknown Song')),
                    theme_text_color="Custom",
                    text_color=[0, 0, 0, 1],  # Black text
                    font_size='14sp',
                    size_hint_y=None,
                    height=dp(24)
                )
                
                # Add artist and album info
                info_label = MDLabel(
                    text=f"{song.get('artist', 'Unknown Artist')} • {song.get('album', 'Unknown Album')}",
                    theme_text_color="Secondary",
                    font_size='12sp',
                    size_hint_y=None,
                    height=dp(20)
                )
                
                # Create the list item with an icon and the content
                item = OneLineIconListItem(
                    text=str(song.get('title', 'Unknown Song')),
                    theme_text_color="Custom",
                    text_color=[0, 0, 0, 1],  # Black text
                    on_release=lambda x, s=song: self.on_song_selected(s)
                )
                item.add_widget(IconLeftWidget(
                    icon="music-note",
                    theme_text_color="Custom",
                    text_color=[0.13, 0.59, 0.95, 1]  # Blue color for the icon
                ))
                
                # Add the item to the recommended songs list
                if hasattr(self, 'recommended_songs_list'):
                    self.recommended_songs_list.add_widget(item)
                    
        except Exception as e:
            self.show_error_dialog("Error", f"Failed to add song to recommendations: {str(e)}")
    
    def on_song_ended(self):
        """Handle when the current song ends and trigger next song or recommendations."""
        try:
            if not hasattr(self, 'current_playlist') or not self.current_playlist:
                return
                
            # If we have more songs in the playlist, play the next one
            if hasattr(self, 'current_index') and self.current_index < len(self.current_playlist) - 1:
                next_index = self.current_index + 1
                self.play_song(index=next_index)
            # Otherwise, generate new recommendations based on the last played song
            elif hasattr(self, 'recommended_songs') and self.recommended_songs:
                self.show_snackbar("Generating new recommendations...")
                # Get the last played song's features to find similar songs
                last_song = self.current_playlist[-1] if self.current_playlist else None
                if last_song:
                    self.on_song_selected(last_song)
        except Exception as e:
            print(f"Error in on_song_ended: {str(e)}")
            import traceback
            traceback.print_exc()

    def on_song_selected(self, song):
        """Handle when a recommended song is selected."""
        try:
            if not song or not isinstance(song, (dict, str)):
                self.show_error_dialog("Error", "Invalid song data")
                return
                
            # Handle both string paths and song dictionaries
            if isinstance(song, dict):
                song_title = song.get('title', 'Unknown Song')
                song_artist = song.get('artist', 'Unknown Artist')
                file_path = song.get('file_path', song.get('path', ''))
            else:  # It's a string path
                song_title = os.path.basename(song)
                song_artist = "Unknown Artist"
                file_path = song
                
            self.show_snackbar(f"Selected: {song_title}")
            
            # Check if we have a valid file path
            if not file_path or not os.path.exists(file_path):
                self.show_error_dialog("File Not Found", f"Could not find audio file: {file_path}")
                return
                
            # Update the selected song and UI
            self.selected_song = song
            self.current_song = song_title
            self.current_artist = song_artist
            self.current_album = song.get('album', 'Unknown Album') if isinstance(song, dict) else 'Unknown Album'
            
            # Update the selected song label in the UI
            if hasattr(self, 'root') and hasattr(self.root, 'ids'):
                if 'selected_song_label' in self.root.ids:
                    self.root.ids.selected_song_label.text = song_title
            
            # Play the selected song
            if not music_player.play(file_path):
                self.show_error_dialog("Playback Error", f"Could not play: {song_title}")
                return
                
            # Update playback state
            self.is_playing = True
            self.total_time = music_player.duration
            
            # Update the current playlist if needed
            if hasattr(self, 'current_playlist') and song not in self.current_playlist:
                if isinstance(song, dict):
                    self.current_playlist.append(song)
                else:
                    self.current_playlist.append({
                        'title': song_title,
                        'artist': song_artist,
                        'file_path': file_path,
                        'album': song.get('album', 'Unknown Album') if isinstance(song, dict) else 'Unknown Album'
                    })
                self.current_index = len(self.current_playlist) - 1
            
            # Show loading dialog for recommendations
            content = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(10))
            content.add_widget(Label(
                text="Analyzing song and finding similar tracks...",
                color=(0, 0, 0, 1),  # Black text
                font_size='16sp',
                halign='center'
            ))
            
            # Add a progress bar
            progress = ProgressBar(max=100, value=0)
            content.add_widget(progress)
            
            # Create and show the loading popup
            dialog = Popup(
                title='Finding Similar Songs',
                content=content,
                size_hint=(0.8, 0.3),
                separator_color=[0.13, 0.59, 0.95, 1],
                title_color=[0, 0, 0, 1],
                title_size='18sp'
            )
            dialog.open()
            
            def update_progress(dt):
                """Update progress bar"""
                if progress.value < 90:  # Don't go to 100% to show we're still working
                    progress.value += 10
                    return True
                return False
                
            # Start progress updates
            progress_ev = Clock.schedule_interval(update_progress, 0.3)
            
            def generate_playlist_async(dt):
                """Generate recommendations asynchronously."""
                try:
                    # Generate new recommendations based on the selected song
                    success = self.simulate_recommendations()
                    
                    # Stop progress updates
                    progress_ev.cancel()
                    
                    # Close the dialog
                    if dialog and hasattr(dialog, 'dismiss'):
                        dialog.dismiss()
                    
                    if success:
                        self.show_snackbar("Recommendations updated!")
                    
                except Exception as e:
                    # Stop progress updates on error
                    progress_ev.cancel()
                    
                    # Close the dialog
                    if dialog and hasattr(dialog, 'dismiss'):
                        dialog.dismiss()
                    
                    error_msg = f"Error generating recommendations: {str(e)}"
                    print(error_msg)
                    self.show_error_dialog("Error", "Could not generate recommendations. Please try again.")
            
            # Schedule the async generation with a small delay
            Clock.schedule_once(generate_playlist_async, 0.5)
            
        except Exception as e:
            error_msg = f"Error selecting song: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.show_error_dialog("Error", "An unexpected error occurred. Please try again.")
    
    def show_error_dialog(self, title, message):
        """Show an error dialog using a custom popup with standard Kivy Button"""
        # Create a custom popup
        content = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(20))
        
        # Add message label
        from kivy.uix.label import Label
        content.add_widget(Label(
            text=message,
            halign='center',
            valign='middle',
            size_hint_y=0.8,
            text_size=(Window.width * 0.7, None),
            color=(0, 0, 0, 1)  # Black text
        ))
        
        # Add OK button with custom styling
        ok_btn = Button(
            text="OK",
            size_hint=(0.5, None),
            height=dp(48),
            pos_hint={'center_x': 0.5},
            background_color=[0.13, 0.59, 0.95, 1],  # Blue background
            color=(1, 1, 1, 1),  # White text
            background_normal='',
            background_down='',
            font_size='16sp',
            bold=True
        )
        
        # Create and show the popup
        self.popup = Popup(
            title=title,
            content=content,
            size_hint=(0.8, 0.4),
            title_size='18sp',
            title_align='center',
            title_color=(0, 0, 0, 1),  # Black title
            separator_color=[0.13, 0.59, 0.95, 1],  # Blue separator
            auto_dismiss=False
        )
        
        # Bind the button to dismiss the popup
        ok_btn.bind(on_release=lambda x: self.popup.dismiss())
        
        # Add button to content
        content.add_widget(ok_btn)
        
        # Open the popup
        self.popup.open()
        
    # Music Player Controls
    def play_song(self, index=None, file_path=None, song_data=None):
        """Play a song from the current playlist, a file path, or a song dictionary.
        
        Args:
            index: Index of the song in the current_playlist (optional)
            file_path: Path to an audio file to play (optional)
            song_data: Dictionary containing song data (optional)
                Expected keys: 'file_path', 'title', 'artist', 'album', 'duration'
                
        Returns:
            bool: True if playback started successfully, False otherwise
        """
        try:
            # If song_data is provided, use it
            if song_data and isinstance(song_data, dict):
                if 'file_path' not in song_data or not song_data['file_path']:
                    self.show_error_dialog("Playback Error", "No file path provided in song data")
                    return False
                    
                file_path = song_data['file_path']
                
            # Play from file path if provided
            if file_path:
                if not os.path.exists(file_path):
                    self.show_error_dialog("File Not Found", f"Could not find audio file: {file_path}")
                    return False
                    
                if music_player.play(file_path):
                    # Update UI with song info
                    self.current_song = song_data.get('title', os.path.basename(file_path)) if song_data else os.path.basename(file_path)
                    self.current_artist = song_data.get('artist', 'Unknown Artist') if song_data else 'Unknown Artist'
                    self.current_album = song_data.get('album', 'Unknown Album') if song_data else 'Unknown Album'
                    self.total_time = music_player.duration
                    self.is_playing = True
                    
                    # If this is a recommended song, add it to the recent plays
                    if song_data and hasattr(self, 'recommended_songs') and song_data in self.recommended_songs:
                        # In a real app, you might want to add this to a "recently played" list
                        pass
                        
                    return True
                return False
            
            # Play from playlist by index
            elif index is not None and hasattr(self, 'current_playlist') and 0 <= index < len(self.current_playlist):
                song = self.current_playlist[index]
                if 'file_path' not in song or not song['file_path']:
                    self.show_error_dialog("Playback Error", "No file path available for this song")
                    return False
                    
                if music_player.play(song['file_path']):
                    self.current_index = index
                    self.current_song = song.get('title', os.path.basename(song['file_path']))
                    self.current_artist = song.get('artist', 'Unknown Artist')
                    self.current_album = song.get('album', 'Unknown Album')
                    self.total_time = music_player.duration
                    self.is_playing = True
                    return True
                return False
            
            # No valid input
            self.show_error_dialog("Playback Error", "No song selected or invalid song data")
            return False
            
        except Exception as e:
            error_msg = f"Could not play song: {str(e)}"
            print(error_msg)
            self.show_error_dialog("Playback Error", error_msg)
            return False
    
    def toggle_play_pause(self):
        """Toggle between play and pause."""
        if music_player.is_playing:
            music_player.pause()
            self.is_playing = False
        else:
            if music_player.current_song:
                music_player.play()
                self.is_playing = True
            elif self.current_playlist:
                self.play_song(0)  # Start playing the first song
    
    def stop_playback(self):
        """Stop the current playback."""
        music_player.stop()
        self.is_playing = False
    
    def next_song(self):
        """Play the next song in the playlist."""
        if not self.current_playlist or self.current_index == -1:
            return
            
        next_index = (self.current_index + 1) % len(self.current_playlist)
        self.play_song(next_index)
    
    def previous_song(self):
        """Play the previous song in the playlist."""
        if not self.current_playlist or self.current_index == -1:
            return
            
        prev_index = (self.current_index - 1) % len(self.current_playlist)
        self.play_song(prev_index)
    
    def set_volume(self, volume):
        """Set the player volume (0.0 to 1.0)."""
        volume = max(0.0, min(1.0, float(volume)))
        self.volume = volume
        music_player.volume = volume
    
    def seek(self, position):
        """Seek to a specific position in the current song."""
        music_player.seek(position)
    
    def update_time_display(self, instance, position):
        """Update the current time display."""
        self.current_time = position
    
    def update_duration(self, instance, duration):
        """Update the total duration display."""
        self.total_time = duration
    
    def update_play_button_state(self, instance, is_playing):
        """Update the play/pause button state."""
        self.is_playing = is_playing
    
    def format_duration(self, seconds):
        """Format seconds into MM:SS format."""
        if not isinstance(seconds, (int, float)) or seconds < 0:
            return "00:00"
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def on_pause(self):
        # Handle app pause (e.g., when phone call comes in)
        return True
        
    def toggle_theme(self, *args):
        """Toggle between light and dark theme"""
        try:
            if self.theme_cls.theme_style == "Light":
                self.theme_cls.theme_style = "Dark"
                # Update any specific dark mode colors here
            else:
                self.theme_cls.theme_style = "Light"
                # Update any specific light mode colors here
                
            # Force update the navigation buttons to reflect theme changes
            if hasattr(self, 'root') and hasattr(self.root, 'ids'):
                screen_manager = self.root.ids.get('screen_manager')
                if screen_manager:
                    self.update_nav_buttons(screen_manager.current)
                    
        except Exception as e:
            print(f"Error toggling theme: {str(e)}")

    def on_stop(self):
        """Clean up when the app is closed."""
        music_player.cleanup()
        return True

if __name__ == '__main__':
    DCMApp().run()