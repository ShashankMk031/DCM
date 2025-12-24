#!/usr/bin/env python3
"""
DCM System Tray Application
A menu bar / system tray app for the DCM Music Recommender.
"""

import os
import sys
import threading
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pystray
from PIL import Image, ImageDraw

from dcm.player import player
from dcm.core.metadata import get_metadata
from dcm.core.suggest_next import SongRecommender

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global state
class AppState:
    current_song_path = None
    current_song_meta = {}
    auto_queue = False
    loop_current = False  # Loop current song
    recommender = None
    recommendations = []
    play_history = []  # History of played songs for "Previous" button

state = AppState()

# --- Icon Creation ---
def create_icon_image():
    """Create a simple music note icon."""
    size = 64
    image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    # Draw a simple music note shape
    # Circle (note head)
    draw.ellipse([10, 35, 30, 55], fill='white')
    # Stem
    draw.rectangle([26, 10, 30, 40], fill='white')
    # Flag
    draw.polygon([(30, 10), (50, 20), (50, 30), (30, 25)], fill='white')
    
    return image

# --- File Picker ---
def open_file_dialog():
    """Open a file dialog to select a song."""
    import platform
    import subprocess
    
    file_path = None
    
    if platform.system() == 'Darwin':
        # macOS: Use AppleScript via subprocess (works from any thread)
        try:
            script = '''
            tell application "System Events"
                activate
                set theFile to choose file with prompt "Select a Song" of type {"mp3", "m4a", "flac", "wav", "ogg"}
                return POSIX path of theFile
            end tell
            '''
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                file_path = result.stdout.strip()
                logger.info(f"Selected file: {file_path}")
            else:
                logger.info("File selection cancelled or failed")
                return
        except Exception as e:
            logger.error(f"AppleScript error: {e}")
            return
    else:
        # Linux/Windows: Use Tkinter
        try:
            from tkinter import Tk, filedialog
            root = Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                title="Select a Song",
                filetypes=[
                    ("Audio Files", "*.mp3 *.m4a *.flac *.wav *.ogg"),
                    ("All Files", "*.*")
                ]
            )
            root.destroy()
        except Exception as e:
            logger.error(f"Tkinter error: {e}")
            return
    
    if file_path:
        play_song(file_path)

def play_song(file_path):
    """Play a song and update state."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return
    
    logger.info(f"Playing: {file_path}")
    
    if player.play(file_path):
        state.current_song_path = file_path
        state.current_song_meta = get_metadata(file_path)
        
        # Add to play history for "Previous" button
        state.play_history.append(file_path)
        # Keep history limited to last 50 songs
        if len(state.play_history) > 50:
            state.play_history.pop(0)
        
        # Load recommendations in background
        threading.Thread(target=load_recommendations, daemon=True).start()
    else:
        logger.error(f"Failed to play: {file_path}")

def load_recommendations():
    """Load recommendations for the current song."""
    if not state.current_song_path:
        return
    
    try:
        if state.recommender is None:
            features_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'features', 'all_features.csv'
            )
            if os.path.exists(features_path):
                state.recommender = SongRecommender()
                state.recommender.load_features(features_path)
                state.recommender.train_model()
                logger.info("Recommender loaded successfully")
            else:
                logger.warning(f"Features file not found: {features_path}")
                return
        
        
        # Get recommendations, excluding recently played songs to prevent loops
        # Exclude the last 50 songs from history to ensure variety
        exclude_list = state.play_history[-50:] if len(state.play_history) > 0 else []
        
        similar = state.recommender.find_similar_songs(
            state.current_song_path, 
            n_songs=10,  # Request more to have options after filtering
            exclude_paths=exclude_list
        )
        state.recommendations = []
        for _, row in similar.iterrows():
            if row['file_path'] != state.current_song_path:
                meta = get_metadata(row['file_path'])
                state.recommendations.append({
                    'path': row['file_path'],
                    'title': meta.get('title', os.path.basename(row['file_path'])),
                    'similarity': row['similarity_score']
                })
        logger.info(f"Loaded {len(state.recommendations)} recommendations")
    except Exception as e:
        logger.error(f"Error loading recommendations: {e}")

# --- Menu Building ---
def get_now_playing_text(item):
    """Get the 'Now Playing' text for the menu."""
    if state.current_song_path and player.is_playing:
        title = state.current_song_meta.get('title', 'Unknown')
        artist = state.current_song_meta.get('artist', '')
        if artist:
            return f"🎵 {title} - {artist}"
        return f"🎵 {title}"
    elif state.current_song_path:
        return "⏸️ Paused"
    return "No song playing"

def toggle_play_pause(icon, item):
    """Toggle play/pause."""
    if player.is_playing:
        player.pause()
    elif state.current_song_path:
        player.unpause()

def toggle_auto_queue(icon, item):
    """Toggle auto-queue feature."""
    state.auto_queue = not state.auto_queue
    logger.info(f"Auto-Queue: {'ON' if state.auto_queue else 'OFF'}")

def toggle_loop(icon, item):
    """Toggle loop current song."""
    state.loop_current = not state.loop_current
    logger.info(f"Loop: {'ON' if state.loop_current else 'OFF'}")

def play_next(icon, item):
    """Play the next recommended song."""
    if state.recommendations:
        next_song = state.recommendations[0]['path']
        logger.info(f"Playing next: {next_song}")
        play_song(next_song)
    else:
        logger.info("No recommendations available")

def play_previous(icon, item):
    """Play the previous song from history."""
    if len(state.play_history) >= 2:
        # Current song is at index -1, previous is at -2
        prev_song = state.play_history[-2]
        state.play_history.pop()  # Remove current from history
        logger.info(f"Playing previous: {prev_song}")
        # Play without adding to history again
        if player.play(prev_song):
            state.current_song_path = prev_song
            state.current_song_meta = get_metadata(prev_song)
    else:
        logger.info("No previous song in history")

def play_recommendation(path):
    """Play a recommended song."""
    def action(icon, item):
        play_song(path)
    return action

def build_menu():
    """Build the tray menu."""
    menu_items = [
        pystray.MenuItem("📁 Open Song...", lambda icon, item: open_file_dialog()),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(get_now_playing_text, None, enabled=False),
        pystray.MenuItem(
            "⏯️ Play/Pause", 
            toggle_play_pause,
            enabled=lambda item: state.current_song_path is not None
        ),
        pystray.MenuItem(
            "⏮️ Previous", 
            play_previous,
            enabled=lambda item: len(state.play_history) >= 2
        ),
        pystray.MenuItem(
            "⏭️ Next", 
            play_next,
            enabled=lambda item: len(state.recommendations) > 0
        ),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(
            "🔁 Loop", 
            toggle_loop,
            checked=lambda item: state.loop_current
        ),
        pystray.MenuItem(
            "Auto-Queue", 
            toggle_auto_queue,
            checked=lambda item: state.auto_queue
        ),
        pystray.Menu.SEPARATOR,
    ]
    
    # Add recommendations submenu
    if state.recommendations:
        rec_items = []
        for rec in state.recommendations[:5]:
            title = rec['title'][:30] + "..." if len(rec['title']) > 30 else rec['title']
            score = f"{int(rec['similarity'] * 100)}%"
            rec_items.append(
                pystray.MenuItem(f"  {title} ({score})", play_recommendation(rec['path']))
            )
        menu_items.append(pystray.MenuItem("📋 Up Next:", pystray.Menu(*rec_items)))
    else:
        menu_items.append(pystray.MenuItem("📋 Up Next: (play a song first)", None, enabled=False))
    
    menu_items.extend([
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", lambda icon, item: icon.stop()),
    ])
    
    return pystray.Menu(*menu_items)

# --- Auto-Queue Background Thread ---
def auto_queue_monitor():
    """Monitor playback and auto-queue next song when current ends."""
    import time
    last_song = None
    
    while True:
        time.sleep(1)
        
        if state.auto_queue and state.current_song_path:
            # Check if song ended (not playing but we have a current song)
            if not player.is_playing and state.current_song_path == last_song:
                # Song likely ended, queue next
                if state.recommendations:
                    next_song = state.recommendations[0]['path']
                    logger.info(f"Auto-Queue: Playing next - {next_song}")
                    play_song(next_song)
            
            last_song = state.current_song_path if player.is_playing else last_song

# --- Main Entry Point ---
def main():
    """Main entry point for the tray app."""
    logger.info("Starting DCM Tray App...")
    
    # Start auto-queue monitor thread
    monitor_thread = threading.Thread(target=auto_queue_monitor, daemon=True)
    monitor_thread.start()
    
    # Create the tray icon
    icon = pystray.Icon(
        "DCM",
        create_icon_image(),
        "DCM Music Player",
        menu=build_menu()
    )
    
    # Run the icon (this blocks)
    icon.run()

if __name__ == "__main__":
    main()
