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
import pygame  # For checking music playback status

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
    recommendations = []  # Top 5 for display
    play_history = []  # History of played songs for "Previous" button
    
    # Queue system
    queue = []  # Full queue of up to 10 songs
    queue_index = 0  # Current position in queue
    songs_played_since_refresh = 0  # Counter for auto-refresh
    
    # Tray icon reference (for updating menu)
    icon = None

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
        
        # Rebuild tray menu to reflect new song
        if state.icon:
            state.icon.menu = build_menu()
    else:
        logger.error(f"Failed to play: {file_path}")

def build_queue_from_seed(seed_song_path):
    """Build a queue of 10 songs starting from a seed song."""
    if not seed_song_path:
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
        
        # Get recommendations, excluding recently played songs
        exclude_list = state.play_history[-50:] if len(state.play_history) > 0 else []
        
        similar = state.recommender.find_similar_songs(
            seed_song_path, 
            n_songs=15,  # Request more than 10 to have options after filtering
            exclude_paths=exclude_list
        )
        
        # Build queue from recommendations
        state.queue = []
        for _, row in similar.iterrows():
            if row['file_path'] != seed_song_path and len(state.queue) < 10:
                state.queue.append({
                    'path': row['file_path'],
                    'title': get_metadata(row['file_path']).get('title', os.path.basename(row['file_path'])),
                    'similarity': row['similarity_score']
                })
        
        state.queue_index = 0
        state.songs_played_since_refresh = 0
        
        # Also populate recommendations (top 5) for menu display
        state.recommendations = state.queue[:5] if state.queue else []
        
        logger.info(f"Built queue with {len(state.queue)} songs")
    except Exception as e:
        logger.error(f"Error building queue: {e}")

def refresh_queue_if_needed():
    """Refresh queue if 6 songs have been played since last refresh."""
    if state.songs_played_since_refresh >= 6 and len(state.queue) > 0:
        # Use current song as new seed
        if state.current_song_path:
            logger.info(f"Refreshing queue (played {state.songs_played_since_refresh} songs)")
            build_queue_from_seed(state.current_song_path)

def load_recommendations():
    """Legacy function - now builds queue instead."""
    if state.current_song_path:
        build_queue_from_seed(state.current_song_path)

# --- Menu Building ---
def format_time(seconds):
    """Format seconds as MM:SS."""
    if seconds <= 0:
        return "0:00"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"

def get_now_playing_text(item):
    """Get the 'Now Playing' text for the menu."""
    MAX_LENGTH = 50  # Maximum length for menu item text
    
    if state.current_song_path and player.is_playing:
        title = state.current_song_meta.get('title', 'Unknown')
        artist = state.current_song_meta.get('artist', '')
        
        if artist:
            full_text = f"🎵 {title} - {artist}"
        else:
            full_text = f"🎵 {title}"
        
        # Truncate if too long (simple truncation with ellipsis)
        if len(full_text) > MAX_LENGTH:
            full_text = full_text[:MAX_LENGTH-3] + "..."
        
        return full_text
    elif state.current_song_path:
        title = state.current_song_meta.get('title', 'Unknown')
        paused_text = f"⏸️ {title} (Paused)"
        
        # Truncate if too long
        if len(paused_text) > MAX_LENGTH:
            paused_text = paused_text[:MAX_LENGTH-3] + "..."
        
        return paused_text
    
    return "♪ No song playing"

def get_progress_text(item):
    """Get the progress text showing current time / total time."""
    if state.current_song_path:
        current = format_time(player.current_position)
        total = format_time(player.duration)
        return f"⏱️ {current} / {total}"
    return "⏱️ --:-- / --:--"

def toggle_play_pause(icon, item):
    """Toggle play/pause."""
    if player.is_playing:
        player.pause()
    elif state.current_song_path:
        player.unpause()

def seek_to_position(percentage):
    """Create a seek handler for a specific percentage."""
    def handler(icon, item):
        if state.current_song_path and player.duration > 0:
            position = (percentage / 100.0) * player.duration
            player.seek(position)
            logger.info(f"Seeked to {percentage}% ({format_time(position)})")
    return handler

def toggle_auto_queue(icon, item):
    """Toggle auto-queue feature."""
    state.auto_queue = not state.auto_queue
    
    # If turning on Auto-Queue, turn off Loop
    if state.auto_queue and state.loop_current:
        state.loop_current = False
        logger.info("Loop disabled (Auto-Queue enabled)")
    
    logger.info(f"Auto-Queue: {'ON' if state.auto_queue else 'OFF'}")
    
    # Rebuild menu to reflect changes
    if state.icon:
        state.icon.menu = build_menu()

def toggle_loop(icon, item):
    """Toggle loop current song."""
    state.loop_current = not state.loop_current
    
    # If turning on Loop, turn off Auto-Queue
    if state.loop_current and state.auto_queue:
        state.auto_queue = False
        logger.info("Auto-Queue disabled (Loop enabled)")
    
    logger.info(f"Loop: {'ON' if state.loop_current else 'OFF'}")
    
    # Rebuild menu to reflect changes
    if state.icon:
        state.icon.menu = build_menu()

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
    # Build seek submenu
    seek_items = [
        pystray.MenuItem("Start (0%)", seek_to_position(0)),
        pystray.MenuItem("25%", seek_to_position(25)),
        pystray.MenuItem("50%", seek_to_position(50)),
        pystray.MenuItem("75%", seek_to_position(75)),
        pystray.MenuItem("End (100%)", seek_to_position(100)),
    ]
    
    menu_items = [
        pystray.MenuItem("📁 Open Song...", lambda icon, item: open_file_dialog()),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(get_now_playing_text, None, enabled=False),
        # Progress timer removed - will be in mini player
        pystray.MenuItem(
            "⏩ Seek to...",
            pystray.Menu(*seek_items),
            enabled=lambda item: state.current_song_path is not None and player.duration > 0
        ),
        pystray.Menu.SEPARATOR,
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
    # Add queue/recommendations display - Show queue when Auto-Queue is enabled
    if state.auto_queue and state.queue and len(state.queue) > 0:
        queue_items = []
        # Show current position indicator
        queue_items.append(pystray.MenuItem(f"[Playing #{state.queue_index + 1} of {len(state.queue)}]", None, enabled=False))
        queue_items.append(pystray.Menu.SEPARATOR)
        
        # Show all songs in queue (up to 10)
        for i, song in enumerate(state.queue):
            title = song['title'][:35] + "..." if len(song['title']) > 35 else song['title']
            # Mark currently playing song
            if i == state.queue_index:
                prefix = "▶"
            else:
                prefix = f"{i + 1}."
            queue_items.append(
                pystray.MenuItem(f"{prefix} {title}", play_recommendation(song['path']))
            )
        
        menu_items.append(pystray.MenuItem("📋 Queue", pystray.Menu(*queue_items)))
    else:
        # Show instruction when Auto-Queue is off
        if state.auto_queue:
            menu_items.append(pystray.MenuItem("📋 Queue: (play a song first)", None, enabled=False))
        else:
            menu_items.append(pystray.MenuItem("📋 Queue: (enable Auto-Queue)", None, enabled=False))
    
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
    was_playing = False
    
    while True:
        time.sleep(0.5)  # Check more frequently
        
        # Check if music is actually playing using pygame's get_busy()
        is_busy = pygame.mixer.music.get_busy()
        
        # Detect song end: was playing before, now not busy, same song
        # BUT: Don't trigger if player.is_playing is False (means paused, not ended)
        # When paused: player.is_playing = False (set by pause())
        # When natural end: player.is_playing might still be True (not yet updated)
        # So we check: if player.is_playing is False, it's paused; if True, it might be natural end
        if was_playing and not is_busy and state.current_song_path == last_song:
            # Check if this is a pause (player.is_playing is False) or natural end (player.is_playing is True)
            if not player.is_playing:
                # This is a pause, not an end - reset was_playing to prevent false trigger
                was_playing = False
                continue
            
            # Song has ended naturally (player.is_playing is still True but pygame is not busy)
            logger.info("Song ended")
            
            # Handle based on mode: Loop takes precedence over Auto-Queue
            if state.loop_current:
                # Loop: replay the same song
                logger.info("Looping current song")
                if state.current_song_path:
                    player.play(state.current_song_path)
            elif state.auto_queue:
                # Auto-Queue: play next from queue
                logger.info("Auto-queueing next...")
                
                # Increment counters
                state.queue_index += 1
                state.songs_played_since_refresh += 1
                
                # Check if we need to refresh queue
                refresh_queue_if_needed()
                
                # Play next song from queue
                if state.queue and state.queue_index < len(state.queue):
                    next_song = state.queue[state.queue_index]
                    logger.info(f"Playing next from queue [{state.queue_index + 1}/{len(state.queue)}]: {next_song['title']}")
                    play_song(next_song['path'])
                elif state.queue:  # Queue exists but we're at the end
                    # Loop back to start
                    logger.info("Reached end of queue, restarting...")
                    state.queue_index = 0
                    play_song(state.queue[0]['path'])
                else:
                    logger.warning("No queue available")
            # else: neither Loop nor Auto-Queue - do nothing, song ends normally
            
            was_playing = False
            last_song = None
        elif is_busy:
            # Song is playing
            was_playing = True
            last_song = state.current_song_path

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
    
    # Store icon reference for menu updates
    state.icon = icon
    
    # Run the icon (this blocks)
    icon.run()

if __name__ == "__main__":
    main()
