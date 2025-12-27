
import logging
from dcm.player import MusicPlayer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global state
class AppState:
    current_song_path = None
    current_song_meta = {}
    auto_queue = False
    loop_current = False
    recommender = None
    recommendations = []
    play_history = []
    
    # Queue system
    queue = []
    queue_index = 0
    songs_played_since_refresh = 0
    
    # Icon reference (for menu updates)
    icon = None  # Will hold QSystemTrayIcon reference

state = AppState()
player = MusicPlayer()
