
import os
import logging
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import core DCM modules
# We need to ensure the project root is in path if running directly
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcm.player import player
from dcm.core.metadata import get_metadata, get_album_art
from dcm.core.suggest_next import SongRecommender

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DCM Music Player", description="Backend API for DCM Web Interface")

# Enable CORS for frontend development (Vite runs on 5173 by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Song(BaseModel):
    file_path: str
    title: str
    artist: str
    album: str
    duration: float = 0.0
    
class PlayerStatus(BaseModel):
    is_playing: bool
    current_song: Optional[Song] = None
    position: float = 0.0
    duration: float = 0.0
    volume: float = 1.0
    auto_queue: bool = False

# Global Auto-Queue State
AUTO_QUEUE_ENABLED = False
LAST_PLAYED_SONG = None

# Initialize Recommender (Global)
# We'll load this lazily or on startup
recommender = None

DEFAULT_FEATURES_PATH = Path("features/all_features.csv")

def get_recommender():
    global recommender
    if recommender is None:
        if DEFAULT_FEATURES_PATH.exists():
            recommender = SongRecommender()
            recommender.load_features(str(DEFAULT_FEATURES_PATH))
            recommender.train_model()
    return recommender

# --- Endpoints ---

@app.get("/")
async def root():
    return {"message": "DCM Backend is running"}

@app.get("/status", response_model=PlayerStatus)
async def get_status():
    """Get current player status"""
    current_song_data = None
    if player.current_song:
        # Get metadata for current song
        meta = get_metadata(player.current_song)
        current_song_data = Song(
            file_path=player.current_song,
            title=meta.get('title', 'Unknown'),
            artist=meta.get('artist', 'Unknown'),
            album=meta.get('album', 'Unknown'),
            duration=player.duration
        )

    return PlayerStatus(
        is_playing=player.is_playing,
        current_song=current_song_data,
        position=player.current_position,
        duration=player.duration,
        volume=player.volume,
        auto_queue=AUTO_QUEUE_ENABLED
    )

@app.post("/toggle_auto_queue")
async def toggle_auto_queue(enabled: bool):
    global AUTO_QUEUE_ENABLED
    AUTO_QUEUE_ENABLED = enabled
    return {"status": "success", "auto_queue": AUTO_QUEUE_ENABLED}

# Check for song end and auto-queue
# Note: This is a simple implementation hooked into the status poll.
# In a robust production app, this should be a separate background thread/worker.
@app.on_event("startup")
async def startup_event():
    import asyncio
    asyncio.create_task(auto_queue_worker())

async def auto_queue_worker():
    """Background task to check if song ended and queue next"""
    import asyncio
    global LAST_PLAYED_SONG
    
    while True:
        await asyncio.sleep(1) # Check every second
        
        if AUTO_QUEUE_ENABLED and player.current_song:
            # If player stopped naturally (not paused) and we have a current song
            # Note: Pygame stop state + position check is tricky. 
            # Simplified logic: If not playing but we have a song and position is near 0 (reset) or duration
            # Ideally player.py would have an 'on_ended' event.
            
            # Better approach: Check if position >= duration - 1 (near end)
            try:
                if player.is_playing:
                    # Update last played to avoid re-queueing immediately
                    LAST_PLAYED_SONG = player.current_song
                    
                elif not player.is_playing and LAST_PLAYED_SONG == player.current_song:
                    # Song stopped. Check if it finished?
                    # Pygame resets position to 0 on stop.
                    # We assume if it stopped and we have AutoQueue, we want more music.
                    # To avoid loop on manual stop, we could check a 'manual_stop' flag in player, 
                    # but for now let's just queue if it's not playing.
                    
                    logger.info("Auto-Queue: Song ended, fetching recommendation...")
                    rec = get_recommender()
                    if rec:
                        similar = rec.find_similar_songs(player.current_song, n_songs=2) # Get top 2
                        # The first one might be the song itself if distance is 0, or very close.
                        # suggestion_next usually filters input, but let's be safe.
                        
                        next_song = None
                        for _, row in similar.iterrows():
                            if row['file_path'] != player.current_song:
                                next_song = row['file_path']
                                break
                        
                        if next_song:
                            logger.info(f"Auto-Queue: Playing next - {next_song}")
                            player.play(next_song)
                            LAST_PLAYED_SONG = next_song
            except Exception as e:
                logger.error(f"Auto-queue error: {e}")

@app.post("/play")
async def play_song(file_path: str):
    """Play a specific file"""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    success = player.play(file_path)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to play song")
        
    return {"status": "playing", "file": file_path}

@app.post("/control/{action}")
async def player_control(action: str):
    """Control playback: pause, resume, stop, next, prev"""
    if action == "pause":
        player.pause()
    elif action == "resume":
        if player.current_song:
            player.play() # Resume
    elif action == "stop":
        player.stop()
    elif action == "next":
        player.next()
    elif action == "prev":
        player.previous()
    else:
        raise HTTPException(status_code=400, detail="Invalid action")
        
    return {"status": "success", "action": action}

@app.post("/recommend")
async def recommend_similar(file_path: str, n: int = 5):
    """Get recommendations for a song"""
    rec = get_recommender()
    if not rec:
        raise HTTPException(status_code=503, detail="Recommendation engine not ready (features not extracted?)")
        
    try:
        similar_df = rec.find_similar_songs(file_path, n_songs=n)
        # Convert to list of dicts
        recommendations = similar_df.to_dict('records')
        
        # Add metadata to recommendations
        enhanced_recs = []
        for r in recommendations:
            path = r['file_path']
            meta = get_metadata(path)
            meta['file_path'] = path
            meta['similarity'] = r['similarity_score']
            enhanced_recs.append(meta)
            
        return enhanced_recs
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/album_art")
async def get_art(file_path: str):
    """Get album art path for a file"""
    art_path = get_album_art(file_path)
    if not art_path:
        # You might want to return a default image URL or 404
        return {"found": False}
    return {"found": True, "path": art_path}

# Mount static files for the frontend (once built)
# app.mount("/", StaticFiles(directory="dcm/ui/web/dist", html=True), name="static")
