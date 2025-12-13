import os
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random

class MusicDatabase:
    def __init__(self, db_path: str = 'music_library.db'):
        """Initialize the music database."""
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create songs table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS songs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                title TEXT,
                artist TEXT,
                album TEXT,
                duration INTEGER,  -- in seconds
                genre TEXT,
                mood TEXT,
                bpm INTEGER,
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            
            # Create playlists table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS playlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            
            # Create playlist_songs junction table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS playlist_songs (
                playlist_id INTEGER,
                song_id INTEGER,
                position INTEGER,
                PRIMARY KEY (playlist_id, song_id),
                FOREIGN KEY (playlist_id) REFERENCES playlists(id) ON DELETE CASCADE,
                FOREIGN KEY (song_id) REFERENCES songs(id) ON DELETE CASCADE
            )''')
            
            conn.commit()
    
    def add_song(self, file_path: str, title: str = None, artist: str = None, 
                album: str = None, genre: str = None, mood: str = None, 
                bpm: int = None, duration: int = None) -> int:
        """Add a song to the database."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # If title is not provided, use the filename without extension
            if title is None:
                title = os.path.splitext(os.path.basename(file_path))[0]
            
            try:
                cursor.execute('''
                INSERT INTO songs (file_path, title, artist, album, duration, genre, mood, bpm)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (file_path, title, artist, album, duration, genre, mood, bpm))
                
                song_id = cursor.lastrowid
                conn.commit()
                return song_id
                
            except sqlite3.IntegrityError:
                # Song already exists, return its ID
                cursor.execute('SELECT id FROM songs WHERE file_path = ?', (file_path,))
                return cursor.fetchone()[0]
    
    def get_song(self, song_id: int) -> Optional[Dict]:
        """Get a song by its ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM songs WHERE id = ?', (song_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    def get_similar_songs(self, song_id: int, limit: int = 5) -> List[Dict]:
        """Get songs similar to the given song ID."""
        song = self.get_song(song_id)
        if not song:
            return []
            
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Simple similarity: same genre or mood
            # In a real app, you'd use more sophisticated algorithms
            query = '''
            SELECT * FROM songs 
            WHERE id != ? 
            AND (genre = ? OR mood = ?)
            ORDER BY RANDOM()
            LIMIT ?
            '''
            
            cursor.execute(query, (song_id, song.get('genre'), song.get('mood'), limit))
            
            # If not enough results, get any random songs
            results = [dict(row) for row in cursor.fetchall()]
            
            if len(results) < limit:
                cursor.execute('''
                SELECT * FROM songs 
                WHERE id != ? 
                ORDER BY RANDOM()
                LIMIT ?
                ''', (song_id, limit - len(results)))
                
                results.extend([dict(row) for row in cursor.fetchall()])
            
            return results
    
    def get_songs_by_mood(self, mood: str, limit: int = 10) -> List[Dict]:
        """Get songs by mood."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM songs 
            WHERE mood = ? 
            ORDER BY RANDOM()
            LIMIT ?
            ''', (mood, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def search_songs(self, query: str, limit: int = 20) -> List[Dict]:
        """Search for songs by title, artist, or album."""
        search_term = f"%{query}%"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM songs 
            WHERE title LIKE ? OR artist LIKE ? OR album LIKE ?
            LIMIT ?
            ''', (search_term, search_term, search_term, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def add_playlist(self, name: str, song_ids: List[int] = None) -> int:
        """Create a new playlist with optional songs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('INSERT INTO playlists (name) VALUES (?)', (name,))
            playlist_id = cursor.lastrowid
            
            if song_ids:
                for i, song_id in enumerate(song_ids):
                    cursor.execute('''
                    INSERT INTO playlist_songs (playlist_id, song_id, position)
                    VALUES (?, ?, ?)
                    ''', (playlist_id, song_id, i))
            
            conn.commit()
            return playlist_id
    
    def get_playlist_songs(self, playlist_id: int) -> List[Dict]:
        """Get all songs in a playlist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT s.* FROM songs s
            JOIN playlist_songs ps ON s.id = ps.song_id
            WHERE ps.playlist_id = ?
            ORDER BY ps.position
            ''', (playlist_id,))
            
            return [dict(row) for row in cursor.fetchall()]

# Singleton instance
db = MusicDatabase()

# Add some sample data if the database is empty
def _add_sample_data():
    """Add sample data for testing."""
    sample_songs = [
        {
            'file_path': '/path/to/song1.mp3',
            'title': 'Sample Song 1',
            'artist': 'Artist A',
            'album': 'Album X',
            'genre': 'Pop',
            'mood': 'Happy',
            'bpm': 120,
            'duration': 180
        },
        # Add more sample songs as needed
    ]
    
    with sqlite3.connect('music_library.db') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM songs')
        if cursor.fetchone()[0] == 0:
            for song in sample_songs:
                db.add_song(**song)

# Initialize sample data (comment out in production)
# _add_sample_data()
