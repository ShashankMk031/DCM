
import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, QStyle)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QIcon, QPixmap, QAction, QImage
from PIL import Image

# Import shared state/logic
from dcm.state import state, player
from dcm.core.metadata import get_metadata

class MiniPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DCM Mini Player")
        self.setFixedSize(320, 160)
        
        # Window flags: Frameless, Always on Top, Tool
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | 
                            Qt.WindowType.WindowStaysOnTopHint |
                            Qt.WindowType.Tool)
        
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Styles
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                border-radius: 10px;
            }
            QLabel { font-family: 'Segoe UI', sans-serif; }
            QSlider::groove:horizontal {
                border: 1px solid #3d3d3d;
                height: 4px;
                background: #3d3d3d;
                margin: 0px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #a0a0ff;
                border: 1px solid #a0a0ff;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 15px;
            }
            QPushButton:hover { background-color: #3d3d3d; }
            QPushButton:pressed { background-color: #505050; }
        """)

        # Layouts
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_widget.setLayout(main_layout)
        
        # 1. Album Art (Left)
        self.art_label = QLabel()
        self.art_label.setFixedSize(140, 140)
        self.art_label.setStyleSheet("background-color: #1a1a1a; border-radius: 8px;")
        self.art_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.art_label)
        
        # 2. Controls & Info (Right)
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout)
        
        # Song Info
        self.title_label = QLabel("Not Playing")
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.artist_label = QLabel("")
        self.artist_label.setStyleSheet("font-size: 12px; color: #aaaaaa;")
        
        right_layout.addWidget(self.title_label)
        right_layout.addWidget(self.artist_label)
        right_layout.addStretch()
        
        # Seek Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.sliderReleased.connect(self.seek)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.is_sliding = False
        right_layout.addWidget(self.slider)
        
        # Time Labels
        time_layout = QHBoxLayout()
        self.current_time_label = QLabel("0:00")
        self.total_time_label = QLabel("0:00")
        self.current_time_label.setStyleSheet("font-size: 10px; color: #888;")
        self.total_time_label.setStyleSheet("font-size: 10px; color: #888;")
        time_layout.addWidget(self.current_time_label)
        time_layout.addStretch()
        time_layout.addWidget(self.total_time_label)
        right_layout.addLayout(time_layout)
        
        # Control Buttons
        controls_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("⏮")
        self.play_btn = QPushButton("▶")
        self.next_btn = QPushButton("⏭")
        
        # Style buttons
        for btn in [self.prev_btn, self.play_btn, self.next_btn]:
            btn.setFixedSize(30, 30)
            controls_layout.addWidget(btn)
        
        # Connect buttons
        # Note: We need to import logic from tray_app actions or duplicate it
        # Ideally, we call functions directly if they are accessible
        self.play_btn.clicked.connect(self.toggle_play)
        # Prev/Next will need callbacks or access to tray app functions
        # For now, we'll implement simple versions accessing 'state' if possible
        
        right_layout.addLayout(controls_layout)
        
        # Setup Timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(500)  # Update every 500ms
        
        self.update_ui()
        
        # Drag implementation vars
        self.drag_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self.drag_pos:
            self.move(event.globalPosition().toPoint() - self.drag_pos)
            event.accept()

    def update_ui(self):
        # Update Song Info
        if state.current_song_path:
            meta = state.current_song_meta
            title = meta.get('title', os.path.basename(state.current_song_path))
            artist = meta.get('artist', 'Unknown Artist')
            
            # Truncate if too long
            if len(title) > 20: title = title[:18] + "..."
            if len(artist) > 20: artist = artist[:18] + "..."
            
            self.title_label.setText(title)
            self.artist_label.setText(artist)
            
            # Update Play/Pause Icon
            if player.is_playing:
                self.play_btn.setText("⏸")
            else:
                self.play_btn.setText("▶")
                
            # Update Progress
            if not self.is_sliding:
                pos = player.current_position
                dur = player.duration
                if dur > 0:
                    self.slider.setValue(int((pos / dur) * 100))
                    self.current_time_label.setText(self.format_time(pos))
                    self.total_time_label.setText(self.format_time(dur))
                    
        else:
            self.title_label.setText("Not Playing")
            self.artist_label.setText("")
            self.slider.setValue(0)
            
    def format_time(self, seconds):
        if seconds <= 0: return "0:00"
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}:{s:02d}"

    def slider_pressed(self):
        self.is_sliding = True
        
    def seek(self):
        val = self.slider.value()
        if player.duration > 0:
            target_time = (val / 100.0) * player.duration
            player.seek(target_time)
        self.is_sliding = False

    def toggle_play(self):
        if player.is_playing:
            player.pause()
        elif state.current_song_path:
            player.unpause()
        self.update_ui()

# Global app reference
qt_app = None
mini_player_window = None

def launch_miniplayer():
    global qt_app, mini_player_window
    
    if qt_app is None:
        qt_app = QApplication(sys.argv)
    
    if mini_player_window is None:
        mini_player_window = MiniPlayer()
        
    mini_player_window.show()
    # Note: We don't call app.exec() here because we are running in a separate thread
    # or inside the pystray loop. Actually, mixing pystray and PyQt event loops is tricky.
    # We might need to run PyQt as the MAIN loop and pystray in a thread, 
    # OR run PyQt in a separate process.
    # 
    # For simplicity in this architecture:
    # We will try to run QApplication.processEvents() in our main loop or uses threads.
    # BUT standard way for tray app + GUI is: GUI is main, tray is secondary.
    # Since our 'tray_app' is main, launching PyQt window is hard without blocking.
    #
    # Solution: We will run the PyQt event loop in a separate thread? 
    # PyQt requires to be in the MAIN thread usually.
    # 
    # ALTERNATIVE: Rewrite tray_app.py to be a PyQt app that HAS a system tray icon.
    # This is much cleaner and robust. 
    # Given the constraint to not rewrite everything:
    # We will try running PyQt in a Process.
    pass 

if __name__ == "__main__":
    # Test standalone
    app = QApplication(sys.argv)
    window = MiniPlayer()
    window.show()
    sys.exit(app.exec())
