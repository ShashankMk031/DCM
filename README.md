# DCM - Dynamic Content Music

A powerful music player and playlist generator that creates personalized playlists based on your music preferences and listening habits. Features intelligent recommendations, seamless playback, and a beautiful Material Design interface.

## Features

- **Smart Playback**: Seamless music playback with auto-advance and continuous play
- **Intelligent Recommendations**: Get personalized song suggestions based on your listening history
- **Auto-Recommendations**: Automatically generates new recommendations when your playlist ends
- **Modern UI**: Clean, responsive interface built with KivyMD
- **Cross-Platform**: Runs on Windows, macOS, Linux, and Android
- **Audio Analysis**: Advanced audio processing for accurate music recommendations
- **Playlist Management**: Create, save, and manage your playlists with ease

## Installation

1. Clone the repository
   ```bash
   git clone https://github.com/ShashankMk031/DCM.git
   cd DCM
   ```

2. Set up the environment
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```
   
3. Set up the music library
   - Create a directory named 'music_library' in the project root
   - Add your music files to this directory
   - The application will automatically scan and index these files on first run

## Usage

1. Start the application
   ```bash
   python -m dcm.ui.main
   ```

2. Basic Navigation
   - **Playlists**: View and manage your saved playlists
   - **Library**: Browse your music library
   - **Recommend**: Get song recommendations
   - **Generate**: Create new playlists based on your preferences

3. Playback Controls
   - Click on any song to start playback
   - Use the media controls to play/pause, skip, or adjust volume
   - The app will automatically generate new recommendations when your playlist ends

## Project Structure

```
dcm/
├── ui/                  # User interface components
│   ├── main.py         # Main application entry point
│   ├── main_screen.py  # Main screen implementation
│   └── main.kv         # UI layout and styling
├── core/               # Core functionality
│   ├── __init__.py
│   ├── player.py       # Music playback and playlist management
│   ├── extract_features.py  # Audio feature extraction
│   └── playlist_generator.py  # Playlist generation logic
├── database/           # Database models and operations
│   └── database.py     # Database connection and queries
└── music_library/      # User's music collection (not version controlled)
    └── ...             # Music files go here
```

## Dependencies

- Python 3.8+
- Kivy 2.3.1
- KivyMD 1.1.1
- Librosa 0.11.0
- NumPy 1.26.4
- Pandas 1.5.3
- SoundFile 0.13.1
- SciPy 1.13.0
- tqdm 4.67.1

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Kivy](https://kivy.org/) - The cross-platform Python framework
- [KivyMD](https://kivymd.readthedocs.io/) - Material Design widgets for Kivy
- [Librosa](https://librosa.org/) - Audio and music analysis in Python
- [SoundFile](https://pysoundfile.readthedocs.io/) - Audio file reading/writing
- [NumPy](https://numpy.org/) - Numerical computing with Python
- [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis

## Note

Create a 'music_library' directory in the project root and add your music files there. The application will automatically scan and index these files on first run. This directory is included in .gitignore to prevent large music files from being committed to version control.