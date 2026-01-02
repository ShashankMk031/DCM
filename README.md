# DCM - AI-Powered Music Recommendation System

<div align="center">

![DCM Logo](assets/icon.png)

**Intelligent music recommendations using audio feature analysis and machine learning**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## Features

- **AI-Powered Recommendations** - Uses audio feature analysis with k-NN to find similar songs
- **Smart Queue System** - Auto-generates 10-song queues with refresh after 6 songs
- **M4A Support** - Automatic transcoding to WAV via ffmpeg for seamless playback
- **3 Playback Modes**:
  - **Loop** - Repeat current song
  - **Auto-Queue** - AI-recommended songs
  - **Sequential** - Folder-based playback
- **System Tray App** - Lives in your menu bar (macOS) or system tray (Linux)
- **Offline** - No internet required, all processing local

---

## Installation

### Prerequisites
- Python 3.8+
- ffmpeg (for M4A support)

### macOS
```bash
# Install ffmpeg
brew install ffmpeg

# Clone and setup
git clone https://github.com/ShashankMk031/DCM.git
cd DCM
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Linux (Debian/Ubuntu)
```bash
# Install dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv ffmpeg

# Clone and setup
git clone https://github.com/ShashankMk031/DCM.git
cd DCM
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Linux (Fedora)
```bash
# Install dependencies
sudo dnf install python3 python3-pip ffmpeg

# Clone and setup
git clone https://github.com/ShashankMk031/DCM.git
cd DCM
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Start

### 1. Extract Audio Features

```bash
# Extract features from your music library
./venv/bin/python3 -m dcm.core.extract_features ~/Music -o features/all_features.csv -v -w 1
```

This analyzes your music and creates a database of audio features (MFCCs, spectral features, etc.) for recommendations.

**Time**: ~1-2 minutes for 100 songs

### 2. Run the Tray App

```bash
./venv/bin/python3 -m dcm.tray_app
```

The DCM icon will appear in your system tray/menu bar!

### 3. Play Music

1. Click the DCM icon
2. Select "📁 Open Song..."
3. Choose a song
4. Enable "Auto-Queue" for AI recommendations!

---

## Usage

### Tray Menu Controls
- **Open Song...** - Select a song to play
- **Play/Pause** - Toggle playback
- **Previous** - Go to previous song
- **Next** - Play next recommended song
- **Loop** - Repeat current song
- **Auto-Queue** - Enable AI recommendations
- **Queue** - View upcoming songs (when Auto-Queue is on)

### Playback Modes

**Auto-Queue OFF + Loop OFF (Default)**
- Plays songs sequentially from folder
- Auto-advances to next folder when done

**Auto-Queue ON**
- AI chooses similar songs based on audio features
- 10-song queue, refreshes after 6 songs
- Excludes recently played to prevent loops

**Loop ON**
- Repeats current song endlessly
- Auto-Queue automatically disabled

---

## How It Works

### Feature Extraction
DCM analyzes audio files and extracts:
- **MFCCs** (Mel-frequency cepstral coefficients) - Timbre/texture
- **Spectral Features** (centroid, bandwidth, rolloff) - Brightness/tonality
- **RMS Energy** - Loudness/dynamics
- **Zero-Crossing Rate** - Noisiness

### Recommendation Engine
1. All features normalized and weighted
2. PCA reduces dimensions (keeps 95% variance)
3. k-NN finds 5 nearest neighbors
4. Recommendations filtered by play history

---

## Project Structure

```
DCM/
├── dcm/
│   ├── core/
│   │   ├── extract_features.py  # Audio analysis
│   │   ├── suggest_next.py      # Recommendation engine
│   │   └── metadata.py          # ID3 tag extraction
│   ├── player.py                # Pygame audio player
│   └── tray_app.py             # System tray application
├── features/
│   └── all_features.csv        # Feature database
├── assets/
│   └── icon.png                # App icon
└── requirements.txt
```

---

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built with [librosa](https://librosa.org/) for audio analysis
- [pygame](https://www.pygame.org/) for playback
- [pystray](https://github.com/moses-palmer/pystray) for system tray
- [scikit-learn](https://scikit-learn.org/) for ML

---

## Support

Have questions or issues? [Open an issue](https://github.com/ShashankMk031/DCM/issues) on GitHub!

---

<div align="center">

[⭐ Star this repo](https://github.com/ShashankMk031/DCM) if you find it useful!

</div>