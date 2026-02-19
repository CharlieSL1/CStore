# CStore Web App

Local web interface for generating Csound (.csd) files and audio using the CStore V1.0.1 model.

## Requirements

- Python 3.11+
- [Csound](https://csound.com/) 6.18.0 (for audio rendering)
- Model checkpoint: Download from [Releases](https://github.com/CharlieSL1/CStore/releases) and place at `model/checkpoints/Cstore_V1.0.1/best`

## Run

```bash
cd CStore
pip install -r requirements.txt
python webapp/app.py
```

Open http://127.0.0.1:5000 in your browser.

## Features

- **Generate** — Click to generate CSD and render audio (uses best model V1.0.1)
- **FFT display** — Real-time frequency spectrum when playing audio
- **CSD viewer** — View generated source with copy button
- **Generated Files** — Browse and load previous outputs (click to view)
- **Csound IDE link** — Open in [Csound Web IDE](https://ide.csound.com/editor/OE3qtlvC0RKjq47vPDdx) to edit and run
- **Storage** — All generated files saved in `webapp/generated/<run_id>/` (output.csd, output.wav)
