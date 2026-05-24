# Organ Transplant Model Web App

This project runs a Flask web app that evaluates a small sample text classification model for organ transplant outcome prediction.

## Setup

1. Open PowerShell in this folder:
   ```powershell
   cd "C:\Users\chand\OneDrive\Desktop\Organ Transplant"
   ```

2. Install Python dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Install the spaCy English model:
   ```powershell
   python -m spacy download en_core_web_sm
   ```

## Run the app

```powershell
python main.py
```

Then open this URL in your browser:

```
http://127.0.0.1:5000
```

## Notes

- `main.py` starts a Flask server and uses a trained PyTorch model.
- If you get a dependency error, run the install command again.
- If you want to stop the app, press `Ctrl+C` in the terminal.
