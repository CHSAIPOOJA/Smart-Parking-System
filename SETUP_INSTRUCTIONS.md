# Smart Parking Project - Setup Guide

## Installation (Run Once)

### Option 1: Automatic Installation (Windows)
```
Double-click: install_requirements.bat
```

### Option 2: Manual Installation (PowerShell)
```powershell
cd 'D:\btech projects\SmartParkingProject'
pip install -r requirements.txt
```

## Running the Program

### Using PowerShell:
```powershell
cd 'D:\btech projects\SmartParkingProject'
python code/main.py
```

### Using the Batch File:
```
Double-click: run.bat
```

## What's Fixed

✅ **Permanent Dependencies**: `requirements.txt` contains all required packages
✅ **Virtual Environment**: Uses isolated Python environment (.venv)
✅ **Empty Slot Calculation**: Fixed logic to accurately count free parking slots
✅ **Color Coding**: 
   - GREEN = Empty/Free Slot
   - RED = Occupied Slot
✅ **Frame Counter**: Shows frame number for debugging

## Dependencies Installed

- scikit-image (0.22.0) - Image processing
- opencv-python (4.8.1.78) - Video/image handling
- numpy (1.24.3) - Numerical computations
- Pillow (10.1.0) - Image support

## Troubleshooting

If you still get `ModuleNotFoundError`:
1. Delete the `.venv` folder
2. Run: `python -m venv .venv`
3. Run: `pip install -r requirements.txt`
4. Run your program again

The error will NOT come back with these fixes in place.
