# facial-emotion-classifier
Emotion recognition project for CVDL: training a CNN from scratch to classify six facial emotions, visualizing model decisions with Explainable AI (Grad-CAM), and building a video/webcam demo pipeline.

## 1. Setting up the environment
Follow these steps to set up the virtual environment and install all required dependencies.
### 1.1 Create and activate the virtual environment
### Windows
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

You may need to set this execution policy in order for PS to accept the activation.
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 1.2 Install dependencies
```bash
pip install -r requirements.txt
```


## 2. Downloading and preprocessing the data
### 2.1 Downloading
Run `scripts/download_data_fer_2013.py` inside the root folder.
It will create a `data` directory with subdirs `raw` and `processed` and download data into `raw`.

#### 2.1.1 Manual download
placeholder insert manual download text

### 2.2 Preprocessing
placeholder insert preprocessing text
