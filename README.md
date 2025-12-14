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
Run `scripts\download_data.py` inside the root folder.
It will download data into `data\balanced-raf-db` directory with subdirs `train`, `val`, and `test`.

#### 2.1.1 Manual download
Should the download script fail, you can download the dataset on [Kaggle](https://www.kaggle.com/datasets/dollyprajapati182/balanced-raf-db-dataset-7575-grayscale). Move the downloaded directories into `data\balanced-raf-db`.

Consult the following result structure as reference.
```text
data/
└── balanced-raf-db/
    ├── train/
    ├── test/
    └── val/
```

### 2.2 Preprocessing
Run `scripts\preprocessing_data.py`. It will flatten the structure of `train`, `test`, and `val` and 
generate multiple label cvs files.