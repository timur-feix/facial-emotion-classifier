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

Important: All scripts in this repo are intended to be ran as modules, not as standalone scripts.

Correct:
```sh
python -m src.inference --args...
```

Incorrect:
```sh
python src\inference.py --args...
```


### 2.1 Downloading
Run `scripts.data_utils.download_data` as a module inside the root folder.
It will download data into `data\balanced-raf-db` directory with subdirs `train`, `val`, and `test`.

Arguments: None.

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
Run `scripts.data_utils.preprocess_data`. It will flatten the structure of `train`, `test`, and `val` and 
generate label cvs files for each split.

Arguments: None.

## 3 Using the model
### 3.1 Training
Now, the world is your oyster. You can train the custom `FacialEmotionRecognitionCNN` by running `src.train`.

Required arguments: None.

Optionial arguments:

`--data-dir` - Path to data folder, default is `~\data\balanced-raf-db`.

`--out-dir` - Path to folder that saves checkpoints `best.pt`, `last.pt`, default is `~\checkpoints`.

`--epochs` - Number of epochs, default is 10.

`--batch-size` - Batch size, default is 64.

`--lr` - Learning rate (start), default is `1e-3`.

`--num-workers` - Workers for data loader, default is 4.

`--resume` - Use to resume from a saved checkpoint, default is None.

`--debug` - (``action = "store_true"``) This flag prints a timer for each epoch, default is False.

`--sched` - LR scheduler, default is cosine, choices are none, cosine, and onecycle.

`--warmup` - Warmup fraction of total steps (cosine), default is 0.05.

`--min-lr` - Lowest learning rate for cosine decay, default is `2e-5`.

### 3.2 Testing
Test the model by running `src.eval`. It will run the test data through the model with `checkpoints\best.pt` as the weights.

Arguments: None.

### 3.3 Inference
Run `src.inference` to generate a CSV file with the corresponding logits for each image in a directory.

Required arguments:

`--input-dir` - Path to the input directory with images to score.

Optional arguments:

`--weights` - Model weights, default is `~\checkpoints\best.pt`.

`--output-csv` - Output CSV, default is `predictions.csv`.

`--batch-size` - Batch size, default is 256.

## 4 Demo
### 4.1 Video Demo
Run `scripts.video_demo` to generate a video with the model prediction overlay and the gradCAM - heatmap.

Required arguments:

`--input-file` - File path of video to be processed.

Optional arguments:

``--output-dir` - Directory to store processed video, default is `~\videos`.

``--weights`` - Weights to be loaded into model, default is `~\checkpoints\best.pt`.

``--gradcam`` - (``action = "store_true"``) Render gradCAM - heatmap.

### 4.2 Webcam Demo
To start the webcam demo, run `scripts.webcam_demo`. This will start a program overlaying the model predictions to your webcam feed.
Press `G` on your keyboard to toggle the gradCAM - heatmap overlay on and off.
Press `Q` to quit the program.

Arguments: None.