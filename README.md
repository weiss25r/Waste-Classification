# Waste Classification

## Description
The project consists in a full pipeline, from data collection to deployement, with output a deep learning model capable of classifying waste in one of six categories:
- plastic
- glass
- metal
- paper
- cardboard
- trash

Waste classification could be performed by low-powered devices: Consider, for example, a smart trash bin that scans waste and sorts it into the correct containers based on its classification. Such a system could be equipped with an embedded device with low computing power. Considering similiar cases, the MobileNet architecture was chosen, as it is small and properly designed to run on mobile and embedded systems, making it a right choice. MobileNetV3 was fine-tuned on a waste dataset with good result.

## Tech stack

**Data collection:**
- Kaggle API
- Pandas

**Training**:
- Pytorch Lightning

**Logging**:
- Tensorboard

**Inference**
- ONNX
- FastAPI

## Project Structure
Waste Classification
 ┣ 📂models
 ┃ ┣ 📜model_large.onnx
 ┃ ┗ 📜model_small.onnx
 ┣ 📂notebooks
 ┃ ┣ 📜dataset_details.ipynb
 ┃ ┣ 📜export_model.ipynb
 ┃ ┣ 📜inference_onnx.ipynb
 ┃ ┣ 📜mean_std.ipynb
 ┃ ┣ 📜prepare_dataset.ipynb
 ┃ ┗ 📜train_models.ipynb
 ┣ 📂training
 ┃ ┣ 📂checkpoints
 ┃ ┃ ┣ 📂waste_cls_jitter_hue_small
 ┃ ┃ ┃ ┣ 📜waste_cls_jitter_hue_small_best.ckpt
 ┃ ┃ ┃ ┗ 📜waste_cls_jitter_hue_small_last.ckpt
 ┃ ┃ ┣ 📂waste_cls_jitter_large
 ┃ ┃ ┃ ┣ 📜waste_cls_jitter_large_best.ckpt
 ┃ ┃ ┃ ┗ 📜waste_cls_jitter_large_last.ckpt
 ┃ ┃ ┣ 📂waste_cls_jitter_small
 ┃ ┃ ┃ ┣ 📜waste_cls_jitter_small_best.ckpt
 ┃ ┃ ┃ ┗ 📜waste_cls_jitter_small_last.ckpt
 ┃ ┃ ┗ 📂waste_cls_jitter_vf_small
 ┃ ┃ ┃ ┣ 📜waste_cls_jitter_vf_small_best.ckpt
 ┃ ┃ ┃ ┗ 📜waste_cls_jitter_vf_small_last.ckpt
 ┃ ┣ 📂config
 ┃ ┃ ┣ 📜config_large_v1.yaml
 ┃ ┃ ┣ 📜config_small_v1.yaml
 ┃ ┃ ┣ 📜config_small_v2.yaml
 ┃ ┃ ┣ 📜config_small_v3.yaml
 ┃ ┃ ┗ 📜config_tests.yaml
 ┃ ┗ 📂metrics
 ┃ ┃ ┣ 📂waste_cls_jitter_hue_small
 ┃ ┃ ┃ ┗ 📂version_1
 ┃ ┃ ┃ ┃ ┣ 📜hparams.yaml
 ┃ ┃ ┃ ┃ ┗ 📜metrics.csv
 ┃ ┃ ┣ 📂waste_cls_jitter_hue_small_best
 ┃ ┃ ┃ ┗ 📂version_1
 ┃ ┃ ┃ ┃ ┣ 📜hparams.yaml
 ┃ ┃ ┃ ┃ ┗ 📜metrics.csv
 ┃ ┃ ┣ 📂waste_cls_jitter_large
 ┃ ┃ ┃ ┗ 📂version_1
 ┃ ┃ ┃ ┃ ┣ 📜hparams.yaml
 ┃ ┃ ┃ ┃ ┗ 📜metrics.csv
 ┃ ┃ ┣ 📂waste_cls_jitter_large_best
 ┃ ┃ ┃ ┗ 📂version_1
 ┃ ┃ ┃ ┃ ┣ 📜hparams.yaml
 ┃ ┃ ┃ ┃ ┗ 📜metrics.csv
 ┃ ┃ ┣ 📂waste_cls_jitter_small
 ┃ ┃ ┃ ┗ 📂version_1
 ┃ ┃ ┃ ┃ ┣ 📜hparams.yaml
 ┃ ┃ ┃ ┃ ┗ 📜metrics.csv
 ┃ ┃ ┣ 📂waste_cls_jitter_small_best
 ┃ ┃ ┃ ┗ 📂version_1
 ┃ ┃ ┃ ┃ ┣ 📜hparams.yaml
 ┃ ┃ ┃ ┃ ┗ 📜metrics.csv
 ┃ ┃ ┣ 📂waste_cls_jitter_vf_small
 ┃ ┃ ┃ ┗ 📂version_1
 ┃ ┃ ┃ ┃ ┣ 📜hparams.yaml
 ┃ ┃ ┃ ┃ ┗ 📜metrics.csv
 ┃ ┃ ┗ 📂waste_cls_jitter_vf_small_best
 ┃ ┃ ┃ ┗ 📂version_1
 ┃ ┃ ┃ ┃ ┣ 📜hparams.yaml
 ┃ ┃ ┃ ┃ ┗ 📜metrics.csv
 ┣ 📂wastenet
 ┃ ┣ 📜dataset.py
 ┃ ┣ 📜inference.py
 ┃ ┣ 📜model.py
 ┃ ┣ 📜tests.ipynb
 ┃ ┣ 📜test_metrics.ipynb
 ┃ ┣ 📜train.py
 ┃ ┗ 📜__init__.py
 ┣ 📜.gitignore
 ┣ 📜README.md
 ┣ 📜requirements.txt

## Data
Data collection includes downloading and merging two datasets from Kaggle. For each class, at least 800 examples were collecting. The datasets used are the following:
- [TrashNet dataset with annotations](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification?select=Garbage+classification)
- [Garbage Classification](www.kaggle.com/datasets/mostafaabla/garbage-classification)

To download and the prepare data, the notebook prepare_dataset.ipynb is used. Using Kaggle API, Pandas and other system libraries, the two dataset are downloaded, classes of interest from the second dataset are selected, the two datasets are merged and annotations for train, validation e testing set are produced. The dataset is saved in data/.

## Training
Pytorch Lightning was used for training the network. The directory wastenet/ and its files are related to training as described above:

- dataset.py: contains implementation of a DataLoader and a DataModule for loading the dataset;

- model.py: contains implementation of a LightningModule, which contains all methods about training, testing and logging.

- train.py: contains the WasteClassifierTrainer class, which is a wrapper for a Pytorch Lightning Trainer object . This class is useful for repating the training process with different configurations of hyperparameters and augmentation using a YAML file.

## ONNX

## Experiments
Experiments are described in detail in docs/report.pdf. Inside the training/ folder are available metrics, checkpoints and configuration files for each conducted experiment.

## Inference
Inference is served by the InferenceSession class, that loads a MobileNet exported in ONNX to run inference on Images. This class can be used directly with the notebook inference_onnx.ipynb or with the FastAPI server The API was tested on a **Raspberry Pi 4** (4 GB), which is capable of running the "large" version of the trained model.
### FastAPI
A FastAPI server for prediction was made with endopoint /predict. To run the API run the following command in the project directory:
### Notebook
fastapi run app/app.py

You can then POST images for inference.
## Cross-platform app
Part of the project is the [WasteScanner]() app, a front-end for the API. It consists in a simple cross platform that lets users pick an image for prediction. The app was built with React-Native and thus can ran in Web, Android and iOS.
