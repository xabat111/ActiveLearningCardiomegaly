# ActiveLearningCardiomegaly
Deep learning has shown great potential for detecting diseases in medical images, but its development requires large amounts of labeled data. This project explores Active Learning strategies to reduce annotation requirements while maintaining performance, using a CheXNet-based model to detect cardiomegaly in chest X-rays.

## Contents

- `FINETUNING_ACTIVE_LEARNING.ipynb` — fine-tuning of the CheXNet (DenseNet-121) model and the Active Learning loop used to select which samples to annotate next.
- `INFERENCIA_SOBRE_EL_MODELO.ipynb` — inference and evaluation of a trained checkpoint, including ROC curves and per-image prediction visualization.
- `sample_labels.csv` — label metadata for the NIH Chest X-ray sample subset (image index, finding labels, patient data and image geometry).
- `models/` — trained checkpoints, stored with Git LFS. See `models/README.md`.

## Running the notebooks (Google Colab)

Both notebooks were written and executed in **Google Colab** with a GPU runtime, so they are not meant to be run as plain local scripts. They start by mounting Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

and then read the dataset and the checkpoint from a Drive folder. To run them:

1. Open the notebook in Colab and select a GPU runtime (Runtime → Change runtime type → GPU).
2. Upload the NIH Chest X-ray sample images and `sample_labels.csv` to your Drive.
3. Update the path variables defined in the first cell so they point at your own Drive folders:
   - `CSV_PATH` — location of `sample_labels.csv`
   - `IMAGES_DIR` — folder containing the PNG images
   - `MODEL_PATH` — starting CheXNet checkpoint (`m-30012020-104001.pth`)

Running outside Colab works too, but you have to drop the `drive.mount` cell and point those same variables at local paths.

## Model weights

The checkpoints are roughly 85 MB, which is above GitHub's 50 MB recommended limit, so they are tracked with [Git LFS](https://git-lfs.com/) via `.gitattributes`. After cloning, run:

```bash
git lfs install
git lfs pull
```

Since the weights are too large for the GitHub web uploader, `models/README.md` describes the two ways to publish them: pushing straight from Colab with Git LFS (the checkpoint is already in Drive), or attaching them to a GitHub Release.
