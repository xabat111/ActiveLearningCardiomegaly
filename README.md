# ActiveLearningCardiomegaly
Deep learning has shown great potential for detecting diseases in medical images, but its development requires large amounts of labeled data. This project explores Active Learning strategies to reduce annotation requirements while maintaining performance, using a CheXNet-based model to detect cardiomegaly in chest X-rays.

## Contents

- `FINETUNING_ACTIVE_LEARNING.ipynb` — fine-tuning of the CheXNet (DenseNet-121) model and the Active Learning loop used to select which samples to annotate next.
- `INFERENCIA_SOBRE_EL_MODELO.ipynb` — inference and evaluation of a trained checkpoint, including ROC curves and per-image prediction visualization.
- `sample_labels.csv` — label metadata for the NIH Chest X-ray sample subset (image index, finding labels, patient data and image geometry).

The notebooks were written for Google Colab and expect the images and checkpoints under a Google Drive path; update the `*_PATH` / `*_DIR` variables at the top of each notebook to run them elsewhere.
