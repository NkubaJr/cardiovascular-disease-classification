# Cardiovascular Disease Classification: Traditional ML vs Deep Learning

Summative assignment, Machine Learning (Model Training and Evaluation).
This project predicts the presence of cardiovascular disease from routine clinical and lifestyle measurements and compares scikit-learn models with Keras neural networks under one evaluation protocol.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NkubaJr/cardiovascular-disease-classification/blob/main/cardiovascular_disease_classification.ipynb)

## Links
- Written report: see the submitted PDF
- Google Drive folder (raw dataset zip, best model weights, experiment logs): YOUR_DRIVE_LINK
- Video presentation: YOUR_VIDEO_LINK

## Dataset
Cardiovascular Disease dataset by Svetlana Ulianova, Kaggle: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
70,000 records, 11 predictors, binary target `cardio`. The raw data is **not** stored in this repository; the notebook downloads it with `kagglehub`, and a zipped copy is in the Drive folder above.

After cleaning (duplicates and implausible blood pressure, height and weight values removed), 67,936 records remain, split 70/15/15 (train/validation/test), stratified, seed 42.

## Method
- Four engineered features: BMI, pulse pressure, mean arterial pressure, blood-pressure stage
- 12 experiments: logistic regression, random forest, histogram gradient boosting (scikit-learn, with randomised and grid search), and five Keras networks (Sequential and Functional API, `tf.data` pipeline, dropout, batch normalisation, L2, learning-rate scheduling)
- Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC and precision-recall curves, learning curves, threshold analysis, permutation importance
- Logging: Weights & Biases (offline mode) and TensorBoard

## Main results (held-out test set, 10,191 records)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Histogram gradient boosting (E4) | 0.7335 | 0.7438 | 0.7080 | 0.7254 | 0.7966 |
| Functional neural network (E10) | 0.7341 | 0.7550 | 0.6888 | 0.7204 | 0.7966 |

Neither approach gave a meaningful advantage on this dataset. The default random forest (E3) overfitted (train-validation AUC gap 0.233) until its depth was restricted.

## How to run
1. Click the "Open in Colab" badge above (or upload the notebook to Google Colab).
2. Choose Runtime, then Run all. The first cell installs dependencies; the data cell downloads the dataset from Kaggle, or asks you to upload `cardio_train.csv` if the download fails.
3. Run time is roughly 20 minutes. All random seeds are fixed (42), so results are reproducible.

To run locally, install the packages in `requirements.txt` and open the notebook in Jupyter.

## Files
- `cardiovascular_disease_classification.ipynb`: full pipeline (loading, preprocessing, experiments, evaluation, error analysis)
- `requirements.txt`: dependencies

## Acknowledgements
Dataset from Kaggle (Ulianova). Built with scikit-learn, TensorFlow/Keras, Weights & Biases and TensorBoard. AI assistance was used for guidance on code and experiment design; the analysis and report were written by the author.

## Disclaimer
This is a research exercise on a public dataset, not a validated diagnostic tool.
