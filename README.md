## Setup

Python: 3.11.9

pip install -r requirements.txt


# Download the dataset using below script and keep it inside data folder

import kagglehub
path = kagglehub.dataset_download("ketangangal/heart-disease-dataset-uci")
print("Path to dataset files:", path)


## Perform EDA and Pre-processing

Use eda_and_preprocess.ipynb to perform EDA and preprocessing. This step cleans the raw data and generates cleaned_data.csv inside data directory. cleaned_data.csv is used in the following steps.

## Train
RandomForestClassifier:  python src/train.py --data_path data/cleaned_data.csv --model rf
LogisticRegression:   python src/train.py --data_path data/cleaned_data.csv --model logreg
SVM:   python src/train.py --data_path data/cleaned_data.csv --model logreg


## Evaluate
RandomForestClassifier:  python src/evaluate.py --model_path outputs_rf/model.pkl --data_path data/cleaned_data.csv
LogisticRegression:   python src/evaluate.py --model_path outputs_logreg/model.pkl --data_path data/cleaned_data.csv
SVM:  python src/evaluate.py --model_path outputs_svm/model.pkl --data_path data/cleaned_data.csv


## Explainability

RandomForestClassifier: python src/explain.py --model_path outputs_rf/model.pkl --data_path data/cleaned_data.csv
LogisticRegression: python src/explain.py --model_path outputs_logreg/model.pkl --data_path data/cleaned_data.csv
