import argparse
import joblib
import json
import os

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay

from data_loader import load_data
from preprocess import preprocess_data

def evaluate(model_path, data_path):
    model = joblib.load(model_path)

    df = load_data(data_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    out_path = model_path.split('/')[0]

    plots_path=os.path.join(out_path, "plots")

    os.makedirs(plots_path, exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig(os.path.join(plots_path,"confusion_matrix.png"))
    plt.close()

    # ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.savefig(os.path.join(plots_path,"roc_curve.png"))
    plt.close()

    with open(os.path.join(out_path,"metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)

    args = parser.parse_args()

    evaluate(args.model_path, args.data_path)