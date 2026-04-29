import argparse
import joblib
import shap
import os
import matplotlib.pyplot as plt

from data_loader import load_data
from preprocess import preprocess_data


def explain(model_path, data_path):
    model = joblib.load(model_path)

    df = load_data(data_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    out_path = model_path.split('/')[0]
    plot_path = os.path.join(out_path, "plots")

    os.makedirs(plot_path, exist_ok=True)

    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(plot_path, "shap_summary.png"))
    plt.close()

    print("SHAP plots saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)

    args = parser.parse_args()

    explain(args.model_path, args.data_path)