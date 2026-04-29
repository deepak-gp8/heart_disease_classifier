import argparse
import joblib
import os

from data_loader import load_data
from preprocess import preprocess_data
from model import get_model


def train(data_path, model_name):
    df = load_data(data_path)

    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    model = get_model(model_name)
    model.fit(X_train, y_train)

    output_dir = "outputs_{}".format(model_name)

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir,"model.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    print("Model trained and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model", default="rf")

    args = parser.parse_args()

    train(args.data_path, args.model)