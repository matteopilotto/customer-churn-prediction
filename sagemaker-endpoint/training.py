
import argparse
import joblib
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    print("extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=3)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train_data.csv")
    parser.add_argument("--test-file", type=str, default="test_data.csv")
    parser.add_argument("--features", type=str)  # in this script we ask user to explicitly name features
    parser.add_argument("--target", type=str)  # in this script we ask user to explicitly name the target

    args, _ = parser.parse_known_args()

    print("[INFO] Reading data...")
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    print("[INFO] Building training and testing datasets...")
    X_train = train_df[args.features.split()]
    X_test = test_df[args.features.split()]
    y_train = train_df[args.target]
    y_test = test_df[args.target]

    # train
    print("[INFO] Training model...")
    model = RandomForestClassifier(n_estimators=args.n_estimators, min_samples_leaf=args.min_samples_leaf, n_jobs=-1)

    model.fit(X_train, y_train)

    # print accuracy
    y_preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_preds)
    print(f"[INFO] {model.__class__.__name__} Accuracy: {accuracy:.2%}")

    # persist model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print("[INFO] Model persisted at " + path)
