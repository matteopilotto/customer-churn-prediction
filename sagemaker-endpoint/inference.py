import joblib
import os
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import boto3
import pickle


def model_fn(model_dir):
    """Load the model from the model_dir"""

    model = joblib.load(os.path.join(model_dir, "model.joblib"))

    s3 = boto3.client("s3")

    # Specify your S3 bucket and file name
    bucket_name = "customer-churn-prediction-1"
    file_name = "models/scaler.pkl"
    
    # Download the pickle file from S3
    print("[INFO] Copying scaler...")
    s3.download_file(bucket_name, file_name, "/tmp/scaler.pkl")
    print("[INFO] Successfully copied scaler.")
    
    # Load the StandardScaler from the downloaded file
    with open("/tmp/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    return {"model": model, "scaler": scaler}


def input_fn(input_data, content_type):
    """Read input data"""

    print(f"[INFO] Input Data: {input_data}")
    if content_type == "application/json":
        input_data = json.loads(input_data)
        print(f"[INFO] Input Data Loaded: {input_data}")

    else:
        raise ValueError(f"Unsupported content type: {content_type}")
    
    return input_data

def _preprocess_fn(input_data, scaler):
    print("[INFO] Preparing data for ML...")
    for record in input_data:
        geo = record.pop("Geography")
        record["Geography_France"] = 1 if geo == "France" else 0
        record["Geography_Germany"] = 1 if geo == "Germany" else 0
        record["Geography_Spain"] = 1 if geo == "Spain" else 0
        
        gender = record.pop("Gender")
        record["Gender_Female"] = 1 if gender == "Female" else 0
        record["Gender_Male"] = 1 if gender == "Male" else 0

    input_data = np.array([list(d.values()) for d in input_data])
    
    # Apply scaling to the input data
    print("[INFO] Scaling data for ML...")
    scaled_input = scaler.transform(input_data)
    print("[INFO] Successfully preprocessed data for ML.")

    return scaled_input

def predict_fn(input_data, model):
    """Perform prediction using the loaded model and scaler"""
    
    scaler = model["scaler"]
    clf = model["model"]
    
    scaled_input = _preprocess_fn(input_data, scaler)
    
    # Make predictions using the model
    predictions = clf.predict(scaled_input)
    
    return predictions


        # df = pd.DataFrame.from_dict(input_data)
        # print(f"[INFO] DF Input Data: {df}")
    # elif content_type == "text/csv":
    #     df = pd.read_csv(StringIO(input_data))