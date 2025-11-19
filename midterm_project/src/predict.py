import sys
import json
import joblib
import pandas as pd

def load_model():
    # Load the saved model, vectorizer and feature lists
    model_data = joblib.load("models/best_model.pkl")
    return model_data["model"], model_data["dv"], model_data["categorical"], model_data["numerical"]

def preprocess(data, categorical, numerical):
    """Converts raw input data into the format required by DictVectorizer."""
    df = pd.DataFrame([data])

    # Ensure all numerical columns exist
    for col in numerical:
        if col not in df:
            df[col] = 0

    # Ensure all categorical columns exist
    for col in categorical:
        if col not in df:
            df[col] = ""

    # Keep only the required columns
    df = df[categorical + numerical]

    return df.to_dict(orient="records")

if __name__ == "__main__":
    # Run:
    # python predict.py '{"Age":41,"BusinessTravel":"Travel_Rarely","OverTime":"Yes", ... }'
    
    raw = sys.argv[1]
    data = json.loads(raw)

    model, dv, categorical, numerical = load_model()

    record = preprocess(data, categorical, numerical)

    X = dv.transform(record)
    pred = model.predict_proba(X)[0, 1]

    print(pred)
