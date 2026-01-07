import joblib
import numpy as np
import pandas as pd
import os

# ---------------- Load Model Safely ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "fraud_detection_model.joblib")

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
feature_names = model_data["feature_names"]

# ---------------- Input Preparation ----------------
def prepare_input(
    step,
    tx_type,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest
):
    # ---- Type safety (important for Streamlit inputs)
    step = int(step)
    amount = float(amount)
    oldbalanceOrg = float(oldbalanceOrg)
    newbalanceOrig = float(newbalanceOrig)
    oldbalanceDest = float(oldbalanceDest)
    newbalanceDest = float(newbalanceDest)

    # ---- Prevent negative values (safety)
    oldbalanceOrg = max(oldbalanceOrg, 0)
    newbalanceOrig = max(newbalanceOrig, 0)
    oldbalanceDest = max(oldbalanceDest, 0)
    newbalanceDest = max(newbalanceDest, 0)

    input_data = {
        "step": step,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,

        # -------- Engineered Features --------
        "balanceDiffOrg": oldbalanceOrg - newbalanceOrig,
        "balanceDiffDest": newbalanceDest - oldbalanceDest,

        # -------- One-Hot Encoding --------
        "type_TRANSFER": 1 if tx_type == "TRANSFER" else 0,
        "type_CASH_OUT": 1 if tx_type == "CASH_OUT" else 0,
        "type_PAYMENT": 1 if tx_type == "PAYMENT" else 0,
        "type_DEBIT": 1 if tx_type == "DEBIT" else 0,
        "type_CASH_IN": 1 if tx_type == "CASH_IN" else 0,
    }

    df = pd.DataFrame([input_data])

    # ---- Ensure correct feature order
    df = df.reindex(columns=feature_names, fill_value=0)

    # ---- Final safety check
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    return df


# ---------------- Prediction Function ----------------
def predict_fraud(
    step,
    tx_type,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest
):
    input_df = prepare_input(
        step,
        tx_type,
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest
    )

    prediction = model.predict(input_df)[0]

    # ---- Handle models without predict_proba
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_df)[0][1]
    else:
        probability = 0.0

    return {
        "is_fraud": int(prediction),
        "fraud_probability": round(probability, 4)  # UI formats it as %
    }
