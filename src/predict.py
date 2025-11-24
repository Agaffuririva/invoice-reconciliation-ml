import pandas as pd
import joblib
from pathlib import Path
from .features import create_features

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def predict_single(invoice, payment):
    """
    invoice e payment devono essere dizionari con campi:
      amount_inv, amount_pay, days_diff, amount_diff
    """
    model_path = MODEL_DIR / "match_model.pkl"
    model = joblib.load(model_path)

    df = pd.DataFrame([{
        "amount_inv": invoice["amount"],
        "amount_pay": payment["amount"],
        "amount_diff": abs(invoice["amount"] - payment["amount"]),
        "days_diff": abs((pd.to_datetime(invoice["paid_date"]) - pd.to_datetime(payment["payment_date"])).days),
        "is_match": 0  # placeholder
    }])

    df_feat = create_features(df).drop(columns=["is_match"])

    proba = model.predict_proba(df_feat)[0][1]
    pred = model.predict(df_feat)[0]

    return {
        "prediction": int(pred),
        "probability": float(proba)
    }


if __name__ == "__main__":
    # Esempio veloce
    invoice = {"amount": 120.0, "paid_date": "2025-01-15"}
    payment = {"amount": 119.5, "payment_date": "2025-01-16"}
    print(predict_single(invoice, payment))
