import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

from .features import create_features
from .data_prep import main as data_prep_main  # 👈 importiamo il data prep

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def train_model():
    """
    Carica pairs_train.csv → se non esiste lo genera → crea feature → addestra modello → salva modello.
    """
    pairs_path = DATA_DIR / "processed" / "pairs_train.csv"

    # 👇 Se il file non esiste (caso CI pulita), generiamo i dati
    if not pairs_path.exists():
        print("[TRAIN] pairs_train.csv non trovato, lancio data_prep...")
        data_prep_main()  # questo genera il file in data/processed

    df = pd.read_csv(pairs_path)

    # Feature engineering centralizzato
    df_feat = create_features(df)

    X = df_feat.drop(columns=["is_match"])
    y = df_feat["is_match"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("=== Evaluation ===")
    print(classification_report(y_test, preds))

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "match_model.pkl"
    joblib.dump(model, model_path)

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    train_model()
