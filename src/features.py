import pandas as pd


def create_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Trasforma il dataset pairs in una matrice di feature pulita.
    """
    df = pairs.copy()

    # Feature numeriche semplici
    df["amount_diff_norm"] = df["amount_diff"] / (df["amount_inv"] + 1e-6)

    # Days diff già calcolata in data_prep
    df["days_diff"] = df["days_diff"].fillna(999)

    # Se vuoi, aggiungi flag booleani
    df["same_amount_flag"] = (df["amount_diff"] <= 0.5).astype(int)

    # Se 'is_match' esiste, la teniamo come label
    return df[[
        "amount_diff",
        "amount_diff_norm",
        "days_diff",
        "same_amount_flag",
        "is_match"
    ]]
