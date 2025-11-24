from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Tuple

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def generate_synthetic_raw(n_invoices: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Genera dati sintetici di fatture e pagamenti.
    - Ogni fattura appartiene a un customer_id
    - Una parte delle fatture ha un pagamento con importo simile e data vicina.
    """
    import numpy as np

    rng = np.random.default_rng(seed=42)

    invoice_ids = range(1, n_invoices + 1)
    customer_ids = rng.integers(1, 6, size=n_invoices)  # 5 clienti
    amounts = rng.uniform(50, 500, size=n_invoices).round(2)

    invoice_dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 30, size=n_invoices), unit="D"
    )
    due_dates = invoice_dates + pd.to_timedelta(
        rng.integers(7, 30, size=n_invoices), unit="D"
    )

    # Alcune fatture sono pagate
    is_paid = rng.random(size=n_invoices) < 0.7
    paid_dates = []
    for paid, inv_date in zip(is_paid, invoice_dates):
        if paid:
            lag_days = int(rng.integers(0, 20))
            paid_dates.append(inv_date + pd.to_timedelta(lag_days, unit="D"))
        else:
            paid_dates.append(pd.NaT)
    paid_dates = pd.to_datetime(paid_dates)

    invoices = pd.DataFrame(
        {
            "invoice_id": invoice_ids,
            "customer_id": customer_ids,
            "amount": amounts,
            "invoice_date": invoice_dates,
            "due_date": due_dates,
            "paid_date": paid_dates,
        }
    )

    # Crea pagamenti solo per le fatture pagate, con un po' di rumore su amount/data
    payments_rows = []
    payment_id = 1
    for _, row in invoices[invoices["paid_date"].notna()].iterrows():
        pay_amount = float(row["amount"]) + float(rng.normal(0, 1.0))  # ±1€
        pay_amount = round(pay_amount, 2)
        pay_date = row["paid_date"] + pd.to_timedelta(
            int(rng.integers(-2, 3)), unit="D"
        )
        payments_rows.append(
            {
                "payment_id": payment_id,
                "customer_id": int(row["customer_id"]),
                "amount": pay_amount,
                "payment_date": pay_date,
            }
        )
        payment_id += 1

    payments = pd.DataFrame(payments_rows)

    return invoices, payments


def save_raw(invoices: pd.DataFrame, payments: pd.DataFrame) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    invoices.to_csv(RAW_DIR / "invoices_raw.csv", index=False)
    payments.to_csv(RAW_DIR / "payments_raw.csv", index=False)


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    invoices_path = RAW_DIR / "invoices_raw.csv"
    payments_path = RAW_DIR / "payments_raw.csv"

    if not invoices_path.exists() or not payments_path.exists():
        # Se non esistono ancora, generiamo dati sintetici
        invoices, payments = generate_synthetic_raw()
        save_raw(invoices, payments)
    else:
        invoices = pd.read_csv(invoices_path, parse_dates=["invoice_date", "due_date", "paid_date"])
        payments = pd.read_csv(payments_path, parse_dates=["payment_date"])

    return invoices, payments


def build_pairs(invoices: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    """
    Costruisce un dataset di coppie invoice-payment con una label 'is_match'.
    Regola sintetica per la label (per training):
      - stesso customer_id
      - importo simile (diff <= 2€)
      - data pagamento entro 5 giorni dalla paid_date (se esiste)
    """
    # Join candidato solo per customer_id per limitare combinazioni
    candidates = invoices.merge(
        payments,
        on="customer_id",
        suffixes=("_inv", "_pay"),
        how="inner",
    )

    # Calcola differenze di importo e giorni
    candidates["amount_diff"] = (candidates["amount_inv"] - candidates["amount_pay"]).abs()

    candidates["days_diff"] = (
        pd.to_datetime(candidates["paid_date"]) - pd.to_datetime(candidates["payment_date"])
    ).abs().dt.days

    # Se paid_date è NaT, days_diff sarà NaN → li mettiamo molto grandi
    candidates["days_diff"] = candidates["days_diff"].fillna(9999)

    # Regola per identificare match "veri" (solo per costruire label sintetica)
    candidates["is_match"] = (
        (candidates["amount_diff"] <= 2.0) &
        (candidates["days_diff"] <= 5)
    ).astype(int)

    return candidates


def main() -> None:
    invoices, payments = load_raw_data()
    pairs = build_pairs(invoices, payments)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "pairs_train.csv"
    pairs.to_csv(out_path, index=False)
    print(f"Saved {len(pairs)} candidate pairs to {out_path}")


if __name__ == "__main__":
    main()
