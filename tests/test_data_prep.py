from src.data_prep import build_pairs
import pandas as pd


def test_build_pairs_basic():
    invoices = pd.DataFrame(
        {
            "invoice_id": [1],
            "customer_id": [10],
            "amount": [100.0],
            "invoice_date": ["2025-01-01"],
            "due_date": ["2025-01-10"],
            "paid_date": ["2025-01-05"],
        }
    )

    payments = pd.DataFrame(
        {
            "payment_id": [101],
            "customer_id": [10],
            "amount": [100.5],
            "payment_date": ["2025-01-06"],
        }
    )

    pairs = build_pairs(invoices, payments)
    assert len(pairs) == 1
    # con amount_diff = 0.5 e days_diff = 1 deve essere match
    assert pairs.iloc[0]["is_match"] == 1
