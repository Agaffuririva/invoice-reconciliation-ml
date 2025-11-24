from src.train_model import train_model
from pathlib import Path

def test_train_model_runs():
    train_model()
    assert Path("models/match_model.pkl").exists()
