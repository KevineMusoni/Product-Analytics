"""Loading and light cleaning of the two public datasets used in this project."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def load_cookie_cats() -> pd.DataFrame:
    """Cookie Cats A/B test: one row per player (90,189 players)."""
    return pd.read_csv(RAW / "cookie_cats.csv")


def load_cdnow_transactions() -> pd.DataFrame:
    """CDNOW transaction log: 69,659 purchases by 23,570 customers, Jan 1997 to Jun 1998.

    The raw file is whitespace-aligned text; this returns a tidy frame and
    also writes ``data/processed/cdnow_transactions.csv`` for the SQL queries.
    """
    tx = pd.read_csv(RAW / "CDNOW_master.txt", sep=r"\s+", dtype={"customer_id": str, "date": str})
    tx["date"] = pd.to_datetime(tx["date"], format="%Y%m%d")
    tx = tx.rename(columns={"number_of_cds": "units", "dollar_value": "value"})
    PROCESSED.mkdir(parents=True, exist_ok=True)
    tx.to_csv(PROCESSED / "cdnow_transactions.csv", index=False, date_format="%Y-%m-%d")
    return tx
