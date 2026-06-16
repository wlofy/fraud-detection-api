"""Fetch the Kaggle credit-card fraud dataset into ``data/creditcard.csv``.

Requires Kaggle API credentials (``~/.kaggle/kaggle.json`` or the KAGGLE_USERNAME
/ KAGGLE_KEY env vars). If they're absent, this prints manual instructions and
exits 0 — the rest of the pipeline still works on the synthetic fallback.

    python scripts/download_data.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASET = "mlg-ulb/creditcardfraud"
TARGET = DATA_DIR / "creditcard.csv"

MANUAL = f"""
Could not download automatically. To use the REAL dataset:

  1. Create a Kaggle account and an API token (Account → Create New API Token),
     which downloads kaggle.json. Place it at:
         {Path.home() / '.kaggle' / 'kaggle.json'}
  2. Re-run:  python scripts/download_data.py
     (or download manually from
      https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
      and unzip creditcard.csv into {DATA_DIR})

Until then the pipeline trains on a synthetic dataset with the same schema.
"""


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        print(f"Already present: {TARGET}")
        return 0
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore

        api = KaggleApi()
        api.authenticate()
        print(f"Downloading {DATASET} → {DATA_DIR} …")
        api.dataset_download_files(DATASET, path=str(DATA_DIR), quiet=False)
        zip_path = DATA_DIR / "creditcardfraud.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(DATA_DIR)
            zip_path.unlink()
        print(f"Done: {TARGET}" if TARGET.exists() else "Download finished.")
        return 0
    except Exception as exc:  # noqa: BLE001 — any failure → manual fallback
        print(f"[download_data] {type(exc).__name__}: {exc}")
        print(MANUAL)
        return 0


if __name__ == "__main__":
    sys.exit(main())
