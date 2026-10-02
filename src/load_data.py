"""
load_data.py
------------
Loads the Amazon Electronics Reviews dataset from a local Parquet file.

Dataset: Amazon Reviews 2023 – Electronics (sampled, 7,109 reviews)
Source:  https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
"""

import pandas as pd
from pathlib import Path

# Default path relative to the project root
DEFAULT_PARQUET_PATH = Path(__file__).resolve().parent.parent / "data" / "train-00000-of-00001.parquet"

# Required columns for contradiction detection
REQUIRED_COLUMNS = ["parent_asin", "product_title", "rating", "review_text"]


def load_reviews(parquet_path: str | Path = DEFAULT_PARQUET_PATH) -> pd.DataFrame:
    """
    Load reviews from the local Parquet file.

    Parameters
    ----------
    parquet_path : str or Path
        Path to the .parquet file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing at least the four required columns.

    Raises
    ------
    FileNotFoundError
        If the Parquet file does not exist at the given path.
    KeyError
        If any required column is missing from the dataset.
    """
    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {parquet_path}\n"
            "Download instructions: see README.md → Dataset section."
        )

    df = pd.read_parquet(parquet_path, columns=REQUIRED_COLUMNS)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Required columns missing from dataset: {missing}")

    print(f"[load_data] Loaded {len(df):,} reviews from '{parquet_path.name}'.")
    return df
