"""
preprocess.py
-------------
Cleans reviews and converts numerical ratings to sentiment labels.

Sentiment mapping
-----------------
  Rating 1–2  → Negative
  Rating 3    → Neutral
  Rating 4–5  → Positive
"""

import re
import pandas as pd


def _clean_text(text: str) -> str:
    """Strip excessive whitespace and invisible unicode characters from a review string."""
    if not isinstance(text, str):
        return ""
    # Strip zero-width and invisible unicode characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def rating_to_sentiment(rating: float) -> str:
    """
    Convert a 1–5 star rating to a sentiment label.

    Parameters
    ----------
    rating : float
        Numeric star rating (1.0–5.0).

    Returns
    -------
    str
        'Positive', 'Neutral', or 'Negative'.
    """
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:  # 1 or 2
        return "Negative"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich the raw reviews DataFrame.

    Steps
    -----
    1. Drop rows with missing or empty review_text / parent_asin / rating.
    2. Clean review text (strip whitespace, normalise).
    3. Add a 'sentiment' column derived from the rating.
    4. Reset the index.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame loaded by load_data.load_reviews().

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with an additional 'sentiment' column.
    """
    df = df.copy()

    # Drop rows missing critical fields
    df.dropna(subset=["review_text", "parent_asin", "rating"], inplace=True)

    # Clean text
    df["review_text"] = df["review_text"].apply(_clean_text)

    # Drop rows where text became empty after cleaning
    df = df[df["review_text"].str.len() > 0]

    # Derive sentiment from rating
    df["sentiment"] = df["rating"].apply(rating_to_sentiment)

    df.reset_index(drop=True, inplace=True)

    print(f"[preprocess] {len(df):,} reviews retained after cleaning.")
    sentiment_counts = df["sentiment"].value_counts().to_dict()
    print(f"[preprocess] Sentiment distribution: {sentiment_counts}")

    return df
