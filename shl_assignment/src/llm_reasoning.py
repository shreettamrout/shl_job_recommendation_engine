import pandas as pd
from typing import List


def select_recommendations(
    query: str,
    candidates: pd.DataFrame,
    k: int = 5,
) -> pd.DataFrame:
    """
    Final recommendation selection logic.

    This function represents the LLM-based reasoning stage.
    For stability and reproducibility, it currently applies
    deterministic selection over retrieved candidates.

    Args:
        query: Original user query
        candidates: DataFrame of retrieved assessment candidates
        k: Number of final recommendations

    Returns:
        DataFrame with top-k recommended assessments
    """

    if candidates is None or candidates.empty:
        return pd.DataFrame()

    # --- Minimal reasoning logic ---
    # At this stage, retrieval already provides relevance.
    # We keep selection stable and interpretable.

    selected = candidates.head(k).copy()

    return selected

