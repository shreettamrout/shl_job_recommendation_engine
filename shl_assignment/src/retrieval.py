import numpy as np
import pandas as pd
import faiss
from typing import Tuple


class DenseRetriever:
    """
    Handles FAISS-based dense retrieval over assessment embeddings.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
    ):
        self.embeddings = embeddings
        self.metadata = metadata.reset_index(drop=True)

        self._build_index()

    def _build_index(self):
        """
        Build FAISS index using cosine similarity (inner product).
        """
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def search(
        self,
        query_embedding: np.ndarray,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Search FAISS index and return top-N metadata rows.

        Args:
            query_embedding: Normalized query embedding (1 x dim)
            top_n: Number of candidates to retrieve

        Returns:
            DataFrame of top-N retrieved assessments
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        _, indices = self.index.search(query_embedding, top_n)

        candidates = self.metadata.iloc[indices[0]].copy()
        candidates.reset_index(drop=True, inplace=True)

        return candidates

