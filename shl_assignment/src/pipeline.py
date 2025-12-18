import os
import numpy as np
import pandas as pd
from typing import List, Dict

import faiss
from sentence_transformers import SentenceTransformer

from src.utils import extract_slug
from src.llm_reasoning import select_recommendations


class AssessmentRecommenderPipeline:
    """
    End-to-end inference pipeline:
    Query → Retrieval → (Optional) LLM reasoning → Recommendations
    """

    def __init__(
        self,
        use_llm_reasoning: bool = True,
        top_n_retrieval: int = 20,
        final_k: int = 5,
    ):
        self.use_llm_reasoning = use_llm_reasoning
        self.top_n_retrieval = top_n_retrieval
        self.final_k = final_k

        self._resolve_paths()
        self._load_resources()

    def _resolve_paths(self):
        """Resolve dataset paths relative to this file."""

        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))

        self.data_dir = os.path.join(project_root, "GenAI-Dataset")

        self.metadata_path = os.path.join(self.data_dir, "shl_metadata.csv")
        self.embeddings_path = os.path.join(self.data_dir, "shl_embeddings.npy")

    def _load_resources(self):
        """Load models, metadata, embeddings, and FAISS index."""

        # Load metadata
        self.metadata = pd.read_csv(self.metadata_path)
        self.metadata["slug"] = self.metadata["url"].apply(extract_slug)
        self.metadata = self.metadata.dropna(subset=["slug"]).reset_index(drop=True)

        # Load embeddings
        self.embeddings = np.load(self.embeddings_path)

        # Embedding model
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # FAISS index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def _embed_query(self, query: str):
        return self.embedder.encode([query], normalize_embeddings=True)

    def _retrieve_candidates(self, query_embedding):
        _, indices = self.index.search(query_embedding, self.top_n_retrieval)
        return self.metadata.iloc[indices[0]].reset_index(drop=True)

    def recommend(self, query: str) -> List[Dict]:
        """
        Public method used by API.
        """

        # Embed query
        query_embedding = self._embed_query(query)

        # Retrieve candidates
        candidates = self._retrieve_candidates(query_embedding)

        # Optional LLM-based selection
        if self.use_llm_reasoning:
            selected = select_recommendations(
                query=query,
                candidates=candidates,
                k=self.final_k,
            )
        else:
            selected = candidates.head(self.final_k)

        return self._format_output(selected)

    def _format_output(self, df: pd.DataFrame) -> List[Dict]:
        results = []

        for _, row in df.iterrows():
            results.append(
                {
                    "name": row["name"],
                    "url": row["url"],
                    "test_type": (
                        [t.strip() for t in row["test_type"].split(",")]
                        if isinstance(row["test_type"], str)
                        else []
                    ),
                    "remote_testing": bool(row.get("remote_testing", False)),
                    "adaptive_irt": bool(row.get("adaptive_irt", False)),
                    "duration_minutes": row.get("duration_minutes", None),
                }
            )

        return results

