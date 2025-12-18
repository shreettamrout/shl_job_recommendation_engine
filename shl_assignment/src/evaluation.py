import pandas as pd
from typing import Tuple

from src.pipeline import AssessmentRecommenderPipeline
from src.utils import extract_slug


def evaluate_retrieval(
    labels_df: pd.DataFrame,
    pipeline: AssessmentRecommenderPipeline,
    k: int = 5,
) -> Tuple[float, float]:
    """
    Evaluate Recall@K and MRR for the recommendation pipeline.

    Args:
        labels_df: DataFrame with columns ['Query', 'Assessment_url']
        pipeline: Initialized AssessmentRecommenderPipeline
        k: Cutoff for Recall@K

    Returns:
        recall_at_k, mrr
    """

    recall_hits = 0
    mrr_total = 0.0
    total = len(labels_df)

    for _, row in labels_df.iterrows():
        query = row["Query"]
        true_url = row["Assessment_url"]

        true_slug = extract_slug(true_url)
        if not true_slug:
            continue

        results = pipeline.recommend(query)
        predicted_slugs = [
            extract_slug(item["url"]) for item in results
        ]

        if true_slug in predicted_slugs:
            recall_hits += 1
            rank = predicted_slugs.index(true_slug) + 1
            mrr_total += 1.0 / rank

    recall_at_k = recall_hits / total if total > 0 else 0.0
    mrr = mrr_total / total if total > 0 else 0.0

    return recall_at_k, mrr


def run_evaluation(
    labels_path: str,
    pipeline: AssessmentRecommenderPipeline,
    k: int = 5,
):
    """
    Load labels and run evaluation.
    """

    labels_df = pd.read_excel(labels_path)

    # Normalize column names
    labels_df = labels_df.rename(
        columns={
            "Query": "Query",
            "Assessment_url": "Assessment_url",
        }
    )

    recall, mrr = evaluate_retrieval(labels_df, pipeline, k)

    print("EVALUATION RESULTS")
    print("==================")
    print(f"Total Queries : {len(labels_df)}")
    print(f"Recall@{k}     : {recall:.4f}")
    print(f"MRR           : {mrr:.4f}")

