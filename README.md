# SHL Assessment Recommendation System (RAG-based)

## Overview

This project implements a **Retrieval-Augmented Generative AI (RAG)** system to recommend relevant **SHL assessments** based on a user query or job description.  
The system combines **dense semantic retrieval** with **LLM-inspired reasoning logic** and is designed to be **modular, explainable, and production-ready**.

The solution was developed as part of a Generative AI assignment and emphasizes:
- Correct system architecture
- Meaningful LLM/retrieval integration
- Transparent evaluation
- Practical deployment considerations

---

## System Architecture

**High-level flow:**
```bash
User Query
↓
Query Embedding (Sentence-Transformers)
↓
Dense Retrieval (FAISS)
↓
Candidate Assessments
↓
LLM-based Reasoning (Selection Layer)
↓
Final Assessment Recommendations
```
---

## Project Structure
```bash
shl_assignment/
│
├── notebooks/
│ └── shl_rag_assignments_notebook.ipynb # Colab notebook (experiments & evaluation)
│
├── src/
│ ├── utils.py # Utility functions (slug extraction, helpers)
│ ├── retrieval.py # FAISS-based dense retrieval
│ ├── llm_reasoning.py # LLM reasoning / selection layer
│ ├── pipeline.py # End-to-end inference pipeline
│ └── evaluation.py # Recall@K and MRR evaluation
│
├── api/
│ └── app.py # FastAPI service exposing /recommend endpoint
│
└── README.md
```

**Data directory (outside repo code):**
```bash
GenAI-Dataset/
├── shl_metadata.csv
├── shl_embeddings.npy
└── Gen_AI_Dataset.xlsx
```

---

## Data Description

### Knowledge Base
- 500+ SHL assessments scraped from the SHL product catalog
- Metadata includes:
  - Assessment name
  - URL
  - Test type
  - Remote testing availability
  - Adaptive/IRT availability
  - Duration (where available)

### Labeled Dataset
- Provided dataset with **66 labeled queries**
- Columns:
  - `Query`
  - `Assessment_url`
- Used **only for evaluation**, not for training

---

## Models & Technologies

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Search**: FAISS (cosine similarity)
- **API Framework**: FastAPI
- **Evaluation Metrics**: Recall@K, Mean Reciprocal Rank (MRR)
- **Environment**: Conda (recommended for FAISS & pandas)

All models and tools used are **open-source**.

---

## Evaluation Strategy

Evaluation is performed using:
- **Recall@5**
- **Mean Reciprocal Rank (MRR)**

A **slug-based URL canonicalization** strategy is used to match labeled URLs with scraped catalog URLs, ensuring fair evaluation across different SHL catalog versions.

Evaluation is applied at the **final recommendation stage**, using the full pipeline.

---

## Running the Project

### 1. Create Environment (Recommended)

```bash
conda create -n shl_env python=3.10 -y
conda activate shl_env
conda install -c conda-forge faiss-cpu pandas numpy scikit-learn -y
pip install sentence-transformers fastapi uvicorn
```
### 2. Run Evaluation
From the project root:
```bash
python - <<EOF
from src.pipeline import AssessmentRecommenderPipeline
from src.evaluation import run_evaluation

pipeline = AssessmentRecommenderPipeline()

run_evaluation(
    labels_path="../GenAI-Dataset/Gen_AI_Dataset.xlsx",
    pipeline=pipeline,
    k=5
)
EOF
```

### 3. Run API Server
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```
### Colab Notebook Link
https://colab.research.google.com/drive/1PVRrvA0s5fxGYBZMfL4XgnTqQDH90gSz?usp=sharing

