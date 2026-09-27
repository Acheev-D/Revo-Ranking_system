# Intelligent Candidate Ranking & Discovery System

An enterprise-grade, high-performance candidate discovery and ranking engine designed to evaluate large candidate datasets (100,000+ profiles) against any target job description using hybrid semantic retrieval, cross-encoder re-ranking, and local LLM reasoning.

---

## Overview & Architecture

The system uses a decoupled two-tier architecture designed for maximum accuracy, scalability, and speed:

1. **The Offline Intelligence Builder (`build_manifest.py`)**:
   Performs deep ML evaluations without runtime constraints, combining sparse keyword matching, dense semantic embeddings, cross-attention re-ranking, behavioral heuristics, and LLM-generated justifications into a compiled manifest.

2. **The High-Performance Runtime Ranker (`rank.py`)**:
   An ultra-lightweight, memory-efficient engine that streams candidate datasets line-by-line in O(1) memory, scores them against the manifest in seconds, and outputs ranked candidates with calibrated match probabilities and reasoning.

```
[Job Description] ──┐
                    ▼
[Candidates Data] ──► [Integrity Honeypot Filter]
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  Hybrid Retrieval                     │
        │  • BM25 Sparse Keyword Scoring        │
        │  • all-MiniLM-L6-v2 Dense Embeddings  │
        └───────────────────┬───────────────────┘
                            ▼
        ┌───────────────────────────────────────┐
        │  Reciprocal Rank Fusion (RRF)         │
        │  • Filter to Top 1,000 Candidates     │
        └───────────────────┬───────────────────┘
                            ▼
        ┌───────────────────────────────────────┐
        │  Cross-Encoder Re-Ranking             │
        │  • ms-marco-MiniLM-L-6-v2             │
        │  • Behavioral Signals Adjustment      │
        └───────────────────┬───────────────────┘
                            ▼
        ┌───────────────────────────────────────┐
        │  Local LLM Reasoning Generation       │
        │  • Qwen/Qwen2.5-0.5B-Instruct         │
        │  • Export to precomputed_manifest.pkl │
        └───────────────────┬───────────────────┘
                            ▼
        ┌───────────────────────────────────────┐
        │  Runtime Streaming Engine (rank.py)   │
        │  • O(1) Memory Stream                 │
        │  • Top-N Output (submission.csv)      │
        └───────────────────────────────────────┘
```

---

## Ranking Pipeline Details

1. **Integrity & Anomaly Detection**:
   Quarantines suspect or fraudulent profiles (negative job tenure, overlapping full-time employment intervals > 12 months, unearned expert skill claims, and inflated years of experience).

2. **Hybrid Retrieval (Sparse + Dense)**:
   * **BM25 Okapi**: Indexes exact technical terms, libraries, frameworks, and titles.
   * **Sentence-Transformers (`all-MiniLM-L6-v2`)**: Generates 384-dimensional dense vectors to measure semantic conceptual relevance.

3. **Reciprocal Rank Fusion (RRF)**:
   Combines BM25 and dense embedding ranks using reciprocal scoring (1 / (k + rank)) to identify the Top 1,000 candidate pool.

4. **Deep Cross-Encoder Re-Ranking**:
   Applies `cross-encoder/ms-marco-MiniLM-L-6-v2` across (Job Description, Candidate Profile) pairs for deep attention-based matching.

5. **Behavioral Signals Integration**:
   Adjusts candidate relevance logits based on engagement signals:
   * **GitHub Activity**: Up to +2.0 bonus based on verified public activity.
   * **Profile Completeness**: Up to +1.0 bonus.
   * **Recruiter Response Rate**: -5.0 penalty for unengaged candidates (< 20% response rate).
   * **Notice Period**: -2.0 penalty for notice periods exceeding 60 days.

6. **Local LLM Justification**:
   Generates a concise, analytical justification for each top candidate using `Qwen/Qwen2.5-0.5B-Instruct`.

---

## Repository Structure

```
.
├── job_description.md        # Target job description used for scoring candidates
├── candidates.jsonl          # Candidate dataset (100,000 JSONL records)
├── rank.py                   # High-performance runtime ranker
├── build_manifest.py         # Offline ML intelligence and manifest builder
├── precomputed_manifest.pkl  # Compiled candidate intelligence manifest
├── intermediate_scores.pkl   # Checkpoint for intermediate ranking scores
├── requirements-runtime.txt  # Lightweight runtime dependencies (pandas only)
├── requirements-offline.txt  # Deep Learning dependencies (PyTorch, Transformers)
├── submission.csv            # Final ranked output CSV
└── README.md                 # System documentation
```

---

## Getting Started

### Prerequisites
* Python 3.10 or 3.11
* Pip package manager

---

### Step 1: Run the Candidate Ranker (Fast Execution)

This evaluates the candidates using the precomputed manifest and generates the ranked CSV in **under 15 seconds** with minimal memory usage:

```bash
# 1. Install lightweight runtime dependencies
pip install -r requirements-runtime.txt

# 2. Run the ranker
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

The output CSV will contain:
* `candidate_id`: Unique identifier for each candidate.
* `rank`: Rank position (1 to N).
* `score`: Calibrated relevance score between `[0.8000, 0.9999]`.
* `reasoning`: Analytical justification generated by the LLM.

---

### Step 2: Re-Building the Intelligence Brain (Optional)

If you update the job description or modify scoring weights, re-run the full offline ML pipeline:

```bash
# 1. Install ML dependencies (PyTorch CPU, Transformers, BM25)
pip install -r requirements-offline.txt

# 2. Re-compute rankings and regenerate manifest
python build_manifest.py

# 3. Generate updated rankings
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

> **Note:** To re-run the complete ML stages from scratch when changing `job_description.md`, delete `intermediate_scores.pkl` first so it doesn't reuse previously cached scores.

---

## Customization

* **Changing the Job Description**:
  Edit `job_description.md` with any new role specifications, then execute `python build_manifest.py`.

* **Changing the Number of Ranked Candidates**:
  * In `build_manifest.py`, modify line 179:
    ```python
    top_100 = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:100]
    ```
  * In `rank.py`, modify line 113:
    ```python
    df_top100 = df.head(100).copy()
    ```
