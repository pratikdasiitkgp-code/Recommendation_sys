# Sequential Product Recommendation System

A two-stage recommendation pipeline (retrieval + ranking) built on the
Amazon Reviews 2023 dataset, modeled after how production systems at
companies like Amazon/Netflix/Meta are structured.

## Architecture

```
User interaction history
        |
        v
[Stage 1: Candidate Generation]   <- implicit-feedback ALS
   fast, wide net, ~100 candidates/user
        |
        v
[Stage 2: Ranking]                <- LightGBM (lambdarank)
   precise, uses rich features, re-orders top candidates
        |
        v
   Final Top-K recommendations
```

**Why two stages instead of one model?** Scoring every item for every
user with a heavy model doesn't scale. Real systems split the problem:
a cheap, high-recall retrieval stage narrows millions of items down to
a few hundred candidates, and a more expensive, high-precision ranking
stage re-orders just those candidates using richer features. This is
the same pattern Netflix/YouTube/Amazon describe publicly for their
recommendation systems.

## Pipeline stages

| Script | Stage | What it does |
|---|---|---|
| `01_data_prep.py` | Data | Loads Amazon Reviews 2023, k-core filters, time-based leave-last-out split |
| `02_candidate_generation.py` | Retrieval | Trains implicit ALS, generates top-K candidates per user |
| `03_ranking.py` | Ranking | Builds features (recency, popularity, price, ALS score), trains LightGBM lambdarank |
| `04_evaluation.py` | Eval | Precision@K, Recall@K, NDCG@K — compares retrieval-only vs. full pipeline |
| `app.py` | Serve | FastAPI endpoint (`/recommend/{user_id}`) for programmatic access |
| `streamlit_app.py` | Demo UI | Interactive web app — pick a user, browse ranked recommendations, view live metrics |

## Running the full pipeline

```bash
pip install -r requirements.txt

python 01_data_prep.py --category All_Beauty --min_interactions 5
python 02_candidate_generation.py --data_dir ./data --factors 64 --top_k 100
python 03_ranking.py --data_dir ./data
python 04_evaluation.py --data_dir ./data --k 10
```

## Demo UI (Streamlit)

Once the pipeline above has run at least once (so `data/ranked_candidates.parquet`
exists), launch the interactive demo:

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. Lets you pick any user from a dropdown and see
their top-K ranked recommendations as cards (title, price, rating, rank score),
alongside a live sidebar comparing retrieval-only vs. full-pipeline NDCG@10 —
useful both as a portfolio demo and as a way to sanity-check the pipeline output.

Deploys for free on [Streamlit Community Cloud](https://share.streamlit.io) by
connecting this repo and pointing it at `streamlit_app.py`.


## Design decisions worth explaining in an interview

**Why leave-last-out, time-based split instead of random split?**
A random split lets future interactions leak into training, which
inflates offline metrics and doesn't reflect how the model will
actually be used (predicting the *next* action, not an arbitrary
held-out one).

**Why NDCG/Precision/Recall instead of RMSE?**
This is a ranking/retrieval problem, not a rating-prediction problem —
what matters is whether the right item shows up near the top of a
list, not how close a predicted rating is to a true rating.

**Cold-start strategy:**
- *New users* (no interaction history): fall back to popularity-based
  or category-trending recommendations until enough interactions
  accumulate to generate personalized ALS candidates.
- *New items* (no interactions yet): boosted into candidate sets via
  content-based similarity (title/category embedding) until they
  accumulate enough interaction signal for collaborative filtering to
  pick them up.

**Scaling retrieval in production:**
ALS produces dense user/item embeddings. At scale, exact dot-product
search over millions of items is too slow — production systems use
approximate nearest neighbor search (e.g., FAISS, ScaNN) to retrieve
candidates in milliseconds instead of scanning the full catalog.

**What the ranking stage adds:**
The evaluation script explicitly reports the NDCG lift from adding
the ranking stage on top of raw ALS ordering — this is the single
most convincing number to lead with when explaining the project.

## Possible extensions (if time allows)
- Replace ALS with a two-tower neural retrieval model (user/session
  tower + item tower, trained with in-batch negatives) for a more
  modern retrieval stage.
- Add session-based sequence modeling (GRU4Rec-style) instead of
  static ALS embeddings, to better capture short-term intent.
- Add a simple online A/B test simulation comparing the two-stage
  pipeline against a popularity baseline.
