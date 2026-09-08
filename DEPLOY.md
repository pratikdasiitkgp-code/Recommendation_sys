# Git & Deployment Guide

## 1. Push to GitHub

```bash
cd recsys_project
git init
git add .
git commit -m "Two-stage recommendation system: ALS retrieval + LightGBM ranking"
```

Create a new empty repo on GitHub (no README/license, since you already have one),
then:

```bash
git remote add origin https://github.com/<your-username>/recsys-project.git
git branch -M main
git push -u origin main
```

**Important:** `.gitignore` excludes `data/` (raw dataset + generated parquet
files) since they can be large and are regenerable. For the *deployed API* to
work, it needs `data/ranked_candidates.parquet` — see the note in step 3.

## 2. Run the pipeline once to generate model outputs

```bash
pip install -r requirements.txt
python 01_data_prep.py --category All_Beauty --min_interactions 5
python 02_candidate_generation.py --data_dir ./data
python 03_ranking.py --data_dir ./data
python 04_evaluation.py --data_dir ./data --k 10
```

This produces `data/ranked_candidates.parquet`, which `app.py` serves.

## 3. Test the API locally

```bash
uvicorn app:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive Swagger docs. Try:
- `GET /sample_users` — get some valid user_ids to test with
- `GET /recommend/{user_id}?k=10` — get top-10 recommendations for that user

## 4. Deploy — two good free options

### Option A: Render (simplest, good for a portfolio link)
1. Push your repo to GitHub (step 1)
2. Since `data/` is gitignored, either:
   - Commit just `data/ranked_candidates.parquet` explicitly (`git add -f data/ranked_candidates.parquet`) if it's small enough (a filtered category like `All_Beauty` usually is), or
   - Add a Render **build command** that runs the pipeline at deploy time: `python 01_data_prep.py && python 02_candidate_generation.py --data_dir ./data && python 03_ranking.py --data_dir ./data`
3. On [render.com](https://render.com): New → Web Service → connect your GitHub repo
4. Render auto-detects the `Dockerfile` — just confirm and deploy
5. You'll get a live URL like `https://recsys-project.onrender.com/docs`

### Option B: Hugging Face Spaces (popular for ML/DS portfolios, free GPU-free tier is fine here)
1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space), choose **Docker** as the SDK
2. Push this same repo to the Space's git remote (HF gives you a git URL, same `git push` flow as GitHub)
3. HF builds your `Dockerfile` automatically and gives you a live URL like `https://huggingface.co/spaces/<you>/recsys-project`
4. This option is worth mentioning by name in interviews — it signals familiarity with the ML deployment ecosystem, not just software deployment in general

## 5. What to put in your resume / portfolio link
- Link directly to the `/docs` endpoint (Swagger UI) — it's interactive, so an
  interviewer can try `/recommend/{user_id}` themselves without reading code first
- In your README or resume bullet, lead with the NDCG lift number from
  `eval_results.json` (retrieval-only vs. retrieval+ranking) — a concrete
  metric is more convincing than "built a recommendation system"
