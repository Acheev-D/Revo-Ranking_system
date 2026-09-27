import os
import json
import gzip
import pickle
from datetime import datetime
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import pipeline

def load_candidates(filepath):
    candidates = []
    open_func = gzip.open if filepath.endswith('.gz') else open
    with open_func(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    return candidates

def is_honeypot(candidate):
    # 1. Negative Job Tenure
    career_history = candidate.get('career_history', [])
    total_duration_months = 0
    intervals = []
    
    for job in career_history:
        if job.get('duration_months', 0) < 0:
            return True
        start_str = job.get('start_date')
        end_str = job.get('end_date')
        if start_str and end_str:
            try:
                start = datetime.strptime(start_str, "%Y-%m-%d")
                end = datetime.strptime(end_str, "%Y-%m-%d")
                if end < start:
                    return True
                intervals.append((start, end))
            except ValueError:
                pass
        total_duration_months += job.get('duration_months', 0)
    
    # 2. Overlapping full-time roles (basic interval overlap check)
    intervals.sort()
    overlap_months = 0
    for i in range(len(intervals) - 1):
        if intervals[i][1] > intervals[i+1][0]:
            overlap = min(intervals[i][1], intervals[i+1][1]) - intervals[i+1][0]
            overlap_months += overlap.days / 30
    if overlap_months > 12: # more than a year of overlap is highly suspicious
        return True

    # 3. Expert skills with 0 months of usage
    for skill in candidate.get('skills', []):
        if skill.get('proficiency') == 'expert' and skill.get('duration_months', -1) == 0:
            return True
            
    # 4. Experience inflation
    reported_years = candidate.get('profile', {}).get('years_of_experience', 0)
    actual_years = total_duration_months / 12
    if reported_years > actual_years + 5:  # e.g., claims 10 years but history sums to 3 years
        return True

    return False

def build_profile_text(candidate):
    prof = candidate.get('profile', {})
    headline = prof.get('headline', '')
    summary = prof.get('summary', '')
    skills = ", ".join([s.get('name', '') for s in candidate.get('skills', [])])
    exp = []
    for job in candidate.get('career_history', []):
        exp.append(f"{job.get('title', '')} at {job.get('company', '')}: {job.get('description', '')}")
    exp_text = " | ".join(exp)
    return f"{headline}. {summary}. Skills: {skills}. Experience: {exp_text}"

def compute_rrf(rankings_list, k=60):
    scores = {}
    for ranks in rankings_list:
        for rank, candidate_id in enumerate(ranks):
            if candidate_id not in scores:
                scores[candidate_id] = 0
            scores[candidate_id] += 1 / (k + rank + 1)
    return scores

def main():
    print("Loading data...")
    jd_path = '../job_description.md'
    if not os.path.exists(jd_path):
        jd_path = 'job_description.md' # fallback if run from parent
    
    with open(jd_path, 'r', encoding='utf-8') as f:
        jd_text = f.read()

    candidates_path = '../candidates.jsonl'
    if os.path.exists('../candidates.jsonl.gz'):
        candidates_path = '../candidates.jsonl.gz'
    elif not os.path.exists(candidates_path):
        candidates_path = 'candidates.jsonl'
        if os.path.exists('candidates.jsonl.gz'):
            candidates_path = 'candidates.jsonl.gz'
    
    candidates = load_candidates(candidates_path)
    
    print(f"Loaded {len(candidates)} candidates.")
    
    valid_candidates = []
    honeypot_ids = set()
    for c in candidates:
        if is_honeypot(c):
            honeypot_ids.add(c['candidate_id'])
        else:
            valid_candidates.append(c)
            
    print(f"Found {len(honeypot_ids)} honeypots.")
    
    # Text representations
    corpus = [build_profile_text(c) for c in valid_candidates]
    candidate_ids = [c['candidate_id'] for c in valid_candidates]
    id_to_candidate = {c['candidate_id']: c for c in valid_candidates}
    
    intermediate_path = 'intermediate_scores.pkl'
    if os.path.exists(intermediate_path):
        print("Found intermediate scores! Skipping 40-minute ML pipeline...")
        with open(intermediate_path, 'rb') as f:
            top_100, honeypot_ids = pickle.load(f)
    else:
        print("Computing BM25 scores...")
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = jd_text.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        
        print("Computing Dense Embedding scores...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        jd_emb = embedder.encode(jd_text, convert_to_tensor=True)
        corpus_embs = embedder.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
        from sentence_transformers import util
        cosine_scores = util.cos_sim(jd_emb, corpus_embs)[0].cpu().numpy()
        
        print("Computing RRF...")
        bm25_ranking = [candidate_ids[i] for i in bm25_scores.argsort()[::-1]]
        dense_ranking = [candidate_ids[i] for i in cosine_scores.argsort()[::-1]]
        
        rrf_scores = compute_rrf([bm25_ranking, dense_ranking])
        # Sort by RRF
        rrf_sorted = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_1000_ids = [x[0] for x in rrf_sorted[:1000]]
        
        print("Cross-Encoder re-ranking for top 1000...")
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Build pairs
        id_to_corpus = dict(zip(candidate_ids, corpus))
        pairs = [[jd_text, id_to_corpus[cid]] for cid in top_1000_ids]
        ce_scores = cross_encoder.predict(pairs, show_progress_bar=True)
        
        # Combine with Behavioral Signals
        id_to_candidate = {c['candidate_id']: c for c in valid_candidates}
        final_scores = {}
        
        for i, cid in enumerate(top_1000_ids):
            base_score = float(ce_scores[i])
            c = id_to_candidate[cid]
            signals = c.get('redrob_signals', {})
            
            # Heuristics
            gh_score = signals.get('github_activity_score', 0)
            gh_bonus = (gh_score / 100.0) * 2.0 if gh_score > 0 else 0
            
            prof_comp = signals.get('profile_completeness_score', 50)
            comp_bonus = (prof_comp / 100.0) * 1.0
            
            resp_rate = signals.get('recruiter_response_rate', 1.0)
            resp_penalty = -5.0 if resp_rate < 0.2 else 0.0
            
            # notice period penalty (long notice period)
            notice_days = signals.get('notice_period_days', 30)
            notice_penalty = -2.0 if notice_days > 60 else 0.0
            
            final_scores[cid] = base_score + gh_bonus + comp_bonus + resp_penalty + notice_penalty

        # Sort to get Top 100
        top_100 = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:100]
    top_100_ids = [x[0] for x in top_100]
    
    # Save intermediate scores so we don't lose 40 minutes if the download hangs!
    intermediate_path = 'intermediate_scores.pkl'
    with open(intermediate_path, 'wb') as f:
        pickle.dump((top_100, honeypot_ids), f)
    
    print("Generating LLM reasoning for Top 100...")
    generator = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
    
    manifest = {}
    for rank, (cid, score) in enumerate(top_100):
        c = id_to_candidate[cid]
        prof = c.get('profile', {})
        exp = prof.get('years_of_experience', 0)
        title = prof.get('current_title', '')
        gh = c.get('redrob_signals', {}).get('github_activity_score', 0)
        
        prompt = (f"Candidate has {exp} years of experience. Current title: {title}. "
                  f"GitHub score: {gh}. "
                  f"Write a 1-sentence concise justification for why they fit the AI Engineer role. "
                  f"Be specific and analytical. Do not use generic templates.")
        
        # Generation
        try:
            messages = [{"role": "user", "content": prompt}]
            out = generator(messages, max_new_tokens=60, do_sample=False)
            reasoning = out[0]['generated_text'][-1]['content'].replace('\n', ' ').strip()
            # Clean up cutoff sentences by stopping at the first valid punctuation mark
            if '.' in reasoning:
                reasoning = reasoning[:reasoning.rfind('.')+1]
        except Exception as e:
            reasoning = f"{exp} years of experience as {title} with a strong GitHub activity score of {gh}, making them a solid fit."

        manifest[cid] = {
            "score": score,
            "reasoning": reasoning
        }
        if (rank + 1) % 10 == 0 or (rank + 1) == len(top_100):
            print(f"Generated reasoning for {rank + 1}/{len(top_100)} candidates...", flush=True)
        
    for cid in honeypot_ids:
        manifest[cid] = {
            "score": 0.0,
            "reasoning": "Candidate profile triggered integrity honeypot rules."
        }
        
    print("Saving manifest...")
    with open('precomputed_manifest.pkl', 'wb') as f:
        pickle.dump(manifest, f)
        
    print("Done! precomputed_manifest.pkl created.")

if __name__ == "__main__":
    main()
