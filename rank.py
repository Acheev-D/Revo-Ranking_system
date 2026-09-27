import os
import sys
import json
import gzip
import pickle
import argparse
import pandas as pd

def load_manifest(manifest_path):
    if not os.path.exists(manifest_path):
        print(f"Warning: Manifest not found at {manifest_path}. Using empty manifest.", file=sys.stderr)
        return {}
    with open(manifest_path, 'rb') as f:
        return pickle.load(f)

def fallback_score(candidate):
    keywords = {"ai", "machine learning", "llm", "rag", "retrieval", "embeddings", "python", "nlp"}
    score = 0.0
    prof = candidate.get('profile', {})
    summary = prof.get('summary', '').lower()
    headline = prof.get('headline', '').lower()
    
    text_to_check = summary + " " + headline
    for skill in candidate.get('skills', []):
        text_to_check += " " + skill.get('name', '').lower()
        
    for kw in keywords:
        if kw in text_to_check:
            score += 0.01
            
    return 0.400 + score

def process_candidates(candidates_path, manifest):
    open_func = gzip.open if candidates_path.endswith('.gz') else open
    
    # Calculate min and max for Min-Max scaling
    valid_scores = [v['score'] for v in manifest.values() if v['score'] > -100]
    if valid_scores:
        max_score = max(valid_scores)
        min_score = min(valid_scores)
        score_range = max_score - min_score if max_score > min_score else 1.0
    else:
        max_score, min_score, score_range = 1.0, 0.0, 1.0
        
    with open_func(candidates_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON line.", file=sys.stderr)
                continue
                
            cid = candidate.get('candidate_id')
            if not cid:
                print(f"Warning: Skipping candidate with missing candidate_id.", file=sys.stderr)
                continue
                
            if cid in manifest:
                entry = manifest[cid]
                raw_score = entry['score']
                
                if entry['reasoning'] == "Candidate profile triggered integrity honeypot rules.":
                    final_score = 0.0000
                else:
                    # Scale to [0.8000, 0.9999]
                    final_score = 0.8000 + 0.1999 * ((raw_score - min_score) / score_range)
                    final_score = round(final_score, 4)
                    
                yield {
                    'candidate_id': cid,
                    'score': final_score,
                    'reasoning': entry['reasoning']
                }
            else:
                score = fallback_score(candidate)
                yield {
                    'candidate_id': cid,
                    'score': round(min(score, 0.4999), 4),
                    'reasoning': "Candidate matched based on basic keyword overlap (fallback)."
                }

def main():
    parser = argparse.ArgumentParser(description="Rank candidates for Redrob Hackathon")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", required=True, help="Path to output submission.csv")
    args = parser.parse_args()

    # Load precomputed manifest (assumes it's in the same directory as the script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    manifest_path = os.path.join(script_dir, 'precomputed_manifest.pkl')
    manifest = load_manifest(manifest_path)
    
    # Process stream
    results = list(process_candidates(args.candidates, manifest))
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    if df.empty:
        # Create empty template
        df = pd.DataFrame(columns=['candidate_id', 'rank', 'score', 'reasoning'])
        df.to_csv(args.out, index=False)
        return

    # Sort strictly by score DESCENDING, break ties by candidate_id ASCENDING
    df = df.sort_values(by=['score', 'candidate_id'], ascending=[False, True])
    
    # Slice exactly top 100 rows
    df_top100 = df.head(100).copy()
    
    # Add rank column (1 to 100)
    df_top100['rank'] = range(1, len(df_top100) + 1)
    
    # Ensure column order matches spec exactly
    columns_order = ['candidate_id', 'rank', 'score', 'reasoning']
    df_top100 = df_top100[columns_order]
    
    # Output to CSV exactly matching the headers
    df_top100.to_csv(args.out, index=False)
    print(f"Successfully wrote top 100 candidates to {args.out}")

if __name__ == "__main__":
    main()
