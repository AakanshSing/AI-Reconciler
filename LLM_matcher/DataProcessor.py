import pandas as pd
import LLM_matcher.config as config
import json
import os

def read(path):
    """Smart reader that auto-detects rulebook or dataset based on file path."""
    # You can make this smarter if your file types/names vary.
    fname = os.path.basename(path).lower()
    if 'rulebook' in fname:
        return load_rulebook(path)
    else:
        return load_dataset(path)
    
def load_dataset(path=None):
    """Load main input CSV as DataFrame."""
    return pd.read_csv(path or config.DATA_CSV)

def save_dataset(df, path=None):
    """Save DataFrame to output CSV."""
    out = path or config.OUTPUT_CSV
    df.to_csv(out, index=False)
    # print(f"Results written to → {out}")

def load_rulebook(path):
    """Load a rulebook CSV and drop empty rows."""
    return pd.read_csv(path).dropna(how="all")

def build_rulebook_examples(df, key1, key2, score_col, comment_col=None):
    """
    Convert rulebook DataFrame into JSON-lines string for few-shot LLM prompting.
    """
    lines = []
    for _, row in df.dropna(subset=[key1, key2, score_col]).iterrows():
        entry = {
            key1:     row[key1],
            key2:     row[key2],
            "binary": int(row[score_col]),
        }
        if comment_col and comment_col in row:
            entry["comment"] = str(row[comment_col])[:80]
        lines.append(json.dumps(entry, ensure_ascii=False))
    return "\n".join(lines)

def validate_columns(df, required_cols):
    """Check that all required columns are present."""
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df

# Memory & cache helpers (move from memory.py)
def summarize(ctx: str) -> str:
    """Compress a long memory string into 2–3 sentences via the LLM."""
    import requests, re
    prompt = (
        "You are a concise summarizer. Given the past reconciliation decisions below,\n"
        "produce a 2–3 sentence summary of the key patterns and outcomes:\n\n"
        f"{ctx}\n\nSummary:"
    )
    payload = {
        "model":       config.LLM_MODEL,
        "prompt":      prompt,
        "temperature": 0.0,
        "stream":      False,
    }
    r = requests.post(config.LLM_URL, json=payload, timeout=60)
    r.raise_for_status()
    text = r.json().get("response", "").strip()
    return re.sub(r"^```.*|```$", "", text, flags=re.S).strip()

def load_cache():
    """Read the JSON cache file, or return empty if missing/corrupt."""
    try:
        with open(config.CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_cache(cache: dict):
    """Persist the cache dict to disk."""
    config.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def write_results_df(results_df, out_path="output_data.csv"):
    """
    Writes a full results DataFrame (with all input and result columns, tags, etc.) to a CSV file.
    """
    results_df.to_csv(out_path, index=False)
    # print(f"Results written to → {out_path}")
    print("\nFinal columns in output file:")
    print(list(results_df.columns))
    

