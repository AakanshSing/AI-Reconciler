# config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_CSV = r"input\input_data.csv"                              # Path to input CSV file
OUTPUT_CSV = r"output\output.csv"                               # Path to output CSV file

RULEBOOKS_DIR = ROOT / "rulebooks"
NAME_RULEBOOK = RULEBOOKS_DIR / "nameRuleBook.csv"              # Path to name rulebook
LOCATION_RULEBOOK = RULEBOOKS_DIR / "locationRuleBook.csv"      # Path to location rulebook

ML_MODEL_PATH = ROOT / "model.pkl"                              # Path to the ML model file

LLM_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "mistral"
TEMPERATURE = 0.0
MAX_MEM_CHARS = 12_000
CACHE_FILE = ROOT / "llm_cache.json"

STRATEGIES = [
    {
        "prefix": "name",
        "method": "match_name",
        "columns": ["Customer Name_BS", "Customer Name_PR"],
        "rulebook": {
            "path": RULEBOOKS_DIR / "nameRuleBook.csv",
            "keys": ["Name1", "Name2", "Score", "Comments"]
        }
    },
    {
        "prefix": "location",
        "method": "match_location",
        "columns": ["Location_BS", "Location_PR"],
        "rulebook": {
            "path": RULEBOOKS_DIR / "locationRuleBook.csv",
            "keys": ["Location1", "Location2", "Score", "Comment"]
        }
    }
    
]
