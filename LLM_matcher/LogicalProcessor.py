import re
from thefuzz import fuzz
import jellyfish

TITLES_SUFFIXES = {"mr", "mrs", "ms", "miss", "dr", "prof", "jr", "sr"}
COMMON_SURNAME_TOKENS = {"singh", "kumar"}  # Expand as needed
NICKNAME_MAP = {
    "liz": "elizabeth",
    "johnny": "john",
    "mike": "michael",
    # Add more mappings here
}
JACCARD_THRESHOLD = 0.75
TOKEN_SIM_THRESHOLD = 0.85

def preprocess(name):
    name = str(name).lower()
    name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
    tokens = name.split()
    tokens = [t for t in tokens if t not in TITLES_SUFFIXES]
    return " ".join(tokens)

def tokenize(name):
    return name.split()

def standardize_tokens(tokens):
    return [NICKNAME_MAP.get(token, token) for token in tokens]

def sort_tokens(tokens):
    return sorted(tokens)

def vectorize(tokens):
    return set(tokens)

def jaccard_similarity(set1, set2):
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def token_similarity(token1, token2):
    jw = jellyfish.jaro_winkler_similarity(token1, token2)
    fuzz_ratio = fuzz.ratio(token1, token2) / 100.0
    return max(jw, fuzz_ratio)

def average_token_similarity(tokens1, tokens2):
    if not tokens1 or not tokens2:
        return 0.0
    matches_1 = [max(token_similarity(t1, t2) for t2 in tokens2) for t1 in tokens1]
    matches_2 = [max(token_similarity(t2, t1) for t1 in tokens1) for t2 in tokens2]
    return (sum(matches_1) + sum(matches_2)) / (len(matches_1) + len(matches_2))

def handle_edge_cases(tokens1, tokens2):
    set1 = set(tokens1) - COMMON_SURNAME_TOKENS
    set2 = set(tokens2) - COMMON_SURNAME_TOKENS
    if set1 and set2:
        return list(set1), list(set2)
    return tokens1, tokens2

def initials_match(tokens1, tokens2):
    if all(len(t) == 1 for t in tokens1) and tokens1:
        temp_tokens2 = tokens2.copy()
        for initial in tokens1:
            matched = False
            for idx, token in enumerate(temp_tokens2):
                if token.startswith(initial):
                    matched = True
                    temp_tokens2.pop(idx)
                    break
            if not matched:
                return False
        return True
    if all(len(t) == 1 for t in tokens2) and tokens2:
        temp_tokens1 = tokens1.copy()
        for initial in tokens2:
            matched = False
            for idx, token in enumerate(temp_tokens1):
                if token.startswith(initial):
                    matched = True
                    temp_tokens1.pop(idx)
                    break
            if not matched:
                return False
        return True
    return False

def is_match(jaccard, avg_token_sim, jaccard_threshold, token_sim_threshold):
    if jaccard >= jaccard_threshold or avg_token_sim >= token_sim_threshold:
        return 1
    return 0

def compare_names(name1, name2):
    n1 = preprocess(name1)
    n2 = preprocess(name2)
    tokens1 = standardize_tokens(sort_tokens(tokenize(n1)))
    tokens2 = standardize_tokens(sort_tokens(tokenize(n2)))
    if initials_match(tokens1, tokens2):
        return 1
    tokens1, tokens2 = handle_edge_cases(tokens1, tokens2)
    set1 = vectorize(tokens1)
    set2 = vectorize(tokens2)
    jaccard = jaccard_similarity(set1, set2)
    avg_token_sim = average_token_similarity(tokens1, tokens2)
    return is_match(jaccard, avg_token_sim, JACCARD_THRESHOLD, TOKEN_SIM_THRESHOLD)

def name_matcher(name1, name2):
    """
    Returns 1 if names match as per custom logic, 0 if not, and -1 if either is missing.
    """
    # Treat empty string, None, or NaN as missing
    if (name1 is None or name2 is None or
        str(name1).strip() == "" or str(name2).strip() == "" or
        str(name1).strip().lower() in {"nan", "none"} or str(name2).strip().lower() in {"nan", "none"}):
        return -1
    return compare_names(name1, name2)

def exact_matcher(val1, val2):
    """
    Returns 1 if val1 matches val2 exactly (case-insensitive, stripped), 0 if not, -1 if either is missing.
    """
    if (val1 is None or val2 is None or
        str(val1).strip() == "" or str(val2).strip() == "" or
        str(val1).strip().lower() in {"nan", "none"} or str(val2).strip().lower() in {"nan", "none"}):
        return -1
    return int(str(val1).strip().lower() == str(val2).strip().lower())

def amount_matcher(a, b):
    """
    Returns a float (0.0–1.0) representing how close the two amounts are,
    -1 if either is missing or not a number.
    """
    try:
        if (a is None or b is None or
            str(a).strip() == "" or str(b).strip() == "" or
            str(a).strip().lower() in {"nan", "none"} or str(b).strip().lower() in {"nan", "none"}):
            return -1
        a, b = float(a), float(b)
        if a == 0 and b == 0:
            return 1.0
        diff = abs(a - b)
        max_ab = max(abs(a), abs(b))
        score = max(0.0, (diff / max_ab))
        return round(score,3)
    except:
        return -1
