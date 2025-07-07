from LLM_matcher.LLMProcessor import LLMProcessor
import LLM_matcher.config as config
import LLM_matcher.DataProcessor as dataprocessor
import LLM_matcher.LogicalProcessor as logicalprocessor
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
from pathlib import Path

# 1. Load rulebooks and build examples

name_rulebook_df      = dataprocessor.read(config.NAME_RULEBOOK)
location_rulebook_df  = dataprocessor.read(config.LOCATION_RULEBOOK)

name_examples     = dataprocessor.build_rulebook_examples(name_rulebook_df,
                                                         "Name1", "Name2", "Score", "Comments")
location_examples = dataprocessor.build_rulebook_examples(location_rulebook_df,
                                                         "Location1", "Location2", "Score", "Comment")

examples_dict = {"name": name_examples, "location": location_examples}
nameProcessor      = LLMProcessor(examples_dict)
locationProcessor  = LLMProcessor(examples_dict)

# 2. Load input data and sanity-check columns

df = dataprocessor.read(config.DATA_CSV)
required_cols = [
    "Customer Name_BS", "Customer Name_PR",
    "location_BS",      "location_PR",
    "Amount_BS",        "Amount_PR",
    "Policy No_BS",     "Policy No_PR",
    "Customer ID_BS",   "Customer ID_PR",
    "Product_BS",       "Product_PR",
    "channel_BS",       "channel_PR",
]
dataprocessor.validate_columns(df, required_cols)

# 3. Prepare result structure

results = []
for _ in range(len(df)):               # a mutable dict per row
    results.append({
        "recon_tag": "","customer_id_binary": "", "policy_no_binary": "", "name_binary": "",
        "location_binary": "", "amount_pct": "", "product_binary": "", "channel_binary": "",
    })

# 4. MEMORY STRINGS

memory_name      = ""
memory_location  = ""

# 5-A.  Pass 1  ——  Reconciling names

for idx, (row_index, row) in enumerate(tqdm(df.iterrows(),
                                            total=len(df),
                                            desc="Reconciling names")):

    # -------------------- Unpack fields --------------------
    cust_id_bs, cust_id_pr = row["Customer ID_BS"],   row["Customer ID_PR"]
    pol_no_bs,  pol_no_pr  = row["Policy No_BS"], row["Policy No_PR"]
    name_bs,    name_pr    = row["Customer Name_BS"], row["Customer Name_PR"]

    # -------------------- Missing flags --------------------
    cust_id_missing = pd.isna(cust_id_bs) or str(cust_id_bs).strip() == "" \
                   or pd.isna(cust_id_pr) or str(cust_id_pr).strip() == ""
    pol_no_missing  = pd.isna(pol_no_bs)  or str(pol_no_bs).strip()  == "" \
                   or pd.isna(pol_no_pr)  or str(pol_no_pr).strip()  == ""
    name_missing    = pd.isna(name_bs)    or str(name_bs).strip()    == "" \
                   or pd.isna(name_pr)    or str(name_pr).strip()    == ""

    # -------------------- Exact-match checks --------------------
    cust_id_match = logicalprocessor.exact_matcher(cust_id_bs, cust_id_pr) if not cust_id_missing else -1
    pol_no_match  = logicalprocessor.exact_matcher(pol_no_bs,  pol_no_pr)  if not pol_no_missing  else -1

    # -------------------- RECON TAG decision --------------------
    if (not cust_id_missing and cust_id_match == 0) or (not pol_no_missing and pol_no_match == 0):
        recon_tag = "definite_mismatch"
    elif (not cust_id_missing and cust_id_match == 1) and (not pol_no_missing and pol_no_match == 1):
        recon_tag = "matched_on_id"
    else:
        recon_tag = "needs_llm"

    # -------------------- NAME binary --------------------
    if recon_tag == "definite_mismatch":
        name_binary = ""
    elif recon_tag == "matched_on_id":
        name_binary = 1 if not name_missing else -1
    else:   # needs_llm
        if name_missing:
            name_binary = -1
        else:
            if len(memory_name) > config.MAX_MEM_CHARS:
                memory_name = dataprocessor.summarize(memory_name)
            match_res    = nameProcessor.match_name(name_bs, name_pr, memory_name)
            name_binary  = match_res.binary
            tag_txt      = "MATCH" if name_binary else "NO_MATCH"
            memory_name += f"\n#{idx}: name|{name_bs}|{name_pr} → {tag_txt}"

    # -------------------- Save partial row --------------------
    results[idx].update({                       
        "recon_tag":           recon_tag,
        "customer_id_binary":  cust_id_match if not cust_id_missing else -1,
        "policy_no_binary":    pol_no_match  if not pol_no_missing  else -1,
        "name_binary":         name_binary,
    })

# 5-B.  Pass 2  ——  Reconciling locations

for idx, (row_index, row) in enumerate(tqdm(df.iterrows(),
                                            total=len(df),
                                            desc="Reconciling locations")):

    recon_tag = results[idx]["recon_tag"]

    # -------------------- Unpack fields --------------------
    location_bs, location_pr = row["location_BS"], row["location_PR"]
    amount_bs,   amount_pr   = row["Amount_BS"],   row["Amount_PR"]
    product_bs,  product_pr  = row["Product_BS"],  row["Product_PR"]
    channel_bs,  channel_pr  = row["channel_BS"],  row["channel_PR"]

    location_missing = pd.isna(location_bs) or str(location_bs).strip() == "" \
                    or pd.isna(location_pr) or str(location_pr).strip() == ""

    # -------------------- LOCATION binary --------------------
    if recon_tag == "definite_mismatch":
        location_binary = ""
    elif recon_tag == "matched_on_id":
        if location_missing:
            location_binary = -1
        else:
            if len(memory_location) > config.MAX_MEM_CHARS:
                memory_location = dataprocessor.summarize(memory_location)
            match_res        = locationProcessor.match_location(location_bs,
                                                                location_pr,
                                                                memory_location)
            location_binary  = match_res.binary
            tag_txt          = "MATCH" if location_binary else "NO_MATCH"
            memory_location += f"\n#{idx}: location|{location_bs}|{location_pr} → {tag_txt}"
    else:  # needs_llm
        if location_missing:
            location_binary = -1
        else:
            if len(memory_location) > config.MAX_MEM_CHARS:
                memory_location = dataprocessor.summarize(memory_location)
            match_res        = locationProcessor.match_location(location_bs,
                                                                location_pr,
                                                                memory_location)
            location_binary  = match_res.binary
            tag_txt          = "MATCH" if location_binary else "NO_MATCH"
            memory_location += f"\n#{idx}: location|{location_bs}|{location_pr} → {tag_txt}"

    # -------------------- Non-LLM fields --------------------
    if recon_tag != "definite_mismatch":
        amount_pct       = logicalprocessor.amount_matcher(amount_bs, amount_pr)
        product_binary   = logicalprocessor.exact_matcher(product_bs,  product_pr)
        channel_binary   = logicalprocessor.exact_matcher(channel_bs,  channel_pr)
    else:
        amount_pct = product_binary = channel_binary = ""

    # -------------------- Save final row --------------------
    results[idx].update({
        "location_binary": location_binary, 
        "amount_pct":      amount_pct,
        "product_binary":  product_binary,
        "channel_binary":  channel_binary,
    })
    
# 6. Save combined results
results_df = pd.concat([df, pd.DataFrame(results)], axis=1)
#dataprocessor.write_results_df(results_df, out_path=str(config.OUTPUT_CSV))
print("Orchestration complete")

# filtered version for the ML model
mask          = results_df["recon_tag"].isin(["matched_on_id", "needs_llm"])
filtered_df   = results_df[mask].copy()

print("Filtered rows saved")

# 7. Apply ML model to filtered rows

MODEL_PATH = config.ML_MODEL_PATH
model      = joblib.load(MODEL_PATH)

FEATURE_COLS = [
    "customer_id_binary",
    "policy_no_binary",
    "name_binary",
    "location_binary",
    "product_binary",
    "channel_binary",
    "amount_pct",
]                                    

X_model = (
    filtered_df.loc[:, FEATURE_COLS]     # enforce order
              .apply(pd.to_numeric, errors="coerce")
              .fillna(0.0)
)

#X_model = filtered_df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)

# Find the column index that corresponds to class 2 (“matcher”)
matcher_col = list(model.classes_).index(2)

prob = model.predict_proba(X_model)[:, matcher_col]      # P(class==2)
filtered_df["match_pct"] = (prob * 100).round(2)
# set blank / NaN to [4,6), then enforce floor of  for everything else

rand_series = pd.Series(
    np.round(np.random.uniform(4, 6, size=len(filtered_df)), 2),
    index=filtered_df.index
)

filtered_df["match_pct"] = (
    filtered_df["match_pct"]
      .fillna(rand_series)
      .clip(lower=rand_series)
)

# Assemble final output

# Drop helper column if present
filtered_df.drop(columns=[c for c in ["model_pred"] if c in filtered_df.columns],
                 inplace=True, errors="ignore")

remaining_df = results_df[~mask].copy()
final_df     = pd.concat([filtered_df, remaining_df], ignore_index=True).sort_values("id")

# Ensure match_pct exists and every cell = [4, 6)
rand_series = pd.Series(
    np.round(np.random.uniform(4, 6, size=len(final_df)), 2),
    index=final_df.index
)

final_df["match_pct"] = (
    pd.to_numeric(final_df["match_pct"], errors="coerce")  
      .fillna(rand_series)                                 
      .clip(lower=rand_series)                             
)

# Finally, we have come to the output!!!

FINAL_OUT = config.OUTPUT_CSV
dataprocessor.write_results_df(final_df, out_path=str(FINAL_OUT))
print(f"Final reconciled file written: {FINAL_OUT}")
