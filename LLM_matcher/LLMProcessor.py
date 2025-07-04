import json
import re
import requests
from tenacity import retry, wait_exponential, stop_after_attempt
import LLM_matcher.config as config
from LLM_matcher.DataProcessor import summarize          
from tqdm import tqdm                       

class MatchResult:
    def __init__(self, binary):
        self.binary = binary


class LLMProcessor:
    def __init__(self, examples):
        self.examples = examples

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30),
           stop=stop_after_attempt(5))
    def _call_llm(self, prompt: str) -> dict:
        payload = {
            "model":       config.LLM_MODEL,
            "prompt":      prompt,
            "temperature": config.TEMPERATURE,
            "stream":      False,
        }
        r = requests.post(config.LLM_URL, json=payload, timeout=120)
        r.raise_for_status()
        raw = r.json().get("response", "")
        if not raw:
            return {}
        clean = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.S|re.I).strip()
        m = re.search(r"\{.*\}", clean, flags=re.S)
        if m:
            body = m.group(0)
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                pass
        # fallback: bullet-style parsing
        result = {}
        for line in clean.splitlines():
            m = re.match(r'\s*\d+\.\s*"(?P<key>[^"]+)"\s*→\s*(?P<val>.+)', line)
            if not m:
                continue
            key, val = m.group("key"), m.group("val").strip()
            if key == "binary":
                nums = re.findall(r"-?\d+", val)
                result[key] = int(nums[0]) if nums else 0
            else:
                val = val.strip()
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                result[key] = val
        return result

    # NAME

    def match_name(self, left: str, right: str, memory: str = "") -> MatchResult:
        #memory section
        mem_section = f"### Memory Summary:\n{memory}\n\n" if memory else ""
        prompt = f"""
You are an expert in Indian customer-name matching. See the rulebook examples, their binary scores and comments below for guidance.

### Instructions
Return only a JSON object with a single field:

  "binary"  → 0 if the names do not match, 1 if they do match

{mem_section}### Rulebook examples
{self.examples.get("name","")}

### Pair
{{"name1":"{left}","name2":"{right}"}}

### Your JSON:
"""
        #print(prompt)
        
        data = self._call_llm(prompt)
        if not data:
            return MatchResult(0)
        try:
            binary_val = int(data.get("binary", 0))
        except Exception:
            binary_val = 0
        binary_val = 1 if binary_val == 1 else 0
        return MatchResult(binary_val)

    # LOCATION
    
    def match_location(self, left: str, right: str, memory: str = "") -> MatchResult:
        mem_section = f"### Memory Summary:\n{memory}\n\n" if memory else ""
        prompt = f"""
You are an expert in Indian location-matching. See the rulebook examples, their binary scores and comments below for guidance.

### Instructions
Return only a JSON object with a single field:

  "binary"  → 0 if the locations do not match, 1 if they do match

{mem_section}### Rulebook examples
{self.examples.get("location","")}

### Pair
{{"location1":"{left}","location2":"{right}"}}

### Your JSON:
"""
        #print(prompt)
        
        data = self._call_llm(prompt)
        if not data:
            return MatchResult(0)
        try:
            binary_val = int(data.get("binary", 0))
        except Exception:
            binary_val = 0
        binary_val = 1 if binary_val == 1 else 0
        return MatchResult(binary_val)

    # Bulk compare helper
    
    def compare(self, pairs, field="name"):
        results = []
        if field == "name":
            fn = self.match_name
        elif field == "location":
            fn = self.match_location
        else:
            raise ValueError(f"Unknown field '{field}' for compare()")
        for left, right in tqdm(pairs, desc=f"Matching {field}", unit="pair"):
            results.append(fn(left, right))
        return results
