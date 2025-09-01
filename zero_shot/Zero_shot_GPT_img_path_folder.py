import os
import base64
import mimetypes
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from tqdm import tqdm

# ---- OpenAI client (v1+) ----
from openai import OpenAI

# =============================
# 1) Configuration
# =============================
# Read API key from file (strip newline/space)
with open("openai_api_key.txt", "r", encoding="utf-8") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

MODEL = "gpt-4o"
FOLDER_PATH = "image_proof" # <- your images folder
OUTPUT_CSV  = "proof_results_image_folder.csv"
DELAY_SECONDS = 8    # set 0 for no delay; increase if you hit rate limits
MAX_RETRIES = 3
BACKOFF_BASE = 2.0   # exponential backoff multiplier

# Accepted image extensions (fast path)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff", ".heic"}

# =============================
# 2) Helpers
# =============================
def encode_image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def guess_mime_type(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    # Prefer real type if image/*; otherwise default to a safe image type
    if mt and mt.startswith("image/"):
        return mt
    # Handle HEIC explicitly if extension hints it
    if Path(path).suffix.lower() == ".heic":
        return "image/heic"
    return "image/jpeg"

def collect_images(folder: str) -> List[Path]:
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = [fp for fp in p.iterdir() if fp.is_file() and fp.suffix.lower() in IMAGE_EXTS]
    # Fallback: also accept files that guess as image even if extension uncommon
    more = [fp for fp in p.iterdir()
            if fp.is_file() and fp not in files and (guess_mime_type(str(fp)) or "").startswith("image/")]
    return files + more

def strip_code_fences(s: str) -> str:
    t = s.strip()
    if t.startswith("```"):
        # Remove starting fence line
        t = t.split("\n", 1)[-1]
        # Remove trailing ```
        if t.endswith("```"):
            t = t[: -3]
    return t.strip()

# =============================
# 3) System prompt (UPDATED)
# =============================
system_prompt = """
You are an expert in architectural analysis and computer vision. Given a facade image of a commercial or industrial 
building, predict the following attributes:
    
1. Use (Occupancy type) – Choose the most likely option:
- Commercial – Retail/Office (e.g., shops, malls, office towers)
- Commercial – Warehouse/Big-box (e.g., supermarkets, logistics centers)
- Industrial – Light (e.g., small factories, workshops, plants)
- Industrial – Heavy (e.g., large plants, refineries, steelworks)
- Mixed-use (commercial/industrial combination)
- Unknown (if not visually inferable)

(Consider signage, glazing style, dock doors, open bays, facade regularity, and building scale.)

2. Height in meters – Estimate the total height of the building in meters.
- Use facade proportions, number of levels, and reference objects (cars, trucks, loading docks, garage doors, people, trees, streetlights, fences).
- Warehouses/factories may have one level but large height; compare with large openings & vehicles.
- If uncertain, provide a range (e.g., "8–12").

3. Primary construction material – Choose the most likely structural material:
- Concrete, Steel, Reinforced masonry, Confined masonry, Unreinforced masonry, Wood, Other, Unknown

Return ONLY compact JSON:
{"use": "...", "height_m": "...", "material": "..."}
"""

REQUIRED_KEYS = {"use", "height_m", "material"}

def request_prediction(image_path: str) -> Dict[str, Any]:
    """Call the API once and return a parsed dict (or raise)."""
    b64 = encode_image_to_base64(image_path)
    mime = guess_mime_type(image_path)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}" }},
                    {"type": "text", "text": "Analyze this single image and return only the JSON described."}
                ],
            },
        ],
        temperature=0.0,
        max_tokens=250,
        # 'seed' isn't supported on all endpoints; omit to avoid errors
    )

    content = resp.choices[0].message.content.strip() if resp.choices else ""
    content = strip_code_fences(content)

    # Parse JSON; if it fails, raise so caller logs raw_response
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError(f"Non-dict JSON: {content}")

    missing = REQUIRED_KEYS - set(parsed.keys())
    if missing:
        raise ValueError(f"Missing JSON keys {missing}: {content}")

    # Minimal sanity checks
    for k in ["use", "height_m", "material"]:
        if not isinstance(parsed[k], str):
            raise ValueError(f'"{k}" must be a string: {content}')

    return {"parsed": parsed, "raw": content}

def predict_with_retries(image_path: str) -> Dict[str, Any]:
    """Retry wrapper with exponential backoff, returning dict ready for CSV row."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.time()
            result = request_prediction(image_path)
            _elapsed = time.time() - t0

            p = result["parsed"]
            return {
                "filename": Path(image_path).name,
                "use": p.get("use", ""),
                "height_m": p.get("height_m", ""),
                "material": p.get("material", ""),
                "raw_json": result["raw"],
                "error": "",
            }
        except Exception as e:
            last_err = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE ** (attempt - 1))
            else:
                return {
                    "filename": Path(image_path).name,
                    "use": "",
                    "height_m": "",
                    "material": "",
                    "raw_json": "",
                    "error": last_err,
                }

# =============================
# 4) Main loop (tqdm style)
# =============================
if __name__ == "__main__":
    image_files = collect_images(FOLDER_PATH)
    if not image_files:
        raise RuntimeError(f"No image files found in: {FOLDER_PATH}")

    results = []
    for fp in tqdm(image_files, desc="Processing images"):
        row = predict_with_retries(str(fp))
        results.append(row)

        # Optional pacing to respect rate limits — set DELAY_SECONDS=0 to disable
        if DELAY_SECONDS > 0:
            time.sleep(DELAY_SECONDS)

    # Save to CSV
    df = pd.DataFrame(results)
    cols = ["filename", "use", "height_m", "material", "raw_json", "error"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n✅ Predictions saved to: {OUTPUT_CSV}")
