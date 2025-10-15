import os
import cv2
import json
import time
import base64
import random
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from tqdm import tqdm

import openai
from ultralytics import YOLO
from get_building_orientation import get_street_view_image  

# =========================
# 0) Configuration
# =========================
# Keys from files (keeps env clean & reproducible)
with open("gsv_api_key.txt", "r") as f:
    GOOGLE_API_KEY = f.read().strip()

with open("openai_api_key.txt", "r") as f:
    openai.api_key = f.read().strip()

####################################################################
################## --- user-------------############################
####################################################################
# I/O
INPUT_CSV  = "proof_coordinates.csv"
OUTPUT_CSV = "proof_results_coordinates.csv"
CHECKPOINT_EVERY = 20  # rows

####################################################################
################## --- user-------------############################
####################################################################

# YOLO
YOLO_WEIGHTS = "building_detector.pt"  # your trained weights
YOLO_CLASS_MAP = {0: "building-xzyh"}            # adapt if needed
YOLO_CONF = 0.15                                  # low threshold -> we’ll pick best anyway
YOLO_IOU = 0.45

# OpenAI
OPENAI_MODEL = "gpt-4o"
TEMPERATURE = 0.0
MAX_TOKENS = 120

# Rate limiting / retries
DELAY_SECONDS = 5         # set >0 if you see rate limits
MAX_RETRIES = 4
BACKOFF_BASE = 2.0          # exponential backoff
TIMEOUT_SEC = 45

# Image prep
MAX_DIM = 1024              # downscale long edge to this (keeps details, reduces cost)
JPEG_QUALITY = 85           # 80–88 is a good balance

# =========================
# 1) Prompt (compact but specific)
# =========================
SYSTEM_PROMPT = """
You are an expert in architectural analysis and computer vision. Given a facade image of a commercial or industrial 
building, predict the following attributes:

1) Use 

2) Height in meters 

3) Primary construction material 

"""

# =========================
# 2) Utilities
# =========================
def exponential_backoff(attempt: int, base: float = BACKOFF_BASE) -> float:
    # jitter to reduce thundering herd
    return (base ** attempt) + random.uniform(0, 0.5)

def resize_for_max_dim(img_bgr: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_dim:
        return img_bgr
    scale = max_dim / long_edge
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

def encode_jpeg_base64(img_bgr: np.ndarray, quality: int = JPEG_QUALITY) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode image.")
    return base64.b64encode(buf).decode("utf-8")

def parse_json_minified(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract first JSON object and parse it."""
    if not text:
        return None
    try:
        # Fast path: direct JSON
        return json.loads(text)
    except Exception:
        # fallback: find first {...}
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                return None
    return None

# =========================
# 3) Model setup (load once)
# =========================
def load_yolo_model() -> YOLO:
    model = YOLO(YOLO_WEIGHTS)
    # Prefer GPU if available
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    model.to(device)
    return model

YOLO_MODEL = load_yolo_model()

# =========================
# 4) Image acquisition & cropping
# =========================
def fetch_gsv_image_with_retry(lat: float, lon: float) -> np.ndarray:
    """Return BGR image from GSV (get_street_view_image returns (url, image_bgr))."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _url, img_bgr = get_street_view_image((float(lat), float(lon)), GOOGLE_API_KEY, angle=0)
            if isinstance(img_bgr, np.ndarray) and img_bgr.size > 0:
                return img_bgr
            last_err = RuntimeError("Empty image from GSV.")
        except Exception as e:
            last_err = e
        # backoff
        time.sleep(exponential_backoff(attempt))
    raise last_err if last_err else RuntimeError("Unknown GSV error.")

def detect_and_crop_building(img_bgr: np.ndarray) -> np.ndarray:
    """
    Runs YOLO on the frame and returns the highest-confidence building crop.
    Falls back to original image if no building box found.
    """
    try:
        results = YOLO_MODEL.predict(
            img_bgr,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            verbose=False
        )
        best_box = None
        best_conf = -1.0

        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue
            boxes = r.boxes
            if boxes.cls is None or boxes.conf is None or boxes.xyxy is None:
                continue

            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy().astype(int)

            for bb, cf, cl in zip(xyxy, confs, cls_ids):
                if YOLO_CLASS_MAP.get(cl, "") == "building-xzyh" and cf > best_conf:
                    best_conf = cf
                    best_box = bb

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            # clamp
            h, w = img_bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                crop = img_bgr[y1:y2, x1:x2]
                # guard tiny crops
                if crop.shape[0] > 40 and crop.shape[1] > 40:
                    return crop
        return img_bgr
    except Exception:
        # Safety: never crash the pipeline for a detection hiccup
        return img_bgr

# =========================
# 5) OpenAI call with retry
# =========================
def classify_image_minified_json(b64_jpeg: str) -> Dict[str, Any]:
    """
    Sends the base64 image to OpenAI and returns a dict with keys: use, height_m, material.
    On failure, raises.
    """
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg}"}},
                        {"type": "text", "text": "Please analyze this building image and provide the classification."}
                    ]}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                seed=42
            )
            content = (resp.choices[0].message.content or "").strip()
            data = parse_json_minified(content)
            if not data or not isinstance(data, dict):
                raise ValueError(f"Non-JSON or empty response: {content[:200]}")
            # Normalize keys
            out = {
                "use":       str(data.get("use", "")).strip(),
                "height_m":  str(data.get("height_m", "")).strip(),
                "material":  str(data.get("material", "")).strip(),
                "raw_response": content
            }
            # quick sanity
            if not out["use"] and not out["material"]:
                raise ValueError("Parsed JSON missing required fields.")
            return out
        except Exception as e:
            last_err = e
            # backoff + optional guard delay
            time.sleep(exponential_backoff(attempt))
    raise last_err if last_err else RuntimeError("Unknown OpenAI error.")

# =========================
# 6) Main
# =========================
def main():
    # Load input
    df = pd.read_csv(INPUT_CSV)
    if not {"latitude", "longitude"}.issubset(df.columns):
        raise ValueError("INPUT_CSV must contain 'latitude' and 'longitude' columns")

    # Resume support (merge if OUTPUT_CSV exists)
    processed: Dict[Tuple[float, float], Dict[str, Any]] = {}
    out_rows = []
    if os.path.exists(OUTPUT_CSV):
        prev = pd.read_csv(OUTPUT_CSV)
        # support both new & old formats
        key_cols = [c for c in ["latitude", "longitude"] if c in prev.columns]
        if len(key_cols) == 2:
            for _, r in prev.iterrows():
                processed[(float(r["latitude"]), float(r["longitude"]))] = r.to_dict()
            out_rows = prev.to_dict(orient="records")

    # Processing loop
    pbar = tqdm(range(len(df)), desc="Processing coordinates")
    new_count = 0
    for i in pbar:
        lat = float(df.at[i, "latitude"])
        lon = float(df.at[i, "longitude"])
        key = (lat, lon)

        if key in processed:
            # already done
            continue

        try:
            # 1) Fetch image
            img_bgr = fetch_gsv_image_with_retry(lat, lon)

            # 2) Crop to building (if detected)
            img_bgr = detect_and_crop_building(img_bgr)

            # 3) Downscale + encode
            img_bgr = resize_for_max_dim(img_bgr, MAX_DIM)
            b64_img = encode_jpeg_base64(img_bgr, JPEG_QUALITY)

            # 4) Classify
            pred = classify_image_minified_json(b64_img)

            row = {
                "latitude": lat,
                "longitude": lon,
                "use": pred.get("use", ""),
                "height_m": pred.get("height_m", ""),
                "material": pred.get("material", ""),
                "raw_response": pred.get("raw_response", "")
            }
        except Exception as e:
            row = {
                "latitude": lat,
                "longitude": lon,
                "use": "ERROR",
                "height_m": "",
                "material": "",
                "raw_response": f"ERROR: {repr(e)}"
            }

        out_rows.append(row)
        processed[key] = row
        new_count += 1

        # Optional pacing
        if DELAY_SECONDS > 0:
            time.sleep(DELAY_SECONDS)

        # Periodic checkpoint
        if new_count > 0 and new_count % CHECKPOINT_EVERY == 0:
            pd.DataFrame(out_rows).to_csv(OUTPUT_CSV, index=False)

    # Final write
    pd.DataFrame(out_rows).to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved predictions to: {OUTPUT_CSV}")
    print(f"   Total rows: {len(out_rows)} | Newly processed: {new_count}")

if __name__ == "__main__":
    # Graceful Ctrl+C handling
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Partial results (if any) were saved.")
