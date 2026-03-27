from __future__ import annotations

from pathlib import Path
import csv
import hashlib
from datetime import datetime, timezone


# ==========================
# USER SETTINGS (EDIT THESE)
# ==========================
ROOT_DIR = Path(r"C:\Users\daniel.gomez\Documents\RUBIC_AI_TRAIN_DL\LLRS_split\train")   
OUTPUT_CSV = Path("train_data.csv")         

# Optional: include file hash (slower on big files)
COMPUTE_SHA256 = False

# Optional: skip some extensions (example)
SKIP_EXTENSIONS = {".tmp"}  # set() to disable


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash of a file (reads in chunks)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"ROOT_DIR not found: {ROOT_DIR}")

    # Collect file rows
    rows = []
    for p in ROOT_DIR.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() in SKIP_EXTENSIONS:
            continue

        stat = p.stat()
        rel_path = p.relative_to(ROOT_DIR)

        row = {
            "root_dir": str(ROOT_DIR),
            "relative_path": str(rel_path),
            "filename": p.name,
            "extension": p.suffix.lower(),
            "size_bytes": stat.st_size,
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "sha256": sha256_file(p) if COMPUTE_SHA256 else "",
        }
        rows.append(row)

    # Write CSV
    fieldnames = ["root_dir", "relative_path", "filename", "extension", "size_bytes", "modified_utc", "sha256"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Saved {len(rows)} files to: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
