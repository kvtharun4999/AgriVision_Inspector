"""
download_hf_dataset.py
-----------------------
Downloads and extracts Dataset B from Hugging Face:
"Project-AgML/turmeric_leaf_disease_classification"

Exact Class Mapping:
  0: Aphids_Disease
  1: Leaf_Blotch
  2: Healthy_Leaf
  3: Leaf_Spot

Features:
- Streaming chunked download with 300s timeout & automatic retries.
- Skips already completed partitions.
- Saves images into:
    mini_project/dataset_hf/
      ├── Leaf_Spot/
      ├── Leaf_Blotch/
      ├── Healthy_Leaf/
      └── Aphids_Disease/
"""

import os
import sys
import io
import time
import json
import logging
import requests
import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_hf")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, ".download_progress.json")

# Verified class label mapping from Hugging Face dataset metadata
LABEL_NAMES = {
    0: "Aphids_Disease",
    1: "Leaf_Blotch",
    2: "Healthy_Leaf",
    3: "Leaf_Spot",
}


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed_partitions": []}


def save_progress(completed):
    try:
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"completed_partitions": completed}, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save progress: {e}")


def download_with_retry(url, max_retries=4, timeout=300):
    """Download large parquet file using streaming with retries."""
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Connecting to {url} (attempt {attempt}/{max_retries})...")
            with requests.get(url, stream=True, timeout=timeout) as res:
                res.raise_for_status()
                chunks = []
                downloaded_bytes = 0
                for chunk in res.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                    if chunk:
                        chunks.append(chunk)
                        downloaded_bytes += len(chunk)
                        if downloaded_bytes % (50 * 1024 * 1024) == 0:
                            logger.info(f"  Downloaded {downloaded_bytes / (1024 * 1024):.1f} MB...")
                return b"".join(chunks)
        except Exception as e:
            logger.warning(f"Download attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(3 * attempt)
            else:
                raise


def download_dataset():
    """Main workflow to download and process Hugging Face parquet partitions."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    progress = load_progress()
    completed = set(progress.get("completed_partitions", []))

    print("=" * 60)
    print("Downloading 'Project-AgML/turmeric_leaf_disease_classification'")
    print("=" * 60)

    parquet_api = "https://huggingface.co/api/datasets/Project-AgML/turmeric_leaf_disease_classification/parquet/raw/train"
    res = requests.get(parquet_api, timeout=15)
    res.raise_for_status()
    parquet_urls = res.json()
    logger.info(f"Total partitions available: {len(parquet_urls)}")

    # Clean legacy Class_3 if present from previous run
    legacy_class3 = os.path.join(OUTPUT_DIR, "Class_3")
    if os.path.exists(legacy_class3):
        import shutil
        shutil.rmtree(legacy_class3, ignore_errors=True)

    saved_counts = {}

    for p_idx, url in enumerate(parquet_urls):
        if p_idx in completed:
            logger.info(f"Partition [{p_idx + 1}/{len(parquet_urls)}] already processed. Skipping.")
            continue

        logger.info(f"\nProcessing partition [{p_idx + 1}/{len(parquet_urls)}]...")
        parquet_bytes = download_with_retry(url, max_retries=4, timeout=300)

        logger.info("Reading parquet data...")
        df = pd.read_parquet(io.BytesIO(parquet_bytes))
        logger.info(f"Loaded {len(df)} images from partition {p_idx + 1}.")

        img_col = next((c for c in ["image", "img"] if c in df.columns), df.columns[0])
        lbl_col = next((c for c in ["label", "labels", "class"] if c in df.columns), df.columns[1])

        for idx, row in df.iterrows():
            raw_lbl = row[lbl_col]
            if isinstance(raw_lbl, int):
                class_name = LABEL_NAMES.get(raw_lbl, f"Class_{raw_lbl}")
            else:
                class_name = str(raw_lbl).strip().replace(" ", "_")

            class_folder = os.path.join(OUTPUT_DIR, class_name)
            os.makedirs(class_folder, exist_ok=True)

            img_data = row[img_col]
            img_obj = None
            if isinstance(img_data, dict) and "bytes" in img_data and img_data["bytes"]:
                img_obj = Image.open(io.BytesIO(img_data["bytes"]))
            elif isinstance(img_data, bytes):
                img_obj = Image.open(io.BytesIO(img_data))
            elif isinstance(img_data, Image.Image):
                img_obj = img_data

            if img_obj:
                save_path = os.path.join(class_folder, f"hf_part{p_idx}_{idx:04d}.jpg")
                img_obj.convert("RGB").save(save_path, "JPEG", quality=92)
                saved_counts[class_name] = saved_counts.get(class_name, 0) + 1

        completed.add(p_idx)
        save_progress(list(completed))
        logger.info(f"Completed partition {p_idx + 1}/{len(parquet_urls)} successfully.")

    print("\n" + "=" * 60)
    print("ALL PARTITIONS DOWNLOADED SUCCESSFULLY!")
    print(f"Location: {OUTPUT_DIR}")
    for d in sorted(os.listdir(OUTPUT_DIR)):
        full_d = os.path.join(OUTPUT_DIR, d)
        if os.path.isdir(full_d):
            print(f"  - {d}: {len(os.listdir(full_d))} images")
    print("=" * 60)


if __name__ == "__main__":
    download_dataset()
