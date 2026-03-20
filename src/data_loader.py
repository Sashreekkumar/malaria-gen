import os
import requests
import zipfile
import shutil
from tqdm import tqdm
from pathlib import Path

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) 

# ─── Config ─────────────────────────────────────────────────────────────────

FEATURES_TSV  = "data_tsv/features.tsv"
LABELS_TSV    = "data_tsv/labels.tsv"
OUTPUT_DIR    = "./extracted"
TEMP_DIR      = "./temp"
PROGRESS_FILE = "progress.txt"
CHUNK_SIZE    = 1024 * 1024 * 4
LIMIT         = 3        # set to None to process all 2660 files


# ─── Progress tracking ───────────────────────────────────────────────────────

def load_progress() -> set:
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def save_progress(label: str):
    with open(PROGRESS_FILE, 'a') as f:
        f.write(label + '\n')


# ─── Data loading ────────────────────────────────────────────────────────────

def row_generator(features_tsv: str, labels_tsv: str):
    with open(features_tsv, 'r') as ff, open(labels_tsv, 'r') as fl:
        for url_line, label_line in zip(ff, fl):
            url   = url_line.strip()
            label = label_line.strip()
            if url and label:
                yield url, label


# ─── Download ────────────────────────────────────────────────────────────────

def stream_download(url: str, dest: str):
    with requests.get(url, stream=True, timeout=60, verify=False) as r:  # ← verify=False
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(dest, 'wb') as f, tqdm(
            desc="  download",
            total=total,
            unit='iB', unit_scale=True, unit_divisor=1024,
            leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                bar.update(len(chunk))


# ─── Extraction ──────────────────────────────────────────────────────────────

def get_gt_folder_prefix(zf: zipfile.ZipFile) -> str:
    """
    Find the path ending in exactly 2L/calldata/GT/
    Ignores 3L, X, etc.
    """
    for member in zf.infolist():
        parts = Path(member.filename).parts
        for i, part in enumerate(parts):
            if (part == "GT"
                    and i >= 2
                    and parts[i-1] == "calldata"
                    and parts[i-2] == "2L"):          # ← locked to 2L only
                gt_prefix = '/'.join(parts[:i+1]) + '/'
                return gt_prefix
    raise FileNotFoundError(
        "Could not find 2L/calldata/GT/ in zip.\n"
        f"Top-level entries: {sorted(set(m.filename.split('/')[0] for m in zf.infolist()))}"
    )


def process_one(url: str, label: str) -> bool:
    """Use sample_id (from URL) as folder name, label is just metadata."""
    # Extract sample ID from URL: AR0047-C.gatk.zarr.zip → AR0047-C
    filename = url.split('/')[-1]                    # AR0047-C.gatk.zarr.zip
    sample_id = filename.replace('.gatk.zarr.zip', '') # AR0047-C

    zip_path    = os.path.join(TEMP_DIR, f"{sample_id}.zip")
    output_path = os.path.join(OUTPUT_DIR, sample_id)

    if os.path.exists(output_path):
        print(f"  ✓ Already done, skipping")
        return True

    try:
        stream_download(url, zip_path)
        extract_gt_folder(zip_path, sample_id)
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        return False

    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)


def extract_gt_folder(zip_path: str, sample_id: str):
    """
    zip structure:  xyz.gatk.zarr/xyz-C/2L/calldata/GT/*
    saved as:       extracted/AR0047-C/gt/*        ← no extra /data
    """
    save_root = os.path.join(OUTPUT_DIR, sample_id, "gt")   # ← removed /data
    os.makedirs(save_root, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        gt_prefix = get_gt_folder_prefix(zf)
        print(f"  Found GT at: {gt_prefix}")

        members = [m for m in zf.infolist() if m.filename.startswith(gt_prefix)]

        if not members:
            raise FileNotFoundError(f"No files found under prefix: {gt_prefix}")

        for member in tqdm(members, desc="  extract", leave=False):
            relative_path = member.filename[len(gt_prefix):]
            if not relative_path:
                continue
            dest_path = os.path.join(save_root, relative_path)
            if member.is_dir():
                os.makedirs(dest_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with zf.open(member) as src, open(dest_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst)


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR,   exist_ok=True)

    done     = load_progress()
    all_rows = list(row_generator(FEATURES_TSV, LABELS_TSV))
    all_rows = all_rows[:LIMIT] if LIMIT else all_rows
    total    = len(all_rows)
    pending  = [(url, label) for url, label in all_rows if url not in done]  # ← key on URL now

    print(f"Total   : {total}")
    print(f"Done    : {total - len(pending)}")
    print(f"Pending : {len(pending)}")
    print(f"Limit   : {LIMIT if LIMIT else 'None (all)'}\n")

    failed = []

    for i, (url, label) in enumerate(pending, 1):
        sample_id = url.split('/')[-1].replace('.gatk.zarr.zip', '')
        print(f"[{i}/{len(pending)}] {sample_id}  (label: {label})")
        success = process_one(url, label)

        if success:
            save_progress(url)   # ← save URL as key, not label
            print(f"  ✓ Saved → {OUTPUT_DIR}/{sample_id}/gt/data/")
        else:
            failed.append((url, label))
            print(f"  ✗ Failed")

    print(f"\n{'─' * 50}")
    print(f"Done.  ✓ {len(pending) - len(failed)}   ✗ {len(failed)}")

    if failed:
        with open("failed.tsv", 'w') as f:
            for url, label in failed:
                f.write(f"{url}\t{label}\n")
        print(f"Failed entries saved → failed.tsv (re-run script to retry)")


if __name__ == "__main__":
    run()