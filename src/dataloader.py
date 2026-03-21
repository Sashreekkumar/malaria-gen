import os
import csv
import requests
import zipfile
import shutil
from tqdm import tqdm
from pathlib import Path

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─── Config ─────────────────────────────────────────────────────────────────

FEATURES_CSV  = "data/data_csv/sampled_100.csv"          # CSV with columns: s.no, features, label
OUTPUT_DIR    = "./extracted"
TEMP_DIR      = "./temp"
PROGRESS_FILE = "progress.txt"
CHUNK_SIZE    = 1024 * 1024 * 4
LIMIT         = 3                   # set to None to process all files


# ─── Progress tracking ───────────────────────────────────────────────────────

def load_progress() -> set:
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def save_progress(url: str):
    with open(PROGRESS_FILE, 'a') as f:
        f.write(url + '\n')


# ─── Data loading ────────────────────────────────────────────────────────────

def row_generator(csv_path: str):
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sno   = row['id'].strip()
            url   = row['features'].strip()
            label = row['labels'].strip()
            if sno and url and label:
                yield sno, url, label


# ─── Download ────────────────────────────────────────────────────────────────

def stream_download(url: str, dest: str):
    with requests.get(url, stream=True, timeout=60, verify=False) as r:
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
    for member in zf.infolist():
        parts = Path(member.filename).parts
        for i, part in enumerate(parts):
            if (part == "GT"
                    and i >= 2
                    and parts[i-1] == "calldata"
                    and parts[i-2] == "2L"):
                gt_prefix = '/'.join(parts[:i+1]) + '/'
                return gt_prefix
    raise FileNotFoundError(
        "Could not find 2L/calldata/GT/ in zip.\n"
        f"Top-level entries: {sorted(set(m.filename.split('/')[0] for m in zf.infolist()))}"
    )


def process_one(sno: str, url: str, label: str) -> bool:
    zip_path    = os.path.join(TEMP_DIR, f"{sno}.zip")
    output_path = os.path.join(OUTPUT_DIR, sno)

    if os.path.exists(output_path):
        print(f"  ✓ Already done, skipping")
        return True

    try:
        stream_download(url, zip_path)
        extract_gt_folder(zip_path, sno)
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        return False

    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)


def extract_gt_folder(zip_path: str, sno: str):
    save_root = os.path.join(OUTPUT_DIR, sno, "gt")
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
    all_rows = list(row_generator(FEATURES_CSV))
    all_rows = all_rows[:LIMIT] if LIMIT else all_rows
    total    = len(all_rows)
    pending  = [(sno, url, label) for sno, url, label in all_rows if url not in done]

    print(f"Total   : {total}")
    print(f"Done    : {total - len(pending)}")
    print(f"Pending : {len(pending)}")
    print(f"Limit   : {LIMIT if LIMIT else 'None (all)'}\n")

    failed = []

    for i, (sno, url, label) in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] s.no: {sno}  (label: {label})")
        success = process_one(sno, url, label)

        if success:
            save_progress(url)
            print(f"  ✓ Saved → {OUTPUT_DIR}/{sno}/gt/")
        else:
            failed.append((sno, url, label))
            print(f"  ✗ Failed")

    print(f"\n{'─' * 50}")
    print(f"Done.  ✓ {len(pending) - len(failed)}   ✗ {len(failed)}")

    if failed:
        with open("failed.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['s.no', 'features', 'label'])
            for sno, url, label in failed:
                writer.writerow([sno, url, label])
        print(f"Failed entries saved → failed.csv (re-run script to retry)")


if __name__ == "__main__":
    run()