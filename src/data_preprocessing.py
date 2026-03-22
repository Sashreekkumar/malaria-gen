import zarr
import numpy as np
import os

# Applying 0/1/2 Encoding
def encode_012(arr: np.ndarray, ref_allele_val: int = 0) -> np.ndarray:
    arr = np.asarray(arr)

    missing_mask = np.any(arr == -1, axis=-1)
    encoded = np.sum(arr != ref_allele_val, axis=-1).astype(np.int8)
    encoded[missing_mask] = -1

    return encoded

# Check Call Rate
'''
Call Rate = 90%
Missing Rate =10%

'''
def check_call_rate(arr: np.ndarray, threshold: float = 0.1) -> tuple[bool, float]:
    missing_rate = np.mean(arr == -1)
    flag = missing_rate > threshold
    return flag, missing_rate


def compute_maf_mask(csv_path: str, snp_folder: str,
                     maf_threshold: float = 0.05,
                     chunk_size: int = 100000,
                     out_path: str = "maf_mask.npy") -> np.ndarray:
    df         = pd.read_csv(csv_path, header=0)
    sample_ids = df.iloc[:, 0].tolist()

    n_variants = None
    valid_ids  = []
    for sid in sample_ids:
        p = Path(snp_folder) / str(sid) / "gt"
        if p.exists():
            store      = zarr.open(str(Path(snp_folder) / str(sid) / "gt"), mode='r')
            n_variants = store.shape[0]
            valid_ids.append(sid)

    print(f"n_variants: {n_variants}, valid samples: {len(valid_ids)}")

    alt_count   = np.zeros(n_variants, dtype=np.int32)
    total_count = np.zeros(n_variants, dtype=np.int32)

    for sid in valid_ids:
        store = zarr.open(str(Path(snp_folder) / str(sid) / "gt"), mode='r')
        print(f"[MAF] counting {sid}", flush=True)

        for start in range(0, n_variants, chunk_size):
            end          = min(start + chunk_size, n_variants)
            chunk        = store[start:end].squeeze(1)
            missing      = np.any(chunk == -1, axis=-1)
            alt_count[start:end]   += np.sum(chunk == 1, axis=-1) * ~missing
            total_count[start:end] += 2 * ~missing

    with np.errstate(invalid='ignore', divide='ignore'):
        freq = np.where(total_count > 0, alt_count / total_count, 0.0)
        maf  = np.minimum(freq, 1 - freq)

    mask = maf >= maf_threshold
    np.save(out_path, mask)
    print(f"[MAF] {mask.sum()} / {n_variants} variants pass MAF >= {maf_threshold}")
    return mask


def encode_and_save_filtered(sample_path: str, out_dir: str, sample_id: int,
                             maf_mask: np.ndarray, chunk_size: int = 100000) -> str:
    
    if os.path.exists(str(sample_path) + "/GT"):
        gt_path = str(sample_path) + "/GT"
    elif os.path.exists(str(sample_path) + "/gt"):
        gt_path = str(sample_path) + "/gt"
    else:
        raise FileNotFoundError(f"No gt or GT folder found in {sample_path}")
    store = zarr.open(gt_path, mode='r')  # ← use the detected path
    n        = store.shape[0]
    n_kept   = int(maf_mask.sum())
    out_path = os.path.join(out_dir, f"{sample_id}.dat")
    mm       = np.memmap(out_path, dtype=np.int8, mode='w+', shape=(n_kept,))

    out_idx = 0
    for start in range(0, n, chunk_size):
        end        = min(start + chunk_size, n)
        chunk_mask = maf_mask[start:end]
        if not chunk_mask.any():
            continue
        chunk    = store[start:end].squeeze(1)
        encoded  = encode_012(chunk)
        filtered = encoded[chunk_mask]
        mm[out_idx:out_idx + len(filtered)] = filtered
        out_idx += len(filtered)

    mm.flush()
    del mm
    return out_path