from sklearn.decomposition import IncrementalPCA
import numpy as np
import pandas as pd
from pathlib import Path
from data_preprocessing import load_zarr_to_numpy, encode_012, check_call_rate
import zarr

import numpy as np
import os

def encode_and_save(sample_path: str, out_dir: str, sample_id: int, chunk_size: int = 100000) -> str:
    """Encode a sample and save to disk as memmap. Returns path."""
    import zarr
    from data_preprocessing import encode_012

    store    = zarr.open(str(sample_path) + "/gt", mode='r')
    n        = store.shape[0]
    out_path = os.path.join(out_dir, f"{sample_id}.dat")

    mm = np.memmap(out_path, dtype=np.int8, mode='w+', shape=(n,))

    for start in range(0, n, chunk_size):
        end              = min(start + chunk_size, n)
        chunk            = store[start:end].squeeze(1)
        mm[start:end]    = encode_012(chunk)

    mm.flush()
    del mm
    return out_path


def fit_incremental_pca(csv_path: str, snp_folder: str, encoded_dir: str,
                        n_components: int = 50, chunk_size: int = 100000):
    from data_preprocessing import check_call_rate
    import pandas as pd
    from pathlib import Path

    os.makedirs(encoded_dir, exist_ok=True)

    df         = pd.read_csv(csv_path, header=0)
    sample_ids = df.iloc[:, 0].tolist()

    ipca    = IncrementalPCA(n_components=n_components)
    batch   = []
    n_vars  = None

    for sample_id in sample_ids:
        sample_path = Path(snp_folder) / str(sample_id)
        if not sample_path.exists():
            continue
        try:
            out_path = encode_and_save(str(sample_path), encoded_dir, sample_id, chunk_size)

            # load as memmap — no RAM copy
            mm       = np.memmap(out_path, dtype=np.int8, mode='r')
            n_vars   = len(mm)

            flag, _  = check_call_rate(mm)
            if flag:
                del mm
                os.remove(out_path)
                print(f"[REJECT] {sample_id}: missing rate > 5%")
                continue

            batch.append(mm.astype(np.float32).reshape(1, -1))
            del mm  # release memmap handle, file stays on disk

            if len(batch) >= n_components:
                ipca.partial_fit(np.vstack(batch))
                print(f"[PCA] Fitted batch of {len(batch)}")
                batch = []

        except Exception as e:
            print(f"[ERROR] {sample_id}: {e}")

    if len(batch) >= n_components:
        ipca.partial_fit(np.vstack(batch))
        print(f"[PCA] Fitted final batch of {len(batch)}")

    return ipca


def transform_with_pca(ipca, csv_path: str, snp_folder: str, encoded_dir: str):
    import pandas as pd
    from pathlib import Path

    df         = pd.read_csv(csv_path, header=0)
    sample_ids = df.iloc[:, 0].tolist()
    labels     = df.iloc[:, 2].tolist()

    X, y = [], []

    for sample_id, label in zip(sample_ids, labels):
        out_path = os.path.join(encoded_dir, f"{sample_id}.dat")
        if not os.path.exists(out_path):
            continue
        try:
            mm      = np.memmap(out_path, dtype=np.int8, mode='r')
            vec_pca = ipca.transform(mm.astype(np.float32).reshape(1, -1))
            X.append(vec_pca[0])
            y.append(label)
            del mm
            print(f"[TRANSFORM] {sample_id}")
        except Exception as e:
            print(f"[ERROR] {sample_id}: {e}")

    return np.array(X), np.array(y)