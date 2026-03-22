import zarr
import numpy as np
import pandas as pd
import os
import gc
from pathlib import Path
from scipy.sparse import csr_matrix, vstack
from sklearn.decomposition import TruncatedSVD
from data_preprocessing import encode_012, check_call_rate



def fit_pca_sparse(csv_path: str, encoded_dir: str,
                   n_components: int = 50):
    df         = pd.read_csv(csv_path, header=0)
    sample_ids = df.iloc[:, 0].tolist()

    rows      = []
    valid_ids = []

    for sample_id in sample_ids:
        out_path = os.path.join(encoded_dir, f"{sample_id}.dat")
        if not os.path.exists(out_path):
            continue
        try:
            mm      = np.memmap(out_path, dtype=np.int8, mode='r')
            flag, _ = check_call_rate(mm, threshold=0.10)
            if flag:
                del mm
                print(f"[REJECT] {sample_id}")
                continue

            sparse_vec = csr_matrix(mm.astype(np.float32))
            rows.append(sparse_vec)
            valid_ids.append(sample_id)
            del mm
            print(f"[SPARSE] {sample_id}", flush=True)

        except Exception as e:
            print(f"[ERROR] {sample_id}: {e}")

    print(f"Stacking {len(rows)} sparse rows...")
    X_sparse = vstack(rows)
    del rows
    gc.collect()

    print(f"Fitting TruncatedSVD with {n_components} components...")
    svd   = TruncatedSVD(n_components=n_components, random_state=42)
    X_pca = svd.fit_transform(X_sparse)

    print(f"Explained variance: {svd.explained_variance_ratio_.sum():.2%}")
    return svd, X_pca, valid_ids