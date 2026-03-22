from data_preprocessing import encode_012, check_call_rate, encode_and_save_filtered, compute_maf_mask
from pca import fit_pca_sparse
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import os
import pickle
from sklearn.preprocessing import LabelEncoder
import joblib

def train_xgboost_from_arrays(X: np.ndarray, y: np.ndarray, chunk_size: int = 50):
    le        = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes: {le.classes_}")

    booster = None
    n       = len(X)

    for start in range(0, n, chunk_size):
        end   = min(start + chunk_size, n)
        chunk = xgb.DMatrix(X[start:end], label=y_encoded[start:end])

        booster = xgb.train(
            params={
                "objective":   "multi:softmax",
                "num_class":   len(le.classes_),
                "max_depth":   6,
                "eta":         0.1,
                "tree_method": "hist",
                "device":      "cuda",
                "eval_metric": "mlogloss",
            },
            dtrain=chunk,
            num_boost_round=10,
            xgb_model=booster,
            verbose_eval=False,
        )
        print(f"[CHUNK] {end}/{n} samples processed")

    booster.save_model("model.ubj")
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print("Model saved to model.ubj")
    return booster, le


if __name__ == "__main__":
    CSV         = "/home/sashreekkumar/Documents/Projects/malariagen/data/data_csv/sampled_100.csv"
    FOLDER      = "/home/sashreekkumar/Documents/Projects/malariagen/extracted/"
    MASK_PATH   = "/home/sashreekkumar/Documents/Projects/malariagen/cache/maf_mask.npy"
    ENCODED_DIR = "/home/sashreekkumar/Documents/Projects/malariagen/cache/encoded_filtered/"

    os.makedirs(os.path.dirname(MASK_PATH), exist_ok=True)
    os.makedirs(ENCODED_DIR, exist_ok=True)

    # step 1: MAF mask
    if os.path.exists(MASK_PATH):
        print("Loading existing MAF mask...")
        maf_mask = np.load(MASK_PATH)
    else:
        maf_mask = compute_maf_mask(CSV, FOLDER, maf_threshold=0.05,
                                    chunk_size=100000, out_path=MASK_PATH)

    # step 2: encode + filter
    df = pd.read_csv(CSV, header=0)
    for sid in df.iloc[:, 0].tolist():
        sample_path = Path(FOLDER) / str(sid)
        out_path    = os.path.join(ENCODED_DIR, f"{sid}.dat")
        if not sample_path.exists() or os.path.exists(out_path):
            continue
        encode_and_save_filtered(str(sample_path), ENCODED_DIR, sid, maf_mask)
        print(f"[ENCODE] {sid}")

    # step 3: PCA
    svd, X_pca, valid_ids = fit_pca_sparse(CSV, ENCODED_DIR, n_components=50)
    joblib.dump(svd, "/home/sashreekkumar/Documents/Projects/malariagen/cache/svd_model.pkl")
    np.save("/home/sashreekkumar/Documents/Projects/malariagen/cache/X_pca.npy", X_pca)
    np.save("/home/sashreekkumar/Documents/Projects/malariagen/cache/valid_ids.npy", np.array(valid_ids))

    # step 4: labels
    id_to_label = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 2]))
    y           = np.array([id_to_label[str(sid)] for sid in valid_ids])

    # step 5: train
    model, le = train_xgboost_from_arrays(X_pca, y)