from data_preprocessing import load_zarr_to_numpy, encode_012, check_call_rate
from pca import transform_with_pca, fit_incremental_pca
import zarr
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_dataset(csv_path: str, snp_folder: str, zarr_key: str = None):
    df = pd.read_csv(csv_path, header=0)  
    sample_ids = df.iloc[:, 0].tolist()
    labels     = df.iloc[:, 2].tolist()

    X, y = [], []
    rejected = []

    for sample_id, label in zip(sample_ids, labels):
        sample_path = Path(snp_folder) / str(sample_id)
        print(f"Trying path: {sample_path}") 

        if not sample_path.exists():
            print(f"[SKIP] ID {sample_id}: folder not found")
            rejected.append((sample_id, "folder not found"))
            continue

        try:
            raw = load_zarr_to_numpy(str(sample_path))
            encoded = encode_012(raw)
            flag, missing_rate = check_call_rate(encoded)

            if flag:
                print(f"[REJECT] ID {sample_id}: {missing_rate:.2%} missing > 10%")
                rejected.append((sample_id, f"missing rate {missing_rate:}"))
                continue

            X.append(encoded.flatten())
            y.append(label)

        except Exception as e:
            print(f"[ERROR] ID {sample_id}: {e}")
            rejected.append((sample_id, str(e)))

    print(f"\nLoaded {len(X)} samples, rejected {len(rejected)}")
    return np.array(X), np.array(y)


# def train_xgboost(csv_path: str, snp_folder: str, zarr_key: str = None, chunk_size: int = 500):
#     X, y = load_dataset(csv_path, snp_folder, zarr_key)

#     if len(X) == 0:
#         raise ValueError("No samples passed the missing rate filter.")

#     booster = None
#     n = len(X)

#     for start in range(0, n, chunk_size):
#         end   = min(start + chunk_size, n)
#         chunk = xgb.DMatrix(X[start:end], label=y[start:end])

#         booster = xgb.train(
#             params={
#                 "objective":   "binary:logistic",
#                 "max_depth":   6,
#                 "eta":         0.1,
#                 "tree_method": "hist",
#                 "device":      "cuda",
#                 "eval_metric": "logloss",
#             },
#             dtrain=chunk,
#             num_boost_round=10,
#             xgb_model=booster,
#             verbose_eval=False,
#         )

#         print(f"[CHUNK] {end}/{n} samples processed")

#     booster.save_model("model.ubj")
#     print("Model saved to model.ubj")
#     return booster

def train_xgboost_from_arrays(X: np.ndarray, y: np.ndarray, chunk_size: int = 500):
    booster = None
    n = len(X)

    for start in range(0, n, chunk_size):
        end   = min(start + chunk_size, n)
        chunk = xgb.DMatrix(X[start:end], label=y[start:end])

        booster = xgb.train(
            params={
                "objective":   "multi:softmax",
                "num_class":   len(np.unique(y)),
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
    print("Model saved to model.ubj")
    return booster

if __name__ == "__main__":
   if __name__ == "__main__":
    CSV          = "/home/sashreekkumar/Documents/Projects/malariagen/data/data_csv/sampled_100.csv"
    FOLDER       = "/home/sashreekkumar/Documents/Projects/malariagen/extracted/"
    ENCODED_DIR  = "/home/sashreekkumar/Documents/Projects/malariagen/encoded_cache/"

    ipca      = fit_incremental_pca(CSV, FOLDER, ENCODED_DIR, n_components=50)
    X, y      = transform_with_pca(ipca, CSV, FOLDER, ENCODED_DIR)
    model     = train_xgboost_from_arrays(X, y)