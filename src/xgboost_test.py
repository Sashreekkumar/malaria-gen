import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import joblib
import os
from pathlib import Path
from scipy.sparse import csr_matrix, vstack
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef
)
from data_preprocessing import encode_and_save_filtered, check_call_rate


MASK_PATH        = "/home/sashreekkumar/Documents/Projects/malariagen/cache/maf_mask.npy"
SVD_PATH         = "/home/sashreekkumar/Documents/Projects/malariagen/cache/svd_model.pkl"
XGB_MODEL_PATH   = "/home/sashreekkumar/Documents/Projects/malariagen/model.ubj"
XGB_ENCODER_PATH = "/home/sashreekkumar/Documents/Projects/malariagen/label_encoder.pkl"
TEST_FOLDER      = "/home/sashreekkumar/Documents/Projects/malariagen/extracted_test"
TEST_CSV         = "/home/sashreekkumar/Documents/Projects/malariagen/data/data_csv/test.csv"
TEST_ENCODED_DIR = "/home/sashreekkumar/Documents/Projects/malariagen/cache/test_encoded/"


if __name__ == "__main__":
    os.makedirs(TEST_ENCODED_DIR, exist_ok=True)

    # step 1: load artifacts
    print("Loading artifacts...")
    maf_mask = np.load(MASK_PATH)
    svd      = joblib.load(SVD_PATH)
    booster  = xgb.Booster()
    booster.load_model(XGB_MODEL_PATH)
    with open(XGB_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    print(f"Classes: {le.classes_}")

    # step 2: encode + filter test samples
    print("\nEncoding test samples...")
    df         = pd.read_csv(TEST_CSV, header=0)
    sample_ids = df.iloc[:, 0].tolist()

    for sid in sample_ids:
        sample_path = Path(TEST_FOLDER) / str(sid)
        out_path    = os.path.join(TEST_ENCODED_DIR, f"{sid}.dat")
        if not sample_path.exists():
            print(f"[SKIP] {sid}: folder not found")
            continue
        if os.path.exists(out_path):
            print(f"[SKIP] {sid}: already encoded")
            continue
        encode_and_save_filtered(str(sample_path), TEST_ENCODED_DIR, sid, maf_mask)
        print(f"[ENCODE] {sid}")

    # step 3: sparse transform — no call rate rejection during inference
    print("\nTransforming test samples...")
    rows        = []
    valid_ids   = []
    y_true      = []
    id_to_label = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 2]))

    for sid in sample_ids:
        out_path = os.path.join(TEST_ENCODED_DIR, f"{sid}.dat")
        if not os.path.exists(out_path):
            continue
        mm       = np.memmap(out_path, dtype=np.int8, mode='r')
        _, rate  = check_call_rate(mm, threshold=0.10)
        if rate > 0.10:
            print(f"[WARN] {sid}: high missing rate {rate:.2%} — predicting anyway")
        rows.append(csr_matrix(mm.astype(np.float32)))
        valid_ids.append(str(sid))
        y_true.append(id_to_label[str(sid)])
        del mm
        print(f"[SPARSE] {sid}")

    # step 4: PCA transform
    print("\nApplying PCA transform...")
    X_sparse = vstack(rows)
    X_pca    = svd.transform(X_sparse)
    print(f"Test PCA shape: {X_pca.shape}")

    # step 5: predict
    print("\nPredicting...")
    dmat       = xgb.DMatrix(X_pca)
    preds      = booster.predict(dmat).astype(int)
    y_pred     = le.inverse_transform(preds)
    y_true_enc = le.transform(y_true)

    # step 6: per sample results
    print("\n── Per Sample Results ───────────────────")
    for sid, true, pred in zip(valid_ids, y_true, y_pred):
        match = "✓" if true == pred else "✗"
        print(f"  {match} ID {sid}: true={true:12s} predicted={pred}")

    # step 7: eval metrics
    print("\n── Evaluation Metrics ───────────────────")
    print(f"Accuracy:             {(y_true_enc == preds).mean():.2%}")
    print(f"Balanced Accuracy:    {balanced_accuracy_score(y_true_enc, preds):.2%}")
    print(f"Matthews CC:          {matthews_corrcoef(y_true_enc, preds):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true_enc, preds, target_names=le.classes_))

    print("Confusion Matrix:")
    cm     = confusion_matrix(y_true_enc, preds)
    header = f"{'':15s}" + "".join(f"{c:12s}" for c in le.classes_)
    print(header)
    for i, row in enumerate(cm):
        print(f"{le.classes_[i]:15s}" + "".join(f"{v:12d}" for v in row))