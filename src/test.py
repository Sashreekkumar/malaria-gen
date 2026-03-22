import zarr
import numpy as np
import pandas as pd
import torch
import pickle
import joblib
import os
from pathlib import Path
from scipy.sparse import csr_matrix, vstack
from data_preprocessing import encode_012, check_call_rate, encode_and_save_filtered
from cnn import SNP_CNN  # wherever you defined the model class
from sklearn.metrics import classification_report, confusion_matrix


# ── paths ─────────────────────────────────────────────────────────────────────
MASK_PATH        = "/home/sashreekkumar/Documents/Projects/malariagen/cache/maf_mask.npy"
SVD_PATH         = "/home/sashreekkumar/Documents/Projects/malariagen/cache/svd_model.pkl"
CNN_MODEL_PATH   = "/home/sashreekkumar/Documents/Projects/malariagen/cnn_model.pt"
CNN_ENCODER_PATH = "/home/sashreekkumar/Documents/Projects/malariagen/cnn_label_encoder.pkl"
TEST_FOLDER      = "/home/sashreekkumar/Documents/Projects/malariagen/extracted_test"
TEST_CSV         = "/home/sashreekkumar/Documents/Projects/malariagen/data/data_csv/test.csv"
TEST_ENCODED_DIR = "/home/sashreekkumar/Documents/Projects/malariagen/cache/test_encoded/"


if __name__ == "__main__":
    os.makedirs(TEST_ENCODED_DIR, exist_ok=True)

    # step 1: load artifacts
    print("Loading artifacts...")
    maf_mask = np.load(MASK_PATH)
    svd      = joblib.load(SVD_PATH)
    with open(CNN_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    model = SNP_CNN(n_components=50, n_classes=len(le.classes_))
    model.load_state_dict(torch.load(CNN_MODEL_PATH))
    model.eval()
    model.to(torch.device("cuda"))
    print(f"Classes: {le.classes_}")

    # step 2: encode + filter test samples
    print("\nEncoding test samples...")
    df         = pd.read_csv(TEST_CSV, header=0)
    sample_ids = df.iloc[:, 0].tolist()

    for sid in sample_ids:
        sample_path = Path(TEST_FOLDER) / str(sid)
        print(f"sample_path: {sample_path}")
        print(f"gt path: {str(sample_path)}/gt")
        print(f"exists: {sample_path.exists()}")
        out_path    = os.path.join(TEST_ENCODED_DIR, f"{sid}.dat")
        if not sample_path.exists():
            print(f"[SKIP] {sid}: folder not found")
            continue
        if os.path.exists(out_path):
            print(f"[SKIP] {sid}: already encoded")
            continue
        encode_and_save_filtered(str(sample_path), TEST_ENCODED_DIR, sid, maf_mask)
        print(f"[ENCODE] {sid}")

    # step 3: call rate check + sparse transform
    print("\nTransforming test samples...")
    rows      = []
    valid_ids = []
    y_true    = []

    id_to_label = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 2]))

    for sid in sample_ids:
        out_path = os.path.join(TEST_ENCODED_DIR, f"{sid}.dat")
        if not os.path.exists(out_path):
            continue
        mm      = np.memmap(out_path, dtype=np.int8, mode='r')
        flag, rate = check_call_rate(mm, threshold=0.10)
        if flag:
            print(f"[REJECT] {sid}: missing rate {rate:.2%}")
            del mm
            continue
        rows.append(csr_matrix(mm.astype(np.float32)))
        valid_ids.append(str(sid))
        y_true.append(id_to_label[str(sid)])
        del mm
        print(f"[SPARSE] {sid}")

    # step 4: PCA transform (use existing SVD, don't refit)
    print("\nApplying PCA transform...")
    X_sparse = vstack(rows)
    X_pca    = svd.transform(X_sparse)  # NOT fit_transform
    print(f"Test PCA shape: {X_pca.shape}")

    # step 5: predict
    print("\nPredicting...")
    device   = torch.device("cuda")
    # no unsqueeze needed, embedding handles it
   # no unsqueeze needed, embedding handles it
    X_tensor = torch.tensor(X_pca, dtype=torch.float32).to(device)  # (5, 50)# (5, 50)

    with torch.no_grad():
        logits = model(X_tensor)
        preds  = logits.argmax(dim=1).cpu().numpy()

    y_pred = le.inverse_transform(preds)

    # step 6: results
    print("\n── Results ──────────────────────────────")
    for sid, true, pred in zip(valid_ids, y_true, y_pred):
        match = "✓" if true == pred else "✗"
        print(f"  {match} ID {sid}: true={true}, predicted={pred}")

    if any(l in le.classes_ for l in y_true):
        y_true_encoded = le.transform(y_true)
        print("\nClassification Report:")
        print(classification_report(y_true_encoded, preds, target_names=le.classes_))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true_encoded, preds))
