import zarr
import numpy as np

# Loading Data
def load_zarr_to_numpy(zarr_path: str, key: str = None) -> np.ndarray:
    store = zarr.open(zarr_path, mode='r')
    
    if key is not None:
        data = store[key]
    else:
        data = store
    
    return np.array(data)


# Applying 0/1/2 Encoding
def encode_012(arr: np.ndarray, ref_allele_val: int = 0) -> np.ndarray:
    arr = np.asarray(arr)

    missing_mask = np.any(arr == -1, axis=-1)
    encoded = np.sum(arr != ref_allele_val, axis=-1).astype(np.int8)
    encoded[missing_mask] = -1

    return encoded

# Check Call Rate
'''
Call Rate = 95%
Missing Rate =5%
'''
def check_call_rate(arr: np.ndarray, threshold: float = 0.05) -> tuple[bool, float]:
    missing_rate = np.mean(arr == -1)
    flag = missing_rate > threshold
    return flag, missing_rate

gt_raw = load_zarr_to_numpy("extracted/54/gt")
encoded = encode_012(gt_raw)
flag, missing_rate = check_call_rate(encoded)

if flag:
    print(f"Warning: {missing_rate:} missing values exceeds 5% threshold")

'''
OUTPUT: Warning: 0.05647189727960293 missing values exceeds 5% threshold
'''