import numpy as np
from data_preprocessing import check_call_rate

ENCODED_DIR = "/home/sashreekkumar/Documents/Projects/malariagen/cache/encoded_filtered/"

missing_ids = [2116, 2766, 160, 2215, 1945, 2395, 2467, 2331, 1925, 2385, 2319, 2276, 2520]

for sid in missing_ids:
    path = f"{ENCODED_DIR}/{sid}.dat"
    import os
    if not os.path.exists(path):
        print(f"{sid}: NO .dat FILE")
        continue
    mm = np.memmap(path, dtype=np.int8, mode='r')
    flag, rate = check_call_rate(mm, threshold=0.10)
    print(f"{sid}: missing_rate={rate:.2%}, rejected={flag}")
    del mm