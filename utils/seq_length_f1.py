# grid_search.py
import subprocess
import itertools
import re
import csv
import pandas as pd

# Search space
# local_windows = [1,3,5,7,9,11]
# strides       = [7]

# Phase definitions search space
# phase1_local_windows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11, 12, 13,14,15,16,17,18,19,20]
# phase1_strides       = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11, 12, 13,14,15,16,17,18,19,20]
#
# # phase2_local_windows = [5]
# # phase2_strides       = [1, 3, 5, 7, 9, 11]
#
# # Updated regex to match "F‑score : 0.9161"
f1_pattern = re.compile(r"F-score\s*:\s*([0-9]+(?:\.[0-9]+)?)")
#
# skip_combinations = set()
# # 1) skip all (2,0) through (2,11)
# skip_combinations.update({(2, st) for st in range(14)})
# skip_combinations.update({(1, st) for st in range(21)})
# skip_combinations.update({(0, st) for st in range(21)})


lw = 2
st = 10
win = [150, 200, 250, 300, 350, 400]
def parse_f1(text):
    m = f1_pattern.search(text)
    if not m:
        raise ValueError("Could not parse F‑score from:\n" + text)
    return float(m.group(1))

with open("results/grid-search/seq_len_f1_sparse_psm.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["local_window","stride","win","f1"])
    writer.writeheader()

    for w in win:
        print(f"→ Testing local_window={lw}, stride={st}, window={w}")


        # train
        train_cmd = [
            "python", "main.py",
            "--anormly_ratio", "1",
            "--num_epochs", "1",
            "--batch_size", "32",
            "--mode", "train",
            "--dataset", "PSM",
            "--data_path", "dataset/PSM",
            "--input_c", "25",
            "--output_c", "25",
            "--local_window", str(lw),
            "--stride", str(st),
            "--win_size", str(w),
        ]
        subprocess.run(train_cmd, check=True)

        # test & capture
        test_cmd = train_cmd.copy()
        test_cmd[test_cmd.index("--mode")+1] = "test"
        output = subprocess.check_output(test_cmd, universal_newlines=True)

        # parse F‑score and record
        f1 = parse_f1(output)
        writer.writerow({"local_window": lw, "stride": st, "win": w, "f1": f1})

# summarize best
df = pd.read_csv("results/grid-search/seq_len_f1_sparse_psm.csv")
best = df.loc[df.f1.idxmax()]
print("\n=== Best combination ===")
print(f"local_window={best.local_window}, stride={best.stride}, F1={best.f1:.4f}")
