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
# seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11, 12, 13,14,15,16,17,18,19,20]
# phase1_strides       = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11, 12, 13,14,15,16,17,18,19,20]

seeds = [2022, 2023, 2024, 2025]
lw = 5
st = 7

# phase2_local_windows = [5]
# phase2_strides       = [1, 3, 5, 7, 9, 11]

# Updated regex to match "F‑score : 0.9161"
f1_pattern = re.compile(r"F-score\s*:\s*([0-9]+(?:\.[0-9]+)?)")

# skip_combinations = set()
# # 1) skip all (2,0) through (2,11)
# skip_combinations.update({(2, st) for st in range(14)})
# skip_combinations.update({(1, st) for st in range(21)})
# skip_combinations.update({(0, st) for st in range(21)})



def parse_f1(text):
    m = f1_pattern.search(text)
    if not m:
        raise ValueError("Could not parse F‑score from:\n" + text)
    return float(m.group(1))

with open("stats.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["seed","f1"])
    writer.writeheader()

    # for lw, st in itertools.product(seeds, phase1_strides):
    for seed in seeds:
        # if (lw, st) in skip_combinations:
        #     continue
        print(f"→ {seed}")


        # train
        train_cmd = [
            "python", "main.py",
            "--anormly_ratio", "1",
            "--seed", str(seed),
            "--num_epochs", "1",
            "--batch_size", "32",
            "--mode", "train",
            "--dataset", "MSL",
            "--data_path", "dataset/MSL",
            "--input_c", "55",
            "--output_c", "55",
            "--local_window", str(lw),
            "--stride", str(st),
        ]
        subprocess.run(train_cmd, check=True)

        # test & capture
        test_cmd = train_cmd.copy()
        test_cmd[test_cmd.index("--mode")+1] = "test"
        output = subprocess.check_output(test_cmd, universal_newlines=True)

        # parse F‑score and record
        f1 = parse_f1(output)
        writer.writerow({f"seed": seed, "f1": f1})

# # summarize best
# df = pd.read_csv("see.csv")
# best = df.loc[df.f1.idxmax()]
# print("\n=== Best combination ===")
# print(f"local_window={best.local_window}, stride={best.stride}, F1={best.f1:.4f}")
