import subprocess
import itertools
import re
import csv
import pandas as pd

# Search space
# local_windows = [1,3,5,7,9,11]
# strides       = [7]

# Phase definitions search space
phase1_local_windows = [1, 3, 5, 7, 9, 11]
phase1_strides       = [7]

phase2_local_windows = [5]
phase2_strides       = [1, 3, 5, 7, 9, 11]

# Updated regex to match "F‑score : 0.9161"
f1_pattern = re.compile(r"F-score\s*:\s*([0-9]+(?:\.[0-9]+)?)")

def parse_f1(text):
    m = f1_pattern.search(text)
    if not m:
        raise ValueError("Could not parse F‑score from:\n" + text)
    return float(m.group(1))

with open("grid_search_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["local_window","stride","f1"])
    writer.writeheader()

    for lw, st in itertools.product(phase1_local_windows, phase1_strides):
        print(f"→ Testing local_window={lw}, stride={st}")

        # train
        train_cmd = [
            "python", "main.py",
            "--anormly_ratio", "1",
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
        writer.writerow({"local_window": lw, "stride": st, "f1": f1})

    for lw, st in itertools.product(phase2_local_windows, phase2_strides):
        print(f"→ Testing local_window={lw}, stride={st}")

        # train
        train_cmd = [
            "python", "main.py",
            "--anormly_ratio", "1",
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
        writer.writerow({"local_window": lw, "stride": st, "f1": f1})

# summarize best
df = pd.read_csv("msl_parameter_sensitivity_results.csv")
best = df.loc[df.f1.idxmax()]
print("\n=== Best combination ===")
print(f"local_window={best.local_window}, stride={best.stride}, F1={best.f1:.4f}")