import subprocess
import itertools
import re
import csv
import pandas as pd

# Phase definitions search space
phase1_local_windows = [4,5,6,7,8,9,10]
phase1_strides       = [1,2,3,4,5,6,7,8,9,10]

# start_combo = (5, 1)
# resuming = False

# regex patterns for all four metrics
accuracy_pattern  = re.compile(r"Accuracy\s*:\s*([0-9]+(?:\.[0-9]+)?)")
precision_pattern = re.compile(r"Precision\s*:\s*([0-9]+(?:\.[0-9]+)?)")
recall_pattern    = re.compile(r"Recall\s*:\s*([0-9]+(?:\.[0-9]+)?)")
f1_pattern        = re.compile(r"F-score\s*:\s*([0-9]+(?:\.[0-9]+)?)")

def parse_metric(pattern, text, name):
    m = pattern.search(text)
    if not m:
        raise ValueError(f"Could not parse {name} from:\n{text}")
    return float(m.group(1))

with open("results/grid-search/SWAT_grid_search_results_8_plain.csv", "w", newline="") as f:
    fieldnames = ["local_window", "stride", "accuracy", "precision", "recall", "f1"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for lw, st in itertools.product(phase1_local_windows, phase1_strides):
        # only start testing once we've reached (7,0)
        # if not resuming:
        #     if (lw, st) == start_combo:
        #         resuming = True
        #     else:
        #         continue
        # skip the equal‐parameter cases
        if lw == st:
            continue
        print(f"→ Testing local_window={lw}, stride={st}")

        base_cmd = [
            "python", "main.py",
            "--anormly_ratio", "0.5",
            "--num_epochs", "1",
            "--batch_size", "256",
            "--dataset", "SWAT",
            "--data_path", "dataset/SWAT",
            "--input_c", "51",
            "--output_c", "51",
            "--local_window", str(lw),
            "--stride", str(st),
        ]

        # train
        train_cmd = base_cmd + ["--mode", "train"]
        subprocess.run(train_cmd, check=True)

        # test & capture
        test_cmd = base_cmd + ["--mode", "test"]
        output = subprocess.check_output(test_cmd, universal_newlines=True)

        # parse all metrics
        accuracy  = parse_metric(accuracy_pattern,  output, "Accuracy")
        precision = parse_metric(precision_pattern, output, "Precision")
        recall    = parse_metric(recall_pattern,    output, "Recall")
        f1        = parse_metric(f1_pattern,        output, "F-score")

        writer.writerow({
            "local_window": lw,
            "stride":       st,
            "accuracy":     accuracy,
            "precision":    precision,
            "recall":       recall,
            "f1":           f1
        })

# summarize best by F1
df = pd.read_csv("results/grid-search/SWAT_grid_search_results_8_plain.csv")
best = df.loc[df.f1.idxmax()]
print("\n=== Best combination by F1 ===")
print(
    f"local_window={best.local_window}, stride={best.stride}, "
    f"Accuracy={best.accuracy:.4f}, Precision={best.precision:.4f}, "
    f"Recall={best.recall:.4f}, F1={best.f1:.4f}"
)
