"""Verify archived Study III artifacts and optionally compare a new full run.

Uses only the standard library; it neither fits models nor modifies results.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

BASELINE = Path(__file__).resolve().parent / "results" / "2019"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(root, name):
    with (root / "tables" / f"{name}.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root):
    metrics = rows(root, "final_test_metrics_all_models")
    require(len(metrics) == 260, "Expected 260 evaluations.")
    require(len({row["Model"] for row in metrics}) == 65, "Expected 65 candidates.")
    require(len({(row["Model"], row["Threshold_Method"]) for row in metrics}) == 260,
            "Duplicate evaluation key.")
    for row in metrics:
        tp, fp, fn, tn = (int(row[key]) for key in ("TP", "FP", "FN", "TN"))
        require(min(tp, fp, fn, tn) >= 0, "Negative confusion count.")
        require(tp + fn == 119 and fp + tn == 272623, "Wrong final-period totals.")
        require(math.isclose(float(row["Recall"]), tp / 119, abs_tol=1e-12), "Inconsistent recall.")
        require(math.isclose(float(row["F1"]), 2 * tp / (2 * tp + fp + fn), abs_tol=1e-12),
                "Inconsistent F1.")
    folds = rows(root, "time_folds")
    require(len(folds) == 5, "Expected five folds.")
    for row in folds:
        require(row["train_end"] < row["validation_start"], "Chronological overlap.")
    require(sum(int(row["validation_rows"]) for row in folds) == 1227994, "Wrong OOF row count.")
    for row in rows(root, "model_fits"):
        require(int(row["fit_normal_rows"]) == 5000, "Wrong recorded fit sample cap.")
    for row in rows(root, "feature_selection"):
        require("Corrected_S4" in row["features"], "Mandatory S4 missing.")
    frozen = json.loads((root / "frozen_selection.json").read_text())
    for name, definition in frozen["ensembles"].items():
        algorithms = {member.split("_")[0] for member in definition["members"]}
        require(definition["is_genuinely_heterogeneous"] == (len(algorithms) >= 2),
                f"Incorrect diversity classification: {name}")
        if definition["family"] == "Heterogeneous":
            require(len(algorithms) >= 2, f"Single-algorithm ensemble classified heterogeneous: {name}")
    return metrics, frozen


def compare(root, reference, frozen):
    actual, new_frozen = validate(root)
    key = lambda row: (row["Model"], row["Threshold_Method"])
    actual = {key(row): row for row in actual}
    for expected in reference:
        require(key(expected) in actual, f"Missing evaluation: {key(expected)}")
        found = actual[key(expected)]
        require(found.keys() == expected.keys(), "Metric schema differs.")
        for column, value in expected.items():
            if found[column] == value:
                continue
            try:
                left, right = float(found[column]), float(value)
                equal = (math.isnan(left) and math.isnan(right)) or math.isclose(
                    left, right, rel_tol=1e-9, abs_tol=1e-12)
            except ValueError:
                equal = False
            require(equal, f"Difference: {key(expected)}, {column}: {found[column]} versus {value}")
    # These definitions include chosen Ks, exact ensemble membership, family
    # champions, operating thresholds, and tie-handling policy.
    require(new_frozen == frozen, "Frozen model/selector/threshold choices differ.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", type=Path, help="New study_iii_results directory")
    args = parser.parse_args()
    try:
        checksums = json.loads((BASELINE / "artifact_checksums.json").read_text())
        for name, expected in checksums["sha256"].items():
            require(digest(BASELINE / name) == expected, f"Artifact checksum differs: {name}")
        metrics, frozen = validate(BASELINE)
        audit = json.loads((BASELINE / "postprocessing_audit.json").read_text())
        require(audit["numeric_artifacts_unchanged"], "Numeric postprocessing audit failed.")
        require(digest(BASELINE / "tables" / "final_test_metrics_all_models.csv") ==
                audit["numeric_artifact_sha256"]["final_test_metrics_all_models.csv"],
                "Historical final metric hash differs.")
        print(f"PASS: {len(checksums['sha256'])} artifact hashes; 65 models / 260 evaluations; "
              "confusion totals, five chronological folds, fit cap, S4 and ensemble diversity.")
        if args.compare:
            compare(args.compare, metrics, frozen)
            print("PASS: new run metrics and frozen model/threshold choices match the recorded baseline.")
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
