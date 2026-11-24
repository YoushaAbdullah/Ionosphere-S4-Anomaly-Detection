"""Tsai & Lin (2021), IEEE Access, DOI 10.1109/ACCESS.2021.3051969.

This is a documented adaptation, not an exact replication. Sections III-A/B/C
give the model/selector/ensemble settings. The user-approved adaptations are
date-grouped expanding five-fold evaluation, mandatory Corrected_S4, and a
5,000-normal-row fitting cap. Only the modelling cells are replaced.
"""
from pathlib import Path
from itertools import combinations
import hashlib
import importlib.metadata
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import types
import urllib.request
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
    roc_curve, auc,
)
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

SEED = 42
N_TIME_FOLDS = 5
MAX_NORMAL_ROWS = 5_000
NORMALIZATION_NORMAL_ROWS = 2_000
PSEUDO_K = (10, 50, 100)
GA_POPULATION = 10
GA_GENERATIONS = 5
GA_CROSSOVER = 0.90
GA_BIT_MUTATION = 0.01
RETAINED_VARIANCE = 0.80
BATCH_SIZE = 20_000
BOOTSTRAP_REPLICATES = 500
MANDATORY_FEATURE = "Corrected_S4"
ALGORITHMS = ("LOF", "OCSVM", "IFOREST")
VARIANTS = ("BASE", "PCA", "GA", "C45")
OUTPUT_DIR = Path(os.environ.get("Q4_PAPER_OUTPUT", "q4_paper_results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for subfolder in ("figures", "tables", "scores", "selectors", "dependencies"):
    (OUTPUT_DIR / subfolder).mkdir(exist_ok=True)

# Retain the prepared candidate features, including S4. Never use the prior
# globally fitted imputer/scaler in time-series CV: each fold fits fresh ones.
PAPER_FEATURES = list(FEATURES)
if MANDATORY_FEATURE not in PAPER_FEATURES or "LABEL" in PAPER_FEATURES:
    raise ValueError("Corrected_S4 must be a predictor and LABEL must not be.")
if not df_clean.index.is_unique:
    raise ValueError("Unique row indices are required for score alignment.")
if not set(df_clean["LABEL"].unique()).issubset({0, 1}):
    raise ValueError("Expected the existing binary S4 reference labels.")
if not np.array_equal(
    df_clean["LABEL"].to_numpy(),
    (df_clean[MANDATORY_FEATURE].to_numpy() >= S4_THRESHOLD).astype(int),
):
    raise ValueError("Existing S4 measurement and reference labels disagree.")

# Methodological choices not fully specified by the paper are explicit:
# - numeric predictors are median-imputed and standardized on normal fit rows;
# - J48 3.8.6 is used as a genuine C4.5-family implementation (not CART);
# - GA fitness is 3-fold pseudo-class accuracy, then fewer columns as a tie-break;
# - one seeded bootstrap of the normal sample per candidate supplies bagging;
# - a boundary-preserving arctan maps scores monotonically into (0,1);
#   its scale uses separate normal training-period rows, never validation/test;
# - paper threshold 0.5 and CV-calibrated F1/F2/FPR operating points are distinct;
# - supervised C4.5/SMOTE baselines are omitted: the user requires class-0-only fit.
RUN_INFO = {
    "paper_doi": "10.1109/ACCESS.2021.3051969",
    "experimental_focus": "Section III-C / Study III: homogeneous and heterogeneous OCC ensembles",
    "primary_operating_point": "paper 0.5 normality rule; CV-calibrated points are supplementary",
    "protocol": "five expanding date-grouped folds; existing final holdout",
    "max_normal_fit_rows": MAX_NORMAL_ROWS,
    "normalization_normal_rows": NORMALIZATION_NORMAL_ROWS,
    "mandatory_predictor": MANDATORY_FEATURE,
    "features": PAPER_FEATURES,
    "seed": SEED,
    "versions": {package: importlib.metadata.version(package) for package in
                 ("numpy", "pandas", "scikit-learn", "matplotlib", "deap")},
    "reference": "LABEL = 1[Corrected_S4 >= existing S4_THRESHOLD]",
    "interpretation": "agreement with S4-derived reference, not independent event truth",
    "holdout_history": "this period was evaluated in the earlier unchanged run",
    "normalization": "0.5 + arctan((anomaly_score - native_boundary) / normal_IQR) / pi",
    "bagging": "one full-size seeded bootstrap per candidate; a stated assumption",
    "ga_fitness": "mean 3-fold pseudo-class accuracy; fewer features breaks ties",
    "c45": "Weka J48 3.8.6, confidence 0.25, minimum leaf size 2",
    "pca_adaptation": "explicit S4 channel plus 80%-variance PCA of other predictors",
    "not_included": "two-class supervised baselines and SMOTE, per normal-only requirement",
}
SELECTOR_RECORDS = []
GA_HISTORY = []
FIT_RECORDS = []
CV_FOLD_RECORDS = []

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")

def save_records():
    write_json(OUTPUT_DIR / "run_manifest.json", RUN_INFO)
    for name, records in (
        ("feature_selection", SELECTOR_RECORDS), ("ga_history", GA_HISTORY),
        ("model_fits", FIT_RECORDS), ("time_folds", CV_FOLD_RECORDS),
    ):
        pd.DataFrame(records).to_csv(OUTPUT_DIR / "tables" / f"{name}.csv", index=False)

save_records()
print("Paper-based output directory:", OUTPUT_DIR.resolve())



def ensure_j48():
    """Download pinned public Weka jars and compile a small J48 graph adapter."""
    dependency_dir = OUTPUT_DIR / "dependencies"
    sources = {
        "weka-stable-3.8.6.jar": (
            "https://repo.maven.apache.org/maven2/nz/ac/waikato/cms/weka/"
            "weka-stable/3.8.6/weka-stable-3.8.6.jar"
        ),
        "bounce-0.18.jar": (
            "https://repo.maven.apache.org/maven2/nz/ac/waikato/cms/weka/"
            "thirdparty/bounce/0.18/bounce-0.18.jar"
        ),
    }
    if not shutil.which("java"):
        raise RuntimeError("Java JDK 11+ is required for genuine J48/C4.5 selection.")
    hashes = {}
    for filename, url in sources.items():
        target = dependency_dir / filename
        if not target.exists():
            print("Downloading pinned dependency:", filename, flush=True)
            with urllib.request.urlopen(url, timeout=60) as response:
                with target.open("wb") as destination:
                    shutil.copyfileobj(response, destination)
        hashes[filename] = {"url": url, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    source = dependency_dir / "Q4J48.java"
    java_source = '''import java.io.BufferedReader;
import java.io.FileReader;
import weka.core.Instances;
import weka.classifiers.trees.J48;
public class Q4J48 {
  public static void main(String[] args) throws Exception {
    Instances data = new Instances(new BufferedReader(new FileReader(args[0])));
    data.setClassIndex(data.numAttributes()-1);
    J48 tree = new J48();
    tree.setConfidenceFactor(0.25f);
    tree.setMinNumObj(2);
    tree.buildClassifier(data);
    System.out.println(tree.graph());
  }
}
'''
    classpath = os.pathsep.join(str((dependency_dir / name).resolve()) for name in sources)
    if (not source.exists() or source.read_text() != java_source
            or not (dependency_dir / "Q4J48.class").exists()):
        source.write_text(java_source, encoding="utf-8")
        compiler = (["javac"] if shutil.which("javac") else
                    ["java", "-m", "jdk.compiler/com.sun.tools.javac.Main"])
        subprocess.run(compiler + ["-cp", classpath, str(source.resolve())], check=True,
                       capture_output=True, text=True)
    RUN_INFO["weka_dependencies"] = hashes
    save_records()
    return os.pathsep.join([str(dependency_dir.resolve()), classpath])

def c45_columns(X, pseudo_labels, tag, seed, classpath):
    """Section III-B: retain variables used by a J48 tree of normal pseudo-classes."""
    arff = OUTPUT_DIR / "selectors" / f"{tag}.arff"
    with arff.open("w", encoding="utf-8") as stream:
        stream.write("@relation normal_pseudo_classes\n")
        for j in range(X.shape[1]):
            stream.write(f"@attribute x{j} numeric\n")
        classes = ",".join(f"c{j}" for j in np.unique(pseudo_labels))
        stream.write(f"@attribute pseudo_class {{{classes}}}\n@data\n")
        for row, label in zip(X, pseudo_labels):
            stream.write(",".join(format(value, ".17g") for value in row) + f",c{label}\n")
    result = subprocess.run(
        ["java", "-Xmx512m", "-cp", classpath, "Q4J48", str(arff.resolve()), str(seed)],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(f"J48/C4.5 failed: {result.stderr.strip()}")
    graph_start = result.stdout.find("digraph")
    if graph_start < 0:
        raise RuntimeError("J48 did not return a feature-selection graph.")
    graph = result.stdout[graph_start:]
    (OUTPUT_DIR / "selectors" / f"{tag}.dot").write_text(graph, encoding="utf-8")
    selected = {int(value) for value in re.findall(r'label="x(\d+)"', graph)}
    mandatory = PAPER_FEATURES.index(MANDATORY_FEATURE)
    forced = mandatory not in selected
    selected.add(mandatory)
    # Keep the generated graph and selection audit, not a redundant training-data file.
    arff.unlink()
    return sorted(selected), forced

def ga_columns(X, pseudo_labels, tag, seed):
    """Section III-B: DEAP GA on normal pseudo-classes; S4 cannot be switched off."""
    if not hasattr(creator, "Q4Fitness"):
        creator.create("Q4Fitness", base.Fitness, weights=(1.0, -1.0))
        creator.create("Q4FeatureMask", list, fitness=creator.Q4Fitness)
    random.seed(seed)
    mandatory = PAPER_FEATURES.index(MANDATORY_FEATURE)
    toolbox = base.Toolbox()
    cache = {}
    warning_count = 0
    def new_mask():
        mask = creator.Q4FeatureMask(random.randint(0, 1) for _ in range(X.shape[1]))
        mask[mandatory] = 1
        return mask
    def evaluate(mask):
        nonlocal warning_count
        mask[mandatory] = 1
        key = tuple(mask)
        if key not in cache:
            estimator = LogisticRegression(max_iter=300, solver="lbfgs", random_state=seed)
            # Inner pseudo-class CV is a feature-search heuristic, not an anomaly metric.
            inner = KFold(n_splits=3, shuffle=True, random_state=seed)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                accuracy = cross_val_score(estimator, X[:, np.flatnonzero(mask)],
                                           pseudo_labels, cv=inner,
                                           scoring="accuracy", error_score="raise").mean()
            warning_count += len(captured)
            cache[key] = (float(accuracy), int(sum(mask)))
        return cache[key]
    toolbox.register("individual", new_mask)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=GA_BIT_MUTATION)
    toolbox.register("select", tools.selTournament, tournsize=3)
    population = toolbox.population(n=GA_POPULATION)
    best = tools.HallOfFame(1)
    for generation in range(GA_GENERATIONS + 1):
        for individual in population:
            individual[mandatory] = 1
            if not individual.fitness.valid:
                individual.fitness.values = toolbox.evaluate(individual)
        best.update(population)
        GA_HISTORY.append({"selector": tag, "generation": generation,
                           "pseudo_class_accuracy": best[0].fitness.values[0],
                           "selected_count": int(sum(best[0])),
                           "evaluated_masks": len(cache),
                           "fit_warnings": warning_count})
        if generation == GA_GENERATIONS:
            break
        offspring = list(map(toolbox.clone, toolbox.select(population, GA_POPULATION)))
        for left, right in zip(offspring[::2], offspring[1::2]):
            if random.random() < GA_CROSSOVER:
                toolbox.mate(left, right)
                del left.fitness.values
                del right.fitness.values
        for child in offspring:
            toolbox.mutate(child)
            child[mandatory] = 1
            del child.fitness.values
        offspring[0] = toolbox.clone(best[0])  # Explicit elitism assumption.
        population = offspring
    return list(np.flatnonzero(best[0]))

def stratified_date_sample(frame, maximum, seed):
    """Sample class-0 rows proportionally by date, with largest-remainder allocation."""
    if not frame["LABEL"].eq(0).all():
        raise ValueError("Only normal/class-0 observations may enter fitting samples.")
    if len(frame) <= maximum:
        return frame.copy()
    groups = list(frame.groupby("_DATE", sort=True))
    exact = np.array([maximum * len(group) / len(frame) for _, group in groups])
    allocation = np.floor(exact).astype(int)
    for j in np.argsort(-(exact - allocation), kind="stable")[:maximum - allocation.sum()]:
        allocation[j] += 1
    rng = np.random.default_rng(seed)
    parts = [group.sample(n=int(number), random_state=int(rng.integers(2**31 - 1)))
             for (_, group), number in zip(groups, allocation) if number]
    return pd.concat(parts).sort_index()



def make_representations(X, tag, seed, classpath, requested=None):
    """Fit selectors on normal rows; never inspect an evaluation period here."""
    mandatory = PAPER_FEATURES.index(MANDATORY_FEATURE)
    other = [j for j in range(X.shape[1]) if j != mandatory]
    representations = {"BASE": {"columns": list(range(X.shape[1]))}}
    # User-required adaptation: PCA acts on the other predictors, with an
    # explicit S4 channel retained. S4 is standardized, never projected away.
    if requested is None or "PCA" in requested:
        pca = PCA(n_components=RETAINED_VARIANCE, svd_solver="full")
        pca.fit(X[:, other])
        representations["PCA"] = {"pca": pca, "other": other, "mandatory": mandatory}
        loading_table = pd.DataFrame(pca.components_.T,
                                     index=[PAPER_FEATURES[j] for j in other],
                                     columns=[f"PC{j+1}" for j in range(pca.n_components_)])
        loading_table.to_csv(OUTPUT_DIR / "selectors" / f"{tag}_pca_loadings.csv")
        SELECTOR_RECORDS.append({"stage": tag, "representation": "PCA",
                                 "features": "Corrected_S4 + PCA(other predictors)",
                                 "retained_variance": float(pca.explained_variance_ratio_.sum()),
                                 "output_dimensions": 1 + pca.n_components_})
    for clusters in PSEUDO_K:
        methods = [method for method in ("GA", "C45")
                   if requested is None or f"{method}_K{clusters}" in requested]
        if not methods:
            continue
        if len(X) < clusters:
            raise ValueError("Normal sample is smaller than the requested pseudo-class count.")
        kmeans = KMeans(n_clusters=clusters, n_init=10, random_state=seed + clusters)
        pseudo = kmeans.fit_predict(X)
        pd.Series(pseudo).value_counts().sort_index().to_csv(
            OUTPUT_DIR / "selectors" / f"{tag}_k{clusters}_pseudo_sizes.csv")
        for method in methods:
            name = f"{method}_K{clusters}"
            print(f"{tag}: selecting {name} on {len(X):,} normal rows", flush=True)
            started = time.monotonic()
            forced = True
            if method == "GA":
                columns = ga_columns(X, pseudo, f"{tag}_{name}", seed + clusters)
            else:
                columns, forced = c45_columns(X, pseudo, f"{tag}_{name}", seed, classpath)
            if mandatory not in columns:
                raise AssertionError("A selector removed mandatory Corrected_S4.")
            representations[name] = {"columns": columns}
            SELECTOR_RECORDS.append({"stage": tag, "representation": name,
                                     "pseudo_classes": clusters,
                                     "features": json.dumps([PAPER_FEATURES[j] for j in columns]),
                                     "output_dimensions": len(columns),
                                     "s4_forced_when_not_used_by_tree": forced if method == "C45" else None,
                                     "seconds": time.monotonic() - started})
            save_records()
    return representations

def transform_representation(X, representation):
    if "pca" in representation:
        return np.column_stack((X[:, representation["mandatory"]],
                                representation["pca"].transform(X[:, representation["other"]])))
    return X[:, representation["columns"]]

def build_paper_model(algorithm, dimensions, seed):
    # Section III-A: fixed published settings, not the old notebook's grid.
    if algorithm == "LOF":
        return LocalOutlierFactor(n_neighbors=10, metric="euclidean",
                                  novelty=True, n_jobs=-1)
    if algorithm == "OCSVM":
        return OneClassSVM(kernel="rbf", nu=0.5, gamma=1.0 / dimensions)
    if algorithm == "IFOREST":
        return IsolationForest(n_estimators=100, max_samples=100,
                               contamination=0.5, random_state=seed, n_jobs=-1)
    raise ValueError(f"Unknown algorithm: {algorithm}")

def score_batches(model, X, algorithm):
    """Higher scores indicate anomalies; only score unseen observations."""
    scores = np.empty(len(X), dtype=float)
    for start in range(0, len(X), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(X))
        if algorithm == "OCSVM":
            scores[start:stop] = -model.decision_function(X[start:stop]).ravel()
        else:
            scores[start:stop] = -model.score_samples(X[start:stop])
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite anomaly scores encountered.")
    return scores

def normalize_scores(scores, center, scale):
    # The paper does not specify its [0,1] normalization formula. This explicit
    # implementation assumption preserves the native 0.5 decision boundary,
    # rank ordering, and score orientation without test-fitted min/max values.
    return 0.5 + np.arctan((scores - center) / scale) / np.pi

def fit_score_stage(training, evaluation, tag, seed, classpath, requested=None):
    """One fold or the final refit. No anomaly labels enter any model fit."""
    if training["_DATE"].max() >= evaluation["_DATE"].min():
        raise ValueError("Temporal ordering violated.")
    normals = training.loc[training["LABEL"].eq(0)]
    fitted_rows = stratified_date_sample(normals, MAX_NORMAL_ROWS, seed)
    remaining = normals.drop(index=fitted_rows.index)
    if remaining.empty:
        raise ValueError("Need separate normal rows to estimate score normalization.")
    calibration = stratified_date_sample(remaining, NORMALIZATION_NORMAL_ROWS, seed + 1)
    if not fitted_rows["LABEL"].eq(0).all() or not calibration["LABEL"].eq(0).all():
        raise AssertionError("Positive observations entered the one-class fitting path.")
    if set(fitted_rows.index) & set(calibration.index):
        raise AssertionError("LOF normalization sample overlaps fitted samples.")
    pd.DataFrame({"Original_Row_Index": fitted_rows.index}).to_csv(
        OUTPUT_DIR / "selectors" / f"{tag}_normal_fit_indices.csv", index=False)
    pd.DataFrame({"Original_Row_Index": calibration.index}).to_csv(
        OUTPUT_DIR / "selectors" / f"{tag}_normal_calibration_indices.csv", index=False)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(imputer.fit_transform(fitted_rows[PAPER_FEATURES]))
    X_calibration = scaler.transform(imputer.transform(calibration[PAPER_FEATURES]))
    X_evaluation = scaler.transform(imputer.transform(evaluation[PAPER_FEATURES]))
    if not all(np.isfinite(X).all() for X in (X_fit, X_calibration, X_evaluation)):
        raise ValueError("Preprocessing produced non-finite inputs.")
    write_json(OUTPUT_DIR / "selectors" / f"{tag}_preprocessing.json", {
        "imputer_statistics": imputer.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "all_missing_normal_features": fitted_rows[PAPER_FEATURES].isna().all().to_dict(),
    })
    required_representations = (None if requested is None else
                                {name.split("|")[1] for name in requested})
    representations = make_representations(X_fit, tag, seed, classpath, required_representations)
    scores = {}
    for number, (representation_name, representation) in enumerate(representations.items()):
        X_model = transform_representation(X_fit, representation)
        X_normalization = transform_representation(X_calibration, representation)
        X_query = transform_representation(X_evaluation, representation)
        for algorithm_number, algorithm in enumerate(ALGORITHMS):
            candidate = f"{algorithm}|{representation_name}"
            if requested is not None and candidate not in requested:
                continue
            model_seed = seed + 100 * number + algorithm_number
            rng = np.random.default_rng(model_seed)
            bootstrap = rng.integers(0, len(X_model), len(X_model))
            model = build_paper_model(algorithm, X_model.shape[1], model_seed)
            started = time.monotonic()
            print(f"{tag}: fitting {candidate} ({len(X_model):,} normal draws)", flush=True)
            model.fit(X_model[bootstrap])
            fit_seconds = time.monotonic() - started
            normal_scores = score_batches(model, X_normalization, algorithm)
            center = 0.0 if algorithm == "OCSVM" else -float(model.offset_)
            scale = float(np.quantile(normal_scores, .75) - np.quantile(normal_scores, .25))
            if scale <= np.finfo(float).eps:
                scale = max(float(np.std(normal_scores)), 1e-9)
            scores[candidate] = normalize_scores(score_batches(model, X_query, algorithm), center, scale)
            FIT_RECORDS.append({"stage": tag, "model": candidate,
                                "fit_normal_rows": len(X_model),
                                "unique_bootstrap_rows": len(np.unique(bootstrap)),
                                "normalization_rows": len(X_calibration),
                                "features": X_model.shape[1],
                                "parameters": json.dumps(model.get_params(), default=str, sort_keys=True),
                                "normalization_center": center, "normalization_scale": scale,
                                "fit_seconds": fit_seconds,
                                "total_seconds": time.monotonic() - started})
            save_records()
    return pd.DataFrame(scores, index=evaluation.index)



def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan

def count_metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    precision = safe_divide(tp, tp + fp) if tp + fp else 0.0
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "Actual_Anomalies": int(tp + fn), "Detected_Anomalies": int(tp + fp),
        "Precision": precision, "Recall": recall,
        "F1": safe_divide(2 * tp, 2 * tp + fp + fn),
        "F2": safe_divide(5 * tp, 5 * tp + 4 * fn + fp),
        "Specificity": specificity, "FPR": safe_divide(fp, fp + tn),
        "NPV": safe_divide(tn, tn + fn), "Accuracy": safe_divide(tp + tn, total),
        "Balanced_Accuracy": (recall + specificity) / 2 if tp + fn and tn + fp else np.nan,
        "MCC": (tp * tn - fp * fn) / np.sqrt(denominator) if denominator else 0.0,
        "Prevalence": safe_divide(tp + fn, total),
    }

def rank_metrics(y, scores):
    positive = np.count_nonzero(y == 1)
    if positive and positive < len(y):
        precision, recall, _ = precision_recall_curve(y, scores)
        return {"ROC_AUC": float(roc_auc_score(y, scores)),
                "Average_Precision": float(average_precision_score(y, scores)),
                "PR_AUC_Trapezoidal": float(auc(recall, precision))}
    return {"ROC_AUC": np.nan, "Average_Precision": np.nan, "PR_AUC_Trapezoidal": np.nan}

def binary_predictions(scores, threshold, operator):
    return np.asarray(scores) > threshold if operator == ">" else np.asarray(scores) >= threshold

def metrics_at(y, scores, threshold, operator, ranking=None):
    y = np.asarray(y, dtype=int)
    predictions = binary_predictions(scores, threshold, operator)
    tp = int(np.count_nonzero(predictions & (y == 1)))
    fp = int(np.count_nonzero(predictions & (y == 0)))
    fn = int(np.count_nonzero(~predictions & (y == 1)))
    tn = int(np.count_nonzero(~predictions & (y == 0)))
    result = count_metrics(tp, fp, fn, tn)
    result.update(rank_metrics(y, scores) if ranking is None else ranking)
    return result

def operating_points(y, scores):
    """Learn thresholds on out-of-fold development predictions only."""
    result = {"Paper_0.5": {"threshold": .5, "operator": ">"}}
    if len(np.unique(y)) != 2:
        raise ValueError("Pooled time-series validation needs both reference classes.")
    precision, recall, thresholds = precision_recall_curve(y, scores)
    for beta in (1, 2):
        beta2 = beta ** 2
        denominator = beta2 * precision[:-1] + recall[:-1]
        fscore = np.divide((1 + beta2) * precision[:-1] * recall[:-1], denominator,
                           out=np.zeros_like(denominator), where=denominator > 0)
        # Conservative deterministic tie-break: highest threshold at equal F-score.
        best = np.flatnonzero(np.isclose(fscore, fscore.max(), rtol=0, atol=1e-12))[-1]
        result[f"CV_F{beta}"] = {"threshold": float(thresholds[best]), "operator": ">="}
    negative = np.sort(np.asarray(scores)[np.asarray(y) == 0])
    allowed = int(np.floor(.001 * len(negative)))
    boundary = negative[-allowed - 1] if allowed else negative[-1]
    result["CV_FPR_0.1pct"] = {"threshold": float(np.nextafter(boundary, np.inf)), "operator": ">="}
    return result

def cv_fitting_signature():
    """Fingerprint numerical fitting code/settings, independent of source line numbers."""
    def canonical(value):
        if isinstance(value, types.CodeType):
            return {"bytecode": value.co_code.hex(),
                    "exception_table": getattr(value, "co_exceptiontable", b"").hex(),
                    "constants": canonical(value.co_consts),
                    "names": value.co_names, "variables": value.co_varnames,
                    "free": value.co_freevars, "cells": value.co_cellvars,
                    "arguments": (value.co_argcount, value.co_posonlyargcount, value.co_kwonlyargcount)}
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if isinstance(value, (tuple, list)):
            return [canonical(item) for item in value]
        if isinstance(value, frozenset):
            return sorted([canonical(item) for item in value], key=repr)
        return value
    functions = (ensure_j48, c45_columns, ga_columns, stratified_date_sample,
                 make_representations, transform_representation, build_paper_model,
                 score_batches, normalize_scores, fit_score_stage)
    payload = {"functions": {function.__name__: canonical(function.__code__) for function in functions},
               "configuration": [SEED, N_TIME_FOLDS, MAX_NORMAL_ROWS, NORMALIZATION_NORMAL_ROWS,
                                 PSEUDO_K, GA_POPULATION, GA_GENERATIONS, GA_CROSSOVER,
                                 GA_BIT_MUTATION, RETAINED_VARIANCE, BATCH_SIZE, PAPER_FEATURES],
               "versions": RUN_INFO["versions"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def verified_cv_cache(development, dates, cache_root):
    """Reuse completed folds only when input, preparation, code and settings match.

    This recovers expensive numerical fits after a reporting/category correction.
    A normal run has no Q4_CV_CACHE and computes all folds from scratch.
    """
    root = Path(cache_root)
    audit = json.loads((root / "cv_cache_audit.json").read_text())
    expected = {"fitting_signature": cv_fitting_signature(),
                "input_sha256": os.environ.get("Q4_INPUT_SHA256"),
                "preparation_sha256": os.environ.get("Q4_PREPARATION_SHA256")}
    if not all(expected.values()) or any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError("CV cache failed input/preparation/fitting compatibility checks.")
    parts = []
    for fold, (training_days, validation_days) in enumerate(
            TimeSeriesSplit(n_splits=N_TIME_FOLDS).split(dates), 1):
        expected_rows = development.loc[development["_DATE"].isin(dates[validation_days])].index
        path = root / "scores" / f"fold{fold}_scores.npz"
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(saved["row_index"], expected_rows.to_numpy()):
                raise ValueError(f"Cached fold {fold} observations/order do not match this dataset.")
            values = saved["scores"]
            if values.shape != (len(expected_rows), 24) or not np.isfinite(values).all():
                raise ValueError(f"Cached fold {fold} has invalid numeric scores.")
            parts.append(pd.DataFrame(values, index=expected_rows, columns=saved["columns"]))
        shutil.copy2(path, OUTPUT_DIR / "scores" / path.name)
    for filename, records, key in (("time_folds", CV_FOLD_RECORDS, None),
                                    ("model_fits", FIT_RECORDS, "stage"),
                                    ("feature_selection", SELECTOR_RECORDS, "stage"),
                                    ("ga_history", GA_HISTORY, "selector")):
        table = pd.read_csv(root / "tables" / f"{filename}.csv")
        if key:
            table = table.loc[table[key].str.startswith("fold")]
        records.extend(table.to_dict("records"))
    if len(CV_FOLD_RECORDS) != N_TIME_FOLDS or len(FIT_RECORDS) != 24 * N_TIME_FOLDS:
        raise ValueError("CV cache does not contain every completed fold/member fit.")
    for path in (root / "selectors").glob("fold*"):
        if path.is_file():
            shutil.copy2(path, OUTPUT_DIR / "selectors" / path.name)
    RUN_INFO["cv_cache_reuse"] = {"root": str(root), "compatibility_audit": audit,
                                  "reason": "correct actual ensemble-diversity categories; numerical fits unchanged"}
    save_records()
    print("Verified and reused all five numerical CV folds; no test metrics used.", flush=True)
    return pd.concat(parts)

def run_time_series_cv(classpath):
    # Preserve the existing last-period test split; use train+validation as development.
    development = df_clean.loc[train_df.index.union(val_df.index)].sort_values(
        ["_DATE", "Second", "PRN"])
    dates = np.sort(development["_DATE"].unique())
    if development["_DATE"].max() >= test_df["_DATE"].min():
        raise ValueError("Development overlaps the final test period.")
    if os.environ.get("Q4_CV_CACHE"):
        return development, verified_cv_cache(development, dates, os.environ["Q4_CV_CACHE"])
    parts = []
    for fold, (training_days, validation_days) in enumerate(
            TimeSeriesSplit(n_splits=N_TIME_FOLDS).split(dates), 1):
        training = development.loc[development["_DATE"].isin(dates[training_days])]
        validation = development.loc[development["_DATE"].isin(dates[validation_days])]
        record = {"fold": fold, "train_start": str(training["_DATE"].min()),
                  "train_end": str(training["_DATE"].max()),
                  "validation_start": str(validation["_DATE"].min()),
                  "validation_end": str(validation["_DATE"].max()),
                  "training_rows": len(training),
                  "normal_training_rows": int(training["LABEL"].eq(0).sum()),
                  "validation_rows": len(validation),
                  "validation_anomalies": int(validation["LABEL"].sum())}
        CV_FOLD_RECORDS.append(record)
        save_records()
        print("\nTIME-SERIES FOLD", fold, record, flush=True)
        fold_scores = fit_score_stage(training, validation, f"fold{fold}",
                                     SEED + 1000 * fold, classpath)
        np.savez_compressed(OUTPUT_DIR / "scores" / f"fold{fold}_scores.npz",
                            row_index=fold_scores.index.to_numpy(),
                            columns=fold_scores.columns.to_numpy(dtype=str),
                            scores=fold_scores.to_numpy())
        parts.append(fold_scores)
    oof = pd.concat(parts)
    if not oof.index.is_unique:
        raise ValueError("An observation was evaluated in more than one CV fold.")
    return development, oof



def choose_twelve(oof):
    """Study II: select each GA/C4.5 cluster count on CV ROC-AUC, never test."""
    y = df_clean.loc[oof.index, "LABEL"].to_numpy()
    rows = [{"Candidate": candidate, **rank_metrics(y, oof[candidate].to_numpy())}
            for candidate in oof.columns]
    ranking = pd.DataFrame(rows).sort_values(
        ["ROC_AUC", "Average_Precision", "Candidate"], ascending=[False, False, True])
    ranking.to_csv(OUTPUT_DIR / "tables" / "all_cluster_candidate_cv_ranking.csv", index=False)
    source_map = {}
    for algorithm in ALGORITHMS:
        for variant in VARIANTS:
            alias = f"{algorithm}_{variant}"
            if variant in ("BASE", "PCA"):
                source_map[alias] = f"{algorithm}|{variant}"
            else:
                matching = ranking.loc[ranking["Candidate"].str.startswith(f"{algorithm}|{variant}_K")]
                if matching["ROC_AUC"].isna().all():
                    raise ValueError("Cannot select pseudo-class count without two-class CV references.")
                source_map[alias] = matching.iloc[0]["Candidate"]
    selected = pd.DataFrame({alias: oof[source] for alias, source in source_map.items()},
                            index=oof.index)
    write_json(OUTPUT_DIR / "selected_feature_variants.json", source_map)
    return source_map, selected

def ensemble_specs(selected):
    """Tables 1/2/6: 33 homogeneous and 19 heterogeneous combinations."""
    y = df_clean.loc[selected.index, "LABEL"].to_numpy()
    ordered = sorted(selected.columns, key=lambda name: (
        -rank_metrics(y, selected[name].to_numpy())["ROC_AUC"], name))
    specs = {}
    for algorithm in ALGORITHMS:
        members = [f"{algorithm}_{variant}" for variant in VARIANTS]
        for size in (2, 3, 4):
            for subset in combinations(members, size):
                specs["HOMO_" + "__".join(subset)] = {"family": "Homogeneous", "members": subset}
    for size in (2, 3):
        for subset in combinations([f"{algorithm}_BASE" for algorithm in ALGORITHMS], size):
            specs["HET_BASE_" + "__".join(subset)] = {"family": "Heterogeneous", "members": subset}
    for variant in ("PCA", "GA", "C45"):
        specs[f"HET_SHARED_{variant}"] = {"family": "Heterogeneous",
                                             "members": tuple(f"{algorithm}_{variant}" for algorithm in ALGORITHMS)}
    best_per_algorithm = tuple(next(name for name in ordered if name.startswith(algorithm + "_"))
                               for algorithm in ALGORITHMS)
    specs["HET_BEST_PER_ALGORITHM"] = {"family": "Heterogeneous", "members": best_per_algorithm}
    for size in range(2, 13):
        specs[f"HET_TOP_{size:02d}"] = {"family": "Heterogeneous", "members": tuple(ordered[:size])}
    assert sum(spec["family"] == "Homogeneous" for spec in specs.values()) == 33
    assert sum(spec["family"] == "Heterogeneous" for spec in specs.values()) == 19
    # Supplemental fixed example reported in Section IV-C / Table 7, distinct
    # from selecting the top three for this GNSS dataset on development data.
    specs["PAPER_REPORTED_TRIO"] = {
        "family": "Paper reported example",
        "members": ("LOF_BASE", "LOF_C45", "IFOREST_C45"),
    }
    # The paper places top-N members in its cross-pool ensemble table, but a
    # particular dataset's top members can all be variants of one algorithm.
    # Preserve every Table-2/6 combination while auditing actual diversity.
    # Such a top-N candidate must not win the genuinely heterogeneous category.
    for spec in specs.values():
        spec["design_family"] = spec["family"]
        algorithms = sorted({member.split("_", 1)[0] for member in spec["members"]})
        spec["base_algorithms"] = algorithms
        spec["is_genuinely_heterogeneous"] = len(algorithms) >= 2
        if spec["family"] == "Heterogeneous" and len(algorithms) == 1:
            spec["family"] = "Top-N (single base algorithm)"
    return specs

def combine_scores(selected, specs):
    combined = selected.copy()
    for name, spec in specs.items():
        # Eq. (1): mean normality; anomaly score = 1 - mean(normality).
        # Algebraically this is the equal mean of oriented anomaly scores.
        combined[name] = selected[list(spec["members"])].mean(axis=1)
    return combined

def freeze_development(oof):
    source_map, selected = choose_twelve(oof)
    specs = ensemble_specs(selected)
    all_scores = combine_scores(selected, specs)
    y = df_clean.loc[all_scores.index, "LABEL"].to_numpy()
    thresholds = {}
    rows = []
    for name in all_scores:
        scores = all_scores[name].to_numpy()
        ranking = rank_metrics(y, scores)
        thresholds[name] = operating_points(y, scores)
        for method, setting in thresholds[name].items():
            rows.append({"Model": name, "Family": specs.get(name, {}).get("family", "Single OCC"),
                         "Threshold_Method": method, **setting,
                         **metrics_at(y, scores, setting["threshold"], setting["operator"], ranking)})
    cv_metrics = pd.DataFrame(rows)
    cv_metrics.to_csv(OUTPUT_DIR / "tables" / "time_series_cv_metrics.csv", index=False)
    paper_rows = cv_metrics.loc[cv_metrics["Threshold_Method"].eq("Paper_0.5")]
    paper_champion = paper_rows.sort_values(
        ["ROC_AUC", "Average_Precision", "Model"], ascending=[False, False, True]).iloc[0]["Model"]
    detection_rows = cv_metrics.loc[cv_metrics["Threshold_Method"].eq("CV_F1")]
    detection_champion = detection_rows.sort_values(
        ["F1", "Recall", "Precision", "Model"], ascending=[False, False, False, True]).iloc[0]["Model"]
    family_champions = {}
    for family in ("Single OCC", "Homogeneous", "Heterogeneous"):
        family_champions[family] = paper_rows.loc[paper_rows["Family"].eq(family)].sort_values(
            ["ROC_AUC", "Average_Precision", "Model"], ascending=[False, False, True]).iloc[0]["Model"]
    # These choices are written BEFORE any final-test model scoring/metrics.
    frozen = {"source_map": source_map, "ensembles": specs, "thresholds": thresholds,
              "paper_auc_champion": paper_champion,
              "detection_f1_champion": detection_champion,
              "study_iii_family_champions": family_champions,
              "selection_data": "pooled predictions from five expanding development folds",
              "selection_warning": "CV used for selection; final holdout is the evaluation set"}
    write_json(OUTPUT_DIR / "frozen_selection.json", frozen)
    pd.DataFrame([{"Model": name, "Family": spec["family"],
                   "Design_Family": spec["design_family"],
                   "Base_Algorithms": " + ".join(spec["base_algorithms"]),
                   "Genuinely_Heterogeneous": spec["is_genuinely_heterogeneous"],
                   "Members": " + ".join(spec["members"])} for name, spec in specs.items()]).to_csv(
                       OUTPUT_DIR / "tables" / "ensemble_membership.csv", index=False)
    fold_rows = []
    for fold in CV_FOLD_RECORDS:
        mask = df_clean.loc[all_scores.index, "_DATE"].between(
            pd.Timestamp(fold["validation_start"]), pd.Timestamp(fold["validation_end"]))
        fold_y = y[mask.to_numpy()]
        for name in all_scores:
            scores = all_scores.loc[mask, name].to_numpy()
            setting = thresholds[name]["CV_F1"]
            fold_rows.append({"Fold": fold["fold"], "Model": name,
                              **metrics_at(fold_y, scores, setting["threshold"], setting["operator"])})
    pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "tables" / "per_fold_metrics_at_frozen_cv_f1.csv", index=False)
    print("Frozen paper AUC champion:", paper_champion)
    print("Frozen detection F1 champion:", detection_champion)
    return frozen, cv_metrics



def evaluate_final(development, frozen, classpath):
    # Thresholds, model families, cluster counts and membership are now frozen.
    evaluation = df_clean.loc[test_df.index].sort_values(["_DATE", "Second", "PRN"])
    requested = set(frozen["source_map"].values())
    raw_names = fit_score_stage(development, evaluation, "final", SEED + 6000,
                               classpath, requested=requested)
    selected = pd.DataFrame({alias: raw_names[source]
                             for alias, source in frozen["source_map"].items()}, index=evaluation.index)
    all_scores = combine_scores(selected, frozen["ensembles"])
    y = evaluation["LABEL"].to_numpy()
    rows = []
    for name in all_scores:
        scores = all_scores[name].to_numpy()
        ranking = rank_metrics(y, scores)
        for method, setting in frozen["thresholds"][name].items():
            row = {"Model": name,
                   "Family": frozen["ensembles"].get(name, {}).get("family", "Single OCC"),
                   "Threshold_Method": method, **setting,
                   **metrics_at(y, scores, setting["threshold"], setting["operator"], ranking)}
            row["False_Alarms_Per_Observed_Day"] = row["FP"] / evaluation["_DATE"].nunique()
            rows.append(row)
    final_metrics = pd.DataFrame(rows)
    final_metrics.to_csv(OUTPUT_DIR / "tables" / "final_test_metrics_all_models.csv", index=False)
    # Numeric score archive contains every row and all 65 candidates (12 singles,
    # 33 homogeneous, 19 heterogeneous, and one fixed paper example).
    np.savez_compressed(OUTPUT_DIR / "scores" / "final_test_scores.npz",
                        row_index=all_scores.index.to_numpy(),
                        columns=all_scores.columns.to_numpy(dtype=str),
                        scores=all_scores.to_numpy())
    metadata_columns = ["_DATE", "Second", "PRN", "Corrected_S4", "LABEL", "Elevation"]
    metadata = evaluation[metadata_columns].copy()
    metadata.index.name = "Original_Row_Index"
    metadata.to_csv(OUTPUT_DIR / "scores" / "final_test_observation_metadata.csv")
    # This check is deliberately separate: it is perfect agreement by
    # construction, not an independent anomaly detector or a ranked ML model.
    consistency = (evaluation[MANDATORY_FEATURE].to_numpy() >= S4_THRESHOLD).astype(int)
    write_json(OUTPUT_DIR / "tables" / "s4_reference_consistency_check.json", {
        "reference_positives": int(y.sum()),
        "mismatches": int(np.count_nonzero(consistency != y)),
        "interpretation": "deterministic S4 label consistency, not ML validation",
    })
    chosen = list(dict.fromkeys([*frozen["study_iii_family_champions"].values(),
                                frozen["paper_auc_champion"], frozen["detection_f1_champion"]]))
    pd.DataFrame([{"Champion_Number": number, "Model": name,
                   "Primary_Threshold_Method": "Paper_0.5"}
                  for number, name in enumerate(chosen, 1)]).to_csv(
                      OUTPUT_DIR / "tables" / "champion_lookup.csv", index=False)
    for number, name in enumerate(chosen, 1):
        scores = all_scores[name].to_numpy()
        setting = frozen["thresholds"][name]["Paper_0.5"]
        predictions = binary_predictions(scores, setting["threshold"], setting["operator"])
        calibrated = frozen["thresholds"][name]["CV_F1"]
        cv_predictions = binary_predictions(scores, calibrated["threshold"], calibrated["operator"])
        output = metadata.copy()
        output["Anomaly_Score"] = scores
        output["Anomaly_Prediction"] = predictions.astype(int)
        output["Reference_Outcome"] = np.select(
            [predictions & (y == 1), predictions & (y == 0), ~predictions & (y == 1)],
            ["TP", "FP", "FN"], default="TN")
        output["CV_F1_Anomaly_Prediction"] = cv_predictions.astype(int)
        output["CV_F1_Reference_Outcome"] = np.select(
            [cv_predictions & (y == 1), cv_predictions & (y == 0), ~cv_predictions & (y == 1)],
            ["TP", "FP", "FN"], default="TN")
        output["Timestamp"] = pd.to_datetime(output["_DATE"]) + pd.to_timedelta(output["Second"], unit="s")
        output.to_csv(OUTPUT_DIR / "scores" / f"champion_{number}_observations.csv")
        for outcome in ("TP", "FP", "FN"):
            output.loc[output["Reference_Outcome"].eq(outcome)].to_csv(
                OUTPUT_DIR / "tables" / f"champion_{number}_{outcome}.csv")
        block_bootstrap_intervals(output, name, number)
        candidate_runs(output, number)
    RUN_INFO["final_test"] = {"rows": len(evaluation), "reference_anomalies": int(y.sum()),
                              "start": str(evaluation["_DATE"].min()),
                              "end": str(evaluation["_DATE"].max()), "models": len(all_scores.columns)}
    save_records()
    return evaluation, all_scores, final_metrics

def block_bootstrap_intervals(observations, model_name, champion_number):
    """Supplementary uncertainty: resample whole dates, not dependent rows."""
    daily = pd.crosstab(observations["_DATE"], observations["Reference_Outcome"]).reindex(
        columns=["TP", "FP", "FN", "TN"], fill_value=0)
    values = daily.to_numpy(dtype=np.int64)
    rng = np.random.default_rng(SEED + champion_number)
    samples = []
    for _ in range(BOOTSTRAP_REPLICATES):
        tp, fp, fn, tn = map(int, values[rng.integers(0, len(values), len(values))].sum(axis=0))
        samples.append(count_metrics(tp, fp, fn, tn))
    samples = pd.DataFrame(samples)
    rows = []
    for metric in ("Precision", "Recall", "F1", "F2", "FPR", "Specificity", "Balanced_Accuracy", "MCC"):
        usable = samples[metric].dropna()
        rows.append({"Model": model_name, "Threshold_Method": "Paper_0.5", "Metric": metric,
                     "Lower_95pct": usable.quantile(.025), "Upper_95pct": usable.quantile(.975),
                     "Usable_Resamples": len(usable), "Resampling_Unit": "whole observation date"})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "tables" / f"champion_{champion_number}_block_bootstrap_ci.csv", index=False)

def candidate_runs(observations, champion_number):
    """Consecutive flagged observations are candidate runs, not verified events."""
    flagged = observations.loc[observations["Anomaly_Prediction"].eq(1)].sort_values(["PRN", "Timestamp"]).copy()
    if flagged.empty:
        return
    gaps = flagged.groupby("PRN")["Timestamp"].diff().dt.total_seconds()
    flagged["Run"] = gaps.ne(60).groupby(flagged["PRN"]).cumsum()
    summary = flagged.groupby(["PRN", "Run"]).agg(
        Start=("Timestamp", "min"), End=("Timestamp", "max"),
        Observations=("Timestamp", "size"), Reference_Positive_Observations=("LABEL", "sum"),
        Maximum_S4=("Corrected_S4", "max"))
    summary["Elapsed_Minutes"] = (summary["End"] - summary["Start"]).dt.total_seconds() / 60
    summary.to_csv(OUTPUT_DIR / "tables" / f"champion_{champion_number}_candidate_runs.csv")



def short_name(name, maximum=58):
    return name if len(name) <= maximum else name[:maximum - 3] + "..."

def save_figure(figure, name):
    figure.savefig(OUTPUT_DIR / "figures" / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / "figures" / f"{name}.pdf", bbox_inches="tight")
    plt.show()
    plt.close(figure)

def publication_outputs(evaluation, scores, test_metrics, frozen, cv_metrics):
    chosen = list(dict.fromkeys([
        "LOF_BASE", "OCSVM_BASE", "IFOREST_BASE", "PAPER_REPORTED_TRIO",
        *frozen["study_iii_family_champions"].values(),
        frozen["paper_auc_champion"], frozen["detection_f1_champion"],
    ]))
    y = evaluation["LABEL"].to_numpy()
    paper_table = test_metrics.loc[test_metrics["Threshold_Method"].eq("Paper_0.5")].copy()
    detection_table = test_metrics.loc[test_metrics["Threshold_Method"].eq("CV_F1")].copy()
    paper_table.to_csv(OUTPUT_DIR / "tables" / "paper_threshold_test_comparison.csv", index=False)
    detection_table.to_csv(OUTPUT_DIR / "tables" / "cv_f1_threshold_test_comparison.csv", index=False)
    comparison = paper_table.set_index("Model").loc[chosen].reset_index()
    comparison.to_csv(OUTPUT_DIR / "tables" / "main_q4_comparison.csv", index=False)
    detection_comparison = detection_table.set_index("Model").loc[chosen].reset_index()
    detection_comparison.to_csv(OUTPUT_DIR / "tables" / "supplementary_cv_f1_comparison.csv", index=False)
    family_rows = paper_table.set_index("Model").loc[
        list(frozen["study_iii_family_champions"].values())].reset_index()
    family_rows.to_csv(OUTPUT_DIR / "tables" / "study_iii_single_homogeneous_heterogeneous.csv", index=False)
    for table_name, table in (("main_q4_comparison", comparison),
                               ("paper_threshold_test_comparison", paper_table)):
        # Optional LaTeX rendering is skipped if the user's environment lacks
        # the pandas/Jinja formatting dependency; numerical CSVs are mandatory.
        try:
            (OUTPUT_DIR / "tables" / f"{table_name}.tex").write_text(
                table.to_latex(index=False, float_format="%.4f"), encoding="utf-8")
        except ImportError as error:
            RUN_INFO.setdefault("optional_export_notes", []).append(str(error))

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    for name in chosen:
        values = scores[name].to_numpy()
        fpr, tpr, _ = roc_curve(y, values)
        precision, recall, _ = precision_recall_curve(y, values)
        # Preserve curve geometry. Uniform index thinning can hide nearly all
        # positive-event steps in a very imbalanced dataset. Retain all ROC
        # corners and both sides of every PR recall change. Intermediate
        # points on constant-recall vertical segments are redundant.
        roc_indices = np.arange(len(fpr))
        changes = np.flatnonzero(np.diff(recall) != 0)
        pr_indices = np.unique(np.r_[0, changes, changes + 1, len(recall) - 1])
        axes[0].plot(fpr[roc_indices], tpr[roc_indices], label=short_name(name, 35))
        axes[1].plot(recall[pr_indices], precision[pr_indices], label=short_name(name, 35))
        if name in (frozen["paper_auc_champion"], frozen["detection_f1_champion"]):
            pd.DataFrame({"FPR": fpr, "TPR": tpr}).to_csv(
                OUTPUT_DIR / "tables" / f"curve_roc_{chosen.index(name)}.csv", index=False)
            pd.DataFrame({"Recall": recall, "Precision": precision}).to_csv(
                OUTPUT_DIR / "tables" / f"curve_pr_{chosen.index(name)}.csv", index=False)
    axes[0].plot([0, 1], [0, 1], "--", color="grey", linewidth=.8)
    axes[1].axhline(y.mean(), linestyle="--", color="grey", label="Reference prevalence")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Final-period ROC curves")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Final-period precision–recall curves")
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend(fontsize=7, loc="best")
    figure.suptitle("Agreement with the S4-derived reference; choices frozen on time-series CV")
    figure.tight_layout()
    save_figure(figure, "heldout_roc_pr_curves")

    figure_rows = int(np.ceil(len(chosen) / 3))
    figure, axes = plt.subplots(figure_rows, 3, figsize=(14, 4 * figure_rows), squeeze=False)
    for axis, name in zip(axes.flat, chosen):
        setting = frozen["thresholds"][name]["Paper_0.5"]
        row = metrics_at(y, scores[name], setting["threshold"], setting["operator"], ranking={})
        matrix = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
        axis.imshow(np.log1p(matrix), cmap="Blues")
        for i in range(2):
            for j in range(2):
                axis.text(j, i, f"{matrix[i,j]:,}", ha="center", va="center",
                          color="white" if np.log1p(matrix[i,j]) > .6 * np.log1p(matrix).max() else "black")
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Anomaly"],
                 yticklabels=["Normal", "S4-positive"], xlabel="Prediction", ylabel="Reference",
                 title=short_name(name, 40))
    for axis in list(axes.flat)[len(chosen):]:
        axis.set_visible(False)
    figure.suptitle("Study III test confusion matrices — paper 0.5 rule (colour uses log counts)")
    figure.tight_layout()
    save_figure(figure, "heldout_confusion_matrices")

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(comparison))
    for offset, column, label in ((-.25, "TP", "Detected S4-positive (TP)"),
                                   (0, "FP", "False alarms (FP)"),
                                   (.25, "FN", "Missed S4-positive (FN)")):
        axes[0].bar(x + offset, comparison[column], width=.25, label=label)
    for offset, column in ((-.25, "Precision"), (0, "Recall"), (.25, "F1")):
        axes[1].bar(x + offset, comparison[column], width=.25, label=column)
    for axis in axes:
        axis.set_xticks(x, [short_name(name, 25) for name in comparison["Model"]], rotation=35, ha="right")
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=.2)
    axes[0].set(title="Study III paper threshold: detected, missed and falsely flagged", ylabel="Observations")
    axes[1].set(title="Paper threshold: precision, recall and F1", ylabel="Score", ylim=(0, 1.05))
    figure.tight_layout()
    save_figure(figure, "heldout_detection_counts_and_scores")

    # Order the feature study by model family, never select winners on test rows.
    single_table = detection_table.loc[detection_table["Family"].eq("Single OCC")]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, metric in zip(axes, ["Average_Precision", "F1"]):
        for offset, variant in zip([-.3, -.1, .1, .3], VARIANTS):
            values = [float(single_table.loc[single_table["Model"].eq(f"{algorithm}_{variant}"), metric].iloc[0])
                      for algorithm in ALGORITHMS]
            axis.bar(np.arange(3) + offset, values, width=.2, label=variant)
        axis.set(xticks=np.arange(3), xticklabels=ALGORITHMS,
                 title=f"Feature-selection study: {metric}", ylim=(0, 1.05))
        axis.legend()
        axis.grid(axis="y", alpha=.2)
    figure.tight_layout()
    save_figure(figure, "feature_selection_comparison")

    champion = frozen["detection_f1_champion"]
    setting = frozen["thresholds"][champion]["CV_F1"]
    predictions = binary_predictions(scores[champion], setting["threshold"], setting["operator"])
    diagnostics = evaluation[["_DATE", "PRN", "LABEL", "Corrected_S4", "Second"]].copy()
    diagnostics["TP"] = predictions & y.astype(bool)
    diagnostics["FP"] = predictions & ~y.astype(bool)
    diagnostics["FN"] = ~predictions & y.astype(bool)
    daily = diagnostics.groupby("_DATE")[["TP", "FP", "FN"]].sum()
    daily.to_csv(OUTPUT_DIR / "tables" / "detection_champion_daily_counts.csv")
    diagnostics.groupby("PRN")[["TP", "FP", "FN"]].sum().to_csv(
        OUTPUT_DIR / "tables" / "detection_champion_satellite_counts.csv")
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    timestamps = pd.to_datetime(diagnostics["_DATE"]) + pd.to_timedelta(diagnostics["Second"], unit="s")
    axes[0].scatter(timestamps, diagnostics["Corrected_S4"], s=1, alpha=.25, rasterized=True)
    axes[0].axhline(S4_THRESHOLD, linestyle="--", color="red", label="S4 reference threshold")
    axes[0].set(ylabel="Corrected S4", title="Final-period measurements (all satellites)")
    axes[0].legend()
    for column in ("TP", "FP", "FN"):
        axes[1].plot(daily.index, daily[column], label=column)
    axes[1].set(xlabel="Recorded date", ylabel="Observations per date", title="Detection-champion daily outcomes")
    axes[1].legend()
    figure.tight_layout()
    save_figure(figure, "heldout_s4_and_daily_outcomes")

    # GA fitness is pseudo-class accuracy, not anomaly-detection accuracy.
    if GA_HISTORY:
        history = pd.DataFrame(GA_HISTORY)
        figure, axis = plt.subplots(figsize=(11, 5))
        for name, group in history.groupby("selector"):
            axis.plot(group["generation"], group["pseudo_class_accuracy"], alpha=.45, label=name)
        axis.set(xlabel="GA generation", ylabel="Inner pseudo-class CV accuracy",
                 title="GA search diagnostics — normal-data pseudo-classes")
        axis.grid(alpha=.2)
        figure.tight_layout()
        save_figure(figure, "ga_search_diagnostics")

    fold_table = pd.DataFrame(CV_FOLD_RECORDS)
    figure, axis = plt.subplots(figsize=(12, 5))
    for j, fold in fold_table.iterrows():
        axis.plot([pd.Timestamp(fold.train_start), pd.Timestamp(fold.train_end)], [j, j], linewidth=8, color="steelblue")
        axis.plot([pd.Timestamp(fold.validation_start), pd.Timestamp(fold.validation_end)], [j, j], linewidth=8, color="darkorange")
    axis.set(yticks=np.arange(len(fold_table)), yticklabels=[f"Fold {j}" for j in fold_table.fold],
             xlabel="Recorded date", title="Expanding time-series folds: blue = training, orange = validation")
    figure.tight_layout()
    save_figure(figure, "time_series_fold_diagram")
    print("\nSTUDY III PRIMARY RESULTS — paper 0.5 rule; families chosen on development CV:")
    display(comparison[["Model", "TP", "FP", "FN", "Precision", "Recall", "F1", "F2",
                        "MCC", "ROC_AUC", "Average_Precision"]])
    return comparison

def write_method_description(frozen):
    text = f'''<!doctype html><html><head><meta charset="utf-8"><title>Q4 paper-based OCC experiment</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:35px auto;line-height:1.5}}img{{max-width:100%}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:6px;border:1px solid #bbb}}</style></head><body>
<h1>Q4: Study III — one-class classifier ensembles</h1>
<p>Source: Tsai and Lin (2021), DOI 10.1109/ACCESS.2021.3051969, Sections III–IV.</p>
<p>This experiment is a documented adaptation: five expanding folds grouped by complete dates;
all fits use only class-0 observations; at most 5,000 normal rows are sampled proportionally by date;
Corrected_S4 is mandatory, including an explicit channel alongside PCA of other predictors.
All earlier data preparation and EDA code is preserved.</p>
<p>Published settings: LOF k=10 / Euclidean; OCSVM RBF, nu=0.5, gamma=1/d;
Isolation Forest 100 trees, max_samples=100, contamination=0.5. PCA retains 80% variance;
K-means creates 10/50/100 normal-data pseudo-classes. DEAP uses population 10,
5 generations, crossover 0.9 and per-bit mutation 0.01. GA selection is lexicographic
pseudo-class accuracy then fewer features; pseudo-class CV, elitism, tournament size 3,
and J48 pruning defaults are implementation choices not fully specified in the paper.</p>
<p>J48 is genuine C4.5-family software, not sklearn CART. Two-class C4.5 and SMOTE baselines
are omitted under the normal-only fitting requirement. The 33 homogeneous and 19 heterogeneous
ensembles follow Tables 1/2/6. The fixed reported trio is an extra, explicitly labelled comparison.
Ensembles equally average oriented scores (Eq. 1). Boundary-preserving arctan normalization
and one seeded bootstrap per member are stated choices where the paper lacks reproducible details.</p>
<p>The paper's 0.5 threshold is reported separately from CV F1/F2 and 0.1% FPR operating points.
Top-N denotes the paper's cross-pool design; it does not guarantee that a given dataset's top
members use different base algorithms. Membership is audited, and the genuinely heterogeneous
family comparison excludes combinations containing only one base algorithm.
CV selects cluster counts, ensemble membership and thresholds. No final-test metrics select winners.
The paper-AUC champion is {frozen['paper_auc_champion']}; the detection-F1 champion is
{frozen['detection_f1_champion']}. Five-fold results are development/model-selection estimates;
they are not an additional untouched test. The final period was evaluated in an earlier run.</p>
<p>REFERENCE LIMITATION: LABEL is derived from Corrected_S4, which remains a predictor.
TP therefore means agreement with this reference, not independently confirmed physical events.
The reference threshold rule is a consistency check and would agree perfectly by construction.
The full cleaned 2019 dataset has an imbalance ratio much higher than the paper's maximum 129.44;
its published improvement is not guaranteed here. Candidate consecutive runs are not verified events.</p>
<p>Uncertainty intervals resample whole dates and cover count-derived metrics. ROC/PR metrics use
all observations; plotted PR compression preserves every recall change. PNG/PDF figures,
CSV/LaTeX tables, full numeric scores, selected feature maps, GA histories, seeds, parameters,
normalization constants and source-row indices are retained. Earlier notebook cell outputs are historical;
this result directory and its manifest identify the actual experiment.</p>
'''
    main = pd.read_csv(OUTPUT_DIR / "tables" / "main_q4_comparison.csv")
    text += "<h2>Primary Study III results: paper 0.5 normality rule</h2>" + main.to_html(index=False, float_format=lambda v: f"{v:.4f}")
    supplementary = pd.read_csv(OUTPUT_DIR / "tables" / "supplementary_cv_f1_comparison.csv")
    text += "<h2>Supplementary operating-point adaptation: CV-selected F1 thresholds</h2>" + supplementary.to_html(index=False, float_format=lambda v: f"{v:.4f}")
    for name in ("heldout_detection_counts_and_scores", "heldout_confusion_matrices", "heldout_roc_pr_curves",
                 "feature_selection_comparison", "time_series_fold_diagram", "heldout_s4_and_daily_outcomes"):
        text += f'<h2>{name.replace("_", " ")}</h2><img src="figures/{name}.png">'
    text += "</body></html>"
    (OUTPUT_DIR / "research_report.html").write_text(text, encoding="utf-8")

def run_paper_workflow():
    started = time.monotonic()
    study_steps = [
        (1, "Build prerequisite variants", "III-C paragraph 1; III-A/B", "12 candidates: 3 algorithms x 4 variants"),
        (2, "Create normal-data pseudo-classes", "III-B prerequisite", "K-means K=10/50/100, normal training rows only"),
        (3, "Select GA/C4.5 cluster settings", "III-C paragraph 1", "Highest development CV ROC-AUC; S4 mandatory"),
        (4, "Bootstrap normal fitting rows", "III-C bagging statement", "One seeded full-size bootstrap per member; paper leaves details unspecified"),
        (5, "Construct homogeneous ensembles", "Table 1", "All size-2/3/4 subsets within each algorithm: 33"),
        (6, "Construct heterogeneous ensembles", "Tables 2 and 6", "Base pairs/trio, shared selectors, best-per-family, top-2 through top-12: 19"),
        (7, "Normalize oriented member scores", "III-C before Eq. 1", "Training-period normal calibration; arctan is an explicit implementation choice"),
        (8, "Equal-average soft scores", "Eq. 1", "Mean normality >=0.5: normal; <0.5: anomaly"),
        (9, "Freeze single/homogeneous/heterogeneous choices", "Study III comparison", "Development CV AUC; never select using final-test metrics"),
        (10, "Evaluate and export", "Tables 5, 6, 7; Q4 extensions", "Primary paper-threshold results; supplementary CV thresholds; counts, metrics, figures"),
    ]
    pd.DataFrame(study_steps, columns=["Step", "Action", "Paper_Reference", "Implementation"]).to_csv(
        OUTPUT_DIR / "tables" / "study_iii_method_steps.csv", index=False)
    classpath = ensure_j48()
    development, oof = run_time_series_cv(classpath)
    frozen, cv_metrics = freeze_development(oof)
    evaluation, scores, final_metrics = evaluate_final(development, frozen, classpath)
    comparison = publication_outputs(evaluation, scores, final_metrics, frozen, cv_metrics)
    RUN_INFO.update(status="completed", total_seconds=time.monotonic() - started)
    save_records()
    write_method_description(frozen)
    # Exclude raw datasets and runtime dependency jars from the result bundle.
    import zipfile
    archive_path = OUTPUT_DIR.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in OUTPUT_DIR.rglob("*"):
            if path.is_file() and "dependencies" not in path.relative_to(OUTPUT_DIR).parts:
                archive.write(path, path.relative_to(OUTPUT_DIR))
    print("Result archive:", archive_path.resolve())
    return comparison



paper_results = run_paper_workflow()
