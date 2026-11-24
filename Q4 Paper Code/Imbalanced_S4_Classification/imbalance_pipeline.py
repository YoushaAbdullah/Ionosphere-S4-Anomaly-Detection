"""Supervised S4-reference benchmark; no work occurs on import.

Scientific decisions and their untested/selected status: README.md, decision table.
The same nine OCC inputs include the variable defining LABEL. This deliberately
measures reference reconstruction, not independent scintillation detection.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (auc, average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

FEATURES = ['L1 CNo', 'L2 CNo', 'Azimuth_sin', 'Azimuth_cos', 'Elevation',
            'Average of Carrier divergence', 'Corrected_S4', 'TEC at TOW',
            'Sigma of Carrier Divergence']
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    """Fixed before execution; reasons and trade-offs are in README decisions D1–D12."""
    seed: int = 42
    folds: int = 5
    threads: int = 1
    bootstrap_replicates: int = 1000
    fpr_budget: float = .001


def candidates():
    """Small prespecified search, not settings claimed optimal by the review (D5–D7)."""
    return [
        {'name': 'XGB_sqrt_weight', 'family': 'XGBoost', 'weight_power': .5},
        {'name': 'XGB_full_weight', 'family': 'XGBoost', 'weight_power': 1.},
        {'name': 'BRF_sqrt_features', 'family': 'BalancedRandomForest', 'max_features': 'sqrt'},
        {'name': 'BRF_all_features', 'family': 'BalancedRandomForest', 'max_features': None},
        {'name': 'Easy_stumps', 'family': 'EasyEnsemble', 'depth': 1},
        {'name': 'Easy_depth2', 'family': 'EasyEnsemble', 'depth': 2},
    ]


def build_model(spec, y, config):
    """Both classes enter fitting; balancing occurs only inside training (D3, D5–D7)."""
    from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
    from xgboost import XGBClassifier

    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Each training fold must contain both classes; no silent fallback.')
    if spec['family'] == 'XGBoost':
        # Fold-local count ratio; square-root versus full compensation tests how
        # aggressively rare positives should influence loss, without test tuning.
        ratio = np.count_nonzero(y == 0) / np.count_nonzero(y == 1)
        model = XGBClassifier(
            objective='binary:logistic', tree_method='hist', device='cpu',
            n_estimators=300, learning_rate=.05, max_depth=3, min_child_weight=1,
            reg_lambda=10., reg_alpha=0., max_bin=1024, subsample=1.,
            colsample_bytree=1., scale_pos_weight=ratio ** spec['weight_power'],
            eval_metric='aucpr', random_state=config.seed, n_jobs=config.threads)
    elif spec['family'] == 'BalancedRandomForest':
        model = BalancedRandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=1,
            max_features=spec['max_features'], sampling_strategy='all',
            replacement=True, bootstrap=False, class_weight=None,
            random_state=config.seed, n_jobs=config.threads)
    elif spec['family'] == 'EasyEnsemble':
        weak = DecisionTreeClassifier(max_depth=spec['depth'], random_state=config.seed)
        boosted = AdaBoostClassifier(estimator=weak, n_estimators=50,
                                     learning_rate=1., random_state=config.seed)
        model = EasyEnsembleClassifier(
            n_estimators=10, estimator=boosted, sampling_strategy='auto',
            replacement=False, random_state=config.seed, n_jobs=config.threads)
    else:
        raise ValueError(spec)
    # Tree splits need no standard scaling. Fit the median solely on this
    # training partition, before internal ensemble resampling (D4).
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), model)


def divide(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan


def count_metrics(tp, fp, fn, tn):
    """Same definitions/zero-denominator policy as OCC; Detected_Anomalies=TP+FP."""
    recall, specificity = divide(tp, tp + fn), divide(tn, tn + fp)
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
        'Actual_Anomalies': int(tp + fn), 'Detected_Anomalies': int(tp + fp),
        'Precision': divide(tp, tp + fp) if tp + fp else 0., 'Recall': recall,
        'F1': divide(2 * tp, 2 * tp + fp + fn),
        'F2': divide(5 * tp, 5 * tp + 4 * fn + fp),
        'Specificity': specificity, 'FPR': divide(fp, fp + tn),
        'NPV': divide(tn, tn + fn), 'Accuracy': divide(tp + tn, tp + fp + fn + tn),
        'Balanced_Accuracy': (recall + specificity) / 2 if tp + fn and tn + fp else np.nan,
        'MCC': (tp * tn - fp * fn) / np.sqrt(denominator) if denominator else 0.,
        'Prevalence': divide(tp + fn, tp + fp + fn + tn),
    }


def rank_metrics(y, scores):
    if len(np.unique(y)) != 2:
        return dict.fromkeys(['ROC_AUC', 'Average_Precision', 'PR_AUC_Trapezoidal'], np.nan)
    precision, recall, _ = precision_recall_curve(y, scores)
    return {'ROC_AUC': float(roc_auc_score(y, scores)),
            'Average_Precision': float(average_precision_score(y, scores)),
            'PR_AUC_Trapezoidal': float(auc(recall, precision))}


def confusion(y, predictions):
    positive, predictions = np.asarray(y) == 1, np.asarray(predictions, dtype=bool)
    return (int(np.sum(positive & predictions)), int(np.sum(~positive & predictions)),
            int(np.sum(positive & ~predictions)), int(np.sum(~positive & ~predictions)))


def metrics(y, scores, threshold, observed_days):
    result = count_metrics(*confusion(y, scores >= threshold))
    result.update(rank_metrics(y, scores))
    result['False_Alarms_Per_Observed_Day'] = divide(result['FP'], observed_days)
    return result


def operating_points(y, scores, config):
    """OCC supplementary objectives, same >= rule and conservative ties (D9)."""
    if len(np.unique(y)) != 2:
        raise ValueError('Pooled development validation must contain both classes.')
    result = {'Score_0.5': .5}
    precision, recall, thresholds = precision_recall_curve(y, scores)
    for beta in (1, 2):
        den = beta ** 2 * precision[:-1] + recall[:-1]
        f = np.divide((1 + beta ** 2) * precision[:-1] * recall[:-1], den,
                      out=np.zeros_like(den), where=den > 0)
        best = np.flatnonzero(np.isclose(f, f.max(), atol=1e-12, rtol=0))[-1]
        result[f'CV_F{beta}'] = float(thresholds[best])
    negative = np.sort(scores[np.asarray(y) == 0])
    allowed = int(np.floor(config.fpr_budget * len(negative)))
    boundary = negative[-allowed - 1] if allowed else negative[-1]
    result['CV_FPR_0.1pct'] = float(np.nextafter(boundary, np.inf))
    return result


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')


def row_hash(frame):
    """Save ordered observation identities and reference for future paired comparisons."""
    values = frame[['_DATE', 'Second', 'PRN', 'Corrected_S4', 'LABEL']].copy()
    values.insert(0, 'Original_Row_Index', frame.index)
    return hashlib.sha256(values.to_csv(index=False).encode()).hexdigest()


def fit_predict(training, evaluation, spec, config, audits, stage):
    if training['_DATE'].max() >= evaluation['_DATE'].min():
        raise ValueError('Future-to-past or overlapping-date training detected.')
    model = build_model(spec, training['LABEL'].to_numpy(), config)
    started = time.monotonic()
    model.fit(training[FEATURES], training['LABEL'].to_numpy())
    # Uniform float64 avoids float32 threshold-casting surprises at tied scores.
    scores = model.predict_proba(evaluation[FEATURES])[:, 1].astype(np.float64)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite predictions.')
    audits.append({'stage': stage, 'candidate': spec, 'training_rows': len(training),
                   'training_positives': int(training['LABEL'].sum()),
                   'train_identity_sha256': row_hash(training),
                   'evaluation_identity_sha256': row_hash(evaluation),
                   'fit_and_score_seconds': time.monotonic() - started,
                   'imputation_medians': model[0].statistics_.tolist(),
                   'parameters': model.get_params(deep=True)})
    print(stage, spec['name'], f'{time.monotonic() - started:.1f}s', flush=True)
    return scores


def validate_partitions(clean, train, val, test, features):
    """Consume unchanged preparation frames, never its globally transformed matrices."""
    if list(features) != FEATURES or not clean.index.is_unique:
        raise ValueError('Features or source row identities differ from OCC.')
    if not np.array_equal(clean['LABEL'], (clean['Corrected_S4'] >= .20).astype(int)):
        raise ValueError('Reference definition changed.')
    if any(set(a.index) & set(b.index) for a, b in [(train, val), (train, test), (val, test)]):
        raise ValueError('Overlapping partitions.')
    development = clean.loc[train.index.union(val.index)].sort_values(
        ['_DATE', 'Second', 'PRN'], kind='stable')
    final = clean.loc[test.index].sort_values(['_DATE', 'Second', 'PRN'], kind='stable')
    if development['_DATE'].max() >= final['_DATE'].min():
        raise ValueError('Development/test dates overlap.')
    if len(development) + len(final) != len(clean):
        raise ValueError('Partitions do not cover the cleaned observations.')
    if len(final) != 272742 or int(final['LABEL'].sum()) != 119:
        raise ValueError('Final-period population differs from the recorded 2019 benchmark.')
    if str(final['_DATE'].min().date()) != '2019-11-29' or str(final['_DATE'].max().date()) != '2020-01-01':
        raise ValueError('Final-period dates differ from OCC.')
    return development, final


def export_predictions(frame, scores, threshold, destination):
    """All four confusion outcomes, not only attractive detections (D10)."""
    records = frame[['_DATE', 'Second', 'PRN', 'Corrected_S4', 'LABEL', 'Elevation']].copy()
    records.index.name = 'Original_Row_Index'
    records['Score'] = scores
    records['Prediction'] = (scores >= threshold).astype(int)
    records['Outcome'] = np.select(
        [(records.LABEL == 1) & (records.Prediction == 1),
         (records.LABEL == 0) & (records.Prediction == 1),
         (records.LABEL == 1) & (records.Prediction == 0)], ['TP', 'FP', 'FN'], default='TN')
    records.to_csv(destination, compression='gzip')
    return records


def group_counts(records, group):
    table = pd.crosstab(records[group], records['Outcome']).reindex(
        columns=['TP', 'FP', 'FN', 'TN'], fill_value=0)
    table['Observations'] = table[['TP', 'FP', 'FN', 'TN']].sum(axis=1)
    table['Reference_Positives'] = table.TP + table.FN
    return table


def bootstrap_dates(records, config, name, method):
    """Exploratory one-observed-date blocks, conditional on frozen fits (D11)."""
    daily = group_counts(records, '_DATE')
    counts = daily[['TP', 'FP', 'FN', 'TN']].to_numpy()
    rng = np.random.default_rng(config.seed)
    rows = []
    for _ in range(config.bootstrap_replicates):
        tp, fp, fn, tn = counts[rng.integers(0, len(counts), len(counts))].sum(axis=0)
        rows.append(count_metrics(int(tp), int(fp), int(fn), int(tn)))
    samples = pd.DataFrame(rows)
    return [{'Model': name, 'Threshold_Method': method, 'Metric': key,
             'Lower_2.5pct': samples[key].quantile(.025),
             'Upper_97.5pct': samples[key].quantile(.975),
             'Finite_Replicates': int(samples[key].notna().sum()),
             'Replicates': config.bootstrap_replicates,
             'Scope': 'one-observed-date blocks; frozen-fit count metrics only'}
            for key in ['Precision', 'Recall', 'F1', 'F2', 'FPR', 'MCC']]


def plot_results(final, scores, table, out):
    """Unsmooth PR/ROC plus frozen-rule confusion counts and daily errors (D12)."""
    import matplotlib.pyplot as plt

    y = final['LABEL'].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, values in scores.items():
        fpr, tpr, _ = roc_curve(y, values)
        precision, recall, _ = precision_recall_curve(y, values)
        axes[0].plot(fpr, tpr, label=name)
        axes[1].step(recall, precision, where='post', label=name)
    axes[0].plot([0, 1], [0, 1], 'k:', label='chance ranking')
    axes[1].axhline(y.mean(), color='k', linestyle=':', label='prevalence')
    axes[0].set(xlabel='False-positive rate', ylabel='True-positive rate', xlim=(0, 1), ylim=(0, 1.01))
    axes[1].set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.01))
    for ax in axes:
        ax.legend(fontsize=7)
    save_figure(fig, out / 'figures' / 'roc_pr')
    selected = table[table.Threshold_Method == 'CV_FPR_0.1pct']
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, (_, row) in zip(axes, selected.iterrows()):
        matrix = np.array([[row.TN, row.FP], [row.FN, row.TP]], dtype=int)
        ax.imshow(np.log1p(matrix), cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='red')
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel='Prediction', ylabel='S4 reference', title=row.Model)
    save_figure(fig, out / 'figures' / 'confusion_cv_fpr')
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax, (_, row) in zip(axes, selected.iterrows()):
        prediction = scores[row.Model] >= row.threshold
        daily = pd.DataFrame({'Date': final['_DATE'].to_numpy(),
                              'TP': prediction & (y == 1), 'FP': prediction & (y == 0),
                              'FN': ~prediction & (y == 1)}).groupby('Date').sum()
        for key, style in [('TP', '-'), ('FP', '--'), ('FN', ':')]:
            ax.plot(daily.index, daily[key], linestyle=style, label=key)
        ax.set_ylabel(f'{row.Model}\nobservations')
        ax.legend(fontsize=7)
    axes[-1].set_xlabel('Dataset date (gaps are unobserved; lines connect observed dates)')
    save_figure(fig, out / 'figures' / 'daily_detection_counts')


def save_figure(fig, path):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    for extension in ['png', 'pdf']:
        fig.savefig(path.with_suffix('.' + extension), dpi=300, bbox_inches='tight')
    plt.close(fig)


def run_workflow(clean, train, val, test, features, config=None):
    """Select on development, freeze on disk, refit, then evaluate historical test.

    Requires an explicit runner --run or notebook opt-in after owner approval.
    Execution permission is not inferred from elapsed time or presence of data.
    """
    if os.environ.get('S4_IMBALANCE_APPROVED') != '1':
        raise RuntimeError('Training is awaiting owner approval. Use --run only after approval.')
    config = config or Config()
    if config.folds != 5 or config.fpr_budget != .001:
        raise ValueError('This benchmark requires five folds and the declared 0.1% comparison budget.')
    out = Path(os.environ['S4_IMBALANCE_OUTPUT']).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output is nonempty; select a new run directory.')
    for folder in ['tables', 'scores', 'detections', 'figures', 'audit']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    development, final = validate_partitions(clean, train, val, test, features)
    dates = np.sort(development['_DATE'].unique())
    specs, folds, parts, fit_audit, fold_metrics = candidates(), [], [], [], []
    planned_folds = list(TimeSeriesSplit(n_splits=config.folds).split(dates))
    for number, (td, vd) in enumerate(planned_folds, 1):
        training = development.loc[development['_DATE'].isin(dates[td])]
        validation = development.loc[development['_DATE'].isin(dates[vd])]
        folds.append({'fold': number, 'train_start': str(training['_DATE'].min()),
                      'train_end': str(training['_DATE'].max()),
                      'validation_start': str(validation['_DATE'].min()),
                      'validation_end': str(validation['_DATE'].max()),
                      'training_rows': len(training),
                      'normal_training_rows': int(training.LABEL.eq(0).sum()),
                      'validation_rows': len(validation),
                      'validation_anomalies': int(validation.LABEL.sum())})
    fold_table = pd.DataFrame(folds)
    fold_table.to_csv(out / 'tables' / 'time_folds.csv', index=False)
    # Check protocol before expensive fits, without changing the OCC archive.
    occ = ROOT.parent / 'OCC_Esemble_Anomaly_Detection' / 'results' / '2019'
    archived_folds = pd.read_csv(occ / 'tables' / 'time_folds.csv')
    if fold_table.astype(str).to_dict('list') != archived_folds.astype(str).to_dict('list'):
        raise ValueError('Folds do not match archived OCC; do not claim a matched comparison.')
    for number, (td, vd) in enumerate(planned_folds, 1):
        training = development.loc[development['_DATE'].isin(dates[td])]
        validation = development.loc[development['_DATE'].isin(dates[vd])]
        scored = pd.DataFrame(index=validation.index)
        for spec in specs:
            values = fit_predict(training, validation, spec, config, fit_audit, f'fold{number}')
            scored[spec['name']] = values
            fold_metrics.append({'Fold': number, 'Model': spec['name'],
                                 **metrics(validation.LABEL.to_numpy(), values, .5,
                                           validation['_DATE'].nunique())})
            write_json(out / 'audit' / 'fits.json', fit_audit)
        scored['Fold'] = number
        parts.append(scored)
    oof = pd.concat(parts)
    if not oof.index.is_unique:
        raise ValueError('Repeated OOF observations.')
    oof.to_csv(out / 'scores' / 'development_oof.csv.gz', compression='gzip', index_label='Original_Row_Index')
    y = clean.loc[oof.index, 'LABEL'].to_numpy()
    ranking = pd.DataFrame([{'Model': spec['name'], 'Family': spec['family'],
                             **rank_metrics(y, oof[spec['name']].to_numpy())} for spec in specs])
    # AP is selected a priori for precision–recall ranking under severe imbalance.
    # Pooled OOF selection is exploratory, not nested unbiased performance (D8).
    ranking = ranking.sort_values(['Average_Precision', 'Model'], ascending=[False, True])
    chosen = ranking.groupby('Family', sort=False).head(1)
    selected = [next(s for s in specs if s['name'] == name) for name in chosen.Model]
    thresholds = {s['name']: operating_points(y, oof[s['name']].to_numpy(), config) for s in selected}
    frozen = {'selected': selected, 'champion': ranking.iloc[0].Model,
              'thresholds': thresholds, 'primary_rule': 'CV_FPR_0.1pct',
              'config': vars(config), 'features': FEATURES, 'training': 'both classes',
              'selection': 'pooled five-fold development AP; lexical candidate ties',
              'interpretation': 'S4-reference reconstruction, not independent physical detection',
              'test_status': 'previously evaluated historical benchmark',
              'development_identity_sha256': row_hash(development),
              'final_identity_sha256': row_hash(final)}
    # Freeze BEFORE the first final fit/score. Never select a winner from final metrics.
    write_json(out / 'frozen_selection.json', frozen)
    ranking.to_csv(out / 'tables' / 'development_candidate_ranking.csv', index=False)
    pd.DataFrame(fold_metrics).to_csv(out / 'tables' / 'per_fold_score_0.5_metrics.csv', index=False)
    cv_rows = []
    for spec in selected:
        name = spec['name']
        for rule, threshold in thresholds[name].items():
            cv_rows.append({'Model': name, 'Family': spec['family'], 'Threshold_Method': rule,
                            'threshold': threshold, 'operator': '>=',
                            **metrics(y, oof[name].to_numpy(), threshold,
                                      clean.loc[oof.index, '_DATE'].nunique())})
    pd.DataFrame(cv_rows).to_csv(out / 'tables' / 'time_series_cv_metrics.csv', index=False)
    final_scores, rows, intervals = {}, [], []
    for spec in selected:
        name = spec['name']
        values = fit_predict(development, final, spec, config, fit_audit, 'final')
        final_scores[name] = values
        for rule, threshold in thresholds[name].items():
            rows.append({'Model': name, 'Family': spec['family'], 'Threshold_Method': rule,
                         'threshold': threshold, 'operator': '>=',
                         **metrics(final.LABEL.to_numpy(), values, threshold, final['_DATE'].nunique())})
            records = export_predictions(final, values, threshold,
                                          out / 'detections' / f'{name}__{rule}.csv.gz')
            for group, label in [('_DATE', 'daily'), ('PRN', 'satellite')]:
                group_counts(records, group).to_csv(out / 'tables' / f'{name}__{rule}__{label}.csv')
            intervals.extend(bootstrap_dates(records, config, name, rule))
        write_json(out / 'audit' / 'fits.json', fit_audit)
    final_table = pd.DataFrame(rows)
    final_table.to_csv(out / 'tables' / 'final_test_metrics_all_models.csv', index=False)
    pd.DataFrame(intervals).to_csv(out / 'tables' / 'date_block_count_intervals.csv', index=False)
    final[['_DATE', 'Second', 'PRN', 'Corrected_S4', 'LABEL']].to_csv(
        out / 'scores' / 'final_observation_metadata.csv.gz', compression='gzip', index_label='Original_Row_Index')
    pd.DataFrame(final_scores, index=final.index).to_csv(
        out / 'scores' / 'final_scores.csv.gz', compression='gzip', index_label='Original_Row_Index')
    baselines = []
    for name, values in [('All_normal', np.zeros(len(final))),
                          ('S4_reference_rule', (final.Corrected_S4.to_numpy() >= .20).astype(float))]:
        baselines.append({'Model': name, 'Family': 'Definition/control baseline',
                          'Threshold_Method': 'Baseline', 'threshold': .5, 'operator': '>=',
                          **metrics(final.LABEL.to_numpy(), values, .5, final['_DATE'].nunique())})
    baseline_table = pd.DataFrame(baselines)
    baseline_table.to_csv(out / 'tables' / 'baselines.csv', index=False)
    historical = pd.read_csv(occ / 'tables' / 'final_test_metrics_all_models.csv')
    historical['Training_Regime'] = 'normal-only fit; development label selection; 5000-row cap'
    historical['Experiment'] = 'archived OCC Study III'
    comparison = final_table.assign(Training_Regime='both-class fit; algorithm-specific weighting/resampling',
                                    Experiment='supervised reference reconstruction')
    combined = pd.concat([historical, comparison, baseline_table.assign(
        Training_Regime='no fitting', Experiment='definition/control')], ignore_index=True)
    combined['Claim_Limit'] = 'matched population/features; unequal label supervision and fit budgets'
    combined.to_csv(out / 'tables' / 'occ_comparison_all_operating_points.csv', index=False)
    final_table.to_latex(out / 'tables' / 'final_metrics.tex', index=False, float_format='%.6g')
    plot_results(final, final_scores, final_table, out)
    write_json(out / 'audit' / 'comparison.json', {
        'occ_results': str(occ), 'identical_fold_table': True,
        'archive_sha256': os.environ.get('Q4_INPUT_SHA256'),
        'preparation_sha256': os.environ.get('Q4_PREPARATION_SHA256'),
        'occ_metrics_sha256': hashlib.sha256((occ / 'tables' / 'final_test_metrics_all_models.csv').read_bytes()).hexdigest(),
        'reference_baseline_mismatches': int(np.count_nonzero(
            final.LABEL.to_numpy() != (final.Corrected_S4.to_numpy() >= .20))),
        'warning': 'No independent event truth; no causal or algorithm-only superiority claim.'})
    (out / 'README.md').write_text(
        '# Executed historical benchmark\n\n'
        'See frozen_selection.json for development-only choices; all prespecified final rules are reported.\n'
        'Metrics measure S4-reference agreement. Both-class fitting differs from normal-only OCC.\n'
        'Intervals resample individual observed dates conditional on fixed fits; no ROC/AP, selection, '
        'seed or between-date-dependence uncertainty is covered.\n', encoding='utf-8')
    print(final_table.to_string(index=False))
    return final_table
