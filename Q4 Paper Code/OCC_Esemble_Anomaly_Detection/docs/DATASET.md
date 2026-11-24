# Dataset: the recorded 2019 SKN archive

## Obtain the input

Use the owner's original [`2019.zip` Drive file](https://drive.google.com/file/d/1OG_aty11RvXqJ8G3JuXn6sv2QWq_6L-h/view). Access may require permission. The repository does not redistribute the archive, establish a public dataset license, or guarantee that the owner's link remains available. Place a local copy in `data/2019.zip`; that directory is ignored by Git.

| Identity | Recorded value |
|---|---|
| Filename | `2019.zip` |
| Bytes | 122,198,332 |
| SHA-256 | `e7b2e4882ac7c856d438b4e2210993c1a643af52abb6f59ca19c79afe175504e` |
| CSV members | 221, excluding macOS metadata |
| Station header | `GNSS scintillation data from SKN (69.43, 20.38)` |
| Actual observation dates | 2019-01-01 through 2020-01-01 |

The portable runner checks this hash. Renaming/recompressing the input changes its byte identity; use the original ZIP to reproduce the recorded run. `--allow-different-data` explicitly permits a new experiment, but it cannot reproduce this baseline by definition.

## CSV layout

The loader reads the first 19 columns, skips two introductory lines, and uses the third line as the header. It trims whitespace, sorts files by member name, ignores `__MACOSX` and `._` entries, validates shared columns, and stably sorts observations by date, seconds since midnight, and PRN.

```text
YYYY, MM, DD, Second, Total satellite, PRN, Azimuth, Elevation,
L1 CNo, Total S4, Correction to total S4, 60-second phase sigma,
Average of Carrier divergence, Sigma of Carrier Divergence,
TEC at TOW, L1 Lock time, Channel status, L2 Lock Time, L2 CNo
```

`Second` is seconds since midnight; the preparation treats intervals as minutely. Clock timezone is not independently confirmed. Do not label the plots as UTC or local time without additional provenance.

## Preserved cleaning and reference

- Quality filters: `L1 Lock time > 60` and `Elevation > 25`.
- Corrected S4: `sqrt(max(0, Total S4**2 - Correction to total S4**2))`.
- Invalid dates, geometry, PRNs, or S4 measurements are excluded according to the original preparation.
- Missing optional TEC is retained for training-only imputation; S4 is not artificially capped at 1.
- Duplicates and continuity gaps are reported; no new removal/interpolation rule was added.
- Reference: `LABEL = (Corrected_S4 >= 0.20).astype(int)`.

The cleaned dataset contains **1,783,960 rows**, with 584 positives and 1,783,376 negatives. The original raw count is **2,733,026**. All first 12 preparation/EDA cells remain unchanged relative to the initial notebook.

## Predictors and interpretation

The nine candidate predictors are `L1 CNo`, `L2 CNo`, `Azimuth_sin`, `Azimuth_cos`, `Elevation`, `Average of Carrier divergence`, `Corrected_S4`, `TEC at TOW`, and `Sigma of Carrier Divergence`.

`LABEL` is never a predictor. `Corrected_S4` remains mandatory in every representation by the research owner's explicit choice. However, it directly determines the reference; excluding the `LABEL` column does not eliminate this dependency. Treat performance as S4-reference agreement, not independent physical-event validation or forecasting.

## Chronological partitions

| Existing partition | Rows | Reference positives |
|---|---:|---:|
| Training | 1,246,808 | 381 |
| Validation | 264,410 | 84 |
| Final test | 272,742 | 119 |

The modelling stage pools training and validation as development data, makes five expanding date-grouped folds, and leaves the existing final period for the frozen evaluation. See [exact fold boundaries](../results/2019/tables/time_folds.csv). Five-fold out-of-fold evaluation covers 1,227,994 observations, including 368 positives; the initial training-only dates have no out-of-fold predictions.

The final period is 2019-11-29 through 2020-01-01. It had already been evaluated in the earlier baseline experiment. New independent data would be needed to support a strictly prospective generalisation claim.
