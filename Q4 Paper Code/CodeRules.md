# Code rules — Q4 Paper Code

## Working style
- Inspect the current implementation before proposing any change. Explain the issue and the smallest viable correction, then wait for explicit approval.
- Prefer clarity, simplicity and maintainability over abstractions or cleverness. Follow established Python conventions (PEP 8), meaningful names, small cohesive functions, and straightforward control flow.
- Reuse existing functions, DataFrames, configuration and file structure. Avoid duplicate pipelines, repeated DataFrames, unused imports, dead code and unnecessary dependencies.
- Do not introduce new modules, notebooks, folders, configuration files, or tooling merely for style. Add a file only when it solves a concrete approved need.
- Avoid hardcoded machine-specific paths; preserve Windows compatibility where relevant and prefer `pathlib`.
- Use reproducible random seeds where stochastic methods are used. Record important settings, software versions and experimental parameters using the project's existing mechanisms.
- Keep input data immutable unless an approved transformation explicitly requires otherwise. Validate required columns, timestamps, labels and units before modelling.
- Do not silently change outputs, column semantics, file names, model parameters, thresholds or existing public interfaces.
- Make small, reviewable edits. Avoid unrelated formatting churn and large rewrites.

## Quality gates
- Check syntax/imports and run relevant existing tests or minimal smoke checks when possible; do not run expensive full experiments without user approval.
- Guard against leakage across train/validation/test partitions, including preprocessing, feature selection, resampling, hyperparameter tuning and threshold choice.
- Report exact commands, failures, skipped checks, and environmental limitations. Never describe unexecuted checks as passed.
- Keep generated outputs, caches, environments, credentials and large datasets out of Git unless explicitly authorised. Review `.gitignore` before staging.

## Git discipline
- Diagnose first; edit only after approval; show diff and request separate approval before commit.
- Commit only task-related files with a concise imperative or conventional-commit message that accurately describes the change.
- Do not push without explicit approval. Do not overwrite unrelated work or rewrite history.
