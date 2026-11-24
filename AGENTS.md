# Codex project router — Ionosphere S4 research

## Active experiment

The canonical home of the OCC Study III project is:

**`Q4 Paper Code/OCC_Esemble_Anomaly_Detection/`**

For any request about OCC ensembles, the 2019 S4 benchmark, `run_experiment.py`,
its notebook, or its outputs, read in this order before analysing or changing files:

1. This root `AGENTS.md`.
2. `Q4 Paper Code/OCC_Esemble_Anomaly_Detection/PROJECT_CONTEXT.md`, completely.
3. For manuscript planning/writing, literature reviews, research-direction advice,
   or journal submissions, also read that folder's
   `Q4_S4_Paper_Writing_Rules.md` completely. It contains journal requirements,
   checked literature, writing standards, evidence limits and candidate directions.
4. That folder's `README.md`; consult its dataset/reproduction guides as relevant.
5. Relevant source/configuration, current branch, and `git status`.

The project context guide contains the approved methodology, known limitations, commands,
source-of-truth files, and project state. Do not rely on access to an earlier chat.
If an applicable instruction file is missing, report it before changing the project.

## Repository layout and workspace hygiene

- Keep each experiment self-contained under its named directory in `Q4 Paper Code/`.
- Keep its runner, notebook, dependency files, citation, docs, tests, local references,
  and curated results together. Do not scatter experiment files at repository root.
- Root files are limited to this router, the repository README, and shared Git/CI
  configuration. GitHub workflows remain in `.github/workflows/` and explicitly
  select the experiment working directory.
- Put raw input in the experiment's ignored `data/`, fresh outputs in ignored
  `runs/<unique-run>/`, and optional local environments in ignored `.venv/`.
- Keep archived results immutable except explicitly approved documentation/audit
  corrections. Never silently replace prior metrics with new output.
- Do not remove unknown/untracked files, user data, or another experiment to make
  the workspace look clean. Inspect first; relocate only task-related approved files.
- After a move, update runtime paths, imports, docs/Colab links, CI, instruction
  routing, and affected checksum manifests. Verify the documented entry points.
- Keep this root **`AGENTS.md`** as the only Codex instruction entry point.
  Keep experiment-specific context, coding rules and research rules together in
  its `PROJECT_CONTEXT.md`, and link it here. Do not create nested `AGENTS.md`
  files, duplicate `AGENT.md` files or depend on chat history as project memory.

## Authority and Git safety

Diagnose concrete issues and explain the minimal fix. Implement only changes
covered by the user's request/approval. Obtain approval before changing scientific
methodology, labels, mandatory features, partitions, or reported results outside
that scope. Existing authorization applies when it clearly covers the action;
do not ask again for the same approved work.

Inspect branch/status before editing. Preserve unrelated changes. Do not switch
branches, reset, rebase, merge, delete user data, or force-push without approval.
Do not commit or push without explicit authorization covering those actions.
Stage only task-related files; never include credentials or raw data by accident.

Report changed paths, checks actually run, scientific implications, limitations,
Git status, a proposed commit message, and a commit hash only if a commit was made.
