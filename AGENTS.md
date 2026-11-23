# Codex agent instructions — Ionosphere Q4 Paper

## Mandatory reading order
At the start of EVERY task, before analysing, proposing, editing, testing or committing:
1. Read this file completely.
2. Read `Q4 Paper Code/CodeRules.md` completely.
3. Read `Q4 Paper Code/ResearchRules.md` completely.
4. Inspect only the relevant existing scripts, configuration, data schema and Git status.
If either rules file is missing or inaccessible, stop and ask the user. Do not silently proceed.

## Scope and authority
- Work within this research project. Preserve existing code and results unless explicitly authorised.
- First diagnose and report concrete issues, their evidence, priority, and proposed minimal fixes.
- **Do not edit, create, delete, rename or move project files until the user approves the specific changes.** Implement only approved items.
- Ask before changing scientific methodology, labels, features, evaluation splits, or reported results, even if a general coding change was approved.
- Do not create unnecessary files or directories. Prefer small changes to existing files.
- Do not automatically implement previously discussed ideas (GA, C4.5, time-series CV) without explicit approval.
- If instructions conflict, stop and ask; follow higher-priority platform and user instructions.

## Git safety and delivery
- Before editing, inspect `git status` and the current branch; never overwrite or stage unrelated user changes.
- Do not switch branches, force push, reset, rebase, merge or delete files without approval.
- After approved changes, run relevant available checks and report exactly what ran and what could not run.
- Show the changed-file list, meaningful diff summary, any scientific implications, and proposed commit message.
- **Do not commit or push until the user explicitly approves.** On approval, stage only relevant files and create a focused, descriptive commit on the current branch. Push only with separate explicit approval.
- Never commit raw research datasets, credentials, tokens, machine-specific paths, large generated artifacts or sensitive files without explicit approval.
- Suggested commit messages: `fix: prevent temporal leakage in validation`, `refactor: simplify feature preprocessing`, `docs: document research validation protocol`.

## Completion report
Summarise: issues found; approved actions taken; files changed; checks and results; limitations; Git status and commit hash (only if actually committed). Never fabricate execution or results.
