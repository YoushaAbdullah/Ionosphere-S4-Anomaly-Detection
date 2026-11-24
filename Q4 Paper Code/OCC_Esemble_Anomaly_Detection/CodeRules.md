# Code rules — OCC ensemble anomaly detection

## Scope and authorization

Read root/scoped `AGENTS.md` and `ResearchRules.md` first. Inspect current code and
Git status before proposing changes. Explain the concrete issue and minimal fix;
implement only work covered by the user's request/approval. Ask before scientific
changes not already approved. Do not commit/push without applicable explicit approval.

## Canonical structure and cleanliness — mandatory project rule

- Keep all experiment-specific code, notebook, dependencies, docs, tests, rules,
  citation, local references and curated results in this experiment directory.
- Do not create root-level experiment scripts or duplicate active implementations.
  The notebook owns the scientific pipeline; the runner only adapts execution.
- Use `data/` for local immutable inputs, `runs/<unique-run>/` for generated output,
  `.venv/` for local environments, `references/` for local third-party references,
  and `results/2019/` only for the curated historical benchmark.
- Ignore raw datasets, new runs, environments, compiled code, downloaded jars,
  credentials and unlicensed reference PDFs. Do not stage them incidentally.
- Keep shared repository routing/Git/CI files at root because their tools require it.
- Do not clean up by deleting unknown user files. Move only authorized related
  files, preserve content, and remove only known expendable generated caches.
- A relocation is complete only after runtime paths, documentation links, Colab
  paths, CI working directories, Codex routing and relevant checksums are verified.
- Preserve exact historical numerical snapshots and their hashes. Record a new
  relocation/documentation audit rather than rewriting execution history.

## Implementation

Prefer straightforward PEP 8 Python, `pathlib`, small cohesive functions, explicit
names and existing data structures. Avoid unrelated rewrites or new abstractions.
Resolve source/config paths from `__file__`, not a hardcoded machine directory.
The portable runner's default generated output belongs under this experiment's
`runs/`, irrespective of terminal location. For direct interactive notebook use,
set `Q4_PAPER_OUTPUT` to a fresh directory under this experiment's `runs/` before
section 4; the historical notebook fallback is preserved, not rewritten during moves.
An explicit absolute output path is an intentional user override.
Document where relative input paths resolve. Never overwrite a nonempty run folder.

Keep source notebook outputs separate from newly executed copies. Preserve the
first 12 preparation/EDA cells unless their alteration is approved. Never edit
archived numerical code to masquerade as a newly executed current source version.

Keep seeded stochastic choices, locked dependencies, input/source hashes, model
parameters, row counts, dates and selected features in manifests. Validate schema,
input identity and the full JDK before expensive fitting. Do not silently substitute
an algorithm, change public flags or bypass a failed input/runtime check.

## Quality and delivery

Run checks proportional to the change: syntax/CLI/import and offline artifact
checks, path/link checks for moves, actual-data preflight/preparation when relevant,
and the reduced synthetic smoke test for execution changes. Full experiments need
explicit authorization; do not rerun them merely to tidy files.

Report exact checks and failures/skips. A preparation-only or synthetic run is not
a completed 2019 scientific rerun. Keep generated test output out of Git. Before
staging, inspect `.gitignore`, diff, credentials/large-file risks and unrelated work.
Report changed files, scientific implications, Git status and a proposed descriptive
commit message. Never force-push or rewrite history without approval.
