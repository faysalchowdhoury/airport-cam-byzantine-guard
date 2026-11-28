# Work history and resume guide

Read this file first when resuming work on this repository. It records how the project was built, its current
state, the decisions behind it, and what is still open.

Last updated: 2026-11-28. Last results commit: `ed25772`.

## 1. Origin

The project was started from a single agent prompt. Both versions are saved verbatim:

- `docs/prompts/original_prompt_v1.md`: the first draft, with a different subtitle and a looser spec.
- `docs/prompts/original_prompt_v2.md`: **the governing spec.** Everything was built against this version.

Rules from the prompt that still apply:

- **Data:** real data only, with no synthetic frames or ADS-B. Every download is SHA-256 logged in
  `artifacts/data_provenance.json`. Gate 0 fails closed with `NOT_EVALUABLE`.
- **Seeds and methods:** seeds `[42, 101, 202, 303, 404]`; 4 methods; `k/n` in [0.1, 0.45].
- **Figures:** 5 figures at 400 DPI, white background, Times New Roman, Okabe-Ito/Viridis, no overlapping labels.
- **Paper:** IEEEtran; tables come from `artifacts/results_summary.json`.
- **Git identity:** `git config user.name "S M Tanvir Faysal Alam Chowdhoury"` and
  `git config user.email "faysalchowdhoury@users.noreply.github.com"`.
- **Commits:** no Claude/Anthropic co-author trailers in any commit message. faysalchowdhoury is the sole
  contributor.

Paper author block, requested later by the owner:

> Faysal A. Chowdhoury^1,\* and Lu Sun^2. Both are at Rochester Institute of Technology, as given in
> `paper/main.tex`. Correspondence: smcite@rit.edu.

## 2. Timeline

1. **Commit `59cfb7b` (WIP):** pipeline, theory, ST-GNN attestor, MILP hand-off, paper draft. The first
   session ran out of context during the WILDTRACK download, which crashed with a `MemoryError`.
2. **Resumed session:**
   - **Downloader fix:** `data_pipeline/wildtrack.py` now streams frames to `.npy` memmaps with resumable
     checkpoints and 2 workers. All 2,800 frames (7 cameras × 400) were downloaded.
   - **Gate 0:** passed all checks.
   - **Runs:** the theory check, phase boundary, five-seed benchmark, latency, and ATL case study all ran.
   - **Figure fixes:** Fig 1 labels and arrows, the Fig 3 symlog floor, and the Fig 4 and 5 label positions.
   - **Paper:** wrote `paper/results.tex`, with every number coming from `generated/macros.tex`.
   - **LaTeX fixes:** load order `amsmath,amsthm` before `newtx`; a custom `\makecell`, because MiKTeX hung
     installing the package; `\inputtable` built on the primitive `\@@input`, because `\input` breaks
     `\bottomrule`.
   - **Claim fixes:** corrected over-claims against the actual numbers.
3. **Commit `6777142`:** full results, figures, results section, compiled paper.
4. **Commit `ed25772`:** the author block changed to Chowdhoury & Sun with RIT affiliations.

## 3. Current results

Mean ± std over 5 seeds, from `artifacts/results_summary.json`:

| Method | Blind min/h | Deceived min/h | False alarms/h | Frame AUROC |
|---|---|---|---|---|
| Single-camera | 3.36 | 1.17 | 238.6 | 0.730 |
| Majority vote | 1.91 | 1.06 | 56.7 | 0.648 |
| Static redundancy | 3.04 | 0.85 | 12.5 | 0.599 |
| **SCSPV attestor + MILP** | **1.79 ± 0.05** | **0.76** | 37.0 ± 28.2 | **0.852** |

Other results:

- **Phase boundary:** the ground plane agrees with the theorem in every cell (1.00). Off-plane agreement is 0.80,
  with the boundary one witness later, as the probabilistic extension predicts.
- **Latency, p95:** the attestor takes 11.99 ms per frame for 7 cameras. The MILP takes 50 ms. Horn–Schunck
  optical flow takes 214 ms per camera, which is outside the 15 ms budget.
- **ATL Concourse F case study:** blind minutes are 4.64 with no hand-off, 2.96 with greedy, and 2.95 with MILP.
- **ST-GNN size:** 17,841 parameters.

Known weaknesses, all stated honestly in the paper's Limitations:

- **Blind time:** only about 6% lower than majority vote. The main gain is in AUROC.
- **False alarms:** high variance across seeds. The selected θ was 0.9 on every seed, which is the edge of the
  grid in `experiments/run_benchmark.py` (`THETAS`).
- **Fabrication:** detected in only 13% of episodes, below the single-camera baseline.
- **Speed:** with CPU optical flow, the full pipeline is far outside the 15 ms target.
- **Length:** the paper is 8 pages.

## 4. Open items, waiting on the owner

1. **Optional:** extend `THETAS` with 0.95 and 0.99, then rerun the benchmark (about 50 minutes) and regenerate
   the summary, tables, figures and PDF. This is fair because θ is still chosen on validation data. It would
   likely shrink the variance in false alarms.
2. **Optional:** add Lu Sun to `authors` in `pyproject.toml`.

## 5. How to resume on a fresh machine

```bash
git clone https://github.com/faysalchowdhoury/airport-cam-byzantine-guard
cd airport-cam-byzantine-guard
git config user.name "S M Tanvir Faysal Alam Chowdhoury"
git config user.email "faysalchowdhoury@users.noreply.github.com"
pip install -e .[test]        # use Python 3.12; on the original machine, Python 3.13's numpy DLL was broken
```

- **Data:** `data/` (about 360 MB) is git-ignored and was deleted locally. Rebuild it with the download step in
  `scripts/reproduce.sh`. WILDTRACK is fetched with HTTP range requests from the remote ZIP and resumes if it is
  interrupted.
- **OpenSky is live:** a fresh capture differs from the one in the paper, whose hashes are in
  `artifacts/data_provenance.json`. Keep the committed artifacts unless a full rerun is intended.
- **Tests:** `pytest`. There are 22 tests and all pass.
- **Paper:** run `python paper/make_tables.py`, then run `pdflatex main.tex` twice inside `paper/` (with
  bibtex the first time, if the references change). It uses MiKTeX on Windows.
- **Python in Git Bash:** `python` may point to the wrong interpreter. Set `PYTHON=<path to 3.12>` for
  `scripts/reproduce.sh`.
