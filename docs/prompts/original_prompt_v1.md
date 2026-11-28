

You are an autonomous research and software engineering agent working on Windows 11 in `C:\Users\sheel\Downloads\airport-cam-byzantine-guard`.

### 1. OBJECTIVE & SCOPE
Build, execute, verify, and publish a complete, publication-grade research repository and LaTeX manuscript for **Paper 1 (Scenario 1: Airport Cyber-Physical Camera & Drone Security — Availability-First)**:
- **Title:** *Passive-Chain Attestation: Bounded-Redundancy Visual Non-Invasion and Coverage-Weighted Blind-Time Minimization for Airport Surveillance*
- **Core Thesis:** Unlike zero-passenger nuclear plants (integrity-first), international airports must prioritize **availability and continuous airside/terminal coverage** under high passenger throughput. We formalize a **Site-Class-Aware Security Priority Vector (SCSPV)** for airports, model $n$ surveillance cameras ($k$ Byzantine-compromised via passive RF/protocol degradation or drone-relayed stream injection) alongside untrusted FAA Remote ID / ADS-B priors, derive the minimum cross-camera epipolar & optical-flow redundancy $m(n,k)$ needed to bound coverage-weighted blind minutes, and build a real-time PyTorch Spatio-Temporal Graph Neural Network (ST-GNN) + integer-programming hand-off controller.

### 2. STRICT DATA & INTEGRITY RULES
- **REAL DATA ONLY — ZERO SYNTHETIC DATA:** Download and use real public surveillance, crowd-tracking, and aviation telemetry datasets (e.g., **PETS 2009 / DukeMTMC / WILDTRACK** multi-camera overlapping pedestrian/apron-like benchmarks + real public **OpenSky Network ADS-B** aircraft/airspace tracks and **OpenStreetMap** airport terminal/runway geometries for JFK/ATL/DCA).
- Never generate mock/synthetic video frames or fake ADS-B coordinates. If any dataset URL requires authentication or fails, automatically fetch an verified open mirror (e.g., Zenodo, HuggingFace, OpenSky public historical dump) and log exact SHA-256 checksums and provenance in `artifacts/data_provenance.json`.
- **Gate 0 Feasibility Check:** Verify cross-camera overlapping field-of-view (FOV) homography and ground-truth labels before running experiments; fail closed with `NOT_EVALUABLE` if alignment fails.

### 3. REPOSITORY ARCHITECTURE & EXECUTION PIPELINE
Create `C:\Users\sheel\Downloads\airport-cam-byzantine-guard` with:
- `data_pipeline/`: Automated downloader + SHA-256 verifier for multi-camera overlapping benchmarks, OpenSky ADS-B tracks, and OSM airport layout polygons.
- `theory/`: Symbolic/numerical verification of the Byzantine visual quorum bound $m(n,k)$ and coverage-weighted blind-time optimization solver (`scipy.optimize.milp` / `cvxpy`).
- `models/`: PyTorch Spatio-Temporal GNN (`st_gnn_attestor.py`) fusing cross-camera epipolar consistency residuals, RTSP frame-rate/jitter telemetry, and untrusted ADS-B/Remote-ID weak priors.
- `experiments/`: Multi-seed (`seeds = [42, 101, 202, 303, 404]`) evaluation comparing: (1) Single-Camera Analytics Baseline, (2) Naive Majority Voting, (3) Fixed Redundancy, and (4) Our Availability-First SCSPV Attestor + Optical Hand-off across $k/n \in [0.1, 0.45]$ under RTSP saturation, Wi-Fi deauth frame drops, and drone-relayed replay injection.
- `figures/`: Publication-grade plotting scripts generating 5 distinct, attractive figures at **400 DPI**, **pure white background (`#FFFFFF`)**, **Times New Roman typography**, colorblind-safe Okabe-Ito/Viridis palettes, zero overlapping labels:
  1. `fig1_airport_architecture.png` (**Horizontal Swimlane & Multi-Layer Pipeline Diagram**): Airside/Terminal Camera Layer $\rightarrow$ Untrusted ADS-B/Remote-ID Weak-Prior Layer $\rightarrow$ ST-GNN Epipolar Attestation & MILP Optical Hand-off Controller.
  2. `fig2_byzantine_quorum_bound.png` (**Analytical Surface & Contour Phase Diagram**): Theoretical vs. empirical phase boundary of minimum redundancy $m(n,k)$ vs. compromised fraction $k/n$ across high-density passenger zones.
  3. `fig3_blind_time_pareto.png` (**Multi-Panel Pareto Frontier & Violin Distribution**): Coverage-weighted blind minutes vs. false-alarm terminal disruption cost across 5 random seeds.
  4. `fig4_zone_criticality_heatmap.png` (**Airport Spatial Zone Risk & Camera Hand-off Graph**): Rendered over real OSM runway/apron/terminal geometry showing dynamic camera re-allocation during a simulated airside protocol attack.
  5. `fig5_ablation_latency.png` (**Grouped Error-Bar & Cumulative Latency CDF**): Real-time inference latency (<15 ms/frame) and ablation over epipolar geometry, frame-jitter, and ADS-B priors.
- `paper/`: Complete IEEE Transactions (`IEEEtran`) LaTeX manuscript (`main.tex`, `references.bib`, compiled `main.pdf`) with formal theorems, proofs, tables directly populated from `artifacts/results_summary.json`, and all 5 figures embedded.
- `Dockerfile`, `pyproject.toml`, `tests/` (pytest suite), and `README.md`.

### 4. GIT & GITHUB PUBLICATION (SOLE CONTRIBUTOR)
- Configure local git identity strictly as:
  `git config user.name "S M Tanvir Faysal Alam Chowdhoury"`
  `git config user.email "faysalchowdhoury@users.noreply.github.com"`
  (Ensure NO Claude/Anthropic co-author trailers appear in any commit message.)
- Run `pytest` and verify all 5 figures and `paper/main.pdf` compile cleanly.
- Use GitHub CLI (`gh repo create faysalchowdhoury/airport-cam-byzantine-guard --public --source=. --remote=origin --push`) to create the public repository and push all commits so `faysalchowdhoury` is the sole contributor.
