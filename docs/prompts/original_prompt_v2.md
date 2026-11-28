

You are an autonomous research and software engineering agent working on Windows 11 in `C:\Users\sheel\Downloads\airport-cam-byzantine-guard`.

### 1. OBJECTIVE & SCOPE
Build, execute, verify, and publish a complete, publication-grade research repository and LaTeX manuscript for **Paper 1 (Scenario 1: Airport Cyber-Physical Camera & Drone Security — Availability-First)**:
- **Title:** *Passive-Chain Attestation: Bounded-Degree Visibility Rigidity and Coverage-Weighted Blind-Time Minimization for Airport Surveillance*
- **Core Thesis & Proof Architecture:** Unlike zero-passenger nuclear plants (integrity-first), international airports must prioritize **availability and continuous airside/terminal coverage** under dense passenger throughput. Formalize a **Site-Class-Aware Security Priority Vector (SCSPV)** for airports and model the $n$ surveillance cameras ($k$ Byzantine-compromised via Wi-Fi deauth, RTSP saturation, DHCP starvation, or drone-relayed stream injection) as a **bounded-degree visibility graph**. Prove a **Compression-by-Contradiction Theorem**: an adversary cannot compress fabricated trajectories across shared epipolar overlap planes ($p^T E p' = 0$) below a minimum cross-camera redundancy threshold $m(n,k)$ without violating multi-view geometric distinctness across healthy nodes. Pair this lower bound with a constructive **Spatio-Temporal Graph Neural Network (ST-GNN)** + **Mixed-Integer Linear Programming (MILP)** optical hand-off controller that treats FAA Remote ID / ADS-B strictly as an untrusted weak prior (never ground truth).

### 2. STRICT DATA & INTEGRITY RULES
- **REAL DATA ONLY — ZERO SYNTHETIC DATA:** Download and use real public multi-camera overlapping surveillance datasets (e.g., **PETS 2009 / WILDTRACK** multi-camera pedestrian/apron benchmarks on Zenodo/HuggingFace) paired with real public **OpenSky Network ADS-B** airspace tracks and **OpenStreetMap (OSM)** airport terminal/runway geometries (JFK/ATL/DCA).
- Never generate mock video frames or fake ADS-B coordinates. If any mirror fails, fetch a verified open mirror and log exact URLs, licensing, and SHA-256 checksums in `artifacts/data_provenance.json`.
- **Gate 0 Feasibility Check:** Verify cross-camera overlapping field-of-view (FOV) homography calibration and ground-truth labels before running experiments; fail closed with `NOT_EVALUABLE` if alignment fails.

### 3. REPOSITORY ARCHITECTURE & EXECUTION PIPELINE
Create `C:\Users\sheel\Downloads\airport-cam-byzantine-guard` with:
- `data_pipeline/`: Automated downloader + SHA-256 verifier for overlapping multi-camera benchmarks, OpenSky ADS-B tracks, and OSM airport layout polygons.
- `theory/`: Formal derivation and numerical solver proving the **Bounded-Degree Visibility Graph Compression Contradiction** ($m(n,k)$ redundancy threshold) and the availability-weighted MILP optical hand-off formulation (`scipy.optimize.milp`).
- `models/`: PyTorch Spatio-Temporal GNN (`st_gnn_attestor.py`, zero PINNs) fusing cross-camera epipolar residuals, RTSP frame-rate/jitter telemetry, and untrusted ADS-B/Remote-ID weak priors.
- `experiments/`: Multi-seed (`seeds = [42, 101, 202, 303, 404]`) benchmarks comparing: (1) Single-Camera Analytics Baseline, (2) Naive Majority Voting, (3) Static Redundancy, and (4) Our Availability-First SCSPV Attestor + MILP Hand-off across $k/n \in [0.1, 0.45]$ under RTSP saturation, Wi-Fi deauth frame drops, and drone-relayed replay injection.
- `figures/`: Publication-grade plotting scripts generating 5 distinct, attractive figures at **400 DPI**, **pure white background (`#FFFFFF`)**, **Times New Roman typography**, colorblind-safe Okabe-Ito/Viridis palettes, zero overlapping labels:
  1. `fig1_airport_architecture.png` (**Horizontal Swimlane & Multi-Layer Pipeline Diagram**): Airside/Terminal Camera Visibility Graph $\rightarrow$ Untrusted ADS-B/Remote-ID Weak-Prior Layer $\rightarrow$ ST-GNN Epipolar Attestation & MILP Optical Hand-off Controller.
  2. `fig2_byzantine_quorum_bound.png` (**Analytical Surface & Contour Phase Diagram**): Theoretical vs. empirical phase boundary of minimum redundancy $m(n,k)$ vs. compromised fraction $k/n$ showing the compression-by-contradiction threshold.
  3. `fig3_blind_time_pareto.png` (**Multi-Panel Pareto Frontier & Violin Distribution**): Coverage-weighted blind minutes vs. false-alarm terminal disruption cost across 5 random seeds.
  4. `fig4_zone_criticality_heatmap.png` (**Airport Spatial Zone Risk & Camera Hand-off Graph**): Rendered over real OSM runway/apron/terminal geometry showing dynamic camera re-allocation during an airside protocol attack.
  5. `fig5_ablation_latency.png` (**Grouped Error-Bar & Cumulative Latency CDF**): Real-time inference latency (<15 ms/frame) and ablation over epipolar geometry, frame-jitter, and untrusted ADS-B priors.
- `paper/`: Complete IEEE Transactions (`IEEEtran`) LaTeX manuscript (`main.tex`, `references.bib`, compiled `main.pdf`) with formal theorems, proofs, tables directly populated from `artifacts/results_summary.json`, and all 5 figures embedded.
- `Dockerfile`, `pyproject.toml`, `tests/` (pytest suite), and `README.md`.

### 4. GIT & GITHUB PUBLICATION (SOLE CONTRIBUTOR)
- Configure local git identity strictly as:
  `git config user.name "S M Tanvir Faysal Alam Chowdhoury"`
  `git config user.email "faysalchowdhoury@users.noreply.github.com"`
  (Ensure NO Claude/Anthropic co-author trailers appear in any commit message.)
- Run `pytest` and verify all 5 figures and `paper/main.pdf` compile cleanly.
- Use GitHub CLI (`gh repo create faysalchowdhoury/airport-cam-byzantine-guard --public --source=. --remote=origin --push`) to create the public repository and push all commits so `faysalchowdhoury` is the sole contributor.
