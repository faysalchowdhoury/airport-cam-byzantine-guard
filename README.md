# Passive-Chain Attestation for Airport Surveillance

**Bounded-Degree Visibility Rigidity and Coverage-Weighted Blind-Time Minimization**

This repository holds the research code, real-data pipeline, and IEEEtran manuscript (`paper/main.pdf`) for an
availability-first defence of airport camera networks against Byzantine (compromised or degraded) cameras.

* **Theory** (`theory/`): the *Compression-by-Contradiction* theorem gives the per-zone redundancy needed when `k`
  cameras collude:

  | target | detect | mask |
  |---|---|---|
  | ground plane | `k+1` | `2k+1` |
  | off-plane (epipolar-degenerate) | `k+2` | `2k+2` |

  The bounds are tight. Every tight case has an explicit pair of indistinguishable worlds, and an exhaustive
  enumeration matches the closed forms. Because the cameras covering a zone form a clique in the visibility graph,
  the bound also caps tolerable collusion at `k <= floor(Delta/2)`.
* **Method** (`models/`, `theory/handoff_milp.py`): a causal ST-GNN attestor fuses cross-view epipolar and
  homography residuals, Horn–Schunck optical flow, frame-jitter telemetry, and an untrusted ADS-B/Remote-ID prior. The
  prior is gated so its effect on the logit is at most `2*beta`. The attestor's risk scores drive an availability-weighted
  MILP (`scipy.optimize.milp`) that hands analytics slots off to healthy overlapping cameras.
* **Evaluation** (`experiments/`): five seeds `[42, 101, 202, 303, 404]` and `k/n` from 0.14 to 0.43. The four
  methods compared are single-camera analytics, naive majority vote, static redundancy, and the SCSPV attestor with
  MILP hand-off. The attacks are drone-relayed replay, Wi-Fi deauth burst loss, RTSP saturation, colluding metadata
  fabrication, and a mixed campaign.

## Data (real only, nothing synthesized)

| Source | Use | Access |
|---|---|---|
| [WILDTRACK](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) (Chavdarova et al., CVPR 2018) | 7 calibrated, overlapping cameras; 400 annotated frames | HTTP range requests on the 6.8 GB ZIP, fetching only the needed members and repairing 32-bit-wrapped offsets |
| [OpenSky Network](https://opensky-network.org) REST API | live ADS-B state vectors, JFK / ATL / DCA | 90 snapshots at 10 s intervals |
| [OpenStreetMap](https://www.openstreetmap.org) via Overpass | runway, taxiway, apron, terminal and stand polygons | Overpass API with mirror fallback |

Every downloaded member or response is SHA-256 hashed into `artifacts/data_provenance.json`, along with its URL and
licence. The raw data are **not redistributed** here because of their licence terms (`data/` is git-ignored). Running
`scripts/reproduce.sh` downloads them again.

**Emulated rather than recorded:** attack effects (re-routing of real frames, metadata tampering) and transport
telemetry (latency, jitter, burst loss). WILDTRACK carries no network traces. ADS-B spoofing is emulated by
transplanting one airport's real traffic onto another. The paper states all of this explicitly.

**Gate 0** (`data_pipeline/gate0.py`) checks four things and fails closed with `NOT_EVALUABLE` if any check fails:
homography invertibility, label reprojection, cross-view transfer error, and visibility-graph connectivity.

## Layout

```
data_pipeline/   downloaders (WILDTRACK range-zip, OpenSky, OSM), SHA-256 ledger, Gate 0
theory/          quorum_bound.py (bounds + exhaustive verification), handoff_milp.py (MILP / greedy / static)
models/          st_gnn_attestor.py (causal ST-GNN, bounded untrusted prior)
experiments/     scene, optical flow, attack emulation, features, controllers, benchmark,
                 phase boundary, latency, airport case study (ATL Concourse F), summarizer
figures/         make_figures.py -> fig1..fig5 (400 DPI, Times New Roman, Okabe-Ito / Viridis)
paper/           main.tex, references.bib, make_tables.py (tables/macros from results_summary.json), main.pdf
artifacts/       provenance, gate0, phase boundary, per-seed runs, latency, results_summary.json
tests/           pytest suite
```

## Reproduce

```bash
pip install -e .[test]          # Python >= 3.10, CPU-only torch is sufficient
bash scripts/reproduce.sh       # download -> Gate 0 -> theory -> benchmark -> figures -> tests -> paper
```

or `docker build -t accg . && docker run --rm accg`.

The OpenSky step captures a *live* window, so a fresh run yields different ADS-B snapshots. The hashes of the capture
used in the paper are recorded in `artifacts/data_provenance.json`.

## Headline results

Test episodes, mean ± std over five seeds (`artifacts/results_summary.json`; the paper's tables are generated from it):

| Method | Blind min/h ↓ | Deceived min/h ↓ | False alarms/h ↓ | Frame AUROC ↑ |
|---|---|---|---|---|
| Single-camera analytics | 3.36 ± 0.26 | 1.17 ± 0.22 | 238.6 ± 10.0 | 0.730 ± 0.008 |
| Naive majority vote | 1.91 ± 0.18 | 1.06 ± 0.15 | 56.7 ± 1.4 | 0.648 ± 0.009 |
| Static redundancy | 3.04 ± 0.28 | 0.85 ± 0.15 | **12.5 ± 1.9** | 0.599 ± 0.008 |
| **SCSPV attestor + MILP** | **1.79 ± 0.05** | **0.76 ± 0.09** | 37.0 ± 28.2 | **0.852 ± 0.010** |

* Ground-plane phase boundary on real views matches the theorem in every cell. Off-plane it appears one witness later,
  as the probabilistic extension predicts for crowded scenes.
* The attestor (geometry + ST-GNN) runs in 12.0 ms p95 per frame for 7 cameras on one CPU thread. Optical flow
  (214 ms p95 per camera on CPU) and uncached MILP solves (50 ms p95) are outside that budget.
* Known weaknesses, reported in the paper: false alarms vary widely across seeds (the selected threshold sits at the
  edge of the grid), and coordinated metadata fabrication is detected in only 13% of episodes.

See `paper/main.pdf`, Section VII.

## Licence

Code: MIT. Data remain under their original licences: WILDTRACK research terms, the OpenSky terms of use, and ODbL for
OpenStreetMap.
