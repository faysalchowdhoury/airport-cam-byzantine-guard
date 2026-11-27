#!/usr/bin/env bash
# End-to-end reproduction. Requires network access for the public datasets.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
$PY -m data_pipeline.wildtrack            # WILDTRACK (range-streamed; ~8 GB transferred, ~0.4 GB stored)
$PY -m data_pipeline.osm                  # OpenStreetMap aeroway geometry (JFK / ATL / DCA)
$PY -m data_pipeline.opensky --minutes 15 # OpenSky ADS-B state vectors (live window; differs per run)
$PY -m data_pipeline.gate0                # fail-closed feasibility gate -> NOT_EVALUABLE on failure
$PY -m theory.quorum_bound
$PY -m experiments.phase_boundary
$PY -m experiments.run_benchmark
$PY -m experiments.latency
$PY -m experiments.summarize              # pass 1: detection-latency pool for the case study
$PY -m experiments.airport_case_study
$PY -m experiments.summarize
$PY -m figures.make_figures
$PY -m pytest -q
(cd paper && $PY make_tables.py && latexmk -pdf -interaction=nonstopmode main.tex)
