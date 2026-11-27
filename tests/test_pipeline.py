"""Pipeline tests. Data-dependent tests skip cleanly when the datasets are absent."""
import io
import json
import zipfile

import numpy as np
import pytest
import torch

from data_pipeline.provenance import ARTIFACTS, DATA
from data_pipeline.remote_zip import repair_wrapped_offsets
from data_pipeline.wildtrack import position_id_to_world, rodrigues
from experiments.controller import quarantine
from experiments.streams import EDGE_FEATS, NODE_FEATS
from models.st_gnn_attestor import STGNNAttestor

HAVE_META = (DATA / "wildtrack" / "raw" / "calibrations").exists()
HAVE_FRAMES = (DATA / "wildtrack" / "frames" / "C1.npy").exists()


def test_position_id_mapping():
    assert position_id_to_world(0) == (-300.0, -900.0)
    assert position_id_to_world(481) == (-297.5, -897.5)


def test_rodrigues_orthonormal():
    R = rodrigues(np.array([0.3, -1.2, 0.7]))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_repair_wrapped_offsets_noop_on_valid_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i in range(5):
            z.writestr(f"f{i}.txt", "x" * (100 + i))
    z = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
    assert repair_wrapped_offsets(z) == 0
    assert z.read("f3.txt") == b"x" * 103


def test_quarantine_hysteresis():
    s = np.array([[0, 0.9, 0.9, 0.9, 0, 0, 0, 0, 0, 0, 0]])
    Q = quarantine(s, 0.5)
    assert not Q[0, 1] and Q[0, 2] and Q[0, 6] and not Q[0, 10]


def test_stgnn_causal_and_prior_bounded():
    torch.manual_seed(0)
    m = STGNNAttestor(len(NODE_FEATS), len(EDGE_FEATS), node_names=NODE_FEATS, beta=0.5)
    m.eval()
    Xn = torch.randn(1, 12, 5, len(NODE_FEATS))
    Xe = torch.rand(1, 12, 5, 5, len(EDGE_FEATS))
    adj = torch.ones(1, 5, 5) - torch.eye(5)
    with torch.no_grad():
        a = m(Xn, Xe, adj)
        Xn2 = Xn.clone()
        Xn2[:, 8:] += 3.0                      # perturb the future only
        b = m(Xn2, Xe, adj)
        assert torch.allclose(a[:, :8], b[:, :8])
        Xn3 = Xn.clone()
        Xn3[..., NODE_FEATS.index("adsb_prior")] = 1e6   # adversarial prior
        c = m(Xn3, Xe, adj)
        assert (c - a).abs().max() <= 2 * m.beta + 1e-5


@pytest.mark.skipif(not HAVE_META, reason="WILDTRACK metadata not downloaded")
def test_gate0_and_scene():
    from data_pipeline.gate0 import run
    rep = run(require_frames=HAVE_FRAMES)
    assert rep["status"] == "PASS"
    from experiments.scene import load_scene
    sc = load_scene()
    assert sc.n_cams == 7 and sc.n_frames == 400
    assert np.isclose(sc.weights.sum(), 1.0)


@pytest.mark.skipif(not HAVE_META, reason="WILDTRACK metadata not downloaded")
def test_epipolar_constraint_on_real_calibration():
    from experiments.scene import load_scene
    sc = load_scene()
    X = np.array([[150.0, 600.0]])
    for i, j in [(0, 5), (1, 2)]:
        xi, _ = sc.project(i, X, 170.0)
        xj, _ = sc.project(j, X, 170.0)
        F = sc.fundamental(i, j)
        r = np.r_[xj[0], 1] @ F @ np.r_[xi[0], 1]
        line = F @ np.r_[xi[0], 1]
        assert abs(r) / np.hypot(line[0], line[1]) < 1e-6


@pytest.mark.skipif(not (ARTIFACTS / "data_provenance.json").exists(), reason="no provenance yet")
def test_provenance_has_sha256():
    prov = json.loads((ARTIFACTS / "data_provenance.json").read_text(encoding="utf-8"))
    for key, src in prov["sources"].items():
        blob = json.dumps(src)
        assert "sha256" in blob, key


@pytest.mark.skipif(not (ARTIFACTS / "results_summary.json").exists(), reason="benchmark not run")
def test_results_summary_complete():
    S = json.loads((ARTIFACTS / "results_summary.json").read_text(encoding="utf-8"))
    assert S["benchmark"]["seeds"] == [42, 101, 202, 303, 404]
    assert set(S["benchmark"]["methods"]) == {"single_camera", "majority_vote", "static_redundancy", "scspv_attestor"}
    assert S["theory"]["mismatches"] == 0
    assert S["gate0"]["status"] == "PASS"
