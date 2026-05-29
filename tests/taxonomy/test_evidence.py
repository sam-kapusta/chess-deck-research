import json, os, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_evidence_built(tmp_path):
    out = tmp_path / "evidence.json"
    ver = tmp_path / "verification.json"
    r = subprocess.run([sys.executable,
        os.path.join(ROOT, "scripts/sae/taxonomy/build_evidence.py"),
        "--labels", os.path.join(ROOT, "output/taxonomy_v2/_inputs/feature_labels.json"),
        "--profiles", os.path.join(ROOT, "output/taxonomy_v2/_inputs/profiles.json"),
        "--out-evidence", str(out), "--out-verification", str(ver)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    ev = json.load(open(out))
    assert len(ev) >= 1990
    f0 = ev["0"]
    assert "description" in f0 and "label" in f0
    assert "fingerprint" in f0 and f0["fingerprint"]["dom_piece"] == "pawn"
    assert "verification" in f0
