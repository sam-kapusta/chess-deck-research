import json, os, sys, subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_assembled_shape(tmp_path):
    # minimal evidence + assignments + vocab, check merge shape
    ev = {
        "0": {"feature_id": 0, "label": "A slow a-pawn push.", "description": "8/10 a-pawn pushes.",
              "old_chip": "Slow flank pawn", "fingerprint": {"dom_piece": "pawn"},
              "verification": {"verdict": "partial"}},
    }
    asg = {"assignments": {"0": {"feature_id": 0, "category": "slow_play_punished",
                                 "chip": "a-pawn push over center", "confidence": 90, "corrected": False}}}
    vocab = {"source": "test", "categories": [
        {"id": "slow_play_punished", "name": "Slow Play Punished", "definition": "x"}]}
    ev_p = tmp_path / "ev.json"; asg_p = tmp_path / "asg.json"; voc_p = tmp_path / "voc.json"
    out = tmp_path / "tax.json"
    json.dump(ev, open(ev_p, "w")); json.dump(asg, open(asg_p, "w")); json.dump(vocab, open(voc_p, "w"))
    r = subprocess.run([sys.executable,
        os.path.join(ROOT, "scripts/sae/taxonomy/assemble.py"),
        "--evidence", str(ev_p), "--assignments", str(asg_p),
        "--vocab", str(voc_p), "--out", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t = json.load(open(out))
    assert "features" in t and "categories" in t and "category_index" in t
    f0 = t["features"]["0"]
    for k in ("feature_id", "chip", "title", "description", "category", "category_name", "fingerprint"):
        assert k in f0, k
    assert f0["category"] == "slow_play_punished"
    assert f0["category_name"] == "Slow Play Punished"
    assert t["category_index"]["slow_play_punished"] == [0]
