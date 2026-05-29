"""Assemble per-feature evidence: description + label + fingerprint + verification.

Pure local computation. Skips features whose chip is INSUFFICIENT/ERROR or
confidence==0. Output keyed by feature_id (string).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
from collections import Counter

from verify_descriptions import move_fingerprint, verify_description


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--out-evidence", required=True)
    ap.add_argument("--out-verification", required=True)
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    labels = json.load(open(args.labels))
    profiles = json.load(open(args.profiles))

    evidence = {}
    verification = {}
    for fid, lab in labels.items():
        chip = lab.get("chip", "")
        if chip in ("INSUFFICIENT", "ERROR") or lab.get("confidence", 0) == 0:
            continue
        prof = profiles.get(fid, {})
        positions = [(ex["fen"], ex["uci"]) for ex in prof.get("examples", [])[: args.top_n]]
        fp = move_fingerprint(positions)
        ver = verify_description(lab.get("description", ""), fp)
        evidence[fid] = {
            "feature_id": int(fid),
            "label": lab.get("label", ""),
            "description": lab.get("description", ""),
            "old_chip": chip,
            "fingerprint": fp,
            "verification": ver,
        }
        verification[fid] = ver
    json.dump(evidence, open(args.out_evidence, "w"), indent=2)
    json.dump(verification, open(args.out_verification, "w"), indent=2)
    print(f"Built evidence for {len(evidence)} features.")
    h = Counter(v["verdict"] for v in verification.values())
    print("verdicts:", dict(h))


if __name__ == "__main__":
    main()
