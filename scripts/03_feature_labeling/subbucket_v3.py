#!/usr/bin/env python3
"""Sub-bucket the v3 taxonomy assignments — MECHANICAL, no LLM.

Each top-level bucket splits the way its features actually vary (read off the chip vocab):
  - MATERIAL buckets (Left Hanging / Greedy / Missed Hanging / Premature Trade / Pointless Check)
    split BY PIECE — the piece word is in the chip ("Hangs queen", "Greedy pawn grab", "Missed free
    rook capture"). Fall back to the SEE moved/captured-piece distribution when the chip is generic.
  - THEMATIC buckets (Missed Tactic / Missed Check-Mate / King Safety / Unsound Aggression /
    Endgame / Abandoning Defense / Passive) split BY THEME via chip keywords (fork/pin, mate/check,
    pawn/king, sac/lunge...).

No model call — the sub-label is derived from the chip + SEE, so it's transparent and tunable.
Output feeds render_taxonomy_tree (repointed to v3) for the browsable tree.

Run locally:
  python3 subbucket_v3.py --labels output/relabel_v2_neutral_d2048_k6.json \
    --assign output/feature_buckets_v3_v2labels_d2048_k6.json --stats output/see_stats_d2048_k6.json \
    --buckets output/buckets_v3_d2048_k6.json --out output/feature_leaf_v3_d2048_k6.json
"""
import argparse, json, re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--assign", required=True)
ap.add_argument("--stats", required=True)
ap.add_argument("--buckets", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

lab = json.load(open(a.labels))
asg = json.load(open(a.assign))["assignments"]
st = json.load(open(a.stats))
buckets = json.load(open(a.buckets))
name = {b["id"]: b["name"] for b in buckets}
def S(f): return st.get("f" + f) or st.get(f) or {}

PIECES = ["queen", "rook", "bishop", "knight", "pawn", "king"]
PIECE_TITLE = {p: p.title() for p in PIECES}


def piece_from_chip(chip):
    c = chip.lower()
    for p in PIECES:
        if p in c: return p
    if "minor" in c: return "minor"
    return None


def piece_from_see(f, key):
    d = S(f).get(key, {}) or {}
    if not d: return None
    top = max(d, key=d.get)
    return top if top in PIECES + ["minor"] else None


def by_piece(f, see_key="moved_piece_pct"):
    """Sub-label = the piece, from chip first then SEE distribution."""
    p = piece_from_chip(lab[f]["chip"]) or piece_from_see(f, see_key)
    return PIECE_TITLE.get(p, "Minor" if p == "minor" else "Other")


def by_keywords(chip, rules, default):
    c = chip.lower()
    for label, kws in rules:
        if any(k in c for k in kws): return label
    return default


# --- per-bucket sub-rules ---
def sub_missed_tactic(f):
    # Specific motifs first; everything unspecified collapses to one honest "General Tactic" sub
    # (the chips for these are genuinely non-specific — "Missed knight tactic", "Missed crushing
    # tactic" — so they can't be subdivided further without re-chipping).
    return by_keywords(lab[f]["chip"], [
        ("Fork / Double Attack", ["fork", "double attack", "double"]),
        ("Pin / Skewer", ["pin", "skewer"]),
        ("Zwischenzug / Intermezzo", ["zwischenzug", "intermezzo", "intermediate", "in-between"]),
        ("Discovered Attack", ["discover"]),
        ("Trap / Wins Piece", ["trap", "wins piece", "win piece", "traps"]),
        ("Sacrifice / Greek Gift", ["sacrifice", "sac ", "greek gift", "bxf7", "bxh7", "bxh6"]),
        ("Missed Capture", ["capture", "captures", "takes"]),
    ], "General / Forcing Tactic")

def sub_check_mate(f):
    return by_keywords(lab[f]["chip"], [
        ("Missed Mate", ["mate", "checkmate", "#"]),
        ("Missed Back-Rank", ["back-rank", "back rank", "backrank"]),
        ("Missed Winning Check", ["check"]),
    ], "Missed Forcing Check")

def sub_endgame(f):
    return by_keywords(lab[f]["chip"], [
        ("Pawn Promotion", ["promotion", "promote", "queening", "underpromot"]),
        ("Pawn Push / Break", ["pawn", "push", "passed", "break", "advance"]),
        ("King Technique", ["king", "opposition", "blockade"]),
        ("Rook Endgame", ["rook"]),
    ], "Other Endgame")

def sub_king_safety(f):
    return by_keywords(lab[f]["chip"], [
        ("Walks Into Mate", ["mate", "mating"]),
        ("Ignored Threat to King", ["ignor", "neglect", "lethal", "danger", "threat"]),
        ("Castling Error", ["castl"]),
        ("Weakened Shelter", ["shelter", "expos", "weaken", "back-rank", "back rank"]),
    ], "King Exposure")

def sub_unsound(f):
    return by_keywords(lab[f]["chip"], [
        ("Unsound Sacrifice", ["sac", "sacrifice"]),
        ("Premature Attack", ["premature", "lunge", "sortie", "attack"]),
        ("Impulsive Forcing", ["impulsive", "speculative", "flashy"]),
    ], "Overextension")

def sub_abandon(f):
    return by_keywords(lab[f]["chip"], [
        ("Removed Key Defender", ["defender", "defense", "defensive", "removes", "removing", "abandon"]),
        ("Unblocked Critical Line", ["unblock", "line", "diagonal", "file"]),
    ], "Neglected Duty")

def sub_passive(f):
    return by_keywords(lab[f]["chip"], [
        ("Missed Exchange", ["trade", "exchange", "simplif"]),
        ("Passive / Drifting", ["passive", "aimless", "retreat", "drift", "wasted", "tempo", "routine", "quiet"]),
    ], "Ignored Initiative")

RULES = {
    "left_hanging": lambda f: "Hangs " + by_piece(f),
    "greedy_capture": lambda f: "Greedy " + by_piece(f, "captured_piece_pct") + " Grab",
    "missed_hanging": lambda f: "Missed " + by_piece(f, "best_captured_piece_pct") + " Capture",
    "premature_trade": lambda f: by_piece(f) + " Trade",
    "pointless_check": lambda f: by_piece(f) + " Check",
    "missed_tactic": sub_missed_tactic,
    "missed_check_mate": sub_check_mate,
    "endgame_technique": sub_endgame,
    "king_safety": sub_king_safety,
    "unsound_aggression": sub_unsound,
    "abandoned_defense": sub_abandon,
    "passive_play": sub_passive,
}

leaf = {}
for f, bid in asg.items():
    if bid == "unassignable":
        leaf[f] = {"bucket": bid, "bucket_name": "UNASSIGNABLE", "sub": "-"}; continue
    sub = RULES.get(bid, lambda f: "-")(f)
    leaf[f] = {"bucket": bid, "bucket_name": name[bid], "sub": sub}
json.dump(leaf, open(a.out, "w"), indent=1)

# rollup report: bucket -> sub -> (count, fire)
def fr(f): return S(f).get("fire_rate", 0)
roll = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
for f, v in leaf.items():
    roll[v["bucket"]][v["sub"]][0] += 1
    roll[v["bucket"]][v["sub"]][1] += fr(f)
order = {b["id"]: i for i, b in enumerate(buckets)}
print(f"wrote {a.out}\n")
for bid in sorted(roll, key=lambda k: order.get(k, 99)):
    if bid == "unassignable": continue
    subs = roll[bid]
    tot = sum(v[0] for v in subs.values())
    print(f"{name[bid]} ({tot})")
    for sub, (n, tf) in sorted(subs.items(), key=lambda kv: -kv[1][0]):
        print(f"    {sub:28s} {n:>4} feats   {tf*100:5.1f}% fire")
