"""Regression: known-answer blunders must produce the expected motif/direction.
Run: python3 scripts/04_tagger/tests/test_cook_adapter.py"""
import sys, json, chess, os
HERE=os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE,'..'))
from mistake import Mistake
import cook_adapter as CA

# (fen, played, best, best_line_san, refutation_san, expected_tag, expected_direction)
CASES=[
 # f59: g6h5 allows a forced mate (doubleBishop) — the hard tier, must work
 ("r2qr1k1/1b2bp1p/pnp1p1pQ/1p1nP2N/2pP4/5N2/PPB2PPP/R1B1R1K1 b - - 3 16","g6h5","e7f8",
  ["Bf8","Qd2","gxh5","Bxh7+"],["Qxh7+","Kf8","Qh8#"],"mate","allowed"),
]
def run():
    ok=0; tot=0
    for fen,pl,bm,bl,rf,etag,edir in CASES:
        b=chess.Board(fen)
        m=Mistake(fen,pl,bm,bl,rf,0,0,400,b.turn)
        tags=CA.tag_mistake(m)
        hit=any(t==etag and d==edir for t,d,_ in tags)
        tot+=1; ok+=hit
        print(f"  [{'PASS' if hit else 'FAIL'}] expect {etag}/{edir} | got {tags}")
    print(f"{ok}/{tot} passed")
    return ok==tot
if __name__=="__main__":
    sys.exit(0 if run() else 1)
