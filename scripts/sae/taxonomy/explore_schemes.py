"""Explore multiple TOP-LEVEL category schemes for the chess blunder taxonomy.

For each scheme, we define categories by a short text definition, embed those
definitions with bge-m3, and assign each feature to its nearest definition by
cosine (using the already-computed feature embeddings). Reports real counts +
fire-rate share so we can compare how each scheme actually carves the 1996
features — no LLM calls needed.

Schemes encode genuinely different organizing PHILOSOPHIES (not just relabels):
the point is to see the shape of each before committing to one.
"""
import json
import numpy as np

ROOT = "output/taxonomy_v2"


def load():
    rows = json.load(open(f"{ROOT}/cluster_input.json"))
    fids = sorted(rows, key=lambda k: int(k))
    emb = np.load(f"{ROOT}/embeddings.npy")  # concept (piece-stripped mechanism) embeddings
    fire = np.load(f"{ROOT}/firerate_flat_v2_k32.npy")
    fr = np.array([rows[f]["fire_rate"] for f in fids])
    return rows, fids, emb, fr


def embed_defs(defs):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("BAAI/bge-m3")
    return m.encode(defs, normalize_embeddings=True, show_progress_bar=False)


# ---- SCHEMES: each is {name: definition_text} ----
# The definitions are written to match the CONCEPT embedding (mechanism, piece-neutral).

SCHEMES = {
    # 1. Heisman-style "what skill failed" — the classic coaching diagnosis axis
    "A_coaching_skill": {
        "Board Vision / Missed Threats": "failed to see the opponent's threat or a piece left hanging; overlooked an attack on own material",
        "Calculation / Tactics": "missed a concrete tactical shot, combination, fork, pin, or forcing sequence that was available",
        "King Safety": "exposed own king, walked it into danger, or weakened the pawn shelter around it",
        "Time Management / Slow Play": "played a slow, passive, non-forcing move when urgent concrete action was demanded",
        "Material Greed": "grabbed material — a pawn or piece — ignoring the consequences and losing more",
        "Piece Activity / Placement": "placed a piece on a bad square, abandoned an active post, or misplaced it",
        "Pawn Play": "made a pawn move or push that created weaknesses or ignored the position's needs",
        "Endgame Technique": "mishandled an endgame — king-and-pawn race, promotion, or conversion",
    },
    # 2. Phase of game — when in the game does the mistake happen
    "B_game_phase": {
        "Opening Mistakes": "an opening-phase error in development, early king safety, or premature aggression",
        "Middlegame Mistakes": "a middlegame error in piece coordination, tactics, or planning",
        "Endgame Mistakes": "an endgame error with few pieces — king activity, pawn races, promotion",
    },
    # 3. Defensive vs offensive failure — what the player was TRYING to do
    "C_intent": {
        "Overreaching (too aggressive)": "an over-aggressive move — premature attack, unsound sacrifice, or chasing material/checks",
        "Too Passive (not aggressive enough)": "a passive or slow move that failed to take the required forcing action",
        "Failed Defense": "failed to defend — abandoned a guard, left a piece undefended, ignored a threat",
        "Faulty Conversion": "failed to convert or hold — mishandled an advantage, endgame, or passed pawn",
    },
    # 4. Concrete error mechanism — the geometric/tactical 'what happened on the board'
    "D_mechanism": {
        "Piece Left Hanging": "a move leaves a piece undefended or en prise, free to capture",
        "Moved Into Attack": "moved a piece onto a square where it is immediately attacked or forked",
        "Abandoned Defensive Duty": "moved a piece away from a square where it was guarding something critical",
        "Walked King Into Danger": "a king move into a check, mating net, or fork",
        "Bad Pawn Push": "a pawn advance that weakens the position or ignores urgency",
        "Bad Capture / Trade": "a capture or exchange that backfires, releases tension, or activates the opponent",
        "Pointless Check / Tempo Loss": "a check or move that wastes a tempo and accomplishes nothing",
        "Quiet Move Over Forcing Line": "a quiet positional move played when a concrete forcing line was required",
    },
    # 5. Severity / frequency — organize by how often + how costly (product-facing buckets)
    # (assigned by fire-rate band + cp signal, handled separately below)
    # 6. Player-facing themes — broad, marketable "what to work on" buckets (fewest categories)
    "F_player_themes": {
        "Don't Hang Pieces": "leaving pieces undefended, moving into attacks, or abandoning defenders",
        "See the Whole Board": "missing the opponent's threats and available tactics",
        "Keep Your King Safe": "exposing the king or weakening its shelter",
        "Play With Urgency": "passive slow play when forcing action was needed",
        "Don't Be Greedy": "grabbing material or chasing checks at the cost of the position",
        "Master Endgames": "endgame technique, king activity, pawn races, promotion",
    },
}


def assign(emb, def_emb):
    sims = emb @ def_emb.T          # (n_features, n_cats)
    return sims.argmax(1), sims.max(1)


def report(name, cat_names, labels, fr, conf):
    n = len(labels)
    total_fire = fr.sum()
    print(f"\n=== SCHEME {name}  ({len(cat_names)} categories) ===")
    print(f"{'category':<38} {'feats':>5} {'%feat':>5} {'%fire':>6} {'conf':>5}")
    order = sorted(range(len(cat_names)), key=lambda ci: -(labels == ci).sum())
    for ci in order:
        mask = labels == ci
        nf = int(mask.sum())
        fire_share = fr[mask].sum() / total_fire * 100
        mc = conf[mask].mean() if nf else 0
        print(f"{cat_names[ci]:<38} {nf:>5} {nf/n*100:>4.0f}% {fire_share:>5.0f}% {mc:>5.2f}")
    sizes = np.array([(labels == ci).sum() for ci in range(len(cat_names))])
    print(f"  balance: largest {sizes.max()/n*100:.0f}%  smallest {sizes.min()/n*100:.0f}%  "
          f"mean-conf {conf.mean():.2f}")


def main():
    rows, fids, emb, fr = load()

    # fire-rate-band scheme (scheme E) — no embedding needed
    print("\n=== SCHEME E_severity (fire-rate bands) — frequency of the mistake ===")
    bands = [("Very rare (<0.5%)", fr < 0.005),
             ("Uncommon (0.5-1%)", (fr >= 0.005) & (fr < 0.01)),
             ("Common (1-2%)", (fr >= 0.01) & (fr < 0.02)),
             ("Frequent (2-5%)", (fr >= 0.02) & (fr < 0.05)),
             ("Pervasive (>5%)", fr >= 0.05)]
    for nm, mask in bands:
        nf = int(mask.sum())
        print(f"  {nm:<22} {nf:>5} feats  {fr[mask].sum()/fr.sum()*100:>4.0f}% of fire")

    for sname, defs in SCHEMES.items():
        cat_names = list(defs.keys())
        def_emb = embed_defs(list(defs.values()))
        labels, conf = assign(emb, def_emb)
        report(sname, cat_names, labels, fr, conf)


if __name__ == "__main__":
    main()
