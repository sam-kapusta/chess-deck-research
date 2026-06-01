#!/usr/bin/env python3
"""Per-feature stats from top-100 activating positions.

For each of 2048 features: piece type dist, is_capture, is_check,
piece left hanging after blunder, side, eval trajectory, cp_loss,
motif histogram (Opus join), phase. Fully objective - no LLM.

Usage (on chess-poc):
    python scripts/evaluation/feature_stats_btk.py \
      --weights ~/SageMaker/chess-stage-a/output/maia3_sae/btk_2048_k32_weights.pt \
      --cache ~/SageMaker/chess-stage-a/cache/maia3_l7only_v2_dedup.pt \
      --enrichment ~/SageMaker/position_enrichment_cache.json \
      --opus ~/SageMaker/all_positions_labeled_opus.json \
      --output ~/SageMaker/chess-stage-a/output/feature_stats_btk_2048_k32.json
"""
import argparse, json, os, numpy as np, torch, torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import chess

PIECE_NAMES = {chess.PAWN:"pawn",chess.KNIGHT:"knight",chess.BISHOP:"bishop",
               chess.ROOK:"rook",chess.QUEEN:"queen",chess.KING:"king"}
TOP_N = 100   # top positions per feature for stats
TOP_PROFILE = 15  # top positions saved for labeling profiles


class BatchTopKSAE(nn.Module):
    def __init__(self, d_input, d_hidden, k):
        super().__init__()
        self.W_enc = nn.Parameter(torch.empty(d_input, d_hidden))
        self.W_dec = nn.Parameter(torch.empty(d_hidden, d_input))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_input))
        self.d_hidden = d_hidden; self.k = k
        self.register_buffer("num_batches_not_active", torch.zeros(d_hidden))

    def encode_threshold(self, x, threshold):
        """Deterministic threshold inference — no batch dependency (inference_example.py pattern)."""
        z = (x - self.b_dec) @ self.W_enc + self.b_enc
        z_relu = F.relu(z)
        return z_relu * (z_relu > threshold)


def normalize(raw, mean, std):
    x = (raw - mean) / std
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def eval_num(s):
    """Parse white-relative Stockfish eval string to centipawns (mate=+-10000)."""
    if not s: return 0
    s = str(s).strip()
    if s.startswith("#"):
        m = int(s[1:].replace("−","-"))
        return (10000 - abs(m)*10) * (1 if m >= 0 else -1)
    try: return int(float(s) * 100)
    except: return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    required=True)
    parser.add_argument("--cache",      required=True)
    parser.add_argument("--enrichment", required=True)
    parser.add_argument("--opus",       required=True)
    parser.add_argument("--output",     required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    d_input, dict_size, k = cfg["d_input"], cfg["dict_size"], cfg["k"]

    stats_path = args.weights.replace(".pt", "_stats.json")
    with open(stats_path) as f: ns = json.load(f)
    mean = torch.tensor(ns["mean"], dtype=torch.float32)
    std  = torch.tensor(ns["std"],  dtype=torch.float32)

    print("Loading corpus + auxiliary data...")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    meta  = cache["metadata"]
    raw   = cache["activations"].float()
    keys  = [m["fen"] + "|" + m["blunder_uci"] for m in meta]
    x_norm = normalize(raw, mean, std); del raw

    print("Loading enrichment + Opus labels...")
    enr  = json.load(open(args.enrichment))
    opus = json.load(open(args.opus))
    motif_map = {}
    for k_op, v in opus.items():
        a = v.get("analysis", v)
        if isinstance(a, dict) and a.get("tactical_motif"):
            motif_map[k_op] = a["tactical_motif"]

    model = BatchTopKSAE(d_input, dict_size, k)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval().to(device)

    # Load calibrated threshold for deterministic inference (no batch dependency)
    cal_path = args.weights.replace("_weights.pt", "_calibration.json")
    if os.path.exists(cal_path):
        threshold = json.load(open(cal_path))["global_threshold"]
        print(f"Using calibrated threshold θ={threshold:.6f} (L0≈{json.load(open(cal_path))['mean_l0']:.1f})")
    else:
        raise FileNotFoundError(f"Calibration file not found: {cal_path}\nRun calibrate_threshold_v2.py first.")

    print("Encoding full corpus (threshold inference)...")
    loader = DataLoader(TensorDataset(x_norm), batch_size=8192, shuffle=False)
    all_acts = []
    with torch.no_grad():
        for (batch,) in loader:
            all_acts.append(model.encode_threshold(batch.to(device), threshold).cpu().numpy())
    acts = np.concatenate(all_acts)
    print(f"  Acts shape: {acts.shape}, mean L0: {(acts>0).sum(1).mean():.1f}")

    print("Computing per-feature stats...")
    out = {}
    for fid in range(dict_size):
        col = acts[:, fid]
        n_activating = int((col > 0).sum())
        if n_activating == 0:
            out[str(fid)] = {"fid": fid, "n_activating": 0, "top100_acts": []}
            continue

        top_idx = np.argsort(-col)[:TOP_N]
        top_acts = col[top_idx].tolist()

        piece_counts = Counter()
        side_white = 0
        traj_already_losing = 0
        traj_made_worse = 0
        traj_threw_winning = 0
        cp_losses = []
        motifs = Counter()
        phases = Counter()
        motif_covered = 0
        is_capture = 0
        is_check = 0
        piece_left_hanging = Counter()   # most valuable piece en prise after blunder
        best_move_piece = Counter()      # piece type of Stockfish best move
        best_move_is_capture = 0        # best move was a capture

        for i, idx in enumerate(top_idx):
            m = meta[int(idx)]
            key = keys[int(idx)]

            try:
                b = chess.Board(m["fen"])
                mv = chess.Move.from_uci(m["blunder_uci"])
                pc = b.piece_at(mv.from_square)
                if pc: piece_counts[PIECE_NAMES.get(pc.piece_type, "other")] += 1
                if b.is_capture(mv): is_capture += 1
                if b.gives_check(mv): is_check += 1
                # after the blunder: find the most valuable piece left en prise
                b2 = b.copy(); b2.push(mv)
                opp = not b.turn
                best_hanging = None; best_val = 0
                vals = {chess.QUEEN: 9, chess.ROOK: 5, chess.BISHOP: 3,
                        chess.KNIGHT: 3, chess.PAWN: 1}
                for sq in chess.SQUARES:
                    p2 = b2.piece_at(sq)
                    if p2 and p2.color == b.turn:  # our piece still on board
                        if b2.is_attacked_by(opp, sq):
                            defenders = len(b2.attackers(b.turn, sq))
                            attackers = len(b2.attackers(opp, sq))
                            if attackers > defenders:  # genuinely en prise
                                v = vals.get(p2.piece_type, 0)
                                if v > best_val:
                                    best_val = v
                                    best_hanging = PIECE_NAMES.get(p2.piece_type, "other")
                if best_hanging: piece_left_hanging[best_hanging] += 1
            except: pass

            if m.get("is_white"): side_white += 1

            en = enr.get(key, {})
            if en and not en.get("error"):
                eb = eval_num(en.get("eval_before", 0))
                ea = eval_num(en.get("eval_after", 0))
                is_white = m.get("is_white", True)
                mover_before = eb if is_white else -eb
                mover_after  = ea if is_white else -ea
                if mover_before < -150:
                    traj_already_losing += 1
                    if mover_after < mover_before:
                        traj_made_worse += 1
                elif mover_before > 150 and mover_after < 50:
                    traj_threw_winning += 1
                cp = m.get("cp_loss") or en.get("cp_loss")
                if cp is not None: cp_losses.append(float(cp))
                # best move piece type — parse from best_uci if available
                best_uci = m.get("best_uci") or en.get("best_uci")
                if best_uci and len(best_uci) >= 4:
                    try:
                        bp = b.piece_at(chess.parse_square(best_uci[:2]))
                        if bp: best_move_piece[PIECE_NAMES.get(bp.piece_type, "other")] += 1
                        best_mv = chess.Move.from_uci(best_uci[:4])
                        if b.is_capture(best_mv): best_move_is_capture += 1
                    except: pass

            mt = motif_map.get(key)
            if mt:
                motifs[mt] += 1
                motif_covered += 1

            ph = en.get("phase") if en else None
            if ph: phases[ph] += 1

        n = len(top_idx)
        out[str(fid)] = {
            "fid": fid,
            "n_activating": n_activating,
            "top100_acts": [round(a, 4) for a in top_acts],
            "piece_types": dict(piece_counts),
            "piece_type_pct": {p: round(c/n, 3) for p,c in piece_counts.items()},
            "is_capture_pct": round(is_capture / n, 3),
            "is_check_pct": round(is_check / n, 3),
            "best_move_piece": dict(best_move_piece),
            "best_move_piece_pct": {p: round(c/n, 3) for p,c in best_move_piece.items()},
            "best_move_is_capture_pct": round(best_move_is_capture / n, 3),
            "piece_left_hanging": dict(piece_left_hanging),
            "piece_left_hanging_pct": {p: round(c/n, 3) for p,c in piece_left_hanging.items()},
            "side_white_pct": round(side_white / n, 3),
            "traj_already_losing_pct": round(traj_already_losing / n, 3),
            "traj_made_worse_pct": round(traj_made_worse / n, 3),
            "traj_threw_winning_pct": round(traj_threw_winning / n, 3),
            "cp_loss_p50": round(float(np.percentile(cp_losses, 50)), 1) if cp_losses else None,
            "cp_loss_p90": round(float(np.percentile(cp_losses, 90)), 1) if cp_losses else None,
            "cp_loss_mean": round(float(np.mean(cp_losses)), 1) if cp_losses else None,
            "motif_hist": dict(motifs.most_common(10)),
            "motif_coverage_pct": round(motif_covered / n, 3),
            "phase_hist": dict(phases),
        }

        if fid % 200 == 0:
            print(f"  {fid}/2048 features done", flush=True)

    profiles_out = args.output.replace("feature_stats", "btk_profiles")
    profiles = {}
    for fid in range(dict_size):
        col = acts[:, fid]
        top15 = np.argsort(-col)[:TOP_PROFILE]
        exs = []
        seen = set()
        for idx in top15:
            a = float(col[idx])
            if a <= 0: break
            m = meta[int(idx)]
            key = keys[int(idx)]
            if key in seen: continue
            seen.add(key)
            exs.append({"fen": m["fen"], "uci": m["blunder_uci"],
                        "best_uci": m.get("best_uci"), "cp_loss": m.get("cp_loss"),
                        "act": round(a, 4), "key": key})
        profiles[str(fid)] = {"examples": exs, "fire_rate": float((col > 0).mean())}
    with open(profiles_out, "w") as f:
        json.dump(profiles, f)
    print(f"Profiles saved to {profiles_out}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f"Stats saved to {args.output} ({len(out)} features)")


if __name__ == "__main__":
    main()
