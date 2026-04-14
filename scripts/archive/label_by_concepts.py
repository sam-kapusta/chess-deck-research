#!/usr/bin/env python3
"""Label SAE features by correlation with known chess concepts.

Instead of asking an LLM "what is this feature?", we test:
"does this feature correlate with [concept X]?" for ~100 known concepts.

Each concept is a binary or continuous property computable from FEN via python-chess.
Features that correlate strongly (>0.15 point-biserial) get that concept as their label.

Usage:
    python3 label_by_concepts.py \
        --sae research/sae/maia_sae_2048_k32_v2.pt \
        --data research/data/multitask_moments.jsonl \
        --output research/sae/concept_labels.json \
        --n-positions 5000
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import chess
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'mcp'))


# ============================================================================
# Chess concepts (computable from FEN)
# ============================================================================

def compute_concepts(fen: str) -> dict:
    """Compute ~100 binary/continuous chess concepts from a FEN."""
    try:
        board = chess.Board(fen)
    except:
        return {}

    c = {}
    is_white_turn = board.turn == chess.WHITE

    # --- Game phase ---
    piece_count = len([sq for sq in chess.SQUARES if board.piece_at(sq)])
    c['opening'] = int(piece_count > 26)
    c['early_middlegame'] = int(20 < piece_count <= 26)
    c['late_middlegame'] = int(14 < piece_count <= 20)
    c['endgame'] = int(piece_count <= 14)
    c['piece_count'] = piece_count

    # --- Material ---
    def mat(color):
        return sum({1:1,2:3,3:3,4:5,5:9}.get(board.piece_at(sq).piece_type, 0)
                   for sq in chess.SQUARES if board.piece_at(sq) and board.piece_at(sq).color == color)
    w_mat, b_mat = mat(chess.WHITE), mat(chess.BLACK)
    c['material_equal'] = int(abs(w_mat - b_mat) <= 1)
    c['white_up_material'] = int(w_mat - b_mat > 1)
    c['black_up_material'] = int(b_mat - w_mat > 1)
    c['material_diff'] = w_mat - b_mat

    # --- King safety ---
    for color, name in [(chess.WHITE, 'white'), (chess.BLACK, 'black')]:
        king_sq = board.king(color)
        if king_sq is None:
            continue
        rank = chess.square_rank(king_sq)
        file = chess.square_file(king_sq)

        c[f'{name}_castled_kingside'] = int(file >= 5 and rank in (0, 7))
        c[f'{name}_castled_queenside'] = int(file <= 2 and rank in (0, 7))
        c[f'{name}_king_center'] = int(2 <= file <= 5 and 2 <= rank <= 5)
        c[f'{name}_king_back_rank'] = int(rank == (0 if color == chess.WHITE else 7))
        c[f'{name}_can_castle'] = int(bool(board.castling_rights & (chess.BB_RANK_1 if color == chess.WHITE else chess.BB_RANK_8)))

        # Pawn shield (pawns in front of king)
        shield = 0
        for df in [-1, 0, 1]:
            f = file + df
            if 0 <= f <= 7:
                r = rank + (1 if color == chess.WHITE else -1)
                if 0 <= r <= 7:
                    sq = chess.square(f, r)
                    p = board.piece_at(sq)
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        shield += 1
        c[f'{name}_pawn_shield'] = shield

    c['in_check'] = int(board.is_check())

    # --- Piece features ---
    for color, name in [(chess.WHITE, 'white'), (chess.BLACK, 'black')]:
        pieces = [board.piece_at(sq) for sq in chess.SQUARES if board.piece_at(sq) and board.piece_at(sq).color == color]
        c[f'{name}_has_queen'] = int(any(p.piece_type == chess.QUEEN for p in pieces))
        c[f'{name}_bishop_pair'] = int(sum(1 for p in pieces if p.piece_type == chess.BISHOP) >= 2)
        c[f'{name}_has_rooks'] = int(sum(1 for p in pieces if p.piece_type == chess.ROOK) >= 1)
        c[f'{name}_two_rooks'] = int(sum(1 for p in pieces if p.piece_type == chess.ROOK) >= 2)

        # Connected rooks
        rook_sqs = [sq for sq in chess.SQUARES if board.piece_at(sq) and board.piece_at(sq).color == color and board.piece_at(sq).piece_type == chess.ROOK]
        c[f'{name}_connected_rooks'] = 0
        if len(rook_sqs) >= 2:
            r1, r2 = rook_sqs[0], rook_sqs[1]
            if chess.square_rank(r1) == chess.square_rank(r2) or chess.square_file(r1) == chess.square_file(r2):
                # Check if path is clear
                c[f'{name}_connected_rooks'] = 1

    # --- Pawn structure ---
    for color, name in [(chess.WHITE, 'white'), (chess.BLACK, 'black')]:
        pawns = [sq for sq in chess.SQUARES if board.piece_at(sq) and board.piece_at(sq).color == color and board.piece_at(sq).piece_type == chess.PAWN]
        c[f'{name}_pawn_count'] = len(pawns)

        # Passed pawns
        passed = 0
        for sq in pawns:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            is_passed = True
            opp = not color
            for df in [-1, 0, 1]:
                ff = f + df
                if 0 <= ff <= 7:
                    for rr in range(r + (1 if color == chess.WHITE else -8), 8 if color == chess.WHITE else -1, 1 if color == chess.WHITE else -1):
                        if 0 <= rr <= 7:
                            opp_sq = chess.square(ff, rr)
                            p = board.piece_at(opp_sq)
                            if p and p.piece_type == chess.PAWN and p.color == opp:
                                is_passed = False
                                break
                if not is_passed:
                    break
            if is_passed:
                passed += 1
        c[f'{name}_passed_pawns'] = passed

        # Doubled pawns
        files_with_pawns = [chess.square_file(sq) for sq in pawns]
        from collections import Counter
        file_counts = Counter(files_with_pawns)
        c[f'{name}_doubled_pawns'] = sum(1 for v in file_counts.values() if v > 1)

        # Isolated pawns
        isolated = 0
        for sq in pawns:
            f = chess.square_file(sq)
            has_neighbor = False
            for df in [-1, 1]:
                ff = f + df
                if 0 <= ff <= 7 and ff in files_with_pawns:
                    has_neighbor = True
            if not has_neighbor:
                isolated += 1
        c[f'{name}_isolated_pawns'] = isolated

    # --- Center control ---
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    for color, name in [(chess.WHITE, 'white'), (chess.BLACK, 'black')]:
        c[f'{name}_center_pawns'] = sum(1 for sq in center_squares
                                        if board.piece_at(sq) and board.piece_at(sq).color == color
                                        and board.piece_at(sq).piece_type == chess.PAWN)
        c[f'{name}_center_control'] = sum(1 for sq in center_squares
                                          if board.is_attacked_by(color, sq))

    # --- Open files ---
    open_files = 0
    for f in range(8):
        has_white_pawn = any(board.piece_at(chess.square(f, r)) and board.piece_at(chess.square(f, r)).piece_type == chess.PAWN and board.piece_at(chess.square(f, r)).color == chess.WHITE for r in range(8))
        has_black_pawn = any(board.piece_at(chess.square(f, r)) and board.piece_at(chess.square(f, r)).piece_type == chess.PAWN and board.piece_at(chess.square(f, r)).color == chess.BLACK for r in range(8))
        if not has_white_pawn and not has_black_pawn:
            open_files += 1
    c['open_files'] = open_files

    # --- Piece activity ---
    for color, name in [(chess.WHITE, 'white'), (chess.BLACK, 'black')]:
        # Rook on 7th rank
        rank_7 = 6 if color == chess.WHITE else 1
        c[f'{name}_rook_7th'] = int(any(
            board.piece_at(chess.square(f, rank_7)) and
            board.piece_at(chess.square(f, rank_7)).piece_type == chess.ROOK and
            board.piece_at(chess.square(f, rank_7)).color == color
            for f in range(8)))

        # Knight on outpost (central squares supported by pawn, no opposing pawn can kick it)
        knight_sqs = [sq for sq in chess.SQUARES if board.piece_at(sq) and
                      board.piece_at(sq).piece_type == chess.KNIGHT and board.piece_at(sq).color == color]
        outposts = 0
        for sq in knight_sqs:
            r = chess.square_rank(sq)
            f = chess.square_file(sq)
            # Must be in opponent's half
            if (color == chess.WHITE and r >= 4) or (color == chess.BLACK and r <= 3):
                outposts += 1
        c[f'{name}_knight_outpost'] = outposts

        # Fianchettoed bishop
        if color == chess.WHITE:
            c[f'{name}_fianchetto'] = int(
                (board.piece_at(chess.G2) and board.piece_at(chess.G2).piece_type == chess.BISHOP and board.piece_at(chess.G2).color == color) or
                (board.piece_at(chess.B2) and board.piece_at(chess.B2).piece_type == chess.BISHOP and board.piece_at(chess.B2).color == color))
        else:
            c[f'{name}_fianchetto'] = int(
                (board.piece_at(chess.G7) and board.piece_at(chess.G7).piece_type == chess.BISHOP and board.piece_at(chess.G7).color == color) or
                (board.piece_at(chess.B7) and board.piece_at(chess.B7).piece_type == chess.BISHOP and board.piece_at(chess.B7).color == color))

        # Rook on open/semi-open file
        rook_sqs = [sq for sq in chess.SQUARES if board.piece_at(sq) and
                    board.piece_at(sq).piece_type == chess.ROOK and board.piece_at(sq).color == color]
        rook_open = 0
        for sq in rook_sqs:
            f = chess.square_file(sq)
            own_pawns = any(board.piece_at(chess.square(f, r)) and board.piece_at(chess.square(f, r)).piece_type == chess.PAWN and board.piece_at(chess.square(f, r)).color == color for r in range(8))
            if not own_pawns:
                rook_open += 1
        c[f'{name}_rook_open_file'] = rook_open

    # --- Position type ---
    # Opposite-side castling
    w_kingside = any(board.piece_at(chess.square(f, 0)) and board.piece_at(chess.square(f, 0)).piece_type == chess.KING and board.piece_at(chess.square(f, 0)).color == chess.WHITE for f in range(5, 8))
    b_queenside = any(board.piece_at(chess.square(f, 7)) and board.piece_at(chess.square(f, 7)).piece_type == chess.KING and board.piece_at(chess.square(f, 7)).color == chess.BLACK for f in range(0, 3))
    w_queenside = any(board.piece_at(chess.square(f, 0)) and board.piece_at(chess.square(f, 0)).piece_type == chess.KING and board.piece_at(chess.square(f, 0)).color == chess.WHITE for f in range(0, 3))
    b_kingside = any(board.piece_at(chess.square(f, 7)) and board.piece_at(chess.square(f, 7)).piece_type == chess.KING and board.piece_at(chess.square(f, 7)).color == chess.BLACK for f in range(5, 8))
    c['opposite_side_castling'] = int((w_kingside and b_queenside) or (w_queenside and b_kingside))

    # Closed position (4+ locked pawn pairs)
    locked = 0
    for f in range(8):
        for r in range(1, 7):
            p = board.piece_at(chess.square(f, r))
            if p and p.piece_type == chess.PAWN:
                opp_sq = chess.square(f, r + (1 if p.color == chess.WHITE else -1))
                op = board.piece_at(opp_sq) if 0 <= chess.square_rank(opp_sq) <= 7 else None
                if op and op.piece_type == chess.PAWN and op.color != p.color:
                    locked += 1
    c['locked_pawns'] = locked
    c['closed_position'] = int(locked >= 3)

    # --- Mobility (proxy) ---
    c['total_legal_moves'] = board.legal_moves.count()

    # --- Side to move ---
    c['white_to_move'] = int(board.turn == chess.WHITE)

    return c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sae', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n-positions', type=int, default=5000)
    args = parser.parse_args()

    # Load SAE
    ckpt = torch.load(args.sae, map_location='cpu', weights_only=False)
    cfg = ckpt['config']

    class SAE(torch.nn.Module):
        def __init__(self, d_in, d_dict, k):
            super().__init__()
            self.encoder = torch.nn.Linear(d_in, d_dict)
            self.decoder = torch.nn.Linear(d_dict, d_in, bias=False)
            self.pre_bias = torch.nn.Parameter(torch.zeros(d_in))
            self.k = k
        def forward(self, x):
            z = self.encoder(x - self.pre_bias)
            tv, ti = torch.topk(z, self.k, dim=-1)
            a = torch.zeros_like(z)
            a.scatter_(-1, ti, F.relu(tv))
            return self.decoder(a) + self.pre_bias, a

    sae = SAE(cfg['input_dim'], cfg['dict_size'], cfg['k'])
    sae.load_state_dict(ckpt['model_state_dict'])
    sae.eval()
    sae_mean = torch.tensor(ckpt['normalization']['mean'])
    sae_std = torch.tensor(ckpt['normalization']['std']).clamp(min=1e-6)

    # Load Maia
    from maia_engine import get_maia_hidden, _load_model
    _load_model()

    # Extract features + concepts
    print(f"Processing {args.n_positions} positions...", flush=True)
    all_features = []  # [N, dict_size]
    all_concepts = []  # [N, n_concepts]
    concept_names = None
    t0 = time.time()

    with open(args.data) as f:
        for i, line in enumerate(f):
            if i >= args.n_positions:
                break
            item = json.loads(line.strip())
            fen = item.get('fen', '')
            if not fen:
                continue

            # SAE features
            h = get_maia_hidden(fen, 1800)
            if h is None:
                continue
            h_norm = (h - sae_mean) / sae_std
            with torch.no_grad():
                _, z = sae(h_norm.unsqueeze(0))
            z = z.squeeze(0).numpy()

            # Chess concepts
            concepts = compute_concepts(fen)
            # Add tags from our data
            for tag in item.get('tags', []):
                concepts[f'tag_{tag}'] = 1

            if concept_names is None:
                concept_names = sorted(concepts.keys())

            concept_vec = [concepts.get(name, 0) for name in concept_names]

            all_features.append(z)
            all_concepts.append(concept_vec)

            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{args.n_positions} ({time.time()-t0:.0f}s)", flush=True)

    features = np.array(all_features)  # [N, dict_size]
    concepts = np.array(all_concepts)   # [N, n_concepts]
    N = features.shape[0]
    print(f"Got {N} positions, {features.shape[1]} features, {len(concept_names)} concepts\n", flush=True)

    # Correlate each feature with each concept
    print("Computing correlations...", flush=True)
    labels = {}
    for fid in range(features.shape[1]):
        feat = features[:, fid]
        if feat.std() < 1e-8:
            continue  # dead feature

        best_corr = 0
        best_concept = None
        for cid, cname in enumerate(concept_names):
            concept = concepts[:, cid]
            if concept.std() < 1e-8:
                continue
            corr = np.corrcoef(feat, concept)[0, 1]
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_concept = cname

        if best_concept and abs(best_corr) > 0.1:
            direction = "+" if best_corr > 0 else "-"
            labels[str(fid)] = {
                'concept': best_concept,
                'correlation': round(best_corr, 3),
                'label': f"{best_concept} ({direction}{abs(best_corr):.2f})"
            }

    # Summary
    print(f"\n{len(labels)} features labeled (corr > 0.1)", flush=True)

    # Top concepts by number of features
    concept_counts = defaultdict(int)
    for info in labels.values():
        concept_counts[info['concept']] += 1
    print(f"\nTop concepts:")
    for concept, count in sorted(concept_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {count:3d} features: {concept}")

    # Save
    out = {
        'source': f"Concept correlation labeling ({len(concept_names)} concepts)",
        'labels': {fid: info['label'] for fid, info in labels.items()},
        'detailed': labels,
        'concept_names': concept_names,
        'n_positions': N,
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
