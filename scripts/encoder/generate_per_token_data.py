#!/usr/bin/env python3
"""Generate per-token alignment data for contrastive training.

Each position produces 77 (token_index, text_description) pairs.
Tokens 0: side. Tokens 1-64: board squares. Tokens 65+: metadata.

Usage:
    python3 generate_per_token_data.py \
        --input data/eval_positions.jsonl \
        --output data/per_token_alignment.jsonl \
        --n 200000
"""
import argparse
import json
import chess

SQUARE_ORDER = []  # Tokenizer processes rank 8→1, file a→h
for rank in range(7, -1, -1):
    for file in range(8):
        SQUARE_ORDER.append(chess.square(file, rank))

PIECE_TEXT = {
    (chess.PAWN, chess.WHITE): "white pawn",
    (chess.KNIGHT, chess.WHITE): "white knight",
    (chess.BISHOP, chess.WHITE): "white bishop",
    (chess.ROOK, chess.WHITE): "white rook",
    (chess.QUEEN, chess.WHITE): "white queen",
    (chess.KING, chess.WHITE): "white king",
    (chess.PAWN, chess.BLACK): "black pawn",
    (chess.KNIGHT, chess.BLACK): "black knight",
    (chess.BISHOP, chess.BLACK): "black bishop",
    (chess.ROOK, chess.BLACK): "black rook",
    (chess.QUEEN, chess.BLACK): "black queen",
    (chess.KING, chess.BLACK): "black king",
}


def generate_descriptions(fen: str) -> list[str]:
    """Generate 77 text descriptions for each encoder token."""
    board = chess.Board(fen)
    descs = []

    # Token 0: side to move
    descs.append("white to move" if board.turn == chess.WHITE else "black to move")

    # Tokens 1-64: board squares (rank 8→1, file a→h)
    for sq in SQUARE_ORDER:
        piece = board.piece_at(sq)
        sq_name = chess.square_name(sq)
        if piece:
            text = PIECE_TEXT.get((piece.piece_type, piece.color), "unknown piece")
            descs.append(f"{sq_name}: {text}")
        else:
            descs.append(f"{sq_name}: empty")

    # Tokens 65-68: castling
    castling = board.castling_rights
    descs.append("white kingside castle" if castling & chess.BB_H1 else "no white kingside castle")
    descs.append("white queenside castle" if castling & chess.BB_A1 else "no white queenside castle")
    descs.append("black kingside castle" if castling & chess.BB_H8 else "no black kingside castle")
    descs.append("black queenside castle" if castling & chess.BB_A8 else "no black queenside castle")

    # Tokens 69-70: en passant
    ep = board.ep_square
    if ep is not None:
        descs.append(f"en passant file: {chess.square_name(ep)[0]}")
        descs.append(f"en passant rank: {chess.square_name(ep)[1]}")
    else:
        descs.append("no en passant")
        descs.append("no en passant")

    # Tokens 71-73: halfmove clock (3 chars)
    hm = str(board.halfmove_clock)
    for c in hm.ljust(3, '.'):
        descs.append(f"halfmove: {c}")

    # Tokens 74-76: fullmove number (3 chars)
    fm = str(board.fullmove_number)
    for c in fm.ljust(3, '.'):
        descs.append(f"fullmove: {c}")

    assert len(descs) == 77, f"Expected 77, got {len(descs)}"
    return descs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n', type=int, default=200000)
    args = parser.parse_args()

    count = 0
    with open(args.output, 'w') as out:
        with open(args.input) as f:
            for line in f:
                if count >= args.n:
                    break
                item = json.loads(line.strip())
                fen = item.get('fen', '')
                if not fen:
                    continue

                parts = fen.split()
                if len(parts) == 4:
                    fen += ' 0 1'
                elif len(parts) == 5:
                    fen += ' 1'

                try:
                    descs = generate_descriptions(fen)
                    out.write(json.dumps({'fen': fen, 'descriptions': descs}) + '\n')
                    count += 1
                except Exception as e:
                    continue

                if count % 10000 == 0:
                    print(f"  {count} positions...", flush=True)

    print(f"Written {count} positions to {args.output}", flush=True)

    # Sample
    with open(args.output) as f:
        sample = json.loads(f.readline())
    print(f"\nSample (first 10 descriptions):")
    for i, d in enumerate(sample['descriptions'][:10]):
        print(f"  token {i}: {d}")


if __name__ == '__main__':
    main()
