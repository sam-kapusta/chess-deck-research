"""Extract chess coaching concepts from the DeepMind 270M encoder.

Two types of concepts:
1. FEN-based (python-chess): material, game phase, king safety, piece activity
2. Encoder-based (neural probes): tactical volatility, attack detection, position eval

Usage:
    from concept_extractor import ConceptExtractor
    ext = ConceptExtractor("chess_encoder_270m.pt", "concept_probes.pkl")
    concepts = ext.extract("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    # Returns list of coaching-ready concept strings
"""
import pickle
import chess
import numpy as np
from pathlib import Path

try:
    import torch
    from chess_encoder import ChessEncoder, load_encoder, tokenize_fen
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ConceptExtractor:
    """Extract coaching concepts from chess positions."""

    def __init__(self, encoder_path=None, probes_path=None, device='cpu'):
        self.device = device
        self.encoder = None
        self.probes = {}

        if encoder_path and Path(encoder_path).exists() and HAS_TORCH:
            self.encoder = load_encoder(encoder_path, device)

        if probes_path and Path(probes_path).exists():
            self.probes = pickle.loads(Path(probes_path).read_bytes())

    def extract(self, fen):
        """Extract all concepts for a position."""
        concepts = self._fen_concepts(fen)
        if self.encoder:
            concepts += self._encoder_concepts(fen)
        return concepts

    def format_for_prompt(self, fen, played_move=None, best_move=None, tags=None):
        """Format concepts as text for a coaching LLM prompt."""
        concepts = self.extract(fen)
        lines = ["Position analysis:"]
        for c in concepts:
            lines.append(f"- {c}")
        if played_move:
            line = f"\nPlayer played {played_move}"
            if tags:
                line += f" (patterns: {', '.join(tags[:3])})"
            lines.append(line + ".")
        if best_move:
            lines.append(f"Engine recommends {best_move} instead.")
        return '\n'.join(lines)

    def _fen_concepts(self, fen):
        """Concepts from FEN parsing (no encoder needed)."""
        try:
            board = chess.Board(fen)
        except:
            return []

        concepts = []

        # Check
        if board.is_check():
            concepts.append("King is IN CHECK — must respond immediately")

        # Material
        pv = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        w = sum(pv.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.WHITE)
        b = sum(pv.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.BLACK)
        d = w - b
        if abs(d) >= 3:
            concepts.append(f"{'White' if d > 0 else 'Black'} has significant material advantage ({abs(d)} pts)")
        elif abs(d) >= 1:
            concepts.append(f"{'White' if d > 0 else 'Black'} has slight material edge ({abs(d)} pts)")
        else:
            concepts.append("Material is equal")

        # Game phase
        total = len(board.piece_map())
        if total <= 10:
            concepts.append("ENDGAME — technique and precision matter most")
        elif total <= 20:
            concepts.append("Middlegame position")
        else:
            concepts.append("Opening phase — development and center control are key")

        # King safety
        for color, name in [(chess.WHITE, "White"), (chess.BLACK, "Black")]:
            sq = board.king(color)
            if sq is None:
                continue
            rank = chess.square_rank(sq)
            home = 0 if color == chess.WHITE else 7
            if rank != home and rank != home + (1 if color == chess.WHITE else -1):
                concepts.append(f"{name}'s king is EXPOSED on {chess.square_name(sq)}")

        # Development
        if total > 20:
            for color, name in [(chess.WHITE, "White"), (chess.BLACK, "Black")]:
                back = 0 if color == chess.WHITE else 7
                undeveloped = sum(1 for pt in [chess.KNIGHT, chess.BISHOP]
                                 for sq in board.pieces(pt, color) if chess.square_rank(sq) == back)
                if undeveloped >= 2:
                    concepts.append(f"{name} has {undeveloped} undeveloped minor pieces")

        # Open files with rooks
        for color, name in [(chess.WHITE, "White"), (chess.BLACK, "Black")]:
            for sq in board.pieces(chess.ROOK, color):
                file = chess.square_file(sq)
                pawns = sum(1 for r in range(8) if board.piece_at(chess.square(file, r))
                           and board.piece_at(chess.square(file, r)).piece_type == chess.PAWN)
                if pawns == 0:
                    concepts.append(f"{name}'s rook on {chess.square_name(sq)} controls an open file")

        return concepts

    def _encoder_concepts(self, fen):
        """Concepts from encoder probes (needs encoder + probes loaded)."""
        tokens = tokenize_fen(fen).unsqueeze(0).to(self.device)
        hidden = self.encoder(tokens)
        embedding = hidden.mean(dim=1).cpu().float().numpy()[0]  # [1024]

        concepts = []

        if 'volatility' in self.probes:
            vol = self.probes['volatility'].predict(embedding.reshape(1, -1))[0]
            if vol > 3.0:
                concepts.append("CRITICAL MOMENT — this position is extremely sharp, one move changes everything")
            elif vol > 1.5:
                concepts.append("Important decision point — significant tactical tension")

        if 'attacking' in self.probes:
            prob = self.probes['attacking'].predict_proba(embedding.reshape(1, -1))[0]
            if prob[1] > 0.7:
                concepts.append("ACTIVE ATTACK available — look for forcing moves (checks, captures, threats)")
            elif prob[1] < 0.3:
                concepts.append("No immediate attacking chances — improve piece placement")

        if 'eval' in self.probes:
            ev = self.probes['eval'].predict(embedding.reshape(1, -1))[0]
            if ev > 2.0:
                concepts.append(f"White has a WINNING advantage (~{ev:.1f})")
            elif ev > 0.5:
                concepts.append(f"White has a comfortable advantage (~{ev:.1f})")
            elif ev < -2.0:
                concepts.append(f"Black has a WINNING advantage (~{abs(ev):.1f})")
            elif ev < -0.5:
                concepts.append(f"Black has a comfortable advantage (~{abs(ev):.1f})")

        return concepts
