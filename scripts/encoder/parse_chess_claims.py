"""Parse chess claims from model output for Stockfish verification.

Handles Stage A structured output format:
  "Best: Nxd5 (+2.1). Played: Qd7. Classification: blunder. Line: Nxd5 exd5 Bxd5."

Also handles natural text claims for Stage B evaluation.

No NLP. Just regex + python-chess disambiguation.
"""
import re
import chess


def parse_best_move(text):
    """Extract best move claim."""
    patterns = [
        r'Best:\s*(\S+)',
        r'best move (?:is|here is|would be)\s+\*?\*?(\S+)',
        r'(?:should|could|must) (?:play|have played)\s+\*?\*?(\S+)',
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            move = match.group(1).rstrip('.,;:)')
            return move
    return None


def parse_eval(text):
    """Extract eval claim. Returns float."""
    # Structured: "Best: Nxd5 (+2.1)"
    match = re.search(r'Best:\s*\S+\s*\(([+\-]?\d+\.?\d*)\)', text)
    if match:
        try:
            return float(match.group(1))
        except:
            pass

    # "Eval: +1.5"
    match = re.search(r'Eval:\s*([+\-]?\d+\.?\d*)', text)
    if match:
        try:
            return float(match.group(1))
        except:
            pass

    # Natural text direction
    text_lower = text.lower()
    if any(p in text_lower for p in ['white is winning', 'decisive for white']):
        return 3.0
    if any(p in text_lower for p in ['white is clearly better', 'white is much better']):
        return 2.0
    if any(p in text_lower for p in ['white has a slight', 'white is slightly']):
        return 0.7
    if any(p in text_lower for p in ['equal', 'balanced', 'roughly equal']):
        return 0.0
    if any(p in text_lower for p in ['black has a slight', 'black is slightly']):
        return -0.7
    if any(p in text_lower for p in ['black is clearly better', 'black is much better']):
        return -2.0
    if any(p in text_lower for p in ['black is winning', 'decisive for black']):
        return -3.0

    return None


def parse_classification(text):
    """Extract classification claim."""
    match = re.search(r'Classification:\s*(\w+)', text)
    if match:
        return match.group(1).lower()
    return None


def parse_played_move(text):
    """Extract played move claim."""
    match = re.search(r'Played:\s*(\S+)', text)
    if match:
        return match.group(1).rstrip('.,;:)')
    return None


def parse_pv_line(text):
    """Extract PV line moves."""
    match = re.search(r'Line:\s*([^.]+)', text)
    if match:
        return match.group(1).strip().split()
    return []


def parse_alternatives(text):
    """Extract alternative move claims."""
    match = re.search(r'Alt:\s*(.+?)(?:\.|$)', text)
    if match:
        alts = []
        for am in re.finditer(r'(\S+)\s*\(([+\-]?\d+\.?\d*)\)', match.group(1)):
            alts.append({'move': am.group(1), 'eval': float(am.group(2))})
        return alts
    return []


def parse_squares(text):
    """Extract all chess square mentions."""
    return set(re.findall(r'\b([a-h][1-8])\b', text.lower()))


def parse_moves(text):
    """Extract all chess move mentions (SAN)."""
    san = re.findall(r'\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b', text)
    castling = re.findall(r'\b(O-O(?:-O)?)\b', text)
    return san + castling


def parse_all(text):
    """Parse all chess claims from text."""
    return {
        'best_move': parse_best_move(text),
        'eval': parse_eval(text),
        'classification': parse_classification(text),
        'played_move': parse_played_move(text),
        'pv': parse_pv_line(text),
        'alternatives': parse_alternatives(text),
        'squares': parse_squares(text),
        'moves': parse_moves(text),
    }
