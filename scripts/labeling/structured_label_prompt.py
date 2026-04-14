#!/usr/bin/env python3
"""Structured labeling prompts for chess SAE features.

Two modes:
1. BATCH (thorough): System prompt + 10 positive + 10 negative examples + analysis framework.
   ~30-60 sec per feature. For building labels.json.
2. GAME (fast): Short user prompt, LABEL/CATEGORY/PIECE format, maxTokens=40.
   ~3 sec per feature. For per-game eval.

Adapted from Sandstone persona pipeline (auto_interp_prompts.py).
"""

# ══════════════════════════════════════════════════════════════════════
# CATEGORIES
# ══════════════════════════════════════════════════════════════════════

CATEGORIES = {
    'hanging_pieces': 'Undefended or inadequately defended pieces that can be captured for free',
    'overloaded_defenders': 'Pieces defending too many targets — deflection, removing the guard',
    'forks': 'One piece attacking two or more targets simultaneously (any piece, not just knights)',
    'pins': 'Piece unable to move without exposing a more valuable piece behind it',
    'skewers': 'High-value piece attacked and forced to move, exposing lower-value piece behind',
    'discovered_attacks': 'Moving a piece to reveal an attack from a piece behind it (including discovered check)',
    'back_rank': 'Back rank mate threats, king trapped on home rank by own pieces',
    'king_safety': 'Weak pawn shelter, exposed king, castling issues, king walking into danger',
    'checkmate_patterns': 'Mating threats, mating nets, forced checkmate sequences',
    'overloaded_defenders': 'Pieces defending too many targets — deflection, removing the guard',
    'quiet_moves': 'Non-forcing winning moves, prophylaxis, zwischenzug, intermediate moves',
    'trapped_pieces': 'Pieces with no escape squares, boxed in, restricted mobility',
    'sacrifice': 'Giving up material for positional or tactical compensation',
    'passed_pawns': 'Passed pawn creation, advancement, promotion threats, blockade failures',
    'rook_endgames': 'Rook endgame technique — R+P vs R, active rook, 7th rank, Lucena/Philidor',
    'pawn_endgames': 'King and pawn endgame technique — opposition, king activity, pawn races',
    'other': 'Patterns not fitting other categories (en passant, zugzwang, stalemate tricks)',
}

PIECES = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king', 'mixed']


# ══════════════════════════════════════════════════════════════════════
# BATCH MODE — Thorough labeling (for labels.json)
# ══════════════════════════════════════════════════════════════════════

BATCH_SYSTEM_PROMPT = """You are an expert at interpreting sparse autoencoder (SAE) features learned from a chess encoder. Each feature corresponds to a direction in the encoder's hidden space that activates on specific chess move patterns — a recognizable tactical or positional concept defined by the type of mistake or opportunity it detects.

## Task Overview
You will be given:
- 10 POSITIVE examples: blunder positions where this feature activates strongly
- 10 NEGATIVE examples: blunder positions where this feature does NOT activate
- For each position: FEN, the blunder move played, the best move, and cp loss

Your goal is to identify the **chess pattern** this feature encodes. A chess pattern is a specific tactical or positional concept — the *type of mistake* these positions share, not just surface-level similarities.

## Granularity Guide

The right interpretation is a pattern a chess coach would immediately recognize as a specific lesson topic.

TOO BROAD (would match >15% of all blunder positions — reject these):
  - "Tactical mistakes"
  - "Piece left undefended"
  - "Bad move in middlegame"
  - "Missing better move"
  - "Overloaded position"

RIGHT LEVEL (would match roughly 1-10% of positions — aim for these):
  - "Knight fork with check winning queen"
  - "King walks into discovered check"
  - "Rook passivity instead of 7th rank invasion"
  - "Queen abandons back rank defense"
  - "Pawn advance weakening king shelter"

TOO NARROW (would match <0.1% of positions — reject these):
  - "Nf3+ forking Ke1 and Qh8 in Vienna Game"
  - "g2-g3 instead of d2-d4 on move 13"
  - "Bishop retreating to e7 specifically"

## Exclusions — Do NOT interpret the feature as:
- Generic mistake severity ("big blunder", "small inaccuracy")
- Game phase alone ("opening mistake", "endgame error")
- Move type alone ("capture", "check", "pawn move")
- Evaluation-based ("losing move", "missing winning move")

If the positives share one of these, look deeper for the SPECIFIC tactical pattern.

## Using Activation Strength
Customers with highest activations are purest examples. Form hypothesis from top-3 first. Lower-activation positives may be noisier — don't weaken interpretation to accommodate noise.

## Analysis Framework

### Step 1: Position Fingerprint
Before forming any hypothesis, tabulate:
1. What piece is being moved in the blunder? (across all positives)
2. What tactical/positional theme appears? (hanging pieces, pins, forks, king safety, endgame technique)
3. What's the game phase? (opening/middlegame/endgame)
4. What do the BEST moves have in common that the BLUNDERS don't?
5. What appears in BOTH positive and negative examples? (not discriminative — ignore these)

### Step 2: Hypothesis
From highest-activation positives, identify the specific chess pattern:
- What *type of mistake* are these? What tactical concept explains all of them?
- Look across categories — diapers + baby food = parent. Hanging knight + hanging bishop = "leaving pieces undefended"

### Step 3: Discriminative Validation
- Verify hypothesis fits ≥7/10 positives
- Verify it does NOT fit negatives
- If it fits both: too broad — narrow it

### Step 4: Granularity Check
- Would a chess coach assign this as a specific lesson?
- ~1-10% of blunder positions?
- Describing a *type of mistake* not just *a move*?

## Output Format
Respond with ONLY this JSON, no other text:

{
  "label": "2-5 word specific label",
  "category": "<CATEGORY>",
  "description": "One sentence: what specific chess pattern this feature detects",
  "piece": "<PIECE>",
  "phase": "opening|middlegame|endgame|any",
  "confidence": 0.XX,
  "estimated_frequency_pct": "X-Y%",
  "reasoning": {
    "position_fingerprint": {
      "common_piece_moved": "piece type in blunders",
      "common_theme": "tactical/positional theme",
      "common_phase": "game phase",
      "best_move_pattern": "what the best moves share",
      "non_discriminative": "what appears in both pos and neg"
    },
    "hypothesis": "What type of mistake are these",
    "positive_evidence": "X/10 positives fit because...",
    "negative_exclusion": "Why negatives don't fit",
    "granularity_check": "Why this is at the right level"
  }
}

CATEGORIES: """ + ', '.join(CATEGORIES.keys()) + """
PIECES: """ + ', '.join(PIECES)


def build_batch_prompt(feature_id, positive_examples, negative_examples=None):
    """Build the user message for batch labeling.

    Args:
        feature_id: Feature ID
        positive_examples: List of dicts with fen, uci/blunder, best_uci, cp_loss, strength
        negative_examples: List of dicts (positions where feature does NOT fire)

    Returns:
        (system_prompt, user_prompt) tuple
    """
    user = f"# Chess Pattern Interpretation: Feature {feature_id}\n\n"
    user += "## POSITIVE EXAMPLES (Feature activates strongly)\n"
    user += "Sorted by activation strength (highest = purest examples).\n\n"

    for i, ex in enumerate(positive_examples[:10]):
        fen = ex.get('fen', '?')
        uci = ex.get('uci', ex.get('blunder', '?'))
        best = ex.get('best_uci', ex.get('best', '?'))
        cp = ex.get('cp_loss', '?')
        strength = ex.get('strength', '?')
        user += f"### P{i+1} (Activation: {strength})\n"
        user += f"FEN: {fen}\nBlunder: {uci}  Best: {best}  CP loss: {cp}\n\n"

    if negative_examples:
        user += "\n## NEGATIVE EXAMPLES (Feature does NOT activate)\n\n"
        for i, ex in enumerate(negative_examples[:10]):
            fen = ex.get('fen', '?')
            uci = ex.get('uci', ex.get('blunder', '?'))
            best = ex.get('best_uci', ex.get('best', '?'))
            cp = ex.get('cp_loss', '?')
            user += f"### N{i+1}\n"
            user += f"FEN: {fen}\nBlunder: {uci}  Best: {best}  CP loss: {cp}\n\n"

    user += "## TASK\nAnalyze the positions above. Follow the analysis framework in your instructions. Output ONLY the JSON."

    return BATCH_SYSTEM_PROMPT, user


# ══════════════════════════════════════════════════════════════════════
# GAME MODE — Fast labeling (for per-game eval)
# ══════════════════════════════════════════════════════════════════════

GAME_LABEL_PROMPT = """Chess SAE feature on blunder position.
FEN: {fen}
Blunder: {played} ({side}). Best: {best}. Loss: {cp_loss}cp. Strength: {strength}.
Feature fires on {move_type} NOT the other.

Three lines only:
LABEL: <2-5 words>
CATEGORY: <{categories}>
PIECE: <pawn|knight|bishop|rook|queen|king|mixed>"""

GAME_LABEL_PROMPT_WITH_PROFILES = """Chess SAE feature on blunder position.
Top activations from training data:
{profile_examples}
Game context — Move {ply}: FEN: {fen}
Blunder: {played} ({side}). Best: {best}. Loss: {cp_loss}cp.
Feature fires on {move_type} NOT the other.

Three lines only:
LABEL: <2-5 words>
CATEGORY: <{categories}>
PIECE: <pawn|knight|bishop|rook|queen|king|mixed>"""


def build_game_prompt(feature_id, fen, played, best, cp_loss, strength, side, on_played,
                      profile_examples=None, ply=None):
    """Build fast labeling prompt for per-game eval.

    Returns prompt string (no system prompt — maxTokens=40 constrains output).
    """
    move_type = 'PLAYED' if on_played else 'BEST'
    categories = '|'.join(CATEGORIES.keys())

    if profile_examples:
        examples_text = ""
        for i, ex in enumerate(profile_examples[:5]):
            examples_text += f"{i+1}. FEN: {ex.get('fen','?')[:50]}  Move: {ex.get('uci', ex.get('blunder','?'))}  CP: {ex.get('cp_loss','?')}\n"
        return GAME_LABEL_PROMPT_WITH_PROFILES.format(
            profile_examples=examples_text, ply=ply or '?',
            fen=fen, played=played, best=best, cp_loss=cp_loss,
            side=side, move_type=move_type, categories=categories,
        )
    else:
        return GAME_LABEL_PROMPT.format(
            fen=fen, played=played, best=best, cp_loss=cp_loss,
            strength=strength, side=side, move_type=move_type,
            categories=categories,
        )


# ══════════════════════════════════════════════════════════════════════
# BATCH RECORDS (for Bedrock Batch API)
# ══════════════════════════════════════════════════════════════════════

def build_batch_records(profiles, feature_ids=None, negative_examples_fn=None,
                        model_id='global.anthropic.claude-sonnet-4-6'):
    """Build Bedrock Batch JSONL records.

    Args:
        profiles: Dict of {fid_str: {examples: [...], fire_rate: float, ...}}
        feature_ids: Optional list of feature IDs to label (default: all)
        negative_examples_fn: Optional function(fid) -> list of negative examples
        model_id: Bedrock model ID

    Returns:
        List of batch record dicts
    """
    records = []
    fids = feature_ids or list(profiles.keys())

    for fid_str in fids:
        prof = profiles.get(str(fid_str))
        if not prof or len(prof.get('examples', [])) < 5:
            continue

        positive = prof['examples'][:10]
        negative = negative_examples_fn(fid_str) if negative_examples_fn else None

        system, user = build_batch_prompt(fid_str, positive, negative)

        record = {
            'recordId': f'label_{fid_str}',
            'modelInput': {
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 1000,
                'system': system,
                'messages': [{'role': 'user', 'content': user}],
            }
        }
        records.append(record)

    return records


# ══════════════════════════════════════════════════════════════════════
# VALIDATION / REVISION (optional second pass)
# ══════════════════════════════════════════════════════════════════════

REVISION_SYSTEM_PROMPT = """You are validating a previous SAE feature interpretation against new examples. Test whether the interpretation generalizes, and revise if needed.

Rules:
- If >7/10 new positives fit and <2/10 new negatives fit: HOLDS — keep it, raise confidence
- If <6/10 new positives fit: TOO NARROW — abstract up to the pattern level
- If >3/10 new negatives also fit: TOO BROAD — find what specifically distinguishes positives
- If <4/10 new positives fit: FAILED — re-examine from scratch

Output ONLY JSON:
{
  "validation": "HOLDS|TOO_NARROW|TOO_BROAD|FAILED",
  "original_label": "previous label",
  "revised_label": "new label (or same if HOLDS)",
  "revised_category": "category",
  "revised_description": "updated description",
  "confidence": 0.XX,
  "coverage_new_positives": "X/10",
  "false_positive_negatives": "Y/10",
  "revision_summary": "What changed and why"
}"""


def build_revision_prompt(feature_id, original_label, original_category,
                          new_positives, new_negatives):
    """Build validation/revision prompt for second pass."""
    user = f"# Validation: Feature {feature_id}\n"
    user += f"Original interpretation: [{original_category}] {original_label}\n\n"

    user += "## NEW POSITIVE EXAMPLES\n"
    for i, ex in enumerate(new_positives[:10]):
        user += f"P{i+1}: FEN: {ex.get('fen','?')}  Blunder: {ex.get('uci','?')}  Best: {ex.get('best_uci','?')}  CP: {ex.get('cp_loss','?')}\n"

    user += "\n## NEW NEGATIVE EXAMPLES\n"
    for i, ex in enumerate(new_negatives[:10]):
        user += f"N{i+1}: FEN: {ex.get('fen','?')}  Blunder: {ex.get('uci','?')}  Best: {ex.get('best_uci','?')}  CP: {ex.get('cp_loss','?')}\n"

    user += "\nValidate and output JSON only."
    return REVISION_SYSTEM_PROMPT, user


if __name__ == '__main__':
    # Demo
    print("=== BATCH SYSTEM PROMPT (first 500 chars) ===")
    print(BATCH_SYSTEM_PROMPT[:500])
    print(f"\n... ({len(BATCH_SYSTEM_PROMPT)} chars total)")
    print(f"\n=== GAME PROMPT (example) ===")
    print(build_game_prompt(42, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",
                           "e2e4", "d2d4", 50, 3.5, "white", True))
