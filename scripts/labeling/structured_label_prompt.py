#!/usr/bin/env python3
"""Structured labeling prompt for SAE features.

Uses fixed 16-category taxonomy, forced structured output, contrastive examples.
Produces consistent, comparable labels across features.
"""

CATEGORIES = {
    'hanging_pieces': 'Undefended or inadequately defended pieces that can be captured',
    'overloaded_defenders': 'Pieces defending too many targets simultaneously, creating deflection opportunities',
    'forks': 'One piece attacking two or more targets simultaneously',
    'pins': 'Piece unable to move without exposing a more valuable piece behind it',
    'skewers': 'High-value piece attacked, forced to move, exposing a lower-value piece behind',
    'discovered_attacks': 'Moving a piece to reveal an attack from a piece behind it',
    'back_rank': 'Back rank mate threats or king trapped on home rank by own pieces',
    'king_safety': 'Weak pawn shelter, exposed king, castling vulnerabilities',
    'passed_pawns': 'Passed pawn creation, advancement, promotion threats, or blockade',
    'rook_endgames': 'Rook endgame technique including R+P vs R, active rook play, 7th rank',
    'pawn_endgames': 'King and pawn endgame technique, opposition, king activity',
    'checkmate_patterns': 'Mating threats, mating nets, specific checkmate patterns',
    'quiet_moves': 'Non-forcing winning moves, prophylaxis, zwischenzug, intermediate moves',
    'trapped_pieces': 'Pieces with no escape squares, boxed in by enemy or own pieces',
    'sacrifice': 'Giving up material for positional or tactical compensation',
    'other_tactics': 'En passant, zugzwang, attraction, interference, stalemate tricks',
}

PIECES = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king', 'mixed']
PHASES = ['opening', 'middlegame', 'endgame', 'all_phases']
SIDES = ['white_playing', 'black_playing', 'either_side']


def build_labeling_prompt(feature_id, positive_examples, negative_examples=None):
    """Build a structured labeling prompt for one SAE feature.

    Args:
        feature_id: Feature ID (int or str)
        positive_examples: List of dicts with keys: fen, uci, best_uci, cp_loss, strength
        negative_examples: Optional list of dicts (positions where feature does NOT fire)

    Returns:
        Prompt string for Sonnet/Haiku
    """
    cat_list = '\n'.join(f'  - {k}: {v}' for k, v in CATEGORIES.items())

    pos_text = ''
    for i, ex in enumerate(positive_examples[:15]):
        pos_text += f'{i+1}. FEN: {ex["fen"]}\n'
        pos_text += f'   Move played: {ex.get("uci", "?")}'
        if ex.get('best_uci') and ex['best_uci'] != ex.get('uci'):
            pos_text += f'  (best was: {ex["best_uci"]})'
        if ex.get('cp_loss'):
            pos_text += f'  cp_loss={ex["cp_loss"]}'
        pos_text += f'\n   Activation strength: {ex.get("strength", "?")}\n\n'

    neg_text = ''
    if negative_examples:
        neg_text = '\n=== NEGATIVE EXAMPLES (feature does NOT fire on these) ===\n'
        for i, ex in enumerate(negative_examples[:5]):
            neg_text += f'{i+1}. FEN: {ex["fen"]}  Move: {ex.get("uci", "?")}\n'
        neg_text += '\nUse these to understand what the feature is NOT detecting.\n'

    prompt = f"""Analyze SAE feature F{feature_id}. This feature fires on the chess positions below (all are blunder moves — the played move lost evaluation).

=== POSITIVE EXAMPLES (feature fires strongly on these) ===
{pos_text}
{neg_text}
=== YOUR TASK ===

Classify this feature using the structured format below. Be as SPECIFIC as possible.
Do NOT use generic labels like "tactical positions" or "multiple threats."
Instead, identify the SPECIFIC pattern: what piece, what tactic, what context.

CATEGORIES (pick exactly one primary, optionally one secondary):
{cat_list}

Respond in this EXACT JSON format — nothing else:
{{
  "primary_category": "<one of the category keys above>",
  "secondary_category": "<one of the category keys above, or null>",
  "specific_label": "<2-5 words: what SPECIFIC pattern within the category>",
  "piece_involved": "<{'/'.join(PIECES)}>",
  "game_phase": "<{'/'.join(PHASES)}>",
  "side": "<{'/'.join(SIDES)}>",
  "explanation": "<1 sentence: what distinguishes this feature from other features in the same category>",
  "confidence": "<high/medium/low>"
}}

RULES:
- "specific_label" must be 2-5 words describing the SPECIFIC variant, not the category name
  BAD: "overloaded defenders" (that's just the category)
  GOOD: "queen defending rook and back rank"
  GOOD: "knight fork with check"
  GOOD: "hanging bishop after castling"
- If positions involve a specific piece type, name it
- If positions are all in one game phase, say which
- If you can't identify a specific pattern, set confidence to "low"
- secondary_category is for features that span two themes (e.g., a fork that also creates a pin)
"""
    return prompt


def build_batch_records(profiles, feature_ids=None, negative_examples_fn=None):
    """Build Bedrock Batch JSONL records for multiple features.

    Args:
        profiles: Dict of {fid_str: {examples: [...], fire_rate: float, ...}}
        feature_ids: Optional list of feature IDs to label (default: all)
        negative_examples_fn: Optional function(fid) -> list of negative examples

    Returns:
        List of batch record dicts
    """
    records = []
    fids = feature_ids or list(profiles.keys())

    for fid_str in fids:
        prof = profiles.get(str(fid_str))
        if not prof or len(prof.get('examples', [])) < 5:
            continue

        positive = prof['examples'][:15]
        negative = negative_examples_fn(fid_str) if negative_examples_fn else None

        prompt = build_labeling_prompt(fid_str, positive, negative)

        record = {
            'recordId': f'label_{fid_str}',
            'modelInput': {
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 400,
                'messages': [{'role': 'user', 'content': prompt}],
            }
        }
        records.append(record)

    return records


# Example usage
if __name__ == '__main__':
    # Demo with fake data
    examples = [
        {'fen': 'r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P1Q1/2N5/PPPP1PPP/R1B1K1NR w KQkq - 4 4',
         'uci': 'd1g4', 'best_uci': 'd1h5', 'cp_loss': 55, 'strength': 10.3},
    ]
    print(build_labeling_prompt(42, examples))
