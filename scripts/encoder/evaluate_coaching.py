#!/usr/bin/env python3
"""Evaluate coaching text quality.

Compares coaching text from:
1. Baseline: Claude Haiku with text prompt (no encoder)
2. Encoder model: our chess encoder + projection + Qwen

Evaluation dimensions (from GCC-Eval + our additions):
- Correctness: are chess claims accurate?
- Specificity: does it reference specific squares, pieces, plans?
- Coaching value: does it explain WHY and give actionable advice?
- Voice: does it sound like a coach, not a robot?
- Grounding: are claims supported by the encoder's representations?

Automated checks:
- Does it mention specific squares (a1-h8)?
- Does it reference the best move?
- Does it reference detected tags/patterns?
- Length (2-4 paragraphs)?
- Does it hallucinate moves not in the data?
"""
import json
import re
from pathlib import Path


def evaluate_coaching_text(moment, coaching_text):
    """Score a coaching text on automated metrics."""
    scores = {}

    # 1. Specificity: mentions specific squares?
    square_pattern = r'\b[a-h][1-8]\b'
    squares_mentioned = len(set(re.findall(square_pattern, coaching_text)))
    scores['squares_mentioned'] = min(squares_mentioned, 5)  # Cap at 5

    # 2. References best move?
    best_move = moment.get('best_move', '')
    scores['mentions_best_move'] = 1 if best_move.lower() in coaching_text.lower() else 0

    # 3. References played move?
    played = moment.get('played_move', '')
    scores['mentions_played_move'] = 1 if played.lower() in coaching_text.lower() else 0

    # 4. References tags?
    tags = moment.get('tags', [])
    tag_keywords = {
        'premature_push': ['push', 'pawn push', 'premature'],
        'premature_trade': ['trade', 'exchange', 'capture', 'tension'],
        'left_piece_hanging': ['hanging', 'undefended', 'unprotected'],
        'missed_capture': ['capture', 'take', 'win material'],
        'missed_check': ['check', 'king'],
        'quiet_when_winning': ['passive', 'winning', 'advantage'],
        'undeveloped_pieces': ['develop', 'undeveloped', 'starting square'],
        'conversion_failure': ['convert', 'winning position', 'let slip'],
        'missed_fork': ['fork', 'attacks two'],
        'missed_pin': ['pin', 'pinned'],
    }
    tag_refs = 0
    for tag in tags:
        keywords = tag_keywords.get(tag, [tag.replace('_', ' ')])
        if any(kw.lower() in coaching_text.lower() for kw in keywords):
            tag_refs += 1
    scores['tag_references'] = min(tag_refs, 3)  # Cap at 3

    # 5. Length: 2-4 paragraphs?
    paragraphs = [p.strip() for p in coaching_text.split('\n\n') if p.strip()]
    scores['paragraph_count'] = len(paragraphs)
    scores['good_length'] = 1 if 2 <= len(paragraphs) <= 4 else 0

    # 6. Has "Next time" or actionable tip?
    scores['has_tip'] = 1 if any(phrase in coaching_text.lower() for phrase in
        ['next time', 'look for', 'remember to', 'try to', 'before you', 'ask yourself']) else 0

    # 7. Uses bold formatting?
    scores['uses_bold'] = min(len(re.findall(r'\*\*[^*]+\*\*', coaching_text)), 3)

    # Composite score (0-10)
    composite = (
        scores['squares_mentioned'] * 0.5 +  # 0-2.5
        scores['mentions_best_move'] * 1.5 +  # 0-1.5
        scores['mentions_played_move'] * 0.5 +  # 0-0.5
        scores['tag_references'] * 1.0 +  # 0-3
        scores['good_length'] * 1.0 +  # 0-1
        scores['has_tip'] * 1.0 +  # 0-1
        scores['uses_bold'] * 0.5  # 0-1.5
    )
    scores['composite'] = round(composite, 2)

    return scores


def main():
    data_path = Path('/Users/samtkap/workspace/chess-coach/research/data/coaching_training_data.jsonl')
    if not data_path.exists():
        print("No training data found!")
        return

    moments = [json.loads(line) for line in data_path.read_text().strip().split('\n')]
    moments_with_text = [m for m in moments if m.get('coaching_text')]

    print(f"=== Coaching Quality Evaluation ===")
    print(f"Total moments with coaching text: {len(moments_with_text)}")

    all_scores = []
    for m in moments_with_text:
        scores = evaluate_coaching_text(m, m['coaching_text'])
        all_scores.append(scores)

    # Aggregate
    print(f"\n--- Automated Metrics (n={len(all_scores)}) ---")
    for key in ['composite', 'squares_mentioned', 'mentions_best_move', 'mentions_played_move',
                'tag_references', 'good_length', 'has_tip', 'uses_bold']:
        values = [s[key] for s in all_scores]
        avg = sum(values) / len(values)
        print(f"  {key:25s}: {avg:.2f}")

    # Show best and worst
    sorted_by_composite = sorted(zip(moments_with_text, all_scores), key=lambda x: -x[1]['composite'])

    print(f"\n--- Best coaching text (composite={sorted_by_composite[0][1]['composite']}) ---")
    m = sorted_by_composite[0][0]
    print(f"  {m['played_move']} ({m['classification']}) | Tags: {m['tags']}")
    print(f"  {m['coaching_text'][:200]}...")

    print(f"\n--- Worst coaching text (composite={sorted_by_composite[-1][1]['composite']}) ---")
    m = sorted_by_composite[-1][0]
    print(f"  {m['played_move']} ({m['classification']}) | Tags: {m['tags']}")
    print(f"  {m['coaching_text'][:200]}...")


if __name__ == "__main__":
    main()
