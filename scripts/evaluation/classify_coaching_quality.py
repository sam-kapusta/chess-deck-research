#!/usr/bin/env python3
"""Classify Lichess study comments as coaching-relevant or generic.

Coaching-relevant = explains WHY a move is good/bad, teaches a concept,
discusses alternatives with reasoning, or identifies tactical/strategic themes.

Generic = opening labels, bare evaluations, move narration without insight,
humor/commentary without chess reasoning, links/references, non-English.

Usage:
  python research/scripts/classify_coaching_quality.py
  python research/scripts/classify_coaching_quality.py --input research/data/lichess_studies.jsonl
  python research/scripts/classify_coaching_quality.py --sample  # print 20 from each category
  python research/scripts/classify_coaching_quality.py --llm-script  # write LLM scoring script
"""
import json
import re
import argparse
import random
from pathlib import Path
from collections import Counter


# ---------------------------------------------------------------------------
# Coaching signal keywords — phrases that indicate the comment explains WHY
# ---------------------------------------------------------------------------
COACHING_PATTERNS = [
    # Reason / explanation
    (r'\bbecause\b', 2),
    (r'\bsince\b(?!.*(lichess|chess\.com))', 1),
    (r'\bthe (?:idea|point|reason|problem|issue|trick|key) (?:is|here|being)\b', 3),
    (r'\bin order to\b', 2),
    (r'\bso that\b', 2),
    (r'\bthis (?:is|was) (?:important|critical|crucial|necessary)\b', 2),

    # Consequence explanation
    (r'\bif\b.{3,60}\b(?:then|wins?|loses?|draws?|mates?)\b', 2),
    (r'\b(?:leads? to|results? in|allows?|prevents?|stops?)\b', 1),
    (r'\bwould (?:have been|be|allow|give|lose|win)\b', 2),
    (r'\bthreatens?\b', 1),
    (r'\bforced\b', 1),

    # Alternatives / comparison
    (r'\bbetter (?:was|is|would be|move|option|choice)\b', 3),
    (r'\binstead\b', 2),
    (r'\balternative\b', 2),
    (r'\b(?:stronger|weaker|worse|more accurate|more precise) (?:is|was|move|would)\b', 2),
    (r'\bshould (?:have|play|consider|try)\b', 2),
    (r'\bcould (?:have|also)\b', 1),

    # Strategic / positional concepts
    (r'\b(?:control|pressure|initiative|tempo|development|activity|coordination)\b', 1),
    (r'\b(?:weakness|weak square|outpost|open file|diagonal|pawn structure)\b', 1),
    (r'\b(?:king safety|castling|centrali[sz]|piece placement|space advantage)\b', 1),
    (r'\b(?:prophylaxis|overprotect|blockade|restrict)\b', 2),

    # Tactical motifs
    (r'\b(?:pin|fork|skewer|discovery|discovered|deflection|decoy|sacrifice|zwischenzug)\b', 2),
    (r'\b(?:back rank|mating (?:net|pattern|attack)|checkmate pattern)\b', 2),
    (r'\b(?:trapped piece|overloaded|double attack|x-ray)\b', 2),

    # Evaluative with reason (not just "good move")
    (r'\b(?:mistake|blunder|inaccuracy|error)\b.{0,40}\bbecause\b', 3),
    (r'\b(?:mistake|blunder|inaccuracy|error)\b', 1),
    (r'\b(?:winning|losing|advantage|disadvantage)\b.{0,40}\b(?:because|since|due to)\b', 3),

    # Imperative / teaching voice
    (r'\b(?:always|never|don.t|avoid|remember|notice|note that|observe|look at)\b', 1),
    (r'\b(?:the (?:lesson|takeaway|rule|principle) (?:is|here))\b', 3),
    (r'\bask yourself\b', 2),
    (r'\btry to\b', 1),

    # Deep analysis markers
    (r'\bvariation\b.{0,30}\b(?:leads?|gives?|wins?|loses?)\b', 2),
    (r'\bafter\b.{3,40}\b(?:white|black)\b.{3,40}\b(?:has|gets|wins?|stands?)\b', 2),
    (r'\bposition (?:is|demands?|requires?|calls for)\b', 2),
]

# ---------------------------------------------------------------------------
# Generic / non-coaching signals
# ---------------------------------------------------------------------------
GENERIC_PATTERNS = [
    # Opening labels without reasoning
    (r'^(?:this is|we (?:now )?have|this (?:is|was) (?:the|called|known as))\b.{0,80}(?:opening|variation|defense|defence|gambit|system)', 3),
    (r'\b(?:also known as|named after)\b', 2),

    # Bare evaluation
    (r'^[+-]?\d+\.\d+\s*$', 3),
    (r'stockfish (?:says?|gives?|evaluates?|approved)', 2),

    # URL / reference dumps
    (r'https?://\S+', 1),
    (r'lichess\.org/practice', 2),

    # Non-chess / humor without substance
    (r'\bJK\b|\bjust kidding\b|\blol\b|\bhaha\b', 1),
    (r'(?:wink|sorry|tired)', 1),

    # Pure narration ("and now he plays", "white plays", game result)
    (r'^(?:and )?(?:white|black) (?:plays?|moves?|resigned?)\b.{0,40}$', 2),
    (r'\bresigned\b.{0,20}$', 1),

    # Checkmate pattern labels without explanation
    (r'^(?:correct|well done|congratulations)', 2),

    # Pure move list
    (r'^\d+\.\s*[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8]', 1),
]

# Non-English detection
NON_ENGLISH_RE = re.compile(r'[àáâãäéèêëíìîïóòôõöúùûüñçßæøåÄÖÜ¿¡]')


def score_comment(comment: str) -> tuple[int, list[str]]:
    """Return (net_coaching_score, matched_categories).

    Positive = coaching, negative = generic.
    """
    coaching_score = 0
    generic_score = 0
    categories: list[str] = []

    comment_lower = comment.lower()

    # Non-English penalty
    if NON_ENGLISH_RE.search(comment):
        # Allow if it's mostly English (< 10% non-ASCII-letter chars)
        alpha = sum(1 for c in comment if c.isalpha())
        non_en = len(NON_ENGLISH_RE.findall(comment))
        if alpha > 0 and non_en / alpha > 0.05:
            generic_score += 5
            categories.append('non_english')

    # Coaching signals
    for pattern, weight in COACHING_PATTERNS:
        if re.search(pattern, comment, re.IGNORECASE):
            coaching_score += weight

    # Generic signals
    for pattern, weight in GENERIC_PATTERNS:
        if re.search(pattern, comment, re.IGNORECASE):
            generic_score += weight

    # Length bonus — longer comments with coaching signals are more valuable
    if len(comment) >= 200 and coaching_score > 0:
        coaching_score += 2
    elif len(comment) >= 300 and coaching_score > 0:
        coaching_score += 3

    # Classify coaching type
    if coaching_score > generic_score:
        # Determine primary coaching category
        if re.search(r'\b(?:pin|fork|skewer|discovery|sacrifice|deflection|decoy|zwischenzug|double attack|x-ray|back rank|trapped|overloaded)\b', comment_lower):
            categories.append('tactical_motif')
        if re.search(r'\b(?:because|since|the (?:idea|point|reason|problem))\b', comment_lower):
            categories.append('evaluative_with_reason')
        if re.search(r'\b(?:better|instead|alternative|should have|could have|more accurate)\b', comment_lower):
            categories.append('comparison_alternatives')
        if re.search(r'\bif\b.{3,60}\b(?:then|wins?|loses?|draws?|mates?)\b', comment_lower):
            categories.append('consequence_explanation')
        if re.search(r'\b(?:control|pressure|initiative|weakness|pawn structure|king safety|outpost|open file|space)\b', comment_lower):
            categories.append('strategic_explanation')
        if re.search(r'\b(?:always|never|don.t|avoid|remember|lesson|principle|rule)\b', comment_lower):
            categories.append('teaching_imperative')
        if len(comment) >= 300:
            categories.append('analytical_deep_dive')
        if not categories:
            categories.append('general_coaching')

    net = coaching_score - generic_score
    return net, categories


def classify_dataset(data: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split data into coaching and generic."""
    coaching = []
    generic = []

    for d in data:
        comment = d.get('comment', '')
        net_score, categories = score_comment(comment)

        d_out = {**d, 'coaching_score': net_score, 'coaching_categories': categories}

        if net_score >= 2:  # Threshold: need at least 2 net coaching points
            coaching.append(d_out)
        else:
            generic.append(d_out)

    return coaching, generic


def print_samples(items: list[dict], label: str, n: int = 20):
    """Print n random samples from a list."""
    sample = random.sample(items, min(n, len(items)))
    print(f"\n{'='*80}")
    print(f"  {label} SAMPLES ({n} of {len(items)})")
    print(f"{'='*80}")
    for i, d in enumerate(sample, 1):
        score = d.get('coaching_score', '?')
        cats = ', '.join(d.get('coaching_categories', []))
        comment = d['comment'][:300]
        if len(d['comment']) > 300:
            comment += '...'
        print(f"\n--- [{i}] score={score} cats=[{cats}] phase={d.get('phase','?')} ---")
        print(f"  Move: {d['move']} | Study: {d.get('study_name', '?')}")
        print(f"  {comment}")


def print_stats(coaching: list[dict], generic: list[dict]):
    """Print classification statistics."""
    total = len(coaching) + len(generic)
    print(f"\n{'='*80}")
    print(f"  CLASSIFICATION RESULTS")
    print(f"{'='*80}")
    print(f"Total:    {total}")
    print(f"Coaching: {len(coaching)} ({100*len(coaching)/total:.1f}%)")
    print(f"Generic:  {len(generic)} ({100*len(generic)/total:.1f}%)")

    # Score distribution
    c_scores = [d['coaching_score'] for d in coaching]
    g_scores = [d['coaching_score'] for d in generic]
    print(f"\nCoaching scores: min={min(c_scores)}, max={max(c_scores)}, "
          f"mean={sum(c_scores)/len(c_scores):.1f}")
    print(f"Generic scores:  min={min(g_scores)}, max={max(g_scores)}, "
          f"mean={sum(g_scores)/len(g_scores):.1f}")

    # Category distribution
    all_cats = []
    for d in coaching:
        all_cats.extend(d['coaching_categories'])
    cat_counts = Counter(all_cats)
    print(f"\nCoaching category distribution:")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count} ({100*count/len(coaching):.1f}%)")

    # Phase distribution
    for label, items in [('Coaching', coaching), ('Generic', generic)]:
        phases = Counter(d.get('phase', 'unknown') for d in items)
        print(f"\n{label} phases: {dict(phases)}")

    # Comment length distribution
    for label, items in [('Coaching', coaching), ('Generic', generic)]:
        lengths = [len(d['comment']) for d in items]
        print(f"{label} comment length: mean={sum(lengths)/len(lengths):.0f}, "
              f"median={sorted(lengths)[len(lengths)//2]}")


def write_llm_script(output_path: str):
    """Write a script that uses Bedrock Claude Haiku for borderline classification."""
    script = '''#!/usr/bin/env python3
"""LLM-based quality scoring for borderline coaching comments.

Uses Bedrock Claude Haiku to classify comments that the heuristic scorer
rated between -1 and 3 (the borderline zone).

Cost estimate for 30K comments at ~150 tokens avg:
  - Input: 30K * 250 tokens (prompt + comment) = 7.5M tokens
  - Output: 30K * 30 tokens (label + reason) = 0.9M tokens
  - Haiku pricing: $0.25/1M input, $1.25/1M output
  - Total: ~$1.88 + ~$1.13 = ~$3.00 for ALL 30K
  - Borderline only (~8K): ~$0.80

Usage:
  python research/scripts/llm_classify_borderline.py --dry-run  # cost estimate only
  python research/scripts/llm_classify_borderline.py --max 100  # test on 100
  python research/scripts/llm_classify_borderline.py             # run all borderline
"""
import json
import time
import argparse
from pathlib import Path

import boto3

PROMPT_TEMPLATE = """You are classifying chess study comments for a coaching dataset.

Rate this comment as either COACHING or GENERIC.

COACHING = explains WHY a move is good/bad, teaches a concept, discusses alternatives
with reasoning, identifies tactical/strategic themes with explanation.

GENERIC = opening name labels, bare evaluations, move narration without insight,
humor without chess reasoning, non-English, just links/references.

Comment on the move {move} in this position:
"{comment}"

Respond with exactly one line: COACHING or GENERIC, then a pipe, then a 1-sentence reason.
Example: COACHING | Explains why trading bishops weakens the dark squares.
"""


def classify_with_llm(client, comment: str, move: str, model_id: str) -> tuple[str, str]:
    """Classify a single comment using Bedrock Haiku."""
    prompt = PROMPT_TEMPLATE.format(move=move, comment=comment[:500])

    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 60, "temperature": 0.0},
    )

    text = response["output"]["message"]["content"][0]["text"].strip()
    parts = text.split("|", 1)
    label = parts[0].strip().upper()
    reason = parts[1].strip() if len(parts) > 1 else ""

    if "COACHING" in label:
        return "coaching", reason
    return "generic", reason


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="research/data/prepared/coaching_lichess.jsonl")
    parser.add_argument("--output", default="research/data/prepared/llm_classified.jsonl")
    parser.add_argument("--model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--max", type=int, default=0, help="Max comments to classify (0=all borderline)")
    parser.add_argument("--dry-run", action="store_true", help="Just estimate cost")
    parser.add_argument("--profile", default="chess-deck")
    args = parser.parse_args()

    # Load data
    data = [json.loads(l) for l in Path(args.input).read_text().strip().split("\\n")]

    # Import the heuristic scorer
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from classify_coaching_quality import score_comment

    # Find borderline comments (heuristic score between -1 and 3)
    borderline = []
    for d in data:
        score, _ = score_comment(d["comment"])
        if -1 <= score <= 3:
            borderline.append({**d, "heuristic_score": score})

    print(f"Total comments: {len(data)}")
    print(f"Borderline (-1 to 3): {len(borderline)}")

    # Cost estimate
    avg_input_tokens = 250  # prompt + comment
    avg_output_tokens = 30
    n = args.max if args.max > 0 else len(borderline)
    input_cost = n * avg_input_tokens * 0.25 / 1_000_000
    output_cost = n * avg_output_tokens * 1.25 / 1_000_000
    print(f"Estimated cost for {n} comments: ${input_cost + output_cost:.2f}")

    if args.dry_run:
        return

    # Classify
    session = boto3.Session(profile_name=args.profile)
    client = session.client("bedrock-runtime", region_name="us-east-1")

    results = []
    items = borderline[:n] if args.max > 0 else borderline
    for i, d in enumerate(items):
        label, reason = classify_with_llm(client, d["comment"], d["move"], args.model)
        d["llm_label"] = label
        d["llm_reason"] = reason
        results.append(d)

        if (i + 1) % 50 == 0:
            print(f"  Classified {i+1}/{len(items)}...")
            time.sleep(0.5)  # Rate limit courtesy

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for d in results:
            f.write(json.dumps(d) + "\\n")

    # Stats
    labels = [d["llm_label"] for d in results]
    from collections import Counter
    print(f"\\nLLM classification: {dict(Counter(labels))}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
'''
    Path(output_path).write_text(script)
    print(f"Wrote LLM scoring script to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Classify Lichess coaching quality')
    parser.add_argument('--input', default='research/data/lichess_studies.jsonl',
                        help='Raw JSONL input (default: full 30K raw dataset)')
    parser.add_argument('--prepared', default='research/data/prepared/coaching_lichess.jsonl',
                        help='Prepared 20K dataset (used if --input does not exist)')
    parser.add_argument('--output', default='research/data/prepared/coaching_only.jsonl')
    parser.add_argument('--sample', action='store_true', help='Print 20 samples from each category')
    parser.add_argument('--llm-script', action='store_true', help='Write LLM borderline scoring script')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load data — prefer raw if available (more data), fall back to prepared
    input_path = args.input
    if not Path(input_path).exists():
        input_path = args.prepared
    data = [json.loads(l) for l in Path(input_path).read_text().strip().split('\n')]
    print(f"Loaded {len(data)} pairs from {input_path}")

    # Add phase if missing
    try:
        import chess
        for d in data:
            if 'phase' not in d or d['phase'] in ('none', None):
                try:
                    board = chess.Board(d['fen'])
                    n = len(board.piece_map())
                    d['phase'] = 'opening' if n > 24 else ('middlegame' if n > 10 else 'endgame')
                except ValueError:
                    d['phase'] = 'unknown'
    except ImportError:
        print("Warning: python-chess not installed, skipping phase classification")

    # Filter non-English first
    before = len(data)
    data_en = []
    non_english_count = 0
    for d in data:
        if NON_ENGLISH_RE.search(d.get('comment', '')):
            alpha = sum(1 for c in d['comment'] if c.isalpha())
            non_en = len(NON_ENGLISH_RE.findall(d['comment']))
            if alpha > 0 and non_en / alpha > 0.05:
                non_english_count += 1
                continue
        data_en.append(d)
    print(f"Removed {non_english_count} non-English comments ({before} -> {len(data_en)})")

    # Filter very short comments (< 50 chars rarely coaching)
    data_en = [d for d in data_en if len(d.get('comment', '')) >= 50]
    print(f"After min length filter: {len(data_en)}")

    # Classify
    coaching, generic = classify_dataset(data_en)

    # Print stats
    print_stats(coaching, generic)

    if args.sample:
        print_samples(coaching, 'COACHING')
        print_samples(generic, 'GENERIC')

    # Deduplicate coaching by FEN
    seen_fens = set()
    coaching_deduped = []
    for d in coaching:
        fen_key = d['fen'].split(' ')[0]
        if fen_key not in seen_fens:
            seen_fens.add(fen_key)
            coaching_deduped.append(d)
    print(f"\nAfter FEN dedup: {len(coaching_deduped)} coaching pairs "
          f"(removed {len(coaching) - len(coaching_deduped)} duplicates)")

    # Save coaching-only dataset (strip internal scoring fields)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        for d in coaching_deduped:
            out = {k: v for k, v in d.items() if k not in ('coaching_score',)}
            f.write(json.dumps(out) + '\n')

    print(f"\nSaved {len(coaching_deduped)} coaching pairs to {out_path}")

    # Final phase balance
    phases = Counter(d.get('phase', 'unknown') for d in coaching_deduped)
    print(f"Phase balance: {dict(phases)}")

    cats = []
    for d in coaching_deduped:
        cats.extend(d.get('coaching_categories', []))
    cat_counts = Counter(cats)
    print(f"Category distribution:")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count} ({100*count/len(coaching_deduped):.1f}%)")

    # Write LLM script if requested
    if args.llm_script:
        write_llm_script('research/scripts/llm_classify_borderline.py')


if __name__ == '__main__':
    main()
