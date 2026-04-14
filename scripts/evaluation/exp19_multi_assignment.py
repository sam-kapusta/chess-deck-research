#!/usr/bin/env python3
"""Experiment 19: Multi-assignment coaching taxonomy.

Hypothesis: Features can be meaningfully assigned to multiple coaching categories.
Prediction: >40% of tactical features belong to 2+ categories, and the secondary
            category adds coaching value (not just noise).

Method: For each feature, check which coaching themes its label+explanation match.
Use keyword/semantic matching against a curated coaching taxonomy.
"""
import json
import re
import numpy as np
from collections import Counter, defaultdict


# Curated coaching taxonomy — what a player should practice
COACHING_TAXONOMY = {
    'hanging_material': {
        'description': 'Leaving pieces undefended or inadequately defended',
        'keywords': ['hanging', 'undefended', 'unprotected', 'en prise', 'left piece',
                     'inadequately defended', 'loose piece', 'free piece', 'capture.*free'],
        'coaching_question': 'Am I leaving anything undefended?',
    },
    'overloaded_defenders': {
        'description': 'Pieces defending too many things at once',
        'keywords': ['overloaded', 'overworked', 'deflection', 'defender.*multiple',
                     'defending.*two', 'double duty', 'stretched'],
        'coaching_question': 'Is any defender doing too much?',
    },
    'forks_double_attacks': {
        'description': 'Attacking two or more pieces/targets simultaneously',
        'keywords': ['fork', 'double attack', 'two.*targets', 'simultaneous.*threat',
                     'multiple.*threat', 'attacking.*two', 'knight.*fork'],
        'coaching_question': 'Can I attack two things at once?',
    },
    'pins_skewers': {
        'description': 'Restricting piece movement through alignment',
        'keywords': ['pin', 'skewer', 'alignment', 'x-ray', 'discovered',
                     'battery', 'absolute pin', 'relative pin'],
        'coaching_question': 'Are any pieces aligned for a pin/skewer?',
    },
    'king_safety': {
        'description': 'Attacks on the king, mating threats, weak shelter',
        'keywords': ['king.*safe', 'king.*attack', 'checkmate', 'mating', 'back rank',
                     'king.*exposed', 'pawn.*shelter', 'king.*danger', 'check'],
        'coaching_question': 'Is my king safe? Is their king vulnerable?',
    },
    'piece_activity': {
        'description': 'Improving piece placement, development, coordination',
        'keywords': ['develop', 'activity', 'centrali', 'improve.*piece', 'piece.*place',
                     'coordination', 'active.*piece', 'passive', 'trapped'],
        'coaching_question': 'Are all my pieces doing something useful?',
    },
    'pawn_play': {
        'description': 'Pawn breaks, passed pawns, pawn structure',
        'keywords': ['passed pawn', 'pawn break', 'pawn structure', 'pawn chain',
                     'promotion', 'advance.*pawn', 'push.*pawn', 'pawn.*weak',
                     'isolated pawn', 'doubled pawn', 'backward pawn'],
        'coaching_question': 'What should my pawns be doing?',
    },
    'forcing_moves': {
        'description': 'Checks, captures, and threats that limit opponent options',
        'keywords': ['forcing', 'check', 'capture', 'threat', 'zwischenzug',
                     'intermediate', 'in-between', 'tempo'],
        'coaching_question': 'Am I considering all checks, captures, and threats?',
    },
    'calculation': {
        'description': 'Seeing deeper variations, avoiding blunders from miscalculation',
        'keywords': ['calculat', 'variation', 'tactic.*complex', 'sequence',
                     'combination', 'sacrifice.*follow', 'deeper'],
        'coaching_question': 'Did I calculate far enough?',
    },
    'endgame_technique': {
        'description': 'Specific endgame knowledge and technique',
        'keywords': ['endgame', 'king.*pawn', 'rook.*endgame', 'queen.*rook',
                     'opposition', 'zugzwang', 'fortress', 'conversion',
                     'bishop.*knight.*endgame', 'theoretical'],
        'coaching_question': 'Do I know the right technique for this endgame?',
    },
}


def match_categories(label, explanation):
    """Return list of (category, match_strength) for a feature."""
    text = (label + ' ' + explanation).lower()
    matches = []
    for cat, info in COACHING_TAXONOMY.items():
        score = 0
        for kw in info['keywords']:
            if re.search(kw, text):
                score += 1
        if score > 0:
            matches.append((cat, score))
    matches.sort(key=lambda x: -x[1])
    return matches


def main():
    print('Experiment 19: Multi-assignment coaching taxonomy')
    print('Hypothesis: Features meaningfully belong to 2+ coaching categories')
    print('Prediction: >40% of tactical features have 2+ category matches')
    print()

    with open('/home/ec2-user/SageMaker/chess-deck-research/output/labels_blunder_mt_k32.json') as f:
        labels = json.load(f)

    # Filter to quality features
    quality_fids = []
    for fid_str, info in labels.items():
        if info.get('confidence') in ['high', 'medium']:
            quality_fids.append(fid_str)

    print('Quality features: ' + str(len(quality_fids)))
    print('Taxonomy categories: ' + str(len(COACHING_TAXONOMY)))
    print()

    # Assign categories
    assignments = {}
    category_counts = Counter()
    n_multi = 0
    n_zero = 0

    for fid_str in quality_fids:
        info = labels[fid_str]
        matches = match_categories(info.get('label', ''), info.get('explanation', ''))
        assignments[fid_str] = matches

        if len(matches) == 0:
            n_zero += 1
        elif len(matches) >= 2:
            n_multi += 1

        for cat, score in matches:
            category_counts[cat] += 1

    total = len(quality_fids)
    print('=== Assignment distribution ===')
    print('  0 categories: ' + str(n_zero) + ' (' + str(round(n_zero / total * 100, 1)) + '%)')
    print('  1 category:   ' + str(total - n_multi - n_zero) + ' (' + str(round((total - n_multi - n_zero) / total * 100, 1)) + '%)')
    print('  2+ categories: ' + str(n_multi) + ' (' + str(round(n_multi / total * 100, 1)) + '%)')
    print()

    # Distribution of matches per feature
    match_counts = [len(assignments[f]) for f in quality_fids]
    for n in range(5):
        count = sum(1 for m in match_counts if m == n)
        if count > 0:
            print('  ' + str(n) + ' matches: ' + str(count))

    print()
    print('=== Category sizes ===')
    for cat, count in category_counts.most_common():
        print('  ' + cat + ': ' + str(count) + ' features')

    # For multi-assigned features, show the primary+secondary combinations
    print()
    print('=== Top primary+secondary combos ===')
    combos = Counter()
    for fid_str in quality_fids:
        matches = assignments[fid_str]
        if len(matches) >= 2:
            combos[(matches[0][0], matches[1][0])] += 1

    for (primary, secondary), count in combos.most_common(15):
        print('  ' + primary + ' + ' + secondary + ': ' + str(count))

    # Show examples of multi-assigned features
    print()
    print('=== Multi-assigned examples ===')
    shown = 0
    for fid_str in quality_fids:
        matches = assignments[fid_str]
        if len(matches) >= 2:
            info = labels[fid_str]
            cats_str = ' + '.join(c + '(' + str(s) + ')' for c, s in matches)
            print('  F' + fid_str + ': ' + info.get('label', '')[:50])
            print('    → ' + cats_str)
            shown += 1
            if shown >= 10:
                break

    # For unassigned features, check what they are
    print()
    print('=== Unassigned features (sample) ===')
    shown = 0
    for fid_str in quality_fids:
        if len(assignments[fid_str]) == 0:
            info = labels[fid_str]
            print('  F' + fid_str + ' [' + info.get('category', '?') + '] ' + info.get('label', '')[:60])
            shown += 1
            if shown >= 10:
                break

    # Quality check: does secondary category add information?
    # Compare: "primary only" labels vs "primary + secondary" — does overlap make sense?
    print()
    print('=== Secondary category coherence ===')
    print('For the top combos, are the secondary assignments coaching-meaningful?')
    for (primary, secondary), count in combos.most_common(5):
        print()
        print(primary + ' + ' + secondary + ' (' + str(count) + ' features):')
        print('  Primary question: ' + COACHING_TAXONOMY[primary]['coaching_question'])
        print('  Secondary question: ' + COACHING_TAXONOMY[secondary]['coaching_question'])
        # Show 2 examples
        shown = 0
        for fid_str in quality_fids:
            matches = assignments[fid_str]
            if len(matches) >= 2 and matches[0][0] == primary and matches[1][0] == secondary:
                print('  Example: ' + labels[fid_str].get('label', '')[:60])
                shown += 1
                if shown >= 2:
                    break

    # Verdict
    pct_multi = n_multi / total * 100
    print()
    print('=== Verdict ===')
    print('Features with 2+ categories: ' + str(round(pct_multi, 1)) + '%')
    print('Prediction was >40%: ' + ('CONFIRMED' if pct_multi > 40 else 'FAILED'))


if __name__ == '__main__':
    main()
