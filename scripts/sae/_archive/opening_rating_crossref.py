#!/usr/bin/env python3
"""Cross-reference opening-specific features with rating gradient.

Produces: "In the Sicilian Defense, higher-rated players see [X] more"
Run locally after downloading opening_feature_labels.json from notebook.
"""
import json, sys

def main():
    opening_path = sys.argv[1] if len(sys.argv) > 1 else 'research/sae/opening_feature_labels.json'
    gradient_path = 'research/sae/rating_gradient_k32.json'
    labels_path = 'research/sae/maia_2048_k32_final_labels.json'

    with open(opening_path) as f:
        openings = json.load(f)
    with open(gradient_path) as f:
        gradient = json.load(f)
    with open(labels_path) as f:
        label_data = json.load(f)

    opening_labels = openings['labels']
    neutral_labels = label_data['labels']
    rating_labels = label_data.get('rating_aware_labels', {})

    # Build gradient lookup
    grad_lookup = {}
    for g in gradient['all_gradients']:
        grad_lookup[str(g['feature'])] = g

    print("OPENING-SPECIFIC COACHING INSIGHTS")
    print("="*70)
    print()

    # Group features by opening
    from collections import defaultdict
    opening_features = defaultdict(list)
    for fid, info in opening_labels.items():
        opening = info['opening']
        corr = info['correlation']
        grad = grad_lookup.get(fid, {})
        grad_corr = grad.get('correlation', 0)
        rates = grad.get('rates', {})
        neutral = neutral_labels.get(fid, '?')
        rating = rating_labels.get(fid, '?')

        opening_features[opening].append({
            'fid': fid,
            'opening_corr': corr,
            'gradient_corr': grad_corr,
            'neutral_label': neutral,
            'rating_label': rating,
            'rates': rates,
        })

    # For each opening, show features that change with rating
    for opening, features in sorted(opening_features.items(), key=lambda x: -len(x[1])):
        if len(features) < 2:
            continue

        # Split into increasing/decreasing with rating
        inc = [f for f in features if f['gradient_corr'] > 0.3]
        dec = [f for f in features if f['gradient_corr'] < -0.3]

        if not inc and not dec:
            continue

        print(f"\n{opening} ({len(features)} features):")
        if inc:
            print(f"  Higher-rated sees MORE:")
            for f in sorted(inc, key=lambda x: -x['gradient_corr'])[:3]:
                print(f"    {f['neutral_label']}")
        if dec:
            print(f"  Lower-rated over-focuses on:")
            for f in sorted(dec, key=lambda x: x['gradient_corr'])[:3]:
                print(f"    {f['neutral_label']}")


if __name__ == '__main__':
    main()
