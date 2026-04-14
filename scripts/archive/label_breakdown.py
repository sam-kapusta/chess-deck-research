#!/usr/bin/env python3
"""Break down labels by category, uniqueness, and most common patterns.

Usage:
    python3 label_breakdown.py --labels output/labels_k32.json output/labels_k64.json
"""
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', required=True, nargs='+')
    args = parser.parse_args()

    all_cats = {}
    all_label_counts = {}

    for path in args.labels:
        with open(path) as f:
            labels = json.load(f)

        name = path.split('/')[-1].replace('labels_', '').replace('.json', '')
        cats = {}
        conf = {'high': 0, 'medium': 0, 'low': 0}

        for fid, v in labels.items():
            c = v.get('category', 'unknown')
            cats[c] = cats.get(c, 0) + 1
            conf[v.get('confidence', 'low')] += 1

        total = len(labels)
        print(name + ' (' + str(total) + ' features):')
        print('  Confidence: high=' + str(conf['high']) + ' (' + str(round(conf['high'] / total * 100)) + '%) med=' + str(conf['medium']) + ' low=' + str(conf['low']))

        sorted_cats = sorted(cats.items(), key=lambda x: -x[1])
        print('  Categories:')
        for cat, count in sorted_cats[:12]:
            print('    ' + cat + ': ' + str(count) + ' (' + str(round(count / total * 100, 1)) + '%)')

        unique = set(v['label'] for v in labels.values())
        print('  Unique labels: ' + str(len(unique)) + ' / ' + str(total) + ' (' + str(round(len(unique) / total * 100)) + '%)')
        print()

        for c, n in cats.items():
            if c not in all_cats:
                all_cats[c] = {}
            all_cats[c][name] = n

        for fid, v in labels.items():
            lbl = v['label']
            all_label_counts[lbl] = all_label_counts.get(lbl, 0) + 1

    if len(args.labels) > 1:
        print('=== Most common labels (across all) ===')
        for lbl, count in sorted(all_label_counts.items(), key=lambda x: -x[1])[:15]:
            print('  ' + str(count) + 'x  ' + lbl)


if __name__ == '__main__':
    main()
