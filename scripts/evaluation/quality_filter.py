#!/usr/bin/env python3
"""Filter SAE features by quality gates: confidence, fire rate, polysemantic.

Usage:
    python3 quality_filter.py --labels output/labels_blunder_mt_k32.json --profiles prof1.json prof2.json
"""
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', required=True, nargs='+', help='Label JSON files')
    parser.add_argument('--profiles', nargs='+', help='Profile JSON files (for fire rates)')
    parser.add_argument('--max-fire-rate', type=float, default=5.0, help='Max fire rate %')
    args = parser.parse_args()

    # Load profiles for fire rates
    profiles = {}
    if args.profiles:
        for ppath in args.profiles:
            with open(ppath) as f:
                p = json.load(f)
            n_features = len(p)
            tag = str(n_features)
            for fid, data in p.items():
                profiles[tag + '_' + fid] = data

    for label_path in args.labels:
        with open(label_path) as f:
            labels = json.load(f)

        total = 0
        pass_conf = 0
        pass_fr = 0
        pass_poly = 0
        pass_all = 0
        categories_passing = {}

        for fid, v in labels.items():
            total += 1
            conf = v.get('confidence', 'low')
            poly = v.get('polysemantic', False)
            cat = v.get('category', 'unknown')

            # Try to find fire rate from profiles
            fr = None
            for tag in profiles:
                key = tag.split('_')[0] + '_' + fid
                if key in profiles:
                    fr = profiles[key].get('fire_rate', 0)
                    break

            is_high_conf = conf in ['high', 'medium']
            is_low_fr = fr is not None and fr <= args.max_fire_rate
            is_mono = not poly

            if is_high_conf:
                pass_conf += 1
            if is_low_fr:
                pass_fr += 1
            if is_mono:
                pass_poly += 1

            if is_high_conf and is_mono and (fr is None or fr <= args.max_fire_rate):
                pass_all += 1
                categories_passing[cat] = categories_passing.get(cat, 0) + 1

        # Unique labels
        quality_labels = {}
        for fid, v in labels.items():
            conf = v.get('confidence', 'low')
            poly = v.get('polysemantic', False)
            if conf in ['high', 'medium'] and not poly:
                lbl = v['label']
                quality_labels[lbl] = quality_labels.get(lbl, 0) + 1

        unique_labels = len(quality_labels)
        redundant = sum(v - 1 for v in quality_labels.values() if v > 1)

        print(label_path + ':')
        print('  Total: ' + str(total))
        print('  High/med confidence: ' + str(pass_conf) + ' (' + str(round(pass_conf / total * 100)) + '%)')
        print('  Fire rate <= ' + str(args.max_fire_rate) + '%: ' + str(pass_fr) + ' (' + str(round(pass_fr / total * 100)) + '%)')
        print('  Monosemantic: ' + str(pass_poly) + ' (' + str(round(pass_poly / total * 100)) + '%)')
        print('  PASS ALL: ' + str(pass_all) + ' (' + str(round(pass_all / total * 100)) + '%)')
        print('  Unique labels: ' + str(unique_labels))
        print('  Redundant: ' + str(redundant))
        print('  Categories:')
        for cat, n in sorted(categories_passing.items(), key=lambda x: -x[1])[:10]:
            print('    ' + cat + ': ' + str(n))
        print()


if __name__ == '__main__':
    main()
