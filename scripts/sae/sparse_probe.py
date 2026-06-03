"""k-sparse probing of SAE features against SEE ground-truth concepts.

Question (Gao et al. 2024; SAEBench): how FEW SAE features do you need to predict a known
concept? Few features at high accuracy => the SAE cleanly isolates that concept into a
small number of monosemantic features. Many => the concept is smeared.

We trust the SEE labels (see_labels_168k.npz); we are SKEPTICAL of the probe — so we
report balanced accuracy / F1 AND the majority-class baseline, never raw accuracy, and we
select features on TRAIN only (no leakage).

NOTE on naming: the SAE's sparsity is k (top-k). The probe's feature budget is p, to avoid
confusion. We sweep p in {1,2,4,8,16,32}.

Protocol per (model, concept):
  1. encode all positions -> SAE activations f(x)  [N x D], gated at calibrated threshold
  2. 70/30 train/test split (fixed seed)
  3. rank features by mutual information with the label ON TRAIN
  4. for each p: train logistic regression on the top-p features (train), eval on test
  5. report balanced_accuracy@p and f1@p, plus base rate + majority baseline

Run on chess-poc from ~/SageMaker:
  python sparse_probe.py --models k4,k6,k8,k16 --out sparse_probe_results.json
"""
import torch, numpy as np, json, argparse, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import balanced_accuracy_score, f1_score
B = '/home/ec2-user/SageMaker'; BASE = B + '/chess-stage-a'
ap = argparse.ArgumentParser()
ap.add_argument('--models', default='k4,k6,k8,k16')
ap.add_argument('--dict', type=int, default=2048)
ap.add_argument('--out', required=True)
ap.add_argument('--ps', default='1,2,4,8,16,32')
a = ap.parse_args()
PS = [int(p) for p in a.ps.split(',')]

c = torch.load(BASE + '/cache/maia3_l7only_v2_dedup.pt', map_location='cpu', weights_only=False)
craw = c['activations'].float(); zmean = craw.mean(0); zstd = craw.std(0).clamp(min=1e-6)
x = (craw - zmean) / zstd; N = len(x)
lab = np.load(B + '/see_labels_168k.npz')
CONCEPTS = ['any_hang','hang_major','hang_minor','hang_queen','hang_rook','hang_knight',
            'hang_bishop','blunder_capture','best_capture','best_check','best_quiet','severe','endgame']

# fixed split
rng = np.random.default_rng(0)
perm = rng.permutation(N); ntr = int(0.7 * N)
tr, te = perm[:ntr], perm[ntr:]

def encode(tag, kk):
    sd = torch.load(BASE + f'/output/maia3_sae/btk_{a.dict}_{tag}_nol2.pt', map_location='cpu', weights_only=False)['state_dict']
    kth = []
    for i in range(0, 40000, 8192):
        z = F.relu((x[i:i+8192] - sd['b_dec']) @ sd['W_enc'] + sd['b_enc'])
        kth.append(torch.topk(z, kk, 1).values[:, -1].numpy())
    th = float(np.concatenate(kth).mean())
    A = np.zeros((N, sd['W_enc'].shape[1]), np.float32)
    for i in range(0, N, 8192):
        z = F.relu((x[i:i+8192] - sd['b_dec']) @ sd['W_enc'] + sd['b_enc']).numpy()
        A[i:i+z.shape[0]] = z * (z > th)
    return A

results = {}
for tag in a.models.split(','):
    kk = int(''.join(ch for ch in tag if ch.isdigit()))
    print(f"=== {tag} (encoding) ===", flush=True)
    A = encode(tag, kk)
    Atr, Ate = A[tr], A[te]
    results[tag] = {}
    for con in CONCEPTS:
        y = lab[con].astype(int); ytr, yte = y[tr], y[te]
        base = float(yte.mean())                       # positive rate on test
        maj = max(base, 1 - base)                      # majority-class accuracy
        if ytr.sum() < 50 or ytr.sum() > len(ytr) - 50:   # too rare/common to probe
            results[tag][con] = {'base_rate': round(base,3), 'majority_acc': round(maj,3), 'skipped': 'degenerate'}
            continue
        # MI feature ranking on TRAIN only (subsample for speed)
        sub = rng.choice(len(tr), min(20000, len(tr)), replace=False)
        mi = mutual_info_classif(Atr[sub], ytr[sub], discrete_features=False, random_state=0)
        rank = np.argsort(-mi)
        per_p = {}
        for p in PS:
            feats = rank[:p]
            clf = LogisticRegression(max_iter=300, class_weight='balanced', C=1.0)
            clf.fit(Atr[:, feats], ytr)
            pred = clf.predict(Ate[:, feats])
            per_p[p] = {'bal_acc': round(balanced_accuracy_score(yte, pred), 3),
                        'f1': round(f1_score(yte, pred, zero_division=0), 3),
                        'top_feat': int(rank[0])}
        results[tag][con] = {'base_rate': round(base,3), 'majority_acc': round(maj,3),
                             'top_feature': int(rank[0]), 'per_p': per_p}
        b1 = per_p[1]['bal_acc']; b8 = per_p[8]['bal_acc']
        print(f"  {con:16s} base {base:4.2f} | bal_acc@1 {b1:.2f} @8 {b8:.2f} | top f{rank[0]}", flush=True)
    json.dump(results, open(a.out, 'w'), indent=1)
print(f"wrote {a.out}", flush=True)
