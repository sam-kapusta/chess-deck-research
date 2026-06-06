#!/usr/bin/env python3
"""Train BatchTopK SAEs on the l7only v2 diff cache — z-score ONLY (no L2), the `_nol2` recipe.

Reconstructs the exact recipe that produced btk_2048_k6_nol2.pt (recovered from that model's embedded
config + how every consumer normalizes it): 1024-dim Maia3 L7 best-blunder mean-pool diff cache,
z-SCORE normalization with NO L2 (the deliberate divergence from SandstonePersonas — chess diffs are
magnitude-meaningful, so L2 would erase severity). BatchTopK + AuxK + unit-norm decoder.

NOTE on the L2 question: the k6 model's saved config says `l2_normalized: True`, but that field is a
stale default — dict_size_compare.py, encode_game_blunders.py, and knowledge.md all normalize these
models z-score-only at inference, and they produce coherent features. Training with L2 + inferring
without would scramble activations. So z-score-only is the faithful, internally-consistent recipe.

Trains a grid of (dict, k) and saves each as btk_{dict}_k{k}_nol2.pt with config + training_log,
matching the existing naming so dict_size_compare.py / the v7 pipeline pick them up unchanged.

Run on chess-poc:
  cd ~/SageMaker && python3 train_l7only_nol2.py --configs 512:4,512:2,256:4,256:2
"""
import argparse, json, time, os
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.backends.cudnn.benchmark = True
BASE = "/home/ec2-user/SageMaker/chess-stage-a"
CACHE = BASE + "/cache/maia3_l7only_v2_dedup.pt"
OUTDIR = BASE + "/output/maia3_sae"


class BatchTopKSAE(nn.Module):
    def __init__(self, d_input, d_hidden, k, k_aux=64, aux_alpha=1/32, n_dead=5, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_input, d_hidden)))
        self.W_dec = nn.Parameter(self.W_enc.data.clone().T)
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_input))
        self.d_hidden, self.k, self.k_aux, self.aux_alpha, self.n_dead = d_hidden, k, k_aux, aux_alpha, n_dead
        self.register_buffer("dead_cnt", torch.zeros(d_hidden))

    def forward(self, x):
        z = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        flat = z.reshape(-1)
        total_k = min(x.shape[0] * self.k, flat.numel())
        tv, ti = torch.topk(flat, k=total_k)
        acts = torch.zeros_like(flat); acts[ti] = tv; acts = acts.reshape(z.shape)
        if self.training:
            active = (acts > 0).any(dim=0)
            self.dead_cnt[active] = 0; self.dead_cnt[~active] += 1
        x_hat = acts @ self.W_dec + self.b_dec
        loss = (x_hat - x).pow(2).mean()
        if self.training and self.k_aux > 0:
            dead = self.dead_cnt >= self.n_dead
            if dead.sum() > 0:
                err = (x - x_hat).detach()
                dp = F.relu(((x - self.b_dec) @ self.W_enc + self.b_enc)[:, dead])
                ka = min(self.k_aux, int(dead.sum()))
                tkv = torch.topk(dp, k=ka, dim=-1)
                da = torch.zeros_like(dp).scatter(-1, tkv.indices, tkv.values)
                loss = loss + self.aux_alpha * (da @ self.W_dec[dead] - err).pow(2).mean()
        return loss, x_hat, acts

    @torch.no_grad()
    def norm_decoder(self):
        n = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.W_dec.grad -= (self.W_dec.grad * n).sum(-1, keepdim=True) * n
        self.W_dec.data = n


def load_zscore():
    """z-score ONLY — no L2. The defining choice of the _nol2 recipe."""
    c = torch.load(CACHE, map_location="cpu", weights_only=False)
    raw = c["activations"].float()
    mean = raw.mean(0); std = raw.std(0).clamp(min=1e-6)
    x = (raw - mean) / std
    torch.manual_seed(42)
    n = x.shape[0]; n_val = int(n * 0.1); perm = torch.randperm(n)
    return x[perm[:n - n_val]], x[perm[n - n_val:]], x.shape[1], n


def train(model, loader, n_epochs, warmup, device, val):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler("cuda")
    step = 0; log = []
    for ep in range(n_epochs):
        model.train()
        for (batch,) in loader:
            for g in opt.param_groups:               # linear lr warmup
                g["lr"] = 3e-4 * min(1.0, (step + 1) / warmup)
            batch = batch.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss, _, _ = model(batch)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            model.norm_decoder(); scaler.step(opt); scaler.update(); opt.zero_grad()
            step += 1
        if ep == 0 or (ep + 1) % 50 == 0 or ep == n_epochs - 1:
            vl, fire = eval_model(model, val, device)
            log.append({"epoch": ep + 1, "val_loss": vl, "n_dead": int((fire == 0).sum()),
                        "n_active_gt_0.5pct": int((fire >= 0.005).sum())})
    return log


@torch.no_grad()
def eval_model(model, val, device):
    model.eval(); xs, xhs, ac = [], [], []
    for i in range(0, len(val), 8192):
        b = val[i:i+8192].to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            _, xh, a = model(b)
        xs.append(b.cpu()); xhs.append(xh.float().cpu()); ac.append((a > 0).cpu())
    x = torch.cat(xs); xh = torch.cat(xhs); fire = torch.cat(ac).float().mean(0).numpy()
    vl = float(((x - xh) ** 2).mean())
    return vl, fire


ap = argparse.ArgumentParser()
ap.add_argument("--configs", default="512:4,512:2,256:4,256:2", help="comma list of dict:k")
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--warmup", type=int, default=500)
a = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
train_x, val_x, d_in, ntot = load_zscore()
loader = DataLoader(TensorDataset(train_x), batch_size=4096, shuffle=True, num_workers=2, pin_memory=True)
print(f"cache {CACHE} | d_input={d_in} | train {len(train_x)} val {len(val_x)} | device {device}", flush=True)

for cfg in a.configs.split(","):
    dct, k = (int(x) for x in cfg.split(":"))
    torch.manual_seed(42)
    model = BatchTopKSAE(d_in, dct, k).to(device)
    t0 = time.time()
    log = train(model, loader, a.epochs, a.warmup, device, val_x)
    vl, fire = eval_model(model, val_x, device)
    dead = int((fire == 0).sum()); live = dct - dead
    blob = int((fire >= 0.05).sum()); band = int(((fire >= 0.001) & (fire < 0.05)).sum())
    out = f"{OUTDIR}/btk_{dct}_k{k}_nol2.pt"
    torch.save({"state_dict": model.state_dict(),
                "config": {"model": "maia3", "architecture": "BatchTopKSAE", "d_input": d_in,
                           "dict_size": dct, "k": k, "k_aux": 64, "aux_alpha": 1/32, "lr": 3e-4,
                           "batch_size": 4096, "n_epochs": a.epochs, "warmup_steps": a.warmup,
                           "l2_normalized": False, "seed": 42, "n_train": len(train_x)},
                "training_log": log, "source_activations": "chess-stage-a/cache/maia3_l7only_v2_dedup.pt"},
               out)
    print(f"d{dct}_k{k}: live={live}/{dct} dead={dead} blob(>5%)={blob} band={band} "
          f"val_loss={vl:.3f} maxfire={fire.max()*100:.0f}% [{time.time()-t0:.0f}s] -> {os.path.basename(out)}", flush=True)
