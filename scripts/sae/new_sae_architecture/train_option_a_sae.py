"""
Train BatchTopK SAE on Option-A diffs: h[best_to_sq] - h[blunder_to_sq]
Config: dict_size=2048, k=32 (matches current production SAE)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, time, sys
BASE="/home/ec2-user/SageMaker/chess-stage-a"
CACHE=BASE+"/cache/maia3_option_a_diff.pt"
OUT_DIR=BASE+"/output/maia3_sae"
import os; os.makedirs(OUT_DIR, exist_ok=True)
OUT=OUT_DIR+"/maia3_option_a_2048_k32.pt"

print("Loading cache...")
d=torch.load(CACHE, map_location="cpu", weights_only=False)
raw=d["activations"].float()
mean=torch.tensor(d["mean"]).float()
std=torch.tensor(d["std"]).float().clamp(min=1e-6)
meta=d["metadata"]
print(f"Loaded {len(raw)} diffs, shape={raw.shape}")

# Normalize: z-score then L2
x=(raw-mean)/std
x=x/x.norm(dim=-1,keepdim=True).clamp(min=1e-8)
print(f"Normalized. Norm range: {x.norm(dim=-1).min():.3f}-{x.norm(dim=-1).max():.3f}")

D_IN=x.shape[1]  # 512
D_HIDDEN=2048; K=32

class BatchTopKSAE(nn.Module):
    def __init__(self, d_in, d_hidden, k, k_aux=256, aux_alpha=1/32, n_dead=5):
        super().__init__()
        self.k=k; self.k_aux=k_aux; self.aux_alpha=aux_alpha; self.n_dead=n_dead
        self.W_enc=nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_in,d_hidden)))
        self.W_dec=nn.Parameter(self.W_enc.data.clone().T)
        self.W_dec.data[:]=self.W_dec/self.W_dec.norm(dim=-1,keepdim=True)
        self.b_enc=nn.Parameter(torch.zeros(d_hidden))
        self.b_dec=nn.Parameter(torch.zeros(d_in))
        self.register_buffer("dead_cnt",torch.zeros(d_hidden))

    def forward(self, x):
        z=F.relu((x-self.b_dec)@self.W_enc+self.b_enc)
        flat=z.reshape(-1); tk=x.shape[0]*self.k
        tv,ti=torch.topk(flat,k=min(tk,flat.numel()))
        acts=torch.zeros_like(flat); acts[ti]=tv; acts=acts.reshape(z.shape)
        if self.training:
            active=(acts>0).any(dim=0)
            self.dead_cnt[active]=0; self.dead_cnt[~active]+=1
        x_hat=acts@self.W_dec+self.b_dec
        loss=(x_hat-x).pow(2).mean()
        if self.training and self.k_aux>0:
            dead=self.dead_cnt>=self.n_dead
            if dead.sum()>0:
                err=(x-x_hat).detach()
                dp=F.relu(((x-self.b_dec)@self.W_enc+self.b_enc)[:,dead])
                ka=min(self.k_aux,int(dead.sum()))
                tkv=torch.topk(dp,k=ka,dim=-1)
                da=torch.zeros_like(dp).scatter(-1,tkv.indices,tkv.values)
                loss=loss+self.aux_alpha*(da@self.W_dec[dead]-err).pow(2).mean()
        return loss, acts

    @torch.no_grad()
    def norm_dec(self):
        self.W_dec.data=self.W_dec/self.W_dec.norm(dim=-1,keepdim=True)

torch.manual_seed(42)
sae=BatchTopKSAE(D_IN,D_HIDDEN,K)
device="cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
sae=sae.to(device); x=x.to(device)

opt=torch.optim.Adam(sae.parameters(),lr=3e-4,betas=(0.9,0.999))

# Train/val split
n=len(x); n_val=min(20000,n//5)
idx=torch.randperm(n,generator=torch.Generator().manual_seed(42))
x_val=x[idx[:n_val]]; x_train=x[idx[n_val:]]
print(f"Train: {len(x_train)}, Val: {len(x_val)}")

BS=4096; EPOCHS=200
print(f"\nTraining {D_HIDDEN}d k={K} for {EPOCHS} epochs...")
t0=time.time()
for ep in range(EPOCHS):
    sae.train()
    perm=torch.randperm(len(x_train),device=device)
    ep_loss=0; nb=0
    for i in range(0,len(x_train),BS):
        batch=x_train[perm[i:i+BS]]
        loss,_=sae(batch)
        opt.zero_grad(); loss.backward(); opt.step(); sae.norm_dec()
        ep_loss+=loss.item(); nb+=1

    if (ep+1)%20==0 or ep==0:
        sae.eval()
        with torch.no_grad():
            val_loss,acts=sae(x_val)
            dead=(sae.dead_cnt>=sae.n_dead).sum().item()
            nz=(acts>0).float().mean().item()
        elapsed=time.time()-t0
        print(f"  ep {ep+1:>3}/{EPOCHS}: train={ep_loss/nb:.4f} val={val_loss:.4f} "
              f"dead={dead} nz/pos={nz:.3f} ({elapsed:.0f}s)", flush=True)

# Final eval
sae.eval()
with torch.no_grad():
    _,all_acts=sae(x)
all_acts=all_acts.cpu().numpy()
fire_rates=(all_acts>0).mean(0)
live=(fire_rates>0.001).sum()
print(f"\nFinal: {live}/{D_HIDDEN} live features, "
      f"fire rate range [{fire_rates[fire_rates>0].min():.4f},{fire_rates.max():.4f}]")

# Save
torch.save({
    "state_dict": {k:v.cpu() for k,v in sae.state_dict().items()},
    "config": {"d_input":D_IN,"dict_size":D_HIDDEN,"k":K,"architecture":"BatchTopKSAE",
               "construction":"option_a","n_train":len(x_train),"n_val":n_val,
               "epochs":EPOCHS},
    "norm": {"mean":d["mean"],"std":d["std"]},
    "fire_rates": fire_rates,
}, OUT)
print(f"Saved to {OUT}")
