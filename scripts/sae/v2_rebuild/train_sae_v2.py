"""Generic BatchTopK SAE trainer for v2 diff caches.
Usage: python3 train_sae_v2.py <cache.pt> <out.pt> <k> [dict_size]
Default k=16 (validated operating point), dict_size=2048."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, sys, os

CACHE=sys.argv[1]; OUT=sys.argv[2]; K=int(sys.argv[3]); D_HIDDEN=int(sys.argv[4]) if len(sys.argv)>4 else 2048
print(f"cache={CACHE} out={OUT} k={K} dict={D_HIDDEN}",flush=True)
d=torch.load(CACHE,map_location="cpu",weights_only=False)
raw=d["activations"].float()
mean=torch.tensor(np.array(d["mean"])).float(); std=torch.tensor(np.array(d["std"])).float().clamp(min=1e-6)
x=(raw-mean)/std; x=x/x.norm(dim=-1,keepdim=True).clamp(min=1e-8)
D_IN=x.shape[1]
print(f"{len(raw)} diffs dim={D_IN}",flush=True)

class BatchTopKSAE(nn.Module):
    def __init__(self,d_in,d_hidden,k,k_aux=256,aux_alpha=1/32,n_dead=5):
        super().__init__()
        self.k=k;self.k_aux=k_aux;self.aux_alpha=aux_alpha;self.n_dead=n_dead
        self.W_enc=nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_in,d_hidden)))
        self.W_dec=nn.Parameter(self.W_enc.data.clone().T)
        self.W_dec.data[:]=self.W_dec/self.W_dec.norm(dim=-1,keepdim=True)
        self.b_enc=nn.Parameter(torch.zeros(d_hidden)); self.b_dec=nn.Parameter(torch.zeros(d_in))
        self.register_buffer("dead_cnt",torch.zeros(d_hidden))
    def forward(self,x):
        z=F.relu((x-self.b_dec)@self.W_enc+self.b_enc)
        flat=z.reshape(-1); tk=x.shape[0]*self.k
        tv,ti=torch.topk(flat,k=min(tk,flat.numel()))
        acts=torch.zeros_like(flat); acts[ti]=tv; acts=acts.reshape(z.shape)
        if self.training:
            active=(acts>0).any(dim=0); self.dead_cnt[active]=0; self.dead_cnt[~active]+=1
        x_hat=acts@self.W_dec+self.b_dec; loss=(x_hat-x).pow(2).mean()
        if self.training and self.k_aux>0:
            dead=self.dead_cnt>=self.n_dead
            if dead.sum()>0:
                err=(x-x_hat).detach(); dp=F.relu(((x-self.b_dec)@self.W_enc+self.b_enc)[:,dead])
                ka=min(self.k_aux,int(dead.sum())); tkv=torch.topk(dp,k=ka,dim=-1)
                da=torch.zeros_like(dp).scatter(-1,tkv.indices,tkv.values)
                loss=loss+self.aux_alpha*(da@self.W_dec[dead]-err).pow(2).mean()
        return loss,acts
    @torch.no_grad()
    def norm_dec(self): self.W_dec.data=self.W_dec/self.W_dec.norm(dim=-1,keepdim=True)

torch.manual_seed(42)
sae=BatchTopKSAE(D_IN,D_HIDDEN,K)
device="cuda" if torch.cuda.is_available() else "cpu"; print("device",device,flush=True)
sae=sae.to(device); x=x.to(device)
opt=torch.optim.Adam(sae.parameters(),lr=3e-4)
n=len(x); n_val=min(20000,n//5)
idx=torch.randperm(n,generator=torch.Generator().manual_seed(42))
x_val=x[idx[:n_val]]; x_train=x[idx[n_val:]]
BS=4096; EPOCHS=200; t0=time.time()
for ep in range(EPOCHS):
    sae.train(); perm=torch.randperm(len(x_train),device=device); el=0;nb=0
    for i in range(0,len(x_train),BS):
        loss,_=sae(x_train[perm[i:i+BS]]); opt.zero_grad(); loss.backward(); opt.step(); sae.norm_dec()
        el+=loss.item(); nb+=1
    if (ep+1)%40==0 or ep==0:
        sae.eval()
        with torch.no_grad():
            vl,acts=sae(x_val); dead=(sae.dead_cnt>=sae.n_dead).sum().item(); nz=(acts>0).float().mean().item()
        print(f"  ep{ep+1}/{EPOCHS} train={el/nb:.4f} val={vl:.4f} dead={dead} nz={nz:.3f} ({time.time()-t0:.0f}s)",flush=True)
sae.eval()
with torch.no_grad(): _,aa=sae(x)
aa=aa.cpu().numpy(); fr=(aa>0).mean(0); live=(fr>0.001).sum()
interp=((fr>=0.001)&(fr<=0.01)).sum()
print(f"FINAL live={live}/{D_HIDDEN} interp(0.1-1%)={interp}",flush=True)
torch.save({"state_dict":{k:v.cpu() for k,v in sae.state_dict().items()},
    "config":{"d_input":D_IN,"dict_size":D_HIDDEN,"k":K,"architecture":"BatchTopKSAE","epochs":EPOCHS},
    "norm":{"mean":d["mean"],"std":d["std"]},"fire_rates":fr},OUT)
print(f"SAVED {OUT}",flush=True)
