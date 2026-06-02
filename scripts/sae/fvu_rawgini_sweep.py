"""Threshold-free concentration (raw Gini, no activation gate) + reconstruction FVU vs k.
Settles whether the Gini decline was itself a threshold confound, and measures the REAL
cost of high k: does reconstruction improve, and is the dictionary's raw mass concentrated."""
import torch,numpy as np,torch.nn.functional as F,os
B='/home/ec2-user/SageMaker';BASE=B+'/chess-stage-a'
c=torch.load(BASE+'/cache/maia3_l7only_v2_dedup.pt',map_location='cpu',weights_only=False)
craw=c['activations'].float();zmean=craw.mean(0);zstd=craw.std(0).clamp(min=1e-6)
x=(craw-zmean)/zstd;N=len(x)
def gini(v):
    v=np.sort(v[v>=0]);n=len(v)
    if n==0 or v.sum()==0:return 0.0
    cum=np.cumsum(v);return float((n+1-2*(cum/cum[-1]).sum())/n)
print(f"{'k':>4s} | {'rawGini':>7s} {'threshGini':>10s} | {'FVU':>6s} {'frac_var_expl':>13s}")
print('-'*56)
for tag,kk in [('k4',4),('k6',6),('k8',8),('k10',10),('k12',12),('k16',16),('k32',32)]:
    wp=BASE+f'/output/maia3_sae/btk_2048_{tag}_nol2.pt'
    if not os.path.exists(wp):continue
    sd=torch.load(wp,map_location='cpu',weights_only=False)['state_dict']
    kth=[]
    for i in range(0,40000,8192):
        z=F.relu((x[i:i+8192]-sd['b_dec'])@sd['W_enc']+sd['b_enc']);kth.append(torch.topk(z,kk,1).values[:,-1].numpy())
    th=float(np.concatenate(kth).mean())
    rawmass=np.zeros(2048);thrmass=np.zeros(2048);sse=0.0;sst=0.0;xm=x.mean(0)
    for i in range(0,N,8192):
        xb=x[i:i+8192];z=F.relu((xb-sd['b_dec'])@sd['W_enc']+sd['b_enc'])
        zt=torch.topk(z,kk,1)                       # top-k per position (eval recon)
        zk=torch.zeros_like(z).scatter_(1,zt.indices,zt.values)
        recon=zk@sd['W_dec']+sd['b_dec']
        sse+=((xb-recon)**2).sum().item();sst+=((xb-xm)**2).sum().item()
        zn=z.numpy();rawmass+=zn.sum(0);za=zn*(zn>th);thrmass+=za.sum(0)
    fvu=sse/sst
    print(f"{tag:>4s} | {gini(rawmass):>7.3f} {gini(thrmass):>10.3f} | {fvu:>6.3f} {1-fvu:>12.1%}")
