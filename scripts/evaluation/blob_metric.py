"""Reusable blob-concentration metric for any BTK SAE weights file.
Calibrates threshold (mean k-th largest), then reports on a corpus sample:
  n_dead, n_blob(>10%), pct_positions_top_is_blob, specific-top activation p50, FVU.
Appends one JSON line to --out.

Usage: python3 blob_metric.py --weights <path.pt> --k <k> --out blob_sweep.jsonl
"""
import argparse,json,numpy as np,torch,torch.nn.functional as F
B='/home/ec2-user/SageMaker';BASE=B+'/chess-stage-a'
p=argparse.ArgumentParser()
p.add_argument('--weights',required=True); p.add_argument('--k',type=int,required=True)
p.add_argument('--cache',default=BASE+'/cache/maia3_l7only_v2_dedup.pt')
p.add_argument('--out',default=B+'/blob_sweep.jsonl')
p.add_argument('--tag',default='')
a=p.parse_args()
wd=torch.load(a.weights,map_location='cpu',weights_only=False); sd=wd['state_dict']
ns=json.load(open(a.weights.replace('.pt','_stats.json')))
mean=torch.tensor(ns['mean']);std=torch.tensor(ns['std']).clamp(min=1e-6)
c=torch.load(a.cache,map_location='cpu',weights_only=False)
raw=c['activations'].float();x=(raw-mean)/std;x=x/x.norm(dim=-1,keepdim=True).clamp(min=1e-8)
N=len(x)
# calibrate threshold on 40k
kth=[]
for i in range(0,min(40000,N),8192):
    z=F.relu((x[i:i+8192]-sd['b_dec'])@sd['W_enc']+sd['b_enc'])
    kth.append(torch.topk(z,a.k,dim=1).values[:,-1].numpy())
theta=float(np.concatenate(kth).mean())
# fire rates over full corpus
fire=np.zeros(2048)
for i in range(0,N,8192):
    z=F.relu((x[i:i+8192]-sd['b_dec'])@sd['W_enc']+sd['b_enc']).numpy()
    fire+=((z*(z>theta))>0).sum(0)
fire/=N
n_dead=int((fire==0).sum()); blob=fire>0.10; n_blob=int(blob.sum())
# top-feature stats + FVU on 20k
raws=[];specs=[];blobtop=0;ntot=0;mse=0.0;varx=0.0;nb=0
for i in range(0,min(20000,N),8192):
    xb=x[i:i+8192]
    z=F.relu((xb-sd['b_dec'])@sd['W_enc']+sd['b_enc'])
    za=(z*(z>theta))
    xhat=za@sd['W_dec']+sd['b_dec']
    mse+=((xhat-xb)**2).sum().item(); varx+=((xb-xb.mean())**2).sum().item(); nb+=xb.numel()
    zan=za.numpy(); top=zan.argmax(1)
    raws.append(zan.max(1)); zs=zan.copy(); zs[:,blob]=0; specs.append(zs.max(1))
    blobtop+=blob[top].sum(); ntot+=len(top)
raw_top=np.concatenate(raws); spec_top=np.concatenate(specs)
fvu=mse/varx
rec={'tag':a.tag or f'k{a.k}','k':a.k,'theta':round(theta,4),'n_dead':n_dead,'n_blob':n_blob,
     'pct_top_is_blob':round(100*blobtop/ntot,1),'raw_top_p50':round(float(np.percentile(raw_top,50)),3),
     'spec_top_p50':round(float(np.percentile(spec_top,50)),3),'fvu':round(fvu,4),
     'n_useful_0.1pct':int((fire>=0.001).sum()),'n_active_1to10pct':int(((fire>=0.01)&(fire<=0.10)).sum())}
print(json.dumps(rec))
with open(a.out,'a') as f: f.write(json.dumps(rec)+'\n')
