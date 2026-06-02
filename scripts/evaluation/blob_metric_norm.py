"""Norm-aware blob metric: --norm {l2,zscore,raw} matches how the model was trained.
Critical: eval normalization MUST match train normalization or comparison is meaningless."""
import argparse,json,numpy as np,torch,torch.nn.functional as F
B='/home/ec2-user/SageMaker';BASE=B+'/chess-stage-a'
p=argparse.ArgumentParser()
p.add_argument('--weights',required=True);p.add_argument('--k',type=int,default=16)
p.add_argument('--norm',choices=['l2','zscore','raw'],required=True)
p.add_argument('--cache',default=BASE+'/cache/maia3_l7only_v2_dedup.pt')
p.add_argument('--out',default=B+'/blob_norm.jsonl');p.add_argument('--tag',default='')
a=p.parse_args()
wd=torch.load(a.weights,map_location='cpu',weights_only=False);sd=wd['state_dict']
c=torch.load(a.cache,map_location='cpu',weights_only=False);raw=c['activations'].float();N=len(raw)
# normalization to match training
if a.norm=='raw':
    x=raw
else:
    # z-score uses the cache's own mean/std (train computed from same cache)
    mean=raw.mean(0);std=raw.std(0).clamp(min=1e-6)
    x=(raw-mean)/std
    if a.norm=='l2':
        x=x/x.norm(dim=-1,keepdim=True).clamp(min=1e-8)
# calibrate threshold (mean k-th largest)
kth=[]
for i in range(0,min(40000,N),8192):
    z=F.relu((x[i:i+8192]-sd['b_dec'])@sd['W_enc']+sd['b_enc'])
    kth.append(torch.topk(z,a.k,dim=1).values[:,-1].numpy())
theta=float(np.concatenate(kth).mean())
fire=np.zeros(2048)
for i in range(0,N,8192):
    z=F.relu((x[i:i+8192]-sd['b_dec'])@sd['W_enc']+sd['b_enc']).numpy();fire+=((z*(z>theta))>0).sum(0)
fire/=N
# top-feature activation stats on 20k + raw activation magnitude
raws=[];specs=[];blob=fire>0.10;blobtop=0;ntot=0
peakacts=[]
for i in range(0,min(20000,N),8192):
    z=F.relu((x[i:i+8192]-sd['b_dec'])@sd['W_enc']+sd['b_enc'])
    za=(z*(z>theta)).numpy()
    peakacts.append(za.max(1))
    top=za.argmax(1);raws.append(za.max(1));zs=za.copy();zs[:,blob]=0;specs.append(zs.max(1))
    blobtop+=blob[top].sum();ntot+=len(top)
peak=np.concatenate(peakacts)
rec={'tag':a.tag,'norm':a.norm,'theta':round(theta,4),'n_dead':int((fire==0).sum()),
     'n_blob':int(blob.sum()),'pct_top_is_blob':round(100*blobtop/ntot,1),
     'raw_top_p50':round(float(np.percentile(np.concatenate(raws),50)),3),
     'raw_top_p90':round(float(np.percentile(peak,90)),3),
     'spec_top_p50':round(float(np.percentile(np.concatenate(specs),50)),3),
     'useful_0.1pct':int((fire>=0.001).sum())}
print(json.dumps(rec))
open(a.out,'a').write(json.dumps(rec)+'\n')
