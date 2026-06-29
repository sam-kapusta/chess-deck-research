import json, time
from collections import defaultdict
import chess
import motifs as MO
BANDS=['600-800','800-1000','1000-1200','1200-1400','1400-1600','1600-1800','1800-2000','2000-2200','2200-2400','2400-2600','2600-2800']
enrich=json.load(open('fifa_enrich_200k.json')); sweep=json.load(open('fifa_sweep_200k.json'))
def l2s(line): return [t for t in (line or '').split() if not t.replace('.','').isdigit() and t not in ('1-0','0-1','1/2-1/2','*')]
den=defaultdict(int); newf=defaultdict(int); n=0
for row in sweep:
    band=row['band']; den[band]+=1
    ce=enrich.get(f'{row["fen"]}|{row["uci"]}')
    if not ce: continue
    try:
        b=chess.Board(row['fen']); mover=b.turn
        bl=l2s(ce.get('top_3_best',[{}])[0].get('line','')) if ce.get('top_3_best') else []
        if not bl: continue
        bb=b.copy(); ucis=[]
        for san in bl:
            try: mv=bb.parse_san(san)
            except: break
            ucis.append(mv.uci()); bb.push(mv)
        if not ucis: continue
        if MO.detect_line(b, ucis, mover).get('advancedPawn'): newf[band]+=1
        n+=1
    except: pass
rates=[round(1000*newf[b]/den[b],3) if den[b] else 0 for b in BANDS]
nz=[r for r in rates if r]
json.dump({'n':n,'fires':[newf[b] for b in BANDS],'rate':rates,'ratio':round(nz[0]/nz[-1],1) if nz and nz[-1] else 0}, open('advpawn_result.json','w'))
