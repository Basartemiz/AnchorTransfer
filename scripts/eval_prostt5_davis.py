"""Eval ProstT5 models on Davis with anchor quartile breakdown."""
import torch, torch.nn as nn, numpy as np, pandas as pd, sys, json, logging
from sklearn.metrics import mean_squared_error
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger()
sys.path.insert(0, 'src')

device = torch.device('cuda')

prostt5 = torch.load('data/processed/prostt5_all.pt', map_location='cpu', weights_only=False)
log.info(f'ProstT5: {len(prostt5)}')

davis = pd.read_csv('data/raw/davis/davis_benchmark.csv')
d_seqs = dict(zip(davis.protein_name, davis.protein_sequence))

CHARISOSMISET = {"#":29,"%":30,")":31,"(":1,"+":32,"-":33,"/":34,".":2,"1":35,"0":3,"3":36,"2":4,"5":37,"4":5,"7":38,"6":6,"9":39,"8":7,"=":40,"A":41,"@":8,"C":42,"B":9,"E":43,"D":10,"G":44,"F":11,"I":45,"H":12,"K":46,"M":47,"L":13,"O":48,"N":14,"P":15,"S":49,"R":16,"[":50,"]":51,"_":19,"a":20,"c":21,"e":22,"g":23,"i":24,"l":25,"n":26,"o":27,"s":28,"r":17,"u":18}
def enc_smi(s, ml=100): return [CHARISOSMISET.get(c,0) for c in s[:ml]] + [0]*(ml-len(s[:ml]))
def ci_fn(y, f):
    ind=np.argsort(y); y=y[ind]; f=f[ind]
    z=np.sum((np.tile(y,(len(y),1)).T<np.tile(y,(len(y),1)))*(np.tile(f,(len(f),1)).T<np.tile(f,(len(f),1))))+0.5*np.sum((np.tile(y,(len(y),1)).T<np.tile(y,(len(y),1)))*(np.tile(f,(len(f),1)).T==np.tile(f,(len(f),1))))
    n=np.sum(np.tile(y,(len(y),1)).T<np.tile(y,(len(y),1))); return z/n if n>0 else 0
def rmse(y, p): return np.sqrt(mean_squared_error(y, p))

from idr_gat.model.anchor_transfer_v2 import AnchorTransferDTAv2
from idr_gat.model.conplex import ConPlex
from idr_gat.model.esm_dta import EsmDTAModel
from pathlib import Path

models = {}
def try_load(name, cls, path, kwargs, mtype):
    if not Path(path).exists():
        log.info(f'Skipping {name}: {path} not found')
        return
    m = cls(**kwargs).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model_state_dict']); m.eval()
    models[name] = (mtype, m)

try_load('V2-PT5-DTC', AnchorTransferDTAv2, 'models/v2_prostt5_dtc/best_model.pt', {'esm2_dim': 1024}, 'v2')
try_load('V2-PT5-BDB', AnchorTransferDTAv2, 'models/v2_prostt5_bdb/best_model.pt', {'esm2_dim': 1024}, 'v2')
try_load('CPX-PT5-DTC', ConPlex, 'models/conplex_prostt5_dtc/best_model.pt', {'esm2_dim': 1024}, 'cpx')
try_load('CPX-PT5-BDB', ConPlex, 'models/conplex_prostt5_bdb/best_model.pt', {'esm2_dim': 1024}, 'cpx')
try_load('ESM-PT5-DTC', EsmDTAModel, 'models/esm_dta_prostt5_dtc/best_model.pt', {'esm2_dim': 1024}, 'esm')
try_load('ESM-PT5-BDB', EsmDTAModel, 'models/esm_dta_prostt5_bdb/best_model.pt', {'esm2_dim': 1024}, 'esm')
log.info(f'Loaded {len(models)} ProstT5 models')

df = davis.rename(columns={'protein_name':'uniprot_id','drug_smiles':'ligand_smiles','pKd':'pki'})
valid = set(prostt5.keys()) & set(d_seqs.keys())
df = df[df.uniprot_id.isin(valid)].copy()

strongest, second = {}, {}
for smi, grp in df.groupby('ligand_smiles'):
    s = grp.sort_values('pki', ascending=False); p, pk = s.uniprot_id.values, s.pki.values
    strongest[smi] = (p[0], pk[0])
    if len(p) > 1: second[smi] = (p[1], pk[1])
rows, anc_u, anc_p = [], [], []
for i, row in df.iterrows():
    uid, smi = row['uniprot_id'], row['ligand_smiles']
    if smi not in strongest: continue
    au, ap = strongest[smi]
    if au == uid:
        if smi not in second: continue
        au, ap = second[smi]
    if au not in valid: continue
    if ap < 7.0: continue
    rows.append(i); anc_u.append(au); anc_p.append(ap)
sub = df.loc[rows].copy(); sub['anc_uid'] = anc_u; sub['anc_pki'] = anc_p
log.info(f'Davis with anchors>=7: {len(sub)}')

for name, (mtype, model) in models.items():
    preds = []
    for start in range(0, len(sub), 512):
        b = sub.iloc[start:start+512]
        if mtype == 'v2':
            a_e = torch.stack([prostt5[a] for a in b.anc_uid]).to(device)
            q_e = torch.stack([prostt5[u] for u in b.uniprot_id]).to(device)
            dt = torch.tensor([enc_smi(s) for s in b.ligand_smiles], dtype=torch.long, device=device)
            with torch.no_grad(): preds.extend(model(a_e, q_e, dt)['pki_pred'].cpu().tolist())
        elif mtype == 'cpx':
            q_e = torch.stack([prostt5[u] for u in b.uniprot_id]).to(device)
            dt = torch.tensor([enc_smi(s) for s in b.ligand_smiles], dtype=torch.long, device=device)
            with torch.no_grad(): preds.extend(model(q_e, dt)['score'].cpu().tolist())
        elif mtype == 'esm':
            q_e = torch.stack([prostt5[u] for u in b.uniprot_id]).to(device)
            dt = torch.tensor([enc_smi(s) for s in b.ligand_smiles], dtype=torch.long, device=device)
            with torch.no_grad(): preds.extend(model(dt, q_e).cpu().tolist())
    sub[name] = preds
    log.info(f'Predicted {name}')

sub['aq'] = pd.qcut(sub.anc_pki, 4, labels=['Q1','Q2','Q3','Q4'])
mlist = list(models.keys())

log.info(f'\n{"="*80}')
log.info(f'Davis (ProstT5 models) — CI')
log.info(f'{"="*80}')
hdr = f"{'Quartile':<12} {'n':<7}"
for m in mlist: hdr += f" {m:<14}"
log.info(hdr)
log.info('-' * len(hdr))
for q in ['Q1','Q2','Q3','Q4']:
    s = sub[sub.aq == q]; t = s.pki.values
    line = f"{q:<12} {len(s):<7}"
    for m in mlist: line += f" {ci_fn(t, s[m].values):<14.3f}"
    log.info(line)
t = sub.pki.values
line = f"{'Overall':<12} {len(sub):<7}"
for m in mlist: line += f" {ci_fn(t, sub[m].values):<14.3f}"
log.info(line)

log.info(f'\nDavis (ProstT5 models) — RMSE')
hdr = f"{'Quartile':<12} {'n':<7}"
for m in mlist: hdr += f" {m:<14}"
log.info(hdr)
for q in ['Q1','Q2','Q3','Q4']:
    s = sub[sub.aq == q]; t = s.pki.values
    line = f"{q:<12} {len(s):<7}"
    for m in mlist: line += f" {rmse(t, s[m].values):<14.3f}"
    log.info(line)
t = sub.pki.values
line = f"{'Overall':<12} {len(sub):<7}"
for m in mlist: line += f" {rmse(t, sub[m].values):<14.3f}"
log.info(line)
