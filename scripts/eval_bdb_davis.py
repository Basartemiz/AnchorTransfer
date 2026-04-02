"""Eval all DTC and BDB models on Davis with overlap filtering."""
import torch, torch.nn as nn, numpy as np, pandas as pd, sys, json, random, logging
from sklearn.metrics import roc_auc_score, mean_squared_error
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger()
sys.path.insert(0, 'src')

device = torch.device('cuda')

esm35 = {}
for f in ['data/processed/esm2_35m_dtc_proteins_full.pt','data/processed/esm2_35m_davis.pt',
          'data/processed/esm2_35m_pdbbind.pt','data/processed/esm2_35m_benchmark.pt']:
    try: esm35.update(torch.load(f, map_location='cpu', weights_only=False))
    except: pass
esm650 = torch.load('data/processed/esm2_650m_all.pt', map_location='cpu', weights_only=False)
log.info(f'ESM35: {len(esm35)}, ESM650: {len(esm650)}')

davis = pd.read_csv('data/raw/davis/davis_benchmark.csv')
d_seqs = dict(zip(davis.protein_name, davis.protein_sequence))

CHARISOSMISET = {"#":29,"%":30,")":31,"(":1,"+":32,"-":33,"/":34,".":2,"1":35,"0":3,"3":36,"2":4,"5":37,"4":5,"7":38,"6":6,"9":39,"8":7,"=":40,"A":41,"@":8,"C":42,"B":9,"E":43,"D":10,"G":44,"F":11,"I":45,"H":12,"K":46,"M":47,"L":13,"O":48,"N":14,"P":15,"S":49,"R":16,"[":50,"]":51,"_":19,"a":20,"c":21,"e":22,"g":23,"i":24,"l":25,"n":26,"o":27,"s":28,"r":17,"u":18}
CHARPROTSET = {"A":1,"C":2,"B":3,"E":4,"D":5,"G":6,"F":7,"I":8,"H":9,"K":10,"M":11,"L":12,"O":13,"N":14,"Q":15,"P":16,"S":17,"R":18,"U":19,"T":20,"W":21,"V":22,"Y":23,"X":24,"Z":25}
def enc_smi(s, ml=100): return [CHARISOSMISET.get(c,0) for c in s[:ml]] + [0]*(ml-len(s[:ml]))
def enc_prot(s, ml=1000): return [CHARPROTSET.get(c,0) for c in s[:ml]] + [0]*(ml-len(s[:ml]))
def ci_fn(y, f):
    ind=np.argsort(y); y=y[ind]; f=f[ind]
    z=np.sum((np.tile(y,(len(y),1)).T<np.tile(y,(len(y),1)))*(np.tile(f,(len(f),1)).T<np.tile(f,(len(f),1))))+0.5*np.sum((np.tile(y,(len(y),1)).T<np.tile(y,(len(y),1)))*(np.tile(f,(len(f),1)).T==np.tile(f,(len(f),1))))
    n=np.sum(np.tile(y,(len(y),1)).T<np.tile(y,(len(y),1))); return z/n if n>0 else 0
def rmse(y, p): return np.sqrt(mean_squared_error(y, p))

class DeepDTAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.smiles_embed = nn.Embedding(66, 128, padding_idx=0)
        self.protein_embed = nn.Embedding(26, 128, padding_idx=0)
        self.sc1=nn.Conv1d(128,32,8); self.sc2=nn.Conv1d(32,64,8); self.sc3=nn.Conv1d(64,96,8)
        self.pc1=nn.Conv1d(128,32,8); self.pc2=nn.Conv1d(32,64,8); self.pc3=nn.Conv1d(64,96,8)
        self.pool=nn.AdaptiveMaxPool1d(1)
        self.fc1=nn.Linear(192,1024); self.fc2=nn.Linear(1024,1024); self.fc3=nn.Linear(1024,512); self.out=nn.Linear(512,1)
        self.relu=nn.ReLU(); self.drop=nn.Dropout(0.1)
    def forward(self, drug, prot):
        d=self.relu(self.sc1(self.smiles_embed(drug).permute(0,2,1))); d=self.relu(self.sc2(d)); d=self.pool(self.relu(self.sc3(d))).squeeze(-1)
        p=self.relu(self.pc1(self.protein_embed(prot).permute(0,2,1))); p=self.relu(self.pc2(p)); p=self.pool(self.relu(self.pc3(p))).squeeze(-1)
        x=torch.cat([d,p],1); x=self.drop(self.relu(self.fc1(x))); x=self.drop(self.relu(self.fc2(x))); x=self.drop(self.relu(self.fc3(x)))
        return self.out(x).squeeze(-1)

from idr_gat.model.anchor_transfer_v2 import AnchorTransferDTAv2
from idr_gat.model.conplex import ConPlex
from idr_gat.model.esm_dta import EsmDTAModel

models = {}
def load_v2(name, path, dim):
    m = AnchorTransferDTAv2(esm2_dim=dim).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model_state_dict']); m.eval()
    models[name] = ('v2', m, dim)
def load_dta(name, path):
    m = DeepDTAModel().to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model_state_dict']); m.eval()
    models[name] = ('dta', m, 0)
def load_cpx(name, path, dim):
    m = ConPlex(esm2_dim=dim).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model_state_dict']); m.eval()
    models[name] = ('cpx', m, dim)
def load_esm(name, path, dim):
    m = EsmDTAModel(esm2_dim=dim).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model_state_dict']); m.eval()
    models[name] = ('esm', m, dim)

load_v2('V2-650M-DTC', 'models/v2_650m_dtc/best_model.pt', 1280)
load_v2('V2-35M-DTC', 'models/v2_dtc/best_model.pt', 480)
load_dta('DTA-DTC', 'models/deepdta_dtc/best_model.pt')
load_cpx('CPX-DTC', 'models/conplex_dtc/best_model.pt', 480)
load_esm('ESM-DTA-DTC', 'models/esm_dta_dtc/best_model.pt', 480)
load_v2('V2-35M-BDB', 'models/v2_bdb/best_model.pt', 480)
load_v2('V2-650M-BDB', 'models/v2_650m_bdb/best_model.pt', 1280)
load_dta('DTA-BDB', 'models/deepdta_bdb/best_model.pt')
load_cpx('CPX-BDB', 'models/conplex_bdb/best_model.pt', 480)
log.info(f'Loaded {len(models)} models')

df = davis.rename(columns={'protein_name':'uniprot_id','drug_smiles':'ligand_smiles','pKd':'pki'})
valid = set(esm35.keys()) & set(esm650.keys()) & set(d_seqs.keys())
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

def get_emb(dim):
    return esm650 if dim == 1280 else esm35

for name, (mtype, model, dim) in models.items():
    preds = []
    emb = get_emb(dim)
    for start in range(0, len(sub), 512):
        b = sub.iloc[start:start+512]
        if mtype == 'v2':
            a_e = torch.stack([emb[a] for a in b.anc_uid]).to(device)
            q_e = torch.stack([emb[u] for u in b.uniprot_id]).to(device)
            dt = torch.tensor([enc_smi(s) for s in b.ligand_smiles], dtype=torch.long, device=device)
            with torch.no_grad(): preds.extend(model(a_e, q_e, dt)['pki_pred'].cpu().tolist())
        elif mtype == 'dta':
            dt = torch.tensor([enc_smi(s) for s in b.ligand_smiles], dtype=torch.long, device=device)
            pe = torch.tensor([enc_prot(d_seqs[u]) for u in b.uniprot_id], dtype=torch.long, device=device)
            with torch.no_grad(): preds.extend(model(dt, pe).cpu().tolist())
        elif mtype == 'cpx':
            q_e = torch.stack([emb[u] for u in b.uniprot_id]).to(device)
            dt = torch.tensor([enc_smi(s) for s in b.ligand_smiles], dtype=torch.long, device=device)
            with torch.no_grad(): preds.extend(model(q_e, dt)['score'].cpu().tolist())
        elif mtype == 'esm':
            q_e = torch.stack([emb[u] for u in b.uniprot_id]).to(device)
            dt = torch.tensor([enc_smi(s) for s in b.ligand_smiles], dtype=torch.long, device=device)
            with torch.no_grad(): preds.extend(model(dt, q_e).cpu().tolist())
    sub[name] = preds
    log.info(f'Predicted {name}')

sub['aq'] = pd.qcut(sub.anc_pki, 4, labels=['Q1','Q2','Q3','Q4'])

for label, mlist in [('DTC-trained', [n for n in models if 'DTC' in n]),
                      ('BDB-trained', [n for n in models if 'BDB' in n])]:
    log.info(f'\n{"="*80}')
    log.info(f'Davis ({label}) — CI')
    log.info(f'{"="*80}')
    hdr = f"{'Quartile':<12} {'n':<7}"
    for m in mlist: hdr += f" {m:<16}"
    log.info(hdr)
    log.info('-' * len(hdr))
    for q in ['Q1','Q2','Q3','Q4']:
        s = sub[sub.aq == q]; t = s.pki.values
        line = f"{q:<12} {len(s):<7}"
        for m in mlist: line += f" {ci_fn(t, s[m].values):<16.3f}"
        log.info(line)
    t = sub.pki.values
    line = f"{'Overall':<12} {len(sub):<7}"
    for m in mlist: line += f" {ci_fn(t, sub[m].values):<16.3f}"
    log.info(line)

    log.info(f'\nDavis ({label}) — RMSE')
    for q in ['Q1','Q2','Q3','Q4']:
        s = sub[sub.aq == q]; t = s.pki.values
        line = f"{q:<12} {len(s):<7}"
        for m in mlist: line += f" {rmse(t, s[m].values):<16.3f}"
        log.info(line)
    t = sub.pki.values
    line = f"{'Overall':<12} {len(sub):<7}"
    for m in mlist: line += f" {rmse(t, sub[m].values):<16.3f}"
    log.info(line)
