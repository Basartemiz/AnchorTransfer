"""Per-protein CI, RMSE, AUROC distributions on Davis for all models."""
import torch, torch.nn as nn, numpy as np, pandas as pd, sys, json, logging
from sklearn.metrics import roc_auc_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
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
prostt5 = torch.load('data/processed/prostt5_all.pt', map_location='cpu', weights_only=False)
log.info(f'ESM35: {len(esm35)}, ESM650: {len(esm650)}, ProstT5: {len(prostt5)}')

davis = pd.read_csv('data/raw/davis/davis_benchmark.csv')
d_seqs = dict(zip(davis.protein_name, davis.protein_sequence))

CHARISOSMISET = {"#":29,"%":30,")":31,"(":1,"+":32,"-":33,"/":34,".":2,"1":35,"0":3,"3":36,"2":4,"5":37,"4":5,"7":38,"6":6,"9":39,"8":7,"=":40,"A":41,"@":8,"C":42,"B":9,"E":43,"D":10,"G":44,"F":11,"I":45,"H":12,"K":46,"M":47,"L":13,"O":48,"N":14,"P":15,"S":49,"R":16,"[":50,"]":51,"_":19,"a":20,"c":21,"e":22,"g":23,"i":24,"l":25,"n":26,"o":27,"s":28,"r":17,"u":18}
CHARPROTSET = {"A":1,"C":2,"B":3,"E":4,"D":5,"G":6,"F":7,"I":8,"H":9,"K":10,"M":11,"L":12,"O":13,"N":14,"Q":15,"P":16,"S":17,"R":18,"U":19,"T":20,"W":21,"V":22,"Y":23,"X":24,"Z":25}
def enc_smi(s, ml=100): return [CHARISOSMISET.get(c,0) for c in s[:ml]] + [0]*(ml-len(s[:ml]))
def enc_prot(s, ml=1000): return [CHARPROTSET.get(c,0) for c in s[:ml]] + [0]*(ml-len(s[:ml]))

def ci_fn(y, f):
    if len(y) < 2: return np.nan
    ind=np.argsort(y); y=y[ind]; f=f[ind]
    n=np.sum(np.tile(y,(len(y),1)).T<np.tile(y,(len(y),1)))
    if n == 0: return np.nan
    z=np.sum((np.tile(y,(len(y),1)).T<np.tile(y,(len(y),1)))*(np.tile(f,(len(f),1)).T<np.tile(f,(len(f),1))))+0.5*np.sum((np.tile(y,(len(y),1)).T<np.tile(y,(len(y),1)))*(np.tile(f,(len(f),1)).T==np.tile(f,(len(f),1))))
    return z/n

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
from pathlib import Path

models = {}
def try_load(name, cls, path, kwargs, mtype, emb_key):
    if not Path(path).exists():
        log.info(f'Skip {name}'); return
    m = cls(**kwargs).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model_state_dict']); m.eval()
    models[name] = (mtype, m, emb_key)

# DTC models
try_load('V2-650M', AnchorTransferDTAv2, 'models/v2_650m_dtc/best_model.pt', {'esm2_dim':1280}, 'v2', '650')
try_load('V2-35M', AnchorTransferDTAv2, 'models/v2_dtc/best_model.pt', {'esm2_dim':480}, 'v2', '35')
try_load('DeepDTA', DeepDTAModel, 'models/deepdta_dtc/best_model.pt', {}, 'dta', '35')
try_load('ConPlex', ConPlex, 'models/conplex_dtc/best_model.pt', {'esm2_dim':480}, 'cpx', '35')
try_load('ESM-DTA', EsmDTAModel, 'models/esm_dta_dtc/best_model.pt', {'esm2_dim':480}, 'esm', '35')
try_load('V2-PT5', AnchorTransferDTAv2, 'models/v2_prostt5_dtc/best_model.pt', {'esm2_dim':1024}, 'v2', 'pt5')
log.info(f'Loaded {len(models)} models')

emb_map = {'35': esm35, '650': esm650, 'pt5': prostt5}

df = davis.rename(columns={'protein_name':'uniprot_id','drug_smiles':'ligand_smiles','pKd':'pki'})
valid = set(esm35.keys()) & set(esm650.keys()) & set(prostt5.keys()) & set(d_seqs.keys())
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
log.info(f'Davis: {len(sub)} interactions, {sub.uniprot_id.nunique()} proteins')

# Predict
for name, (mtype, model, emb_key) in models.items():
    emb = emb_map[emb_key]
    preds = []
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

# Per-protein metrics
model_names = list(models.keys())
per_protein = {m: {'ci': [], 'rmse': [], 'auroc': []} for m in model_names}

for uid, grp in sub.groupby('uniprot_id'):
    t = grp.pki.values
    if len(t) < 5: continue
    has_binders = (t >= 7.0).sum() > 0 and (t < 7.0).sum() > 0
    for m in model_names:
        p = grp[m].values
        per_protein[m]['ci'].append(ci_fn(t, p))
        per_protein[m]['rmse'].append(np.sqrt(mean_squared_error(t, p)))
        if has_binders:
            per_protein[m]['auroc'].append(roc_auc_score((t >= 7.0).astype(int), p))
        else:
            per_protein[m]['auroc'].append(np.nan)

# Print summary
log.info(f'\n=== Per-protein metrics (n={len(per_protein[model_names[0]]["ci"])} proteins) ===')
log.info(f'{"Model":<12} {"CI mean":<10} {"CI med":<10} {"RMSE mean":<10} {"RMSE med":<10} {"AUC mean":<10} {"AUC med":<10}')
log.info('-' * 70)
for m in model_names:
    ci_vals = [x for x in per_protein[m]['ci'] if not np.isnan(x)]
    rmse_vals = per_protein[m]['rmse']
    auc_vals = [x for x in per_protein[m]['auroc'] if not np.isnan(x)]
    log.info(f'{m:<12} {np.mean(ci_vals):<10.3f} {np.median(ci_vals):<10.3f} {np.mean(rmse_vals):<10.3f} {np.median(rmse_vals):<10.3f} {np.mean(auc_vals):<10.3f} {np.median(auc_vals):<10.3f}')

# === FIGURES ===
colors = {'V2-650M': '#2166ac', 'V2-35M': '#67a9cf', 'DeepDTA': '#ef8a62',
          'ConPlex': '#b2182b', 'ESM-DTA': '#d6604d', 'V2-PT5': '#4393c3'}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: CI distribution
ax = axes[0]
ci_data = []
ci_labels = []
for m in model_names:
    vals = [x for x in per_protein[m]['ci'] if not np.isnan(x)]
    ci_data.append(vals)
    ci_labels.append(m)
parts = ax.violinplot(ci_data, positions=range(len(model_names)), showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors.get(model_names[i], 'gray'))
    pc.set_alpha(0.7)
parts['cmedians'].set_color('black')
parts['cmeans'].set_color('red')
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('Concordance Index', fontsize=12)
ax.set_title('(A) Per-Protein CI Distribution', fontsize=13, fontweight='bold')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='random')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
# Add mean annotations
for i, m in enumerate(model_names):
    vals = [x for x in per_protein[m]['ci'] if not np.isnan(x)]
    ax.text(i, np.mean(vals) + 0.02, f'{np.mean(vals):.3f}', ha='center', fontsize=8, color='red')

# Panel B: RMSE distribution
ax = axes[1]
rmse_data = [per_protein[m]['rmse'] for m in model_names]
parts = ax.violinplot(rmse_data, positions=range(len(model_names)), showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors.get(model_names[i], 'gray'))
    pc.set_alpha(0.7)
parts['cmedians'].set_color('black')
parts['cmeans'].set_color('red')
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('RMSE (pKi units)', fontsize=12)
ax.set_title('(B) Per-Protein RMSE Distribution', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, m in enumerate(model_names):
    ax.text(i, np.mean(per_protein[m]['rmse']) + 0.05, f'{np.mean(per_protein[m]["rmse"]):.3f}', ha='center', fontsize=8, color='red')

# Panel C: AUROC distribution
ax = axes[2]
auc_data = []
for m in model_names:
    vals = [x for x in per_protein[m]['auroc'] if not np.isnan(x)]
    auc_data.append(vals)
parts = ax.violinplot(auc_data, positions=range(len(model_names)), showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors.get(model_names[i], 'gray'))
    pc.set_alpha(0.7)
parts['cmedians'].set_color('black')
parts['cmeans'].set_color('red')
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('AUROC (binder $\\geq$ 7 pKi)', fontsize=12)
ax.set_title('(C) Per-Protein AUROC Distribution', fontsize=13, fontweight='bold')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='random')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
for i, m in enumerate(model_names):
    vals = [x for x in per_protein[m]['auroc'] if not np.isnan(x)]
    ax.text(i, np.mean(vals) + 0.02, f'{np.mean(vals):.3f}', ha='center', fontsize=8, color='red')

plt.suptitle('Davis Kinase Benchmark: Per-Protein Metric Distributions', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('paper/figures/fig_davis_per_protein_distributions.png', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/fig_davis_per_protein_distributions.pdf', bbox_inches='tight')
log.info('Saved distribution figures')

# Also save raw data as JSON for paper
import json as jsonlib
results = {}
for m in model_names:
    results[m] = {
        'ci': [float(x) for x in per_protein[m]['ci'] if not np.isnan(x)],
        'rmse': [float(x) for x in per_protein[m]['rmse']],
        'auroc': [float(x) for x in per_protein[m]['auroc'] if not np.isnan(x)],
    }
with open('paper/figures/davis_per_protein_metrics.json', 'w') as f:
    jsonlib.dump(results, f)
log.info('Saved metrics JSON')
