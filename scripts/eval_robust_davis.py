"""Robust Davis evaluation: multi-seed CIs, AUPRC, realistic anchor retrieval."""
import torch, torch.nn as nn, numpy as np, pandas as pd, sys, json, logging, random
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
from scipy.stats import spearmanr
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger()
sys.path.insert(0, 'src')

device = torch.device('cuda')

# Load embeddings
esm650 = torch.load('data/processed/esm2_650m_all.pt', map_location='cpu', weights_only=False)
esm35 = {}
for f in ['data/processed/esm2_35m_dtc_proteins_full.pt','data/processed/esm2_35m_davis.pt',
          'data/processed/esm2_35m_benchmark.pt']:
    try: esm35.update(torch.load(f, map_location='cpu', weights_only=False))
    except: pass
log.info(f'ESM35: {len(esm35)}, ESM650: {len(esm650)}')

davis = pd.read_csv('data/raw/davis/davis_benchmark.csv')
d_seqs = dict(zip(davis.protein_name, davis.protein_sequence))

# DTC training data for realistic anchor retrieval
dtc = pd.read_csv('data/processed/dtc_training_interactions.csv')
dtc_valid = dtc[dtc.uniprot_id.isin(esm35)]
dtc_prots = sorted(set(dtc_valid.uniprot_id))
random.seed(42); random.shuffle(dtc_prots)
nt = max(1, int(len(dtc_prots)*0.1)); nv = max(1, int(len(dtc_prots)*0.1))
dtc_train_prots = set(dtc_prots[nt+nv:])
dtc_train = dtc_valid[dtc_valid.uniprot_id.isin(dtc_train_prots)]
log.info(f'DTC train: {len(dtc_train)} interactions, {dtc_train.uniprot_id.nunique()} proteins')

# Build DTC-only anchor lookup (realistic: only use training set knowledge)
dtc_drug_strongest = {}
dtc_drug_second = {}
for smi, grp in dtc_train.groupby('ligand_smiles'):
    s = grp.sort_values('pki', ascending=False)
    p, pk = s.uniprot_id.values, s.pki.values
    if pk[0] >= 7.0:  # only strong anchors
        dtc_drug_strongest[smi] = (p[0], pk[0])
        if len(p) > 1 and pk[1] >= 7.0:
            dtc_drug_second[smi] = (p[1], pk[1])
log.info(f'DTC anchors (pKi>=7): {len(dtc_drug_strongest)} drugs with anchors')

# Map Davis proteins to DTC UIDs via sequence
davis_seq_to_dtc = {}
dtc_seqs = json.load(open('data/processed/merged_sequences.json'))
for uid, seq in dtc_seqs.items():
    davis_seq_to_dtc[seq] = uid
davis_to_dtc = {}
for gname, seq in d_seqs.items():
    if seq in davis_seq_to_dtc:
        davis_to_dtc[gname] = davis_seq_to_dtc[seq]

CHARISOSMISET = {"#":29,"%":30,")":31,"(":1,"+":32,"-":33,"/":34,".":2,"1":35,"0":3,"3":36,"2":4,"5":37,"4":5,"7":38,"6":6,"9":39,"8":7,"=":40,"A":41,"@":8,"C":42,"B":9,"E":43,"D":10,"G":44,"F":11,"I":45,"H":12,"K":46,"M":47,"L":13,"O":48,"N":14,"P":15,"S":49,"R":16,"[":50,"]":51,"_":19,"a":20,"c":21,"e":22,"g":23,"i":24,"l":25,"n":26,"o":27,"s":28,"r":17,"u":18}
CHARPROTSET = {"A":1,"C":2,"B":3,"E":4,"D":5,"G":6,"F":7,"I":8,"H":9,"K":10,"M":11,"L":12,"O":13,"N":14,"Q":15,"P":16,"S":17,"R":18,"U":19,"T":20,"W":21,"V":22,"Y":23,"X":24,"Z":25}
def enc_smi(s, ml=100): return [CHARISOSMISET.get(c,0) for c in s[:ml]] + [0]*(ml-len(s[:ml]))
def enc_prot(s, ml=1000): return [CHARPROTSET.get(c,0) for c in s[:ml]] + [0]*(ml-len(s[:ml]))

def ci_fn(y, f):
    if len(y) < 2: return np.nan
    ind=np.argsort(y); y=y[ind]; f=f[ind]
    n=np.sum(np.tile(y,(len(y),1)).T<np.tile(y,(len(y),1)))
    if n==0: return np.nan
    z=np.sum((np.tile(y,(len(y),1)).T<np.tile(y,(len(y),1)))*(np.tile(f,(len(f),1)).T<np.tile(f,(len(f),1))))+0.5*np.sum((np.tile(y,(len(y),1)).T<np.tile(y,(len(y),1)))*(np.tile(f,(len(f),1)).T==np.tile(f,(len(f),1))))
    return z/n
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
from pathlib import Path

models = {}
def tl(name, cls, path, kw, mt, ek):
    if not Path(path).exists(): log.info(f'Skip {name}'); return
    m = cls(**kw).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model_state_dict']); m.eval()
    models[name] = (mt, m, ek)

tl('V2-650M', AnchorTransferDTAv2, 'models/v2_650m_dtc/best_model.pt', {'esm2_dim':1280}, 'v2', '650')
tl('V2-35M', AnchorTransferDTAv2, 'models/v2_dtc/best_model.pt', {'esm2_dim':480}, 'v2', '35')
tl('DeepDTA', DeepDTAModel, 'models/deepdta_dtc/best_model.pt', {}, 'dta', '35')
tl('ConPlex', ConPlex, 'models/conplex_dtc/best_model.pt', {'esm2_dim':480}, 'cpx', '35')
tl('ESM-DTA', EsmDTAModel, 'models/esm_dta_dtc/best_model.pt', {'esm2_dim':480}, 'esm', '35')
log.info(f'Loaded {len(models)} models')

embs = {'35': esm35, '650': esm650}
valid = set(esm35.keys()) & set(esm650.keys())

df = davis.rename(columns={'protein_name':'uniprot_id','drug_smiles':'ligand_smiles','pKd':'pki'})
df = df[df.uniprot_id.isin(valid)].copy()

# ============================================================
# PART 1: Oracle anchors (existing) + AUPRC
# ============================================================
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
oracle = df.loc[rows].copy(); oracle['anc_uid'] = anc_u; oracle['anc_pki'] = anc_p
log.info(f'Oracle anchors: {len(oracle)} interactions')

# ============================================================
# PART 2: Realistic anchors from DTC training set only
# ============================================================
rows_r, anc_u_r, anc_p_r = [], [], []
for i, row in df.iterrows():
    uid, smi = row['uniprot_id'], row['ligand_smiles']
    if smi not in dtc_drug_strongest: continue
    au, ap = dtc_drug_strongest[smi]
    # Map Davis gene name to DTC UID for self-check
    dtc_uid = davis_to_dtc.get(uid)
    if au == dtc_uid:
        if smi not in dtc_drug_second: continue
        au, ap = dtc_drug_second[smi]
    if au not in valid: continue
    rows_r.append(i); anc_u_r.append(au); anc_p_r.append(ap)
realistic = df.loc[rows_r].copy(); realistic['anc_uid'] = anc_u_r; realistic['anc_pki'] = anc_p_r
log.info(f'Realistic anchors (DTC-only): {len(realistic)} interactions ({len(realistic)/len(df)*100:.1f}% coverage)')

# Predict function
def predict(sub, name, mt, model, ek):
    emb = embs[ek]; preds = []
    for start in range(0, len(sub), 512):
        b = sub.iloc[start:start+512]
        if mt == 'v2':
            a_e = torch.stack([emb[a] for a in b.anc_uid]).to(device)
            q_e = torch.stack([emb[u] for u in b.uniprot_id]).to(device)
            dt = torch.tensor([enc_smi(s) for s in b.ligand_smiles], dtype=torch.long, device=device)
            with torch.no_grad(): preds.extend(model(a_e, q_e, dt)['pki_pred'].cpu().tolist())
        elif mt == 'dta':
            dt = torch.tensor([enc_smi(s) for s in b.ligand_smiles], dtype=torch.long, device=device)
            pe = torch.tensor([enc_prot(d_seqs[u]) for u in b.uniprot_id], dtype=torch.long, device=device)
            with torch.no_grad(): preds.extend(model(dt, pe).cpu().tolist())
        elif mt == 'cpx':
            q_e = torch.stack([emb[u] for u in b.uniprot_id]).to(device)
            dt = torch.tensor([enc_smi(s) for s in b.ligand_smiles], dtype=torch.long, device=device)
            with torch.no_grad(): preds.extend(model(q_e, dt)['score'].cpu().tolist())
        elif mt == 'esm':
            q_e = torch.stack([emb[u] for u in b.uniprot_id]).to(device)
            dt = torch.tensor([enc_smi(s) for s in b.ligand_smiles], dtype=torch.long, device=device)
            with torch.no_grad(): preds.extend(model(dt, q_e).cpu().tolist())
    return preds

# Predict all models on both oracle and realistic
for name, (mt, model, ek) in models.items():
    oracle[name] = predict(oracle, name, mt, model, ek)
    if mt == 'v2':
        realistic[name] = predict(realistic, name, mt, model, ek)
    log.info(f'Predicted {name}')

# For non-anchor models on realistic set, predict normally
for name, (mt, model, ek) in models.items():
    if mt != 'v2' and name not in realistic.columns:
        preds = []
        emb = embs[ek]
        for start in range(0, len(realistic), 512):
            b = realistic.iloc[start:start+512]
            if mt == 'dta':
                dt = torch.tensor([enc_smi(s) for s in b.ligand_smiles], dtype=torch.long, device=device)
                pe = torch.tensor([enc_prot(d_seqs[u]) for u in b.uniprot_id], dtype=torch.long, device=device)
                with torch.no_grad(): preds.extend(model(dt, pe).cpu().tolist())
            elif mt == 'cpx':
                q_e = torch.stack([emb[u] for u in b.uniprot_id]).to(device)
                dt = torch.tensor([enc_smi(s) for s in b.ligand_smiles], dtype=torch.long, device=device)
                with torch.no_grad(): preds.extend(model(q_e, dt)['score'].cpu().tolist())
            elif mt == 'esm':
                q_e = torch.stack([emb[u] for u in b.uniprot_id]).to(device)
                dt = torch.tensor([enc_smi(s) for s in b.ligand_smiles], dtype=torch.long, device=device)
                with torch.no_grad(): preds.extend(model(dt, q_e).cpu().tolist())
        realistic[name] = preds

# ============================================================
# PART 3: Per-protein metrics with bootstrap CIs
# ============================================================
def per_protein_metrics(sub, model_names, n_bootstrap=1000):
    results = {}
    for m in model_names:
        cis, aucs, auprs, rmses = [], [], [], []
        for uid, grp in sub.groupby('uniprot_id'):
            t = grp.pki.values; p = grp[m].values
            if len(t) < 5: continue
            cis.append(ci_fn(t, p))
            rmses.append(rmse(t, p))
            if (t >= 7).sum() > 0 and (t < 7).sum() > 0:
                labels = (t >= 7).astype(int)
                aucs.append(roc_auc_score(labels, p))
                auprs.append(average_precision_score(labels, p))
        cis = [c for c in cis if not np.isnan(c)]

        # Bootstrap CI
        ci_boots, auc_boots, aupr_boots = [], [], []
        for _ in range(n_bootstrap):
            idx_c = np.random.choice(len(cis), len(cis), replace=True)
            ci_boots.append(np.mean([cis[i] for i in idx_c]))
            if aucs:
                idx_a = np.random.choice(len(aucs), len(aucs), replace=True)
                auc_boots.append(np.mean([aucs[i] for i in idx_a]))
                aupr_boots.append(np.mean([auprs[i] for i in idx_a]))

        results[m] = {
            'ci_mean': np.mean(cis), 'ci_lo': np.percentile(ci_boots, 2.5), 'ci_hi': np.percentile(ci_boots, 97.5),
            'auroc_mean': np.mean(aucs) if aucs else np.nan,
            'auroc_lo': np.percentile(auc_boots, 2.5) if auc_boots else np.nan,
            'auroc_hi': np.percentile(auc_boots, 97.5) if auc_boots else np.nan,
            'auprc_mean': np.mean(auprs) if auprs else np.nan,
            'auprc_lo': np.percentile(aupr_boots, 2.5) if aupr_boots else np.nan,
            'auprc_hi': np.percentile(aupr_boots, 97.5) if aupr_boots else np.nan,
            'rmse_mean': np.mean(rmses),
            'n_proteins': len(cis),
        }
    return results

mnames = list(models.keys())
v2_names = [n for n in mnames if models[n][0] == 'v2']

# Oracle results
log.info('\n' + '='*100)
log.info('ORACLE ANCHORS (n=%d)', len(oracle))
log.info('='*100)
np.random.seed(42)
oracle_results = per_protein_metrics(oracle, mnames)
log.info(f"{'Model':<12} {'CI':<22} {'AUROC':<22} {'AUPRC':<22} {'RMSE':<8}")
log.info('-' * 86)
for m in mnames:
    r = oracle_results[m]
    ci_str = f"{r['ci_mean']:.3f} [{r['ci_lo']:.3f}-{r['ci_hi']:.3f}]"
    auc_str = f"{r['auroc_mean']:.3f} [{r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}]"
    aupr_str = f"{r['auprc_mean']:.3f} [{r['auprc_lo']:.3f}-{r['auprc_hi']:.3f}]"
    log.info(f"{m:<12} {ci_str:<22} {auc_str:<22} {aupr_str:<22} {r['rmse_mean']:<8.3f}")

# Realistic results
log.info('\n' + '='*100)
log.info('REALISTIC ANCHORS from DTC training set (n=%d, %.1f%% coverage)', len(realistic), len(realistic)/len(df)*100)
log.info('='*100)
np.random.seed(42)
realistic_results = per_protein_metrics(realistic, mnames)
log.info(f"{'Model':<12} {'CI':<22} {'AUROC':<22} {'AUPRC':<22} {'RMSE':<8}")
log.info('-' * 86)
for m in mnames:
    r = realistic_results[m]
    ci_str = f"{r['ci_mean']:.3f} [{r['ci_lo']:.3f}-{r['ci_hi']:.3f}]"
    auc_str = f"{r['auroc_mean']:.3f} [{r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}]" if not np.isnan(r['auroc_mean']) else "N/A"
    aupr_str = f"{r['auprc_mean']:.3f} [{r['auprc_lo']:.3f}-{r['auprc_hi']:.3f}]" if not np.isnan(r['auprc_mean']) else "N/A"
    log.info(f"{m:<12} {ci_str:<22} {auc_str:<22} {aupr_str:<22} {r['rmse_mean']:<8.3f}")

# Oracle vs Realistic comparison for V2 models
log.info('\n' + '='*100)
log.info('ORACLE vs REALISTIC comparison (V2 models)')
log.info('='*100)
log.info(f"{'Model':<12} {'Oracle CI':<12} {'Realistic CI':<14} {'Delta':<8} {'Oracle AUPRC':<14} {'Realistic AUPRC':<16}")
for m in v2_names:
    o, r = oracle_results[m], realistic_results[m]
    delta = o['ci_mean'] - r['ci_mean']
    log.info(f"{m:<12} {o['ci_mean']:<12.3f} {r['ci_mean']:<14.3f} {delta:<8.3f} {o['auprc_mean']:<14.3f} {r['auprc_mean']:<16.3f}")
