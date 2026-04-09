"""Robust Davis eval: realistic anchors via Tanimoto, bootstrap CIs, AUPRC."""
import torch, torch.nn as nn, numpy as np, pandas as pd, sys, json, logging, random
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger()
sys.path.insert(0, 'src')

device = torch.device('cuda')

esm650 = torch.load('data/processed/esm2_650m_all.pt', map_location='cpu', weights_only=False)
esm35 = {}
for f in ['data/processed/esm2_35m_dtc_proteins_full.pt','data/processed/esm2_35m_davis.pt',
          'data/processed/esm2_35m_benchmark.pt']:
    try: esm35.update(torch.load(f, map_location='cpu', weights_only=False))
    except: pass

davis = pd.read_csv('data/raw/davis/davis_benchmark.csv')
d_seqs = dict(zip(davis.protein_name, davis.protein_sequence))

dtc = pd.read_csv('data/processed/dtc_training_interactions.csv')
dtc_valid = dtc[dtc.uniprot_id.isin(esm35)]
dtc_prots = sorted(set(dtc_valid.uniprot_id))
random.seed(42); random.shuffle(dtc_prots)
nt = max(1, int(len(dtc_prots)*0.1)); nv = max(1, int(len(dtc_prots)*0.1))
dtc_train_prots = set(dtc_prots[nt+nv:])
dtc_train = dtc_valid[dtc_valid.uniprot_id.isin(dtc_train_prots)]
log.info(f'DTC train: {len(dtc_train)} interactions, {dtc_train.uniprot_id.nunique()} proteins')

# Build DTC drug->strongest binder lookup (pKi >= 7 only)
dtc_drug_to_anchor = {}
for smi, grp in dtc_train.groupby('ligand_smiles'):
    s = grp.sort_values('pki', ascending=False)
    if s.pki.values[0] >= 7.0 and s.uniprot_id.values[0] in esm650:
        dtc_drug_to_anchor[smi] = (s.uniprot_id.values[0], s.pki.values[0])
log.info(f'DTC drugs with anchors (pKi>=7): {len(dtc_drug_to_anchor)}')

# Compute Tanimoto similarity: find nearest DTC drug for each Davis drug
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

log.info('Computing Morgan fingerprints...')
davis_drugs = list(davis.drug_smiles.unique())
dtc_anchor_drugs = list(dtc_drug_to_anchor.keys())

def smi_to_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

davis_fps = {s: smi_to_fp(s) for s in davis_drugs}
davis_fps = {k: v for k, v in davis_fps.items() if v is not None}

dtc_fps = {}
for i, s in enumerate(dtc_anchor_drugs):
    fp = smi_to_fp(s)
    if fp is not None: dtc_fps[s] = fp
    if (i+1) % 10000 == 0: log.info(f'  FP progress: {i+1}/{len(dtc_anchor_drugs)}')
log.info(f'Davis FPs: {len(davis_fps)}, DTC anchor FPs: {len(dtc_fps)}')

# Find nearest DTC drug for each Davis drug
dtc_fp_list = list(dtc_fps.items())
nearest_dtc = {}
for d_smi, d_fp in davis_fps.items():
    best_sim, best_smi = 0, None
    for c_smi, c_fp in dtc_fp_list:
        sim = DataStructs.TanimotoSimilarity(d_fp, c_fp)
        if sim > best_sim:
            best_sim = sim; best_smi = c_smi
    nearest_dtc[d_smi] = (best_smi, best_sim)
    log.info(f'  Davis drug -> nearest DTC: sim={best_sim:.3f}')

log.info('Nearest DTC drug similarities:')
sims = [v[1] for v in nearest_dtc.values()]
log.info(f'  Min={min(sims):.3f} Median={np.median(sims):.3f} Mean={np.mean(sims):.3f} Max={max(sims):.3f}')

CHARISOSMISET = {"#":29,"%":30,")":31,"(":1,"+":32,"-":33,"/":34,".":2,"1":35,"0":3,"3":36,"2":4,"5":37,"4":5,"7":38,"6":6,"9":39,"8":7,"=":40,"A":41,"@":8,"C":42,"B":9,"E":43,"D":10,"G":44,"F":11,"I":45,"H":12,"K":46,"M":47,"L":13,"O":48,"N":14,"P":15,"S":49,"R":16,"[":50,"]":51,"_":19,"a":20,"c":21,"e":22,"g":23,"i":24,"l":25,"n":26,"o":27,"s":28,"r":17,"u":18}
CHARPROTSET = {"A":1,"C":2,"B":3,"E":4,"D":5,"G":6,"F":7,"I":8,"H":9,"K":10,"M":11,"L":12,"O":13,"N":14,"Q":15,"P":16,"S":17,"R":18,"U":19,"T":20,"W":21,"V":22,"Y":23,"X":24,"Z":25}
def enc_smi(s, ml=100): return [CHARISOSMISET.get(c,0) for c in s[:ml]] + [0]*(ml-len(s[:ml]))
def enc_prot(s, ml=1000): return [CHARPROTSET.get(c,0) for c in s[:ml]] + [0]*(ml-len(s[:ml]))
def ci_fn(y, f):
    if len(y)<2: return np.nan
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

from anchor_transfer.model.anchor_transfer_v2 import AnchorTransferDTAv2
from anchor_transfer.model.conplex import ConPlex
from anchor_transfer.model.esm_dta import EsmDTAModel
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
tl('ESM-DTA', EsmDTAModel, 'models/esm_dta_dtc/best_model.pt', {'esm2_dim':480}, 'esm', '35')
log.info(f'Loaded {len(models)} models')

embs = {'35': esm35, '650': esm650}
valid = set(esm35.keys()) & set(esm650.keys())

df = davis.rename(columns={'protein_name':'uniprot_id','drug_smiles':'ligand_smiles','pKd':'pki'})
df = df[df.uniprot_id.isin(valid)].copy()

# Map Davis gene names to DTC UIDs via sequence for self-check
dtc_seqs = json.load(open('data/processed/merged_sequences.json'))
seq_to_uid = {v: k for k, v in dtc_seqs.items()}
davis_to_dtc = {g: seq_to_uid.get(s) for g, s in d_seqs.items()}

# Oracle anchors
strongest, second = {}, {}
for smi, grp in df.groupby('ligand_smiles'):
    s = grp.sort_values('pki', ascending=False); p, pk = s.uniprot_id.values, s.pki.values
    strongest[smi] = (p[0], pk[0])
    if len(p) > 1: second[smi] = (p[1], pk[1])
rows_o, anc_u_o, anc_p_o = [], [], []
for i, row in df.iterrows():
    uid, smi = row['uniprot_id'], row['ligand_smiles']
    if smi not in strongest: continue
    au, ap = strongest[smi]
    if au == uid:
        if smi not in second: continue
        au, ap = second[smi]
    if au not in valid: continue
    if ap < 7.0: continue
    rows_o.append(i); anc_u_o.append(au); anc_p_o.append(ap)
oracle = df.loc[rows_o].copy(); oracle['anc_uid'] = anc_u_o; oracle['anc_pki'] = anc_p_o
log.info(f'Oracle: {len(oracle)} interactions')

# Realistic anchors via Tanimoto nearest DTC drug
rows_r, anc_u_r, anc_p_r, tani_sims = [], [], [], []
for i, row in df.iterrows():
    uid, smi = row['uniprot_id'], row['ligand_smiles']
    if smi not in nearest_dtc: continue
    dtc_smi, tani = nearest_dtc[smi]
    if dtc_smi not in dtc_drug_to_anchor: continue
    au, ap = dtc_drug_to_anchor[dtc_smi]
    # Self-check: skip if anchor is the query protein
    dtc_uid = davis_to_dtc.get(uid)
    if au == dtc_uid: continue
    if au not in valid: continue
    rows_r.append(i); anc_u_r.append(au); anc_p_r.append(ap); tani_sims.append(tani)
realistic = df.loc[rows_r].copy()
realistic['anc_uid'] = anc_u_r; realistic['anc_pki'] = anc_p_r; realistic['tanimoto'] = tani_sims
log.info(f'Realistic (Tanimoto): {len(realistic)} interactions ({len(realistic)/len(df)*100:.1f}% coverage)')
log.info(f'Tanimoto sims: min={min(tani_sims):.3f} median={np.median(tani_sims):.3f} mean={np.mean(tani_sims):.3f}')

# Predict
def predict_sub(sub, name, mt, model, ek):
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
        elif mt == 'esm':
            q_e = torch.stack([emb[u] for u in b.uniprot_id]).to(device)
            dt = torch.tensor([enc_smi(s) for s in b.ligand_smiles], dtype=torch.long, device=device)
            with torch.no_grad(): preds.extend(model(dt, q_e).cpu().tolist())
    return preds

for name, (mt, model, ek) in models.items():
    oracle[name] = predict_sub(oracle, name, mt, model, ek)
    realistic[name] = predict_sub(realistic, name, mt, model, ek)
    log.info(f'Predicted {name}')

# Per-protein metrics with bootstrap
def pp_metrics(sub, mnames, n_boot=1000):
    results = {}
    for m in mnames:
        cis, aucs, auprs, rmses = [], [], [], []
        for uid, grp in sub.groupby('uniprot_id'):
            t = grp.pki.values; p = grp[m].values
            if len(t) < 5: continue
            cis.append(ci_fn(t, p)); rmses.append(rmse(t, p))
            if (t>=7).sum()>0 and (t<7).sum()>0:
                lb = (t>=7).astype(int)
                aucs.append(roc_auc_score(lb, p)); auprs.append(average_precision_score(lb, p))
        cis = [c for c in cis if not np.isnan(c)]
        ci_b, auc_b, aupr_b = [], [], []
        for _ in range(n_boot):
            ci_b.append(np.mean(np.random.choice(cis, len(cis), replace=True)))
            if aucs:
                idx = np.random.choice(len(aucs), len(aucs), replace=True)
                auc_b.append(np.mean([aucs[i] for i in idx]))
                aupr_b.append(np.mean([auprs[i] for i in idx]))
        results[m] = {
            'ci': np.mean(cis), 'ci_lo': np.percentile(ci_b,2.5), 'ci_hi': np.percentile(ci_b,97.5),
            'auroc': np.mean(aucs) if aucs else np.nan,
            'auroc_lo': np.percentile(auc_b,2.5) if auc_b else np.nan,
            'auroc_hi': np.percentile(auc_b,97.5) if auc_b else np.nan,
            'auprc': np.mean(auprs) if auprs else np.nan,
            'auprc_lo': np.percentile(aupr_b,2.5) if aupr_b else np.nan,
            'auprc_hi': np.percentile(aupr_b,97.5) if aupr_b else np.nan,
            'rmse': np.mean(rmses), 'n': len(cis),
        }
    return results

mnames = list(models.keys())
np.random.seed(42)

log.info('\n' + '='*110)
log.info('ORACLE ANCHORS (n=%d)', len(oracle))
log.info('='*110)
orc = pp_metrics(oracle, mnames)
log.info(f"{'Model':<12} {'CI [95%]':<24} {'AUROC [95%]':<24} {'AUPRC [95%]':<24} {'RMSE':<8}")
log.info('-'*92)
for m in mnames:
    r = orc[m]
    log.info(f"{m:<12} {r['ci']:.3f} [{r['ci_lo']:.3f}-{r['ci_hi']:.3f}]   {r['auroc']:.3f} [{r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}]   {r['auprc']:.3f} [{r['auprc_lo']:.3f}-{r['auprc_hi']:.3f}]   {r['rmse']:.3f}")

log.info('\n' + '='*110)
log.info('REALISTIC ANCHORS via Tanimoto (n=%d, %.1f%% coverage)', len(realistic), len(realistic)/len(df)*100)
log.info('='*110)
rea = pp_metrics(realistic, mnames)
log.info(f"{'Model':<12} {'CI [95%]':<24} {'AUROC [95%]':<24} {'AUPRC [95%]':<24} {'RMSE':<8}")
log.info('-'*92)
for m in mnames:
    r = rea[m]
    log.info(f"{m:<12} {r['ci']:.3f} [{r['ci_lo']:.3f}-{r['ci_hi']:.3f}]   {r['auroc']:.3f} [{r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}]   {r['auprc']:.3f} [{r['auprc_lo']:.3f}-{r['auprc_hi']:.3f}]   {r['rmse']:.3f}")

log.info('\n' + '='*110)
log.info('ORACLE vs REALISTIC (V2 models)')
log.info('='*110)
for m in [n for n in mnames if 'V2' in n]:
    o, r = orc[m], rea[m]
    log.info(f"{m:<12} CI: {o['ci']:.3f} -> {r['ci']:.3f} (delta={o['ci']-r['ci']:+.3f})  AUROC: {o['auroc']:.3f} -> {r['auroc']:.3f}  AUPRC: {o['auprc']:.3f} -> {r['auprc']:.3f}")
