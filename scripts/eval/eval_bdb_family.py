"""Cross-dataset evaluation on BindingDB grouped by protein family.

Models trained on DTC, evaluated on BDB after canonical drug overlap exclusion.
Anchors retrieved from DTC training set via Tanimoto similarity.
Saves anchor metadata, per-protein metrics, and family-stratified violin plots.
"""
import torch, torch.nn as nn, numpy as np, pandas as pd, sys, json, logging, random, os, time
from sklearn.metrics import roc_auc_score, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger()
sys.path.insert(0, 'src')

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import requests

# ============================================================
# SECTION 1: Fetch protein families from UniProt
# ============================================================
FAMILY_CACHE = 'results/bdb_protein_families.json'

def fetch_uniprot_families(uniprot_ids):
    """Query UniProt REST API for protein family annotations."""
    if os.path.exists(FAMILY_CACHE):
        log.info(f'Loading cached families from {FAMILY_CACHE}')
        with open(FAMILY_CACHE) as f:
            return json.load(f)

    families = {}
    ids = list(uniprot_ids)
    batch_size = 100

    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        query = ' OR '.join(f'accession:{uid}' for uid in batch)
        url = 'https://rest.uniprot.org/uniprotkb/search'
        params = {
            'query': query,
            'fields': 'accession,protein_families',
            'format': 'json',
            'size': 500,
        }

        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    for entry in data.get('results', []):
                        uid = entry.get('primaryAccession', '')
                        fam_name = 'Unknown'
                        comments = entry.get('comments', [])
                        for comment in comments:
                            if comment.get('commentType') == 'SIMILARITY':
                                texts = comment.get('texts', [])
                                if texts:
                                    fam_name = texts[0].get('value', 'Unknown')
                                    fam_name = fam_name.replace('Belongs to the ', '').rstrip('.')
                                    break
                        families[uid] = fam_name
                    break
                elif resp.status_code == 429:
                    time.sleep(2 ** attempt)
                else:
                    log.warning(f'UniProt batch {start}: HTTP {resp.status_code}')
                    break
            except Exception as e:
                log.warning(f'UniProt batch {start} attempt {attempt}: {e}')
                time.sleep(2 ** attempt)

        if (start // batch_size + 1) % 10 == 0:
            log.info(f'  UniProt: {start + len(batch)}/{len(ids)} queried, {len(families)} found')

    for uid in uniprot_ids:
        if uid not in families:
            families[uid] = 'Unknown'

    os.makedirs('results', exist_ok=True)
    with open(FAMILY_CACHE, 'w') as f:
        json.dump(families, f, indent=2)
    log.info(f'Saved {len(families)} families to {FAMILY_CACHE}')
    return families

# ============================================================
# SECTION 2: Load data and embeddings
# ============================================================
bdb = pd.read_csv('data/processed/bindingdb_interactions.csv')
dtc = pd.read_csv('data/processed/dtc_training_interactions.csv')
log.info(f'BDB: {len(bdb)} interactions, {bdb.uniprot_id.nunique()} proteins, {bdb.ligand_smiles.nunique()} drugs')
log.info(f'DTC: {len(dtc)} interactions, {dtc.uniprot_id.nunique()} proteins')

esm35 = {}
for f in ['data/processed/esm2_35m_dtc_proteins_full.pt', 'data/processed/esm2_35m_benchmark.pt']:
    try: esm35.update(torch.load(f, map_location='cpu', weights_only=False))
    except: pass
esm650 = torch.load('data/processed/esm2_650m_all.pt', map_location='cpu', weights_only=False)
log.info(f'ESM35: {len(esm35)}, ESM650: {len(esm650)}')

seqs = json.load(open('data/processed/merged_sequences.json'))
log.info(f'Sequences: {len(seqs)}')

# DTC train split (same as all other eval scripts)
dtc_valid = dtc[dtc.uniprot_id.isin(esm35)]
dtc_prots = sorted(set(dtc_valid.uniprot_id))
random.seed(42); random.shuffle(dtc_prots)
nt = max(1, int(len(dtc_prots) * 0.1)); nv = max(1, int(len(dtc_prots) * 0.1))
dtc_train_prots = set(dtc_prots[nt + nv:])
dtc_train = dtc_valid[dtc_valid.uniprot_id.isin(dtc_train_prots)]
log.info(f'DTC train: {len(dtc_train)} interactions, {dtc_train.uniprot_id.nunique()} proteins')

# ============================================================
# SECTION 3: Canonical drug overlap exclusion
# ============================================================
def canonicalize(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    return Chem.MolToSmiles(mol, canonical=True)

dtc_smiles = set(dtc_train.ligand_smiles.unique())
dtc_canon = set()
for i, s in enumerate(dtc_smiles):
    c = canonicalize(s)
    if c: dtc_canon.add(c)
    if (i + 1) % 50000 == 0: log.info(f'  Canonicalizing DTC: {i+1}/{len(dtc_smiles)}')
log.info(f'DTC canonical drugs: {len(dtc_canon)}')

bdb_smiles = bdb.ligand_smiles.unique()
bdb_overlap = set()
bdb_canon_map = {}
for i, s in enumerate(bdb_smiles):
    c = canonicalize(s)
    if c:
        bdb_canon_map[s] = c
        if c in dtc_canon:
            bdb_overlap.add(s)
    if (i + 1) % 50000 == 0: log.info(f'  Canonicalizing BDB: {i+1}/{len(bdb_smiles)}')

log.info(f'BDB drugs overlapping with DTC (canonical): {len(bdb_overlap)}/{len(bdb_smiles)} ({len(bdb_overlap)/len(bdb_smiles)*100:.1f}%)')

bdb_clean = bdb[~bdb.ligand_smiles.isin(bdb_overlap)].copy()
log.info(f'BDB after overlap exclusion: {len(bdb_clean)} interactions, {bdb_clean.uniprot_id.nunique()} proteins, {bdb_clean.ligand_smiles.nunique()} drugs')

# ============================================================
# SECTION 4: Anchor retrieval from DTC training set
# ============================================================
def smi_to_fp(smi, chiral=True):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=chiral)

bdb_clean_canon = set()
for s in bdb_clean.ligand_smiles.unique():
    c = canonicalize(s)
    if c: bdb_clean_canon.add(c)

dtc_drug_to_anchor = {}
for smi, grp in dtc_train.groupby('ligand_smiles'):
    c = canonicalize(smi)
    if c and c in bdb_clean_canon:
        continue
    s = grp.sort_values('pki', ascending=False)
    uid = s.uniprot_id.values[0]
    if s.pki.values[0] >= 7.0 and uid in esm650:
        dtc_drug_to_anchor[smi] = (uid, s.pki.values[0])
log.info(f'DTC anchors (after excluding BDB-canonical drugs): {len(dtc_drug_to_anchor)}')

dtc_anchor_fps = {}
for s in dtc_drug_to_anchor:
    fp = smi_to_fp(s, chiral=True)
    if fp: dtc_anchor_fps[s] = fp
log.info(f'DTC anchor FPs: {len(dtc_anchor_fps)}')

bdb_drug_fps = {}
for s in bdb_clean.ligand_smiles.unique():
    fp = smi_to_fp(s, chiral=True)
    if fp: bdb_drug_fps[s] = fp
log.info(f'BDB drug FPs: {len(bdb_drug_fps)}')

dtc_fp_smiles = list(dtc_anchor_fps.keys())
dtc_fp_vals = list(dtc_anchor_fps.values())
log.info(f'Starting BulkTanimoto search: {len(bdb_drug_fps)} BDB drugs x {len(dtc_fp_vals)} DTC anchors')

nearest_dtc = {}
bdb_drug_list = list(bdb_drug_fps.items())
for i, (b_smi, b_fp) in enumerate(bdb_drug_list):
    sims = DataStructs.BulkTanimotoSimilarity(b_fp, dtc_fp_vals)
    best_idx = int(np.argmax(sims))
    nearest_dtc[b_smi] = (dtc_fp_smiles[best_idx], sims[best_idx])
    if (i + 1) % 5000 == 0:
        log.info(f'  Tanimoto search: {i+1}/{len(bdb_drug_fps)}')
log.info(f'Nearest DTC drug computed for {len(nearest_dtc)} BDB drugs')

sims = [v[1] for v in nearest_dtc.values()]
log.info(f'Tanimoto distribution: Min={min(sims):.3f} Median={np.median(sims):.3f} Mean={np.mean(sims):.3f} Max={max(sims):.3f}')

# Self-anchor check via sequence
seq_to_uids = defaultdict(set)
for uid, seq in seqs.items():
    seq_to_uids[seq].add(uid)

valid = set(esm35.keys()) & set(esm650.keys()) & set(seqs.keys())
bdb_eval = bdb_clean[bdb_clean.uniprot_id.isin(valid)].copy()
log.info(f'BDB proteins with all embeddings + sequences: {bdb_eval.uniprot_id.nunique()}')

rows, anc_uids, anc_pkis, tani_vals, anc_drugs = [], [], [], [], []
for i, row in bdb_eval.iterrows():
    uid, smi = row['uniprot_id'], row['ligand_smiles']
    if smi not in nearest_dtc: continue
    dtc_smi, tani = nearest_dtc[smi]
    if dtc_smi not in dtc_drug_to_anchor: continue
    au, ap = dtc_drug_to_anchor[dtc_smi]
    query_seq = seqs.get(uid, '')
    dtc_equiv = seq_to_uids.get(query_seq, set())
    if au in dtc_equiv: continue
    if au not in valid: continue
    rows.append(i)
    anc_uids.append(au)
    anc_pkis.append(ap)
    tani_vals.append(tani)
    anc_drugs.append(dtc_smi)

sub = bdb_eval.loc[rows].copy()
sub['anc_uid'] = anc_uids
sub['anc_pki'] = anc_pkis
sub['tanimoto'] = tani_vals
sub['anc_drug'] = anc_drugs
log.info(f'Eval set: {len(sub)} interactions, {sub.uniprot_id.nunique()} proteins, {sub.ligand_smiles.nunique()} drugs')
log.info(f'Coverage: {len(sub)/len(bdb_eval)*100:.1f}%')

# ============================================================
# SECTION 5: Fetch protein families
# ============================================================
eval_proteins = set(sub.uniprot_id.unique())
families = fetch_uniprot_families(eval_proteins)
sub['family'] = sub['uniprot_id'].map(families)

fam_counts = sub.groupby('family')['uniprot_id'].nunique()
top_families = set(fam_counts[fam_counts >= 10].index) - {'Unknown'}
sub['family_group'] = sub['family'].apply(lambda f: f if f in top_families else 'Other')
log.info(f'Top families (>=10 proteins): {len(top_families)}')
for fam in sorted(top_families):
    n = fam_counts[fam]
    log.info(f'  {fam}: {n} proteins')
log.info(f'  Other: {fam_counts[~fam_counts.index.isin(top_families)].sum()} proteins')

# Save anchor metadata CSV
os.makedirs('results', exist_ok=True)
anchor_meta = sub[['uniprot_id', 'ligand_smiles', 'pki', 'anc_uid', 'anc_drug', 'anc_pki', 'tanimoto', 'family', 'family_group']].copy()
anchor_meta.to_csv('results/bdb_anchor_metadata.csv', index=False)
log.info(f'Saved anchor metadata: results/bdb_anchor_metadata.csv')

# ============================================================
# SECTION 6: Load models and predict
# ============================================================
device = torch.device('cuda')

CHARISOSMISET = {"#":29,"%":30,")":31,"(":1,"+":32,"-":33,"/":34,".":2,"1":35,"0":3,"3":36,"2":4,"5":37,"4":5,"7":38,"6":6,"9":39,"8":7,"=":40,"A":41,"@":8,"C":42,"B":9,"E":43,"D":10,"G":44,"F":11,"I":45,"H":12,"K":46,"M":47,"L":13,"O":48,"N":14,"P":15,"S":49,"R":16,"[":50,"]":51,"_":19,"a":20,"c":21,"e":22,"g":23,"i":24,"l":25,"n":26,"o":27,"s":28,"r":17,"u":18}
CHARPROTSET = {"A":1,"C":2,"B":3,"E":4,"D":5,"G":6,"F":7,"I":8,"H":9,"K":10,"M":11,"L":12,"O":13,"N":14,"Q":15,"P":16,"S":17,"R":18,"U":19,"T":20,"W":21,"V":22,"Y":23,"X":24,"Z":25}
def enc_smi(s, ml=100): return [CHARISOSMISET.get(c,0) for c in s[:ml]] + [0]*(ml-len(s[:ml]))
def enc_prot(s, ml=1000): return [CHARPROTSET.get(c,0) for c in s[:ml]] + [0]*(ml-len(s[:ml]))

def ci_fn(y, f):
    if len(y) < 2: return np.nan
    ind = np.argsort(y); y = y[ind]; f = f[ind]
    n = np.sum(np.tile(y, (len(y), 1)).T < np.tile(y, (len(y), 1)))
    if n == 0: return np.nan
    z = np.sum((np.tile(y, (len(y), 1)).T < np.tile(y, (len(y), 1))) * (np.tile(f, (len(f), 1)).T < np.tile(f, (len(f), 1)))) + 0.5 * np.sum((np.tile(y, (len(y), 1)).T < np.tile(y, (len(y), 1))) * (np.tile(f, (len(f), 1)).T == np.tile(f, (len(f), 1))))
    return z / n

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
from anchor_transfer.model.esm_dta import EsmDTAModel

models = {}
def try_load(name, cls, path, kwargs, mtype, emb_key):
    if not Path(path).exists():
        log.info(f'Skip {name}: {path} not found'); return
    m = cls(**kwargs).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model_state_dict']); m.eval()
    models[name] = (mtype, m, emb_key)
    log.info(f'Loaded {name}')

try_load('V2-650M', AnchorTransferDTAv2, 'models/v2_650m_dtc/best_model.pt', {'esm2_dim': 1280}, 'v2', '650')
try_load('V2-35M', AnchorTransferDTAv2, 'models/v2_dtc/best_model.pt', {'esm2_dim': 480}, 'v2', '35')
try_load('DeepDTA', DeepDTAModel, 'models/deepdta_dtc/best_model.pt', {}, 'dta', '35')
try_load('ESM-DTA', EsmDTAModel, 'models/esm_dta_dtc/best_model.pt', {'esm2_dim': 480}, 'esm', '35')

emb_map = {'35': esm35, '650': esm650}

for name, (mtype, model, emb_key) in models.items():
    emb = emb_map[emb_key]
    preds = []
    for start in range(0, len(sub), 512):
        b = sub.iloc[start:start + 512]
        if mtype == 'v2':
            a_e = torch.stack([emb[a] for a in b.anc_uid]).to(device)
            q_e = torch.stack([emb[u] for u in b.uniprot_id]).to(device)
            dt = torch.tensor([enc_smi(s) for s in b.ligand_smiles], dtype=torch.long, device=device)
            with torch.no_grad(): preds.extend(model(a_e, q_e, dt)['pki_pred'].cpu().tolist())
        elif mtype == 'dta':
            dt = torch.tensor([enc_smi(s) for s in b.ligand_smiles], dtype=torch.long, device=device)
            pe = torch.tensor([enc_prot(seqs[u]) for u in b.uniprot_id], dtype=torch.long, device=device)
            with torch.no_grad(): preds.extend(model(dt, pe).cpu().tolist())
        elif mtype == 'esm':
            q_e = torch.stack([emb[u] for u in b.uniprot_id]).to(device)
            dt = torch.tensor([enc_smi(s) for s in b.ligand_smiles], dtype=torch.long, device=device)
            with torch.no_grad(): preds.extend(model(dt, q_e).cpu().tolist())
    sub[name] = preds
    log.info(f'Predicted {name}: {len(preds)} interactions')

# ============================================================
# SECTION 7: Per-protein metrics grouped by family
# ============================================================
model_names = list(models.keys())
MIN_INTERACTIONS = 5

per_protein_rows = []
for uid, grp in sub.groupby('uniprot_id'):
    t = grp.pki.values
    if len(t) < MIN_INTERACTIONS: continue
    fam = grp.family_group.values[0]
    for m in model_names:
        p = grp[m].values
        ci = ci_fn(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        per_protein_rows.append({
            'uniprot_id': uid,
            'family': fam,
            'model': m,
            'ci': ci,
            'rmse': rmse,
            'n_interactions': len(t),
            'mean_anc_pki': grp.anc_pki.mean(),
            'mean_tanimoto': grp.tanimoto.mean(),
        })

pp = pd.DataFrame(per_protein_rows)
pp.to_csv('results/bdb_per_protein_metrics.csv', index=False)
log.info(f'Saved per-protein metrics: {len(pp)} rows, {pp.uniprot_id.nunique()} proteins')

log.info(f'\n=== Per-family performance summary ===')
log.info(f'{"Family":<40} {"Model":<12} {"n_prot":<8} {"CI med":<8} {"CI mean":<8} {"RMSE med":<9} {"RMSE mean":<9}')
log.info('-' * 120)
for fam in sorted(pp.family.unique()):
    for m in model_names:
        mask = (pp.family == fam) & (pp.model == m)
        d = pp[mask]
        if len(d) == 0: continue
        ci_vals = d.ci.dropna()
        log.info(f'{fam:<40} {m:<12} {len(d):<8} {ci_vals.median():<8.3f} {ci_vals.mean():<8.3f} {d.rmse.median():<9.3f} {d.rmse.mean():<9.3f}')

log.info(f'\n=== Overall ===')
for m in model_names:
    d = pp[pp.model == m]
    ci_vals = d.ci.dropna()
    log.info(f'{m:<12} n={len(d)} proteins  CI: {ci_vals.mean():.3f} (med {ci_vals.median():.3f})  RMSE: {d.rmse.mean():.3f} (med {d.rmse.median():.3f})')

# ============================================================
# SECTION 8: Family-stratified violin plots (PNG)
# ============================================================
colors = {'V2-650M': '#2166ac', 'V2-35M': '#67a9cf', 'DeepDTA': '#ef8a62', 'ESM-DTA': '#d6604d'}

fam_order = pp.groupby('family')['uniprot_id'].nunique().sort_values(ascending=False).index.tolist()
if 'Other' in fam_order:
    fam_order.remove('Other')
    fam_order.append('Other')

n_families = len(fam_order)
n_models = len(model_names)
width = 0.8 / n_models

# --- Figure A: CI distribution by family ---
fig, ax = plt.subplots(figsize=(max(14, n_families * 2.5), 7))

for j, m in enumerate(model_names):
    positions = []
    data = []
    for i, fam in enumerate(fam_order):
        vals = pp[(pp.family == fam) & (pp.model == m)].ci.dropna().values
        if len(vals) >= 3:
            data.append(vals)
            positions.append(i + (j - n_models / 2 + 0.5) * width)
    if data:
        parts = ax.violinplot(data, positions=positions, widths=width * 0.9, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[m])
            pc.set_alpha(0.7)
        parts['cmedians'].set_color('black')
        parts['cmeans'].set_color('red')

ax.set_xticks(range(n_families))
ax.set_xticklabels([f[:25] for f in fam_order], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Per-Protein Concordance Index', fontsize=12)
ax.set_title('BindingDB Cross-Dataset: Per-Protein CI by Protein Family', fontsize=14, fontweight='bold')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[m], alpha=0.7) for m in model_names]
ax.legend(handles, model_names, loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
os.makedirs('paper/figures', exist_ok=True)
plt.savefig('paper/figures/fig_bdb_family_ci_distribution.png', dpi=300, bbox_inches='tight')
log.info('Saved: paper/figures/fig_bdb_family_ci_distribution.png')
plt.close()

# --- Figure B: RMSE distribution by family ---
fig, ax = plt.subplots(figsize=(max(14, n_families * 2.5), 7))

for j, m in enumerate(model_names):
    positions = []
    data = []
    for i, fam in enumerate(fam_order):
        vals = pp[(pp.family == fam) & (pp.model == m)].rmse.values
        if len(vals) >= 3:
            data.append(vals)
            positions.append(i + (j - n_models / 2 + 0.5) * width)
    if data:
        parts = ax.violinplot(data, positions=positions, widths=width * 0.9, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[m])
            pc.set_alpha(0.7)
        parts['cmedians'].set_color('black')
        parts['cmeans'].set_color('red')

ax.set_xticks(range(n_families))
ax.set_xticklabels([f[:25] for f in fam_order], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Per-Protein RMSE (pKi units)', fontsize=12)
ax.set_title('BindingDB Cross-Dataset: Per-Protein RMSE by Protein Family', fontsize=14, fontweight='bold')
handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[m], alpha=0.7) for m in model_names]
ax.legend(handles, model_names, loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('paper/figures/fig_bdb_family_rmse_distribution.png', dpi=300, bbox_inches='tight')
log.info('Saved: paper/figures/fig_bdb_family_rmse_distribution.png')
plt.close()

log.info('\n=== DONE ===')
