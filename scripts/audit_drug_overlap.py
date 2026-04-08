"""Audit drug overlap between Davis and DTC at multiple levels, then
stratify V2-650M performance by Tanimoto retrieval similarity."""
import torch, torch.nn as nn, numpy as np, pandas as pd, sys, json, logging, random
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
from collections import defaultdict
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger()
sys.path.insert(0, 'src')

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, inchi

# ============================================================
# PART 1: Multi-level overlap audit
# ============================================================
davis = pd.read_csv('data/raw/davis/davis_benchmark.csv')
davis_smiles = set(davis.drug_smiles.unique())
log.info(f'Davis unique drugs (raw SMILES): {len(davis_smiles)}')

dtc = pd.read_csv('data/processed/dtc_training_interactions.csv')
esm35 = {}
for f in ['data/processed/esm2_35m_dtc_proteins_full.pt','data/processed/esm2_35m_davis.pt',
          'data/processed/esm2_35m_benchmark.pt']:
    try: esm35.update(torch.load(f, map_location='cpu', weights_only=False))
    except: pass
esm650 = torch.load('data/processed/esm2_650m_all.pt', map_location='cpu', weights_only=False)

dtc_valid = dtc[dtc.uniprot_id.isin(esm35)]
dtc_prots = sorted(set(dtc_valid.uniprot_id))
random.seed(42); random.shuffle(dtc_prots)
nt = max(1, int(len(dtc_prots)*0.1)); nv = max(1, int(len(dtc_prots)*0.1))
dtc_train_prots = set(dtc_prots[nt+nv:])
dtc_train = dtc_valid[dtc_valid.uniprot_id.isin(dtc_train_prots)]
dtc_smiles = set(dtc_train.ligand_smiles.unique())
log.info(f'DTC train unique drugs (raw SMILES): {len(dtc_smiles)}')

# 1. Raw SMILES overlap
raw_overlap = davis_smiles & dtc_smiles
log.info(f'\n=== OVERLAP AUDIT ===')
log.info(f'1. Raw SMILES overlap: {len(raw_overlap)}/{len(davis_smiles)} ({len(raw_overlap)/len(davis_smiles)*100:.1f}%)')

# 2. Canonical SMILES overlap
def canonicalize(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    return Chem.MolToSmiles(mol, canonical=True)

davis_canon = {canonicalize(s) for s in davis_smiles} - {None}
dtc_canon = set()
for i, s in enumerate(dtc_smiles):
    c = canonicalize(s)
    if c: dtc_canon.add(c)
    if (i+1) % 50000 == 0: log.info(f'  Canonicalizing DTC: {i+1}/{len(dtc_smiles)}')
canon_overlap = davis_canon & dtc_canon
log.info(f'2. Canonical SMILES overlap: {len(canon_overlap)}/{len(davis_canon)} ({len(canon_overlap)/len(davis_canon)*100:.1f}%)')
if canon_overlap:
    log.info(f'   Examples: {list(canon_overlap)[:3]}')

# 3. InChIKey overlap (full)
def get_inchikey(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    try: return inchi.InchiToInchiKey(inchi.MolToInchi(mol))
    except: return None

davis_inchikeys = {get_inchikey(s) for s in davis_smiles} - {None}
dtc_inchikeys = set()
for i, s in enumerate(dtc_smiles):
    ik = get_inchikey(s)
    if ik: dtc_inchikeys.add(ik)
    if (i+1) % 50000 == 0: log.info(f'  InChIKey DTC: {i+1}/{len(dtc_smiles)}')
inchikey_overlap = davis_inchikeys & dtc_inchikeys
log.info(f'3. Full InChIKey overlap: {len(inchikey_overlap)}/{len(davis_inchikeys)} ({len(inchikey_overlap)/len(davis_inchikeys)*100:.1f}%)')

# 4. First-block InChIKey (connectivity layer, ignores stereochem)
davis_ik1 = {ik.split('-')[0] for ik in davis_inchikeys if ik}
dtc_ik1 = {ik.split('-')[0] for ik in dtc_inchikeys if ik}
ik1_overlap = davis_ik1 & dtc_ik1
log.info(f'4. First-block InChIKey overlap: {len(ik1_overlap)}/{len(davis_ik1)} ({len(ik1_overlap)/len(davis_ik1)*100:.1f}%)')

# 5. Morgan FP (no chirality) Tanimoto = 1.0
def smi_to_fp(smi, chiral=False):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=chiral)

davis_fps = {s: smi_to_fp(s, chiral=False) for s in davis_smiles}
davis_fps = {k: v for k, v in davis_fps.items() if v is not None}

# Convert DTC FPs to bit strings for exact matching
dtc_fp_bits = set()
dtc_fp_map = {}
for i, s in enumerate(dtc_smiles):
    fp = smi_to_fp(s, chiral=False)
    if fp:
        bits = fp.ToBitString()
        dtc_fp_bits.add(bits)
        dtc_fp_map[s] = fp
    if (i+1) % 50000 == 0: log.info(f'  FP DTC: {i+1}/{len(dtc_smiles)}')

fp_exact = 0
for s, fp in davis_fps.items():
    if fp.ToBitString() in dtc_fp_bits:
        fp_exact += 1
log.info(f'5. Morgan FP (no chirality) exact match: {fp_exact}/{len(davis_fps)} ({fp_exact/len(davis_fps)*100:.1f}%)')

# 6. Morgan FP (with chirality) Tanimoto = 1.0
davis_fps_chiral = {s: smi_to_fp(s, chiral=True) for s in davis_smiles}
davis_fps_chiral = {k: v for k, v in davis_fps_chiral.items() if v is not None}
dtc_fp_bits_chiral = set()
for i, s in enumerate(dtc_smiles):
    fp = smi_to_fp(s, chiral=True)
    if fp: dtc_fp_bits_chiral.add(fp.ToBitString())
    if (i+1) % 50000 == 0: log.info(f'  Chiral FP DTC: {i+1}/{len(dtc_smiles)}')

fp_chiral_exact = 0
for s, fp in davis_fps_chiral.items():
    if fp.ToBitString() in dtc_fp_bits_chiral:
        fp_chiral_exact += 1
log.info(f'6. Morgan FP (with chirality) exact match: {fp_chiral_exact}/{len(davis_fps_chiral)} ({fp_chiral_exact/len(davis_fps_chiral)*100:.1f}%)')

# ============================================================
# PART 2: Tanimoto-stratified performance
# ============================================================
log.info('\n=== TANIMOTO-STRATIFIED PERFORMANCE ===')

# Build DTC anchor pool (exclude canonicalized Davis compounds)
dtc_drug_to_anchor = {}
for smi, grp in dtc_train.groupby('ligand_smiles'):
    s = grp.sort_values('pki', ascending=False)
    uid = s.uniprot_id.values[0]
    if s.pki.values[0] >= 7.0 and uid in esm650:
        # Exclude if this drug canonicalizes to a Davis drug
        c = canonicalize(smi)
        if c and c in davis_canon:
            continue  # SKIP: same molecule as a Davis drug
        dtc_drug_to_anchor[smi] = (uid, s.pki.values[0])
log.info(f'DTC anchors after excluding Davis-canonical drugs: {len(dtc_drug_to_anchor)}')

# Compute Tanimoto for each Davis drug (using chirality-aware FPs)
dtc_anchor_fps = {}
for s in dtc_drug_to_anchor:
    fp = smi_to_fp(s, chiral=True)
    if fp: dtc_anchor_fps[s] = fp
log.info(f'DTC anchor FPs (chiral): {len(dtc_anchor_fps)}')

davis_fps_eval = {s: smi_to_fp(s, chiral=True) for s in davis_smiles}
davis_fps_eval = {k: v for k, v in davis_fps_eval.items() if v is not None}

dtc_fp_list = list(dtc_anchor_fps.items())
nearest_dtc = {}
for d_smi, d_fp in davis_fps_eval.items():
    best_sim, best_smi = 0, None
    for c_smi, c_fp in dtc_fp_list:
        sim = DataStructs.TanimotoSimilarity(d_fp, c_fp)
        if sim > best_sim:
            best_sim = sim; best_smi = c_smi
    nearest_dtc[d_smi] = (best_smi, best_sim)
log.info(f'Nearest DTC drug computed for {len(nearest_dtc)} Davis drugs')

sims = [v[1] for v in nearest_dtc.values()]
log.info(f'Tanimoto distribution (after canonical exclusion, chiral FP):')
log.info(f'  Min={min(sims):.3f} Q1={np.percentile(sims,25):.3f} Median={np.median(sims):.3f} Q3={np.percentile(sims,75):.3f} Max={max(sims):.3f} Mean={np.mean(sims):.3f}')

# Tanimoto histogram
for lo, hi in [(0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.0), (1.0, 1.01)]:
    c = sum(1 for s in sims if lo <= s < hi)
    log.info(f'  [{lo:.1f}-{hi:.1f}): {c} drugs')

# Self-anchor check: map Davis to DTC via sequence (handle duplicates)
dtc_seqs = json.load(open('data/processed/merged_sequences.json'))
d_seqs = dict(zip(davis.protein_name, davis.protein_sequence))
seq_to_uids = defaultdict(set)
for uid, seq in dtc_seqs.items():
    seq_to_uids[seq].add(uid)
davis_to_dtc_uids = {}
for gname, seq in d_seqs.items():
    davis_to_dtc_uids[gname] = seq_to_uids.get(seq, set())

# Build evaluation set
CHARISOSMISET = {"#":29,"%":30,")":31,"(":1,"+":32,"-":33,"/":34,".":2,"1":35,"0":3,"3":36,"2":4,"5":37,"4":5,"7":38,"6":6,"9":39,"8":7,"=":40,"A":41,"@":8,"C":42,"B":9,"E":43,"D":10,"G":44,"F":11,"I":45,"H":12,"K":46,"M":47,"L":13,"O":48,"N":14,"P":15,"S":49,"R":16,"[":50,"]":51,"_":19,"a":20,"c":21,"e":22,"g":23,"i":24,"l":25,"n":26,"o":27,"s":28,"r":17,"u":18}
def enc_smi(s, ml=100): return [CHARISOSMISET.get(c,0) for c in s[:ml]] + [0]*(ml-len(s[:ml]))
def ci_fn(y, f):
    if len(y)<2: return np.nan
    ind=np.argsort(y); y=y[ind]; f=f[ind]
    n=np.sum(np.tile(y,(len(y),1)).T<np.tile(y,(len(y),1)))
    if n==0: return np.nan
    z=np.sum((np.tile(y,(len(y),1)).T<np.tile(y,(len(y),1)))*(np.tile(f,(len(f),1)).T<np.tile(f,(len(f),1))))+0.5*np.sum((np.tile(y,(len(y),1)).T<np.tile(y,(len(y),1)))*(np.tile(f,(len(f),1)).T==np.tile(f,(len(f),1))))
    return z/n

device = torch.device('cuda')
from anchor_transfer.model.anchor_transfer_v2 import AnchorTransferDTAv2
model = AnchorTransferDTAv2(esm2_dim=1280).to(device)
model.load_state_dict(torch.load('models/v2_650m_dtc/best_model.pt', map_location=device, weights_only=False)['model_state_dict']); model.eval()

valid = set(esm35.keys()) & set(esm650.keys())
df = davis.rename(columns={'protein_name':'uniprot_id','drug_smiles':'ligand_smiles','pKd':'pki'})
df = df[df.uniprot_id.isin(valid)].copy()

rows, anc_u, anc_p, tani_vals = [], [], [], []
for i, row in df.iterrows():
    uid, smi = row['uniprot_id'], row['ligand_smiles']
    if smi not in nearest_dtc: continue
    dtc_smi, tani = nearest_dtc[smi]
    if dtc_smi not in dtc_drug_to_anchor: continue
    au, ap = dtc_drug_to_anchor[dtc_smi]
    # Self-check: skip if anchor is any sequence-equivalent of query
    dtc_uids = davis_to_dtc_uids.get(uid, set())
    if au in dtc_uids: continue
    if au not in valid: continue
    rows.append(i); anc_u.append(au); anc_p.append(ap); tani_vals.append(tani)

sub = df.loc[rows].copy()
sub['anc_uid'] = anc_u; sub['anc_pki'] = anc_p; sub['tanimoto'] = tani_vals
log.info(f'Eval set: {len(sub)} interactions ({len(sub)/len(df)*100:.1f}% coverage)')

# Predict
preds = []
for start in range(0, len(sub), 512):
    b = sub.iloc[start:start+512]
    a_e = torch.stack([esm650[a] for a in b.anc_uid]).to(device)
    q_e = torch.stack([esm650[u] for u in b.uniprot_id]).to(device)
    dt = torch.tensor([enc_smi(s) for s in b.ligand_smiles], dtype=torch.long, device=device)
    with torch.no_grad(): preds.extend(model(a_e, q_e, dt)['pki_pred'].cpu().tolist())
sub['pred'] = preds

# Stratify by Tanimoto bins
log.info(f'\nV2-650M performance by Tanimoto retrieval similarity bin:')
log.info(f'{"Bin":<14} {"n":<7} {"drugs":<7} {"CI":<8} {"AUROC":<8} {"AUPRC":<8} {"RMSE":<8}')
log.info('-' * 60)
for lo, hi, label in [(0, 0.6, '[0-0.6)'), (0.6, 0.8, '[0.6-0.8)'), (0.8, 0.95, '[0.8-0.95)'),
                       (0.95, 0.999, '[0.95-1.0)'), (0.999, 1.01, '[1.0]')]:
    s = sub[(sub.tanimoto >= lo) & (sub.tanimoto < hi)]
    if len(s) < 20: continue
    t = s.pki.values; p = s.pred.values
    # Per-protein
    cis, aucs, auprs = [], [], []
    for uid, grp in s.groupby('uniprot_id'):
        tv = grp.pki.values; pv = grp.pred.values
        if len(tv) < 3: continue
        c = ci_fn(tv, pv)
        if not np.isnan(c): cis.append(c)
        if (tv>=7).sum()>0 and (tv<7).sum()>0:
            aucs.append(roc_auc_score((tv>=7).astype(int), pv))
            auprs.append(average_precision_score((tv>=7).astype(int), pv))
    ci_m = np.mean(cis) if cis else np.nan
    auc_m = np.mean(aucs) if aucs else np.nan
    aupr_m = np.mean(auprs) if auprs else np.nan
    rmse_m = np.sqrt(mean_squared_error(t, p))
    n_drugs = s.ligand_smiles.nunique()
    log.info(f'{label:<14} {len(s):<7} {n_drugs:<7} {ci_m:<8.3f} {auc_m:<8.3f} {aupr_m:<8.3f} {rmse_m:<8.3f}')

# Overall
t = sub.pki.values; p = sub.pred.values
cis, aucs, auprs = [], [], []
for uid, grp in sub.groupby('uniprot_id'):
    tv = grp.pki.values; pv = grp.pred.values
    if len(tv) < 3: continue
    c = ci_fn(tv, pv);
    if not np.isnan(c): cis.append(c)
    if (tv>=7).sum()>0 and (tv<7).sum()>0:
        aucs.append(roc_auc_score((tv>=7).astype(int), pv))
        auprs.append(average_precision_score((tv>=7).astype(int), pv))
log.info(f'{"Overall":<14} {len(sub):<7} {sub.ligand_smiles.nunique():<7} {np.mean(cis):<8.3f} {np.mean(aucs):<8.3f} {np.mean(auprs):<8.3f} {np.sqrt(mean_squared_error(t,p)):<8.3f}')
