import json, random, torch, numpy as np, pandas as pd, sys
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "src")

esm650 = torch.load("data/processed/esm2_650m_all.pt", map_location="cpu", weights_only=False)
esm35 = {}
for f in ["data/processed/esm2_35m_dtc_proteins_full.pt","data/processed/esm2_35m_davis.pt","data/processed/esm2_35m_pdbbind.pt","data/processed/esm2_35m_benchmark.pt"]:
    try: esm35.update(torch.load(f, map_location="cpu", weights_only=False))
    except: pass

valid = set(esm650.keys()) & set(esm35.keys())
device = torch.device("cuda")

CHARISOSMISET = {"#":29,"%":30,")":31,"(":1,"+":32,"-":33,"/":34,".":2,"1":35,"0":3,"3":36,"2":4,"5":37,"4":5,"7":38,"6":6,"9":39,"8":7,"=":40,"A":41,"@":8,"C":42,"B":9,"E":43,"D":10,"G":44,"F":11,"I":45,"H":12,"K":46,"M":47,"L":13,"O":48,"N":14,"P":15,"S":49,"R":16,"[":50,"]":51,"_":19,"a":20,"c":21,"e":22,"g":23,"i":24,"l":25,"n":26,"o":27,"s":28,"r":17,"u":18}
def enc_smi(s, ml=100): return [CHARISOSMISET.get(c,0) for c in s[:ml]] + [0]*(ml-len(s[:ml]))
def ci_fn(y, f):
    ind=np.argsort(y); y=y[ind]; f=f[ind]
    z=np.sum((np.tile(y,(len(y),1)).T<np.tile(y,(len(y),1)))*(np.tile(f,(len(f),1)).T<np.tile(f,(len(f),1))))+0.5*np.sum((np.tile(y,(len(y),1)).T<np.tile(y,(len(y),1)))*(np.tile(f,(len(f),1)).T==np.tile(f,(len(f),1))))
    n=np.sum(np.tile(y,(len(y),1)).T<np.tile(y,(len(y),1))); return z/n if n>0 else 0
def auroc_safe(y,p):
    t=(y>=np.median(y)).astype(int)
    if len(set(t))<2: return float("nan")
    return roc_auc_score(t,p)

from idr_gat.model.anchor_transfer_v2 import AnchorTransferDTAv2
v2_650 = AnchorTransferDTAv2(esm2_dim=1280).to(device)
v2_650.load_state_dict(torch.load("models/v2_650m_dtc/best_model.pt", map_location=device, weights_only=False)["model_state_dict"]); v2_650.eval()
v2_35 = AnchorTransferDTAv2(esm2_dim=480).to(device)
v2_35.load_state_dict(torch.load("models/v2_dtc/best_model.pt", map_location=device, weights_only=False)["model_state_dict"]); v2_35.eval()
print("Models loaded")

def build_anchors(df):
    df = df[df.uniprot_id.isin(valid)].copy()
    strongest, second = {}, {}
    for smi, grp in df.groupby("ligand_smiles"):
        s = grp.sort_values("pki", ascending=False)
        p, pk = s.uniprot_id.values, s.pki.values
        strongest[smi] = (p[0], pk[0])
        if len(p) > 1: second[smi] = (p[1], pk[1])
    rows, anc_u, anc_p = [], [], []
    for i, row in df.iterrows():
        uid, smi = row["uniprot_id"], row["ligand_smiles"]
        if smi not in strongest: continue
        au, ap = strongest[smi]
        if au == uid:
            if smi not in second: continue
            au, ap = second[smi]
        if au not in valid: continue
        rows.append(i); anc_u.append(au); anc_p.append(ap)
    sub = df.loc[rows].copy(); sub["anc_uid"] = anc_u; sub["anc_pki"] = anc_p
    return sub

def predict(sub, model, emb):
    preds = []
    for start in range(0, len(sub), 512):
        b = sub.iloc[start:start+512]
        a_e = torch.stack([emb[a] for a in b.anc_uid]).to(device)
        q_e = torch.stack([emb[u] for u in b.uniprot_id]).to(device)
        dt = torch.tensor([enc_smi(s) for s in b.ligand_smiles], dtype=torch.long, device=device)
        with torch.no_grad(): preds.extend(model(a_e, q_e, dt)["pki_pred"].cpu().tolist())
    return preds

def print_bins(name, sub, bins):
    print(f"\n=== {name} ===")
    print(f"{'Bin':<12} {'n':<7} {'anc':<6} {'650-AUROC':<12} {'650-CI':<10} {'35-AUROC':<12} {'35-CI':<10}")
    print("-" * 72)
    for lo, hi in bins:
        s = sub[(sub.anc_pki >= lo) & (sub.anc_pki < hi)]
        if len(s) < 10: continue
        t = s.pki.values
        a6 = auroc_safe(t, s.pred_650.values); c6 = ci_fn(t, s.pred_650.values)
        a3 = auroc_safe(t, s.pred_35.values); c3 = ci_fn(t, s.pred_35.values)
        a6s = "nan" if np.isnan(a6) else f"{a6:.3f}"
        a3s = "nan" if np.isnan(a3) else f"{a3:.3f}"
        label = f"[{lo}-{hi})"
        print(f"{label:<12} {len(s):<7} {s.anc_uid.nunique():<6} {a6s:<12} {c6:<10.3f} {a3s:<12} {c3:<10.3f}")
    t = sub.pki.values
    a6 = auroc_safe(t, sub.pred_650.values); c6 = ci_fn(t, sub.pred_650.values)
    a3 = auroc_safe(t, sub.pred_35.values); c3 = ci_fn(t, sub.pred_35.values)
    a6s = "nan" if np.isnan(a6) else f"{a6:.3f}"
    a3s = "nan" if np.isnan(a3) else f"{a3:.3f}"
    print(f"{'Overall':<12} {len(sub):<7} {sub.anc_uid.nunique():<6} {a6s:<12} {c6:<10.3f} {a3s:<12} {c3:<10.3f}")

# DAVIS
davis = pd.read_csv("data/raw/davis/davis_benchmark.csv").rename(columns={"protein_name":"uniprot_id","drug_smiles":"ligand_smiles","pKd":"pki"})
sub_d = build_anchors(davis)
sub_d = sub_d[sub_d.anc_pki >= 7.0].copy()
sub_d["pred_650"] = predict(sub_d, v2_650, esm650)
sub_d["pred_35"] = predict(sub_d, v2_35, esm35)
print_bins("DAVIS (anchors >= 7)", sub_d, [(7,8),(8,9),(9,10),(10,11)])

# PDBBIND full
pdb = pd.read_csv("data/raw/pdbbind_benchmark.csv")
sub_p = build_anchors(pdb)
sub_p["pred_650"] = predict(sub_p, v2_650, esm650)
sub_p["pred_35"] = predict(sub_p, v2_35, esm35)
print_bins("PDBBIND (all)", sub_p, [(0,4),(4,6),(6,8),(8,10),(10,12),(12,16)])

# PDBBIND Q4 sub-bins
q4 = sub_p[sub_p.anc_pki >= 8.0].copy()
print_bins("PDBBIND Q4 (anchor >= 8)", q4, [(8,10),(10,12),(12,16)])
