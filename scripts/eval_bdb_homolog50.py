#!/usr/bin/env python3
"""BDB-trained models on Davis + PDBbind with 50% homolog filtering + quartiles."""
import json, random
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

CHARISOSMISET = {"#":29,"%":30,")":31,"(":1,"+":32,"-":33,"/":34,".":2,"1":35,"0":3,"3":36,"2":4,"5":37,"4":5,"7":38,"6":6,"9":39,"8":7,"=":40,"A":41,"@":8,"C":42,"B":9,"E":43,"D":10,"G":44,"F":11,"I":45,"H":12,"K":46,"M":47,"L":13,"O":48,"N":14,"P":15,"S":49,"R":16,"[":50,"T":17,"]":51,"V":18,"Y":19,"c":20,"e":21,"l":22,"n":23,"o":24,"r":25,"s":26,"t":27,"u":28}
CHARPROTSET = {"A":1,"C":2,"B":3,"E":4,"D":5,"G":6,"F":7,"I":8,"H":9,"K":10,"M":11,"L":12,"O":13,"N":14,"Q":15,"P":16,"S":17,"R":18,"U":19,"T":20,"W":21,"V":22,"Y":23,"X":24,"Z":25}
def encode_smi(s,ml=100): return [CHARISOSMISET.get(c,0) for c in s[:ml]]+[0]*max(0,ml-len(s))
def encode_prot(s,ml=1000): return [CHARPROTSET.get(c,0) for c in s[:ml]]+[0]*max(0,ml-len(s))
def ci_fn(yt,yp):
    n=len(yt);yt,yp=np.array(yt),np.array(yp)
    if n<2:return 0.5
    if n*(n-1)//2>100000:i=np.random.randint(0,n,100000);j=np.random.randint(0,n,100000);m=i!=j;i,j=i[m],j[m]
    else:idx=np.triu_indices(n,k=1);i,j=idx
    dt=yt[i]-yt[j];dp=yp[i]-yp[j];t=dt==0
    return float(((dt*dp)>0).sum()/(~t).sum()) if (~t).sum()>0 else 0.5
def auroc_safe(t,p):
    b=t>=7;nb=t<=5;m=b|nb
    if m.sum()==0 or b[m].sum()==0 or nb[m].sum()==0:return float("nan")
    return float(roc_auc_score(b[m].astype(int),p[m]))

random.seed(42);np.random.seed(42);torch.manual_seed(42)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

esm35={}
for p in ["data/processed/esm2_35m_dtc_proteins.pt","data/processed/esm2_35m_davis.pt","data/processed/esm2_35m_pdbbind.pt"]:
    if Path(p).exists():esm35.update(torch.load(p,map_location="cpu",weights_only=False))
esm35={k:v for k,v in esm35.items() if not torch.isnan(v).any()}
seqs=json.load(open("data/processed/dtc_sequences.json"))
for _,r in pd.read_csv("data/raw/davis/davis_benchmark.csv").drop_duplicates("protein_name").iterrows():
    seqs[r["protein_name"]]=r["protein_sequence"]

from idr_gat.model.anchor_transfer_v2 import AnchorTransferDTAv2
v2=AnchorTransferDTAv2(esm2_dim=480).to(device)
v2.load_state_dict(torch.load("models/v2_bdb/best_model.pt",map_location=device,weights_only=False)["model_state_dict"]);v2.eval()

class DeepDTAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.smiles_embed=nn.Embedding(66,128,padding_idx=0);self.protein_embed=nn.Embedding(26,128,padding_idx=0)
        self.sc1=nn.Conv1d(128,32,8);self.sc2=nn.Conv1d(32,64,8);self.sc3=nn.Conv1d(64,96,8)
        self.pc1=nn.Conv1d(128,32,8);self.pc2=nn.Conv1d(32,64,8);self.pc3=nn.Conv1d(64,96,8)
        self.fc1=nn.Linear(192,1024);self.fc2=nn.Linear(1024,1024);self.fc3=nn.Linear(1024,512);self.out=nn.Linear(512,1);self.do=nn.Dropout(0.1)
    def forward(self,s,p):
        s=self.smiles_embed(s).permute(0,2,1);s=F.relu(self.sc1(s));s=F.relu(self.sc2(s));s=F.relu(self.sc3(s));s=s.max(2)[0]
        p=self.protein_embed(p).permute(0,2,1);p=F.relu(self.pc1(p));p=F.relu(self.pc2(p));p=F.relu(self.pc3(p));p=p.max(2)[0]
        x=torch.cat([s,p],1);x=self.do(F.relu(self.fc1(x)));x=self.do(F.relu(self.fc2(x)));x=self.do(F.relu(self.fc3(x)));return self.out(x).squeeze(-1)
dd=DeepDTAModel().to(device)
dd.load_state_dict(torch.load("models/deepdta_bdb/best_model.pt",map_location=device,weights_only=False)["model_state_dict"]);dd.eval()

from idr_gat.model.conplex import ConPlex
cpx=ConPlex(esm2_dim=480).to(device)
cpx.load_state_dict(torch.load("models/conplex_bdb/best_model.pt",map_location=device,weights_only=False)["model_state_dict"]);cpx.eval()

bdb=pd.read_csv("data/processed/bindingdb_interactions.csv")
bdb=bdb[bdb.uniprot_id.isin(esm35)]
bp=sorted(set(bdb.uniprot_id)&set(esm35.keys()));random.seed(42);random.shuffle(bp)
nt=max(1,int(len(bp)*0.1));nv=max(1,int(len(bp)*0.1))
bdb_train_drugs=set(bdb[bdb.uniprot_id.isin(set(bp[nt+nv:]))].ligand_smiles.unique())
print(f"BDB train drugs: {len(bdb_train_drugs)}")

def run(name, df, homolog_file):
    homologs=set(open(homolog_file).read().strip().split("\n"))
    before=len(df)
    df=df[~df.uniprot_id.isin(homologs)].copy()
    df=df[~df.ligand_smiles.isin(bdb_train_drugs)].copy()
    valid=set(esm35.keys())&set(seqs.keys())
    df=df[df.uniprot_id.isin(valid)].copy()
    print(f"\n{name}: {before} -> {len(df)} after homolog+drug filtering ({df.uniprot_id.nunique()} proteins)")

    strongest,second={},{}
    for smi,grp in df.groupby("ligand_smiles"):
        s=grp.sort_values("pki",ascending=False);p,pk=s.uniprot_id.values,s.pki.values
        strongest[smi]=(p[0],pk[0])
        if len(p)>1:second[smi]=(p[1],pk[1])
    rows,aus,aps=[],[],[]
    for i,row in df.iterrows():
        uid,smi=row["uniprot_id"],row["ligand_smiles"]
        if smi not in strongest:continue
        au,ap=strongest[smi]
        if au==uid:
            if smi not in second:continue
            au,ap=second[smi]
        if au not in valid:continue
        rows.append(i);aus.append(au);aps.append(ap)
    sub=df.loc[rows].copy();sub["anchor_uid"]=aus;sub["anchor_pki"]=aps
    if len(sub)<20:print(f"  too few ({len(sub)})");return

    preds={m:[] for m in ["v2","dd","cpx"]}
    for st in range(0,len(sub),512):
        b=sub.iloc[st:st+512]
        uids,smis,ancs=b.uniprot_id.values,b.ligand_smiles.values,b.anchor_uid.values
        a35=torch.stack([esm35[a] for a in ancs]).to(device)
        q35=torch.stack([esm35[u] for u in uids]).to(device)
        dt=torch.tensor([encode_smi(s) for s in smis],dtype=torch.long,device=device)
        pe=torch.tensor([encode_prot(seqs[u]) for u in uids],dtype=torch.long,device=device)
        with torch.no_grad():
            preds["v2"].extend(v2(a35,q35,dt)["pki_pred"].cpu().tolist())
            preds["dd"].extend(dd(dt,pe).cpu().tolist())
            preds["cpx"].extend(cpx(q35,dt)["score"].cpu().tolist())
    for m in preds:sub[m]=preds[m]

    try:sub["aq"]=pd.qcut(sub.anchor_pki,4,labels=["Q1","Q2","Q3","Q4"])
    except:sub["aq"]=pd.qcut(sub.anchor_pki,4,labels=False,duplicates="drop")

    print(f"  Anchored subset: {len(sub)} interactions, {sub.uniprot_id.nunique()} proteins")
    print(f"  {'Quartile':<10} {'Anchor pKi':<14} {'n':<7} {'V2-BDB':<10} {'DD-BDB':<10} {'CPX-BDB'}")
    print(f"  {'-'*65}")
    for q in sorted(sub.aq.unique()):
        s2=sub[sub.aq==q];lo,hi=s2.anchor_pki.min(),s2.anchor_pki.max();t=s2.pki.values
        a1=auroc_safe(t,s2.v2.values);a2=auroc_safe(t,s2.dd.values);a3=auroc_safe(t,s2.cpx.values)
        fmt=lambda x:f"{x:.3f}" if not np.isnan(x) else "N/A  "
        print(f"  {str(q):<10} [{lo:.1f}-{hi:.1f}]{'':>4} {len(s2):<7} {fmt(a1):<10} {fmt(a2):<10} {fmt(a3)}")
    t=sub.pki.values
    a1=auroc_safe(t,sub.v2.values);a2=auroc_safe(t,sub.dd.values);a3=auroc_safe(t,sub.cpx.values)
    c1=ci_fn(t,sub.v2.values);c2=ci_fn(t,sub.dd.values);c3=ci_fn(t,sub.cpx.values)
    print(f"  {'Overall':<10} {'':>14} {len(sub):<7} {a1:<10.3f} {a2:<10.3f} {a3:.3f}")
    print(f"  {'CI':<10} {'':>14} {'':>7} {c1:<10.3f} {c2:<10.3f} {c3:.3f}")

davis=pd.read_csv("data/raw/davis/davis_benchmark.csv").rename(columns={"protein_name":"uniprot_id","drug_smiles":"ligand_smiles","pKd":"pki"})
run("Davis",davis,"/tmp/davis_vs_bdb_homologs_50.txt")

pdb=pd.read_csv("data/raw/pdbbind_benchmark.csv")
run("PDBbind",pdb,"/tmp/pdbbind_vs_bdb_homologs_50.txt")
