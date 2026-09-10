from pathlib import Path
import numpy as np,pandas as pd,torch
SEQ_TYPES=("mutation","protein","ptm")
COUNTS={"disease":19019,"mutation":1055821,"protein":14231,"ptm":166288,"pathway":2791}
class Inputs:
 def __init__(self,root):
  self.root=Path(root);g=self.root/"work/graph_embedding_assets_v1/embeddings";e=self.root/"work/esm2_only_v1/core_embeddings"
  self.graph={k:np.load(g/f"{k}_embeddings.npy",mmap_mode="r") for k in COUNTS};self.esm={k:np.load(e/f"{k}_esm2_embeddings.npy",mmap_mode="r") for k in SEQ_TYPES};self.valid={k:np.load(e/f"{k}_sequence_valid_mask.npy",mmap_mode="r") for k in SEQ_TYPES}
 def batch(self,typ,idx,device):
  g=torch.as_tensor(np.asarray(self.graph[typ][idx],dtype=np.float32),device=device)
  if typ in SEQ_TYPES:
   e=torch.as_tensor(np.asarray(self.esm[typ][idx],dtype=np.float32),device=device);v=torch.as_tensor(np.asarray(self.valid[typ][idx],dtype=bool),device=device);return g,e,v
  return g,None,None
def load_edges(root,sanity=False):
 root=Path(root);ed=root/"work/graph_core/edges";base=[]
 for p in sorted(ed.glob("*.tsv.gz")):
  src,rel,dst=p.name.replace(".tsv.gz","").split("__");x=pd.read_csv(p,sep="\t")[["src_index","dst_index"]].to_numpy(np.int64);base.append([src,rel,dst,x])
 if sanity:
  mp=next(x for s,r,d,x in base if s=="mutation" and d=="protein");mset=set(mp[mp[:,1]<500,0].tolist())
  pp=next(x for s,r,d,x in base if s=="protein" and d=="ptm");pset=set(range(500));ptmset=set(pp[pp[:,0]<500,1].tolist())
  small=[]
  for s,r,d,x in base:
   keep=np.ones(len(x),bool)
   if s=="mutation" and d=="protein":keep=np.isin(x[:,0],list(mset))&np.isin(x[:,1],list(pset))
   elif s=="protein":keep=np.isin(x[:,0],list(pset))
   elif d=="mutation":keep=np.isin(x[:,1],list(mset))
   elif d=="protein":keep=np.isin(x[:,1],list(pset))
   elif d=="ptm":keep=np.isin(x[:,1],list(ptmset))
   x=x[keep][:5000];small.append([s,r,d,x])
  base=small
 out=[]
 for s,r,d,x in base:
  if not len(x):raise RuntimeError("empty sanity relation "+r)
  out.append((s,r,d,x));out.append((d,"reverse_"+r,s,x[:,[1,0]]))
 return out
