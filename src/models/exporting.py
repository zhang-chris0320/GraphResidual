import gzip,hashlib,json,shutil
from pathlib import Path
import numpy as np,pandas as pd,torch
from src.models.model import ResidualV2,SEQ_TYPES
from src.graph.data import Inputs,COUNTS
def export_all(root,checkpoint,out,chunk=50000):
 root=Path(root);out=Path(out);(out/"embeddings").mkdir(parents=True,exist_ok=True);state=torch.load(checkpoint,map_location="cpu",weights_only=False);model=ResidualV2(state["relations"],state["config"]["beta_init"]);model.load_state_dict(state["model_state_dict"]);model.eval();inputs=Inputs(root);m=pd.read_csv(root/"work/graph_embedding_assets_v1/node_id_mapping.tsv",sep="\t");m.to_csv(out/"embeddings/node_id_mapping.tsv",sep="\t",index=False);modes=[];checks=[]
 with torch.no_grad():
  for typ,n in COUNTS.items():
   dst=out/"embeddings"/f"{typ}_residual_embeddings.npy";arr=np.lib.format.open_memmap(dst,mode="w+",dtype=np.float16,shape=(n,640))
   for i in range(0,n,chunk):
    j=min(i+chunk,n);idx=np.arange(i,j);g,e,v=inputs.batch(typ,idx,"cpu");z,_,_=model.encode(typ,g,e,v);arr[i:j]=z.numpy().astype(np.float16)
   arr.flush();a=np.load(dst,mmap_mode="r");checks.append((typ+"_shape",a.shape==(n,640),str(a.shape)));checks.append((typ+"_finite",np.isfinite(a).all(),str(a.dtype)))
   sub=m[m.node_type==typ].sort_values("node_index")
   if typ in SEQ_TYPES:
    v=np.asarray(inputs.valid[typ],bool);mode=np.where(v,"esm_graph_residual","graph_only_fallback")
   else:mode=np.repeat("graph_projected",n)
   modes.append(pd.DataFrame({"node_type":typ,"node_index":np.arange(n),"node_id":sub.node_id.to_numpy(),"fusion_mode":mode}))
 pd.concat(modes).to_csv(out/"embeddings/fusion_mode.tsv.gz",sep="\t",index=False,compression="gzip")
 rep=pd.DataFrame(checks,columns=["check","passed","detail"]);rep["status"]=np.where(rep.passed,"PASS","FAIL");rep[["check","status","detail"]].to_csv(out/"embedding_integrity_report.tsv",sep="\t",index=False)
 if not rep.passed.all():raise RuntimeError("export integrity failed")
 print("EXPORT_PASS",len(checks))
