import argparse,hashlib,sys
from pathlib import Path
import numpy as np,pandas as pd
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
EXPECTED={"disease":(19019,64),"mutation":(1055821,64),"protein":(14231,64),"ptm":(166288,64),"pathway":(2791,64)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--project_dir",type=Path,required=True);ap.add_argument("--out_dir",type=Path,required=True);a=ap.parse_args();r=a.project_dir.resolve();o=(a.out_dir if a.out_dir.is_absolute() else r/a.out_dir);o.mkdir(parents=True,exist_ok=True);checks=[]
 gm=pd.read_csv(r/"work/graph_embedding_assets_v1/node_id_mapping.tsv",sep="\t")
 for typ,shape in EXPECTED.items():
  g=np.load(r/f"work/graph_embedding_assets_v1/embeddings/{typ}_embeddings.npy",mmap_mode="r");checks.append((typ+"_graph_shape",g.shape==shape,str(g.shape)));checks.append((typ+"_graph_finite",np.isfinite(g).all(),str(g.dtype)))
 for typ,n in [("mutation",1055821),("protein",14231),("ptm",166288)]:
  e=np.load(r/f"work/esm2_only_v1/core_embeddings/{typ}_esm2_embeddings.npy",mmap_mode="r");v=np.load(r/f"work/esm2_only_v1/core_embeddings/{typ}_sequence_valid_mask.npy",mmap_mode="r");em=pd.read_csv(r/f"work/esm2_only_v1/core_embeddings/{typ}_mapping.tsv.gz",sep="\t").sort_values("embedding_index");gg=gm[gm.node_type==typ].sort_values("node_index")
  checks += [(typ+"_esm_shape",e.shape==(n,640),str(e.shape)),(typ+"_mask_shape",v.shape==(n,),str(v.shape)),(typ+"_node_id_alignment",np.array_equal(em.node_id.to_numpy(),gg.node_id.to_numpy()),"explicit mapping compare"),(typ+"_esm_finite",np.isfinite(e).all(),str(e.dtype))]
 ed=list((r/"work/graph_core/edges").glob("*.tsv.gz"));checks.append(("six_forward_relations",len(ed)==6,str([p.name for p in ed])));checks.append(("no_mutation_ptm_edge",not any("mutation__" in p.name and "__ptm" in p.name for p in ed),str([p.name for p in ed])))
 for p in ed:
  src,rel,dst=p.name.replace(".tsv.gz","").split("__");d=pd.read_csv(p,sep="\t");checks.append(("edge_bounds:"+rel,int(d.src_index.max())<EXPECTED[src][0] and int(d.dst_index.max())<EXPECTED[dst][0],str(len(d))))
 b=pd.read_csv(r/"data_benchmark/clinvar_pathogenicity_esm_resolved/samples.tsv",sep="\t");mm=pd.read_csv(r/"work/esm2_only_v1/core_embeddings/mutation_mapping.tsv.gz",sep="\t",usecols=["node_id","embedding_index","sequence_valid"]);q=b.merge(mm,left_on="mutation_id",right_on="node_id");checks.append(("clinvar_coverage_valid",len(q)==len(b) and q.sequence_valid.all(),f"{len(q)}/{len(b)}"))
 rep=pd.DataFrame(checks,columns=["check","passed","detail"]);rep["status"]=np.where(rep.passed,"PASS","FAIL");rep[["check","status","detail"]].to_csv(o/"input_integrity_report.tsv",sep="\t",index=False);print("INPUT_STATUS","PASS" if rep.passed.all() else "FAIL","checks",len(rep))
 if not rep.passed.all():raise SystemExit(2)
if __name__=="__main__":main()
