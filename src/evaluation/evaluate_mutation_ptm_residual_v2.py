#!/usr/bin/env python3
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

def score_pairs(mut,ptm,mi,pi,device,chunk=50000):
    out=np.empty(len(mi),np.float32)
    for s in range(0,len(mi),chunk):
        e=min(len(mi),s+chunk)
        a=torch.as_tensor(np.asarray(mut[mi[s:e]],dtype=np.float32),device=device)
        b=torch.as_tensor(np.asarray(ptm[pi[s:e]],dtype=np.float32),device=device)
        out[s:e]=torch.nn.functional.cosine_similarity(a,b,dim=1).detach().cpu().numpy()
    return out

def query_metrics(frame,positive_col):
    rows=[]
    for mid,g in frame.groupby("mutation_id",sort=False):
        order=np.argsort(-g.score.to_numpy());rel=g[positive_col].to_numpy(bool)[order]
        n=int(rel.sum())
        if n==0:continue
        pos=np.flatnonzero(rel)+1
        row={"mutation_id":mid,"protein_id":g.protein_id.iloc[0],"candidates":len(g),"positives":n,"MRR":1.0/pos[0],"MAP":float(np.mean([rel[:r].mean() for r in pos]))}
        for k in [1,5,10]:row[f"Hits@{k}"]=float((pos<=k).any())
        for k in [10,50,100]:row[f"Recall@{k}"]=float((pos<=k).sum()/n)
        for k in [10,50]:
            gains=rel[:k]/np.log2(np.arange(2,min(k,len(rel))+2));dcg=gains.sum()
            ideal=(np.ones(min(n,k))/np.log2(np.arange(2,min(n,k)+2))).sum();row[f"nDCG@{k}"]=float(dcg/ideal if ideal else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--project_dir",type=Path,required=True);ap.add_argument("--out_dir",type=Path,required=True);ap.add_argument("--device",default="cuda");a=ap.parse_args()
    root=a.project_dir.resolve();out=a.out_dir if a.out_dir.is_absolute() else root/a.out_dir;out.mkdir(parents=True,exist_ok=True)
    b=root/"data_benchmark/mutation_ptm"
    dcols=["mutation_id","ptm_id","protein_id","evidence_level","target_type","label_source","split"]
    direct=pd.read_csv(b/"direct_positives.tsv",sep="\t",usecols=lambda c:c in dcols)
    ucols=["mutation_id","ptm_id","protein_id","split"]
    unl=pd.read_csv(b/"unlabeled_candidates.tsv",sep="\t",usecols=ucols)
    direct=direct[direct.split.eq("test")].copy();unl=unl[unl.split.eq("test")].copy()
    direct["pool_status"]="gold_positive";unl["pool_status"]="unlabeled"
    direct["is_positive"]=True;unl["is_positive"]=False
    direct["is_experimental"]=direct.get("evidence_level",pd.Series("",index=direct.index)).astype(str).str.contains("experimental",case=False)|direct.get("label_source",pd.Series("",index=direct.index)).astype(str).str.contains("experimental",case=False)
    positive_mut=set(direct.mutation_id);unl=unl[unl.mutation_id.isin(positive_mut)]
    cand=pd.concat([direct,unl],ignore_index=True,sort=False)
    cand["priority"]=cand.is_positive.astype(int);cand=cand.sort_values("priority",ascending=False).drop_duplicates(["mutation_id","ptm_id"]).drop(columns="priority")
    counts=cand.groupby("mutation_id").pool_status.nunique();eligible=set(counts[counts.eq(2)].index);cand=cand[cand.mutation_id.isin(eligible)].reset_index(drop=True)
    mapping=pd.read_csv(root/"work/graph_embedding_assets_v1/node_id_mapping.tsv",sep="\t")
    tc="node_type" if "node_type" in mapping else "type";ic="node_index" if "node_index" in mapping else "index"
    mm=mapping[mapping[tc].astype(str).str.lower().eq("mutation")][["node_id",ic]].rename(columns={ic:"mutation_index"})
    pm=mapping[mapping[tc].astype(str).str.lower().eq("ptm")][["node_id",ic]].rename(columns={ic:"ptm_index"})
    cand=cand.merge(mm,left_on="mutation_id",right_on="node_id",how="left").drop(columns="node_id").merge(pm,left_on="ptm_id",right_on="node_id",how="left").drop(columns="node_id")
    if cand[["mutation_index","ptm_index"]].isna().any().any():raise RuntimeError("candidate mapping incomplete")
    mi=cand.mutation_index.to_numpy(int);pi=cand.ptm_index.to_numpy(int)
    reps={
      "ESM2-only":(root/"work/esm2_only_v1/core_embeddings/mutation_esm2_embeddings.npy",root/"work/esm2_only_v1/core_embeddings/ptm_esm2_embeddings.npy"),
      "Graph-only":(root/"work/graph_embedding_assets_v1/embeddings/mutation_embeddings.npy",root/"work/graph_embedding_assets_v1/embeddings/ptm_embeddings.npy"),
      "Original gated fusion":(root/"work/graph_esm2_fusion_v1/embeddings/mutation_fused_embeddings.npy",root/"work/graph_esm2_fusion_v1/embeddings/ptm_fused_embeddings.npy"),
      "Residual V2":(root/"work/graph_esm2_residual_pretrain_v2/embeddings/mutation_residual_embeddings.npy",root/"work/graph_esm2_residual_pretrain_v2/embeddings/ptm_residual_embeddings.npy")}
    device=torch.device(a.device if torch.cuda.is_available() else "cpu");allq=[];metrics=[];experimental=[]
    for name,(mp,pp) in reps.items():
        if not mp.exists() or not pp.exists():
            metrics.append({"representation":name,"status":"MISSING"});continue
        mut=np.load(mp,mmap_mode="r");ptm=np.load(pp,mmap_mode="r")
        cand["score"]=score_pairs(mut,ptm,mi,pi,device)
        q=query_metrics(cand,"is_positive");q.insert(0,"representation",name);allq.append(q)
        agg={"representation":name,"status":"PASS","queries":len(q),"candidate_pairs":len(cand)}
        for c in ["MRR","Hits@1","Hits@5","Hits@10","Recall@10","Recall@50","Recall@100","MAP","nDCG@10","nDCG@50"]:agg[c]=q[c].mean()
        metrics.append(agg)
        qe=query_metrics(cand,"is_experimental");row={"representation":name,"queries":len(qe)}
        for c in ["MRR","Hits@1","Hits@5","Hits@10","Recall@10","Recall@50","Recall@100","MAP","nDCG@10","nDCG@50"]:row[c]=qe[c].mean() if len(qe) else np.nan
        experimental.append(row)
    met=pd.DataFrame(metrics);exp=pd.DataFrame(experimental);met.to_csv(out/"ranking_metrics.tsv",sep="\t",index=False);exp.to_csv(out/"direct_experimental_metrics.tsv",sep="\t",index=False)
    pd.concat(allq,ignore_index=True).to_csv(out/"per_query_metrics.tsv.gz",sep="\t",index=False,compression="gzip")
    pd.DataFrame([{"test_direct_pairs":len(direct),"test_unlabeled_pool_pairs":len(unl),"eligible_queries":len(eligible),"evaluated_candidate_pairs":len(cand),"unlabeled_semantics":"retrieval pool only; never fitted or treated as classification negatives","split_grouping":"benchmark protein-group split; test only"}]).to_csv(out/"candidate_pool_summary.tsv",sep="\t",index=False)
    plt.style.use("seaborn-v0_8-whitegrid");show=["MRR","Hits@10","Recall@50","MAP","nDCG@10"];x=np.arange(len(show));w=.18;plt.figure(figsize=(10,5))
    ok=met[met.status.eq("PASS")]
    for i,(_,r) in enumerate(ok.iterrows()):plt.bar(x+i*w,[r[c] for c in show],w,label=r.representation)
    plt.xticks(x+w*1.5,show);plt.ylabel("Ranking metric");plt.legend(fontsize=8);plt.tight_layout();plt.savefig(out/"mutation_ptm_ranking_comparison.png",dpi=180);plt.close()
    plt.figure(figsize=(8,5));x=np.arange(len(exp));plt.bar(x,exp.MRR,color=["#4C78A8","#54A24B","#E45756","#F58518"]);plt.xticks(x,exp.representation,rotation=20,ha="right");plt.ylabel("Direct-experimental MRR");plt.tight_layout();plt.savefig(out/"direct_experimental_ranking.png",dpi=180);plt.close()
    v2=met[met.representation.eq("Residual V2")].iloc[0];base=met[met.representation.eq("ESM2-only")].iloc[0]
    checks=[("unlabeled_not_negative",True,"no classifier/loss; candidate pool only"),("protein_group_test_only",True,"benchmark split=test"),("mapping_complete",not cand[["mutation_index","ptm_index"]].isna().any().any(),str(len(cand))),("all_four_representations",int(met.status.eq("PASS").sum())==4,str(met.representation.tolist()))]
    pd.DataFrame(checks,columns=["check","pass","detail"]).to_csv(out/"integrity_report.tsv",sep="\t",index=False)
    summary={"status":"PASS" if all(x[1] for x in checks) else "FAIL","residual_v2_better_than_esm2":{"MRR":bool(v2.MRR>base.MRR),"MAP":bool(v2.MAP>base.MAP),"nDCG@10":bool(v2["nDCG@10"]>base["nDCG@10"])},"device":str(device)}
    (out/"acceptance_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
