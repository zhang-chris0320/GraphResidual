import copy,json,math,random,resource,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.nn import functional as F
from src.models.model import ResidualV2,SEQ_TYPES
from src.graph.data import Inputs,load_edges,COUNTS
def cpu_mb():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
def train_job(root,out,epochs=20,batch_size=1024,lr=.001,weight_decay=1e-5,clip=1.0,beta_init=0.0,preserve_weight=.1,residual_weight=.01,steps_per_epoch=120,sanity=False,seed=42):
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/"checkpoints").mkdir(exist_ok=True);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 inputs=Inputs(root);edges=load_edges(root,sanity);relations=[r for s,r,d,x in edges];model=ResidualV2(relations,beta_init).to(device);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay);bce=torch.nn.BCEWithLogitsLoss();rng=np.random.default_rng(seed);rows=[];covrows=[];best=None
 def enc(typ,idx):
  g,e,v=inputs.batch(typ,idx,device);return model.encode(typ,g,e,v),(g,e,v)
 for epoch in range(1,epochs+1):
  start=time.time();model.train();acc={k:0.0 for k in ["total","relation","preserve","residual","cos","gcos","ratio","variance","esmnorm","rnorm","scaled"]};counts={r:0 for r in relations};seen=0
  for step in range(steps_per_epoch):
   ri=step%len(edges);st,rn,dt,x=edges[ri];ix=rng.integers(0,len(x),batch_size);pos=x[ix];negdst=rng.integers(0,COUNTS[dt],batch_size)
   (hz,hr,hm),(hg,he,hv)=enc(st,pos[:,0]);(tz,tr,tm),(tg,te,tv)=enc(dt,pos[:,1]);(nz,nr,nm),_=enc(dt,negdst)
   relidx=torch.full((batch_size,),ri,dtype=torch.long,device=device);pl=model.score(hz,tz,relidx);nl=model.score(hz,nz,relidx);relation=bce(pl,torch.ones_like(pl))+bce(nl,torch.zeros_like(nl))
   preserve_terms=[];cosvals=[];gcos=[]
   for typ,z,r,e,mask in [(st,hz,hr,he,hm),(dt,tz,tr,te,tm)]:
    if typ in SEQ_TYPES and mask.any():
     preserve_terms.append(1-F.cosine_similarity(z[mask],e[mask]).mean());cosvals.append(F.cosine_similarity(z[mask],e[mask]).mean());gcos.append(F.cosine_similarity(r[mask],e[mask]).mean())
   preserve=torch.stack(preserve_terms).mean() if preserve_terms else torch.zeros((),device=device);scaled=torch.cat([(model.beta*hr).reshape(len(hr),-1),(model.beta*tr).reshape(len(tr),-1)]);residual=scaled.square().mean();loss=relation+preserve_weight*preserve+residual_weight*residual
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),clip);opt.step();model.beta.data.clamp_(-.05,.05)
   with torch.no_grad():model.beta.clamp_(-.5,.5);esmnorm=torch.cat([q for q in [he,te] if q is not None]).norm(dim=1).mean() if he is not None or te is not None else torch.tensor(0.,device=device);rall=torch.cat([hr,tr]);rnorm=rall.norm(dim=1).mean();sn=scaled.norm(dim=1).mean();ratio=sn/(esmnorm+1e-8);var=torch.cat([hz,tz]).var(dim=0).mean()
   vals=[loss,relation,preserve,residual,torch.stack(cosvals).mean() if cosvals else torch.tensor(1.,device=device),torch.stack(gcos).mean() if gcos else torch.tensor(0.,device=device),ratio,var,esmnorm,rnorm,sn]
   for k,v in zip(acc,vals):acc[k]+=float(v)
   counts[rn]+=batch_size;seen+=batch_size
  n=steps_per_epoch;row={"epoch":epoch,"total_loss":acc["total"]/n,"relation_loss":acc["relation"]/n,"preserve_loss":acc["preserve"]/n,"residual_loss":acc["residual"]/n,"beta":float(model.beta),"abs_beta":abs(float(model.beta)),"esm_norm_mean":acc["esmnorm"]/n,"graph_residual_norm_mean":acc["rnorm"]/n,"scaled_residual_norm_mean":acc["scaled"]/n,"residual_to_esm_norm_ratio":acc["ratio"]/n,"cosine_fused_esm":acc["cos"]/n,"cosine_graph_projected_esm":acc["gcos"]/n,"fused_variance":acc["variance"]/n,"optimizer_steps":steps_per_epoch,"positive_edges_seen":seen,"relation_coverage":sum(v>0 for v in counts.values())/len(counts),"epoch_seconds":time.time()-start,"gpu_peak_mb":torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else 0.0,"cpu_peak_mb":cpu_mb()}
  rows.append(row);covrows += [{"epoch":epoch,"relation":r,"positive_edges_seen":v} for r,v in counts.items()];pd.DataFrame(rows).to_csv(out/"metrics.tsv",sep="\t",index=False);pd.DataFrame(covrows).to_csv(out/"relation_coverage.tsv",sep="\t",index=False)
  score=row["relation_loss"]+preserve_weight*row["preserve_loss"]+residual_weight*row["residual_loss"]
  state={"model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"relations":relations,"config":{"beta_init":beta_init,"preserve_weight":preserve_weight,"residual_weight":residual_weight,"seed":seed},"epoch":epoch,"metrics":row}
  torch.save(state,out/"checkpoints/last_model.pt")
  if best is None or score<best[0]:best=(score,copy.deepcopy(state));torch.save(best[1],out/"checkpoints/best_model.pt")
  print(json.dumps(row))
 df=pd.DataFrame(rows);df[["epoch","beta","abs_beta"]].to_csv(out/"beta_statistics.tsv",sep="\t",index=False);df[["epoch","esm_norm_mean","graph_residual_norm_mean","scaled_residual_norm_mean","residual_to_esm_norm_ratio","cosine_fused_esm","cosine_graph_projected_esm","fused_variance"]].to_csv(out/"embedding_norm_statistics.tsv",sep="\t",index=False)
 safe=bool(np.isfinite(df.select_dtypes("number")).all().all() and df.abs_beta.max()<=.5 and df.residual_to_esm_norm_ratio.max()<.5 and df.cosine_fused_esm.min()>.7 and df.fused_variance.min()>1e-6 and (pd.DataFrame(covrows).groupby("relation").positive_edges_seen.sum()>0).all())
 (out/"training_summary.json").write_text(json.dumps({"status":"PASS" if safe else "FAIL","best_score":best[0],"final":rows[-1],"relations":relations},indent=2)+"\n")
 if not safe:raise RuntimeError("training safety gate failed")
 return rows[-1]
