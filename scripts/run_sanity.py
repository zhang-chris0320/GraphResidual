import argparse,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
from src.training.training import train_job
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--project_dir",type=Path,required=True);ap.add_argument("--out_dir",type=Path,required=True);a=ap.parse_args();r=a.project_dir.resolve();o=(a.out_dir if a.out_dir.is_absolute() else r/a.out_dir);o.mkdir(parents=True,exist_ok=True);rows=[]
 for pw in [.05,.1,.2]:
  for rw in [.001,.01]:
   name=f"preserve_{pw}_residual_{rw}";last=train_job(r,o/name,epochs=4,batch_size=512,steps_per_epoch=24,sanity=True,preserve_weight=pw,residual_weight=rw);rows.append({"name":name,"preserve_weight":pw,"residual_weight":rw,**last})
 safe=[x for x in rows if x["cosine_fused_esm"]>.7 and x["residual_to_esm_norm_ratio"]<.5 and x["relation_coverage"]==1.0 and abs(x["beta"])>1e-8];best=min(safe,key=lambda x:x["relation_loss"]+.1*x["preserve_loss"]+.01*x["residual_loss"]);(o/"sanity_selection.json").write_text(json.dumps({"status":"PASS","selected":best,"all":rows},indent=2)+"\n");print("SANITY_PASS",best["preserve_weight"],best["residual_weight"],best["beta"])
if __name__=="__main__":main()
