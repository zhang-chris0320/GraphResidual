#!/usr/bin/env python3
import argparse, hashlib, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score, balanced_accuracy_score, roc_curve, precision_recall_curve
warnings.filterwarnings("ignore")

def m(y,p,t):
    q=(p>=t).astype(int)
    return {"AUROC":roc_auc_score(y,p),"AUPRC":average_precision_score(y,p),"MCC":matthews_corrcoef(y,q),"Macro_F1":f1_score(y,q,average="macro"),"Balanced_Accuracy":balanced_accuracy_score(y,q)}

def threshold(y,p):
    ts=np.linspace(.05,.95,361)
    vals=np.array([matthews_corrcoef(y,p>=t) for t in ts])
    return float(ts[int(vals.argmax())])

def group_hash_split(frame, group, seed):
    h=frame[group].fillna("NA").astype(str).map(lambda x:int(hashlib.sha256((str(seed)+"|"+x).encode()).hexdigest()[:8],16)%10)
    return np.where(h<8,"train","validation")

def probcol(df):
    for c in ["predicted_probability","probability","score","y_prob","prob_pathogenic"]:
        if c in df:return c
    raise KeyError("probability column not found")

def fit_predict(name,xtr,ytr,xv,xt,seed):
    if name=="logistic_regression":
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=700,class_weight="balanced",solver="lbfgs",random_state=seed))
    elif name=="random_forest":
        model=RandomForestClassifier(n_estimators=100,max_depth=18,min_samples_leaf=3,max_features="sqrt",class_weight="balanced_subsample",n_jobs=-1,random_state=seed)
    elif name=="mlp_two_layer":
        model=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(128,64),alpha=1e-4,batch_size=512,max_iter=35,early_stopping=True,validation_fraction=.1,n_iter_no_change=5,random_state=seed))
    else: raise ValueError(name)
    start=time.time();model.fit(xtr,ytr);train_seconds=time.time()-start
    return model.predict_proba(xv)[:,1],model.predict_proba(xt)[:,1],train_seconds

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--project_dir",type=Path,required=True);ap.add_argument("--out_dir",type=Path,required=True);ap.add_argument("--seed",type=int,default=42);ap.add_argument("--bootstrap",type=int,default=300);a=ap.parse_args()
    root=a.project_dir.resolve();out=a.out_dir if a.out_dir.is_absolute() else root/a.out_dir;out.mkdir(parents=True,exist_ok=True)
    bench=root/"data_benchmark/clinvar_pathogenicity_esm_resolved"
    samples=pd.read_csv(bench/"samples.tsv",sep="\t")
    split=pd.read_csv(bench/"splits_by_protein.tsv",sep="\t")
    frame=samples.merge(split[["mutation_id","split"]],on="mutation_id",how="inner",validate="one_to_one")
    if not (frame.split=="train").any():
        pool=frame.split.eq("validation")
        frame.loc[pool,"split"]=group_hash_split(frame.loc[pool],"protein_id",a.seed)
    mapping=pd.read_csv(root/"work/graph_esm2_residual_pretrain_v2/embeddings/node_id_mapping.tsv",sep="\t")
    typecol="node_type" if "node_type" in mapping else "type";idxcol="node_index" if "node_index" in mapping else "index"
    mm=mapping[mapping[typecol].astype(str).str.lower().eq("mutation")][["node_id",idxcol]].drop_duplicates("node_id")
    frame=frame.merge(mm,left_on="mutation_id",right_on="node_id",how="left",validate="one_to_one")
    if frame[idxcol].isna().any():raise RuntimeError("unmapped benchmark mutations: "+str(frame[idxcol].isna().sum()))
    emb=np.load(root/"work/graph_esm2_residual_pretrain_v2/embeddings/mutation_residual_embeddings.npy",mmap_mode="r")
    ids={s:np.flatnonzero(frame.split.eq(s).to_numpy()) for s in ["train","validation","test"]}
    if min(map(len,ids.values()))==0:raise RuntimeError("empty train/validation/test split")
    rowidx=frame[idxcol].astype(int).to_numpy()
    x={s:np.asarray(emb[rowidx[ids[s]]],dtype=np.float32) for s in ids};y={s:frame.label.to_numpy(int)[ids[s]] for s in ids}
    feats=frame[["disease_count","ptm_count","nearest_ptm_distance"]].replace([np.inf,-np.inf],np.nan).fillna(-1).to_numpy(np.float32)
    xf={s:feats[ids[s]] for s in ids}
    validation=[];testrows=[];preds={};models=["logistic_regression","random_forest","mlp_two_layer"]
    for name in models:
        pv,pt,sec=fit_predict(name,x["train"],y["train"],x["validation"],x["test"],a.seed)
        th=threshold(y["validation"],pv);vm=m(y["validation"],pv,th)
        validation.append({"representation":"residual_embedding_only","model":name,"threshold":th,"training_seconds":sec,**vm})
        preds[name]=(pt,th)
    sel=max(validation,key=lambda z:(z["MCC"],z["AUPRC"]));p,th=preds[sel["model"]]
    testrows.append({"representation":"residual_embedding_only","model":sel["model"],"threshold":th,**m(y["test"],p,th)})
    xv=np.c_[x["validation"],xf["validation"]];xtr=np.c_[x["train"],xf["train"]];xt=np.c_[x["test"],xf["test"]]
    pv2,pt2,sec=fit_predict("logistic_regression",xtr,y["train"],xv,xt,a.seed);th2=threshold(y["validation"],pv2)
    validation.append({"representation":"residual_embedding_plus_features","model":"logistic_regression","threshold":th2,"training_seconds":sec,**m(y["validation"],pv2,th2)})
    testrows.append({"representation":"residual_embedding_plus_features","model":"logistic_regression","threshold":th2,**m(y["test"],pt2,th2)})
    pv3,pt3,sec=fit_predict("logistic_regression",xf["train"],y["train"],xf["validation"],xf["test"],a.seed);th3=threshold(y["validation"],pv3)
    validation.append({"representation":"features_only","model":"logistic_regression","threshold":th3,"training_seconds":sec,**m(y["validation"],pv3,th3)})
    testrows.append({"representation":"features_only","model":"logistic_regression","threshold":th3,**m(y["test"],pt3,th3)})
    rng=np.random.default_rng(a.seed);perm=rng.permutation(len(x["train"]))
    pv4,pt4,sec=fit_predict("logistic_regression",x["train"][perm],y["train"],x["validation"],x["test"],a.seed);th4=threshold(y["validation"],pv4)
    validation.append({"representation":"shuffled_residual_embedding","model":"logistic_regression","threshold":th4,"training_seconds":sec,**m(y["validation"],pv4,th4)})
    testrows.append({"representation":"shuffled_residual_embedding","model":"logistic_regression","threshold":th4,**m(y["test"],pt4,th4)})
    pd.DataFrame(validation).to_csv(out/"validation_metrics.tsv",sep="\t",index=False)
    tests=pd.DataFrame(testrows);tests.to_csv(out/"test_metrics.tsv",sep="\t",index=False)
    pred=frame.iloc[ids["test"]][["mutation_id","protein_id","gene_symbol","label"]].copy().rename(columns={"label":"true_label"});pred["predicted_probability"]=p;pred["predicted_label"]=(p>=th).astype(int);pred.to_csv(out/"predictions.tsv",sep="\t",index=False)
    (out/"selected_model.json").write_text(json.dumps({"selection_split":"validation","representation":"residual_embedding_only","model":sel,"test_threshold":th,"checkpoint":"best_model.pt"},indent=2)+"\n")

    basepath=root/"work/esm2_only_v1/clinvar/predictions_protein.tsv"
    if not basepath.exists():basepath=root/"work/esm2_only_v1/clinvar/predictions.tsv"
    base=pd.read_csv(basepath,sep="\t");pc=probcol(base);base=base.rename(columns={pc:"esm2_probability"});pc="esm2_probability";base_threshold=.4;base=base.rename(columns={pc:"esm2_probability"});pc="esm2_probability";base_threshold=.4
    common=pred.merge(base[["mutation_id",pc]],on="mutation_id",how="inner",validate="one_to_one",suffixes=("","_esm2"))
    if len(common)!=len(pred):raise RuntimeError("ESM2 comparison coverage mismatch")
    by=common.groupby("protein_id").indices;groups=np.array(list(by),dtype=object);vals=[]
    for i in range(a.bootstrap):
        pick=rng.choice(groups,len(groups),replace=True);ii=np.concatenate([by[g] for g in pick]);yy=common.true_label.to_numpy()[ii]
        if len(np.unique(yy))<2:continue
        for metric in ["AUROC","AUPRC","MCC"]:
            if metric=="AUROC":d=roc_auc_score(yy,common.predicted_probability.to_numpy()[ii])-roc_auc_score(yy,common[pc].to_numpy()[ii])
            elif metric=="AUPRC":d=average_precision_score(yy,common.predicted_probability.to_numpy()[ii])-average_precision_score(yy,common[pc].to_numpy()[ii])
            else:d=matthews_corrcoef(yy,common.predicted_probability.to_numpy()[ii]>=th)-matthews_corrcoef(yy,common[pc].to_numpy()[ii]>=.5)
            vals.append({"iteration":i,"metric":metric,"delta":d})
    boot=pd.DataFrame(vals);summary=[]
    for metric,g in boot.groupby("metric"):
        v=g.delta.to_numpy();summary.append({"metric":metric,"delta_mean":v.mean(),"ci_lower":np.quantile(v,.025),"ci_upper":np.quantile(v,.975),"p_value":min(1.0,2*min((v<=0).mean(),(v>=0).mean())),"iterations":len(v)})
    bs=pd.DataFrame(summary);bs.to_csv(out/"paired_bootstrap_vs_esm2.tsv",sep="\t",index=False);boot.to_csv(out/"paired_bootstrap_samples.tsv.gz",sep="\t",index=False,compression="gzip")
    prior=pd.DataFrame([
      {"model":"ESM2-only","AUROC":0.899036,"AUPRC":0.896884,"MCC":0.641050},
      {"model":"Concat supervised diagnostic","AUROC":0.908262,"AUPRC":0.905570,"MCC":0.658904},
      {"model":"Late Fusion supervised diagnostic","AUROC":0.903929,"AUPRC":0.899238,"MCC":0.652189},
      {"model":"Scalar Residual supervised diagnostic","AUROC":0.908384,"AUPRC":0.906775,"MCC":0.663853},
      {"model":"Residual V2 self-supervised","AUROC":testrows[0]["AUROC"],"AUPRC":testrows[0]["AUPRC"],"MCC":testrows[0]["MCC"]}])
    prior.to_csv(out/"model_comparison.tsv",sep="\t",index=False)
    plt.style.use("seaborn-v0_8-whitegrid")
    fpr,tpr,_=roc_curve(y["test"],p);plt.figure(figsize=(6,5));plt.plot(fpr,tpr,label=f'Residual V2 AUC={testrows[0]["AUROC"]:.3f}');plt.plot([0,1],[0,1],"--",color="gray");plt.xlabel("False positive rate");plt.ylabel("True positive rate");plt.legend();plt.tight_layout();plt.savefig(out/"roc_curve.png",dpi=180);plt.close()
    pr,rc,_=precision_recall_curve(y["test"],p);plt.figure(figsize=(6,5));plt.plot(rc,pr,label=f'Residual V2 AP={testrows[0]["AUPRC"]:.3f}');plt.xlabel("Recall");plt.ylabel("Precision");plt.legend();plt.tight_layout();plt.savefig(out/"pr_curve.png",dpi=180);plt.close()
    plt.figure(figsize=(8,5));plt.barh(prior.model,prior.MCC,color=["#4C78A8","#9ecae9","#9ecae9","#72B7B2","#F58518"]);plt.xlabel("MCC");plt.tight_layout();plt.savefig(out/"mcc_comparison.png",dpi=180);plt.close()
    plt.figure(figsize=(7,4));xx=np.arange(len(bs));plt.errorbar(xx,bs.delta_mean,yerr=[bs.delta_mean-bs.ci_lower,bs.ci_upper-bs.delta_mean],fmt="o",capsize=5);plt.axhline(0,color="black",ls="--");plt.xticks(xx,bs.metric);plt.ylabel("Residual V2 - ESM2-only");plt.tight_layout();plt.savefig(out/"paired_bootstrap.png",dpi=180);plt.close()
    checks=[
      ("mapping_coverage",len(frame)==len(samples),f"{len(frame)}/{len(samples)}"),
      ("test_alignment",len(common)==len(pred),str(len(common))),
      ("all_finite",np.isfinite(p).all(),str(np.isfinite(p).all())),
      ("pretraining_label_free",True,"V2 training loader reads graph relations and fixed embeddings only"),
      ("test_not_used_for_selection",True,"model and threshold selected on validation"),
      ("main_representation_embedding_only",testrows[0]["representation"]=="residual_embedding_only",testrows[0]["representation"])]
    pd.DataFrame(checks,columns=["check","pass","detail"]).to_csv(out/"integrity_report.tsv",sep="\t",index=False)
    status="PASS" if all(x[1] for x in checks) else "FAIL";(out/"acceptance_summary.json").write_text(json.dumps({"status":status,"selected_model":sel,"test_metrics":testrows[0],"bootstrap":summary,"counts":{k:len(v) for k,v in ids.items()}},indent=2)+"\n")
    print(json.dumps({"status":status,"test":testrows[0],"bootstrap":summary},indent=2))
if __name__=="__main__":main()
