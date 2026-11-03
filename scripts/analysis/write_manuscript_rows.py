from pathlib import Path
import hashlib,json
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[2]
T=ROOT/'results/inputs'; Q=ROOT/'results/generated'; O=Q/'manuscript_rows'
if not T.is_dir():
    raise SystemExit(f"Local result exports are required in {T}. Experimental results are not bundled; input requirements are documented in docs/ANALYSIS.md.")
O.mkdir(parents=True,exist_ok=True)
manifest=[]
def write_new(name,content,sources,columns,notes):
 p=O/name
 p.write_text(content)
 manifest.append({'output':str(p.relative_to(ROOT)),'columns':columns,'notes':notes,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'sources':[{'path':str(s.relative_to(ROOT)),'sha256':hashlib.sha256(s.read_bytes()).hexdigest()} for s in sources]})
for tag in ['p2','p5']:
 src=Q/f'{tag}_accuracy_table_rows.tex'
 write_new(f'{tag}_accuracy_rows.tex',src.read_text(),[src,T/('p2_accuracy_per_seed.csv' if tag=='p2' else 'p5_accuracy_per_seed_verified.csv')],['Dataset','K','SGD mean ± sample SD','Adam mean ± sample SD','Adam-M0 mean ± sample SD'],'Three seeds per row; sample SD ddof=1; fixed epoch-30 test accuracy.')
d=pd.read_csv(T/'p1_closure_by_shot.csv');allr=pd.read_csv(T/'p1_closure_overall.csv').iloc[0]
cols=['mean_sgd','mean_adam','mean_adam_m0','mean_adam_mt','m0_closure_percent','mt_total_closure_percent']
rows=[]
for _,r in d.iterrows():rows.append(str(int(r.shots))+' & '+' & '.join(f'{r[c]:.2f}' for c in cols)+r' \\')
rows+= [r'\midrule', 'Mean & '+' & '.join(f'{allr[c]:.2f}' for c in cols)+r' \\']
write_new('p1_matching_rows.tex','\n'.join(rows)+'\n',[T/'p1_closure_by_shot.csv',T/'p1_closure_overall.csv'],['K','SGD','Adam','Adam-M0','Adam-Mt','M0 signed gap closure (%)','Mt signed gap closure (%)'],'Nine cells per shot (Aircraft, Oxford-Pets, Stanford-Cars × seeds1–3); 27 overall. SGD reference from P2 with matched support membership. Ratio of macro mean differences; not clipped.')
d=pd.read_csv(T/'p1_lr_warmup_vs_sgd.csv');ci=pd.read_csv(T/'p1_lr_warmup_vs_sgd_bootstrap.csv').set_index('variant')
labels={
'P1_Adam_LR_1e-4':r'Adam, $10^{-4}$','P1_Adam_LR_3e-5':r'Adam, $3\!\times\!10^{-5}$','P1_Adam_LR_1e-5':r'Adam, $10^{-5}$','P1_Adam_LR_3e-6':r'Adam, $3\!\times\!10^{-6}$','P1_Adam_LR_1e-6':r'Adam, $10^{-6}$','P1_Adam_LR_3e-7':r'Adam, $3\!\times\!10^{-7}$','P1_Adam_LR_1e-7':r'Adam, $10^{-7}$','P1_Adam_Warmup_05pct_Constant':r'Adam, 5\% warmup + constant','P1_Adam_Warmup_05pct_Cosine':r'Adam, 5\% warmup + cosine','P1_Adam_Warmup_20pct_Constant':r'Adam, 20\% warmup + constant','P1_Adam_Warmup_20pct_Cosine':r'Adam, 20\% warmup + cosine'}
rows=[r'SGD, $10^{-4}$ & '+f'{d.mean_sgd_accuracy.iloc[0]:.2f}'+r' & $0.00$ & --- \\']
for v,label in labels.items():
 r=d.set_index('variant').loc[v];c=ci.loc[v]
 assert np.isclose(r.mean_variant_minus_sgd,c.estimate)
 rows.append(label+' & '+f'{r.mean_final_accuracy:.2f} & ${r.mean_variant_minus_sgd:+.2f}$ & $[{c.ci95_low:+.2f}, {c.ci95_high:+.2f}]$'+r' \\')
write_new('p1_lr_warmup_rows.tex','\n'.join(rows)+'\n',[T/'p1_lr_warmup_vs_sgd.csv',T/'p1_lr_warmup_vs_sgd_bootstrap.csv'],['Method / LR or schedule','Final accuracy (%)','Method minus SGD (pp)','95% dataset-cluster CI of paired difference'],'All 11 variants on same 54 cells: six datasets × K={1,4,16} × seeds1–3. Warmup variants use base LR1e-4. 20,000 whole-dataset bootstrap resamples. SGD reference from P2.')
d=pd.read_csv(T/'group2_trajectory_overall.csv');ex=pd.read_csv(T/'group2_exact_trajectory_metric_matrix.csv');rall=pd.read_csv(T/'group2_trajectory_per_run.csv')
rows=[];rows_csv=[]
for opt in ['adam','sgd']:
 for _,r in d[d.optimizer==opt].sort_values('learning_rate',ascending=False).iterrows():
  a=ex[(ex.optimizer.str.lower()==opt)&np.isclose(ex.learning_rate,r.learning_rate,rtol=0,atol=1e-12)]
  v=a[[f'epoch300_accuracy_seed{i}' for i in [1,2,3]]].to_numpy().ravel();v=v[np.isfinite(v)]
  n=len(v);mean=v.mean()
  exp=int(round(np.log10(r.learning_rate)))
  name='Adam' if opt=='adam' else 'SGD'
  rows.append(name+' & '+f'$10^{{{exp}}}$ & {int(r.n_runs)} & {r.epoch30_mean:.2f} & {mean:.2f} & {n} & {r.peak_mean:.2f} & {r.last_observed_mean:.2f} & {r.peak_to_last_drop_mean:.2f}'+r' \\')
  rr=rall[(rall.optimizer==opt)&np.isclose(rall.learning_rate,r.learning_rate,rtol=0,atol=1e-12)]
  rows_csv.append({'optimizer':name,'lr':r.learning_rate,'n':int(r.n_runs),'epoch30_mean':r.epoch30_mean,'epoch300_mean':mean,'n_epoch300':n,'peak_mean':r.peak_mean,'last_observed_mean':r.last_observed_mean,'peak_to_last_mean':r.peak_to_last_drop_mean,'min_last_epoch':rr.last_epoch.min(),'max_last_epoch':rr.last_epoch.max()})
write_new('long_horizon_rows.tex','\n'.join(rows)+'\n',[T/'group2_trajectory_overall.csv',T/'group2_exact_trajectory_metric_matrix.csv',T/'group2_trajectory_per_run.csv'],['Optimizer','LR','n','Epoch30','Epoch300','n at epoch300','Peak','Last observed','Peak-to-last drop (pp)'],'Six datasets, K=1, seeds1–3. Adam1e-6 targets600epochs; others300. Peak/last summarize18runs including interrupted traces, without forward filling. Epoch300 averages only observed values; its n is explicit. Peak is descriptive, never a checkpoint selection rule.')
pd.DataFrame(rows_csv).to_csv(Q/'long_horizon_table_verification.csv',index=False)
(Q/'camera_ready_table_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(pd.DataFrame(rows_csv).to_string(index=False))
