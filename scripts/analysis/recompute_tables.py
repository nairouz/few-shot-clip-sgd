# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'results/inputs'
O=ROOT/'results/generated'
if not B.is_dir():
    raise SystemExit(f"Local result exports are required in {B}. Experimental results are not bundled; input requirements are documented in docs/ANALYSIS.md.")
O.mkdir(parents=True,exist_ok=True)
NAMES={'eurosat':'EuroSAT','fgvc':'FGVC-Aircraft','dtd':'DTD','oxford_pets':'Oxford-Pets','caltech101':'Caltech101','stanford_cars':'Stanford-Cars'}
ORDER=['eurosat','oxford_pets','dtd','fgvc','stanford_cars','caltech101']
manifest={}
def record(fn):
 p=B/fn;manifest[fn]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p)
def acc_table(tag,fn,cols):
 d=record(fn)
 assert not d.duplicated(['dataset','shots','seed']).any()
 assert all(d.groupby(['dataset','shots']).seed.apply(lambda x:set(x)=={1,2,3}))
 rows=[]
 for (ds,k),g in d.groupby(['dataset','shots']):
  r={'dataset':ds,'shots':int(k),'n_seeds':len(g)}
  for label,col in cols.items():
   r[label+'_mean']=g[col].mean();r[label+'_sample_sd']=g[col].std(ddof=1)
  rows.append(r)
 tab=pd.DataFrame(rows);tab['order']=tab.dataset.map({x:i for i,x in enumerate(ORDER)})
 tab=tab.sort_values(['order','shots']).drop(columns='order')
 tab.to_csv(O/f'{tag}_accuracy_dataset_shot_sample_sd.csv',index=False,float_format='%.12g')
 md=['| Dataset | Shots | SGD | Adam | Adam-M0 |','|---|---:|---:|---:|---:|']
 tex=[]
 for _,r in tab.iterrows():
  vals=[f'{r[x+"_mean"]:.2f} ± {r[x+"_sample_sd"]:.2f}' for x in cols]
  md.append('| '+NAMES[r.dataset]+' | '+str(int(r.shots))+' | '+' | '.join(vals)+' |')
  tv=[f'${r[x+"_mean"]:.2f}\\pm{r[x+"_sample_sd"]:.2f}$' for x in cols]
  tex.append(NAMES[r.dataset]+' & '+str(int(r.shots))+' & '+' & '.join(tv)+r' \\')
 (O/f'{tag}_accuracy_table.md').write_text('\n'.join(md)+'\n')
 (O/f'{tag}_accuracy_table_rows.tex').write_text('\n'.join(tex)+'\n')
 shot=d.groupby('shots')[list(cols.values())].mean().rename(columns={v:k for k,v in cols.items()})
 shot['SGD_minus_Adam']=shot.SGD-shot.Adam
 shot['M0_signed_gap_recovered_percent']=100*(shot['Adam-M0']-shot.Adam)/(shot.SGD-shot.Adam)
 shot.to_csv(O/f'{tag}_accuracy_by_shot.csv',float_format='%.12g')
 return d,tab
p2,t2=acc_table('p2','p2_accuracy_per_seed.csv',{'SGD':'sgd','Adam':'adam_raw','Adam-M0':'adam_m0'})
p5,t5=acc_table('p5','p5_accuracy_per_seed_verified.csv',{'SGD':'sgd_accuracy','Adam':'adam_accuracy','Adam-M0':'adam_m0_accuracy'})
# Cross-check the independent explicit P2 endpoint grid.
g=record('p2_per_seed_fixed_horizon_and_validation_grid.csv').set_index(['dataset','shots','seed'])
a=p2.set_index(['dataset','shots','seed'])
for c,ref in [('sgd','SGD_fixed_horizon_final_test_accuracy'),('adam_raw','raw_Adam_fixed_horizon_final_test_accuracy'),('adam_m0','Adam_M0_fixed_horizon_final_test_accuracy')]:
 assert np.allclose(a[c],g.loc[a.index,ref],rtol=0,atol=1e-10)
# Global mechanism medians/means from one row per independent dataset-shot-seed run.
for tag,fn,filter_opt in [('p2','p2_mechanism_per_seed_scope.csv',False),('p5','p5_mechanism_diagnostic_per_seed_scope.csv',True)]:
 d=record(fn);d=d[d.scope=='global']
 if filter_opt:d=d[d.optimizer=='sgd']
 assert not d.duplicated(['dataset','shots','seed']).any()
 cols=['initial_r_m','initial_r_v_batch' if tag=='p2' else 'initial_r_v','initial_coordinate_mismatch_mean_abs_log10','initial_fraction_underestimated','initial_adaptive_direction_norm_ratio','denominator_vs_numerator_collapse_factor']
 tab=d.groupby('shots')[cols].agg(['mean','median','count']);tab.columns=['_'.join(c) for c in tab.columns]
 tab.to_csv(O/f'{tag}_global_mechanism_by_shot.csv',float_format='%.12g')
# The published paper-proxy statistic is distinct from Adam-faithful direction due to epsilon placement.
p=record('p2_mechanism_by_shot.csv');p.to_csv(O/'p2_paper_proxy_mechanism_by_shot_source.csv',index=False)
# Global first applied update ratio; all accuracy cells, not only diagnostic subset.
for tag,fn,ratio in [('p2','p2_accuracy_per_seed.csv','actual_first_step_adam_to_sgd'),('p5','p5_first_step_budgets_per_seed_scope.csv','raw_Adam_to_SGD_first_step_budget_ratio')]:
 d=record(fn)
 if 'scope' in d:d=d[d.scope=='global']
 d.groupby(['dataset','shots'])[ratio].agg(['mean','median','min','max']).to_csv(O/f'{tag}_global_first_update_ratios.csv',float_format='%.12g')
 d.groupby('shots')[ratio].agg(['mean','median','min','max']).to_csv(O/f'{tag}_global_first_update_ratios_by_shot.csv',float_format='%.12g')
for tag in ['p2','p5']:
 d=record(tag+'_accuracy_primary_paired_bootstrap.csv');d.to_csv(O/(tag+'_verified_primary_intervals.csv'),index=False)
(O/'source_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Wrote',len(list(O.iterdir())),'files')
