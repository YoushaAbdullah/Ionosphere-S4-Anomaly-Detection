"""Reduced synthetic end-to-end check; never use its outputs as research results.

Run with the pinned environment: python tests/smoke_pipeline.py
This deliberately reduces folds, pseudo-class counts, GA iterations and fit size.
The production notebook and archived 2019 results are never changed.
"""
import os,sys,tempfile,json
from pathlib import Path
import numpy as np
import pandas as pd
os.environ['MPLBACKEND']='Agg'
output=Path(tempfile.mkdtemp(prefix='q4-paper-smoke-'))
os.environ['Q4_PAPER_OUTPUT']=str(output)
rng=np.random.default_rng(12)
n=36*60
dates=pd.Timestamp('2020-01-01')+pd.to_timedelta(np.repeat(np.arange(36),60),unit='D')
features=['L1 CNo','L2 CNo','Azimuth_sin','Azimuth_cos','Elevation','Average of Carrier divergence','Corrected_S4','TEC at TOW','Sigma of Carrier Divergence']
frame=pd.DataFrame(rng.normal(size=(n,9)),columns=features)
frame['Corrected_S4']=np.abs(rng.normal(.05,.015,n))
frame.loc[::23,'Corrected_S4']=rng.uniform(.21,.35,len(frame.loc[::23]))
frame['LABEL']=(frame.Corrected_S4>=.2).astype(int)
frame['_DATE']=dates
frame['Second']=np.tile(np.arange(60)*60,36)
frame['PRN']=np.tile(np.arange(1,61),36)
namespace={'FEATURES':features,'df_clean':frame,'S4_THRESHOLD':.2,'display':lambda *objects:None,
           'train_df':frame.loc[frame._DATE<dates[20*60]],
           'val_df':frame.loc[(frame._DATE>=dates[20*60])&(frame._DATE<dates[28*60])],
           'test_df':frame.loc[frame._DATE>=dates[28*60]]}
root = Path(__file__).resolve().parents[1]
notebook = json.loads((root / 'Q4 Paper Code/scintillation_anomaly_detection.ipynb').read_text())
cells = [cell for cell in notebook['cells'][12:] if cell['cell_type'] == 'code']
source = '\n'.join(''.join(cell['source']) for cell in cells[:-1])
exec(compile(source,'paper_pipeline_smoke','exec'),namespace)
namespace.update(MAX_NORMAL_ROWS=200,NORMALIZATION_NORMAL_ROWS=80,N_TIME_FOLDS=2,
                 PSEUDO_K=(3,5,7),GA_POPULATION=4,GA_GENERATIONS=1,BOOTSTRAP_REPLICATES=20)
assert not namespace['binary_predictions'](np.array([.5]),.5,'>')[0]
assert namespace['binary_predictions'](np.array([.5]),.5,'>=')[0]
namespace['plt'].show=lambda: namespace['plt'].close('all')
result=namespace['run_paper_workflow']()
assert (output/'research_report.html').exists()
frozen=json.loads((output/'frozen_selection.json').read_text())
assert sum(x['design_family']=='Heterogeneous' for x in frozen['ensembles'].values())==19
assert all(len(x['base_algorithms'])>=2 for x in frozen['ensembles'].values() if x['family']=='Heterogeneous')
signature_before=namespace['cv_fitting_signature']()
namespace['SEED']+=1
assert namespace['cv_fitting_signature']()!=signature_before
namespace['SEED']-=1
assert pd.read_csv(output/'tables/main_q4_comparison.csv').Threshold_Method.eq('Paper_0.5').all()
assert (output/'tables/study_iii_method_steps.csv').exists()
assert len(pd.read_csv(output/'tables/study_iii_single_homogeneous_heterogeneous.csv'))==3
assert len(pd.read_csv(output/'tables/final_test_metrics_all_models.csv'))==65*4
folds=pd.read_csv(output/'tables/time_folds.csv')
assert (pd.to_datetime(folds.train_end)<pd.to_datetime(folds.validation_start)).all()
selected=pd.read_csv(output/'tables/feature_selection.csv')
for row in selected.itertuples():
 assert 'Corrected_S4' in row.features
for path in (output/'selectors').glob('*_normal_fit_indices.csv'):
 idx=pd.read_csv(path).Original_Row_Index
 assert frame.loc[idx,'LABEL'].eq(0).all()
 assert len(idx)<=200
for name in ['all_cluster_candidate_cv_ranking.csv','main_q4_comparison.csv','ensemble_membership.csv']:
 assert (output/'tables'/name).exists()
print('SMOKE TEST PASSED',output)
