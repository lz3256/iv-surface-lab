"""Descriptive overlapping +/-1-session windows around expiry switches."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def run(package):
 p=Path(package);f=pd.read_csv(p/'diagnostics/same_endpoint_daily.csv');rows=[]
 for kind in ['monthly_proxy_switch','other_switch']:
  ix=np.flatnonzero(f.switch_category.eq(kind));ix=ix[(ix>0)&(ix<len(f)-1)]
  for sample in ['raw_all','matched_triplets']:
   valid=np.isfinite(f.raw_skew_change.to_numpy()[ix[:,None]+np.array([-1,0,1])]).all(axis=1)
   if sample=='matched_triplets':valid &= np.isfinite(f.same_skew_change.to_numpy()[ix[:,None]+np.array([-1,0,1])]).all(axis=1)
   anchors=ix[valid]
   for lag in [-1,0,1]:
    for signal in (['raw_skew_change'] if sample=='raw_all' else ['raw_skew_change','same_skew_change']):
     values=f[signal].iloc[anchors+lag]
     rows.append({'category':kind,'sample':sample,'signal':signal,'lag_sessions':lag,'n':len(values),'mean':values.mean(),'variance':values.var()})
 pd.DataFrame(rows).to_csv(p/'diagnostics/switch_aligned_groups.csv',index=False)
 print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--package',required=True);args=a.parse_args();run(args.package)
