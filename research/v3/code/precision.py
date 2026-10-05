"""Paired loss-ratio bootstrap; units are fractions, not percentages."""
import numpy as np


def compare(y,baseline,added,block,reps=5000,seed=42,threshold=.001,family=4):
    y,a,b=map(lambda x:np.asarray(x,float),(y,baseline,added))
    valid=np.isfinite(y)&np.isfinite(a)&np.isfinite(b)
    la=np.where(valid,(y-a)**2,np.nan);gain=np.where(valid,(y-a)**2-(y-b)**2,np.nan)
    if valid.sum()<20:return {'n':int(valid.sum()),'status':'insufficient_sample'}
    base=float(np.nanmean(la));g=float(np.nanmean(gain));obs=g/base
    rng=np.random.default_rng(seed);n=len(y);block=min(block,n);ratios=[]
    for _ in range(reps):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        ix=((starts[:,None]+np.arange(block))%n).ravel()[:n]
        if np.isfinite(la[ix]).any():
            denom=np.nanmean(la[ix])
            if denom>0:ratios.append(np.nanmean(gain[ix])/denom)
    boot=np.asarray(ratios);lo,hi=np.quantile(boot,[.025,.975]);upper=np.quantile(boot,.95)
    upper_family=np.quantile(boot,1-.05/family)
    return {'n':int(valid.sum()),'mse_baseline':base,'mse_added':float(np.nanmean(np.where(valid,(y-b)**2,np.nan))),
            'gain':g,'incremental_r2':obs,'r2_low95':float(lo),'r2_high95':float(hi),
            'r2_upper_one_sided95':float(upper),'r2_upper_bonferroni95':float(upper_family),
            'p':float((1+(np.abs(boot-obs)>=abs(obs)).sum())/(len(boot)+1)),
            'threshold':threshold,'upper95_below_threshold':bool(upper<threshold),
            'upper97_5_below_threshold':bool(hi<threshold),
            'simultaneous_upper_below_threshold':bool(upper_family<threshold),
            'within_symmetric_threshold_95':bool(lo>-threshold and hi<threshold),
            'bootstrap_reps':len(boot),'block':block,'status':'exploratory_algorithm_specific'}
