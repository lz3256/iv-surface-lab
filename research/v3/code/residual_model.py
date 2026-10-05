"""Unpenalized partial slope added to an unchanged control forecast."""
import numpy as np

CONTROLS=['ret_1','ret_5','log_rv_22','log_iv_var_30','term_slope','log_har_fwd_21']


def fit(x,s,y):
    mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-12]=1.
    design=np.column_stack([np.ones(len(x)),(x-mean)/scale])
    lo,hi=np.quantile(s,[.01,.99]);clipped=np.clip(s,lo,hi)
    projection,_,rank,_=np.linalg.lstsq(design,clipped,rcond=None)
    z=clipped-design@projection;variance=float(z@z/len(z))
    if variance<=1e-16:return None
    beta=float(z@y/(z@z))
    # FWL identity, checked using a different joint regression.
    joint=np.linalg.lstsq(np.column_stack([design,clipped]),y,rcond=None)[0][-1]
    if not np.isclose(beta,joint,rtol=1e-6,atol=1e-9):raise AssertionError('FWL slope mismatch')
    return {'mean':mean.tolist(),'scale':scale.tolist(),'projection':projection.tolist(),
            'lower':float(lo),'upper':float(hi),'beta':beta,'residual_variance':variance,
            'rank':int(rank),'n':len(x),'orthogonality_max_abs':float(np.abs(design.T@z).max())}


def predict(fitted,x,s):
    if fitted is None or not np.isfinite(s):return np.nan,np.nan
    design=np.r_[1.,(x-np.array(fitted['mean']))/np.array(fitted['scale'])]
    z=float(np.clip(s,fitted['lower'],fitted['upper'])-design@np.array(fitted['projection']))
    return z,fitted['beta']*z
