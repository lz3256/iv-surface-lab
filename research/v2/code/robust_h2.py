"""Exploratory H2 models; every preprocessing statistic is fitted on past labels."""
import numpy as np
import pandas as pd
from ivsurf import core


def feature_specs(panel):
    p = panel.copy()
    p['log_har_fwd_21'] = np.log(p.har_fwd_21.clip(lower=1e-10))
    market = ['ret_1','ret_5','log_rv_22','log_iv_var_30','term_slope']
    return p, {'R0_mean':[], 'R1_market':market,
               'R2_HAR':market+['log_har_fwd_21'],
               'R3_skew_change':market+['log_har_fwd_21','skew_change']}


def fit_preprocessor(x, clip_index, quantiles=(.01,.99)):
    lower, upper = np.quantile(x[:,clip_index],quantiles)
    clean = x.copy()
    clean[:,clip_index] = np.clip(clean[:,clip_index],lower,upper)
    return clean, float(lower), float(upper)


def walk(panel, y, end, models, min_train=504, window=1008, refit=21,
         penalty_per_obs=1., quantiles=(.01,.99)):
    if not (0 <= quantiles[0] < quantiles[1] <= 1) or penalty_per_obs <= 0:
        raise ValueError('Invalid training-only clipping or ridge settings.')
    union = list(dict.fromkeys(c for columns in models.values() for c in columns))
    clip_index = union.index('skew_change')
    x = panel[union].to_numpy(float)
    yy = y.to_numpy(float)
    eligible = np.isfinite(x).all(axis=1)&np.isfinite(yy)&end.notna().to_numpy()
    out = panel[['date','asof_time','entry_time']].copy()
    out['target'] = yy
    out['target_end'] = end
    for name in list(models)+['train_n','fit_row','effective_alpha','skew_lower','skew_upper','skew_used']:
        out[name] = np.nan
    out['train_end_max'] = pd.Series(pd.NaT,index=panel.index,dtype='datetime64[ns, UTC]')
    selectors = {name:[union.index(c) for c in columns] for name,columns in models.items()}
    fitted, fit_at, idx, lo, hi = None, -refit, None, None, None
    fit_audit = []
    for i in range(len(panel)):
        if not np.isfinite(x[i]).all():
            continue
        if fitted is None or i-fit_at >= refit:
            idx = np.flatnonzero(eligible&(end<panel.asof_time.iloc[i]).to_numpy())[-window:]
            if len(idx)<min_train:
                continue
            clean,lo,hi = fit_preprocessor(x[idx],clip_index,quantiles)
            alpha = penalty_per_obs*len(idx)
            fitted = {name:core.ridge_fit(clean[:,sel],yy[idx],alpha) if sel else float(yy[idx].mean())
                      for name,sel in selectors.items()}
            fit_at = i
            fit_audit.append({'fit_row':i,'fit_date':panel.date.iloc[i],
                              'train_n':len(idx),'train_first_date':panel.date.iloc[idx[0]],
                              'train_last_date':panel.date.iloc[idx[-1]],
                              'train_end_max':str(end.iloc[idx].max()),
                              'asof_time':str(panel.asof_time.iloc[i]),
                              'effective_alpha':alpha,'skew_lower':lo,'skew_upper':hi})
        current = x[i].copy()
        current[clip_index] = np.clip(current[clip_index],lo,hi)
        for name,sel in selectors.items():
            out.loc[i,name] = core.ridge_predict(fitted[name],current[sel]) if sel else fitted[name]
        out.loc[i,['train_n','fit_row','effective_alpha','skew_lower','skew_upper','skew_used']] = [
            len(idx),fit_at,penalty_per_obs*len(idx),lo,hi,current[clip_index]]
        out.loc[i,'train_end_max'] = end.iloc[idx].max()
    good = out.R0_mean.notna()
    assert (out.loc[good,'train_end_max']<out.loc[good,'asof_time']).all()
    return out,pd.DataFrame(fit_audit)
