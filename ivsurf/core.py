"""Calendar validation, causal features, HAR and a shared forecasting engine."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

MARKET = ['date','asof_time','reference_time','reference_price','rv_available_time',
          'open','high','low','close','entry_session','entry_time','entry_price']
SURFACE = ['iv_atm_30','iv_put25_30','iv_call25_30','iv_atm_91']
DIVIDENDS = ['cash_dividend','is_ex_dividend','entry_cash_dividend','entry_is_ex_dividend',
             'entry_dividend_available_time']


def require(d, names):
    missing = sorted(set(names)-set(d.columns))
    if missing:
        raise ValueError(f'Missing columns: {missing}')


def dates(s):
    return pd.to_datetime(s, format='%Y-%m-%d', errors='raise').dt.strftime('%Y-%m-%d')


def times(s, name):
    if not s.astype(str).str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$', regex=True).all():
        raise ValueError(f'{name}: explicit timezone required, e.g. 2024-01-03T09:31:00-05:00')
    return pd.to_datetime(s, utc=True, format='mixed', errors='raise')


def calendar_check(calendar):
    c = calendar.copy()
    require(c, ['date','session_open','session_close'])
    c['date'] = dates(c.date)
    if c.date.duplicated().any() or not c.date.is_monotonic_increasing:
        raise ValueError('Calendar dates must be unique and sorted.')
    for name in ['session_open','session_close']:
        c[name] = times(c[name], name)
        if not c[name].is_monotonic_increasing or c[name].duplicated().any():
            raise ValueError('Calendar session times must be unique and sorted.')
    if not (c.session_open < c.session_close).all():
        raise ValueError('Calendar open must precede close.')
    return c.reset_index(drop=True)


def validate(daily, calendar, metadata):
    d = daily.copy().reset_index(drop=True)
    c = calendar_check(calendar)
    require(d, MARKET)
    if len(d) < 2:
        raise ValueError('Need at least two sessions.')
    for key in ['synthetic','surface_kind','availability_basis','reference_instrument','execution_instrument','price_basis']:
        if key not in metadata:
            raise ValueError(f'metadata.json needs {key}')
    if not isinstance(metadata['synthetic'], bool):
        raise ValueError('metadata.synthetic must be a JSON boolean.')
    if metadata['surface_kind'] not in ['curve','proxies']:
        raise ValueError('surface_kind must be curve or proxies.')
    if metadata['availability_basis'] not in ['verified','assumed','synthetic']:
        raise ValueError('availability_basis must be verified, assumed or synthetic.')
    for col in ['date','entry_session']:
        d[col] = dates(d[col])
        if d[col].duplicated().any() or not d[col].is_monotonic_increasing:
            raise ValueError(f'{col} must be unique and sorted.')
    expected = c.loc[c.date.between(d.date.iloc[0], d.date.iloc[-1]), 'date'].tolist()
    if d.date.tolist() != expected:
        raise ValueError('Missing/extra signal sessions relative to calendar. Keep missing-surface rows; never drop market sessions.')
    ordinal = pd.Series(np.arange(len(c)), index=c.date)
    entry_pos = d.entry_session.map(ordinal)
    signal_pos = d.date.map(ordinal)
    if entry_pos.isna().any() or not (np.diff(entry_pos) == 1).all():
        raise ValueError('Entry sessions must be consecutive in the calendar, without gaps.')
    offset = entry_pos-signal_pos
    if offset.nunique() != 1 or offset.iloc[0] < 0:
        raise ValueError('Use one nonnegative, fixed entry-session offset throughout an experiment.')
    for col in ['asof_time','reference_time','rv_available_time','entry_time']:
        d[col] = times(d[col], col)
        if not d[col].is_monotonic_increasing:
            raise ValueError(f'{col} must increase with sessions.')
    if not (d.entry_time > d.asof_time).all():
        raise ValueError('entry_time must be strictly after asof_time.')
    if not (d.reference_time <= d.asof_time).all() or not (d.rv_available_time <= d.asof_time).all():
        raise ValueError('A reference price or daily RV is unavailable at asof_time.')
    ci = c.set_index('date')
    signal_close = d.date.map(ci.session_close)
    if not (d.rv_available_time >= signal_close).all():
        raise ValueError('This EOD model needs complete-session OHLC; rv_available_time cannot precede session_close.')
    op = d.entry_session.map(ci.session_open)
    cl = d.entry_session.map(ci.session_close)
    if not ((d.entry_time >= op) & (d.entry_time <= cl)).all():
        raise ValueError('entry_time is outside its declared trading session.')
    for col in ['reference_price','entry_price','open','high','low','close']:
        d[col] = pd.to_numeric(d[col], errors='raise')
        if not (np.isfinite(d[col]) & (d[col] > 0)).all():
            raise ValueError(f'{col} contains missing/nonpositive values. Resolve market data; do not drop rows.')
    if not ((d.high >= d[['open','close']].max(axis=1)) &
            (d.low <= d[['open','close']].min(axis=1)) & (d.high >= d.low)).all():
        raise ValueError('Invalid OHLC envelope.')
    if 'surface_available_time' in d:
        d['surface_available_time'] = times(d.surface_available_time, 'surface_available_time')
        if not (d.surface_available_time <= d.asof_time).all():
            raise ValueError('Surface unavailable at asof_time.')
    dividend_mode = metadata.get('return_basis') == 'cash_distribution_total_return'
    if dividend_mode or any(col in d for col in DIVIDENDS):
        if not dividend_mode:
            raise ValueError('Dividend columns require metadata.return_basis=cash_distribution_total_return.')
        require(d, DIVIDENDS)
        for col in ['cash_dividend','entry_cash_dividend']:
            d[col] = pd.to_numeric(d[col],errors='raise')
            if not (np.isfinite(d[col]) & (d[col]>=0)).all():
                raise ValueError(f'{col} must be complete, finite and nonnegative; missing is not zero.')
        for amount,flag in [('cash_dividend','is_ex_dividend'),('entry_cash_dividend','entry_is_ex_dividend')]:
            d[flag] = pd.to_numeric(d[flag],errors='raise')
            if not d[flag].isin([0,1]).all() or not d[flag].eq((d[amount]>0).astype(int)).all():
                raise ValueError(f'{flag} inconsistent with dividend amount.')
        d['entry_dividend_available_time'] = times(d.entry_dividend_available_time,'entry_dividend_available_time')
        # This ex-post amount is a label field, never a predictor. RET/RETX need the full exit session.
        if not (d.entry_dividend_available_time >= cl).all():
            raise ValueError('RET/RETX-derived entry dividends cannot be known before their session close.')
        signal_amount = d.set_index('date').cash_dividend
        shared = d.entry_session.isin(signal_amount.index)
        if not np.allclose(d.loc[shared,'entry_cash_dividend'],d.loc[shared,'entry_session'].map(signal_amount),rtol=1e-10,atol=1e-10):
            raise ValueError('Signal/entry dividend mappings disagree for the same session.')
    columns = SURFACE if metadata['surface_kind'] == 'curve' else ['vix_30','vix_3m','skew_index','vvix']
    require(d, columns)
    for name in columns + (['implied_var_30'] if 'implied_var_30' in d else []):
        d[name] = pd.to_numeric(d[name], errors='raise')
        bad = d[name].notna() & (~np.isfinite(d[name]) | (d[name] <= 0))
        if name.startswith('iv_'):
            bad |= d[name] > 3
        if bad.any():
            raise ValueError(f'Invalid {name}; IV is a decimal, VIX/VVIX are index points.')
    d['session_id'] = signal_pos.astype(int)
    return d


def features(d, kind):
    p = d.copy()
    r = np.log(p.reference_price).diff()
    p['ret_1'] = r
    p['ret_5'] = np.log(p.reference_price).diff(5)
    gk = .5*np.log(p.high/p.low)**2 - (2*np.log(2)-1)*np.log(p.close/p.open)**2
    cash = p.cash_dividend if 'cash_dividend' in p else 0.
    p['overnight_log_return_raw'] = np.log(p.open/p.close.shift(1))
    p['overnight_log_return'] = np.log((p.open+cash)/p.close.shift(1))
    p['rv_daily'] = gk.clip(lower=0) + p.overnight_log_return**2
    for h in [1,5,22]:
        p[f'rv_{h}'] = p.rv_daily.rolling(h, min_periods=h).mean()*252
        p[f'log_rv_{h}'] = np.log(p[f'rv_{h}'].clip(lower=1e-10))
    if kind == 'curve':
        p['iv_var_30'] = p['implied_var_30'] if 'implied_var_30' in p else p.iv_atm_30**2
        p['term_slope'] = p.iv_atm_91-p.iv_atm_30
        p['skew'] = p.iv_put25_30-p.iv_call25_30
        p['skew_change'] = p['skew'].diff()
        p['curvature'] = (p.iv_put25_30+p.iv_call25_30)/2-p.iv_atm_30
    else:
        p['iv_var_30'] = (p.vix_30/100)**2
        p['term_slope'] = p.vix_3m/100-p.vix_30/100
        p['skew_index_change'] = p.skew_index.diff()
        p['vvix_ratio'] = p.vvix/p.vix_30
    p['log_iv_var_30'] = np.log(p.iv_var_30.clip(lower=1e-10))
    return p


def target(p, kind, horizon):
    if horizon < 1:
        raise ValueError('Horizon must be positive.')
    if kind == 'return':
        cash = pd.Series(0.,index=p.index)
        if 'entry_cash_dividend' in p:
            future_cash = pd.concat([p.entry_cash_dividend.shift(-k) for k in range(1,horizon+1)],axis=1)
            cash = future_cash.sum(axis=1,min_count=horizon)
        y = (p.entry_price.shift(-horizon)+cash)/p.entry_price-1
        end = p.entry_time.shift(-horizon)
        if 'entry_dividend_available_time' in p:
            known = pd.concat([p.entry_dividend_available_time.shift(-k) for k in range(1,horizon+1)],axis=1).max(axis=1)
            end = pd.concat([end,known],axis=1).max(axis=1).where(y.notna())
    else:
        # Explicit shifts exclude today and preserve the original trading calendar.
        future = pd.concat([p.rv_daily.shift(-k) for k in range(1,horizon+1)], axis=1)
        y = np.log((future.sum(axis=1, min_count=horizon)/horizon*252).clip(lower=1e-10))
        end = p.rv_available_time.shift(-horizon)
    return y.rename('target'), end.rename('target_end')


def ex_dividend_window(p, horizon):
    """Retrospective sensitivity: exclude any ex-date from entry through exit, inclusive."""
    require(p,['entry_is_ex_dividend'])
    flags=pd.concat([p.entry_is_ex_dividend.shift(-k) for k in range(horizon+1)],axis=1)
    return flags.eq(1).any(axis=1)


def ridge_fit(x, y, alpha=10.):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1
    z = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    penalty = np.eye(z.shape[1])*alpha
    penalty[0,0] = 0
    beta = np.linalg.solve(z.T@z+penalty, z.T@y)
    return dict(mean=mean, scale=scale, beta=beta, residual=y-z@beta)


def ridge_predict(m, x):
    return float(np.r_[1., (x-m['mean'])/m['scale']]@m['beta'])


def walk(p, y, end, specs, min_train=504, window=756, refit=21, alpha=10., smear=False):
    union = list(dict.fromkeys(v for cols in specs.values() for v in cols))
    x = p[union].to_numpy(float)
    target_values = y.to_numpy(float)
    good = np.isfinite(x).all(axis=1) & np.isfinite(target_values) & end.notna().to_numpy()
    out = p[['date','asof_time','entry_time']].copy()
    out['target'] = target_values
    out['target_end'] = end
    for name in specs:
        out[name] = np.nan
    out['train_n'] = np.nan
    out['fit_row'] = np.nan
    out['train_end_max'] = pd.Series(pd.NaT,index=p.index,dtype='datetime64[ns, UTC]')
    fit_at, models, train_idx = -refit, None, None
    selections = {name:[union.index(col) for col in cols] for name,cols in specs.items()}
    for i in range(len(p)):
        if not np.isfinite(x[i]).all():
            continue
        if models is None or i-fit_at >= refit:
            train_idx = np.flatnonzero(good & (end < p.asof_time.iloc[i]).to_numpy())[-window:]
            if len(train_idx) < min_train:
                continue
            train_y = target_values[train_idx]
            models = {}
            for name, sel in selections.items():
                models[name] = ridge_fit(x[train_idx][:,sel], train_y, alpha) if sel else float(train_y.mean())
            fit_at = i
        for name, sel in selections.items():
            value = ridge_predict(models[name],x[i,sel]) if sel else models[name]
            if smear:
                residual = models[name]['residual'] if sel else target_values[train_idx]-models[name]
                value = float(np.exp(value)*np.exp(residual).mean())
            out.loc[i,name] = value
        out.loc[i,'train_n'] = len(train_idx)
        out.loc[i,'fit_row'] = fit_at
        out.loc[i,'train_end_max'] = end.iloc[train_idx].max()
    has = out[list(specs)[0]].notna()
    if not (out.loc[has,'train_end_max'] < out.loc[has,'asof_time']).all():
        raise AssertionError('Future label entered training.')
    return out


def add_har(p, min_train=252, window=756, refit=21, alpha=10.):
    y,end = target(p,'vol',21)
    out = walk(p,y,end,{'har_fwd_21':['log_rv_1','log_rv_5','log_rv_22']},
               min_train,window,refit,alpha,smear=True)
    p = p.copy()
    p['har_fwd_21'] = out.har_fwd_21
    p['har_train_end_max'] = out.train_end_max
    p['vrp_proxy'] = p.iv_var_30-p.har_fwd_21
    return p, out


def block_test(a, b, y, reps=1000, block=20, seed=42):
    """Paired loss difference, on the uncompressed session grid; missing pairs excluded per draw."""
    loss = (y-a)**2-(y-b)**2
    loss = np.asarray(loss,float)
    valid = np.isfinite(loss)
    if valid.sum()<20:
        return dict(n=int(valid.sum()),gain=np.nan,low=np.nan,high=np.nan,p=np.nan)
    obs = float(np.nanmean(loss))
    rng = np.random.default_rng(seed)
    n=len(loss); block=min(block,n)
    sample=[]
    for _ in range(reps):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        idx=((starts[:,None]+np.arange(block))%n).ravel()[:n]
        vals=loss[idx]
        if np.isfinite(vals).any():
            sample.append(float(np.nanmean(vals)))
    boot=np.array(sample)
    lo,hi=np.quantile(boot,[.025,.975])
    p=(1+np.sum(np.abs(boot-obs)>=abs(obs)))/(len(boot)+1)
    return dict(n=int(valid.sum()),gain=obs,low=float(lo),high=float(hi),p=float(p))


def holm(pvalues):
    pvalues=np.asarray(pvalues,float)
    result=np.full(len(pvalues),np.nan)
    ix=np.flatnonzero(np.isfinite(pvalues))
    order=ix[np.argsort(pvalues[ix])]
    prev=0.
    for rank,i in enumerate(order):
        prev=max(prev, float(pvalues[i])*(len(order)-rank))
        result[i]=min(1.,prev)
    return result


def hac_ols(y, x, names, lags=5, positions=None):
    """OLS descriptive HAC covariance on original session distances (no compressed gaps)."""
    y=np.asarray(y,float); x=np.asarray(x,float)
    good=np.isfinite(y)&np.isfinite(x).all(axis=1)
    pos=np.arange(len(y))[good] if positions is None else np.asarray(positions)[good]
    y=y[good]; x=x[good]
    if len(y)<x.shape[1]+20:
        return pd.DataFrame()
    x=np.column_stack([np.ones(len(x)),x])
    bread=np.linalg.pinv(x.T@x)
    beta=bread@x.T@y
    scores=x*(y-x@beta)[:,None]
    meat=scores.T@scores
    lookup={int(p):i for i,p in enumerate(pos)}
    for lag in range(1,lags+1):
        pairs=[(i,lookup[int(p)-lag]) for i,p in enumerate(pos) if int(p)-lag in lookup]
        if pairs:
            a,b=map(np.array,zip(*pairs))
            gamma=scores[a].T@scores[b]
            meat+=(1-lag/(lags+1))*(gamma+gamma.T)
    cov=bread@meat@bread*len(y)/(len(y)-x.shape[1])
    se=np.sqrt(np.maximum(np.diag(cov),0))
    return pd.DataFrame({'feature':['intercept']+names,'beta':beta,'hac_se':se,
                         'ci_low':beta-1.96*se,'ci_high':beta+1.96*se,'n':len(y),'lags':lags})


def clark_west(y, model, bench, lags):
    f=(y-bench)**2-((y-model)**2-(bench-model)**2)
    tab=hac_ols(f,np.empty((len(f),0)),[],lags)
    if tab.empty or tab.hac_se.iloc[0]==0:
        return np.nan
    return float(tab.beta.iloc[0]/tab.hac_se.iloc[0])


def portfolio(predictions, entry_prices, h, bps, entry_dividends=None):
    """Equal-weight h-session staggered sleeves, including launch and final liquidation."""
    n=len(entry_prices)
    if n<2:
        raise ValueError('Need two execution prices.')
    pred=np.asarray(predictions,float)
    signal=np.where(np.isfinite(pred),np.sign(pred),0.)
    # Do not initiate a trade whose full holding window is unavailable in the evaluation period.
    signal[max(0,n-h):]=0
    pos=np.zeros(n-1)
    for t in range(n-1):
        pos[t]=sum(signal[j] for j in range(max(0,t-h+1),t+1))/h
    cash=np.zeros(n) if entry_dividends is None else np.asarray(entry_dividends,float)
    if len(cash)!=n or not (np.isfinite(cash)&(cash>=0)).all():
        raise ValueError('Portfolio dividend vector must match prices, finite and nonnegative.')
    returns=(np.asarray(entry_prices[1:])+cash[1:])/np.asarray(entry_prices[:-1])-1
    turnover=np.abs(np.diff(np.r_[0.,pos]))
    turnover[-1]+=abs(pos[-1])
    net=pos*returns-turnover*bps/10000
    wealth=np.cumprod(1+net)
    peak=np.maximum.accumulate(np.r_[1.,wealth])[1:]
    vol=float(net.std(ddof=1)*np.sqrt(252)) if len(net)>1 else 0.
    summary={'total_return':float(wealth[-1]-1),'annual_vol':vol,
             'sharpe_zero_cash':float(net.mean()*252/vol) if vol else np.nan,
             'max_drawdown':float(np.min(wealth/peak-1)),
             'mean_abs_exposure':float(abs(pos).mean()),'annual_turnover':float(turnover.mean()*252)}
    return summary,pd.DataFrame({'position':pos,'return':returns,'dividend_at_interval_end':cash[1:],
                                 'turnover':turnover,'net':net,'wealth':wealth})
