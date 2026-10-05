"""Exploratory gamma analysis: explicit signed positions or an explicitly named OI assumption."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from . import core
from .data import read,save_json


def gamma(spot,strike,years,iv,rate,dividend):
    d1=(np.log(spot/strike)+(rate-dividend+.5*iv**2)*years)/(iv*np.sqrt(years))
    return np.exp(-dividend*years)*np.exp(-.5*d1**2)/np.sqrt(2*np.pi)/(spot*iv*np.sqrt(years))


def exposure(day,mode,grid_width=.15,grid_n=201):
    s=float(day.spot.iloc[0])
    t=((day.expiry_time-day.snapshot_time).dt.total_seconds()/86400/365).to_numpy()[:,None]
    k=day.strike.to_numpy()[:,None]; iv=day.iv.to_numpy()[:,None]
    r=day.rate.to_numpy()[:,None]; q=day.dividend_yield.to_numpy()[:,None]
    if mode=='dealer':
        pos=day.dealer_net_contracts.to_numpy()[:,None]
    else:
        pos=(np.where(day.option_type=='C',1.,-1.)*day.open_interest.to_numpy())[:,None]
    multiplier=day.multiplier.to_numpy()[:,None]
    grid=s*(1+np.linspace(-grid_width,grid_width,grid_n))
    totals=(gamma(grid[None,:],k,t,iv,r,q)*pos*multiplier*grid[None,:]**2*.01).sum(axis=0)
    spot_gex=float((gamma(s,k,t,iv,r,q)*pos*multiplier*s*s*.01).sum())
    roots=[]
    for i in range(len(grid)-1):
        a,b=totals[i:i+2]
        if np.isfinite(a+b) and a*b<0:
            roots.append(float(grid[i]-a*(grid[i+1]-grid[i])/(b-a)))
    for i in range(1,len(grid)-1):
        if totals[i]==0 and totals[i-1]*totals[i+1]<0:
            roots.append(float(grid[i]))
    root=min(roots,key=lambda x:abs(x-s)) if roots else np.nan
    return {'gex':spot_gex,'gamma_flip':root,'flip_count':len(roots),
            'distance_to_flip':s/root-1 if np.isfinite(root) else np.nan,'contracts':len(day)}


def run(args):
    if not 0<args.grid_width<1 or args.grid_n<5:
        raise ValueError('Require 0<grid_width<1 and grid_n>=5.')
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty output directory.')
    meta=json.loads(Path(args.metadata).read_text())
    daily=core.validate(read(args.daily),read(args.calendar),meta)
    if args.holdout<1 or len(daily)<=args.holdout:
        raise ValueError('Need a valid reserved holdout.')
    cutoff=daily.asof_time.iloc[-args.holdout]
    panel=core.features(daily,meta['surface_kind'])
    y,end=core.target(panel,'return',1)
    panel['target']=y; panel['target_end']=end
    panel=panel.loc[end<cutoff].copy() # exploratory module never inspects reserved outcomes
    d=read(args.chain)
    required=['date','snapshot_time','available_time','expiry_time','spot','strike','option_type','iv','multiplier','rate','dividend_yield']
    required+=['dealer_net_contracts'] if args.position_mode=='dealer' else ['open_interest']
    core.require(d,required)
    d['date']=core.dates(d.date)
    for col in ['snapshot_time','available_time','expiry_time']:
        d[col]=core.times(d[col],col)
    for col in ['spot','strike','iv','multiplier','rate','dividend_yield']+required[-1:]:
        d[col]=pd.to_numeric(d[col],errors='raise')
        if not np.isfinite(d[col]).all():
            raise ValueError(f'Missing/nonfinite chain {col}')
    if not (d[['spot','strike','iv','multiplier']]>0).all().all():
        raise ValueError('Chain spot, strike, IV and multiplier must be positive.')
    if not d.option_type.isin(['C','P']).all():
        raise ValueError('option_type must be C/P.')
    if args.position_mode=='heuristic' and (d.open_interest<0).any():
        raise ValueError('OI cannot be negative.')
    if not (d.available_time>=d.snapshot_time).all():
        raise ValueError('Chain availability cannot precede its snapshot.')
    d=d.loc[d.expiry_time>d.snapshot_time].copy() # no arbitrary maturity floor for 0DTE
    state=panel.set_index('date')
    output=[]
    for date,part in d.groupby('date'):
        if date not in state.index:
            continue
        if not (part.available_time<=state.loc[date,'asof_time']).all():
            raise ValueError(f'{date}: chain unavailable at signal time.')
        if part.spot.nunique()!=1 or part.snapshot_time.nunique()!=1:
            raise ValueError('Each date must contain one synchronized chain snapshot and spot.')
        output.append({'date':date,**exposure(part,args.position_mode,args.grid_width,args.grid_n)})
    if not output:
        raise ValueError('No usable chain dates in the development sample.')
    result=pd.DataFrame(output)
    joined=panel.merge(result,on='date',how='inner')
    joined['negative_gamma']=(joined.gex<0).astype(float)
    joined['interaction']=joined.ret_1*joined.negative_gamma
    regress=pd.DataFrame()
    if joined.negative_gamma.nunique()==2 and len(joined)>30:
        cols=['ret_1','negative_gamma','interaction']
        regress=core.hac_ols(joined.target,joined[cols].to_numpy(),cols,5,positions=joined.session_id)
    warnings=['EXPLORATORY only: a position assumption cannot identify true dealer holdings.',
              'IV held fixed at each strike when spot changes; grid roots depend on this assumption and grid range.',
              'Above the nearest flip does not imply positive gamma; multiple or reversed crossings may exist.',
              'Sign-regime regression is descriptive development-sample analysis; no causal identification or tradable claim.']
    if regress.empty:
        warnings.append('Insufficient observations or only one gamma-sign regime: interaction regression not estimated.')
    out.mkdir(parents=True,exist_ok=True)
    joined.to_csv(out/'gex_daily.csv',index=False)
    regress.to_csv(out/'gex_interaction_descriptive.csv',index=False)
    save_json(out/'summary.json',{'position_mode':args.position_mode,'synthetic':meta['synthetic'],
                               'reserved_start':daily.date.iloc[-args.holdout],'n':len(joined),'warnings':warnings})
    print(f'Exploratory GEX saved to {out}. mode={args.position_mode}; rows={len(joined)}')
