"""Descriptive diagnostics of v1 forecast losses and expiry-calendar association."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from ivsurf import core,research


def third_friday(stamp):
    first=pd.Timestamp(stamp).replace(day=1).normalize()
    return first+pd.Timedelta(days=(4-first.weekday())%7+14)


def third_friday_offset(dates):
    dates=pd.DatetimeIndex(dates)
    candidates=pd.date_range(dates.min()-pd.Timedelta(days=50),dates.max()+pd.Timedelta(days=50),freq='MS')
    candidates=pd.DatetimeIndex([third_friday(x) for x in candidates])
    distance=(dates.to_numpy()[:,None]-candidates.to_numpy()[None,:])/np.timedelta64(1,'D')
    return distance[np.arange(len(dates)),np.abs(distance).argmin(axis=1)].astype(int)


def run(v1,out):
    v1,out=Path(v1),Path(out);out.mkdir(parents=True,exist_ok=True)
    raw=core.validate(pd.read_csv(v1/'data/daily.csv'),pd.read_csv(v1/'data/calendar.csv'),json.loads((v1/'data/metadata.json').read_text()))
    p=core.features(raw.iloc[:-252].copy(),'curve')
    saved=pd.read_csv(v1/'reports/development/features.csv').set_index('date')
    for col in ['har_fwd_21','vrp_proxy']:p[col]=p.date.map(saved[col])
    _,specs=research.specs('curve')
    union=list(dict.fromkeys(c for names in specs.values() for c in names))
    x=p[union].to_numpy(float);y,end=core.target(p,'return',1)
    good=np.isfinite(x).all(axis=1)&np.isfinite(y)&end.notna()
    pred=pd.read_csv(v1/'reports/development/H2_h1_predictions.csv')
    pred['loss_R0']=(pred.target-pred.R0_mean)**2
    pred['loss_R1']=(pred.target-pred.R1_market)**2
    pred['loss_R2']=(pred.target-pred.R2_HAR)**2
    pred['excess_R2_vs_R0']=pred.loss_R2-pred.loss_R0
    valid=pred[['loss_R0','loss_R1','loss_R2']].notna().all(axis=1)
    losses=pred.loc[valid].copy()
    losses.to_csv(out/'v1_daily_loss.csv',index=False)
    losses.nlargest(20,'excess_R2_vs_R0').to_csv(out/'largest_excess_loss_days.csv',index=False)
    losses.groupby(losses.date.str[:4]).agg(n=('date','size'),mean_mse_R0=('loss_R0','mean'),
        mean_mse_R2=('loss_R2','mean'),sum_excess_loss=('excess_R2_vs_R0','sum')).to_csv(out/'loss_by_year.csv')
    crash=losses.loc[losses.date.between('2020-03-01','2020-03-31')]
    selectors=[union.index(c) for c in specs['R2_HAR']]
    zrows=[];shrinkrows=[];fitcache={}
    by_date={date:i for i,date in enumerate(p.date)}
    for row in crash.itertuples():
        fit=int(row.fit_row);i=by_date[row.date]
        if fit not in fitcache:
            train=np.flatnonzero(good.to_numpy()&(end<p.asof_time.iloc[fit]).to_numpy())[-1008:]
            model=core.ridge_fit(x[train][:,selectors],y.iloc[train].to_numpy(),10.)
            fitcache[fit]=(train,model)
            standardized=(x[train][:,selectors]-model['mean'])/model['scale']
            eig=np.linalg.eigvalsh(standardized.T@standardized)
            shrinkrows.append({'fit_date':p.date.iloc[fit],'n':len(train),'eigen_min':eig.min(),'eigen_max':eig.max(),
                               'min_direction_retention':(eig/(eig+10)).min(),'max_direction_retention':(eig/(eig+10)).max()})
        train,model=fitcache[fit]
        actual=core.ridge_predict(model,x[i,selectors])
        assert abs(actual-row.R2_HAR)<1e-10
        for j,col in enumerate(specs['R2_HAR']):
            z=(x[i,selectors[j]]-model['mean'][j])/model['scale'][j]
            logged_z=np.nan
            if col in ['rv_22','iv_var_30','har_fwd_21']:
                logs=np.log(np.maximum(x[train,selectors[j]],1e-10))
                logged_z=(np.log(max(x[i,selectors[j]],1e-10))-logs.mean())/logs.std()
            zrows.append({'date':row.date,'fit_date':p.date.iloc[fit],'feature':col,
                          'raw_value':x[i,selectors[j]],'training_mean':model['mean'][j],
                          'training_std':model['scale'][j],'z_score':z,'log_feature_z_same_training_rows':logged_z,
                          'contribution_to_prediction':z*model['beta'][j+1],'prediction_R2':row.R2_HAR})
    z=pd.DataFrame(zrows);z.to_csv(out/'march2020_training_window_zscores.csv',index=False)
    pd.DataFrame(shrinkrows).to_csv(out/'ridge_direction_shrinkage.csv',index=False)

    # Calendar association only; no identification of AM/PM contracts from date alone.
    f=core.features(raw,'curve')
    f['offset_third_friday']=third_friday_offset(pd.to_datetime(f.date))
    f['target30_offset_third_friday']=third_friday_offset(pd.to_datetime(f.date)+pd.Timedelta(days=30))
    f['period']=np.select([f.date<'2013-01-01',f.date<'2020-01-01'],['2010-2012','2013-2019'],default='2020-2023')
    f['month']=f.date.str[:7]
    expiry=pd.read_csv(v1/'data/expiry_nodes.csv')
    e=pd.to_datetime(expiry.expiry)
    friday=e.map(third_friday)
    expiry['monthly_calendar_proxy']=(e==friday)|(e==friday-pd.Timedelta(days=1))
    endpoint_lookup=expiry.groupby(['date','tenor_days']).monthly_calendar_proxy.any().to_dict()
    interp=pd.read_csv(v1/'data/interpolation_audit.csv')
    for node in ['iv_put25_30','iv_call25_30','iv_atm_30']:
        a=interp.loc[interp.node.eq(node)].set_index('date')
        proxy={date:bool(endpoint_lookup.get((date,r.lower_days),False) or endpoint_lookup.get((date,r.upper_days),False))
               if r.available else np.nan for date,r in a.iterrows()}
        f[node+'_monthly_endpoint_proxy']=f.date.map(proxy)
        lower=f.date.map(a.lower_days);upper=f.date.map(a.upper_days)
        # Convert DTE to an approximate absolute calendar ordinal for detecting bracket switches.
        ordinals=(pd.to_datetime(f.date)-pd.Timestamp('1970-01-01')).dt.days
        lo=np.round(ordinals+lower,1);hi=np.round(ordinals+upper,1)
        complete=lo.notna()&hi.notna()&lo.shift().notna()&hi.shift().notna()
        f[node+'_bracket_switch']=((lo!=lo.shift())|(hi!=hi.shift())).where(complete)
    f['skew_change_abs']=f.skew_change.abs()
    tables=[]
    for offset in ['offset_third_friday','target30_offset_third_friday']:
        for period,g in [('all',f)]+list(f.groupby('period')):
            table=g.groupby(offset).agg(n=('skew_change','count'),months=('month','nunique'),
                mean_skew_change=('skew_change','mean'),variance_skew_change=('skew_change','var'),
                median_absolute_change=('skew_change_abs','median'))
            table=table.reset_index().rename(columns={offset:'offset_days'})
            table['alignment']=offset;table['period']=period;tables.append(table)
    pd.concat(tables,ignore_index=True).to_csv(out/'third_friday_groups.csv',index=False)
    group=[]
    for node in ['iv_put25_30','iv_call25_30','iv_atm_30']:
        for suffix in ['monthly_endpoint_proxy','bracket_switch']:
            col=node+'_'+suffix
            a=f.groupby(col).agg(n=('skew_change','count'),mean_skew_change=('skew_change','mean'),
                variance_skew_change=('skew_change','var'),median_absolute_change=('skew_change_abs','median')).reset_index().rename(columns={col:'flag'})
            a['classification']=col;group.append(a)
    pd.concat(group,ignore_index=True).to_csv(out/'expiry_proxy_groups.csv',index=False)
    keep=['date','period','offset_third_friday','target30_offset_third_friday','skew','skew_change']+[c for c in f if c.endswith(('_monthly_endpoint_proxy','_bracket_switch'))]
    f[keep].to_csv(out/'expiry_daily_diagnostics.csv',index=False)
    exclusions=pd.read_csv(v1/'data/curve_exclusions.csv')
    exclusions=exclusions.merge(f[['date','offset_third_friday','target30_offset_third_friday']],on='date',how='left')
    exclusions.to_csv(out/'may2019_exclusion_timing.csv',index=False)
    summary={'status':'descriptive, after observing v1 outcomes; no AM/PM causality identification',
        'development_R2_mse_increase_vs_mean_pct':100*(losses.loss_R2.mean()/losses.loss_R0.mean()-1),
        'march2020_days':len(crash),'march2020_fraction_of_R2_excess_loss':crash.excess_R2_vs_R0.sum()/losses.excess_R2_vs_R0.sum(),
        'max_abs_R2_prediction_march2020':crash.R2_HAR.abs().max(),
        'march2020_max_abs_training_z':z.groupby('feature').z_score.apply(lambda a:a.abs().max()).to_dict(),
        'same_rows_max_abs_log_z':z.groupby('feature').log_feature_z_same_training_rows.apply(lambda a:float(a.abs().max()) if a.notna().any() else None).to_dict(),
        'ridge_direction_retention':shrinkrows,
        'caveats':['Standardization already exists in v1; raw units alone do not establish the mechanism.',
                   'Ridge alpha=10 has direction-specific shrinkage, not a universal equivalence to OLS.',
                   'Month and third-Friday associations may include macro events, flows and changing listings.',
                   'Endpoint calendar proxy can contain AM and PM contracts and cannot be used as a verified contract selector.',
                   'Six extreme May2019 IV gaps do not establish an AM/PM cause.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--v1',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();run(a.v1,a.out)
