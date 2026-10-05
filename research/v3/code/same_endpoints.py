"""Hold yesterday's expiry identifiers fixed; never extrapolate a 30d node."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd


def pair_value(rows, pair, source, target=30.):
    if pair is None:return np.nan,'no_previous_pair'
    if any(e not in rows for e in pair):return np.nan,'missing_expiry'
    points=[(float(rows[e]['tenor_days']),float(rows[e][source])) for e in dict.fromkeys(pair)]
    if not np.isfinite(points).all():return np.nan,'missing_node'
    points=sorted(points)
    if points[0][0]>target+1e-9 or points[-1][0]<target-1e-9:
        return np.nan,'no_longer_brackets_30'
    if len(points)==1:
        return (points[0][1],'ok') if abs(points[0][0]-target)<1e-9 else (np.nan,'single_endpoint')
    (lo,a),(hi,b)=points
    if hi<=lo:return np.nan,'duplicate_tenor'
    weight=(target-lo)/(hi-lo)
    return np.sqrt(((1-weight)*a*a*lo+weight*b*b*hi)/target),'ok'


def third_friday(d):
    d=pd.Timestamp(d).replace(day=1).normalize()
    return d+pd.Timedelta(days=(4-d.weekday())%7+14)


def monthly_proxy(expiry):
    d=pd.Timestamp(expiry);f=third_friday(d)
    return d==f or d==f-pd.Timedelta(days=1)


def offsets(dates):
    d=pd.DatetimeIndex(dates)
    candidates=pd.date_range(d.min()-pd.Timedelta(days=50),d.max()+pd.Timedelta(days=50),freq='MS')
    candidates=pd.DatetimeIndex([third_friday(x) for x in candidates])
    delta=(d.to_numpy()[:,None]-candidates.to_numpy()[None,:])/np.timedelta64(1,'D')
    return delta[np.arange(len(d)),np.abs(delta).argmin(axis=1)].astype(int)


def build(v1,out):
    v1,out=Path(v1),Path(out);out.mkdir(parents=True,exist_ok=True)
    daily=pd.read_csv(v1/'data/daily.csv')
    expiry=pd.read_csv(v1/'data/expiry_nodes.csv')
    interp=pd.read_csv(v1/'data/interpolation_audit.csv')
    by_date={d:g.set_index('expiry').to_dict('index') for d,g in expiry.groupby('date')}
    audit=interp.set_index(['date','node'])
    legs={'put25':'iv_put25_30','call25':'iv_call25_30'}
    def normal_pair(date,source):
        if (date,legs[source]) not in audit.index:return None
        row=audit.loc[(date,legs[source])]
        if not row.available:return None
        rows=by_date.get(date,{})
        result=[]
        for tenor in [row.lower_days,row.upper_days]:
            matches=[e for e,r in rows.items() if np.isclose(r['tenor_days'],tenor,rtol=0,atol=1e-8) and np.isfinite(r[source])]
            if len(matches)!=1:return None
            result.append(matches[0])
        return tuple(result)
    valid=daily[['iv_atm_30','iv_atm_91','iv_put25_30','iv_call25_30']].notna().all(axis=1)
    skew=daily.iv_put25_30-daily.iv_call25_30
    records=[];leg_records=[]
    for i,date in enumerate(daily.date):
        row={'date':date,'previous_date':daily.date.iloc[i-1] if i else None,
             'raw_skew_change':skew.diff().iloc[i],'same_skew_change':np.nan,
             'switch_category':'unavailable','reason':'missing_or_excluded_session'}
        if i and valid.iloc[i] and valid.iloc[i-1]:
            prior=daily.date.iloc[i-1];prevpairs={s:normal_pair(prior,s) for s in legs}
            curpairs={s:normal_pair(date,s) for s in legs}
            if all(prevpairs.values()) and all(curpairs.values()):
                changed=any(prevpairs[s]!=curpairs[s] for s in legs)
                involved={e for pairs in [prevpairs,curpairs] for pair in pairs.values() for e in pair}
                row['switch_category']=('monthly_proxy_switch' if any(monthly_proxy(e) for e in involved)
                                         else 'other_switch') if changed else 'no_switch'
            values={};reasons=[]
            for source,pair in prevpairs.items():
                yesterday,reason0=pair_value(by_date.get(prior,{}),pair,source)
                today,reason1=pair_value(by_date.get(date,{}),pair,source)
                values[source]=(yesterday,today)
                reason=reason0 if reason0!='ok' else reason1
                reasons.append(reason)
                leg_records.append({'date':date,'previous_date':prior,'leg':source,
                    'lower_expiry':pair[0] if pair else None,'upper_expiry':pair[1] if pair else None,
                    'previous_node':yesterday,'current_fixed_node':today,'reason':reason})
            if all(r=='ok' for r in reasons):
                prior_fixed=values['put25'][0]-values['call25'][0]
                assert np.isclose(prior_fixed,skew.iloc[i-1],atol=1e-10,rtol=0)
                row['same_skew_change']=(values['put25'][1]-values['call25'][1])-prior_fixed
                row['reason']='ok'
                if row['switch_category']=='no_switch':
                    assert np.isclose(row['same_skew_change'],row['raw_skew_change'],atol=1e-10,rtol=0)
            else:row['reason']=';'.join(sorted(set(r for r in reasons if r!='ok')))
        records.append(row)
    f=pd.DataFrame(records)
    f['switch_component']=f.raw_skew_change-f.same_skew_change
    f['offset_third_friday']=offsets(pd.to_datetime(f.date))
    f['target30_offset_third_friday']=offsets(pd.to_datetime(f.date)+pd.Timedelta(days=30))
    f['period']=np.select([f.date<'2013-01-01',f.date<'2020-01-01'],['2010-2012','2013-2019'],default='2020-2023')
    f.to_csv(out/'same_endpoint_daily.csv',index=False)
    pd.DataFrame(leg_records).to_csv(out/'same_endpoint_legs.csv',index=False)
    tables=[]
    for period,g in [('all',f)]+list(f.groupby('period')):
        for col in ['offset_third_friday','target30_offset_third_friday','switch_category']:
            for sample,sub,signal in [('raw_all',g,'raw_skew_change'),
                 ('raw_matched',g.loc[g.same_skew_change.notna()],'raw_skew_change'),
                 ('same_matched',g.loc[g.same_skew_change.notna()],'same_skew_change')]:
                a=sub.groupby(col)[signal].agg(['count','mean','var']).reset_index().rename(columns={col:'group'})
                a['alignment']=col;a['period']=period;a['sample']=sample;tables.append(a)
    pd.concat(tables,ignore_index=True).to_csv(out/'same_endpoint_groups.csv',index=False)
    summary={'rows':len(f),'raw_available':int(f.raw_skew_change.notna().sum()),
        'same_available':int(f.same_skew_change.notna().sum()),'reasons':f.reason.value_counts().to_dict(),
        'category_coverage':f.groupby('switch_category').agg(raw_n=('raw_skew_change','count'),same_n=('same_skew_change','count')).to_dict('index'),
        'non_switch_identity_checked':True,'extrapolation_used':False,'root_or_settlement_identified':False}
    (out/'endpoint_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--v1',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();build(a.v1,a.out)
