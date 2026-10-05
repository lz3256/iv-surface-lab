from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from ivsurf import core
from run_daily import load_panel
from precision import compare


def containing_window(entry,exit_time,release):
    ix=np.flatnonzero(((entry<release)&(exit_time>release)).to_numpy())
    if len(ix)>1:raise ValueError('Overlapping holding windows')
    return int(ix[0]) if len(ix) else None


def run(v2,package):
    v2,package=Path(v2),Path(package);out=package/'events';out.mkdir(exist_ok=True)
    p=load_panel(v2/'results/former_holdout/features.csv')
    y,end=core.target(p,'return',1);exit_time=p.entry_time.shift(-1)
    forecasts=pd.concat([pd.read_csv(package/'results'/stage/'H2_h1_predictions.csv')
                         for stage in ['former_development','former_holdout']],ignore_index=True).set_index('date')
    assert not forecasts.index.duplicated().any()
    calendar=pd.read_csv(package/'inputs/events.csv');records=[]
    for e in calendar.itertuples():
        row={'event':e.event,'event_date':e.date,'release_time':e.release_time,'source':e.source,
             'time_basis':e.time_basis,'status':'unverified_time'}
        if e.usable:
            release=pd.Timestamp(e.release_time);i=containing_window(p.entry_time,exit_time,release)
            row['status']='outside_available_holding_windows'
            if i is not None:
                date=p.date.iloc[i]
                row.update({'signal_date':date,'asof_time':str(p.asof_time.iloc[i]),
                    'entry_time':str(p.entry_time.iloc[i]),'exit_time':str(exit_time.iloc[i]),
                    'target':y.iloc[i],'target_end':str(end.iloc[i]),'z':np.nan,'R2':np.nan,
                    'status':'missing_residual_or_control_forecast'})
                if date in forecasts.index:
                    f=forecasts.loc[date];row['z']=f.R3u_raw_z;row['R2']=f.R2_HAR
                    if np.isfinite(row['z']) and np.isfinite(row['R2']):row['status']='available'
                assert p.asof_time.iloc[i]<p.entry_time.iloc[i]<release<exit_time.iloc[i]
        records.append(row)
    mapped=pd.DataFrame(records).sort_values(['event','event_date']).reset_index(drop=True)
    mapped.to_csv(out/'event_window_mapping.csv',index=False)
    metrics=[];audits=[];predictions=[]
    for kind,g in mapped.groupby('event'):
        g=g.reset_index(drop=True).copy()
        for c in ['baseline','added','n_train','beta','train_mean_z','train_mean_error']:g[c]=np.nan
        g['train_end_max']=''
        times=pd.to_datetime(g.target_end,utc=True,errors='coerce')
        for i,row in g.iterrows():
            if row.status!='available':continue
            good=np.isfinite(g.z)&np.isfinite(g.target)&np.isfinite(g.R2)&(times<pd.Timestamp(row.asof_time))
            ix=np.flatnonzero(good.to_numpy())[-80:]
            if len(ix)<24:continue
            train=g.iloc[ix];z=train.z.to_numpy();error=(train.target-train.R2).to_numpy()
            centered=z-z.mean();denom=centered@centered
            if denom/len(z)<=1e-16:continue
            beta=float(centered@(error-error.mean())/denom)
            baseline=row.R2+error.mean();added=baseline+beta*(row.z-z.mean())
            g.loc[i,['baseline','added','n_train','beta','train_mean_z','train_mean_error']]=[
                baseline,added,len(ix),beta,z.mean(),error.mean()]
            g.loc[i,'train_end_max']=str(times.iloc[ix].max())
            assert times.iloc[ix].max()<pd.Timestamp(row.asof_time)
            audits.append({'event':kind,'event_date':row.event_date,'asof_time':row.asof_time,
                'training_event_dates':json.dumps(train.event_date.tolist()),'n':len(ix),
                'train_end_max':str(times.iloc[ix].max()),'beta':beta,'mean_z':float(z.mean()),
                'mean_error':float(error.mean()),'residual_variance':float(denom/len(z))})
        stats=compare(g.target,g.baseline,g.added,block=4,family=2)
        for key in list(stats):
            if 'threshold' in key:stats[key]=None
        good=g.added.notna()&g.target.notna()
        metrics.append({'event':kind,'first_scored_event':g.loc[good,'event_date'].min(),
                        'last_scored_event':g.loc[good,'event_date'].max(),**stats})
        predictions.append(g)
    result=pd.DataFrame(metrics);result['p_holm_2']=core.holm(result.p)
    result.to_csv(out/'event_comparisons.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/'event_predictions.csv',index=False)
    pd.DataFrame(audits).to_csv(out/'event_fit_audit.csv',index=False)
    print(result.to_string(index=False))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--v2',required=True);a.add_argument('--package',required=True)
    args=a.parse_args();run(args.v2,args.package)
