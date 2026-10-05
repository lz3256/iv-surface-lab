"""Audit stored training membership and compare partial slopes to joint OLS."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from ivsurf import core
from residual_model import CONTROLS
from run_daily import load_panel


def run(v2,package):
    v2,package=Path(v2).resolve(),Path(package).resolve()
    freeze=json.loads((package/'protocol_freeze.json').read_text())
    for name,sha in freeze['files'].items():assert hashlib.sha256((package/name).read_bytes()).hexdigest()==sha,name
    # External original paths are available in the working session, optional for portable archives.
    checked_prior=0
    for name,sha in freeze['prior_file_hashes'].items():
        f=package.parents[1]/name
        if f.exists():assert hashlib.sha256(f.read_bytes()).hexdigest()==sha,name;checked_prior+=1
    same=pd.read_csv(package/'diagnostics/same_endpoint_daily.csv').set_index('date').same_skew_change
    records=[];total_fits=0;valid_fits=0;prediction_count=0
    for stage in ['former_development','former_holdout']:
        p=load_panel(v2/'results'/stage/'features.csv');x=p[CONTROLS].to_numpy()
        raw=p.skew_change.to_numpy();adjusted=p.date.map(same).to_numpy();date_to_i={date:i for i,date in enumerate(p.date)}
        for h in [1,5,21]:
            y,end=core.target(p,'return',h)
            eligible=np.isfinite(x).all(axis=1)&np.isfinite(raw)&y.notna().to_numpy()&end.notna().to_numpy()
            pred=pd.read_csv(package/'results'/stage/f'H2_h{h}_predictions.csv')
            old=pd.read_csv(v2/'results'/stage/f'H2_h{h}_predictions.csv')
            for col in ['date','target','R0_mean','R1_market','R2_HAR','R3_skew_change']:
                if col=='date':assert pred[col].equals(old[col])
                else:np.testing.assert_allclose(pred[col],old[col],equal_nan=True,atol=1e-14)
            fits=pd.read_csv(package/'results'/stage/f'H2_h{h}_residual_audit.csv');lookup={}
            for row in fits.itertuples():
                total_fits+=1
                ix=np.array(json.loads(row.training_indices),int)
                expected=np.flatnonzero(eligible&(end<p.asof_time.iloc[row.fit_row]).to_numpy())[-1008:]
                s=raw if row.model!='R3u_same' else adjusted
                if row.model!='R3u_raw':expected=expected[np.isfinite(adjusted[expected])]
                np.testing.assert_array_equal(ix,expected);assert len(ix)==row.n
                m=None if pd.isna(row.parameters) else json.loads(row.parameters)
                lookup[(row.model,row.fit_row)]=m
                if m is None:assert len(ix)<504 or np.var(s[ix])<=1e-16;continue
                valid_fits+=1
                assert end.iloc[ix].max()<pd.Timestamp(row.asof_time)
                np.testing.assert_allclose(np.quantile(s[ix],[.01,.99]),[m['lower'],m['upper']],atol=1e-14)
                design=np.column_stack([np.ones(len(ix)),(x[ix]-np.array(m['mean']))/np.array(m['scale'])])
                clipped=np.clip(s[ix],m['lower'],m['upper'])
                joint=np.linalg.lstsq(np.column_stack([design,clipped]),y.iloc[ix],rcond=None)[0]
                assert np.isclose(joint[-1],m['beta'],rtol=1e-6,atol=1e-9)
                z=clipped-design@np.array(m['projection'])
                assert np.max(np.abs(design.T@z))<1e-7
            for name,s in [('R3u_raw',raw),('R3u_raw_matched',raw),('R3u_same',adjusted)]:
                for row in pred.loc[pred[name].notna()].itertuples():
                    i=date_to_i[row.date];m=lookup[(name,int(row.fit_row))]
                    design=np.r_[1.,(x[i]-np.array(m['mean']))/np.array(m['scale'])]
                    z=np.clip(s[i],m['lower'],m['upper'])-design@np.array(m['projection'])
                    assert np.isclose(getattr(row,name),row.R2_HAR+m['beta']*z,rtol=1e-10,atol=1e-13)
                    prediction_count+=1
            records.append({'stage':stage,'horizon':h,'old_forecasts_unchanged':True,'partial_OLS_and_forecasts_verified':True})
    ep=pd.read_csv(package/'events/event_predictions.csv')
    ea=pd.read_csv(package/'events/event_fit_audit.csv')
    for row in ea.itertuples():
        dates=json.loads(row.training_event_dates)
        g=ep.loc[(ep.event==row.event)&ep.event_date.isin(dates)]
        assert len(g)==row.n and 24<=len(g)<=80
        assert (pd.to_datetime(g.target_end,utc=True)<pd.Timestamp(row.asof_time)).all()
        eligible=ep.loc[(ep.event==row.event)&ep[['target','z','R2']].notna().all(axis=1)&
                        (pd.to_datetime(ep.target_end,utc=True,errors='coerce')<pd.Timestamp(row.asof_time))].tail(80)
        assert eligible.event_date.tolist()==dates
        design=np.column_stack([np.ones(len(g)),g.z]);beta=np.linalg.lstsq(design,g.target-g.R2,rcond=None)[0]
        assert np.isclose(beta[1],row.beta,rtol=1e-9,atol=1e-12)
        current=ep.loc[(ep.event==row.event)&ep.event_date.eq(row.event_date)].iloc[0]
        assert pd.Timestamp(current.asof_time)<pd.Timestamp(current.entry_time)<pd.Timestamp(current.release_time)<pd.Timestamp(current.exit_time)
        assert np.isclose(current.added,current.R2+beta[0]+beta[1]*current.z,rtol=1e-9,atol=1e-13)
    result={'status':'PASS','frozen_v3_files_unchanged':True,'prior_files_checked':checked_prior,
        'daily_fit_records':total_fits,'daily_identified_fits':valid_fits,'daily_forecast_checks':prediction_count,
        'event_fit_records':len(ea),'daily_checks':records,
        'limitations':'Exploratory bootstrap, historical calendar availability assumed, no settlement root identification.'}
    (package/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--v2',required=True);a.add_argument('--package',required=True)
    args=a.parse_args();run(args.v2,args.package)
