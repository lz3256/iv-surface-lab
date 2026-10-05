from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from ivsurf import core
import residual_model as rm
from precision import compare


def load_panel(path):
    p=pd.read_csv(path)
    for c in ['asof_time','entry_time','entry_dividend_available_time']:
        p[c]=pd.to_datetime(p[c],utc=True)
    return p


def run(v1,v2,package):
    v1,v2,package=map(Path,(v1,v2,package));out=package/'results'
    if out.exists() and any(out.iterdir()):raise ValueError('Results already exist')
    out.mkdir(exist_ok=True)
    signal=pd.read_csv(package/'diagnostics/same_endpoint_daily.csv').set_index('date').same_skew_change
    comparisons=[];prior=[];old_ratios=[];verification=[]
    for stage,phase in [('former_development','development'),('former_holdout','final')]:
        folder=out/stage;folder.mkdir()
        p=load_panel(v2/'results'/stage/'features.csv')
        x=p[rm.CONTROLS].to_numpy(float);raw=p.skew_change.to_numpy(float)
        same=p.date.map(signal).to_numpy(float)
        for h in [1,5,21]:
            y,end=core.target(p,'return',h);yy=y.to_numpy(float)
            eligible=np.isfinite(x).all(axis=1)&np.isfinite(raw)&np.isfinite(yy)&end.notna().to_numpy()
            old=pd.read_csv(v2/'results'/stage/f'H2_h{h}_predictions.csv')
            result=old.copy().set_index('date',drop=False)
            for name in ['R3u_raw','R3u_raw_matched','R3u_same']:
                for col in [name,name+'_z',name+'_increment']:result[col]=np.nan
            fits=pd.read_csv(v2/'results'/stage/f'H2_h{h}_fit_audit.csv')
            audit=[]
            for j,row in enumerate(fits.itertuples()):
                start=int(row.fit_row);stop=int(fits.iloc[j+1].fit_row) if j+1<len(fits) else len(p)
                idx=np.flatnonzero(eligible & (end<p.asof_time.iloc[start]).to_numpy())[-1008:]
                assert len(idx)==row.train_n
                subset=idx[np.isfinite(same[idx])]
                for name,s,ix in [('R3u_raw',raw,idx),('R3u_raw_matched',raw,subset),('R3u_same',same,subset)]:
                    model=rm.fit(x[ix],s[ix],yy[ix]) if len(ix)>=504 else None
                    audit.append({'model':name,'horizon':h,'fit_row':start,'fit_date':p.date.iloc[start],
                         'asof_time':str(p.asof_time.iloc[start]),'n':len(ix),'available':model is not None,
                         'train_first_date':p.date.iloc[ix[0]] if len(ix) else None,
                         'train_last_date':p.date.iloc[ix[-1]] if len(ix) else None,
                         'train_end_max':str(end.iloc[ix].max()) if len(ix) else None,
                         'parameters':json.dumps(model),'training_indices':json.dumps(ix.tolist())})
                    if model is None:continue
                    assert end.iloc[ix].max()<p.asof_time.iloc[start]
                    for i in range(start,stop):
                        date=p.date.iloc[i]
                        if date not in result.index or not np.isfinite(result.loc[date,'R2_HAR']):continue
                        if name!='R3u_raw' and not np.isfinite(same[i]):continue
                        z,increment=rm.predict(model,x[i],s[i])
                        result.loc[date,[name,name+'_z',name+'_increment']]=[result.loc[date,'R2_HAR']+increment,z,increment]
            for name in ['R0_mean','R1_market','R2_HAR','R3_skew_change','target']:
                np.testing.assert_array_equal(result[name].to_numpy(),old[name].to_numpy())
            result.to_csv(folder/f'H2_h{h}_predictions.csv',index=False)
            pd.DataFrame(audit).to_csv(folder/f'H2_h{h}_residual_audit.csv',index=False)
            pairs=[('R2_HAR','R3u_raw'),('R2_HAR','R3u_raw_matched'),('R2_HAR','R3u_same'),('R3u_raw_matched','R3u_same')]
            for baseline,added in pairs:
                a=compare(result.target,result[baseline],result[added],max(20,h))
                if h!=1:
                    for key in list(a):
                        if 'threshold' in key:a[key]=None
                comparisons.append({'stage':stage,'horizon':h,'baseline':baseline,'added':added,**a})
            verification.append({'stage':stage,'horizon':h,'old_forecasts_and_targets_unchanged':True,
                                 'refits':len(fits),'raw_valid_fits':sum(r['available'] for r in audit if r['model']=='R3u_raw'),
                                 'same_valid_fits':sum(r['available'] for r in audit if r['model']=='R3u_same')})
            # Retain both the old CI unit conversion and a new ratio bootstrap.
            for version in ['v1','v2']:
                if version=='v1':
                    pred=pd.read_csv(v1/'reports'/phase/f'H2_h{h}_predictions.csv')
                    comp=pd.read_csv(v1/'reports'/phase/'model_comparisons.csv')
                    saved=comp.loc[(comp.family=='H2')&(comp.horizon==h)&comp.baseline.eq('R2_HAR')&comp.added.eq('R3_skew_change')].iloc[0]
                else:
                    pred=old;comp=pd.read_csv(v2/'results/comparisons.csv')
                    saved=comp.loc[(comp.stage==stage)&(comp.horizon==h)&comp.baseline.eq('R2_HAR')].iloc[0]
                mse=((pred.target-pred.R2_HAR)**2).where(pred.R3_skew_change.notna()).mean()
                prior.append({'version':version,'stage':stage,'horizon':h,'n':saved.n,'baseline_mse':mse,
                    'incremental_r2':saved.gain/mse,'low95_fixed_denominator':saved.low/mse,
                    'high95_fixed_denominator':saved.high/mse,'half_width':(saved.high-saved.low)/(2*mse),
                    'threshold':.001 if h==1 else np.nan,'upper_below_threshold':bool(saved.high/mse<.001) if h==1 else None})
                old_ratios.append({'version':version,'stage':stage,'horizon':h,**compare(pred.target,pred.R2_HAR,pred.R3_skew_change,max(20,h))})
    table=pd.DataFrame(comparisons);table['p_holm_12']=np.nan
    for _,ix in table.groupby('stage').groups.items():table.loc[ix,'p_holm_12']=core.holm(table.loc[ix,'p'])
    table.to_csv(out/'comparisons.csv',index=False)
    pd.DataFrame(prior).to_csv(out/'v1_v2_saved_interval_conversion.csv',index=False)
    pd.DataFrame(old_ratios).to_csv(out/'v1_v2_ratio_bootstrap.csv',index=False)
    (out/'run_checks.json').write_text(json.dumps(verification,indent=2))
    print(table.to_string(index=False))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--v1',required=True);a.add_argument('--v2',required=True);a.add_argument('--package',required=True)
    args=a.parse_args();run(args.v1,args.v2,args.package)
