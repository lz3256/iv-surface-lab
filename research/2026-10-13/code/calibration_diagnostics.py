"""Exploratory Mincer-Zarnowitz diagnostics on frozen existing forecasts."""
from pathlib import Path
import argparse, datetime, hashlib, json, math, platform
import numpy as np
import pandas as pd
import original_core as core

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'iv_followup_20261013'
MODELS={'V1_HAR':'HAR','V2_IV':'HAR + ATM IV','V3_surface':'HAR + ATM IV + shape'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def mz_fit(actual,forecast,positions,lags):
    y,f,pos=np.asarray(actual,float),np.asarray(forecast,float),np.asarray(positions,int)
    good=np.isfinite(y)&np.isfinite(f)
    y,f,pos=y[good],f[good],pos[good]
    n=len(y)
    if n<30:raise ValueError('At least 30 paired observations required.')
    if len(np.unique(pos))!=n:raise ValueError('Duplicate calendar positions.')
    scale=float(f.std());center=float(f.mean())
    if scale<=0:raise ValueError('Constant forecast does not identify a slope.')
    x=np.column_stack([np.ones(n),(f-center)/scale])
    theta=np.linalg.lstsq(x,y,rcond=None)[0]
    residual=y-x@theta
    score=x*residual[:,None]
    meat=score.T@score
    lookup={int(p):i for i,p in enumerate(pos)}
    for lag in range(1,lags+1):
        pairs=[(i,lookup[int(p)-lag]) for i,p in enumerate(pos) if int(p)-lag in lookup]
        if pairs:
            i,j=map(np.array,zip(*pairs));gamma=score[i].T@score[j]
            meat+=(1-lag/(lags+1))*(gamma+gamma.T)
    bread=np.linalg.inv(x.T@x)
    cov_theta=bread@meat@bread*n/(n-2)
    transform=np.array([[1.,-center/scale],[0.,1/scale]])
    coefficients=transform@theta
    cov=transform@cov_theta@transform.T
    cov=(cov+cov.T)/2
    alpha,beta=coefficients
    se=np.sqrt(np.maximum(np.diag(cov),0))
    delta=coefficients-np.array([0.,1.])
    eigenvalues=np.linalg.eigvalsh(cov)
    if np.all(eigenvalues>0):
        wald=float(delta@np.linalg.solve(cov,delta));p=math.exp(-max(wald,0)/2)
    else:wald=p=np.nan
    # Exact finite-sample OLS MSE identity, using divisor n throughout.
    mse=float(np.mean((y-f)**2))
    bias=float(f.mean()-y.mean())
    level_component=bias**2
    slope_component=float((beta-1)**2*np.var(f))
    residual_component=float(np.mean(residual**2))
    np.testing.assert_allclose(mse,level_component+slope_component+residual_component,rtol=1e-10,atol=1e-14)
    return {'n':n,'actual_mean':float(y.mean()),'forecast_mean':center,
            'forecast_actual_mean_ratio':center/float(y.mean()),'mean_bias':bias,
            'relative_mean_bias_pct':100*bias/float(y.mean()),
            'alpha':float(alpha),'alpha_hac_se':float(se[0]),'alpha_low':float(alpha-1.96*se[0]),'alpha_high':float(alpha+1.96*se[0]),
            'beta':float(beta),'beta_hac_se':float(se[1]),'beta_low':float(beta-1.96*se[1]),'beta_high':float(beta+1.96*se[1]),
            'mz_r2':float(1-np.var(residual)/np.var(y)),
            'wald_2df':wald,'p_joint_hac':p,'cov_min_eigenvalue':float(eigenvalues.min()),
            'variance_level_mse':mse,'mse_mean_bias_component':level_component,
            'mse_slope_component':slope_component,'mse_residual_component':residual_component,
            'mean_bias_share_pct':100*level_component/mse,'slope_share_pct':100*slope_component/mse,
            'residual_share_pct':100*residual_component/mse,
            'mse_reconciliation_error':mse-level_component-slope_component-residual_component}

def freeze():
    inputs=[SOURCE/f'{phase}_h{h}_evaluation.csv' for phase in ['development','final'] for h in [5,21]]
    inputs+=[SOURCE/'protocol.json',SOURCE/'run_complete.json']
    files=inputs+sorted((ROOT/'code').glob('*.py'))
    protocol={'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status':'exploratory diagnostic on previously viewed 2010-2023 forecasts; no independent confirmation',
              'horizons':[5,21],'phases':['development','final'],'models':MODELS,
              'conversions':['direct exp of log forecast','training-only residual smearing from previous extension'],
              'sample':'same complete paired rows as original H1 comparisons; original calendar row positions preserved',
              'regression':'variance proxy = alpha + beta * predicted variance + error',
              'null':'joint alpha=0, beta=1; not a test of incremental model superiority',
              'inference':'Bartlett Newey-West HAC, main lag21, sensitivity lag42; finite sample n/(n-2); asymptotic Wald chi-square df2',
              'multiple_testing':'Holm across 2 horizons x 3 models x 2 conversions = 12 joint tests within each phase and HAC lag; marginal coefficient intervals unadjusted',
              'mse_decomposition':'mean bias squared + (beta-1)^2 Var(forecast) + mean squared OLS residual; descriptive, not causal',
              'no_model_updates':True,'no_recalibration_applied':True,
              'interpretation':['MZ slope is forecast calibration, not trading alpha or OOS forecasting R-squared',
                                'large generated-forecast uncertainty and overlapping targets can limit asymptotic inference',
                                'targets are OHLC/overnight variance proxies, not latent true conditional variance',
                                'transform methods both reported; neither selected using these diagnostics',
                                'no further VIX subdivision or parameter search in this extension'],
              'files':{str(p):sha(p) for p in files},
              'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}}
    with (ROOT/'protocol.json').open('x') as f:json.dump(protocol,f,indent=2,ensure_ascii=False)
    print('Calibration protocol frozen.')

def run():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    assert all(sha(p)==v for p,v in protocol['files'].items()),'Frozen source changed.'
    if (ROOT/'run_complete.json').exists():raise ValueError('Do not overwrite completed results.')
    rows=[];paired=[]
    for phase in ['development','final']:
        for h in [5,21]:
            d=pd.read_csv(SOURCE/f'{phase}_h{h}_evaluation.csv')
            common=np.isfinite(d[['target']+list(MODELS)+[m+'_level' for m in MODELS]].to_numpy(float)).all(axis=1)
            pos=np.arange(len(d))[common]
            y=d.loc[common,'target_variance'].to_numpy(float)
            for model,name in MODELS.items():
                for conversion in ['direct_exp','training_smearing']:
                    f=np.exp(d.loc[common,model].to_numpy(float)) if conversion=='direct_exp' else d.loc[common,model+'_level'].to_numpy(float)
                    assert (y>0).all() and (f>0).all()
                    for lag in [21,42]:
                        result=mz_fit(y,f,pos,lag)
                        qlike=float(np.mean(y/f-np.log(y/f)-1))
                        rows.append({'phase':phase,'horizon':h,'model':model,'model_label':name,'conversion':conversion,'hac_lag':lag,'qlike_mean':qlike,**result})
                    if conversion=='training_smearing':
                        residualframe=pd.DataFrame({'date':d.loc[common,'date'],'actual_variance':y,'forecast_variance':f,
                                                    'error':f-y,'model':model,'phase':phase,'horizon':h})
                        paired.append(residualframe)
    table=pd.DataFrame(rows);table['holm_p_joint']=np.nan
    for phase in ['development','final']:
        for lag in [21,42]:
            idx=table.index[(table.phase==phase)&(table.hac_lag==lag)]
            table.loc[idx,'holm_p_joint']=core.holm(table.loc[idx,'p_joint_hac'])
    table.to_csv(ROOT/'calibration_results.csv',index=False)
    pd.concat(paired,ignore_index=True).to_csv(ROOT/'paired_variance_forecasts.csv',index=False)
    results={'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'protocol_sha256':sha(ROOT/'protocol.json'),'files_unchanged':all(sha(p)==v for p,v in protocol['files'].items()),
             'fit_count':len(table),'max_mse_reconciliation_error':float(table.mse_reconciliation_error.abs().max()),
             'status':'complete exploratory diagnostics; original predictions and models unchanged'}
    with (ROOT/'run_complete.json').open('x') as f:json.dump(results,f,indent=2)
    print(table[(table.phase=='final')&(table.hac_lag==21)][['horizon','model_label','conversion','n','relative_mean_bias_pct','alpha','beta','beta_low','beta_high','holm_p_joint','variance_level_mse','qlike_mean']].to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run'])
    freeze() if parser.parse_args().action=='freeze' else run()
