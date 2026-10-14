"""Frozen exploratory re-evaluation of existing SPX/SPY variance forecasts."""
import os
from pathlib import Path
import argparse, datetime, hashlib, html, itertools, json, platform
import numpy as np
import pandas as pd
import original_core as core

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('IV_ORIGINAL_RUN', str(ROOT / 'private-inputs/original-run')))
MODELS = ['V0_mean', 'V1_HAR', 'V2_IV', 'V3_surface']
SPECS = {'V0_mean': [], 'V1_HAR': ['log_rv_1','log_rv_5','log_rv_22'],
         'V2_IV': ['log_rv_1','log_rv_5','log_rv_22','log_iv_var_30'],
         'V3_surface': ['log_rv_1','log_rv_5','log_rv_22','log_iv_var_30','term_slope','skew','curvature']}
PAIRS = [('V1_HAR','V2_IV','ATM IV beyond HAR'), ('V2_IV','V3_surface','Shape beyond ATM IV')]
GROUPS = ['all','low','middle','high']

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def qlike(y, forecast):
    y, forecast = np.asarray(y, float), np.asarray(forecast, float)
    if np.any(y <= 0) or np.any(forecast <= 0):
        raise ValueError('QLIKE requires positive realized proxy and forecast.')
    ratio = y / forecast
    return ratio - np.log(ratio) - 1

def assign_regime(dates, calendar_dates, vix):
    # An exact previous XNYS session join; no asof fill and no future VIX.
    cal = pd.DataFrame({'date': calendar_dates})
    cal['vix_date'] = cal.date.shift(1)
    mapping = cal.merge(vix.rename(columns={'date':'vix_date'}), on='vix_date', how='left', validate='many_to_one')
    out = pd.DataFrame({'date': dates}).merge(mapping, on='date', how='left', validate='one_to_one')
    out['regime'] = pd.cut(out.vix_close, [-np.inf,20,30,np.inf], right=False,
                           labels=['low','middle','high']).astype('object')
    out.loc[out.vix_close.isna(),'regime'] = None
    known = out.vix_date.notna()
    assert (out.loc[known,'vix_date'] < out.loc[known,'date']).all()
    return out

def block_counts(n, block, reps=5000, seed=20261013):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0,n,size=(reps,int(np.ceil(n/block))))
    idx = ((starts[:,:,None] + np.arange(block)) % n).reshape(reps,-1)[:,:n]
    counts = np.zeros((reps,n),dtype=np.int16)
    for k in range(reps): counts[k] = np.bincount(idx[k],minlength=n)
    return counts

def summarize_pair(a, b, eligible, counts):
    valid = eligible & np.isfinite(a) & np.isfinite(b)
    n = int(valid.sum())
    if not n: return {'n':0}, None
    baseline, enhanced = float(a[valid].mean()), float(b[valid].mean())
    gain = baseline-enhanced
    denom = counts[:,valid].sum(axis=1)
    ba = counts[:,valid] @ a[valid]
    bb = counts[:,valid] @ b[valid]
    ok = denom>0
    gb = (ba[ok]-bb[ok])/denom[ok]
    rb = 1-bb[ok]/ba[ok]
    out = {'n':n,'baseline_loss':baseline,'enhanced_loss':enhanced,'mean_gain':gain,
           'relative_gain_pct':100*gain/baseline,'ci_low_pct':np.nan,'ci_high_pct':np.nan,
           'gain_low':np.nan,'gain_high':np.nan,'p':np.nan,
           'interval_status':'suppressed: fewer than 30 observations' if n<30 else 'exploratory conditional block bootstrap'}
    if n>=30:
        out.update(ci_low_pct=100*np.quantile(rb,.025),ci_high_pct=100*np.quantile(rb,.975),
                   gain_low=np.quantile(gb,.025),gain_high=np.quantile(gb,.975),
                   p=(1+np.sum(np.abs(gb-gain)>=abs(gain)))/(len(gb)+1))
    draws = np.full(len(counts),np.nan);draws[ok] = 100*rb
    return out, draws

def freeze():
    files = [ROOT/'data/cboe_vix_2009_2023.csv'] + sorted((ROOT/'code').glob('*.py'))
    for phase in ['development','final']:
        files += [SOURCE/'reports'/phase/'features.csv',SOURCE/'reports'/phase/'summary.json']
        files += [SOURCE/'reports'/phase/f'H1_h{h}_predictions.csv' for h in [5,21]]
    protocol = {
        'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status':'exploratory extension on previously viewed historical data; not new confirmation',
        'source':str(SOURCE), 'phases':['development','final'], 'horizons':[5,21],
        'prediction_change':'none: replay original ridge fits solely to recover training-only smearing factors',
        'preserve_common_training_sample':True, 'reproduction_tolerance':1e-10,
        'vix_url':'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv',
        'vix_vintage':'downloaded 2026-10-13; historical revised series, not vintage-perfect',
        'vix_alignment':'previous XNYS session close before the option signal-date session; exact join, no fill',
        'vix_bins':{'low':'<20','middle':'20<=VIX<30','high':'>=30'},
        'metrics':['log-variance MSE','variance-level QLIKE'],
        'qlike_main':'exp(log forecast) times mean(exp(training residuals)) at the original fit date',
        'qlike_sensitivity':'exp(log forecast) without smearing, shown for all comparisons; not selected by results',
        'positive_target':'exp(saved log target); same original 1e-10 flooring',
        'comparisons':PAIRS, 'group_contrasts':'all three pairwise regime differences in relative loss reduction for HAR vs HAR+IV',
        'bootstrap':{'reps':5000,'seed':20261013,'main_block_sessions':21,'sensitivity_block_sessions':42,
                     'preserve_calendar_gaps':True,'joint_resampling_for_groups':True,
                     'refit_inside_bootstrap':False,'min_n_for_intervals':30},
        'p_values':'centered paired block-bootstrap descriptive p; Holm across all eligible cells and regime contrasts within each phase',
        'limitations':['daily OHLC plus overnight variance proxy, not high-frequency realized variance',
                       'smearing is an empirical training-only correction, not guaranteed unbiased under heteroskedasticity',
                       'old holdout already viewed; group differences do not establish causality',
                       'ridge models: no automatic formal Clark-West p-value claim',
                       'small regime samples and overlapping forecast horizons constrain inference'],
        'files':{str(p):sha(p) for p in files},
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}}
    with (ROOT/'protocol.json').open('x') as f: json.dump(protocol,f,indent=2,ensure_ascii=False)
    print('Protocol frozen before extension outcome calculations.')

def run():
    protocol = json.loads((ROOT/'protocol.json').read_text())
    for path,digest in protocol['files'].items():
        assert sha(path)==digest, f'Frozen source changed: {path}'
    if (ROOT/'run_complete.json').exists(): raise ValueError('Completed run exists; do not overwrite.')
    vix = pd.read_csv(ROOT/'data/cboe_vix_2009_2023.csv')
    rows, contrasts, audits, naive_rows, contributions = [],[],[],[],[]
    for phase in ['development','final']:
        folder = SOURCE/'reports'/phase
        p = pd.read_csv(folder/'features.csv')
        for col in ['asof_time','entry_time','rv_available_time']:
            p[col] = pd.to_datetime(p[col],utc=True)
        config = json.loads((folder/'summary.json').read_text())['config']
        for horizon in [5,21]:
            y,end = core.target(p,'vol',horizon)
            kwargs = {k:config[k] for k in ['min_train','window','refit','alpha']}
            logpred = core.walk(p,y,end,SPECS,**kwargs)
            levelpred = core.walk(p,y,end,SPECS,**kwargs,smear=True)
            saved = pd.read_csv(folder/f'H1_h{horizon}_predictions.csv')
            exact = saved.merge(logpred[['date','target']+MODELS],on='date',how='left',suffixes=('_saved','_replay'),validate='one_to_one')
            maxdiff = 0.
            for model in ['target']+MODELS:
                a,b=exact[model+'_saved'].to_numpy(float),exact[model+'_replay'].to_numpy(float)
                assert np.array_equal(np.isfinite(a),np.isfinite(b)),(phase,horizon,model,'coverage mismatch')
                good=np.isfinite(a);d=float(np.max(np.abs(a[good]-b[good]))) if good.any() else 0.
                maxdiff=max(maxdiff,d);assert d<1e-10,(phase,horizon,model,d)
            ev = saved.merge(levelpred[['date']+MODELS],on='date',suffixes=('','_level'),validate='one_to_one')
            regime = assign_regime(ev.date,p.date,vix)
            ev = ev.merge(regime,on='date',validate='one_to_one')
            assert ev.vix_close.notna().all(), 'Missing exact previous-session VIX; do not silently fill.'
            common = np.isfinite(ev[['target']+MODELS+[m+'_level' for m in MODELS]].to_numpy(float)).all(axis=1)
            target_level=np.exp(ev.target.to_numpy(float))
            ev['target_variance']=target_level
            for model in MODELS:
                ev[model+'_smear']=ev[model+'_level']/np.exp(ev[model])
                ev[model+'_log_mse']=(ev.target-ev[model])**2
                ev[model+'_qlike']=qlike(target_level,ev[model+'_level'])
                ev[model+'_qlike_naive']=qlike(target_level,np.exp(ev[model]))
            ev.to_csv(ROOT/f'{phase}_h{horizon}_evaluation.csv',index=False)
            has=logpred.V1_HAR.notna()
            assert (logpred.loc[has,'train_end_max']<logpred.loc[has,'asof_time']).all()
            audits.append({'phase':phase,'horizon':horizon,'max_log_prediction_difference':maxdiff,
                           'scored_common_n':int(common.sum()),'all_labels_known':True,
                           'vix_missing':int(ev.vix_close.isna().sum()),
                           'smear_min':float(ev.loc[common,'V1_HAR_smear'].min()),
                           'smear_max':float(ev.loc[common,'V1_HAR_smear'].max())})
            for block in [21,42]:
                counts=block_counts(len(ev),block)
                for baseline,enhanced,label in PAIRS:
                    for metric in ['log_mse','qlike']:
                        a,b = ev[baseline+'_'+metric].to_numpy(float),ev[enhanced+'_'+metric].to_numpy(float)
                        draws_by_group={};stats_by_group={}
                        all_gain=float(np.mean((a-b)[common]))
                        for group in GROUPS:
                            mask=common if group=='all' else common & (ev.regime==group).to_numpy()
                            stats,draws=summarize_pair(a,b,mask,counts)
                            stats_by_group[group]=stats;draws_by_group[group]=draws
                            rows.append({'phase':phase,'horizon':horizon,'comparison':label,'baseline':baseline,
                                         'enhanced':enhanced,'metric':metric,'group':group,'block':block,**stats})
                            if block==21 and group!='all':
                                contribution=float((a-b)[mask].sum()/int(common.sum()))
                                contributions.append({'phase':phase,'horizon':horizon,'comparison':label,'metric':metric,
                                                      'group':group,'n':int(mask.sum()),'weighted_gain':contribution,
                                                      'total_gain':all_gain,'gain_share_pct':100*contribution/all_gain if abs(all_gain)>1e-12 else np.nan})
                        if baseline=='V1_HAR':
                            for g0,g1 in itertools.combinations(GROUPS[1:],2):
                                s0,s1=stats_by_group[g0],stats_by_group[g1]
                                record={'phase':phase,'horizon':horizon,'metric':metric,'block':block,'contrast':g1+' minus '+g0,
                                        'n0':s0['n'],'n1':s1['n'],'difference_pp':np.nan,'low_pp':np.nan,'high_pp':np.nan,'p':np.nan}
                                if s0['n'] and s1['n']:
                                    obs=s1['relative_gain_pct']-s0['relative_gain_pct'];record['difference_pp']=obs
                                    if min(s0['n'],s1['n'])>=30:
                                        d=draws_by_group[g1]-draws_by_group[g0];d=d[np.isfinite(d)]
                                        record.update(low_pp=np.quantile(d,.025),high_pp=np.quantile(d,.975),
                                                      p=(1+np.sum(np.abs(d-obs)>=abs(obs)))/(len(d)+1))
                                contrasts.append(record)
                # Retransformation sensitivity: all cells, fixed before results.
                if block==21:
                    for baseline,enhanced,label in PAIRS:
                        a,b=ev[baseline+'_qlike_naive'].to_numpy(float),ev[enhanced+'_qlike_naive'].to_numpy(float)
                        for group in GROUPS:
                            mask=common if group=='all' else common & (ev.regime==group).to_numpy()
                            stats,_=summarize_pair(a,b,mask,counts)
                            naive_rows.append({'phase':phase,'horizon':horizon,'comparison':label,'group':group,**stats})
            print(f'Reproduced {phase} h={horizon}: n={common.sum()}, max delta={maxdiff:.2g}',flush=True)
    table=pd.DataFrame(rows);contrast=pd.DataFrame(contrasts)
    table['holm_p']=np.nan;contrast['holm_p']=np.nan
    for phase in ['development','final']:
        for block in [21,42]:
            ix=table.index[(table.phase==phase)&(table.block==block)&table.p.notna()]
            jx=contrast.index[(contrast.phase==phase)&(contrast.block==block)&contrast.p.notna()]
            pvals=np.r_[table.loc[ix,'p'],contrast.loc[jx,'p']]
            adjusted=core.holm(pvals)
            table.loc[ix,'holm_p']=adjusted[:len(ix)];contrast.loc[jx,'holm_p']=adjusted[len(ix):]
    table.to_csv(ROOT/'regime_comparisons.csv',index=False)
    contrast.to_csv(ROOT/'regime_contrasts.csv',index=False)
    pd.DataFrame(naive_rows).to_csv(ROOT/'qlike_naive_sensitivity.csv',index=False)
    pd.DataFrame(contributions).to_csv(ROOT/'gain_contributions.csv',index=False)
    pd.DataFrame(audits).to_csv(ROOT/'reproduction_audit.csv',index=False)
    result={'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':sha(ROOT/'protocol.json'),
            'checks':audits,'source_files_unchanged':all(sha(p)==v for p,v in protocol['files'].items()),
            'status':'complete exploratory analysis; no new holdout, no model selection'}
    with (ROOT/'run_complete.json').open('x') as f:json.dump(result,f,indent=2)
    print(table[(table.phase=='final')&(table.block==21)&(table.baseline=='V1_HAR')][['horizon','metric','group','n','relative_gain_pct','ci_low_pct','ci_high_pct','holm_p']].to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run'])
    action=parser.parse_args().action
    freeze() if action=='freeze' else run()
