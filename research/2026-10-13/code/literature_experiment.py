"""Fixed literature-guided extensions; historical exploration, no method search."""
from pathlib import Path
import argparse,datetime,hashlib,json
import numpy as np
import pandas as pd
import original_core as core
from calibration_diagnostics import mz_fit
from pair_inference import qlike,block_counts,summarize_pair,assign_regime
from legacy_plan import comparison_specs as legacy_comparisons

ROOT=Path(__file__).resolve().parents[1]
MODELS={'V1_HAR':'HAR','V2_IV':'HAR + ATM IV'}
SPECS={'V0_mean':[],'V1_HAR':['log_rv_1','log_rv_5','log_rv_22'],
       'V2_IV':['log_rv_1','log_rv_5','log_rv_22','log_iv_var_30'],
       'V3_surface':['log_rv_1','log_rv_5','log_rv_22','log_iv_var_30','term_slope','skew','curvature']}
LEGACY=['direct_exp','training_smearing','rolling_multiplier','shrunk_multiplier','positive_calibration','shrunk_positive_calibration','qlike_training']
NEW=['recent_direct','recent_variance_correction','window_average','level_nnls_calibration','fixed_blend']
PRIMARY=LEGACY+NEW
SECONDARY=['direct_exp','training_smearing']+NEW
SEED=2026101305
HOLDOUT_START='2022-12-29'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze():
    files=sorted((ROOT/'data').glob('*'))+sorted((ROOT/'code').glob('*.py'))
    files+=[ROOT/'data_audit_results.json',ROOT/'audit_protocol.json',ROOT/'literature_review.json']
    protocol={'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status':'limited fixed literature-guided exploration on already viewed history; no independent confirmation',
              'primary_horizons':[5,21],'secondary_horizons':[1,22],
              'models':MODELS,'legacy_methods':LEGACY,'new_methods':NEW,
              'recent_estimation':{'half_life_calendar_sessions':252,'window':1008,'min_train':504,'refit':21,'alpha':10,
                                   'weights':'2**(-(latest_train_session-train_session)/252), normalized to sum=train_n',
                                   'standardization':'weighted training mean and variance only',
                                   'variance_conversion':'exp(log_prediction+0.5*weighted_training_residual_variance)',
                                   'assumption':'local conditional-normal approximation; no guarantee of unbiasedness'},
              'window_average':{'windows':[252,504,1008],'weights':[1/3,1/3,1/3],
                                'combination':'arithmetic mean of positive direct-exp forecasts',
                                'startup':'common original 504-observation minimum historical pool; individual windows truncate available pool'},
              'level_calibration':{'window':252,'method':'nonnegative least squares y=a+b*f in variance levels, a>=0,b>=0; analytic global optimum',
                                   'history':'latest252 strictly matured originally issued baseline forecasts; no hyperparameter search',
                                   'difference_from_paper':'positive constrained adaptation of recursive MZ, not the unrestricted OLS implementation'},
              'fixed_blend':'.5*original_direct_exp + .5*recent_variance_correction; no learned weights',
              'timing_sensitivity':{'shift':'all IV/surface information delayed one additional calendar trading session, no fill',
                                    'design':'lag0 and lag1 models share the intersection-complete training pool, same ridge fits/targets/refit dates',
                                    'comparisons':'IV increment under lag0, IV increment under lag1, lag1 vs lag0 IV forecast',
                                    'inference':'separate6-test family per phase/block for horizons5/21',
                                    'interpretation':'feature-staleness sensitivity; not proof of actual T+1 vendor delivery'},
              'label_maturity':'target_end strictly < asof_time and earlier-origin forecast for all calibrations/training',
              'primary_sample':'identical original completion common samples; fail if any new method silently drops rows',
              'secondary_sample':'original final-period rows only; all7 methods share matured-calibration eligibility; descriptive, no significance selection',
              'primary_inference':'5000 paired calendar circular-block draws, blocks21/42, seeded; cumulative116 comparisons per phase/block includes previous66 and50 new; Holm correction',
              'mz':'HAC21/42, joint Holm across48 primary tests per phase/lag; secondary estimates descriptive only',
              'secondary_regimes':'same fixed lagged VIX<20,20-30,>=30; point estimates only',
              'validation':'original forecast replay, future-label perturbations, weighted fit matrix identity, NNLS recovery/SciPy comparison, independent production history reconstruction',
              'no_tuning':'do not tune weights/half-life/windows or replace failed methods after results',
              'data_status':'raw epoch/date/tenor consistency checked; actual receipt and AM/PM family remain unverified',
              'new_data_status':'local inputs through2023; vendor current public product lists2010-2023; new-period confirmation unavailable here',
              'literature_relation':'simplified fixed adaptations; not exact replications or guaranteed corrections',
              'files':{str(p):sha(p) for p in files if p.is_file()}}
    with (ROOT/'protocol.json').open('x') as f:json.dump(protocol,f,indent=2,ensure_ascii=False)
    print('Literature variants, added horizons and timing sensitivity frozen.',flush=True)


def weighted_fit(x,y,positions,half_life=252,alpha=10.):
    x=np.asarray(x,float);y=np.asarray(y,float);positions=np.asarray(positions,int)
    w=2.**(-(positions.max()-positions)/half_life);w*=len(y)/w.sum()
    mean=np.average(x,axis=0,weights=w)
    scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));scale[scale<1e-12]=1.
    z=np.column_stack([np.ones(len(y)),(x-mean)/scale])
    penalty=np.eye(z.shape[1])*alpha;penalty[0,0]=0
    beta=np.linalg.solve(z.T@(w[:,None]*z)+penalty,z.T@(w*y))
    residual=y-z@beta
    variance=float(np.average(residual**2,weights=w))
    return {'mean':mean,'scale':scale,'beta':beta,'residual_variance':variance,
            'effective_n':float(w.sum()**2/(w@w)),'weights':w,
            'weighted_normal_equation_max_error':float(np.max(np.abs(z.T@(w*residual)-penalty@beta)))}


def nnls_level(f,y):
    f=np.asarray(f,float);y=np.asarray(y,float)
    if not ((f>0).all() and (y>0).all()):raise ValueError('Positive variance forecasts and targets required.')
    scale=float(f.mean());x=f/scale;target=y/scale
    design=np.column_stack([np.ones(len(f)),x])
    unconstrained=np.linalg.lstsq(design,target,rcond=None)[0]
    candidates=[np.array([target.mean(),0.]),np.array([0.,float(x@target/(x@x))])]
    if (unconstrained>=-1e-12).all():candidates.append(np.maximum(unconstrained,0))
    losses=[float(np.mean((target-design@candidate)**2)) for candidate in candidates]
    coefficient=candidates[int(np.argmin(losses))]
    return {'a':float(scale*coefficient[0]),'b':float(coefficient[1]),'scale':scale,
            'normalized_mse':min(losses),'boundary':bool((coefficient<=1e-12).any())}


def walk_variants(p,log_y,end):
    union=list(dict.fromkeys(col for cols in SPECS.values() for col in cols))
    x=p[union].to_numpy(float);ly=np.asarray(log_y,float);end=pd.DatetimeIndex(end);asof=pd.DatetimeIndex(p.asof_time)
    valid=np.isfinite(x).all(axis=1)&np.isfinite(ly)&~end.isna();positions=np.arange(len(p))
    selections={model:[union.index(col) for col in SPECS[model]] for model in MODELS}
    out=p[['date']].copy();audit=[]
    for model in MODELS:
        for method in ['direct_exp','training_smearing','recent_direct','recent_variance_correction','window_average','fixed_blend']:
            out[model+'__'+method]=np.nan
    fit_at=-21;fitted=None
    for i in positions:
        if not np.isfinite(x[i]).all():continue
        if fitted is None or i-fit_at>=21:
            indices=np.flatnonzero(valid&(positions<i)&(end<asof[i]))[-1008:]
            if len(indices)<504:continue
            fitted={};fit_at=i
            for model,selection in selections.items():
                xx=x[indices][:,selection];yy=ly[indices]
                base=core.ridge_fit(xx,yy,10.);recent=weighted_fit(xx,yy,indices)
                windows=[]
                for length in [252,504,1008]:
                    ix=indices[-length:];windowfit=base if length==1008 else core.ridge_fit(x[ix][:,selection],ly[ix],10.)
                    windows.append(windowfit)
                fitted[model]=(base,recent,windows)
                audit.append({'date':p.date.iloc[i],'model':model,'fit_row':int(i),'train_n':len(indices),
                              'target_end_max':end[indices].max().isoformat(),'asof_time':asof[i].isoformat(),
                              'train_index_sha256':hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
                              'weighted_residual_variance':recent['residual_variance'],'effective_n':recent['effective_n'],
                              'weighted_normal_equation_max_error':recent['weighted_normal_equation_max_error'],
                              'recent_mean_json':json.dumps(recent['mean'].tolist()),'recent_scale_json':json.dumps(recent['scale'].tolist()),
                              'recent_beta_json':json.dumps(recent['beta'].tolist())})
        for model,selection in selections.items():
            base,recent,windows=fitted[model]
            log_base=core.ridge_predict(base,x[i,selection]);log_recent=core.ridge_predict(recent,x[i,selection])
            direct=np.exp(log_base);recent_direct=np.exp(log_recent)
            converted=np.exp(log_recent+.5*recent['residual_variance'])
            vals={'direct_exp':direct,'training_smearing':direct*np.exp(base['residual']).mean(),
                  'recent_direct':recent_direct,'recent_variance_correction':converted,
                  'window_average':float(np.mean([np.exp(core.ridge_predict(w,x[i,selection])) for w in windows])),
                  'fixed_blend':.5*(direct+converted)}
            for method,value in vals.items():out.loc[i,model+'__'+method]=value
    return out,pd.DataFrame(audit)


def calibrate_level(p,log_y,end,out):
    f=out[[model+'__direct_exp' for model in MODELS]].to_numpy(float);y=np.exp(np.asarray(log_y,float))
    end=pd.DatetimeIndex(end);asof=pd.DatetimeIndex(p.asof_time);positions=np.arange(len(p))
    valid=np.isfinite(f).all(axis=1)&np.isfinite(y)&~end.isna();result=np.full(f.shape,np.nan);audit=[]
    for i in positions:
        ix=np.flatnonzero(valid&(positions<i)&(end<asof[i]))[-252:]
        if len(ix)<252 or not np.isfinite(f[i]).all():continue
        for k,model in enumerate(MODELS):
            fit=nnls_level(f[ix,k],y[ix]);result[i,k]=fit['a']+fit['b']*f[i,k]
            if not result[i,k]>0:raise ValueError('Nonpositive level calibration.')
            audit.append({'date':p.date.iloc[i],'model':model,'origin_row':int(i),'history_n':len(ix),
                          'target_end_max':end[ix].max().isoformat(),'asof_time':asof[i].isoformat(),
                          'history_index_sha256':hashlib.sha256(ix.astype('<i8').tobytes()).hexdigest(),**fit})
    return result,pd.DataFrame(audit)


def timing_panel(p,log_y,end):
    q=p.copy();cols=['log_iv_var_30','term_slope','skew','curvature']
    for col in cols:q[col+'_lag1']=q[col].shift(1)
    guard=SPECS['V3_surface']+[col+'_lag1' for col in cols]
    specs={'V1_HAR':SPECS['V1_HAR'],'V2_IV':SPECS['V2_IV'],
           'V2_IV_delay':['log_rv_1','log_rv_5','log_rv_22','log_iv_var_30_lag1']}
    # The guard is only an eligibility requirement, never a fitted extra model.
    x=q[guard].to_numpy(float);ly=np.asarray(log_y,float);end=pd.DatetimeIndex(end);asof=pd.DatetimeIndex(q.asof_time)
    valid=np.isfinite(x).all(axis=1)&np.isfinite(ly)&~end.isna();positions=np.arange(len(q))
    out=q[['date']].copy();fitted=None;fit_at=-21;audit=[]
    for model in specs:out[model]=np.nan
    for i in positions:
        if not np.isfinite(x[i]).all():continue
        if fitted is None or i-fit_at>=21:
            ix=np.flatnonzero(valid&(positions<i)&(end<asof[i]))[-1008:]
            if len(ix)<504:continue
            fitted={model:core.ridge_fit(q[columns].to_numpy(float)[ix],ly[ix],10.) for model,columns in specs.items()};fit_at=i
            audit.append({'fit_row':int(i),'train_n':len(ix),'target_end_max':end[ix].max().isoformat(),'asof_time':asof[i].isoformat(),
                          'train_index_sha256':hashlib.sha256(ix.astype('<i8').tobytes()).hexdigest()})
        for model,columns in specs.items():out.loc[i,model]=np.exp(core.ridge_predict(fitted[model],q.loc[i,columns].to_numpy(float)))
    return out,pd.DataFrame(audit)


def comparison_specs():
    result=legacy_comparisons()
    for model in MODELS:
        for method in NEW:
            for baseline in ['direct_exp','training_smearing']:
                result.append(('literature_extension',model,baseline,model,method))
    for method in NEW:result.append(('iv_increment','V1_HAR',method,'V2_IV',method))
    assert len(result)==58
    return result


def run():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    assert all(sha(path)==v for path,v in protocol['files'].items())
    if (ROOT/'run_complete.json').exists():raise ValueError('Completed run exists.')
    audit=json.loads((ROOT/'data_audit_results.json').read_text())
    assert audit['quote_not_before_assumed_asof_unique']==0 and audit['sources_unchanged']
    p=pd.read_csv(ROOT/'data/original_full_features.csv')
    for c in ['asof_time','rv_available_time']:p[c]=pd.to_datetime(p[c],utc=True)
    vix=pd.read_csv(ROOT/'data/cboe_vix_2009_2023.csv')
    regime=assign_regime(p.date,p.date,vix)
    rows=[];comps=[];secondary=[];groups=[];training=[];calibration=[];timing_stats=[];timing_comp=[];timing_audits=[];checks=[]
    for h in [1,5,21,22]:
        ly,end=core.target(p,'vol',h);out,train=walk_variants(p,ly,end)
        level,cal=calibrate_level(p,ly,end,out)
        for k,model in enumerate(MODELS):out[model+'__level_nnls_calibration']=level[:,k]
        train['horizon']=h;cal['horizon']=h;training.append(train);calibration.append(cal)
        if h in [5,21]:
            d=pd.read_csv(ROOT/'data'/f'h{h}_completed_forecasts.csv')
            joined=d[['date']+[model+'__'+m for model in MODELS for m in ['direct_exp','training_smearing']]].merge(out,on='date',suffixes=('_saved','_new'),validate='one_to_one')
            differences=[]
            for model in MODELS:
                for method in ['direct_exp','training_smearing']:
                    key=model+'__'+method
                    np.testing.assert_allclose(joined[key+'_saved'],joined[key+'_new'],rtol=0,atol=1e-10,equal_nan=True)
                    differences.append(float(np.nanmax(np.abs(joined[key+'_saved']-joined[key+'_new']))))
            d=d.merge(out[['date']+[model+'__'+method for model in MODELS for method in NEW]],on='date',validate='one_to_one')
            common=d.complete_plan_common.to_numpy(bool)
            required=[model+'__'+method for model in MODELS for method in NEW]
            assert np.isfinite(d.loc[common,required].to_numpy(float)).all()
            checks.append({'horizon':h,'baseline_max_abs_difference':max(differences),'primary_sample_unchanged':True})
            methods=PRIMARY
        else:
            d=out.copy();d['target_variance']=np.exp(ly);d['target_end']=end;d['calendar_position']=np.arange(len(p))
            d['phase']=np.where(d.date>=HOLDOUT_START,'final','development')
            d=d.merge(regime,on='date',validate='one_to_one');methods=SECONDARY
            required=[model+'__'+method for model in MODELS for method in methods]
            common=d.target_variance.notna().to_numpy()&np.isfinite(d[required].to_numpy(float)).all(axis=1)
        d['literature_common']=common
        for model in MODELS:
            for method in methods:
                f=d[model+'__'+method].to_numpy(float);yy=d.target_variance.to_numpy(float)
                valid=np.isfinite(f)&np.isfinite(yy);loss=np.full(len(d),np.nan)
                loss[valid]=qlike(yy[valid],f[valid]);d[model+'__'+method+'__qlike']=loss
        d.to_csv(ROOT/f'h{h}_forecasts.csv',index=False)
        phases=['development','final'] if h in [5,21] else ['final']
        for phase in phases:
            stage=d[d.phase==phase].copy();first=int(np.flatnonzero(stage.literature_common)[0]);stage=stage.iloc[first:]
            calendar=np.arange(int(stage.calendar_position.min()),int(stage.calendar_position.max())+1)
            grid=pd.DataFrame({'calendar_position':calendar}).merge(stage,on='calendar_position',how='left',validate='one_to_one')
            good=grid.literature_common.eq(True).to_numpy();yy=grid.loc[good,'target_variance'].to_numpy(float);positions=grid.loc[good,'calendar_position'].to_numpy(int)
            for model in MODELS:
                for method in methods:
                    f=grid.loc[good,model+'__'+method].to_numpy(float)
                    for lag in ([21,42] if h in [5,21] else [22]):
                        result=mz_fit(yy,f,positions,lag)
                        record={'phase':phase,'horizon':h,'model':model,'method':method,'hac_lag':lag,
                                'qlike_mean':float(qlike(yy,f).mean()),'log_mse':float(np.mean((np.log(yy)-np.log(f))**2)),**result}
                        if h in [5,21]:rows.append(record)
                        else:
                            # Secondary horizons are descriptive, with no significance claims.
                            for field in ['p_joint_hac','wald_2df']:record.pop(field,None)
                            secondary.append(record)
                    for g in ['low','middle','high']:
                        mask=good&grid.regime.eq(g).to_numpy();yg=grid.loc[mask,'target_variance'].to_numpy(float);fg=grid.loc[mask,model+'__'+method].to_numpy(float)
                        groups.append({'phase':phase,'horizon':h,'model':model,'method':method,'regime':g,'n':int(mask.sum()),
                                       'qlike_mean':float(qlike(yg,fg).mean()) if mask.any() else np.nan,
                                       'relative_mean_bias_pct':float(100*(fg.mean()/yg.mean()-1)) if mask.any() else np.nan})
            if h in [5,21]:
                for block in [21,42]:
                    counts=block_counts(len(grid),block,5000,SEED)
                    for kind,bm,bt,em,et in comparison_specs():
                        result,_=summarize_pair(grid[bm+'__'+bt+'__qlike'].to_numpy(float),grid[em+'__'+et+'__qlike'].to_numpy(float),good,counts)
                        comps.append({'phase':phase,'horizon':h,'comparison_type':kind,'baseline_model':bm,'baseline_method':bt,
                                      'enhanced_model':em,'enhanced_method':et,'block':block,**result})
            print(f'Completed literature methods h={h}, {phase}, n={good.sum()}',flush=True)
        if h in [5,21]:
            timing,ta=timing_panel(p,ly,end);ta['horizon']=h;timing_audits.append(ta)
            td=d[['date','phase','calendar_position','target_variance','literature_common']].merge(timing,on='date',validate='one_to_one')
            td['timing_common']=td.literature_common&np.isfinite(td[['V1_HAR','V2_IV','V2_IV_delay']].to_numpy(float)).all(axis=1)
            td.to_csv(ROOT/f'h{h}_timing_forecasts.csv',index=False)
            for phase in ['development','final']:
                stage=td[td.phase==phase].copy();first=int(np.flatnonzero(stage.timing_common)[0]);stage=stage.iloc[first:]
                grid=pd.DataFrame({'calendar_position':np.arange(int(stage.calendar_position.min()),int(stage.calendar_position.max())+1)}).merge(stage,on='calendar_position',how='left',validate='one_to_one')
                good=grid.timing_common.eq(True).to_numpy();yy=grid.loc[good,'target_variance'].to_numpy(float)
                for model in ['V1_HAR','V2_IV','V2_IV_delay']:
                    f=grid.loc[good,model].to_numpy(float)
                    timing_stats.append({'phase':phase,'horizon':h,'model':model,'n':int(good.sum()),'qlike_mean':float(qlike(yy,f).mean()),
                                         'relative_mean_bias_pct':float(100*(f.mean()/yy.mean()-1))})
                    loss=np.full(len(grid),np.nan);loss[good]=qlike(yy,f);grid[model+'_qlike']=loss
                for block in [21,42]:
                    counts=block_counts(len(grid),block,5000,SEED)
                    for bm,em in [('V1_HAR','V2_IV'),('V1_HAR','V2_IV_delay'),('V2_IV','V2_IV_delay')]:
                        result,_=summarize_pair(grid[bm+'_qlike'].to_numpy(float),grid[em+'_qlike'].to_numpy(float),good,counts)
                        timing_comp.append({'phase':phase,'horizon':h,'baseline':bm,'enhanced':em,'block':block,**result})
    metrics=pd.DataFrame(rows);comp=pd.DataFrame(comps);tcomp=pd.DataFrame(timing_comp)
    metrics['holm_p_joint']=np.nan;comp['holm_p']=np.nan;tcomp['holm_p']=np.nan
    for phase in ['development','final']:
        for lag in [21,42]:
            ix=metrics.index[(metrics.phase==phase)&(metrics.hac_lag==lag)];assert len(ix)==48
            metrics.loc[ix,'holm_p_joint']=core.holm(metrics.loc[ix,'p_joint_hac'])
        for block in [21,42]:
            ix=comp.index[(comp.phase==phase)&(comp.block==block)];assert len(ix)==116
            comp.loc[ix,'holm_p']=core.holm(comp.loc[ix,'p'])
            ti=tcomp.index[(tcomp.phase==phase)&(tcomp.block==block)];assert len(ti)==6
            tcomp.loc[ti,'holm_p']=core.holm(tcomp.loc[ti,'p'])
    metrics.to_csv(ROOT/'primary_metrics.csv',index=False);comp.to_csv(ROOT/'primary_comparisons.csv',index=False)
    pd.DataFrame(secondary).to_csv(ROOT/'secondary_horizon_metrics.csv',index=False);pd.DataFrame(groups).to_csv(ROOT/'regime_metrics.csv',index=False)
    pd.concat(training,ignore_index=True).to_csv(ROOT/'training_audit.csv',index=False)
    pd.concat(calibration,ignore_index=True).to_csv(ROOT/'level_calibration_audit.csv',index=False)
    pd.concat(timing_audits,ignore_index=True).to_csv(ROOT/'timing_training_audit.csv',index=False)
    pd.DataFrame(timing_stats).to_csv(ROOT/'timing_metrics.csv',index=False);tcomp.to_csv(ROOT/'timing_comparisons.csv',index=False)
    for tables in [training,calibration,timing_audits]:
        for a in tables:assert (pd.to_datetime(a.target_end_max,utc=True)<pd.to_datetime(a.asof_time,utc=True)).all()
    result={'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':sha(ROOT/'protocol.json'),
            'frozen_inputs_unchanged':all(sha(path)==v for path,v in protocol['files'].items()),'original_forecast_reproduction':checks,
            'training_fits':sum(len(t) for t in training),'level_calibration_fits':sum(len(t) for t in calibration),
            'primary_diagnostics':len(metrics),'primary_comparisons':len(comp),'secondary_descriptive_rows':len(secondary),
            'timing_comparisons':len(tcomp),'status':'all fixed literature variants,1/22-day gaps,and timing sensitivity completed; no unseen-period confirmation'}
    with (ROOT/'run_complete.json').open('x') as f:json.dump(result,f,indent=2)
    print(metrics[(metrics.phase=='final')&(metrics.model=='V2_IV')&(metrics.hac_lag==21)&metrics.method.isin(['direct_exp','training_smearing']+NEW)][
        ['horizon','method','qlike_mean','relative_mean_bias_pct','beta','holm_p_joint']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run'])
    freeze() if parser.parse_args().action=='freeze' else run()
