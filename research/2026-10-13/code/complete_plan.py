"""Finish the fixed IV plan: positive two-parameter calibration and QLIKE training."""
from pathlib import Path
import argparse, datetime, hashlib, json, platform
import numpy as np
import pandas as pd
import original_core as core
from calibration_diagnostics import mz_fit
from pair_inference import qlike, block_counts, summarize_pair

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'V1_HAR': 'HAR', 'V2_IV': 'HAR + ATM IV'}
OLD = ['direct_exp', 'training_smearing', 'rolling_multiplier', 'shrunk_multiplier']
NEW = ['positive_calibration', 'shrunk_positive_calibration', 'qlike_training']
METHODS = OLD + NEW
SPECS = {'V0_mean': [], 'V1_HAR': ['log_rv_1','log_rv_5','log_rv_22'],
         'V2_IV': ['log_rv_1','log_rv_5','log_rv_22','log_iv_var_30'],
         'V3_surface': ['log_rv_1','log_rv_5','log_rv_22','log_iv_var_30','term_slope','skew','curvature']}
WINDOW = 252
ALPHA = 10.
SHRINK = .5
MIN_TRAIN = 504
TRAIN_WINDOW = 1008
REFIT = 21
REPS = 5000
SEED = 2026101304


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    files = sorted((ROOT/'data').glob('*.csv')) + sorted((ROOT/'code').glob('*.py'))
    protocol = {
        'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'remaining two planned exploratory experiments on previously viewed historical data',
        'horizons': [5,21], 'models': MODELS, 'methods': METHODS,
        'two_parameter_calibration': {
            'forecast': 'exp(log(f) + d + g*z), z=(log(f)-historical_mean)/historical_std; equivalent exp(a+b*log(f))',
            'constraint': 'b=1+g/std >=0; positive forecast and monotonic response to original forecast',
            'objective': 'sum(QLIKE) + alpha/2*(d^2+g^2), alpha=10; identity at d=g=0',
            'window': WINDOW, 'minimum_matured_forecasts': WINDOW, 'alpha': ALPHA,
            'updates': 'every forecast origin; original issued log forecasts; same mature-history qualification as scalar experiment',
            'sensitivity': 'fixed half log correction, exp(log(f)+0.5*(d+g*z)); no second parameter fit',
            'label_eligibility': 'historical origin < current origin and target_end strictly < current asof_time',
            'phase_boundary': 'continuous history; recovered boundary targets enter only after maturity; not added to development scoring'},
        'qlike_training': {
            'forecast': 'exp(intercept + standardized original log-HAR / log-HAR+IV features)',
            'objective': 'sum(QLIKE) + alpha/2*sum(slope^2); intercept unpenalized; alpha=10 fixed',
            'information_structure': 'same original features and union-complete sample across V0/V1/V2/V3; no new features',
            'min_train': MIN_TRAIN, 'window': TRAIN_WINDOW, 'refit_calendar_sessions': REFIT, 'alpha': ALPHA,
            'standardization': 'within each strictly matured training sample only',
            'initialization': 'original log-target ridge solution; deterministic',
            'no_smearing': True,
            'literature_relation': 'loss-coherent positive-link extension using original log features; not a literal replication of level-linear qlikeHAR',
            'regularization_note': 'same fixed numerical ridge alpha but losses have different curvature; not an exactly matched effective-complexity comparison'},
        'optimizer': {'method': 'analytic gradient/Hessian damped projected Newton; convex log-link QLIKE objective',
                      'max_iterations': 150, 'mean_objective_projected_gradient_tolerance': 1e-7,
                      'armijo': 1e-4, 'max_backtracks': 60,
                      'failure_policy': 'stop and document numerical problem, never silently omit forecasts or switch methods by outcome'},
        'sample': 'all 2 models x 7 methods share scoring rows; compare coverage to prior scalar experiment',
        'primary_metric': 'normalized variance-level QLIKE',
        'secondary_metrics': ['variance-level MSE','relative mean bias','MZ alpha/beta','new forecast log-MSE (not substituted for old log-MSE)'],
        'regimes': 'existing lagged VIX <20,20-30,>=30; descriptive only',
        'loss_family': '66 prespecified paired comparisons per phase/block: inherited scalar 28, new methods vs direct/smearing 24, two-parameter vs corresponding scalar 8, new-method IV increment 6',
        'paired_inference': {'reps': REPS,'seed': SEED,'blocks':[21,42],
                             'multiplicity': 'Holm across the full 66 comparison family per phase/block, including older methods',
                             'calendar_gaps_preserved': True, 'refit_inside_bootstrap': False,
                             'interpretation': 'descriptive centered circular-block bootstrap conditional on issued forecasts'},
        'mz_inference': 'HAC lag21/42; Holm across 28 joint alpha=0,beta=1 tests per phase/lag; marginal intervals unadjusted',
        'no_selection': 'no tuning of alpha, windows, shrinkage, regimes or method using results; all variants reported',
        'independent_validation': 'finite-difference derivatives, known-model recovery, future-label perturbations, SciPy optimizer cross-checks on fixed synthetic and production samples',
        'limitations': ['already viewed historical data; new untouched data needed for independent confirmation',
                        'OHLC/overnight variance proxy, not latent conditional variance',
                        'overlapping horizons reduce independent information',
                        'base-data vendor timestamps and settlement-family issues remain unresolved'],
        'literature': {'coherent_forecasting': 'https://onlinelibrary.wiley.com/doi/full/10.1002/for.70114',
                       'retransformation': 'https://www.sciencedirect.com/science/article/pii/S0169207019302390'},
        'files': {str(path):sha(path) for path in files},
        'runtime': {'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}}
    with (ROOT/'protocol.json').open('x') as f:
        json.dump(protocol,f,ensure_ascii=False,indent=2)
    print('Both remaining steps and cumulative comparison family frozen.',flush=True)


def objective(theta, design, log_y, offset, penalty):
    z = offset + design @ theta
    log_ratio = log_y-z
    if np.any(log_ratio>700) or not np.isfinite(z).all():
        return np.inf, None, None
    ratio = np.exp(log_ratio)
    n = len(log_y)
    value = float(np.mean(ratio-log_ratio-1)+.5*np.sum(penalty*theta**2))
    gradient = design.T@(1-ratio)/n + penalty*theta
    hessian = (design.T*ratio)@design/n + np.diag(penalty)
    return value, gradient, hessian


def solve(design, log_y, offset=None, initial=None, penalty=None, lower=None):
    design = np.asarray(design,float); log_y = np.asarray(log_y,float)
    k = design.shape[1]
    offset = np.zeros(len(log_y)) if offset is None else np.asarray(offset,float)
    theta = np.zeros(k) if initial is None else np.asarray(initial,float).copy()
    penalty = np.zeros(k) if penalty is None else np.asarray(penalty,float)
    lower = np.full(k,-np.inf) if lower is None else np.asarray(lower,float)
    theta = np.maximum(theta,lower)
    for iteration in range(150):
        value, gradient, hessian = objective(theta,design,log_y,offset,penalty)
        if not np.isfinite(value): raise RuntimeError('Nonfinite initial objective.')
        at_lower = theta <= lower+1e-12
        projected = gradient.copy(); projected[at_lower & (gradient>=0)] = 0
        if np.max(np.abs(projected)) <= 1e-7:
            return theta, {'iterations':iteration,'objective':value,'projected_gradient_max':float(np.max(np.abs(projected))),
                           'minimum_hessian_eigenvalue':float(np.linalg.eigvalsh(hessian).min()),
                           'bound_active':bool(at_lower.any())}
        free = ~(at_lower & (gradient>=0))
        direction = np.zeros(k)
        direction[free] = -np.linalg.solve(hessian[np.ix_(free,free)],gradient[free])
        trial = np.maximum(theta+direction,lower)
        if gradient@(trial-theta)>=0:
            direction = -gradient/np.maximum(np.diag(hessian),1e-12)
        accepted = False
        step = 1.
        for _ in range(60):
            trial = np.maximum(theta+step*direction,lower)
            delta = trial-theta
            candidate, _, _ = objective(trial,design,log_y,offset,penalty)
            if np.isfinite(candidate) and candidate <= value+1e-4*gradient@delta:
                theta = trial; accepted = True; break
            step *= .5
        if not accepted: raise RuntimeError('Newton line search failed.')
    raise RuntimeError('Convex fit failed to converge within fixed limit.')


def fit_calibration(log_f, log_y, alpha=ALPHA):
    log_f=np.asarray(log_f,float); log_y=np.asarray(log_y,float)
    center=float(log_f.mean()); scale=float(log_f.std())
    if scale<1e-12: scale=1.
    x=np.column_stack([np.ones(len(log_f)),(log_f-center)/scale])
    coefficients, diagnostics=solve(x,log_y,offset=log_f,penalty=np.full(2,alpha/len(log_f)),lower=np.array([-np.inf,-scale]))
    d,g=coefficients
    return {'center':center,'scale':scale,'d':float(d),'g':float(g),
            'a':float(d-g*center/scale),'b':float(1+g/scale),**diagnostics}


def rolling_positive_calibration(d):
    asof=pd.DatetimeIndex(d.asof_time); end=pd.DatetimeIndex(d.full_target_end)
    log_y=d.full_log_target.to_numpy(float)
    base=d[list(MODELS)].to_numpy(float)
    valid=np.isfinite(base).all(axis=1)&np.isfinite(log_y)&~end.isna()
    valid &= np.isfinite(d[[m+'__training_smearing' for m in MODELS]].to_numpy(float)).all(axis=1)
    out=np.full(base.shape,np.nan); shrunk=out.copy(); audits=[]
    origins=np.arange(len(d))
    for i in origins:
        indices=np.flatnonzero(valid&(origins<i)&(end<asof[i]))[-WINDOW:]
        if len(indices)<WINDOW or not np.isfinite(base[i]).all(): continue
        for k,model in enumerate(MODELS):
            fit=fit_calibration(base[indices,k],log_y[indices])
            correction=fit['d']+fit['g']*(base[i,k]-fit['center'])/fit['scale']
            out[i,k]=np.exp(base[i,k]+correction)
            shrunk[i,k]=np.exp(base[i,k]+SHRINK*correction)
            audits.append({'date':d.date.iloc[i],'phase':d.phase.iloc[i],'model':model,
                           'origin_row':int(i),'history_n':len(indices),'history_first_row':int(indices.min()),
                           'history_last_row':int(indices.max()),'target_end_max':end[indices].max().isoformat(),
                           'asof_time':asof[i].isoformat(),'history_index_sha256':hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
                           'recovered_target_count':int(d.recovered_boundary_target.iloc[indices].sum()),**fit})
    return out,shrunk,pd.DataFrame(audits)


def fit_qlike(x,log_y,alpha=ALPHA):
    x=np.asarray(x,float);log_y=np.asarray(log_y,float)
    original=core.ridge_fit(x,log_y,alpha)
    design=np.column_stack([np.ones(len(x)),(x-original['mean'])/original['scale']])
    penalty=np.r_[0.,np.repeat(alpha/len(x),x.shape[1])]
    beta,diagnostics=solve(design,log_y,initial=original['beta'],penalty=penalty)
    return {'mean':original['mean'],'scale':original['scale'],'beta':beta,**diagnostics}


def walk_qlike(p,log_y,end,min_train=MIN_TRAIN,window=TRAIN_WINDOW,refit=REFIT,alpha=ALPHA,specs=SPECS):
    union=list(dict.fromkeys(col for columns in specs.values() for col in columns))
    x=p[union].to_numpy(float); values=np.asarray(log_y,float)
    end=pd.DatetimeIndex(end); asof=pd.DatetimeIndex(p.asof_time)
    good=np.isfinite(x).all(axis=1)&np.isfinite(values)&~end.isna()
    indices=np.arange(len(p));out=p[['date']].copy();audits=[]
    for model in MODELS: out[model+'__qlike_training']=np.nan
    selections={m:[union.index(col) for col in specs[m]] for m in MODELS}
    fit_at=-refit; fitted=None
    for i in indices:
        if not np.isfinite(x[i]).all():continue
        if fitted is None or i-fit_at>=refit:
            train=np.flatnonzero(good&(indices<i)&(end<asof[i]))[-window:]
            if len(train)<min_train:continue
            fitted={};fit_at=i
            for model,selection in selections.items():
                fit=fit_qlike(x[train][:,selection],values[train],alpha)
                fitted[model]=fit
                audits.append({'date':p.date.iloc[i],'model':model,'fit_row':int(i),'train_n':len(train),
                               'train_first_row':int(train.min()),'train_last_row':int(train.max()),
                               'target_end_max':end[train].max().isoformat(),'asof_time':asof[i].isoformat(),
                               'train_index_sha256':hashlib.sha256(train.astype('<i8').tobytes()).hexdigest(),
                               'feature_mean_json':json.dumps(fit['mean'].tolist()),'feature_scale_json':json.dumps(fit['scale'].tolist()),
                               'coefficients_json':json.dumps(fit['beta'].tolist()),
                               **{key:fit[key] for key in ['iterations','objective','projected_gradient_max','minimum_hessian_eigenvalue','bound_active']}})
        for model,selection in selections.items():
            fit=fitted[model]
            prediction=np.r_[1.,(x[i,selection]-fit['mean'])/fit['scale']]@fit['beta']
            out.loc[i,model+'__qlike_training']=np.exp(prediction)
    return out,pd.DataFrame(audits)


def comparison_specs():
    result=[]
    for model in MODELS:
        for method in OLD[1:]: result.append(('inherited',model,'direct_exp',model,method))
        for method in OLD[2:]: result.append(('inherited',model,'training_smearing',model,method))
        for method in NEW:
            for baseline in ['direct_exp','training_smearing']:
                result.append(('planned_extension',model,baseline,model,method))
        for baseline,enhanced in [('rolling_multiplier','positive_calibration'),('shrunk_multiplier','shrunk_positive_calibration')]:
            result.append(('scale_vs_scale_and_amplitude',model,baseline,model,enhanced))
    for method in METHODS:result.append(('iv_increment','V1_HAR',method,'V2_IV',method))
    assert len(result)==33
    return result


def run():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    assert all(sha(f)==v for f,v in protocol['files'].items()),'Frozen source changed.'
    if (ROOT/'run_complete.json').exists():raise ValueError('Completed results cannot be overwritten.')
    p=pd.read_csv(ROOT/'data/original_full_features.csv')
    for col in ['asof_time','entry_time','rv_available_time']:p[col]=pd.to_datetime(p[col],utc=True)
    statistics=[];comparisons=[];regimes=[];cal_audits=[];training_audits=[];checks=[];coverage=[]
    for h in [5,21]:
        d=pd.read_csv(ROOT/'data'/f'h{h}_forecast_timeline.csv')
        for col in ['asof_time','full_target_end']:d[col]=pd.to_datetime(d[col],utc=True)
        # Reproduce historical log fits and train schedule before altering any objective.
        y,end=core.target(p,'vol',h)
        replay=core.walk(p,y,end,SPECS,MIN_TRAIN,TRAIN_WINDOW,REFIT,ALPHA)
        paired=d[['date','fit_row','train_n']+list(MODELS)].merge(replay[['date','fit_row','train_n']+list(MODELS)],on='date',suffixes=('_saved','_replay'),validate='one_to_one')
        max_difference=0.
        for name in list(MODELS)+['fit_row','train_n']:
            np.testing.assert_allclose(paired[name+'_saved'],paired[name+'_replay'],rtol=0,atol=1e-10,equal_nan=True)
            difference=np.nanmax(np.abs(paired[name+'_saved']-paired[name+'_replay']))
            max_difference=max(max_difference,float(difference))
        pred,fit_audit=walk_qlike(p,y,end)
        fit_audit['horizon']=h;training_audits.append(fit_audit)
        pos,shrunk,cal_audit=rolling_positive_calibration(d)
        cal_audit['horizon']=h;cal_audits.append(cal_audit)
        for k,model in enumerate(MODELS):
            d[model+'__positive_calibration']=pos[:,k]
            d[model+'__shrunk_positive_calibration']=shrunk[:,k]
        d=d.merge(pred,on='date',validate='one_to_one')
        forecast_columns=[model+'__'+method for model in MODELS for method in METHODS]
        common=np.isfinite(d[forecast_columns].to_numpy(float)).all(axis=1)&d.target_variance.notna().to_numpy()
        common&=(d[forecast_columns].to_numpy(float)>0).all(axis=1)
        previous=d.scoring_common.to_numpy(bool)
        if not np.array_equal(common,previous):raise ValueError('New methods changed the common sample; do not silently drop rows.')
        d['complete_plan_common']=common
        for model in MODELS:
            for method in METHODS:
                forecast=d[model+'__'+method].to_numpy(float)
                valid=np.isfinite(forecast)&(forecast>0)&d.target_variance.notna().to_numpy()
                loss=np.full(len(d),np.nan);loss[valid]=qlike(d.target_variance.to_numpy(float)[valid],forecast[valid])
                d[model+'__'+method+'__qlike']=loss
        d.to_csv(ROOT/f'h{h}_completed_forecasts.csv',index=False)
        checks.append({'horizon':h,'max_original_forecast_or_schedule_difference':max_difference,
                       'same_common_sample_as_scalar_experiment':True,'base_model_fit_count':len(fit_audit),
                       'calibration_fit_count':len(cal_audit)})
        print(f'Fitted both planned extensions h={h}; sample reproduced.',flush=True)
        for phase in ['development','final']:
            stage=d[d.phase==phase].copy(); first=int(np.flatnonzero(stage.complete_plan_common)[0]);stage=stage.iloc[first:]
            calendar=np.arange(int(stage.calendar_position.min()),int(stage.calendar_position.max())+1)
            grid=pd.DataFrame({'calendar_position':calendar}).merge(stage,on='calendar_position',how='left',validate='one_to_one')
            good=grid.complete_plan_common.eq(True).to_numpy();actual=grid.loc[good,'target_variance'].to_numpy(float)
            positions=grid.loc[good,'calendar_position'].to_numpy(int)
            coverage.append({'phase':phase,'horizon':h,'n':int(good.sum()),'same_as_previous':True})
            for model,label in MODELS.items():
                for method in METHODS:
                    f=grid.loc[good,model+'__'+method].to_numpy(float)
                    for lag in [21,42]:
                        result=mz_fit(actual,f,positions,lag)
                        statistics.append({'phase':phase,'horizon':h,'model':model,'model_label':label,'method':method,'hac_lag':lag,
                                           'qlike_mean':float(qlike(actual,f).mean()),
                                           'new_forecast_log_mse':float(np.mean((np.log(actual)-np.log(f))**2)),**result})
                    for regime in ['low','middle','high']:
                        mask=good&grid.regime.eq(regime).to_numpy();yy=grid.loc[mask,'target_variance'].to_numpy(float);ff=grid.loc[mask,model+'__'+method].to_numpy(float)
                        regimes.append({'phase':phase,'horizon':h,'model':model,'method':method,'regime':regime,'n':int(mask.sum()),
                                        'qlike_mean':float(qlike(yy,ff).mean()) if mask.any() else np.nan,
                                        'relative_mean_bias_pct':float(100*(ff.mean()/yy.mean()-1)) if mask.any() else np.nan})
            for block in [21,42]:
                counts=block_counts(len(grid),block,REPS,SEED)
                for kind,bm,bt,em,et in comparison_specs():
                    a=grid[bm+'__'+bt+'__qlike'].to_numpy(float);b=grid[em+'__'+et+'__qlike'].to_numpy(float)
                    result,_=summarize_pair(a,b,good,counts)
                    comparisons.append({'phase':phase,'horizon':h,'comparison_type':kind,'baseline_model':bm,'baseline_method':bt,
                                        'enhanced_model':em,'enhanced_method':et,'block':block,**result})
            print(f'Evaluated {phase} h={h}, n={good.sum()}.',flush=True)
    stats=pd.DataFrame(statistics);comp=pd.DataFrame(comparisons);stats['holm_p_joint']=np.nan;comp['holm_p']=np.nan
    for phase in ['development','final']:
        for lag in [21,42]:
            ix=stats.index[(stats.phase==phase)&(stats.hac_lag==lag)];assert len(ix)==28
            stats.loc[ix,'holm_p_joint']=core.holm(stats.loc[ix,'p_joint_hac'])
        for block in [21,42]:
            ix=comp.index[(comp.phase==phase)&(comp.block==block)];assert len(ix)==66
            comp.loc[ix,'holm_p']=core.holm(comp.loc[ix,'p'])
    cal=pd.concat(cal_audits,ignore_index=True);train=pd.concat(training_audits,ignore_index=True)
    for audit in [cal,train]:
        assert (pd.to_datetime(audit.target_end_max,utc=True)<pd.to_datetime(audit.asof_time,utc=True)).all()
    stats.to_csv(ROOT/'complete_metrics.csv',index=False);comp.to_csv(ROOT/'complete_qlike_comparisons.csv',index=False)
    pd.DataFrame(regimes).to_csv(ROOT/'regime_metrics.csv',index=False)
    cal.to_csv(ROOT/'positive_calibration_audit.csv',index=False);train.to_csv(ROOT/'qlike_training_audit.csv',index=False)
    pd.DataFrame(coverage).to_csv(ROOT/'coverage.csv',index=False)
    result={'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':sha(ROOT/'protocol.json'),
            'frozen_files_unchanged':all(sha(f)==v for f,v in protocol['files'].items()),'reproduction_checks':checks,
            'positive_calibration_fit_count':len(cal),'qlike_model_fit_count':len(train),
            'max_calibration_projected_gradient':float(cal.projected_gradient_max.max()),
            'max_qlike_training_projected_gradient':float(train.projected_gradient_max.max()),
            'same_samples_as_previous':True,'diagnostic_count':len(stats),'comparison_count':len(comp),
            'status':'all four planned correction stages completed on existing data; independent unseen-period confirmation not performed'}
    with (ROOT/'run_complete.json').open('x') as file:json.dump(result,file,ensure_ascii=False,indent=2)
    print(stats[(stats.phase=='final')&(stats.hac_lag==21)&(stats.model=='V2_IV')][['horizon','method','n','qlike_mean','relative_mean_bias_pct','beta','holm_p_joint']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run'])
    freeze() if parser.parse_args().action=='freeze' else run()
