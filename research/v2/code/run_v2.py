"""One fixed exploratory comparison on already viewed 2010–2023 data."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from ivsurf import core
import robust_h2


def run(v1, out):
    v1,out = Path(v1),Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new results directory.')
    out.mkdir(parents=True,exist_ok=True)
    raw = core.validate(pd.read_csv(v1/'data/daily.csv'),pd.read_csv(v1/'data/calendar.csv'),
                        json.loads((v1/'data/metadata.json').read_text()))
    boundary = len(raw)-252
    all_metrics, all_comparisons, audits = [], [], []
    for phase in ['development','final']:
        stage = 'former_development' if phase=='development' else 'former_holdout'
        dest = out/stage;dest.mkdir()
        p = core.features(raw.iloc[:boundary].copy() if phase=='development' else raw.copy(),'curve')
        p,har = core.add_har(p,252,1008,21,10.)
        p,models = robust_h2.feature_specs(p)
        p.to_csv(dest/'features.csv',index=False)
        har.to_csv(dest/'har_audit.csv',index=False)
        for h in [1,5,21]:
            y,end = core.target(p,'return',h)
            pred,fit_audit = robust_h2.walk(p,y,end,models)
            first = pred.R0_mean.first_valid_index()
            if first is None:raise ValueError('No valid forecasts.')
            ev = pred.iloc[boundary:].copy() if phase=='final' else pred.loc[first:].copy()
            old = pd.read_csv(v1/'reports'/phase/f'H2_h{h}_predictions.csv').set_index('date')
            ev['v1_R3'] = ev.date.map(old.R3_skew_change)
            ev['v1_R2'] = ev.date.map(old.R2_HAR)
            # Preserve session rows for dependent bootstrap; compute common paired errors only.
            valid = ev[['target']+list(models)+['v1_R3','v1_R2']].notna().all(axis=1)
            common = ev.loc[valid]
            mean_mse = ((common.target-common.R0_mean)**2).mean()
            for model in list(models)+['v1_R2','v1_R3']:
                mse = ((common.target-common[model])**2).mean()
                all_metrics.append({'stage':stage,'horizon':h,'model':model,'n':len(common),'mse':mse,
                                    'r2_vs_mean':1-mse/mean_mse,
                                    'direction_accuracy':float(((common[model]>0)==(common.target>0)).mean())})
            for baseline,added in [('R2_HAR','R3_skew_change'),('R0_mean','R3_skew_change'),('v1_R3','R3_skew_change')]:
                a = ev[baseline].where(valid);b = ev[added].where(valid);target = ev.target.where(valid)
                test = core.block_test(a,b,target,1000,max(20,h),42)
                base_mse = ((target-a)**2).mean()
                all_comparisons.append({'stage':stage,'horizon':h,'baseline':baseline,'added':added,
                                        'incremental_r2':test['gain']/base_mse,**test})
            ev.to_csv(dest/f'H2_h{h}_predictions.csv',index=False)
            fit_audit.to_csv(dest/f'H2_h{h}_fit_audit.csv',index=False)
            audits.append({'stage':stage,'horizon':h,'date_start':ev.date.min(),'date_end':ev.date.max(),
                           'n_common':len(common),'max_abs_prediction':float(ev.R3_skew_change.abs().max()),
                           'known_training_labels':bool((fit_audit.train_end_max.map(pd.Timestamp)<fit_audit.asof_time.map(pd.Timestamp)).all())})
    comparisons = pd.DataFrame(all_comparisons)
    comparisons['p_holm_within_stage'] = np.nan
    for stage,ix in comparisons.groupby('stage').groups.items():
        comparisons.loc[ix,'p_holm_within_stage'] = core.holm(comparisons.loc[ix,'p'])
    pd.DataFrame(all_metrics).to_csv(out/'metrics.csv',index=False)
    comparisons.to_csv(out/'comparisons.csv',index=False)
    summary={'status':'EXPLORATORY; former holdout was already viewed before this patch',
             'primary_confirmatory_tests':0,'h2_penalty_per_observation':1.,'skew_quantiles':[.01,.99],
             'h1_changed':False,'har_alpha':10,'settlement_mapping_repaired':False,'audits':audits,
             'input_sha256':hashlib.sha256((v1/'data/daily.csv').read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(comparisons.to_string(index=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--v1',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();run(args.v1,args.out)
