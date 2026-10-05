"""Check saved forecasts, clipping bounds, source identity and pre-run freeze."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from ivsurf import core
import robust_h2


def run(v1, package):
    v1, package = Path(v1), Path(package)
    freeze = json.loads((package/'protocol_freeze.json').read_text())
    for name, expected in freeze['files'].items():
        assert hashlib.sha256((package/name).read_bytes()).hexdigest() == expected, name
    for f in (package/'code/ivsurf').glob('*.py'):
        assert f.read_bytes() == (v1/'code/ivsurf'/f.name).read_bytes(), f.name
    records = []
    for stage, phase in [('former_development','development'),('former_holdout','final')]:
        dest = package/'results'/stage
        p = pd.read_csv(dest/'features.csv')
        for col in ['asof_time','entry_time','entry_dividend_available_time']:
            p[col] = pd.to_datetime(p[col], utc=True)
        _, specs = robust_h2.feature_specs(p)
        columns = list(dict.fromkeys(c for names in specs.values() for c in names))
        for h in [1,5,21]:
            y, end = core.target(p,'return',h)
            valid = np.isfinite(p[columns]).all(axis=1) & y.notna() & end.notna()
            fits = pd.read_csv(dest/f'H2_h{h}_fit_audit.csv')
            for row in fits.itertuples():
                eligible = p.loc[valid & (end < pd.Timestamp(row.asof_time))].tail(1008)
                assert len(eligible) == row.train_n
                assert row.effective_alpha == row.train_n
                np.testing.assert_allclose(eligible.skew_change.quantile([.01,.99]),
                                           [row.skew_lower,row.skew_upper],rtol=1e-12,atol=1e-15)
                assert end.loc[eligible.index].max() == pd.Timestamp(row.train_end_max)
            pred = pd.read_csv(dest/f'H2_h{h}_predictions.csv').set_index('date')
            old = pd.read_csv(v1/'reports'/phase/f'H2_h{h}_predictions.csv').set_index('date').reindex(pred.index)
            np.testing.assert_allclose(pred.target,old.target,equal_nan=True,atol=1e-14)
            np.testing.assert_allclose(pred.R0_mean,old.R0_mean,equal_nan=True,atol=1e-14)
            good = pred.R3_skew_change.notna()
            assert (pd.to_datetime(pred.loc[good,'train_end_max'],utc=True) <
                    pd.to_datetime(pred.loc[good,'asof_time'],utc=True)).all()
            assert (pred.loc[good,'skew_used'] >= pred.loc[good,'skew_lower']-1e-14).all()
            assert (pred.loc[good,'skew_used'] <= pred.loc[good,'skew_upper']+1e-14).all()
            records.append({'stage':stage,'horizon':h,'refits_checked':len(fits),
                            'saved_forecasts_checked':int(good.sum()),'labels_and_R0_match_v1':True,
                            'all_clipping_bounds_match_past_eligible_rows':True})
    result = {'status':'PASS','pre_run_frozen_files_unchanged':True,'v3_2_engine_unchanged':True,
              'checks':records,'note':'Local hashes and timestamps are audit records, not external preregistration.'}
    (package/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    a = argparse.ArgumentParser();a.add_argument('--v1',required=True);a.add_argument('--package',required=True)
    args=a.parse_args();run(args.v1,args.package)
