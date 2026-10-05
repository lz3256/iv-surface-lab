"""Build an explicitly exploratory daily panel from optionsDX and CRSP.

Never fills missing surface dates or infers an OPRA root/settlement convention.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tempfile
import numpy as np
import pandas as pd
from ivsurf import core, wrds_build


def bracket(x, y, target):
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    left, right = x[x <= target], x[x >= target]
    if not len(left) or not len(right):
        return np.nan, np.nan, np.nan
    lo, hi = left.max(), right.min()
    yl, yh = np.median(y[x == lo]), np.median(y[x == hi])
    value = yl if hi == lo else yl + (yh-yl)*(target-lo)/(hi-lo)
    return float(value), float(lo), float(hi)


def expiry_nodes(frame):
    q = frame.copy()
    eligible = q.DTE.between(7, 200)
    counts = {'raw_rows': len(q), 'eligible_tenor_rows': int(eligible.sum())}
    for side, sign in [('C', 1), ('P', -1)]:
        iv, delta = q[side+'_IV'], sign*q[side+'_DELTA']
        bid, ask = q[side+'_BID'], q[side+'_ASK']
        spread = (ask-bid)/((ask+bid)/2)
        good = (eligible & iv.between(.0001, 3) & delta.between(0, 1)
                & (bid > 0) & (ask >= bid) & np.isfinite(spread) & (spread <= .5))
        q[side+'_ok'] = good
        counts[side+'_accepted'] = int(good.sum())
    rows = []
    for (date, expiry), g in q.loc[eligible].groupby(['QUOTE_DATE', 'EXPIRE_DATE'], sort=True):
        if g.DTE.nunique() != 1:
            raise ValueError(f'Inconsistent tenor: {date}, {expiry}')
        row = {'date': date, 'expiry': expiry, 'tenor_days': float(g.DTE.iloc[0])}
        for side, sign, names in [('C',1,[(.5,'atm_call'),(.25,'call25')]),
                                  ('P',-1,[(.5,'atm_put'),(.25,'put25')])]:
            a = g.loc[g[side+'_ok']]
            x, y = sign*a[side+'_DELTA'].to_numpy(), a[side+'_IV'].to_numpy()
            for target, name in names:
                row[name], row[name+'_delta_low'], row[name+'_delta_high'] = bracket(x,y,target)
        rows.append(row)
    return pd.DataFrame(rows), counts


def constant_nodes(nodes):
    rows, audits = [], []
    for date, g in nodes.groupby('date', sort=True):
        row = {'date':date}
        for name, source, target in [('iv_atm_30','atm_call',30),('iv_atm_91','atm_call',91),
                                     ('iv_put25_30','put25',30),('iv_call25_30','call25',30),
                                     ('iv_atm_put_30','atm_put',30),('iv_atm_put_91','atm_put',91)]:
            variance, lo, hi = bracket(g.tenor_days, g[source]**2*g.tenor_days/365, target)
            row[name] = np.sqrt(variance/(target/365)) if np.isfinite(variance) else np.nan
            audits.append({'date':date,'node':name,'target_days':target,'lower_days':lo,
                           'upper_days':hi,'gap_days':hi-lo,'available':bool(np.isfinite(variance))})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(audits)


def hashfile(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def screen_curve(surface):
    """Pre-results extreme vendor-consistency screen, not put-call parity testing."""
    s = surface.copy()
    gap30 = (s.iv_atm_30-s.iv_atm_put_30).abs()
    gap91 = (s.iv_atm_91-s.iv_atm_put_91).abs()
    bad = (gap30 > .20) | (gap91 > .20)
    flags = s.loc[bad].copy()
    flags['atm_gap30'] = gap30.loc[bad]
    flags['atm_gap91'] = gap91.loc[bad]
    s.loc[bad,core.SURFACE] = np.nan
    return s, flags


def build(source, out):
    out = Path(out)
    if (out/'daily.csv').exists():
        raise ValueError('A completed input already exists; use a fresh directory.')
    out.mkdir(parents=True, exist_ok=True)
    source = Path(source)
    calendar, calmeta = wrds_build.exchange_calendar('2010-01-01','2023-12-31')
    ci = core.calendar_check(calendar).set_index('date')
    sessions = ci.loc['2010-01-01':'2023-12-31'].index
    valid_dates = set(sessions)
    all_nodes, all_audits, source_log, seen, observed = [], [], [], set(), set()
    needed = ['QUOTE_UNIXTIME','QUOTE_READTIME','QUOTE_DATE','EXPIRE_DATE','DTE','STRIKE',
              'C_IV','P_IV','C_DELTA','P_DELTA','C_BID','C_ASK','P_BID','P_ASK']
    for archive in sorted((source/'surface').glob('*.7z')):
        archive_record = {'file':archive.name,'sha256':hashfile(archive),'months':[]}
        with tempfile.TemporaryDirectory(prefix='ivsurface_') as tmp:
            subprocess.run(['bsdtar','-xf',str(archive),'-C',tmp],check=True)
            for path in sorted(Path(tmp).rglob('*.txt')):
                if path.name in seen:
                    raise ValueError('Duplicate month: '+path.name)
                seen.add(path.name)
                header = pd.read_csv(path,nrows=0,skipinitialspace=True)
                mapping = {c:c.strip().strip('[]') for c in header}
                if set(needed)-set(mapping.values()):
                    raise ValueError('Missing columns: '+path.name)
                q = pd.read_csv(path,usecols=[c for c in mapping if mapping[c] in needed],
                                skipinitialspace=True,low_memory=False).rename(columns=mapping)
                for col in ['QUOTE_DATE','EXPIRE_DATE','QUOTE_READTIME']:
                    q[col] = q[col].str.strip()
                for col in set(needed)-{'QUOTE_DATE','EXPIRE_DATE','QUOTE_READTIME'}:
                    q[col] = pd.to_numeric(q[col],errors='coerce')
                if q.duplicated(['QUOTE_DATE','EXPIRE_DATE','STRIKE']).any():
                    raise ValueError('Ambiguous duplicate contracts: '+path.name)
                observed.update(q.QUOTE_DATE.unique())
                excluded_dates = sorted(set(q.QUOTE_DATE)-valid_dates)
                q = q.loc[q.QUOTE_DATE.isin(valid_dates)].copy()
                nodes, counts = expiry_nodes(q)
                all_nodes.append(nodes)
                archive_record['months'].append({'file':path.name,**counts,'excluded_dates':excluded_dates})
        source_log.append(archive_record)
        pd.concat(all_nodes,ignore_index=True).to_csv(out/'expiry_nodes.csv',index=False)
        print('Built nodes:',archive.name,flush=True)
    nodes = pd.concat(all_nodes,ignore_index=True)
    surface, interpolation = constant_nodes(nodes)
    surface.to_csv(out/'surface.csv',index=False)
    interpolation.to_csv(out/'interpolation_audit.csv',index=False)
    surface, curve_flags = screen_curve(surface)
    curve_flags.to_csv(out/'curve_exclusions.csv',index=False)

    spy_path = source/'if92ihpkzaflamwz.csv'
    spy = pd.read_csv(spy_path); spy.columns = spy.columns.str.lower()
    if set(spy.ticker) != {'SPY'}:
        raise ValueError('Expected only SPY.')
    px = wrds_build.single_security(spy,['openprc','askhi','bidlo','prc','cfacpr','ret','retx'],
                                    'permno',84398,'CRSP SPY')
    if set(ci.index)-set(px.index):
        raise ValueError('Missing CRSP prices for calendar.')
    if not px.cfacpr.eq(1).all():
        raise ValueError('Unexpected corporate action factor.')
    cash, dividend_audit = wrds_build.distribution_amounts(px, list(ci.index[1:]))
    entries = ci.index[2:]
    if len(entries) != len(sessions):
        raise AssertionError('Entry/calendar alignment failed.')
    daily = pd.DataFrame({'date':sessions,'entry_session':entries})
    daily['reference_price'] = px.loc[sessions,'prc'].to_numpy()
    daily['reference_time'] = ci.loc[sessions,'session_close'].to_numpy()
    daily['rv_available_time'] = (ci.loc[sessions,'session_close']+pd.Timedelta(minutes=5)).to_numpy()
    # An explicit hypothetical receipt scenario; not a verified vendor delivery schedule.
    available = pd.to_datetime(pd.Series(entries)+' 08:00').dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    daily['surface_available_time'] = available.to_numpy()
    daily['asof_time'] = (available+pd.Timedelta(minutes=1)).to_numpy()
    daily['entry_time'] = ci.loc[entries,'session_open'].to_numpy()
    daily['entry_price'] = px.loc[entries,'openprc'].to_numpy()
    daily['cash_dividend'] = cash.loc[sessions].to_numpy()
    daily['entry_cash_dividend'] = cash.loc[entries].to_numpy()
    daily['is_ex_dividend'] = (daily.cash_dividend>0).astype(int)
    daily['entry_is_ex_dividend'] = (daily.entry_cash_dividend>0).astype(int)
    daily['entry_dividend_available_time'] = (ci.loc[entries,'session_close']+pd.Timedelta(minutes=5)).to_numpy()
    for source_col, target_col in [('openprc','open'),('askhi','high'),('bidlo','low'),('prc','close')]:
        daily[target_col] = px.loc[sessions,source_col].to_numpy()
    daily = daily.merge(surface,on='date',how='left',validate='one_to_one')
    meta = {'synthetic':False,'surface_kind':'curve','availability_basis':'assumed',
            'reference_instrument':'SPY CRSP close; changed from earlier SPX-reference proposal before results',
            'execution_instrument':'SPY CRSP PERMNO 84398 next regular-session open',
            'rv_instrument':'SPY daily Garman-Klass plus dividend-adjusted overnight variance proxy',
            'price_basis':'CRSP positive raw OHLC; constant CFACPR=1; cash distributions included',
            'return_basis':'cash_distribution_total_return','dividend_basis':dividend_audit,
            'iv_level_definition':'optionsDX vendor IV and signed Delta; 50-delta call proxy for ATM; linear IV/Delta and total-variance/tenor interpolation',
            'expiry_basis':'Unverified optionsDX DTE; no inferred OPRA roots or AM/PM settlement corrections',
            'curve_quality_screen':'Before model results: mask main four nodes if 30d or 91d call/put 50-delta IV absolute gap >0.20. 50-delta sides differ in strike; this is only an extreme consistency screen, not a parity test.',
            'availability_scenario':'Previous signal-date file assumed received by next XNYS session 08:00 New York; signal at 08:01; not verified or guaranteed conservative',
            'execution_price_time_basis':'Hypothetical fill at next SPY daily opening price; no actual fill evidence',
            'calendar':calmeta,'research_status':'exploratory; vendor contract and point-in-time history unresolved',
            'registration_verified':False,'raw_data_quality_checked_before_results':True,
            'limitations':['Vendor snapshot timestamps and historical revisions unverified.',
                           'Unknown SPX/SPXW and settlement family; vendor-tenor curve is an approximation.',
                           'Missing surface rows remain NaN; non-XNYS dates excluded.',
                           'SPY daily variance is a proxy, not intraday realized variance.',
                           'ATM IV squared minus HAR is a VRP proxy, not model-free variance risk premium.']}
    daily = core.validate(daily,calendar,meta)
    daily.to_csv(out/'daily.csv',index=False)
    calendar.to_csv(out/'calendar.csv',index=False)
    (out/'metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    complete = daily[core.SURFACE].notna().all(axis=1)
    summary = {'source_archives':source_log,'crsp_sha256':hashfile(spy_path),'adapter_sha256':hashfile(__file__),
               'signal_rows':len(daily),'complete_surface_rows':int(complete.sum()),
               'curve_quality_excluded_dates':curve_flags.date.tolist(),
               'missing_source_dates':sorted(valid_dates-observed),'excluded_source_dates':sorted(observed-valid_dates),
               'holdout_start':daily.date.iloc[-252], 'holdout_end':daily.date.iloc[-1],
               'metadata':meta}
    (out/'build_manifest.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print('Validated daily panel:',len(daily),'complete surfaces:',int(complete.sum()),flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source',required=True)
    ap.add_argument('--out',required=True)
    args = ap.parse_args()
    build(args.source,args.out)
