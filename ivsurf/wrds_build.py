"""WRDS daily-input builder with explicit, unverified availability assumptions."""
from __future__ import annotations
import hashlib
import re
from pathlib import Path
import numpy as np
import pandas as pd
from . import core, data


def exchange_calendar(start, end):
    """Include the preceding dividend base-price session and next execution session."""
    try:
        import exchange_calendars as xc
    except ImportError as exc:
        raise ValueError('Install requirements-calendar.txt (or requirements-wrds.txt) first.') from exc
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tzinfo is not None or end.tzinfo is not None or end < start:
        raise ValueError('start/end must be timezone-free dates with end >= start.')
    cal = xc.get_calendar('XNYS', start=start-pd.Timedelta(days=14), end=end+pd.Timedelta(days=31))
    dates = cal.sessions.strftime('%Y-%m-%d')
    result = pd.DataFrame({'date': dates, 'session_open': cal.opens.to_numpy(),
                           'session_close': cal.closes.to_numpy()})
    within = result.date.between(str(start.date()),str(end.date()))
    if not within.any():
        raise ValueError('Requested interval contains no XNYS session.')
    positions=np.flatnonzero(within)
    result = result.iloc[positions[0]-1:positions[-1]+2].reset_index(drop=True)
    for col in ['session_open', 'session_close']:
        result[col] = pd.to_datetime(result[col], utc=True).astype(str)
    return result, {'name': 'XNYS', 'package': 'exchange_calendars', 'version': xc.__version__}


def single_security(frame, required, key, value, label):
    out = frame.copy()
    core.require(out, required + [key, 'date'])
    if not pd.to_numeric(out[key], errors='raise').eq(value).all():
        raise ValueError(f'{label}: every row must belong to the requested {key}={value}.')
    out['date'] = core.dates(out.date)
    if out.date.duplicated().any():
        raise ValueError(f'{label}: duplicate dates; do not choose a vendor revision silently.')
    return out.set_index('date').sort_index()


def distribution_amounts(px, needed):
    """For constant share/price factors: D_t=C_(t-1)*(RET_t-RETX_t). No missing-to-zero."""
    core.require(px,['ret','retx'])
    ret=pd.to_numeric(px.ret,errors='raise').loc[needed]
    retx=pd.to_numeric(px.retx,errors='raise').loc[needed]
    previous=px.prc.shift(1).loc[needed]
    if not (np.isfinite(ret)&np.isfinite(retx)&(ret>-1)&(retx>-1)&np.isfinite(previous)&(previous>0)).all():
        raise ValueError('Dividend derivation needs finite RET/RETX and a preceding-session positive close; missing is not zero.')
    residual=retx-(px.loc[needed,'prc']/previous-1)
    if not np.isclose(residual,0,rtol=0,atol=1e-6).all():
        raise ValueError('RETX disagrees with raw close-to-close return despite constant CFACPR; inspect corporate actions/field mapping.')
    cash=(ret-retx)*previous
    if (cash < -1e-6).any():
        raise ValueError('Negative implied distribution; inspect RET/RETX and corporate actions.')
    cash=cash.where(cash.abs()>1e-6,0.)
    return cash,{'method':'prev_close * (ret - retx), constant CFACPR required',
        'retx_price_check_max_abs_error':float(residual.abs().max()),
        'retx_check_tolerance':1e-6,'cash_zero_tolerance_dollars':1e-6,
        'ex_distribution_sessions':int((cash>0).sum()),
        'note':'Distribution-equivalent cash; ordinary SPY cash-distribution assumption. Verify material events against distribution records. No announcement-time evidence.'}


def atm_diagnostic(surface, delta_unit, session_count):
    s=surface.copy()
    for col in ['days','delta','impl_volatility']:s[col]=pd.to_numeric(s[col],errors='raise')
    s['delta']=s.delta/(100 if delta_unit=='percent' else 1)
    s['cp_flag']=s.cp_flag.astype(str).str.upper()
    result={}
    for horizon in [30,91]:
        sides=[]
        for cp,delta in [('C',.5),('P',-.5)]:
            part=s.loc[(s.days==horizon)&(s.cp_flag==cp)&np.isclose(s.delta,delta)]
            if part.date.duplicated().any():raise ValueError('Duplicate 50-delta diagnostic nodes.')
            sides.append(part.set_index('date').impl_volatility.rename(cp))
        pairs=pd.concat(sides,axis=1).dropna()
        pairs=pairs.loc[np.isfinite(pairs).all(axis=1)&(pairs>0).all(axis=1)&(pairs<=3).all(axis=1)]
        diff=(pairs.C-pairs.P).abs()
        result[str(horizon)]={'paired_sessions':len(pairs),'pair_coverage':len(pairs)/session_count,
            'mean_abs_iv_call_minus_put':float(diff.mean()) if len(diff) else None,
            'max_abs_iv_call_minus_put':float(diff.max()) if len(diff) else None}
    return {'units':'annualized decimal IV; multiply by 100 for volatility points','maturities':result,
        'note':'50-delta call/put can imply different strikes. This is a definition diagnostic, not a put-call parity/arbitrage test; missing pairs are not imputed.'}


def build_frames(surface, reference, execution, calendar, *, start, end, secid, permno,
                 surface_time='18:00', price_delay_minutes=5, delta_unit='percent'):
    if not re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d', surface_time):
        raise ValueError('surface_time must be local New York HH:MM.')
    if not np.isfinite(price_delay_minutes) or price_delay_minutes < 0:
        raise ValueError('price_delay_minutes must be finite and nonnegative.')
    cal = core.calendar_check(calendar).reset_index(drop=True)
    signals = cal.loc[cal.date.between(str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()))].copy()
    if signals.empty:
        raise ValueError('No signal sessions.')
    positions = cal.index[cal.date.isin(signals.date)].to_numpy()
    if positions[-1]+1 >= len(cal):
        raise ValueError('Calendar must include the next execution session after the last signal.')
    entries = cal.iloc[positions+1].reset_index(drop=True)
    signals = signals.reset_index(drop=True)
    ref = single_security(reference, ['close'], 'secid', secid, 'secprd')
    px = single_security(execution, ['openprc','askhi','bidlo','prc','cfacpr','ret','retx'], 'permno', permno, 'CRSP')
    needed = sorted(set(signals.date) | set(entries.date))
    if positions[0]==0:
        raise ValueError('Calendar must include the preceding session for RET/RETX dividend derivation.')
    previous_session=cal.date.iloc[positions[0]-1]
    price_needed=[previous_session]+needed
    missing_ref = sorted(set(signals.date)-set(ref.index))
    missing_px = sorted(set(price_needed)-set(px.index))
    if missing_ref or missing_px:
        raise ValueError(f'Missing market sessions: secprd={missing_ref[:10]}, CRSP={missing_px[:10]}; no rows dropped.')
    px = px.loc[price_needed].copy()
    for col in ['openprc','askhi','bidlo','prc','cfacpr']:
        px[col] = pd.to_numeric(px[col], errors='raise')
    values = px[['openprc','askhi','bidlo','prc','cfacpr']].to_numpy(float)
    invalid = (~np.isfinite(values) | (values <= 0)).any(axis=1)
    if invalid.any():
        raise ValueError(f'CRSP invalid/nonpositive price or factor on {list(px.index[invalid])[:10]}; negative PRC is a bid/ask marker, not an executable close.')
    if not np.allclose(px.cfacpr, px.cfacpr.iloc[0], rtol=1e-10, atol=0):
        raise ValueError('CRSP CFACPR changes: raw-price returns cross a split/price adjustment. Normalize explicitly before research; builder will not silently back-adjust.')
    if ((px.askhi < px[['openprc','prc']].max(axis=1)) | (px.bidlo > px[['openprc','prc']].min(axis=1)) | (px.askhi < px.bidlo)).any():
        raise ValueError('CRSP OHLC envelope invalid; askhi/bidlo may be quote substitutes. Inspect raw records.')
    cash,dividend_audit=distribution_amounts(px,needed)
    core.require(surface, ['secid','date','days','delta','cp_flag','impl_volatility'])
    if surface.empty or not pd.to_numeric(surface.secid, errors='raise').eq(secid).all():
        raise ValueError('Surface must be nonempty and contain only requested secid.')
    surf = surface.copy(); surf['date'] = core.dates(surf.date)
    surf = surf.loc[surf.date.isin(signals.date)]
    stamp = pd.to_datetime(signals.date+' '+surface_time).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    if (stamp < signals.session_close).any():
        raise ValueError('Assumed EOD surface availability cannot precede the signal session close.')
    available = signals.session_close + pd.Timedelta(minutes=price_delay_minutes)
    asof = pd.concat([stamp, available], axis=1).max(axis=1)+pd.Timedelta(minutes=1)
    market = pd.DataFrame({'date':signals.date, 'reference_time':signals.session_close,
        'reference_price':pd.to_numeric(ref.loc[signals.date,'close'], errors='raise').to_numpy(),
        'rv_available_time':available, 'surface_available_time':stamp, 'asof_time':asof,
        'entry_session':entries.date, 'entry_time':entries.session_open,
        'entry_price':px.loc[entries.date,'openprc'].to_numpy()})
    market['cash_dividend']=cash.loc[signals.date].to_numpy()
    market['is_ex_dividend']=(market.cash_dividend>0).astype(int)
    market['entry_cash_dividend']=cash.loc[entries.date].to_numpy()
    market['entry_is_ex_dividend']=(market.entry_cash_dividend>0).astype(int)
    market['entry_dividend_available_time']=entries.session_close+pd.Timedelta(minutes=price_delay_minutes)
    for source, dest in [('openprc','open'),('askhi','high'),('bidlo','low'),('prc','close')]:
        market[dest] = px.loc[signals.date,source].to_numpy()
    metadata = {'synthetic':False, 'surface_kind':'curve', 'availability_basis':'assumed',
        'reference_instrument':f'SPX (user must verify OptionMetrics secid={secid})',
        'execution_instrument':f'SPY (user must verify CRSP permno={permno})',
        'rv_instrument':'SPY; daily OHLC variance proxy, not SPX intraday realized variance',
        'price_basis':'Raw positive CRSP daily prices; split-factor changes rejected; cash distributions included in labels and daily portfolio returns',
        'return_basis':'cash_distribution_total_return',
        'dividend_basis':dividend_audit,
        'rv_dividend_adjusted':True,
        'label_definition':'(exit_open + sum cash ex-dated strictly after entry session through exit session - entry_open) / entry_open; cash held without reinvestment',
        'label_availability':'No earlier than exit opening price AND exit-session close+price delay, because RET/RETX are ex-post daily data',
        'iv_level_definition':'50-delta call ATM proxy, annual decimal IV; 30/91 calendar days',
        'execution_price_time_basis':'Assumed fill at CRSP opening price at scheduled next XNYS open, normally 09:30 New York. Daily data do not verify opening-print timestamp or fill.',
        'availability_assumptions':{
            'surface':f'Same-date {surface_time} America/New_York; historical archive release latency and later revisions NOT verified',
            'reference_and_rv':f'Complete daily prices assumed available {price_delay_minutes} minutes after actual scheduled XNYS close',
            'asof':'One minute after the later of assumed surface/price availability',
            'execution':'Next scheduled XNYS open; no fabricated 09:31 price'},
        'limitations':['SPX surface / SPX reference / SPY RV and execution creates basis mismatch.',
            'OptionMetrics daily observations need not all refer to exactly XNYS close; this is a daily-lag design, not synchronized price discovery.',
            'CFACPR is inspected for corporate-action jumps, not used as a point-in-time signal.',
            'SPY cash distributions inferred from RET/RETX, accrued at ex-date; payment delay, tax, financing and borrow are not modeled. Not evidence of advance dividend knowledge.',
            'CRSP high/low can contain quote substitutions; positive values and envelope checks cannot authenticate trade-level OHLC.',
            'Calendar package is an independent schedule, not proof of actual source availability.']}
    daily = data.export_surface(surf,market,calendar,metadata,delta_unit)
    quality = {'sessions':len(daily), 'first_signal':daily.date.iloc[0], 'last_signal':daily.date.iloc[-1],
        'last_required_execution_session':entries.date.iloc[-1],
        'missing_surface_cells':int(daily[core.SURFACE].isna().sum().sum()),
        'complete_surface_sessions':int(daily[core.SURFACE].notna().all(axis=1).sum()),
        'missing_market_rows':0, 'price_adjustment_factor_constant':True,
        'preceding_price_session':previous_session,'dividends':dividend_audit,
        'atm_call_put_diagnostic':atm_diagnostic(surf,delta_unit,len(daily))}
    complete=daily[core.SURFACE].notna().all(axis=1)
    skew_change=(daily.iv_put25_30-daily.iv_call25_30).diff()
    quality['surface_coverage']=float(complete.mean())
    quality['skew_change_coverage_excluding_first']=float(skew_change.iloc[1:].notna().mean())
    quality['coverage_by_year']=[{'year':year,'sessions':len(g),'complete_surface_fraction':float(complete.loc[g.index].mean()),
        'skew_change_fraction':float(skew_change.loc[g.index].notna().mean())} for year,g in daily.groupby(daily.date.str[:4])]
    quality['coverage_note']='No filling/interpolation. Differencing can lose both a missing day and its next session. First-row skew change is structurally unavailable.'
    if quality['complete_surface_sessions']==0:
        raise ValueError('No complete 30/91-day surface. Check days, signed delta convention and security ID.')
    return daily,market,metadata,quality


def table_names(pattern, start, end):
    if '{year}' in pattern:
        return [data.identifier(pattern.replace('{year}',str(y))) for y in range(pd.Timestamp(start).year,pd.Timestamp(end).year+1)]
    return [data.identifier(pattern)]


def fetch(db, library, pattern, columns, key, value, start, end, log, surface=False):
    library = data.identifier(library)
    tables = set(db.list_tables(library=library)); chunks=[]
    for table in table_names(pattern,start,end):
        if table not in tables:
            raise ValueError(f'{library}.{table} absent; inspect licensed schema and set the table pattern explicitly.')
        schema = set(db.describe_table(library=library,table=table)['name'])
        if not set(columns)<=schema:
            raise ValueError(f'{library}.{table} missing {sorted(set(columns)-schema)}; inspect schema, do not guess substitute fields.')
        query = f"select {','.join(columns)} from {library}.{table} where {key}=%(security)s and date between %(start)s and %(end)s"
        if surface:
            query += ' and days in (30,91)'
        frame = db.raw_sql(query,params={'security':value,'start':start,'end':end},date_cols=['date'])
        log.append({'table':f'{library}.{table}','fields':columns,'rows':len(frame),'start':start,'end':end,'security':value})
        chunks.append(frame)
    out = pd.concat(chunks,ignore_index=True)
    if out.empty:
        raise ValueError(f'{library}.{pattern} returned no data; check identifier, dates and permissions.')
    return out


def run(args):
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is nonempty. Use a new build directory to preserve source provenance.')
    calendar, calendar_source = exchange_calendar(args.start,args.end)
    signal_dates=calendar.loc[calendar.date.between(str(pd.Timestamp(args.start).date()),str(pd.Timestamp(args.end).date())),'date']
    first,last = signal_dates.iloc[0],signal_dates.iloc[-1]
    local = [args.surface_file,args.reference_file,args.execution_file]
    if any(local) and not all(local):
        raise ValueError('Local mode requires all three: --surface-file, --reference-file, --execution-file.')
    log=[]
    if all(local):
        frames=[data.read(p) for p in local]
        source={'mode':'local_raw_files','files':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in local}}
    else:
        if not args.option_library or not args.crsp_library:
            raise ValueError('Live mode requires --option-library and --crsp-library, plus licensed WRDS access.')
        try:
            import wrds
        except ImportError as exc:
            raise ValueError('Install requirements-wrds.txt first.') from exc
        db=wrds.Connection()
        try:
            surface=fetch(db,args.option_library,args.surface_table,['secid','date','days','delta','cp_flag','impl_volatility'],
                          'secid',args.secid,first,last,log,surface=True)
            reference=fetch(db,args.option_library,args.reference_table,['secid','date','close'],
                            'secid',args.secid,first,last,log)
            execution=fetch(db,args.crsp_library,args.crsp_table,['permno','date','openprc','askhi','bidlo','prc','cfacpr','ret','retx'],
                            'permno',args.spy_permno,calendar.date.iloc[0],calendar.date.iloc[-1],log)
            frames=[surface,reference,execution]
        finally:
            db.close()
        source={'mode':'authenticated_wrds','queries':log}
    daily,market,metadata,quality=build_frames(*frames,calendar,start=first,end=last,
        secid=args.secid,permno=args.spy_permno,surface_time=args.assumed_surface_time,
        price_delay_minutes=args.assumed_price_delay,delta_unit=args.delta_unit)
    metadata['calendar_source']=calendar_source
    metadata['source']=source
    out.mkdir(parents=True,exist_ok=True)
    for name,frame in [('daily',daily),('market',market),('calendar',calendar)]:
        frame.to_csv(out/(name+'.csv'),index=False)
    raw=out/'raw'; raw.mkdir(exist_ok=True)
    for name,frame in zip(['surface_long','secprd','crsp_daily'],frames):
        frame.to_csv(raw/(name+'.csv'),index=False)
    from .research import codehash
    data.save_json(out/'metadata.json',metadata)
    data.save_json(out/'build_manifest.json',{'code_sha256':codehash(),'quality':quality,
        'calendar_source':calendar_source,'sources':source,
        'raw_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(raw.glob('*.csv'))},
        'note':'Assumed availability and opening execution. Successful validation is not point-in-time certification.'})
    print(f'Built and validated {len(daily)} sessions -> {out}; availability_basis=assumed')
    return out
