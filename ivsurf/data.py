"""Data adapters; no inferred availability times or synthetic execution timestamps."""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .core import require, dates, times, validate, SURFACE, MARKET


def read(path):
    path=Path(path)
    return pd.read_parquet(path) if path.suffix.lower()=='.parquet' else pd.read_csv(path)


def save_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def export_surface(surface, market, calendar, metadata, delta_unit='percent'):
    s=surface.copy(); m=market.copy()
    require(s,['date','days','delta','cp_flag','impl_volatility'])
    require(m,MARKET+['surface_available_time'])
    s['date']=dates(s.date); m['date']=dates(m.date)
    for col in ['days','delta','impl_volatility']:
        s[col]=pd.to_numeric(s[col],errors='raise')
    if 'secid' in s and s.secid.nunique()!=1:
        raise ValueError('Surface file must contain exactly one security.')
    if delta_unit=='percent':
        s['delta']=s.delta/100
    s['cp_flag']=s.cp_flag.astype(str).str.upper()
    if not (((s.cp_flag=='C') & s.delta.between(0,1)) |
            ((s.cp_flag=='P') & s.delta.between(-1,0))).all():
        raise ValueError('Check delta units and call/put signs.')
    out=m.copy().set_index('date')
    for name,day,cp,delta in [('iv_atm_30',30,'C',.5),('iv_atm_91',91,'C',.5),
                              ('iv_put25_30',30,'P',-.25),('iv_call25_30',30,'C',.25)]:
        part=s.loc[(s.days==day)&(s.cp_flag==cp)&np.isclose(s.delta,delta)]
        if part.date.duplicated().any():
            raise ValueError('Duplicate standardized nodes; select one security/contract methodology/version.')
        out[name]=part.set_index('date').impl_volatility.reindex(out.index)
    out=out.reset_index()
    return validate(out,calendar,metadata)


def inside(x,y,t):
    x=np.asarray(x,float); y=np.asarray(y,float)
    good=np.isfinite(x)&np.isfinite(y)
    x=x[good]; y=y[good]
    if len(x)==0:
        return np.nan
    order=np.argsort(x); x=x[order]; y=y[order]
    if t<x[0]-1e-9 or t>x[-1]+1e-9:
        return np.nan
    return float(np.interp(t,x,y))


def export_quotes(quotes,market,calendar,metadata,max_age=120,max_spread=.5):
    q=quotes.copy()
    require(q,['date','contract_id','quote_time','expiry_time','option_type','delta','iv','bid','ask'])
    require(market,MARKET+['surface_available_time'])
    q['date']=dates(q.date)
    for col in ['quote_time','expiry_time']:
        q[col]=times(q[col],col)
    for col in ['delta','iv','bid','ask']:
        q[col]=pd.to_numeric(q[col],errors='raise')
    q['option_type']=q.option_type.astype(str).str.upper()
    if not (((q.option_type=='C')&q.delta.between(0,1))|((q.option_type=='P')&q.delta.between(-1,0))).all():
        raise ValueError('Quotes require signed decimal delta.')
    grouped={d:g for d,g in q.groupby('date')}
    rows=[]; audits=[]
    for _,row in market.iterrows():
        snapshot=grouped.get(row['date'],q.iloc[:0]).copy()
        stamp=pd.Timestamp(row['reference_time'])
        if stamp.tzinfo is None:
            raise ValueError('reference_time needs a timezone.')
        snapshot=snapshot.loc[snapshot.quote_time<=stamp]
        snapshot=snapshot.sort_values('quote_time').drop_duplicates('contract_id',keep='last')
        before=len(snapshot)
        age=(stamp-snapshot.quote_time).dt.total_seconds()
        snapshot['days']=(snapshot.expiry_time-stamp).dt.total_seconds()/86400
        spread=(snapshot.ask-snapshot.bid)/((snapshot.ask+snapshot.bid)/2)
        snapshot=snapshot.loc[age.between(0,max_age)&(snapshot.bid>0)&(snapshot.ask>=snapshot.bid)&
                              (spread<=max_spread)&snapshot.iv.between(.0001,3)&snapshot.days.between(7,200)]
        nodes=[]
        for _,part in snapshot.groupby('expiry_time'):
            node={'days':float(part.days.iloc[0])}
            for name,cp,delta in [('atm','C',.5),('put25','P',.25),('call25','C',.25)]:
                side=part.loc[part.option_type==cp].copy()
                side['abs_delta']=side.delta.abs()
                side=side.groupby('abs_delta',as_index=False).iv.median()
                node[name]=inside(side.abs_delta,side.iv,delta)
            nodes.append(node)
        nodes=pd.DataFrame(nodes)
        output=row.to_dict()
        for col,key,days in [('iv_atm_30','atm',30),('iv_atm_91','atm',91),
                             ('iv_put25_30','put25',30),('iv_call25_30','call25',30)]:
            variance=inside(nodes.days,nodes[key]**2*nodes.days/365,days) if len(nodes) else np.nan
            output[col]=np.sqrt(variance/(days/365)) if np.isfinite(variance) else np.nan
        rows.append(output)
        audits.append({'date':row['date'],'snapshots':before,'accepted':len(snapshot),
                       'complete_surface':all(np.isfinite(output[col]) for col in SURFACE)})
    return validate(pd.DataFrame(rows),calendar,metadata),pd.DataFrame(audits)


def export_proxies(directory,market,calendar,metadata):
    """Read local Cboe CSVs. SPY/ES execution data always supplied by the user."""
    out=market.copy().set_index('date')
    require(market,MARKET+['surface_available_time'])
    for source,dest in [('VIX','vix_30'),('VIX3M','vix_3m'),('SKEW','skew_index'),('VVIX','vvix')]:
        d=pd.read_csv(Path(directory)/(source+'.csv'))
        d.columns=[c.strip().lower() for c in d.columns]
        require(d,['date'])
        d['date']=pd.to_datetime(d.date).dt.strftime('%Y-%m-%d')
        if 'close' in d:
            value='close'
        elif source.lower() in d:
            value=source.lower()
        else:
            raise ValueError(f'{source}: explicitly normalize the value field to close; no guessing of last column.')
        if d.date.duplicated().any():
            raise ValueError(f'Duplicate {source} dates.')
        out[dest]=d.set_index('date')[value].reindex(out.index)
    return validate(out.reset_index(),calendar,metadata)


def demo(directory,rows=2200,seed=42,strength=.25):
    if rows<100:
        raise ValueError('Demo needs >=100 rows.')
    out=Path(directory); out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    days=pd.bdate_range('2015-01-05',periods=rows+3)
    op=(days+pd.Timedelta(hours=9,minutes=30)).tz_localize('America/New_York')
    cl=(days+pd.Timedelta(hours=16)).tz_localize('America/New_York')
    calendar=pd.DataFrame({'date':days.strftime('%Y-%m-%d'),'session_open':op.astype(str),'session_close':cl.astype(str)})
    signal=rng.normal(size=len(days))
    skew=.05+.012*np.tanh(signal)
    change=np.r_[0.,np.diff(skew)]
    returns=rng.normal(.0002,.009,len(days))-strength*np.r_[0.,change[:-1]]
    opening=np.r_[300.,300*np.cumprod(1+returns[:-1])]
    close=opening*(1+rng.normal(0,.003,len(days)))
    high=np.maximum(opening,close)*(1+rng.uniform(.001,.005,len(days)))
    low=np.minimum(opening,close)*(1-rng.uniform(.001,.005,len(days)))
    entry_price=opening*(1+rng.normal(0,.0001,len(days)))
    iv=np.clip(.21+.02*np.sin(np.arange(len(days))/90)+rng.normal(0,.008,len(days)),.1,.4)
    market=pd.DataFrame({'date':calendar.date.iloc[:rows].to_numpy(),
                         'reference_time':cl[:rows].astype(str),'asof_time':(cl[:rows]+pd.Timedelta(minutes=30)).astype(str),
                         'surface_available_time':(cl[:rows]+pd.Timedelta(minutes=20)).astype(str),
                         'reference_price':close[:rows],'rv_available_time':(cl[:rows]+pd.Timedelta(minutes=5)).astype(str),
                         'open':opening[:rows],'high':high[:rows],'low':low[:rows],'close':close[:rows],
                         'entry_session':calendar.date.iloc[1:rows+1].to_numpy(),
                         'entry_time':(op[1:rows+1]+pd.Timedelta(minutes=1)).astype(str),'entry_price':entry_price[1:rows+1]})
    d=market.copy()
    d['iv_atm_30']=iv[:rows]; d['iv_atm_91']=iv[:rows]+.015
    d['iv_put25_30']=iv[:rows]+skew[:rows]*.65
    d['iv_call25_30']=iv[:rows]-skew[:rows]*.35
    meta={'synthetic':True,'surface_kind':'curve','availability_basis':'synthetic',
          'reference_instrument':'SYNTHETIC','execution_instrument':'SYNTHETIC','price_basis':'synthetic unadjusted',
          'iv_level_definition':'50-delta call proxy, annualized decimal; constant calendar maturities 30/91',
          'execution_price_time_basis':'Synthetic price assigned to next session 09:31, not historical opening price',
          'calendar_source':'Synthetic weekdays; NOT an exchange calendar','signal_strength':strength,'seed':seed}
    d.to_csv(out/'daily.csv',index=False); market.to_csv(out/'market.csv',index=False)
    calendar.to_csv(out/'calendar.csv',index=False); save_json(out/'metadata.json',meta)
    nodes=[]
    for _,row in d.iterrows():
        for day in [30,91]:
            level=row[f'iv_atm_{day}']
            for cp,delta,value in [('C',50,level),('C',25,level-(row.iv_atm_30-row.iv_call25_30)),
                                   ('P',-25,level+(row.iv_put25_30-row.iv_atm_30))]:
                nodes.append({'date':row.date,'days':day,'cp_flag':cp,'delta':delta,'impl_volatility':value})
    pd.DataFrame(nodes).to_csv(out/'surface_long.csv',index=False)
    # Quote adapter smoke input only; dummy option prices are filter inputs.
    quotes=[]
    for _,row in d.head(35).iterrows():
        for day in [20,40,80,100]:
            for cp,sign in [('C',1),('P',-1)]:
                for delta in [.15,.25,.5,.65]:
                    vol=row.iv_atm_30+.015*(day-30)/61+(1 if cp=='P' else -1)*.02*(.5-delta)/.25
                    quotes.append({'date':row.date,'contract_id':f'{day}_{cp}_{delta}',
                                   'quote_time':row.reference_time,'expiry_time':str(pd.Timestamp(row.reference_time)+pd.Timedelta(days=day)),
                                   'option_type':cp,'delta':delta*sign,'iv':vol,'bid':9.9,'ask':10.1})
    pd.DataFrame(quotes).to_csv(out/'quotes_example.csv',index=False)
    market.head(35).to_csv(out/'market_example.csv',index=False)
    # Events are labels only, assigned beforehand; no event-specific data-generating effect.
    ev=[]
    for i in range(100,rows-2,21):
        ev.append({'event_id':f'synthetic_{i}','event_type':'SYNTHETIC_RELEASE','event_time':str(op[i+1]+pd.Timedelta(hours=1))})
    pd.DataFrame(ev).to_csv(out/'events.csv',index=False)
    chain=[]
    for idx,row in market.tail(min(1000,rows)).iterrows():
        stamp=pd.Timestamp(row.reference_time)
        for cp,k in [('P',.95),('C',1.05)]:
            chain.append({'date':row.date,'snapshot_time':str(stamp),'available_time':str(stamp+pd.Timedelta(minutes=10)),
                          'expiry_time':str(stamp+pd.Timedelta(days=30)),
                          'spot':row.reference_price,'strike':row.reference_price*k,'option_type':cp,'iv':.2,
                          'open_interest':(1600 if idx%2 else 500) if cp=='C' else 1000,
                          'multiplier':100,'rate':.03,'dividend_yield':.01})
    pd.DataFrame(chain).to_csv(out/'chain_example.csv',index=False)
    (out/'DEMO_ONLY.txt').write_text('SYNTHETIC DATA. Artificial signal; weekday-only calendar; dummy quote prices. No market evidence.\n')
    return out


def identifier(value):
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',value):
        raise ValueError(f'Invalid SQL identifier: {value}')
    return value


def wrds_download(args):
    try:
        import wrds
    except ImportError as exc:
        raise ValueError('Install requirements-wrds.txt in a virtual environment first.') from exc
    library=identifier(args.option_library)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    start=pd.Timestamp(args.start); end=pd.Timestamp(args.end)
    if end<start:
        raise ValueError('end must follow start')
    db=wrds.Connection()
    log=[]
    try:
        tables=set(db.list_tables(library=library))
        for year in range(start.year,end.year+1):
            name=f'vsurfd{year}'
            if name not in tables:
                raise ValueError(f'{library}.{name} absent; verify licensed schema. No guessed substitution.')
            schema=db.describe_table(library=library,table=name)
            cols=set(schema['name'])
            required={'secid','date','days','delta','cp_flag','impl_volatility'}
            if not required<=cols:
                raise ValueError(f'{name} missing expected columns {sorted(required-cols)}; inspect schema and normalize explicitly.')
            query=f'''select secid,date,days,delta,cp_flag,impl_volatility from {library}.{name}
                      where secid=%(secid)s and date between %(start)s and %(end)s
                      and days in (30,91)'''
            part=db.raw_sql(query,params={'secid':args.secid,'start':str(start.date()),'end':str(end.date())},date_cols=['date'])
            if part.empty:
                raise ValueError(f'{name} returned no data. Verify security identifier and dates.')
            path=out/f'surface_{year}.csv'; part.to_csv(path,index=False)
            log.append({'file':path.name,'rows':len(part)})
        joined=pd.concat([pd.read_csv(out/item['file']) for item in log],ignore_index=True)
        joined.to_csv(out/'surface_long.csv',index=False)
        save_json(out/'download_manifest.json',{'library':library,'secid':args.secid,'start':args.start,'end':args.end,'files':log,
                  'note':'Raw surface only. Availability, session calendar and executable prices must be supplied separately.'})
    finally:
        db.close()
    return out
