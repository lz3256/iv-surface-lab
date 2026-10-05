import argparse
import json
from pathlib import Path
from . import data,core,research,gex,wrds_build


def export_command(args):
    dest=Path(args.out)
    if dest.exists():
        raise ValueError('Export destination exists; choose a new filename.')
    meta=json.loads(Path(args.metadata).read_text())
    market=data.read(args.market); calendar=data.read(args.calendar)
    if args.command=='export':
        if meta['surface_kind']!='curve':
            raise ValueError('Surface export requires curve metadata.')
        output=data.export_surface(data.read(args.surface),market,calendar,meta,args.delta_unit)
    elif args.command=='quotes':
        if meta['surface_kind']!='curve':
            raise ValueError('Quote export requires curve metadata.')
        output,audit=data.export_quotes(data.read(args.quotes),market,calendar,meta,args.max_age,args.max_spread)
        dest=Path(args.out); dest.parent.mkdir(parents=True,exist_ok=True)
        audit.to_csv(dest.with_suffix('.quality.csv'),index=False)
    else:
        if meta['surface_kind']!='proxies':
            raise ValueError('Cboe indices require surface_kind=proxies; not a 25-delta curve.')
        output=data.export_proxies(args.directory,market,calendar,meta)
    dest=Path(args.out); dest.parent.mkdir(parents=True,exist_ok=True)
    output.to_csv(dest,index=False)
    hashes={key:research.filehash(getattr(args,key)) for key in ['market','calendar','metadata']}
    for key in ['surface','quotes']:
        if getattr(args,key,None):hashes[key]=research.filehash(getattr(args,key))
    if getattr(args,'directory',None):
        hashes['proxy_files']={p.name:research.filehash(p) for p in sorted(Path(args.directory).glob('*.csv'))}
    data.save_json(dest.with_suffix('.manifest.json'),{
        'metadata':meta,'rows':len(output),'input_hashes':hashes,
        'note':'Availability supplied explicitly. No date-to-time fabrication; no market rows deleted.'})
    print(f'Validated {len(output)} rows -> {dest}')


def check(args):
    meta=json.loads(Path(args.metadata).read_text())
    d=core.validate(data.read(args.input),data.read(args.calendar),meta)
    print(json.dumps({'rows':len(d),'start':d.date.iloc[0],'end':d.date.iloc[-1],
                      'synthetic':meta['synthetic'],'availability_basis':meta['availability_basis'],
                      'missing_surface_cells':int(d[core.SURFACE].isna().sum().sum()) if meta['surface_kind']=='curve' else None},indent=2))


def main():
    ap=argparse.ArgumentParser(description='Unified IV-surface research: data -> HAR -> H1/H2 -> audit; GEX exploratory.')
    sub=ap.add_subparsers(dest='command',required=True)
    d=sub.add_parser('demo',help='Synthetic full pipeline inputs; never real market evidence.')
    d.add_argument('--out',default='data/demo'); d.add_argument('--rows',type=int,default=2200)
    d.add_argument('--seed',type=int,default=42); d.add_argument('--signal-strength',type=float,default=.25)
    d.set_defaults(func=lambda a:print(data.demo(a.out,a.rows,a.seed,a.signal_strength)))
    for name in ['export','quotes','cboe']:
        p=sub.add_parser(name,help='Normalize supplied data; explicitly supplied timing and calendar required.')
        for flag in ['market','calendar','metadata','out']:
            p.add_argument('--'+flag,required=True)
        if name=='export':
            p.add_argument('--surface',required=True); p.add_argument('--delta-unit',choices=['percent','decimal'],default='percent')
        elif name=='quotes':
            p.add_argument('--quotes',required=True); p.add_argument('--max-age',type=float,default=120)
            p.add_argument('--max-spread',type=float,default=.5)
        else:
            p.add_argument('--directory',required=True)
        p.set_defaults(func=export_command)
    p=sub.add_parser('validate',help='Check units, market gaps and information/execution times.')
    for flag in ['input','calendar','metadata']:
        p.add_argument('--'+flag,required=True)
    p.set_defaults(func=check)
    p=sub.add_parser('run',help='Unified H1/H2, HAR/VRP controls, OOS inference and cost diagnostics.')
    for flag in ['input','calendar','metadata','out']:
        p.add_argument('--'+flag,required=True)
    p.add_argument('--phase',choices=['development','final'],default='development')
    p.add_argument('--holdout',type=int,default=252); p.add_argument('--min-train',type=int,default=504)
    p.add_argument('--har-min',type=int,default=252); p.add_argument('--window',type=int,default=1008)
    p.add_argument('--refit',type=int,default=21); p.add_argument('--alpha',type=float,default=10.)
    p.add_argument('--return-horizons',type=int,nargs='+',default=[1,5,21])
    p.add_argument('--vol-horizons',type=int,nargs='+',default=[5,21])
    p.add_argument('--bootstrap',type=int,default=1000); p.add_argument('--block',type=int,default=20)
    p.add_argument('--cost-bps',type=float,default=1.); p.add_argument('--events')
    p.add_argument('--exclude-ex-dividend',action='store_true',help='Retrospective sensitivity: mask H2 labels touching an ex-date from entry through exit; keep session rows.')
    p.add_argument('--study-role',choices=['primary','sensitivity'],default='primary')
    p.set_defaults(func=research.run)
    p=sub.add_parser('gex',help='Exploratory gamma positions; default leaves final holdout untouched.')
    for flag in ['chain','daily','calendar','metadata','out']:
        p.add_argument('--'+flag,required=True)
    p.add_argument('--position-mode',choices=['heuristic','dealer'],required=True)
    p.add_argument('--holdout',type=int,default=252); p.add_argument('--grid-width',type=float,default=.15)
    p.add_argument('--grid-n',type=int,default=201); p.set_defaults(func=gex.run)
    p=sub.add_parser('wrds-download',help='Optional authenticated raw surface download with schema checks.')
    p.add_argument('--option-library',required=True); p.add_argument('--secid',required=True,type=int)
    p.add_argument('--start',required=True); p.add_argument('--end',required=True); p.add_argument('--out',required=True)
    p.set_defaults(func=lambda a:print(data.wrds_download(a)))
    p=sub.add_parser('build-wrds',help='Build audited daily/calendar/metadata from WRDS or three local raw files; availability explicitly assumed.')
    for flag in ['start','end','out']:
        p.add_argument('--'+flag,required=True)
    p.add_argument('--secid',required=True,type=int)
    p.add_argument('--spy-permno',required=True,type=int)
    p.add_argument('--option-library'); p.add_argument('--crsp-library')
    p.add_argument('--surface-table',default='vsurfd{year}')
    p.add_argument('--reference-table',default='secprd{year}')
    p.add_argument('--crsp-table',default='dsf')
    for flag in ['surface-file','reference-file','execution-file']:
        p.add_argument('--'+flag)
    p.add_argument('--assumed-surface-time',default='18:00',help='Assumed same-date New York availability HH:MM, never vendor-verified.')
    p.add_argument('--assumed-price-delay',type=float,default=5.,help='Minutes after actual XNYS close when daily prices are assumed known.')
    p.add_argument('--delta-unit',choices=['percent','decimal'],default='percent')
    p.set_defaults(func=wrds_build.run)
    p=sub.add_parser('plots',help='Optional static PNG diagnostics, requires matplotlib.')
    p.add_argument('--report-dir',required=True); p.set_defaults(func=research.plots)
    args=ap.parse_args()
    try:
        args.func(args)
    except (ValueError,KeyError,OSError) as e:
        ap.exit(2,f'Error: {e}\n')


if __name__=='__main__':
    main()
