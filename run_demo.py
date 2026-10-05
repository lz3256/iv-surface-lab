"""One-command, offline synthetic end-to-end demonstration. Uses current Python."""
import argparse
import datetime
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--out',default='runs/demo_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
ap.add_argument('--rows',type=int,default=2200)
args=ap.parse_args()
dest=Path(args.out)
if not dest.is_absolute():dest=root/dest
if dest.exists():raise SystemExit('Output exists; choose a new --out directory.')

def call(*argv):
    subprocess.run([sys.executable,'-m','ivsurf',*map(str,argv)],cwd=root,check=True)

data=dest/'data'
call('demo','--out',data,'--rows',args.rows)
call('export','--surface',data/'surface_long.csv','--market',data/'market.csv',
     '--calendar',data/'calendar.csv','--metadata',data/'metadata.json','--out',data/'roundtrip.csv')
call('run','--input',data/'roundtrip.csv','--calendar',data/'calendar.csv','--metadata',data/'metadata.json',
     '--events',data/'events.csv','--out',dest/'research')
call('gex','--chain',data/'chain_example.csv','--daily',data/'roundtrip.csv','--calendar',data/'calendar.csv',
     '--metadata',data/'metadata.json','--position-mode','heuristic','--out',dest/'gex')
print('\nOpen:',dest/'research/report.html')
print('All results are SYNTHETIC. No real market data were fetched.')
