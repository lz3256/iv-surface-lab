"""Hand-crafted raw-price fixtures; no licensed data and no market performance evidence."""
import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from ivsurf import core, research, wrds_build as wb


def fixture():
    # First row supplies prior close; last row is execution-only.
    days=['2024-11-25','2024-11-26','2024-11-27','2024-11-29','2024-12-02']
    cal=pd.DataFrame({'date':days,
        'session_open':[d+'T09:30:00-05:00' for d in days],
        'session_close':[d+('T13:00:00-05:00' if d=='2024-11-29' else 'T16:00:00-05:00') for d in days]})
    ref=pd.DataFrame({'secid':[108105]*3,'date':days[1:4],'close':[5000.,5010.,5020.]})
    px=pd.DataFrame({'permno':[84398]*5,'date':days,'openprc':[499.,500.,501.,502.,503.],
        'prc':[499.5,500.5,501.5,502.5,503.5],'askhi':[501.,502.,503.,504.,505.],
        'bidlo':[498.,499.,500.,501.,502.],'cfacpr':[1.]*5})
    px['retx']=px.prc/px.prc.shift(1)-1
    px['ret']=px.retx
    surface=pd.DataFrame([{'secid':108105,'date':d,'days':h,'cp_flag':cp,'delta':delta,'impl_volatility':v}
        for d in days[1:4] for h,cp,delta,v in [(30,'C',50,.2),(91,'C',50,.22),(30,'P',-25,.24),(30,'C',25,.19)]])
    return surface,ref,px,cal


def build(frames=None,**kwargs):
    return wb.build_frames(*(fixture() if frames is None else frames),start='2024-11-26',end='2024-11-29',
        secid=108105,permno=84398,**kwargs)


class BuilderTests(unittest.TestCase):
    def test_early_close_next_open_and_explicit_assumptions(self):
        daily,market,meta,q=build()
        self.assertEqual(daily.entry_session.tolist(),['2024-11-27','2024-11-29','2024-12-02'])
        self.assertEqual(daily.entry_price.tolist(),[501.,502.,503.])
        self.assertEqual(daily.rv_available_time.iloc[-1],pd.Timestamp('2024-11-29T13:05:00-05:00'))
        self.assertEqual(daily.surface_available_time.iloc[-1],pd.Timestamp('2024-11-29T18:00:00-05:00'))
        self.assertTrue((daily.entry_time.dt.tz_convert('America/New_York').dt.minute==30).all())
        self.assertEqual(meta['availability_basis'],'assumed')
        self.assertEqual(q['missing_surface_cells'],0)

    def test_next_open_label_matches_execution(self):
        daily,*_=build()
        y,end=core.target(daily,'return',1)
        self.assertAlmostEqual(y.iloc[0],502/501-1)
        self.assertEqual(end.iloc[0],pd.Timestamp('2024-11-29T13:05:00-05:00'))

    def test_missing_price_is_not_deleted(self):
        s,r,p,c=fixture()
        with self.assertRaisesRegex(ValueError,'Missing market sessions'):
            build((s,r,p.drop(index=2),c))

    def test_missing_terminal_execution_rejected(self):
        s,r,p,c=fixture()
        with self.assertRaisesRegex(ValueError,'Missing market sessions'):
            build((s,r,p.iloc[:-1],c))

    def test_negative_close_is_not_abs_converted(self):
        s,r,p,c=fixture();p.loc[1,'prc']=-501.5
        with self.assertRaisesRegex(ValueError,'bid/ask marker'):build((s,r,p,c))

    def test_price_adjustment_jump_rejected(self):
        s,r,p,c=fixture();p.loc[1,'cfacpr']=2.
        with self.assertRaisesRegex(ValueError,'CFACPR changes'):build((s,r,p,c))

    def test_wrong_security_and_duplicate_reference_rejected(self):
        s,r,p,c=fixture();p.loc[1,'permno']=123
        with self.assertRaisesRegex(ValueError,'requested permno'):build((s,r,p,c))
        s,r,p,c=fixture()
        with self.assertRaisesRegex(ValueError,'duplicate dates'):build((s,pd.concat([r,r.iloc[:1]]),p,c))

    def test_missing_surface_keeps_session(self):
        s,r,p,c=fixture();s=s.loc[s.date!='2024-11-27']
        d,_,_,q=build((s,r,p,c))
        self.assertEqual(len(d),3)
        self.assertTrue(d.loc[1,core.SURFACE].isna().all())
        self.assertEqual(q['missing_surface_cells'],4)

    def test_unavailable_prices_cannot_trade_next_open(self):
        with self.assertRaisesRegex(ValueError,'strictly after'):
            build(price_delay_minutes=24*60)

    def test_eod_surface_cannot_be_available_in_the_morning(self):
        with self.assertRaisesRegex(ValueError,'cannot precede'):
            build(surface_time='09:00')

    def test_local_command_emits_auditable_bundle(self):
        s,r,p,c=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,frame in [('surface',s),('reference',r),('execution',p)]:frame.to_csv(root/(name+'.csv'),index=False)
            args=argparse.Namespace(start='2024-11-26',end='2024-11-29',out=str(root/'out'),secid=108105,
                spy_permno=84398,surface_file=str(root/'surface.csv'),reference_file=str(root/'reference.csv'),
                execution_file=str(root/'execution.csv'),assumed_surface_time='18:00',assumed_price_delay=5,delta_unit='percent')
            with patch.object(wb,'exchange_calendar',return_value=(c,{'name':'TEST_FIXTURE'})),contextlib.redirect_stdout(io.StringIO()):
                wb.run(args)
            meta=json.loads((root/'out/metadata.json').read_text())
            manifest=json.loads((root/'out/build_manifest.json').read_text())
            self.assertEqual(len(manifest['raw_sha256']),3)
            self.assertEqual(meta['source']['mode'],'local_raw_files')
            core.validate(pd.read_csv(root/'out/daily.csv'),pd.read_csv(root/'out/calendar.csv'),meta)
            with self.assertRaisesRegex(ValueError,'nonempty'):wb.run(args)

    def test_live_mock_queries_three_sources_and_closes(self):
        s,r,p,c=fixture();owner=self
        class DB:
            closed=False
            def list_tables(self,library):return ['vsurfd2024','secprd2024'] if library=='optionm' else ['dsf']
            def describe_table(self,library,table):
                return pd.DataFrame({'name':list((s if table.startswith('vsurfd') else r if table.startswith('secprd') else p).columns)})
            def raw_sql(self,query,params,date_cols):
                owner.assertIn('%(security)s',query)
                if 'crsp.dsf' in query:
                    owner.assertEqual(params['end'],'2024-12-02')
                    owner.assertEqual(params['start'],'2024-11-25')
                    owner.assertIn('ret,retx',query)
                    return p
                owner.assertEqual(params['end'],'2024-11-29')
                return s if 'vsurfd' in query else r
            def close(self):self.closed=True
        db=DB()
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(start='2024-11-26',end='2024-11-29',out=tmp,secid=108105,spy_permno=84398,
                surface_file=None,reference_file=None,execution_file=None,option_library='optionm',crsp_library='crsp',
                surface_table='vsurfd{year}',reference_table='secprd{year}',crsp_table='dsf',
                assumed_surface_time='18:00',assumed_price_delay=5,delta_unit='percent')
            with patch.object(wb,'exchange_calendar',return_value=(c,{'name':'TEST_FIXTURE'})),patch.dict('sys.modules',{'wrds':SimpleNamespace(Connection=lambda:db)}),contextlib.redirect_stdout(io.StringIO()):
                wb.run(args)
            self.assertTrue(db.closed)
            meta=json.loads((Path(tmp)/'metadata.json').read_text())
            self.assertEqual(len(meta['source']['queries']),3)

    def test_sql_identifier_and_schema_guard(self):
        with self.assertRaisesRegex(ValueError,'Invalid SQL identifier'):wb.table_names('dsf;delete','2024-01-01','2024-12-31')
        db=SimpleNamespace(list_tables=lambda **k:['secprd'],describe_table=lambda **k:pd.DataFrame({'name':['date','secid']}))
        with self.assertRaisesRegex(ValueError,'missing'):
            wb.fetch(db,'optionm','secprd',['date','secid','close'],'secid',108105,'2024-01-01','2024-01-31',[])


@unittest.skipUnless(importlib.util.find_spec('exchange_calendars'),'optional exchange_calendars dependency not installed')
class RealCalendarTests(unittest.TestCase):
    def test_holiday_half_day_and_execution_buffer(self):
        cal,source=wb.exchange_calendar('2024-11-25','2024-11-29')
        self.assertNotIn('2024-11-28',cal.date.tolist())
        self.assertEqual(cal.date.iloc[-1],'2024-12-02')
        close=pd.Timestamp(cal.set_index('date').loc['2024-11-29','session_close']).tz_convert('America/New_York')
        self.assertEqual(close.hour,13)
        self.assertEqual(source['name'],'XNYS')

    def test_dst_and_exceptional_closures(self):
        cal,_=wb.exchange_calendar('2024-03-08','2024-03-11')
        self.assertEqual(pd.Timestamp(cal.set_index('date').loc['2024-03-08','session_open']).hour,14)
        self.assertEqual(pd.Timestamp(cal.set_index('date').loc['2024-03-11','session_open']).hour,13)
        cal,_=wb.exchange_calendar('2012-10-26','2012-10-31')
        self.assertNotIn('2012-10-29',cal.date.tolist())
        self.assertNotIn('2012-10-30',cal.date.tolist())


class InferenceFamilyTests(unittest.TestCase):
    def comparisons(self):
        return pd.DataFrame({'family':['H2','H1','H2'],'horizon':[1,5,21],
            'baseline':['R2_HAR','V2_IV','R2_HAR'],'added':['R3_skew_change','V3_surface','R3_skew_change'],
            'p':[.04,.02,.04]})

    def test_unique_primary_raw_aux_holm_and_global_sensitivity(self):
        out=research.inference_families(self.comparisons(),'curve')
        self.assertEqual(out.test_role.tolist(),['primary','auxiliary','auxiliary'])
        self.assertEqual(out.p_primary.iloc[0],.04)
        self.assertTrue(np.isnan(out.p_holm_aux.iloc[0]))
        np.testing.assert_allclose(out.p_holm_aux.iloc[1:],[.04,.04])
        np.testing.assert_allclose(out.p_holm_all_sensitivity,core.holm(out.p))

    def test_proxy_cannot_claim_original_primary(self):
        out=research.inference_families(self.comparisons(),'proxies')
        self.assertTrue((out.test_role=='auxiliary').all())
        self.assertTrue(out.p_primary.isna().all())

    def test_absent_h1_horizon_does_not_reassign_primary(self):
        out=research.inference_families(self.comparisons().iloc[1:],'curve')
        self.assertTrue((out.test_role=='auxiliary').all())


if __name__=='__main__':unittest.main()
