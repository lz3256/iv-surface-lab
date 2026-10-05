import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
from ivsurf import core,data,research,gex


class UnifiedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)
        data.demo(cls.root/'data',rows=650)
        cls.raw=data.read(cls.root/'data/daily.csv')
        cls.cal=data.read(cls.root/'data/calendar.csv')
        cls.meta=json.loads((cls.root/'data/metadata.json').read_text())
        cls.daily=core.validate(cls.raw,cls.cal,cls.meta)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_calendar_gap_rejected(self):
        with self.assertRaisesRegex(ValueError,'Missing/extra'):
            core.validate(self.raw.drop(index=100),self.cal,self.meta)

    def test_missing_execution_price_rejected(self):
        bad=self.raw.copy(); bad.loc[100,'entry_price']=np.nan
        with self.assertRaisesRegex(ValueError,'do not drop'):
            core.validate(bad,self.cal,self.meta)

    def test_entry_session_gap_rejected(self):
        bad=self.raw.copy(); bad.loc[100,'entry_session']=bad.loc[102,'entry_session']
        with self.assertRaises(ValueError):core.validate(bad,self.cal,self.meta)

    def test_equal_time_execution_rejected(self):
        bad=self.raw.copy(); bad['entry_time']=bad.asof_time
        with self.assertRaisesRegex(ValueError,'strictly after'):core.validate(bad,self.cal,self.meta)

    def test_naive_time_rejected(self):
        bad=self.raw.copy(); bad['asof_time']='2015-01-05 18:00:00'
        with self.assertRaisesRegex(ValueError,'timezone'):core.validate(bad,self.cal,self.meta)

    def test_unknown_feature_availability_rejected(self):
        bad=self.raw.copy(); bad['surface_available_time']=bad.entry_time
        with self.assertRaisesRegex(ValueError,'Surface unavailable'):core.validate(bad,self.cal,self.meta)

    def test_intraday_ohlc_leak_rejected(self):
        bad=self.raw.copy(); bad['rv_available_time']=bad.reference_time
        bad.loc[0,'rv_available_time']='2015-01-05T15:00:00-05:00'
        with self.assertRaisesRegex(ValueError,'complete-session'):core.validate(bad,self.cal,self.meta)

    def test_execution_outside_session_rejected(self):
        bad=self.raw.copy()
        bad['entry_time']=(pd.to_datetime(bad.entry_time,utc=True)-pd.Timedelta(hours=1)).astype(str)
        with self.assertRaisesRegex(ValueError,'outside'):core.validate(bad,self.cal,self.meta)

    def test_ohlc_invalid_rejected(self):
        bad=self.raw.copy(); bad.loc[3,'high']=bad.loc[3,'low']*.5
        with self.assertRaisesRegex(ValueError,'OHLC'):core.validate(bad,self.cal,self.meta)

    def test_overnight_variance(self):
        d=self.daily.head(3).copy()
        for name in ['open','high','low','close']:d[name]=[100.,110.,110.]
        f=core.features(d,'curve')
        self.assertAlmostEqual(f.rv_daily.iloc[1],np.log(1.1)**2)

    def test_forward_targets_alignment(self):
        p=core.features(self.daily,'curve')
        y,end=core.target(p,'return',5)
        self.assertAlmostEqual(y.iloc[20],p.entry_price.iloc[25]/p.entry_price.iloc[20]-1)
        self.assertEqual(end.iloc[20],p.entry_time.iloc[25])
        v,endv=core.target(p,'vol',5)
        self.assertAlmostEqual(v.iloc[20],np.log(p.rv_daily.iloc[21:26].mean()*252))
        self.assertEqual(endv.iloc[20],p.rv_available_time.iloc[25])

    def test_missing_surface_keeps_target(self):
        d=self.daily.copy();d.loc[21,'iv_put25_30']=np.nan
        p=core.features(d,'curve'); y,end=core.target(p,'return',1)
        self.assertAlmostEqual(y.iloc[20],d.entry_price.iloc[21]/d.entry_price.iloc[20]-1)
        self.assertTrue(np.isnan(p.skew_change.iloc[21]))

    def test_walk_label_poison_no_leak(self):
        p=core.features(self.daily.head(300),'curve'); y,end=core.target(p,'return',5)
        s={'base':[],'full':['ret_1','skew_change']}
        a=core.walk(p,y,end,s,40,100,5)
        poison=y.copy(); poison.loc[end>=p.asof_time.iloc[180]]=1e5
        b=core.walk(p,poison,end,s,40,100,5)
        np.testing.assert_allclose(a.full.iloc[:181],b.full.iloc[:181],equal_nan=True)
        good=a.full.notna()
        self.assertTrue((a.loc[good,'train_end_max']<a.loc[good,'asof_time']).all())

    def test_har_no_future_rv_leak(self):
        p=core.features(self.daily.head(300),'curve')
        a,aa=core.add_har(p,40,100,5)
        changed=p.copy(); changed.loc[180:,'rv_daily']=2.
        b,bb=core.add_har(changed,40,100,5)
        np.testing.assert_allclose(a.har_fwd_21.iloc[:180],b.har_fwd_21.iloc[:180],equal_nan=True)
        self.assertTrue((a.har_fwd_21.dropna()>0).all())
        self.assertTrue((aa.loc[aa.har_fwd_21.notna(),'train_end_max']<aa.loc[aa.har_fwd_21.notna(),'asof_time']).all())

    def test_total_variance_interp(self):
        total=data.inside([20,40],[.2**2*20/365,.3**2*40/365],30)
        self.assertAlmostEqual(total,(.2**2*20+.3**2*40)/2/365)
        self.assertTrue(np.isnan(data.inside([20,40],[.2,.3],91)))

    def test_standardized_export_roundtrip_and_91(self):
        s=data.read(self.root/'data/surface_long.csv');m=data.read(self.root/'data/market.csv')
        result=data.export_surface(s,m,self.cal,self.meta)
        np.testing.assert_allclose(result[core.SURFACE],self.daily[core.SURFACE])
        self.assertNotIn('iv_atm_90',result)

    def test_export_missing_market_price_no_drop(self):
        s=data.read(self.root/'data/surface_long.csv');m=data.read(self.root/'data/market.csv')
        m.loc[10,'entry_price']=np.nan
        with self.assertRaisesRegex(ValueError,'do not drop'):data.export_surface(s,m,self.cal,self.meta)

    def test_quote_filter_future_and_contract_snapshot(self):
        q=data.read(self.root/'data/quotes_example.csv');m=data.read(self.root/'data/market_example.csv')
        first=q.date.iloc[0]
        q.loc[q.date==first,'quote_time']=(pd.to_datetime(q.loc[q.date==first,'quote_time'],utc=True)+pd.Timedelta(seconds=1)).astype(str)
        result,audit=data.export_quotes(q,m,self.cal,self.meta)
        self.assertTrue(result.loc[0,core.SURFACE].isna().all())
        self.assertTrue(result.loc[1,core.SURFACE].notna().all())
        self.assertEqual(len(result),len(m))

    def test_one_session_cost_and_final_liquidation(self):
        _,p=core.portfolio(np.array([1.,1.,-1.,0.]),np.ones(4)*100,1,1.)
        np.testing.assert_allclose(p.turnover,[1,0,3])
        np.testing.assert_allclose(p.net,[-.0001,0,-.0003])

    def test_multiday_sleeves_not_overlapping_return_sum(self):
        _,p=core.portfolio(np.ones(6),np.ones(6)*100,3,0.)
        np.testing.assert_allclose(p.position,[1/3,2/3,1,2/3,1/3])
        np.testing.assert_allclose(p.net,0)

    def test_holm_known_values(self):
        np.testing.assert_allclose(core.holm([.01,.04,.03]),[.03,.06,.06])

    def test_hac_gap_and_positive_variance(self):
        rng=np.random.default_rng(2); x=rng.normal(size=(200,1)); y=.3*x[:,0]+rng.normal(0,.01,200)
        y[10:15]=np.nan
        tab=core.hac_ols(y,x,['x'],5)
        self.assertAlmostEqual(tab.beta.iloc[1],.3,delta=.01)
        self.assertTrue((tab.hac_se>=0).all())

    def test_bootstrap_reproducible(self):
        y=np.zeros(100);a=np.ones(100);b=np.zeros(100)
        t=core.block_test(a,b,y,reps=100)
        self.assertEqual(t['gain'],1.)
        self.assertEqual(t,core.block_test(a,b,y,reps=100))

    def test_gamma_flip_and_dealer_sign(self):
        now=pd.Timestamp('2024-06-03T16:00:00-04:00')
        d=pd.DataFrame({'spot':[100.,100.],'strike':[90.,110.],'iv':[.2,.2],
                        'snapshot_time':[now,now],'expiry_time':[now+pd.Timedelta(days=30)]*2,
                        'rate':[0.,0.],'dividend_yield':[0.,0.],'option_type':['P','C'],
                        'open_interest':[1000,1000],'multiplier':[100,100],
                        'dealer_net_contracts':[1000,1000]})
        a=gex.exposure(d,'heuristic')
        self.assertTrue(90<a['gamma_flip']<110)
        b=gex.exposure(d,'dealer')
        self.assertGreater(b['gex'],0)
        self.assertTrue(np.isnan(b['gamma_flip']))

    def test_full_development_and_final_pipeline(self):
        for phase in ['development','final']:
            dest=self.root/f'report_{phase}'
            args=argparse.Namespace(input=str(self.root/'data/daily.csv'),calendar=str(self.root/'data/calendar.csv'),
                metadata=str(self.root/'data/metadata.json'),out=str(dest),phase=phase,holdout=80,min_train=60,
                har_min=40,window=180,refit=10,alpha=10.,return_horizons=[1,5,21],vol_horizons=[5,21],
                bootstrap=100,block=20,cost_bps=1.,events=str(self.root/'data/events.csv'))
            with contextlib.redirect_stdout(io.StringIO()):research.run(args)
            summary=json.loads((dest/'summary.json').read_text())
            self.assertTrue(all(a['all_train_labels_known'] for a in summary['audit']))
            preds=pd.read_csv(dest/'H2_h1_predictions.csv')
            if phase=='development':self.assertLess(preds.date.max(),summary['holdout_start'])
            else:self.assertGreaterEqual(preds.date.min(),summary['holdout_start'])
            self.assertTrue((dest/'report.html').exists())
            comparison=pd.read_csv(dest/'model_comparisons.csv')
            self.assertIn('p_holm_aux',comparison)
            self.assertEqual((comparison.test_role=='primary').sum(),1)
            self.assertEqual((comparison.test_role=='auxiliary').sum(),12)
            self.assertIn('har_fwd_21',summary['feature_specs']['H2']['R3_skew_change'])
            with self.assertRaisesRegex(ValueError,'nonempty'):research.run(args)

    def test_proxy_branch_hypothesis_distinct(self):
        d=self.raw.copy();meta=self.meta.copy();meta['surface_kind']='proxies'
        d['vix_30']=d.iv_atm_30*100;d['vix_3m']=d.iv_atm_91*100
        d['skew_index']=100+(d.iv_put25_30-d.iv_call25_30)*100;d['vvix']=90.
        p=core.features(core.validate(d,self.cal,meta),'proxies')
        self.assertIn('skew_index_change',p)
        self.assertNotIn('skew_change',p)
        self.assertIn('skew_index_change',research.specs('proxies')[1]['R3_skew_change'])

    def test_wrds_parameterized_download_with_mock_connection(self):
        owner=self
        class DB:
            closed=False
            def list_tables(self,library):return ['vsurfd2024']
            def describe_table(self,**kwargs):return pd.DataFrame({'name':['secid','date','days','delta','cp_flag','impl_volatility']})
            def raw_sql(self,sql,params,date_cols):
                owner.assertIn('%(secid)s',sql);owner.assertEqual(params['secid'],108105)
                return pd.DataFrame({'secid':[108105],'date':['2024-01-03'],'days':[30],'delta':[50],'cp_flag':['C'],'impl_volatility':[.2]})
            def close(self):self.closed=True
        conn=DB()
        args=argparse.Namespace(option_library='optionm',secid=108105,start='2024-01-01',end='2024-12-31',out=str(self.root/'wrds_mock'))
        with patch.dict('sys.modules',{'wrds':SimpleNamespace(Connection=lambda:conn)}):data.wrds_download(args)
        self.assertTrue(conn.closed)
        self.assertTrue((self.root/'wrds_mock/surface_long.csv').exists())


if __name__=='__main__':unittest.main()
