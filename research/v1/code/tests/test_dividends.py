"""Entitlement, information timing, and quality tests using artificial distributions."""
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from ivsurf import core, data, research, wrds_build as wb
from test_wrds_build import fixture,build


class DividendTests(unittest.TestCase):
    def make_panel(self):
        dates=pd.bdate_range('2024-01-02',periods=5)
        return pd.DataFrame({'entry_price':[100.,99.,101.,100.,102.],
            'entry_cash_dividend':[9.,1.,0.,2.,0.],
            'entry_is_ex_dividend':[1,1,0,1,0],
            'entry_time':(dates+pd.Timedelta(hours=9,minutes=30)).tz_localize('America/New_York').tz_convert('UTC'),
            'entry_dividend_available_time':(dates+pd.Timedelta(hours=16,minutes=5)).tz_localize('America/New_York').tz_convert('UTC')})

    def test_exit_ex_date_dividend_included_entry_day_excluded(self):
        p=self.make_panel();y,end=core.target(p,'return',1)
        self.assertAlmostEqual(y.iloc[0],0.) # -1 price +1 cash, entry-day $9 not earned.
        self.assertAlmostEqual(y.iloc[1],101/99-1) # Entering ex-date does not earn its $1.
        self.assertEqual(end.iloc[0],p.entry_dividend_available_time.iloc[1])
        self.assertTrue(pd.isna(y.iloc[-1]))
        self.assertTrue(pd.isna(end.iloc[-1]))

    def test_multi_day_distribution_sum_without_reinvestment(self):
        p=self.make_panel();y,_=core.target(p,'return',3)
        self.assertAlmostEqual(y.iloc[0],(100+1+2)/100-1)
        self.assertAlmostEqual(y.iloc[1],(102+2)/99-1)
        self.assertTrue(y.iloc[2:].isna().all())

    def test_rv_dividend_adjustment_removes_mechanical_overnight_jump(self):
        with tempfile.TemporaryDirectory() as tmp:
            data.demo(tmp,rows=100)
            d=pd.read_csv(Path(tmp)/'daily.csv').head(3)
        for col in ['open','high','low','close']:d[col]=[100.,99.,99.]
        d['cash_dividend']=[0.,1.,0.]
        f=core.features(d,'curve')
        self.assertAlmostEqual(f.rv_daily.iloc[1],0.)
        self.assertLess(f.overnight_log_return_raw.iloc[1],0.)
        self.assertAlmostEqual(f.overnight_log_return.iloc[1],0.)

    def test_long_receives_short_pays_same_dividend(self):
        prices=[100.,99.,99.];cash=[5.,1.,0.]
        for signal in [1.,-1.]:
            _,stream=core.portfolio([signal]*3,prices,1,0,cash)
            self.assertAlmostEqual(stream.net.iloc[0],0.)
        _,stream=core.portfolio([1.]*3,[100.,100.,100.],1,0,cash)
        self.assertAlmostEqual(stream.net.iloc[0],.01)
        _,stream=core.portfolio([-1.]*3,[100.,100.,100.],1,0,cash)
        self.assertAlmostEqual(stream.net.iloc[0],-.01)

    def test_builder_recovers_cash_and_maps_signal_and_entry(self):
        s,r,p,c=fixture();row=p.index[p.date=='2024-11-29'][0]
        p.loc[row,'ret']=p.loc[row,'retx']+1.25/p.loc[row-1,'prc']
        d,_,meta,q=build((s,r,p,c))
        self.assertAlmostEqual(d.cash_dividend.iloc[2],1.25)
        self.assertAlmostEqual(d.entry_cash_dividend.iloc[1],1.25)
        self.assertEqual(d.is_ex_dividend.tolist(),[0,0,1])
        self.assertEqual(meta['return_basis'],'cash_distribution_total_return')
        self.assertEqual(q['dividends']['ex_distribution_sessions'],1)

    def test_missing_return_is_not_zero_dividend(self):
        s,r,p,c=fixture();p.loc[2,'ret']=np.nan
        with self.assertRaisesRegex(ValueError,'missing is not zero'):build((s,r,p,c))

    def test_retx_price_inconsistency_rejected(self):
        s,r,p,c=fixture();p.loc[2,'retx']+=.01
        with self.assertRaisesRegex(ValueError,'RETX disagrees'):build((s,r,p,c))

    def test_missing_prior_session_rejected(self):
        s,r,p,c=fixture()
        with self.assertRaisesRegex(ValueError,'Missing market sessions'):build((s,r,p.iloc[1:],c))

    def test_validation_refuses_missing_dividends_or_fake_pre_open_availability(self):
        d,_,meta,_=build();c=fixture()[-1]
        with self.assertRaisesRegex(ValueError,'Missing columns'):core.validate(d.drop(columns='entry_cash_dividend'),c,meta)
        bad=d.copy();bad['entry_dividend_available_time']=bad.entry_time
        with self.assertRaisesRegex(ValueError,'before their session close'):core.validate(bad,c,meta)

    def test_future_distribution_not_a_feature_or_early_training_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            data.demo(tmp,rows=200)
            raw=pd.read_csv(Path(tmp)/'daily.csv');cal=pd.read_csv(Path(tmp)/'calendar.csv')
            meta=json.loads((Path(tmp)/'metadata.json').read_text())
        p=core.features(core.validate(raw,cal,meta),'curve')
        p['entry_cash_dividend']=0.
        p['entry_dividend_available_time']=p.entry_time+pd.Timedelta(hours=7)
        y,end=core.target(p,'return',1)
        a=core.walk(p,y,end,{'base':[],'model':['ret_1','skew_change']},30,80,1)
        future=p.copy();future.loc[future.entry_dividend_available_time>=p.asof_time.iloc[100],'entry_cash_dividend']=50.
        y2,end2=core.target(future,'return',1)
        b=core.walk(future,y2,end2,{'base':[],'model':['ret_1','skew_change']},30,80,1)
        np.testing.assert_allclose(a.model.iloc[:101],b.model.iloc[:101],equal_nan=True)
        names=sum(research.specs('curve')[1].values(),[])
        self.assertNotIn('entry_cash_dividend',names)

    def test_ex_date_mask_keeps_calendar_positions_and_includes_entry(self):
        p=self.make_panel();flags=core.ex_dividend_window(p,1)
        self.assertEqual(flags.tolist(),[True,True,True,True,False])
        y,end=core.target(p,'return',1)
        self.assertEqual(len(y.mask(flags)),len(p))
        self.assertEqual(end.iloc[0],p.entry_dividend_available_time.iloc[1])

    def test_event_after_actual_exit_not_in_holding_window(self):
        p=self.make_panel().iloc[:1].copy()
        p['asof_time']=p.entry_time-pd.Timedelta(hours=15)
        p['session_id']=0
        for col in ['ret_1','ret_5','rv_22','iv_var_30','term_slope','har_fwd_21','skew_change']:p[col]=0.
        pred=pd.DataFrame({'target':[.01],'target_end':[pd.Timestamp('2024-01-03T16:05:00-05:00')],
            'target_exit_time':[pd.Timestamp('2024-01-03T09:30:00-05:00')]})
        events=pd.DataFrame({'event_id':['after_exit'],'event_type':['TEST'],
                             'event_time':['2024-01-03T12:00:00-05:00']})
        groups,_=research.event_analysis(p,pred,events,'curve')
        self.assertEqual(groups.set_index('group').loc['in_holding_window','n_sessions'],0)
        self.assertEqual(groups.set_index('group').loc['neither','n_sessions'],1)

    def test_atm_pair_diagnostic_and_missing_pair_not_zero(self):
        s,r,p,c=fixture()
        _,_,_,q=build((s,r,p,c))
        self.assertIsNone(q['atm_call_put_diagnostic']['maturities']['30']['mean_abs_iv_call_minus_put'])
        extra=s.loc[(s.days==30)&(s.delta==50)].copy()
        extra['cp_flag']='P';extra['delta']=-50;extra['impl_volatility']=[.21,.23,.22]
        _,_,_,q=build((pd.concat([s,extra],ignore_index=True),r,p,c))
        item=q['atm_call_put_diagnostic']['maturities']['30']
        self.assertEqual(item['paired_sessions'],3)
        self.assertAlmostEqual(item['mean_abs_iv_call_minus_put'],.02)
        self.assertAlmostEqual(item['max_abs_iv_call_minus_put'],.03)

    def test_missing_surface_costs_two_skew_changes_without_imputation(self):
        s,r,p,c=fixture();s=s.loc[s.date!='2024-11-27']
        d,_,_,q=build((s,r,p,c))
        self.assertEqual(q['skew_change_coverage_excluding_first'],0.)
        self.assertAlmostEqual(q['surface_coverage'],2/3)
        self.assertEqual(len(d),3)

    def test_sensitivity_reports_do_not_claim_primary_or_filter_trades_with_future_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data.demo(root/'data',rows=650)
            raw=pd.read_csv(root/'data/daily.csv');cal=pd.read_csv(root/'data/calendar.csv')
            meta=json.loads((root/'data/metadata.json').read_text())
            cash=pd.Series(0.,index=cal.date);cash.iloc[::40]=1.
            raw['cash_dividend']=raw.date.map(cash)
            raw['is_ex_dividend']=(raw.cash_dividend>0).astype(int)
            raw['entry_cash_dividend']=raw.entry_session.map(cash)
            raw['entry_is_ex_dividend']=(raw.entry_cash_dividend>0).astype(int)
            raw['entry_dividend_available_time']=raw.entry_session.map(cal.set_index('date').session_close)
            raw.loc[10:50,core.SURFACE]=np.nan # Early coverage, before first possible forecast.
            meta['return_basis']='cash_distribution_total_return'
            raw.to_csv(root/'data/daily.csv',index=False);data.save_json(root/'data/metadata.json',meta)
            args=argparse.Namespace(input=str(root/'data/daily.csv'),calendar=str(root/'data/calendar.csv'),
                metadata=str(root/'data/metadata.json'),out=str(root/'report'),phase='development',holdout=80,min_train=60,
                har_min=40,window=180,refit=10,alpha=10.,return_horizons=[1],vol_horizons=[5],bootstrap=100,
                block=20,cost_bps=1.,events=None,exclude_ex_dividend=True,study_role='sensitivity')
            with contextlib.redirect_stdout(io.StringIO()):research.run(args)
            summary=json.loads((root/'report/summary.json').read_text())
            self.assertEqual(summary['inference_families']['primary_count'],0)
            self.assertTrue(any('Development raw' in w for w in summary['warnings']))
            pred=pd.read_csv(root/'report/H2_h1_predictions.csv')
            self.assertTrue(pred.ex_dividend_label_excluded.any())
            self.assertTrue(pred.loc[pred.ex_dividend_label_excluded,'target'].isna().all())
            self.assertIn('target_exit_time',pred)
            pnl=pd.read_csv(root/'report/H2_h1_R3_skew_change_pnl.csv')
            self.assertEqual(len(pnl),len(pred)-1)
            self.assertTrue((pnl.dividend_at_interval_end>0).any())
            # The dividend-aware final branch must preserve horizon and label timing too.
            args.out=str(root/'final');args.phase='final';args.exclude_ex_dividend=False
            args.study_role='primary';args.return_horizons=[1,5,21];args.vol_horizons=[5,21]
            with contextlib.redirect_stdout(io.StringIO()):research.run(args)
            summary=json.loads((root/'final/summary.json').read_text())
            self.assertTrue(all(a['all_train_labels_known'] for a in summary['audit']))
            self.assertEqual(summary['inference_families']['primary_count'],1)
            pred=pd.read_csv(root/'final/H2_h5_predictions.csv')
            first=len(raw)-80
            expected=(raw.entry_price.iloc[first+5]+raw.entry_cash_dividend.iloc[first+1:first+6].sum())/raw.entry_price.iloc[first]-1
            self.assertAlmostEqual(pred.target.iloc[0],expected)


if __name__=='__main__':unittest.main()
