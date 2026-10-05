import unittest
import numpy as np
import pandas as pd
from residual_model import fit,predict
from same_endpoints import pair_value
from precision import compare
from run_events import containing_window


class ExtensionTests(unittest.TestCase):
    def test_FWL_matches_joint_OLS(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(900,6));s=x[:,0]+rng.normal(size=900)
        y=.03*x[:,1]+.02*s+rng.normal(0,.02,900);m=fit(x,s,y)
        design=np.column_stack([np.ones(len(x)),x,np.clip(s,m['lower'],m['upper'])])
        expected=np.linalg.lstsq(design,y,rcond=None)[0][-1]
        self.assertAlmostEqual(m['beta'],expected,places=10)
        self.assertLess(m['orthogonality_max_abs'],1e-8)

    def test_future_forecast_extremes_cannot_change_fit(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(600,6));s=rng.normal(size=600);y=rng.normal(size=600)
        m=fit(x,s,y);before=repr(m)
        predict(m,np.full(6,1e7),1e9)
        self.assertEqual(before,repr(m))

    def test_residual_signal_is_unpenalized(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(1000,6));s=rng.normal(size=1000)
        clipped=np.clip(s,*np.quantile(s,[.01,.99]));y=.123*clipped+.1*x[:,0]
        self.assertAlmostEqual(fit(x,s,y)['beta'],.123,places=10)

    def test_constant_signal_is_unidentified(self):
        self.assertIsNone(fit(np.zeros((600,6)),np.ones(600),np.ones(600)))

    def test_fixed_pair_ignores_new_expiry(self):
        rows={'A':{'tenor_days':20,'put25':.2},'B':{'tenor_days':40,'put25':.3}}
        original=pair_value(rows,('A','B'),'put25')[0]
        rows['NEW']={'tenor_days':31,'put25':.9}
        self.assertEqual(original,pair_value(rows,('A','B'),'put25')[0])

    def test_roll_past_30_is_missing_not_extrapolated(self):
        rows={'A':{'tenor_days':10,'put25':.2},'B':{'tenor_days':29,'put25':.3}}
        value,reason=pair_value(rows,('A','B'),'put25')
        self.assertTrue(np.isnan(value));self.assertEqual(reason,'no_longer_brackets_30')

    def test_pair_interpolates_total_variance(self):
        rows={'A':{'tenor_days':20,'put25':.2},'B':{'tenor_days':40,'put25':.3}}
        self.assertAlmostEqual(pair_value(rows,('A','B'),'put25')[0],np.sqrt((.2**2*20+.3**2*40)/60))

    def test_ratio_bootstrap_scale_invariance(self):
        rng=np.random.default_rng(4);y=rng.normal(size=100);a=np.ones(100)*.1;b=np.ones(100)*.2
        first=compare(y,a,b,20,reps=200);second=compare(y*10,a*10,b*10,20,reps=200)
        for key in ['incremental_r2','r2_low95','r2_high95']:
            self.assertAlmostEqual(first[key],second[key],places=12)

    def test_event_windows_before_open_and_after_open(self):
        entry=pd.Series(pd.to_datetime(['2023-06-12 13:30Z','2023-06-13 13:30Z','2023-06-14 13:30Z']))
        exits=entry.shift(-1)
        self.assertEqual(containing_window(entry,exits,pd.Timestamp('2023-06-13 12:30Z')),0)
        self.assertEqual(containing_window(entry,exits,pd.Timestamp('2023-06-13 18:00Z')),1)

    def test_holiday_release_uses_previous_trading_session(self):
        entry=pd.Series(pd.to_datetime(['2020-04-09 13:30Z','2020-04-13 13:30Z']))
        self.assertEqual(containing_window(entry,entry.shift(-1),pd.Timestamp('2020-04-10 12:30Z')),0)


if __name__=='__main__':unittest.main()
