import unittest
import numpy as np
import pandas as pd
from scipy.optimize import nnls
import original_core as core
from literature_experiment import weighted_fit,nnls_level,walk_variants,calibrate_level,timing_panel


class LiteratureChecks(unittest.TestCase):
    def test_weighted_fit_normal_equations_and_flat_weight_limit(self):
        rng=np.random.default_rng(301);x=rng.normal(size=(600,4));y=-4+x@np.array([.1,.2,-.1,.05])+rng.normal(0,.3,600)
        fit=weighted_fit(x,y,np.arange(600))
        self.assertLess(fit['weighted_normal_equation_max_error'],1e-9)
        flat=weighted_fit(x,y,np.arange(600),half_life=1e16);original=core.ridge_fit(x,y,10.)
        np.testing.assert_allclose(flat['beta'],original['beta'],rtol=1e-10,atol=1e-11)
        self.assertLess(fit['effective_n'],600)

    def test_level_calibration_known_relationship_units_and_independent_nnls(self):
        rng=np.random.default_rng(302);f=rng.lognormal(-4,.4,300);y=.004+.7*f
        fit=nnls_level(f,y)
        self.assertAlmostEqual(fit['a'],.004,places=12);self.assertAlmostEqual(fit['b'],.7,places=12)
        scaled=nnls_level(252*f,252*y)
        self.assertAlmostEqual(scaled['a'],252*fit['a'],places=11)
        self.assertAlmostEqual(scaled['b'],fit['b'],places=11)
        for response in [y,np.maximum(.04-.3*f,.001),rng.lognormal(-4,.5,300)]:
            candidate=nnls_level(f,response)
            x=np.column_stack([np.ones(len(f)),f/f.mean()]);sci,_=nnls(x,response/f.mean())
            ours=np.array([candidate['a']/f.mean(),candidate['b']])
            np.testing.assert_allclose(x@ours,x@sci,rtol=1e-10,atol=1e-10)
            self.assertGreaterEqual(candidate['a'],0);self.assertGreaterEqual(candidate['b'],0)

    def fixture(self,n=660):
        rng=np.random.default_rng(303);times=pd.date_range('2000-01-01',periods=n,freq='D',tz='UTC')
        p=pd.DataFrame({'date':times.strftime('%Y-%m-%d'),'asof_time':times})
        for col in ['log_rv_1','log_rv_5','log_rv_22','log_iv_var_30']:p[col]=rng.normal(-4,.4,n)
        for col in ['term_slope','skew','curvature']:p[col]=rng.normal(0,.05,n)
        y=-4+.1*(p.log_rv_1.to_numpy()+4)+rng.normal(0,.25,n)
        return p,y,times+pd.Timedelta(days=5)

    def test_future_labels_cannot_change_recent_window_or_combined_forecasts(self):
        p,y,end=self.fixture();a,_=walk_variants(p,y,end)
        y2=y.copy();y2[end>=p.asof_time.iloc[605]]+=10
        b,_=walk_variants(p,y2,end)
        self.assertGreater(np.isfinite(a.iloc[:606,1:].to_numpy()).sum(),0)
        np.testing.assert_allclose(a.iloc[:606,1:],b.iloc[:606,1:],equal_nan=True,rtol=0,atol=0)

    def test_future_labels_cannot_change_recursive_level_calibration(self):
        p,y,end=self.fixture(360);rng=np.random.default_rng(304)
        f=np.exp(rng.normal(-4,.3,(360,2)))
        out=pd.DataFrame({'V1_HAR__direct_exp':f[:,0],'V2_IV__direct_exp':f[:,1]})
        a,_=calibrate_level(p,y,end,out)
        y2=y.copy();y2[end>=p.asof_time.iloc[320]]+=10
        b,_=calibrate_level(p,y2,end,out)
        self.assertGreater(np.isfinite(a[:321]).sum(),0)
        np.testing.assert_allclose(a[:321],b[:321],equal_nan=True,rtol=0,atol=0)

    def test_timing_panel_keeps_missing_sessions_and_common_qualification(self):
        p,y,end=self.fixture();p.loc[560,'log_iv_var_30']=np.nan
        out,audit=timing_panel(p,y,end)
        self.assertTrue(out.iloc[[560,561],1:].isna().all().all())
        eligibility=np.isfinite(out.iloc[:,1:].to_numpy())
        np.testing.assert_array_equal(eligibility[:,0],eligibility[:,1])
        np.testing.assert_array_equal(eligibility[:,1],eligibility[:,2])
        self.assertTrue((pd.to_datetime(audit.target_end_max,utc=True)<pd.to_datetime(audit.asof_time,utc=True)).all())


if __name__=='__main__':unittest.main()
