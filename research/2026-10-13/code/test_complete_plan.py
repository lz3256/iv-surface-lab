import unittest
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from complete_plan import objective, solve, fit_calibration, fit_qlike, walk_qlike, rolling_positive_calibration


class CompletionChecks(unittest.TestCase):
    def test_gradient_and_hessian_by_finite_differences(self):
        rng=np.random.default_rng(1001)
        x=np.column_stack([np.ones(140),rng.normal(size=(140,3))])
        y=rng.normal(-3,.5,140);theta=np.array([-3.,.2,-.1,.08]);offset=np.zeros(140)
        penalty=np.array([0.,.03,.03,.03])
        value,gradient,hessian=objective(theta,x,y,offset,penalty)
        step=1e-5;g=[];hh=[]
        for j in range(len(theta)):
            bump=np.zeros_like(theta);bump[j]=step
            plus=objective(theta+bump,x,y,offset,penalty)
            minus=objective(theta-bump,x,y,offset,penalty)
            g.append((plus[0]-minus[0])/(2*step))
            hh.append((plus[1]-minus[1])/(2*step))
        np.testing.assert_allclose(gradient,g,rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(hessian,np.array(hh).T,rtol=1e-6,atol=1e-8)
        self.assertGreater(np.linalg.eigvalsh(hessian).min(),0)

    def test_known_positive_calibration_recovery_and_units(self):
        lf=np.linspace(-6,-2,300);ly=.4+.85*lf
        fitted=fit_calibration(lf,ly,alpha=0)
        self.assertAlmostEqual(fitted['a'],.4,places=6)
        self.assertAlmostEqual(fitted['b'],.85,places=6)
        a=fit_calibration(lf,ly);b=fit_calibration(lf+np.log(252),ly+np.log(252))
        self.assertAlmostEqual(a['d'],b['d'],places=8)
        self.assertAlmostEqual(a['g'],b['g'],places=8)
        self.assertAlmostEqual(a['b'],b['b'],places=8)
        # A decreasing target must respect the prespecified monotonic boundary.
        bound=fit_calibration(lf,-4-.5*lf,alpha=0)
        self.assertGreaterEqual(bound['b'],-1e-12)
        self.assertTrue(bound['bound_active'])

    def test_qlike_known_model_recovery_and_scipy_crosscheck(self):
        rng=np.random.default_rng(1002);x=rng.normal(size=(400,3))
        ly=-4+x@np.array([.2,-.15,.1])
        model=fit_qlike(x,ly,alpha=0)
        pred=model['beta'][0]+((x-model['mean'])/model['scale'])@model['beta'][1:]
        np.testing.assert_allclose(pred,ly,atol=1e-6)
        design=np.column_stack([np.ones(400),x]);noise=ly+rng.normal(0,.4,400)
        penalty=np.r_[0.,np.repeat(.025,3)]
        beta,diag=solve(design,noise,penalty=penalty,initial=np.array([-4.,0.,0.,0.]))
        result=minimize(lambda t:objective(t,design,noise,np.zeros(400),penalty)[:2],
                        np.array([-4.,0.,0.,0.]),jac=True,method='BFGS',options={'gtol':1e-9,'maxiter':500})
        self.assertLess(abs(objective(beta,design,noise,np.zeros(400),penalty)[0]-result.fun),1e-10)
        np.testing.assert_allclose(beta,result.x,atol=1e-6)

    def test_future_labels_cannot_change_qlike_forecasts(self):
        rng=np.random.default_rng(1003);n=120
        times=pd.date_range('2000-01-01',periods=n,freq='D',tz='UTC')
        p=pd.DataFrame({'date':times.strftime('%Y-%m-%d'),'asof_time':times,
                        'x1':rng.normal(size=n),'x2':rng.normal(size=n)})
        y=-4+.1*p.x1.to_numpy()+rng.normal(0,.2,n)
        ends=times+pd.Timedelta(days=5)
        specs={'V1_HAR':['x1'],'V2_IV':['x1','x2']}
        a,audit=walk_qlike(p,y,ends,min_train=30,window=60,refit=5,specs=specs)
        cutoff=75;y2=y.copy();y2[ends>=times[cutoff]]+=9
        b,_=walk_qlike(p,y2,ends,min_train=30,window=60,refit=5,specs=specs)
        np.testing.assert_allclose(a.iloc[:cutoff+1,1:],b.iloc[:cutoff+1,1:],equal_nan=True,atol=0)
        self.assertTrue((pd.to_datetime(audit.target_end_max,utc=True)<pd.to_datetime(audit.asof_time,utc=True)).all())

    def test_future_labels_cannot_change_positive_calibration(self):
        rng=np.random.default_rng(1004);n=360
        times=pd.date_range('2000-01-01',periods=n,freq='D',tz='UTC')
        log_y=rng.normal(-4,.4,n);f1=rng.normal(-4,.3,n);f2=rng.normal(-4,.3,n)
        d=pd.DataFrame({'date':times.strftime('%Y-%m-%d'),'phase':'development','asof_time':times,
                        'full_target_end':times+pd.Timedelta(days=21),'full_log_target':log_y,
                        'V1_HAR':f1,'V2_IV':f2,'V1_HAR__training_smearing':np.exp(f1),
                        'V2_IV__training_smearing':np.exp(f2),'recovered_boundary_target':False})
        a,sa,audit=rolling_positive_calibration(d)
        altered=d.copy();altered.loc[altered.full_target_end>=times[320],'full_log_target']+=9
        b,sb,_=rolling_positive_calibration(altered)
        np.testing.assert_allclose(a[:321],b[:321],equal_nan=True,atol=0)
        np.testing.assert_allclose(sa[:321],sb[:321],equal_nan=True,atol=0)
        self.assertTrue((audit.history_n==252).all())
        self.assertTrue((pd.to_datetime(audit.target_end_max,utc=True)<pd.to_datetime(audit.asof_time,utc=True)).all())


if __name__=='__main__':unittest.main()
