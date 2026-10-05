import unittest
import numpy as np
import pandas as pd
import robust_h2


class Tests(unittest.TestCase):
    def panel(self):
        n=100;rng=np.random.default_rng(42)
        dates=pd.date_range('2020-01-01',periods=n,tz='UTC')
        p=pd.DataFrame({'date':dates.strftime('%Y-%m-%d'),'asof_time':dates,
                        'entry_time':dates+pd.Timedelta(hours=1),
                        'skew_change':rng.normal(size=n),'x':rng.normal(size=n)})
        y=pd.Series(rng.normal(size=n));end=pd.Series(dates+pd.Timedelta(days=2))
        return p,y,end

    def test_future_values_and_labels_do_not_change_past(self):
        p,y,end=self.panel();models={'R0_mean':[],'R3_skew_change':['x','skew_change']}
        a,_=robust_h2.walk(p,y,end,models,min_train=30,window=50,refit=5)
        future=p.copy();future.loc[70:,['x','skew_change']]=1e9
        yy=y.copy();yy.loc[70:]=-1e9
        b,_=robust_h2.walk(future,yy,end,models,min_train=30,window=50,refit=5)
        np.testing.assert_allclose(a.R3_skew_change.iloc[:70],b.R3_skew_change.iloc[:70],equal_nan=True)
        np.testing.assert_allclose(a.skew_upper.iloc[:70],b.skew_upper.iloc[:70],equal_nan=True)

    def test_current_feature_not_in_its_own_bounds(self):
        p,y,end=self.panel();models={'R0_mean':[],'R3_skew_change':['x','skew_change']}
        a,au=robust_h2.walk(p,y,end,models,min_train=30,window=50,refit=1)
        p.loc[60,'skew_change']=1e9
        b,bu=robust_h2.walk(p,y,end,models,min_train=30,window=50,refit=1)
        self.assertEqual(a.skew_upper.iloc[60],b.skew_upper.iloc[60])
        self.assertEqual(b.skew_used.iloc[60],b.skew_upper.iloc[60])
        good=b.train_n.notna();self.assertTrue((b.loc[good,'effective_alpha']==b.loc[good,'train_n']).all())
        self.assertTrue((b.loc[good,'train_end_max']<b.loc[good,'asof_time']).all())

    def test_missing_row_is_not_deleted(self):
        p,y,end=self.panel();p.loc[60,'skew_change']=np.nan
        out,_=robust_h2.walk(p,y,end,{'R0_mean':[],'R3_skew_change':['x','skew_change']},min_train=30,window=50)
        self.assertEqual(len(out),len(p));self.assertTrue(np.isnan(out.R3_skew_change.iloc[60]))

    def test_duplicate_or_constant_skew_clip(self):
        x=np.zeros((40,2));clean,lo,hi=robust_h2.fit_preprocessor(x,1)
        self.assertEqual(lo,0.);self.assertEqual(hi,0.);np.testing.assert_array_equal(clean,x)


if __name__=='__main__':unittest.main()
