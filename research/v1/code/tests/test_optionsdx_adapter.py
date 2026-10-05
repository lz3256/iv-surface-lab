import unittest
import numpy as np
import pandas as pd
from build_optionsdx import bracket, constant_nodes, screen_curve
from ivsurf import core


class AdapterTests(unittest.TestCase):
    def test_extreme_gap_masks_curve_without_dropping_date(self):
        d = pd.DataFrame({'date':['2020-01-02','2020-01-03'],
                          'iv_atm_30':[.25,.4], 'iv_atm_put_30':[.2,.02],
                          'iv_atm_91':[.25,.49], 'iv_atm_put_91':[.2,.015],
                          'iv_put25_30':[.3,.05], 'iv_call25_30':[.2,.3]})
        s, flags = screen_curve(d)
        self.assertEqual(len(s),2)
        self.assertEqual(flags.date.tolist(),['2020-01-03'])
        self.assertTrue(s.loc[1,core.SURFACE].isna().all())
        self.assertEqual(s.iv_atm_30.iloc[0],.25)

    def test_no_extrapolation_and_duplicate_delta_median(self):
        self.assertTrue(np.isnan(bracket([.2,.4],[.3,.2],.5)[0]))
        self.assertAlmostEqual(bracket([.25,.25,.5],[.2,.4,.2],.25)[0],.3)

    def test_total_variance_interpolation(self):
        d = pd.DataFrame({'date':['2020-01-02']*2,'tenor_days':[20.,40.],
                          'atm_call':[.2,.3],'atm_put':[.2,.3],
                          'call25':[.2,.3],'put25':[.2,.3]})
        surface, audit = constant_nodes(d)
        expected = np.sqrt((.2**2*20+.3**2*40)/2/30)
        self.assertAlmostEqual(surface.iv_atm_30.iloc[0],expected)
        self.assertTrue(np.isnan(surface.iv_atm_91.iloc[0]))

    def test_missing_surface_date_breaks_skew_change(self):
        d = pd.DataFrame({'reference_price':[100.,101.,102.,103.],
                          'open':[100.]*4,'high':[101.]*4,'low':[99.]*4,'close':[100.]*4,
                          'iv_atm_30':[.2,np.nan,.2,.2], 'iv_atm_91':[.21,np.nan,.21,.21],
                          'iv_put25_30':[.25,np.nan,.26,.27], 'iv_call25_30':[.18,np.nan,.18,.18]})
        f = core.features(d,'curve')
        self.assertTrue(np.isnan(f.skew_change.iloc[2]))
        self.assertAlmostEqual(f.skew_change.iloc[3],.01)


if __name__ == '__main__':
    unittest.main()
