import numpy as np
from rockphysics.RockPropertyModels import RockProperties, store_1d_trend_dict_to_hdf
from rockphysics.rpm_abc import RPMABC

#3800 max depth
class RPMTagilsk(RPMABC):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

    def create_1d_trends(self, z = None, store_in_hdf = False):
        if z is None:
            z = np.arange(self.cfg.cube_shape[-1] * self.cfg.digi)
        
        d = dict(
            z=z,
            shale_vp=self.calc_shale_vp(z),
            shale_vs=self.calc_shale_vs(z),
            shale_rho=self.calc_shale_rho(z),
            brine_sand_vp=self.calc_brine_sand_vp(z),
            brine_sand_vs=self.calc_brine_sand_vs(z),
            brine_sand_rho=self.calc_brine_sand_rho(z),
            oil_sand_vp=self.calc_oil_sand_vp(z),
            oil_sand_vs=self.calc_oil_sand_vs(z),
            oil_sand_rho=self.calc_oil_sand_rho(z),
            gas_sand_vp=self.calc_gas_sand_vp(z),
            gas_sand_vs=self.calc_gas_sand_vs(z),
            gas_sand_rho=self.calc_gas_sand_rho(z),
        )
        if store_in_hdf:
            store_1d_trend_dict_to_hdf(self.cfg, d, z)

        return d
    
    @staticmethod
    def calc_shale_rho(z):
        param = [-3.27905787e-12, 1.86750139e-08, 9.64773845e-05, 2.09709627e+00]
        return np.polyval(param, z)

    @staticmethod
    def calc_shale_vp(z):
        param = [-2.86940692e-04, 2.02356702e+00, 5.13645163e+02]
        return np.polyval(param, z)

    @staticmethod
    def calc_shale_vs(z):
        param = [-1.51184658e-04, 1.11423506e+00, 2.17341849e+02]
        return np.polyval(param, z)

    @staticmethod
    def calc_brine_sand_rho(z):
        param = [1.62260122e-08, 4.19501863e-05, 2.10717208e+00]
        return np.polyval(param, z)

    @staticmethod
    def calc_brine_sand_vp(z):
        param = [-1.72905613e-04, 1.39902406e+00, 1.15717554e+03]
        return np.polyval(param, z)

    @staticmethod
    def calc_brine_sand_vs(z):
        param = [-4.86767619e-05, 6.08845432e-01, 7.40710471e+02]
        return np.polyval(param, z)

    @staticmethod
    def calc_oil_sand_rho(z):
        param = [ 1.82457149e-11, -1.54547807e-07, 5.46493718e-04, 1.66744366e+00]
        return np.polyval(param, z)

    @staticmethod
    def calc_oil_sand_vp(z):
        param = [-1.05896142e-07, 5.08492087e-04, 3.15498496e-01, 1.44786139e+03]
        return np.polyval(param, z)

    @staticmethod
    def calc_oil_sand_vs(z):
        param = [-1.05101834e-07, 5.90901816e-04, -2.71462741e-01, 7.21829913e+02]
        return np.polyval(param, z)

    @staticmethod
    def calc_gas_sand_rho(z):
        param = [-5.43088129e-08, 4.16689269e-04, 1.69501400e+00]
        return np.polyval(param, z)
    
    @staticmethod
    def calc_gas_sand_vp(z):
        def exp_func(d, a, b, c):
            return a * np.exp(b * d) + c
        param = [ 2.03767992e+07, 3.90733465e-08, -2.03752253e+07]
        return exp_func(z, *param)

    @staticmethod
    def calc_gas_sand_vs(z):
        def exp_func(d, a, b, c):
            return a * np.exp(b * d) + c
        param = [ 2.03767992e+07, 3.90733465e-08, -2.03752253e+07]
        return exp_func(z, *param)/np.sqrt(2)