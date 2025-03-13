import numpy as np
from rockphysics.RockPropertyModels import RockProperties, store_1d_trend_dict_to_hdf
from bruges.rockphysics import moduli


class RPMExample:
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg

    def create_1d_trends(self, z=None, store_in_hdf=True):
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
        return 7.7e-12 * z**3 + -8.8e-08 * z**2 + 0.0004 * z + 1.957

    @staticmethod
    def calc_shale_vp(z):
        return -0.00013 * z**2 + 1.13 * z + 1580

    @staticmethod
    def calc_shale_vs(z):
        return -0.0001 * z**2 + 0.96 * z + 279

    @staticmethod
    def calc_brine_sand_rho(z):
        return -7.8e-09 * z**2 + 0.00012 * z + 2.021

    @staticmethod
    def calc_brine_sand_vp(z):
        return -1.34e-05 * z**2 + 0.49 * z + 2317

    @staticmethod
    def calc_brine_sand_vs(z):
        return -1.0785e-05 * z**2 + 0.391 * z + 1007

    @staticmethod
    def calc_oil_sand_rho(z):
        return -9.23e-09 * z**2 + 0.00014 * z + 1.916

    @staticmethod
    def calc_oil_sand_vp(z):
        return -8.876e-06 * z**2 + 0.505 * z + 1998

    @staticmethod
    def calc_oil_sand_vs(z):
        return -1.126e-05 * z**2 + 0.391 * z + 1036

    @staticmethod
    def calc_gas_sand_rho(z):
        return -1.818e-08 * z**2 + 0.000247 * z + 1.612

    @staticmethod
    def calc_gas_sand_vp(z):
        return -3.216e-06 * z**2 + 0.4796 * z + 1996

    @staticmethod
    def calc_gas_sand_vs(z):
        return -1.0687e-05 * z**2 + 0.3662 * z + 1135

    def calc_shale_properties(self, z_rho, z_vp, z_vs):
        rho = self.calc_shale_rho(z_rho)
        vp = self.calc_shale_vp(z_vp)
        vs = self.calc_shale_vs(z_vs)
        shales = RockProperties(rho, vp, vs)
        return shales

    def calc_brine_sand_properties(self, z_rho, z_vp, z_vs):
        rho = self.calc_brine_sand_rho(z_rho)
        vp = self.calc_brine_sand_vp(z_vp)
        vs = self.calc_brine_sand_vs(z_vs)
        brine_sand = RockProperties(rho, vp, vs)
        return brine_sand

    def calc_oil_sand_properties(self, z_rho, z_vp, z_vs):
        rho = self.calc_oil_sand_rho(z_rho)
        vp = self.calc_oil_sand_vp(z_vp)
        vs = self.calc_oil_sand_vs(z_vs)
        oil_sand = RockProperties(rho, vp, vs)
        return oil_sand

    def calc_gas_sand_properties(self, z_rho, z_vp, z_vs):
        rho = self.calc_gas_sand_rho(z_rho)
        vp = self.calc_gas_sand_vp(z_vp)
        vs = self.calc_gas_sand_vs(z_vs)
        gas_sand = RockProperties(rho, vp, vs)
        return gas_sand
