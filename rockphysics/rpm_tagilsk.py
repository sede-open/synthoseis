import numpy as np
from rockphysics.RockPropertyModels import RockProperties, store_1d_trend_dict_to_hdf
import sys
import os
# Add the parent directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
import time
from src.util.filehandler import load_gp_grid_params
from src.util.gaussian_processes import sample_gp_grid

class RPMTagilsk():
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg
        self._shale_grid = None
        self._brine_grid = None
        self._oil_grid = None
        self._gas_grid = None

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
        return -1.34e-05 * z**2 + 0.49 * z + 2117

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
        return -1.126e-05 * z**2 + 0.391 * z + 636

    @staticmethod
    def calc_gas_sand_rho(z):
        return -1.818e-08 * z**2 + 0.000247 * z + 1.912

    @staticmethod
    def calc_gas_sand_vp(z):
        return -3.216e-06 * z**2 + 0.4796 * z + 1996

    @staticmethod
    def calc_gas_sand_vs(z):
        return -1.0687e-05 * z**2 + 0.3662 * z + 1135
    
    def _load_shale_grid(self):
        if self._shale_grid is None:
            self._shale_grid = load_gp_grid_params('../data/grids/shale_pl_vp_vs.npz')
        return self._shale_grid

    def _load_brine_grid(self):
        if self._brine_grid is None:
            self._brine_grid = load_gp_grid_params('../data/grids/brine_pl_vp_vs.npz')
        return self._brine_grid

    def _load_oil_grid(self):
        if self._oil_grid is None:
            self._oil_grid = load_gp_grid_params('../data/grids/oil_pl_vp_vs.npz')
        return self._oil_grid

    def _load_gas_grid(self):
        if self._gas_grid is None:
            self._gas_grid = load_gp_grid_params('../data/grids/gas_pl_vp.npz')
        return self._gas_grid

    def calc_shale_properties(self, z_rho, z_vp, z_vs):
        """
        Sample shale properties at specified depths.
        
        Args:
            z_rho: Depth array for density sampling
            z_vp: Depth array for Vp sampling
            z_vs: Depth array for Vs sampling
            
        Returns:
            RockProperties object with sampled values
        """
        grid = self._load_shale_grid()
        
        # Sample GP with separate z arrays and linear extrapolation
        samples = sample_gp_grid(
            grid,
            z_new=[z_rho, z_vp, z_vs],
            n_samples=1,
            linear_extrap_funcs=[
                self.calc_shale_rho,
                self.calc_shale_vp,
                self.calc_shale_vs
            ]
        )
        
        rho_gp = samples[0, :len(z_rho), 0]
        vp_gp = samples[0, :len(z_vp), 1]
        vs_gp = samples[0, :len(z_vs), 2]
        
        return RockProperties(rho_gp, vp_gp, vs_gp)

    def calc_brine_sand_properties(self, z_rho, z_vp, z_vs):
        grid = self._load_brine_grid()
        
        samples = sample_gp_grid(
            grid,
            z_new=[z_rho, z_vp, z_vs],
            n_samples=1,
            linear_extrap_funcs=[
                self.calc_brine_sand_rho,
                self.calc_brine_sand_vp,
                self.calc_brine_sand_vs
            ]
        )
        
        rho_gp = samples[0, :len(z_rho), 0]
        vp_gp = samples[0, :len(z_vp), 1]
        vs_gp = samples[0, :len(z_vs), 2]
        
        return RockProperties(rho_gp, vp_gp, vs_gp)

    def calc_oil_sand_properties(self, z_rho, z_vp, z_vs):
        grid = self._load_oil_grid()
        
        samples = sample_gp_grid(
            grid,
            z_new=[z_rho, z_vp, z_vs],
            n_samples=1,
            linear_extrap_funcs=[
                self.calc_oil_sand_rho,
                self.calc_oil_sand_vp,
                self.calc_oil_sand_vs
            ]
        )
        
        rho_gp = samples[0, :len(z_rho), 0]
        vp_gp = samples[0, :len(z_vp), 1]
        vs_gp = samples[0, :len(z_vs), 2]
        
        return RockProperties(rho_gp, vp_gp, vs_gp)

    def calc_gas_sand_properties(self, z_rho, z_vp, z_vs):
        grid = self._load_gas_grid()
        
        # Note: gas grid only has 2 properties (rho, vp)
        samples = sample_gp_grid(
            grid,
            z_new=[z_rho, z_vp],
            n_samples=1,
            linear_extrap_funcs=[
                self.calc_gas_sand_rho,
                self.calc_gas_sand_vp
            ]
        )
        
        rho_gp = samples[0, :len(z_rho), 0]
        vp_gp = samples[0, :len(z_vp), 1]
        vs_gp = self.calc_gas_sand_vs(z_vs)  # Use linear model for Vs
        
        return RockProperties(rho_gp, vp_gp, vs_gp)