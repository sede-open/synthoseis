from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np


class RPMABC(ABC):
    """Abstract Base Class for Rock Property Models."""

    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg

    @abstractmethod
    def create_1d_trends(
        self, z: Optional[np.ndarray] = None, store_in_hdf: bool = True
    ) -> Dict[str, np.ndarray]:
        """Create 1D depth trends for all rock properties."""
        pass

    @staticmethod
    @abstractmethod
    def calc_shale_rho(z: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def calc_shale_vp(z: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def calc_shale_vs(z: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def calc_brine_sand_rho(z: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def calc_brine_sand_vp(z: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def calc_brine_sand_vs(z: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def calc_oil_sand_rho(z: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def calc_oil_sand_vp(z: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def calc_oil_sand_vs(z: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def calc_gas_sand_rho(z: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def calc_gas_sand_vp(z: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def calc_gas_sand_vs(z: np.ndarray) -> np.ndarray:
        pass

    def calc_shale_properties(self, z_rho, z_vp, z_vs):
        from rockphysics.RockPropertyModels import RockProperties
        rho = self.calc_shale_rho(z_rho)
        vp = self.calc_shale_vp(z_vp)
        vs = self.calc_shale_vs(z_vs)
        return RockProperties(rho, vp, vs)

    def calc_brine_sand_properties(self, z_rho, z_vp, z_vs):
        from rockphysics.RockPropertyModels import RockProperties
        rho = self.calc_brine_sand_rho(z_rho)
        vp = self.calc_brine_sand_vp(z_vp)
        vs = self.calc_brine_sand_vs(z_vs)
        return RockProperties(rho, vp, vs)

    def calc_oil_sand_properties(self, z_rho, z_vp, z_vs):
        from rockphysics.RockPropertyModels import RockProperties
        rho = self.calc_oil_sand_rho(z_rho)
        vp = self.calc_oil_sand_vp(z_vp)
        vs = self.calc_oil_sand_vs(z_vs)
        return RockProperties(rho, vp, vs)

    def calc_gas_sand_properties(self, z_rho, z_vp, z_vs):
        from rockphysics.RockPropertyModels import RockProperties
        rho = self.calc_gas_sand_rho(z_rho)
        vp = self.calc_gas_sand_vp(z_vp)
        vs = self.calc_gas_sand_vs(z_vs)
        return RockProperties(rho, vp, vs)