import numpy as np
from rockphysics.RockPropertyModels import RockProperties, store_1d_trend_dict_to_hdf
from bruges.rockphysics import moduli


class RPMABC:
    """
    Custom Rock Property Model for ABC Project.
    
    TODO: Customize the calculation methods below with your specific
    rock physics equations and trends for your geological setting.
    
    All calculation methods receive depth (z) in meters and should return
    rock property values at those depths.
    
    Units:
        - Rho (density): g/cc
        - Vp (compressional velocity): m/s
        - Vs (shear velocity): m/s
    """
    
    def __init__(self, cfg, *args, **kwargs):
        """
        Initialize the RPMABC rock property model.
        
        Parameters
        ----------
        cfg : Parameters
            Configuration object containing model parameters
        """
        self.cfg = cfg

    def create_1d_trends(self, z=None, store_in_hdf=True):
        """
        Create 1D rock property trends vs depth.
        
        This method generates the baseline rock property trends for all
        lithologies and fluid types. These trends are used throughout
        the model to assign properties based on depth and lithology.
        
        Parameters
        ----------
        z : np.ndarray, optional
            Depth array in meters. If None, will be created from config
        store_in_hdf : bool, optional
            Whether to store trends in HDF file (default True)
            
        Returns
        -------
        dict
            Dictionary containing depth and all property arrays
        """
        if z is None:
            # Create depth array based on cube dimensions and digitization
            z = np.arange(self.cfg.cube_shape[-1] * self.cfg.digi)

        d = dict(
            z=z,
            # Shale properties
            shale_vp=self.calc_shale_vp(z),
            shale_vs=self.calc_shale_vs(z),
            shale_rho=self.calc_shale_rho(z),
            # Brine sand properties
            brine_sand_vp=self.calc_brine_sand_vp(z),
            brine_sand_vs=self.calc_brine_sand_vs(z),
            brine_sand_rho=self.calc_brine_sand_rho(z),
            # Oil sand properties
            oil_sand_vp=self.calc_oil_sand_vp(z),
            oil_sand_vs=self.calc_oil_sand_vs(z),
            oil_sand_rho=self.calc_oil_sand_rho(z),
            # Gas sand properties
            gas_sand_vp=self.calc_gas_sand_vp(z),
            gas_sand_vs=self.calc_gas_sand_vs(z),
            gas_sand_rho=self.calc_gas_sand_rho(z),
        )
        
        if store_in_hdf:
            store_1d_trend_dict_to_hdf(self.cfg, d, z)

        return d

    # =========================================================================
    # SHALE PROPERTY CALCULATIONS
    # TODO: Customize these equations for your shale characteristics
    # =========================================================================
    
    @staticmethod
    def calc_shale_rho(z):
        """
        Calculate shale density vs depth.
        
        Parameters
        ----------
        z : np.ndarray
            Depth in meters
            
        Returns
        -------
        np.ndarray
            Density in g/cc
            
        TODO: Replace with your shale density trend equation
        Common approaches:
            - Linear: a * z + b
            - Polynomial: a*z^2 + b*z + c
            - Exponential: a * exp(b * z) + c
            - Gardner's relation based on Vp
        """
        # Placeholder: Linear increase with depth
        # Replace with your actual shale density trend
        rho = 0.0001 * z + 2.0
        return rho

    @staticmethod
    def calc_shale_vp(z):
        """
        Calculate shale Vp vs depth.
        
        Parameters
        ----------
        z : np.ndarray
            Depth in meters
            
        Returns
        -------
        np.ndarray
            Vp in m/s
            
        TODO: Replace with your shale Vp trend equation
        """
        # Placeholder: Linear increase with depth
        vp = 0.3 * z + 1500
        return vp

    @staticmethod
    def calc_shale_vs(z):
        """
        Calculate shale Vs vs depth.
        
        Parameters
        ----------
        z : np.ndarray
            Depth in meters
            
        Returns
        -------
        np.ndarray
            Vs in m/s
            
        TODO: Replace with your shale Vs trend equation
        """
        # Placeholder: Linear increase with depth
        vs = 0.2 * z + 800
        return vs

    # =========================================================================
    # BRINE SAND PROPERTY CALCULATIONS
    # TODO: Customize these equations for your brine sand characteristics
    # =========================================================================
    
    @staticmethod
    def calc_brine_sand_rho(z):
        """
        Calculate brine sand density vs depth.
        
        TODO: Replace with your brine sand density trend equation
        """
        # Placeholder
        rho = 0.00008 * z + 2.1
        return rho

    @staticmethod
    def calc_brine_sand_vp(z):
        """
        Calculate brine sand Vp vs depth.
        
        TODO: Replace with your brine sand Vp trend equation
        """
        # Placeholder
        vp = 0.25 * z + 1800
        return vp

    @staticmethod
    def calc_brine_sand_vs(z):
        """
        Calculate brine sand Vs vs depth.
        
        TODO: Replace with your brine sand Vs trend equation
        """
        # Placeholder
        vs = 0.15 * z + 900
        return vs

    # =========================================================================
    # OIL SAND PROPERTY CALCULATIONS
    # TODO: Customize these equations for your oil sand characteristics
    # =========================================================================
    
    @staticmethod
    def calc_oil_sand_rho(z):
        """
        Calculate oil sand density vs depth.
        
        TODO: Replace with your oil sand density trend equation
        """
        # Placeholder
        rho = 0.00009 * z + 2.05
        return rho

    @staticmethod
    def calc_oil_sand_vp(z):
        """
        Calculate oil sand Vp vs depth.
        
        TODO: Replace with your oil sand Vp trend equation
        Note: Oil typically reduces Vp compared to brine sand
        """
        # Placeholder
        vp = 0.28 * z + 1700
        return vp

    @staticmethod
    def calc_oil_sand_vs(z):
        """
        Calculate oil sand Vs vs depth.
        
        TODO: Replace with your oil sand Vs trend equation
        """
        # Placeholder
        vs = 0.16 * z + 850
        return vs

    # =========================================================================
    # GAS SAND PROPERTY CALCULATIONS
    # TODO: Customize these equations for your gas sand characteristics
    # =========================================================================
    
    @staticmethod
    def calc_gas_sand_rho(z):
        """
        Calculate gas sand density vs depth.
        
        TODO: Replace with your gas sand density trend equation
        Note: Gas significantly reduces density
        """
        # Placeholder
        rho = 0.00007 * z + 1.9
        return rho

    @staticmethod
    def calc_gas_sand_vp(z):
        """
        Calculate gas sand Vp vs depth.
        
        TODO: Replace with your gas sand Vp trend equation
        Note: Gas significantly reduces Vp (Bright Spot indicator)
        """
        # Placeholder
        vp = 0.22 * z + 1600
        return vp

    @staticmethod
    def calc_gas_sand_vs(z):
        """
        Calculate gas sand Vs vs depth.
        
        TODO: Replace with your gas sand Vs trend equation
        Note: Gas has minimal effect on Vs
        """
        # Placeholder
        vs = 0.14 * z + 800
        return vs

    # =========================================================================
    # PROPERTY OBJECT METHODS (Optional but Recommended)
    # These methods create RockProperties objects for advanced calculations
    # =========================================================================
    
    def calc_shale_properties(self, z_rho, z_vp, z_vs):
        """
        Create RockProperties object for shale.
        
        Parameters
        ----------
        z_rho : np.ndarray
            Depth array for density calculation
        z_vp : np.ndarray
            Depth array for Vp calculation
        z_vs : np.ndarray
            Depth array for Vs calculation
            
        Returns
        -------
        RockProperties
            Object containing all elastic properties for shale
        """
        rho = self.calc_shale_rho(z_rho)
        vp = self.calc_shale_vp(z_vp)
        vs = self.calc_shale_vs(z_vs)
        return RockProperties(rho, vp, vs)

    def calc_brine_sand_properties(self, z_rho, z_vp, z_vs):
        """Create RockProperties object for brine sand."""
        rho = self.calc_brine_sand_rho(z_rho)
        vp = self.calc_brine_sand_vp(z_vp)
        vs = self.calc_brine_sand_vs(z_vs)
        return RockProperties(rho, vp, vs)

    def calc_oil_sand_properties(self, z_rho, z_vp, z_vs):
        """Create RockProperties object for oil sand."""
        rho = self.calc_oil_sand_rho(z_rho)
        vp = self.calc_oil_sand_vp(z_vp)
        vs = self.calc_oil_sand_vs(z_vs)
        return RockProperties(rho, vp, vs)

    def calc_gas_sand_properties(self, z_rho, z_vp, z_vs):
        """Create RockProperties object for gas sand."""
        rho = self.calc_gas_sand_rho(z_rho)
        vp = self.calc_gas_sand_vp(z_vp)
        vs = self.calc_gas_sand_vs(z_vs)
        return RockProperties(rho, vp, vs)
