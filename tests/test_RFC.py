from unittest import TestCase
import numpy as np


class RFC:
    """Reflection Coefficient object — moved from Seismic.py (test/study helper only)."""

    def __init__(self, vp_upper, vs_upper, rho_upper, vp_lower, vs_lower, rho_lower, theta):
        self.theta = theta
        self.vp_upper = vp_upper
        self.vs_upper = vs_upper
        self.rho_upper = rho_upper
        self.vp_lower = vp_lower
        self.vs_lower = vs_lower
        self.rho_lower = rho_lower

    def normal_incidence_rfc(self):
        z0 = self.rho_upper * self.vp_upper
        z1 = self.rho_lower * self.vp_lower
        return (z1 - z0) / (z1 + z0)

    def shuey(self):
        """Use 3-term Shuey to calculate RFC."""
        radians = self._degrees_to_radians()
        sin_squared = np.sin(radians) ** 2
        tan_squared = np.tan(radians) ** 2
        d_vp = self.vp_lower - self.vp_upper
        d_vs = self.vs_lower - self.vs_upper
        d_rho = self.rho_lower - self.rho_upper
        avg_vp = np.mean([self.vp_lower, self.vp_upper])
        avg_vs = np.mean([self.vs_lower, self.vs_upper])
        avg_rho = np.mean([self.rho_lower, self.rho_upper])
        r0 = 0.5 * (d_vp / avg_vp + d_rho / avg_rho)
        g = 0.5 * (d_vp / avg_vp) - 2.0 * avg_vs ** 2 / avg_vp ** 2 * (
            d_rho / avg_rho + 2.0 * d_vs / avg_vs
        )
        f = 0.5 * (d_vp / avg_vp)
        return r0 + g * sin_squared + f * (tan_squared - sin_squared)

    def zoeppritz_reflectivity(self):
        """Calculate PP reflectivity using Zoeppritz equation."""
        theta = self._degrees_to_radians(dtype="complex")
        p = np.sin(theta) / self.vp_upper
        theta2 = np.arcsin(p * self.vp_lower)
        phi1 = np.arcsin(p * self.vs_upper)
        phi2 = np.arcsin(p * self.vs_lower)
        a = self.rho_lower * (1 - 2 * np.sin(phi2) ** 2.0) - self.rho_upper * (
            1 - 2 * np.sin(phi1) ** 2.0
        )
        b = (
            self.rho_lower * (1 - 2 * np.sin(phi2) ** 2.0)
            + 2 * self.rho_upper * np.sin(phi1) ** 2.0
        )
        c = (
            self.rho_upper * (1 - 2 * np.sin(phi1) ** 2.0)
            + 2 * self.rho_lower * np.sin(phi2) ** 2.0
        )
        d = 2 * (
            self.rho_lower * self.vs_lower ** 2 - self.rho_upper * self.vs_upper ** 2
        )
        e = (b * np.cos(theta) / self.vp_upper) + (c * np.cos(theta2) / self.vp_lower)
        f = (b * np.cos(phi1) / self.vs_upper) + (c * np.cos(phi2) / self.vs_lower)
        g = a - d * np.cos(theta) / self.vp_upper * np.cos(phi2) / self.vs_lower
        h = a - d * np.cos(theta2) / self.vp_lower * np.cos(phi1) / self.vs_upper
        d = e * f + g * h * p ** 2
        zoep_pp = (
            f
            * (
                b * (np.cos(theta) / self.vp_upper)
                - c * (np.cos(theta2) / self.vp_lower)
            )
            - h
            * p ** 2
            * (a + d * (np.cos(theta) / self.vp_upper) * (np.cos(phi2) / self.vs_lower))
        ) * (1 / d)
        return np.squeeze(zoep_pp)

    def _degrees_to_radians(self, dtype="float"):
        if dtype == "float":
            return np.radians(self.theta)
        else:
            return np.radians(self.theta).astype("complex")


class TestRFC(TestCase):

    def setUpZeroDeg(self):
        # Upper
        self.vp1 = 2500
        self.vs1 = 1800
        self.rho1 = 2.5
        # Lower
        self.vp2 = 3000
        self.vs2 = 2000
        self.rho2 = 2.69
        self.theta = 0

    def setUpZeroDegRandomValues(self):
        import random
        # Upper
        self.vp1 = random.uniform(1500, 8000)
        self.vs1 = random.uniform(500, 5000)
        self.rho1 = random.uniform(1.0, 3.0)
        # Lower
        self.vp2 = random.uniform(1500, 8000)
        self.vs2 = random.uniform(500, 5000)
        self.rho2 = random.uniform(1.0, 3.0)
        self.theta = 0

    def test_normal_incidence_rfc(self):
        self.setUpZeroDeg()

        zo_rfc = ((self.rho2 * self.vp2) - (self.rho1 * self.vp1)) / (
                (self.rho2 * self.vp2) + (self.rho1 * self.vp1))

        rfc = RFC(self.vp1, self.vs1, self.rho1, self.vp2, self.vs2, self.rho2, self.theta)
        z = rfc.normal_incidence_rfc()

        self.assertEqual(z, zo_rfc)

    def test_normal_incidence_rfc_random(self):
        for _ in range(1001):
            self.setUpZeroDegRandomValues()

            zo_rfc = ((self.rho2 * self.vp2) - (self.rho1 * self.vp1)) / (
                    (self.rho2 * self.vp2) + (self.rho1 * self.vp1))

            rfc = RFC(self.vp1, self.vs1, self.rho1, self.vp2, self.vs2, self.rho2,
                      self.theta)
            z = rfc.normal_incidence_rfc()

            self.assertEqual(z, zo_rfc)

    def test_zoeppritz_reflectivity_selfrees(self):
        self.setUpZeroDeg()
        rfc = RFC(self.vp1, self.vs1, self.rho1, self.vp2, self.vs2, self.rho2, self.theta)
        rfc_zo = rfc.normal_incidence_rfc()
        rfc_zp = rfc.zoeppritz_reflectivity().real  # use real part only
        self.assertAlmostEqual(rfc_zo, rfc_zp, places=15)

    def test_zoeppritz_reflectivity_selfrees_random(self):
        # Test 1000 times selecting vp, vs, rho randomly each time
        for _ in range(1001):
            self.setUpZeroDegRandomValues()
            rfc = RFC(self.vp1, self.vs1, self.rho1, self.vp2, self.vs2, self.rho2,
                      self.theta)
            rfc_zo = rfc.normal_incidence_rfc()
            rfc_zp = rfc.zoeppritz_reflectivity().real  # use real part only
            self.assertAlmostEqual(rfc_zo, rfc_zp, places=15)

    def test__degrees_to_radians(self):
        import random
        import numpy as np
        # Do 1000 random tests
        for counter in range(1001):
            angle_in_degrees = random.uniform(0., 360.)
            radians = angle_in_degrees * np.pi / 180.
            radians_complex = np.asanyarray(radians).astype('complex')

            # Class obj
            self.setUpZeroDeg()
            self.theta = angle_in_degrees
            rfc = RFC(self.vp1, self.vs1, self.rho1, self.vp2, self.vs2, self.rho2,
                      self.theta)
            ang_in_radians = rfc._degrees_to_radians('float')
            ang_in_radians_complex = rfc._degrees_to_radians('complex')

            # Test they are identical to 14 decimal places (fails for some at 15 decimal places...)
            self.assertAlmostEqual(radians, ang_in_radians, places=14)
            self.assertAlmostEqual(radians_complex, ang_in_radians_complex, places=14)
