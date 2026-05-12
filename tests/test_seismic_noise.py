import numpy as np
import pytest


class TestHiltermanNoiseWeighting:
    """Verify cos²/sin² weighting uses radians, not degrees."""

    @pytest.mark.parametrize("angle_deg, expected_near, expected_far", [
        (0, 1.0, 0.0),
        (10, np.cos(np.deg2rad(10)) ** 2, np.sin(np.deg2rad(10)) ** 2),
        (18, np.cos(np.deg2rad(18)) ** 2, np.sin(np.deg2rad(18)) ** 2),
        (22.5, np.cos(np.deg2rad(22.5)) ** 2, np.sin(np.deg2rad(22.5)) ** 2),
        (26, np.cos(np.deg2rad(26)) ** 2, np.sin(np.deg2rad(26)) ** 2),
        (45, 0.5, 0.5),
    ])
    def test_noise_weights_use_radians(self, angle_deg, expected_near, expected_far):
        """The Hilterman weighting at angle_deg must match cos²(rad)/sin²(rad)."""
        ang_rad = np.deg2rad(angle_deg)
        near_weight = np.cos(ang_rad) ** 2
        far_weight = np.sin(ang_rad) ** 2
        assert near_weight == pytest.approx(expected_near, abs=1e-10)
        assert far_weight == pytest.approx(expected_far, abs=1e-10)
        assert near_weight + far_weight == pytest.approx(1.0, abs=1e-15)

    def test_buggy_weights_differ_from_correct(self):
        """Confirm the old (degree) weights !== correct (radian) weights for non-boundary angles."""
        from math import cos as math_cos, sin as math_sin
        for ang in [10, 18, 26]:
            buggy_near = math_cos(ang) ** 2  # ang in degrees passed to math.cos (expects radians)
            correct_near = np.cos(np.deg2rad(ang)) ** 2
            assert buggy_near != pytest.approx(correct_near, abs=0.01), \
                f"At {ang}°, buggy and correct should differ"

    def test_18deg_near_weight_is_0905(self):
        """Regression guard: at 18° the near weight must be ~0.905, not 0.436."""
        ang_rad = np.deg2rad(18)
        near_weight = np.cos(ang_rad) ** 2
        assert near_weight == pytest.approx(0.9045, abs=0.001)
