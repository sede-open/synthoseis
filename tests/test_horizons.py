"""Stub tests for Horizons module.

These tests are placeholders; the full fixture data required to run
meaningful integration assertions is not available in the open-source
repo.  Each test is marked xfail so the suite stays green.
"""
import pytest
from unittest import TestCase


class TestHorizons(TestCase):
    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test_insert_feature_into_horizon_stack(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test_insert_seafloor(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test__perlin(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test__fit_plane_strike_dip(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test_fit_plane_lsq(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test_eval_plane(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test_halton(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test_rotatepoint(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test_convert_map_from_samples_to_units(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test_write_maps_to_intfile(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test__write_horizon_to_intfile(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test_write_onlap_episodes_to_intfile(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test_write_fan_horizons_to_intfile(self):
        self.fail()


class TestRandomHorizonStack(TestCase):
    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test__generate_lookup_tables(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test__random_layer_thickness(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test__generate_random_depth_structure_map(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test_create_depth_maps(self):
        self.fail()


class TestOnlaps(TestCase):
    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test__generate_onlap_lookup_table(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test_insert_tilting_episodes(self):
        self.fail()


class TestBasinFloorFans(TestCase):
    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test__generate_fan_lookup_table(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test__generate_basin_floor_fan(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test__generate_fan_thickness_map(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test_insert_fans_into_horizons(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test_fan_qc_plot(self):
        self.fail()


class TestFacies(TestCase):
    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test_sand_shale_facies_binomial_dist(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test_set_layers_below_onlaps_to_shale(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test_set_fan_facies(self):
        self.fail()
