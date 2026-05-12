"""Stub tests for Faults module.

These tests are placeholders; the full fixture data required to run
meaningful integration assertions is not available in the open-source
repo.  Each test is marked xfail so the suite stays green.
"""
import pytest
from unittest import TestCase


class TestFaults(TestCase):
    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test_fault_parameters(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test__get_fault_mode(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test__fault_params_random(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test__fault_params_self_branching(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test__fault_params_stairs(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test__fault_params_relay_ramps(self):
        self.fail()

    @pytest.mark.xfail(reason="Requires proprietary fixture data", strict=False)
    def test__fault_params_horst_graben(self):
        self.fail()
