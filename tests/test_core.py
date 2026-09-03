from __future__ import annotations

import unittest

import numpy as np
from astropy.io import fits

from untangling_eit_waves.cadence import (
    displacement_mm,
    endpoint_quantization_sigma_kms,
    quantization_speed_kms,
)
from untangling_eit_waves.energy import (
    front_fraction,
    full_sun_flux_erg_cm2_s,
    heating_requirement_percent,
    kinetic_energy_range_erg,
)
from untangling_eit_waves.wcs_geometry import check_wcs, pixel_to_world, world_to_pixel


def synthetic_header() -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 1024
    header["NAXIS2"] = 1024
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 512.5
    header["CRPIX2"] = 512.5
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CDELT1"] = 2.4
    header["CDELT2"] = 2.4
    header["RSUN_OBS"] = 960.0
    return header


class TestCadence(unittest.TestCase):
    def test_displacement(self):
        self.assertAlmostEqual(displacement_mm(500, 120), 60.0)

    def test_ring_quantization(self):
        self.assertAlmostEqual(quantization_speed_kms(91.6, 110), 832.7272727)

    def test_endpoint_scale(self):
        self.assertAlmostEqual(endpoint_quantization_sigma_kms(91.6, 420), 154.2, places=1)


class TestEnergy(unittest.TestCase):
    def test_podladchikova_range(self):
        lo, hi = kinetic_energy_range_erg(2e8, 14, (0.05, 0.10), 3e27)
        self.assertTrue(2e21 < lo < 4e21)
        self.assertTrue(1e22 < hi < 1.3e22)

    def test_innes_range(self):
        lo, hi = kinetic_energy_range_erg(2e8, 45, (0.05, 0.10), 3e28)
        self.assertTrue(2e23 < lo < 4e23)
        self.assertTrue(1e24 < hi < 1.3e24)

    def test_population_flux_and_percentage(self):
        flux = full_sun_flux_erg_cm2_s(1e23, 470)
        self.assertAlmostEqual(flux, 8.93e-3, delta=5e-5)
        self.assertAlmostEqual(heating_requirement_percent(flux), 8.93e-6, delta=5e-8)

    def test_front_fraction_requires_same_event_denominator(self):
        self.assertAlmostEqual(front_fraction(2e22, 1e24), 0.02)
        with self.assertRaises(ValueError):
            front_fraction(2e22, 0.0)


class TestWCS(unittest.TestCase):
    def test_check_and_roundtrip(self):
        header = synthetic_header()
        self.assertTrue(check_wcs(header).valid)
        x = np.array([100.0, 511.5, 900.0])
        y = np.array([200.0, 511.5, 850.0])
        lon, lat = pixel_to_world(header, x, y)
        x2, y2 = world_to_pixel(header, lon, lat)
        np.testing.assert_allclose(x2, x, atol=1e-7)
        np.testing.assert_allclose(y2, y, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
