import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from netCDF4 import Dataset

from dronevision.bathymetry import contours, parse_levels


class BathymetryTest(unittest.TestCase):
    def test_levels_are_sorted_and_checked(self):
        self.assertEqual(parse_levels("50, 10; 20, 10"), [10.0, 20.0, 50.0])
        with self.assertRaises(ValueError):
            parse_levels("0, 10")

    def test_reads_only_local_geographic_subset_and_builds_isobaths(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gebco.nc"
            with Dataset(path, "w") as dataset:
                dataset.createDimension("lat", 121)
                dataset.createDimension("lon", 161)
                lat = dataset.createVariable("lat", "f8", ("lat",))
                lon = dataset.createVariable("lon", "f8", ("lon",))
                elevation = dataset.createVariable("elevation", "i2", ("lat", "lon"))
                lat.standard_name = "latitude"
                lon.standard_name = "longitude"
                elevation.standard_name = "height"
                lat[:] = np.linspace(69.5, 70.5, 121)
                lon[:] = np.linspace(29.0, 31.0, 161)
                yy, xx = np.meshgrid(lat[:], lon[:], indexing="ij")
                elevation[:] = np.round(-10 - 180 * (xx - 29) / 2 + 15 * np.sin((yy - 70) * 12))
            with patch.dict(os.environ, {"DRONEVISION_BATHYMETRY": str(path)}):
                result = contours(70, 30, {"east": [-30_000, 30_000], "north": [-30_000, 30_000]},
                                  [20, 50, 100, 150])
            self.assertNotIn("error", result)
            self.assertTrue(all(layer["lines"] for layer in result["contours"]))
            self.assertLess(max(abs(point[0]) for layer in result["contours"]
                                for line in layer["lines"] for point in line), 100_000)


if __name__ == "__main__":
    unittest.main()
