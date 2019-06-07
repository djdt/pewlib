import unittest
import numpy as np
import os.path

from laserlib import io
from laserlib.io import thermo

data_path_csv = os.path.join(os.path.dirname(__file__), "data", "csv")
data_path = os.path.join(os.path.dirname(__file__), "data", "thermo")


class ThermoIOTest(unittest.TestCase):
    def test_fail_csv(self):
        with self.assertRaises(io.error.LaserLibException):
            thermo.load(os.path.join(data_path_csv, "csv.csv"))

    def test_load_icap(self):
        data = thermo.load(os.path.join(data_path, "icap.csv"))

        self.assertEqual(data.shape, (3, 3))
        self.assertEqual(data.dtype.names, ("1A", "2B"))
        self.assertAlmostEqual(np.sum(data["1A"]), 9.0)
        self.assertAlmostEqual(np.sum(data["2B"]), 0.9)
