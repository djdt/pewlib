import unittest
import numpy as np
import os.path

from laserlib import io
from laserlib.io import agilent

data_path = os.path.join(os.path.dirname(__file__), "data", "agilent")


class AgilentIOTest(unittest.TestCase):
    def test_load_7700(self):
        data = agilent.load(os.path.join(data_path, "7700.b"))

        self.assertEqual(data.shape, (3, 3))
        self.assertEqual(data.dtype.names, ("A1", "B2"))
        self.assertAlmostEqual(np.sum(data["A1"]), 9.0)
        self.assertAlmostEqual(np.sum(data["B2"]), 0.9)

    def test_load_7500(self):
        data = agilent.load(os.path.join(data_path, "7500.b"))

        self.assertEqual(data.shape, (3, 3))
        self.assertEqual(data.dtype.names, ("A1", "B2"))
        self.assertAlmostEqual(np.sum(data["A1"]), 9.0)
        self.assertAlmostEqual(np.sum(data["B2"]), 0.9)

    def test_load_missing_line(self):
        with self.assertRaises(io.error.LaserLibException):
            agilent.load("laserlib/tests/data/agilent/missing_line.b")
