import unittest
import numpy as np
import os.path
import tempfile

from laserlib import io
from laserlib.io import csv

data_path = os.path.join(os.path.dirname(__file__), "data", "csv")
data_path_thermo = os.path.join(os.path.dirname(__file__), "data", "thermo")


class CSVIOTest(unittest.TestCase):
    def test_fail_thermo(self):
        with self.assertRaises(io.error.LaserLibException):
            csv.load(os.path.join(data_path_thermo, "icap.csv"))

    def test_load(self):
        data = csv.load(os.path.join(data_path, "csv.csv"))

        self.assertEqual(data.dtype.names, ("CSV",))
        self.assertEqual(data.shape, (3, 3))
        self.assertEqual(np.sum(data["CSV"]), 12.0)

    def test_save(self):
        data = csv.load(os.path.join(data_path, "csv.csv"))
        temp = tempfile.NamedTemporaryFile()
        csv.save(temp.name, data["CSV"])
        self.assertTrue(np.all(data["CSV"] == csv.load(temp.name)["CSV"]))

        temp.close()
