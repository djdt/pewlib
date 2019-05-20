import unittest
import numpy as np
import os.path
import tempfile

from laserlib import io
from laserlib.io import csv

data_path = os.path.join(os.path.dirname(__file__), "data", "csv")


class CSVIOTest(unittest.TestCase):
    def test_load_raw(self):
        data = csv.load_raw(os.path.join(data_path, "csv.csv"))

        self.assertEqual(data.shape, (3, 3))
        self.assertEqual(np.sum(data), 12.0)

        data = csv.load_raw(os.path.join(data_path, "csv_header.csv"))

        self.assertEqual(data.shape, (3, 3))
        self.assertEqual(np.sum(data), 12.0)

    def test_load(self):
        laser = csv.load(os.path.join(data_path, "csv.csv"))

        self.assertIn("_", laser.data)
        self.assertEqual(laser.data["_"].data.shape, (3, 3))
        self.assertEqual(np.sum(laser.data["_"].data), 12.0)
        self.assertEqual(laser.data["_"].calibration.intercept, 0.0)
        self.assertEqual(laser.data["_"].calibration.gradient, 1.0)
        self.assertEqual(laser.data["_"].calibration.unit, "")

    def test_load_header(self):
        laser = csv.load(os.path.join(data_path, "csv_header.csv"), read_header=True)

        self.assertIn("_", laser.data)
        self.assertEqual(laser.data["_"].data.shape, (3, 3))
        self.assertEqual(np.sum(laser.data["_"].data), 12.0)

        self.assertEqual(laser.data["_"].calibration.intercept, 2.0)
        self.assertEqual(laser.data["_"].calibration.gradient, 3.0)
        self.assertEqual(laser.data["_"].calibration.unit, "u")

        self.assertEqual(laser.config.spotsize, 0.1)
        self.assertEqual(laser.config.speed, 0.2)
        self.assertEqual(laser.config.scantime, 0.3)

    def test_load_missing_header(self):
        with self.assertRaises(io.error.LaserLibException):
            csv.load(os.path.join(data_path, "csv.csv"), read_header=True)

    def test_save(self):
        data = csv.load_raw(os.path.join(data_path, "csv.csv"))
        temp = tempfile.NamedTemporaryFile()
        csv.save(temp.name, data, header="")
        self.assertTrue(np.all(data == csv.load_raw(temp.name)))

        temp.close()

    def test_save_header(self):
        laser = csv.load(os.path.join(data_path, "csv_header.csv"), read_header=True)
        temp = tempfile.NamedTemporaryFile()
        csv.save(
            temp.name, laser.data["_"].data, header=csv.make_header(laser, "_")
        )

        laser2 = csv.load(temp.name, read_header=True)

        temp.close()

        self.assertEqual(laser.config.spotsize, laser2.config.spotsize)
        self.assertEqual(laser.config.speed, laser2.config.speed)
        self.assertEqual(laser.config.scantime, laser2.config.scantime)

        self.assertIn("_", laser2.data)

        self.assertTrue(np.all(laser.data["_"].data == laser2.data["_"].data))

        self.assertEqual(
            laser.data["_"].calibration.intercept,
            laser2.data["_"].calibration.intercept,
        )
        self.assertEqual(
            laser.data["_"].calibration.gradient,
            laser2.data["_"].calibration.gradient,
        )
        self.assertEqual(
            laser.data["_"].calibration.unit,
            laser2.data["_"].calibration.unit,
        )
