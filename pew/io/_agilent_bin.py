import numpy as np

from typing import BinaryIO

msscan_magic_number = 257
msprofile_magic_number = 258
msscan_xspecific_magic_number = 275

msscan_header_size = 68
msprofile_header_size = 68
msscan_xspecific_header_size = 68

scan_record_dtype = np.dtype(
    [
        ("ScanID", np.int32),
        ("ScanMethodID", np.int32),
        ("TimeSegmentID", np.int32),
        ("ScanTime", np.float64),
        ("MSLevel", np.int32),
        ("ScanType", np.int32),
        ("TIC", np.float64),
        ("BasePeakMZ", np.float64),
        ("BasePeakValue", np.float64),
        ("Status", np.int32),
        ("IonMode", np.int32),
        ("IonPolarity", np.int32),
        ("SamplingPeriod", np.float64),
        (
            "SpectrumParamValues",
            np.dtype(
                [
                    ("SpectrumFormatID", np.int32),
                    ("SpectrumOffset", np.int64),
                    ("ByteCount", np.int32),
                    ("PointCount", np.int32),
                    ("MinX", np.float64),
                    ("MaxX", np.float64),
                    ("MinY", np.float64),
                    ("MaxY", np.float64),
                ]
            ),
        ),
        (
            "XSpecificParamType",
            np.dtype([("Offset", np.int64), ("ByteCount", np.int32)]),
        ),
    ]
)


def get_ms_profile_dtype(n: int):
    # Not sure about these
    return np.dtype(
        [
            ("ID", np.float32, n),
            ("MinY", np.float64, n),
            ("MaxY", np.float64, n),
            ("TotalTime", np.float64),
            ("Time", np.float64, n - 1),
        ]
    )


def read_ms_scan_xspecific(path: str) -> np.ndarray:
    with open(path, "rb") as fp:
        if int.from_bytes(fp.read(4), "little") != msscan_xspecific_magic_number:
            raise IOError("Invalid header for MSScan.")
        fp.seek(msscan_xspecific_header_size)
        return np.frombuffer(fp.read(), dtype=[(("_", np.int32), ("MZ", np.float64))])


def read_ms_scan(path: str):
    with open(path, "rb") as fp:
        if int.from_bytes(fp.read(4), "little") != msscan_magic_number:
            raise IOError("Invalid header for MSScan.")
        fp.seek(msscan_header_size + 20)
        offset = int.from_bytes(fp.read(4), "little")
        fp.seek(offset)
        return np.frombuffer(fp.read(), dtype=scan_record_dtype)


def read_ms_profile(path: str, n: int) -> np.ndarray:
    dtype = get_ms_profile_dtype(n)
    with open(path, "rb") as fp:
        if int.from_bytes(fp.read(4), "little") != msprofile_magic_number:
            raise IOError("Invalid header for MSProfile.")
        fp.seek(msprofile_header_size)
        return np.frombuffer(fp.read(), dtype=dtype)


def parse_msts_xaddition_xml(path: str) -> None:
    pass


print(read_ms_scan("/home/tom/Downloads/20200630_agar_test_1.b/001.d/AcqData/MSScan.bin")["XSpecificParamType"])
# Size = 100 scan records
print(
    read_ms_profile(
        "/home/tom/Downloads/20200630_agar_test_1.b/001.d/AcqData/MSProfile.bin", 4
    ).size
)
