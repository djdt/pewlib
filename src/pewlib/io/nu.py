"""
Nu Instruments data import.
"""

import json
import logging
from pathlib import Path
from typing import BinaryIO, Callable, Generator

import numpy as np

logger = logging.getLogger(__name__)


def is_nu_acquisition_directory(path: Path) -> bool:
    """Checks path is directory containing a 'run.info' and 'integrated.index'"""

    if not path.is_dir() or not path.exists():
        return False
    if not path.joinpath("run.info").exists():
        return False
    if not path.joinpath("integrated.index").exists():  # pragma: no cover
        return False

    return True


def is_nu_image_directory(path: Path) -> bool:
    """Checks if directory has a 'laser.info' file and some acquistions."""
    if len(list(path.glob("laser.info"))) == 0:
        return False

    return any(is_nu_acquisition_directory(dir) for dir in path.iterdir())


def blanking_regions_from_autob(
    autob_events: np.ndarray,
    num_acc: int,
    start_coef: tuple[float, float],
    end_coef: tuple[float, float],
) -> tuple[list[tuple[int, int]], list[np.ndarray]]:
    """Extract blanking regions from autoblank data.

    Args:
        autob: list of events from `read_nu_autob_binary`
        num_acc: number of accumulations per acquisition
        start_coef: blanker open coefs 'BlMassCalStartCoef'
        end_coef: blanker close coefs 'BlMassCalEndCoef'

    Returns:
        list of (start, end) of each region, array of (start, end) masses
    """
    regions: list[tuple[int, int]] = []
    mass_regions = []

    start_event = None
    for event in autob_events:
        if event["type"] == 0 and start_event is None:
            start_event = event
        elif event["type"] == 1 and start_event is not None:
            regions.append(
                (
                    int(start_event["acq_number"] // num_acc) - 1,
                    int(event["acq_number"] // num_acc) - 1,
                )
            )

            start_masses = (
                start_coef[0]
                + start_coef[1]
                * start_event["edges"][: start_event["num_edges"]][::2]
                * 1.25
            ) ** 2
            end_masses = (
                end_coef[0]
                + end_coef[1]
                * start_event["edges"][: start_event["num_edges"]][1::2]
                * 1.25
            ) ** 2
            valid = start_masses < end_masses
            mass_regions.append(
                np.stack([start_masses[valid], end_masses[valid]], axis=0)
            )

            start_event = None

    return regions, mass_regions


def apply_autoblanking(
    autob_events: np.ndarray,
    signals: np.ndarray,
    masses: np.ndarray,
    num_acc: int,
    start_coef: tuple[float, float],
    end_coef: tuple[float, float],
) -> np.ndarray:
    """Apply the auto-blanking to the integrated data.
    There must be one cycle / segment and no missing acquisitions / data!

    Args:
        autob: list of events from `read_nu_autob_binary`
        signals: 2d array of signals from `get_signals_from_nu_data`
        masses: 1d array of masses, from `get_masses_from_nu_data`
        num_acc: number of accumulations per acquisition
        start_coef: blanker open coefs 'BlMassCalStartCoef'
        end_coef: blanker close coefs 'BlMassCalEndCoef'

    Returns:
        blanked data
    """
    regions, mass_regions_list = blanking_regions_from_autob(
        autob_events, num_acc, start_coef, end_coef
    )
    for region, mass_regions in zip(regions, mass_regions_list):
        mass_idx = np.searchsorted(masses, mass_regions)
        # There are a bunch of useless blanking regions
        mass_idx = mass_idx[mass_idx[:, 0] != mass_idx[:, 1]]
        for s, e in mass_idx:
            signals[region[0] : region[1], s:e] = np.nan

    return signals


def read_autob_binary(
    path: Path,
    first_cyc_number: int | None = None,
    first_seg_number: int | None = None,
    first_acq_number: int | None = None,
) -> np.ndarray:
    data_dtype = np.dtype(
        [
            ("cyc_number", np.uint32),
            ("seg_number", np.uint32),
            ("acq_number", np.uint32),
            ("trig_start_time", np.uint32),
            ("trig_end_time", np.uint32),
            ("type", np.uint8),
            ("num_edges", np.int32),
            ("edges", np.uint32, 12),  # so far 12 is the maximum
        ]
    )

    def read_autoblank_events(fp: BinaryIO) -> Generator[np.ndarray, None, None]:
        while fp:
            partial = fp.read(25)
            if len(partial) < 25:
                return
            autob = np.zeros(1, dtype=data_dtype)
            autob.data.cast("B")[:25] = partial
            num = autob["num_edges"][0]
            if num > 0:
                autob["edges"][:num] = np.frombuffer(fp.read(num * 4), dtype=np.uint32)
            yield autob

    with path.open("rb") as fp:
        autob = np.concatenate(list(read_autoblank_events(fp)))

    if autob.size > 0:  # pragma: no cover
        if first_cyc_number is not None and autob[0]["cyc_number"] != first_cyc_number:
            raise ValueError("read_integ_binary: incorrect FirstCycNum")
        if first_seg_number is not None and autob[0]["seg_number"] != first_seg_number:
            raise ValueError("read_integ_binary: incorrect FirstSegNum")
        if first_acq_number is not None and autob[0]["acq_number"] != first_acq_number:
            raise ValueError("read_integ_binary: incorrect FirstAcqNum")

    return autob


def read_integ_binary(
    path: Path,
    first_cyc_number: int | None = None,
    first_seg_number: int | None = None,
    first_acq_number: int | None = None,
) -> np.ndarray:
    def integ_dtype(size: int) -> np.dtype:
        data_dtype = np.dtype(
            {
                "names": ["center", "signal"],
                "formats": [np.float32, np.float32],
                "itemsize": 4 + 4 + 4 + 1,  # unused f32, unused i8
            }
        )
        return np.dtype(
            [
                ("cyc_number", np.uint32),
                ("seg_number", np.uint32),
                ("acq_number", np.uint32),
                ("num_results", np.uint32),
                ("result", data_dtype, size),
            ]
        )

    with path.open("rb") as fp:
        cyc_number = int.from_bytes(fp.read(4), "little")
        if (
            first_cyc_number is not None and cyc_number != first_cyc_number
        ):  # pragma: no cover
            raise ValueError("read_integ_binary: incorrect FirstCycNum")
        seg_number = int.from_bytes(fp.read(4), "little")
        if (
            first_seg_number is not None and seg_number != first_seg_number
        ):  # pragma: no cover
            raise ValueError("read_integ_binary: incorrect FirstSegNum")
        acq_number = int.from_bytes(fp.read(4), "little")
        if (
            first_acq_number is not None and acq_number != first_acq_number
        ):  # pragma: no cover
            raise ValueError("read_integ_binary: incorrect FirstAcqNum")
        num_results = int.from_bytes(fp.read(4), "little")
        fp.seek(0)

        return np.frombuffer(fp.read(), dtype=integ_dtype(num_results))


def read_pulse_binary(
    path: Path,
    first_cyc_number: int | None = None,
    first_seg_number: int | None = None,
    first_acq_number: int | None = None,
) -> np.ndarray:
    dtype = np.dtype(
        [
            ("cyc_number", np.uint32),
            ("seg_number", np.uint32),
            ("acq_number", np.uint32),
            ("overflow", np.bool),
        ]
    )
    with path.open("rb") as fp:
        pulse = np.frombuffer(fp.read(), dtype=dtype)

    if pulse.size > 0:  # pragma: no cover
        if first_cyc_number is not None and pulse[0]["cyc_number"] != first_cyc_number:
            raise ValueError("read_integ_binary: incorrect FirstCycNum")
        if first_seg_number is not None and pulse[0]["seg_number"] != first_seg_number:
            raise ValueError("read_integ_binary: incorrect FirstSegNum")
        if first_acq_number is not None and pulse[0]["acq_number"] != first_acq_number:
            raise ValueError("read_integ_binary: incorrect FirstAcqNum")

    return pulse


def read_binaries_in_index(
    root: Path,
    index: list[dict],
    binary_ext: str,
    binary_read_fn: Callable[[Path, int, int, int], np.ndarray],
    cyc_number: int | None = None,
    seg_number: int | None = None,
) -> list[np.ndarray]:
    """Reads Nu binaries listed in an index file.

    Args:
        root: directory containing files and index
        index: list of indices from json.loads
        binary_ext: extension of binary files, e.g. '.integ'
        binary_read: function to read binary file
        cyc_number: restrict to cycle
        seg_number: restrict to segments

    Returns:
        binary data as a list of arrays
    """
    datas = []
    for idx in index:
        binary_path = root.joinpath(f"{idx['FileNum']}.{binary_ext}")
        if binary_path.exists():
            data = binary_read_fn(
                binary_path,
                idx["FirstCycNum"],
                idx["FirstSegNum"],
                idx["FirstAcqNum"],
            )
            if cyc_number is not None:
                data = data[data["cyc_number"] == cyc_number]
            if seg_number is not None:
                data = data[data["seg_number"] == seg_number]
            datas.append(data)
        else:
            logger.warning(  # pragma: no cover, missing files
                f"collect_data_from_index: missing data file {idx['FileNum']}.{binary_ext}, skipping"
            )
    return datas


def eventtime_from_info(info: dict) -> float:
    """Reads the dwelltime (total acquistion time) from run.info.
    Rounds to the nearest ns.

    Args:
        info: dict of parameters, as returned by `read_nu_directory`

    Returns:
        dwelltime in s
    """
    seg = info["SegmentInfo"][0]
    acqtime = seg["AcquisitionPeriodNs"] * 1e-9
    accumulations = info["NumAccumulations1"] * info["NumAccumulations2"]
    return np.around(acqtime * accumulations, 9)


def masses_from_integ(
    integ: np.ndarray, cal_coef: tuple[float, float], segment_delays: dict[int, float]
) -> np.ndarray:
    """Converts Nu peak centers into masses.

    Args:
        integ: from `read_integ_binary`
        cal_coef: from run.info 'MassCalCoefficients'
        segment_delays: dict of segment nums and delays from `SegmentInfo`

    Returns:
        2d array of masses
    """

    delays = np.zeros(max(segment_delays.keys()))
    for k, v in segment_delays.items():
        delays[k - 1] = v
    delays = np.atleast_1d(delays[integ["seg_number"] - 1])

    masses = (integ["result"]["center"] * 0.5) + delays[:, None]
    # Convert from time to mass (sqrt(m/q) = a + t * b)
    return (cal_coef[0] + masses * cal_coef[1]) ** 2


def get_times_from_data(data: np.ndarray, run_info: dict) -> np.ndarray:
    times = 0.0
    seg_times = np.array(
        [
            seg["AcquisitionPeriodNs"] * seg["AcquisitionCount"]
            for seg in run_info["SegmentInfo"]
        ]
    )
    seg_periods = np.array(
        [seg["AcquisitionPeriodNs"] for seg in run_info["SegmentInfo"]]
    )
    times = np.sum(seg_times) * (data["cyc_number"] - 1)
    times += np.cumsum(np.concatenate([[0], seg_times]))[data["seg_number"] - 1]
    times += data["acq_number"] * seg_periods[data["seg_number"] - 1]
    return times


def read_laser_acquisition(
    path: str | Path,
    autoblank: bool = True,
    cycle: int | None = None,
    segment: int | None = None,
    raw: bool = False,
    max_integs: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Read the Nu Instruments raw data directory, retuning data and run info.

    Directory must contain 'run.info', 'integrated.index' and at least one '.integ'
    file. Data is read from '.integ' files listed in the 'integrated.index' and
    are checked for correct starting cycle, segment and acquisition numbers.

    Args:
        path: path to data directory
        max_integ_files: maximum number of files to read
        autoblank: apply autoblanking to overrange regions
        cycle: limit import to cycle
        segment: limit import to segment
        raw: return raw ADC counts
        max_integs: only read the first n integ files

    Returns:
        signals in counts
        masses from first acquisition
        times in s
        laser pulse data in s
        dict of parameters from run.info
    """

    path = Path(path)
    if not is_nu_acquisition_directory(path):  # pragma: no cover
        raise ValueError("read_nu_directory: missing 'run.info' or 'integrated.index'")

    with path.joinpath("run.info").open("r") as fp:
        run_info = json.load(fp)

    with path.joinpath("autob.index").open("r") as fp:
        autob_index = json.load(fp)
    with path.joinpath("integrated.index").open("r") as fp:
        integ_index = json.load(fp)
    with path.joinpath("pulse.index").open("r") as fp:
        pulse_index = json.load(fp)

    if max_integs is not None:  # pragma: no cover
        integ_index = integ_index[:max_integs]

    # Collect integrated data
    integs = np.concatenate(
        read_binaries_in_index(
            path,
            integ_index,
            "integ",
            read_integ_binary,
            cyc_number=cycle,
            seg_number=segment,
        )
    )

    # Collect laser trigger data
    pulses = np.concatenate(
        read_binaries_in_index(
            path,
            pulse_index,
            "pulse",
            read_pulse_binary,
            cyc_number=cycle,
            seg_number=segment,
        ),
    )

    # Get masses from data
    segment_delays = {
        s["Num"]: s["AcquisitionTriggerDelayNs"] for s in run_info["SegmentInfo"]
    }
    masses = masses_from_integ(
        integs[0], run_info["MassCalCoefficients"], segment_delays
    )[0]

    signals = integs["result"]["signal"]

    if not raw:
        signals /= run_info["AverageSingleIonArea"]

    # Blank out overrange regions
    if autoblank:
        accumulations = run_info["NumAccumulations1"] * run_info["NumAccumulations2"]
        autobs = np.concatenate(
            read_binaries_in_index(
                path,
                autob_index,
                "autob",
                read_autob_binary,
                cyc_number=cycle,
                seg_number=segment,
            )
        )
        signals = apply_autoblanking(
            autobs,
            signals,
            masses,
            accumulations,
            run_info["BlMassCalStartCoef"],
            run_info["BlMassCalEndCoef"],
        )

    pulse_times = get_times_from_data(pulses, run_info) * 1e-9
    times = get_times_from_data(integs, run_info) * 1e-9

    return signals, masses, times, pulse_times, run_info


def apply_trigger_correction(times: np.ndarray, corrections: dict) -> np.ndarray:
    """Return times with trigger time removed.

    Args:
        times: times in seconds
        corrections: corrections from TriggerCorrections.dat

    Returns:
        corrected times
    """
    if corrections["CorrectionMode"] == 0:
        return times + corrections["Transit1Time"] * 1e-3
    else:  # pragma: no cover
        c1 = (
            (corrections["Transit2time"] - corrections["Transit1Time"])
            / (corrections["Trigger2Time"] - corrections["Trigger1Time"])
            * 1e-3
        )
        c2 = corrections["Transit1Time"] - (c1 * corrections["Trigger1Time"]) * 1e-3
        return c1 * times + c2


def read_laser_image(
    path: Path | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Read a laser image from a Nu Vitesse ICP-TOF-MS.

    Calls ``read_laser_acquistion`` on valid sub directories and concatenates.

    Args:
        path: path to the Image directory

    Returns:
        signals
        masses
        times
        pulses
        laser_info
    """
    path = Path(path)

    if not is_nu_image_directory(path):  # pragma: no cover
        try:
            path = next(d for d in path.iterdir() if is_nu_image_directory(d))
            logger.info(f"invalid image path, reading '/{path.stem}")
        except StopIteration:
            raise ValueError(f"{path} is not a valid Nu image directory")

    with Path(path.joinpath("laser.info")).open("r") as fp:
        laser_info = json.load(fp)
    with Path(path.joinpath("TriggerCorrections.dat")).open("r") as fp:
        corrections = json.load(fp)

    masses = None
    signals_list = []
    times_list = []
    pulse_list = []

    acqusitions = sorted(
        [d for d in path.iterdir() if is_nu_acquisition_directory(d)],
        key=lambda d: int(d.stem),
    )

    for i, acq_dir in enumerate(acqusitions):
        _signals, _masses, _times, _pulses, _info = read_laser_acquisition(acq_dir)
        if masses is None:
            masses = _masses
        elif not np.all(masses == _masses):  # pragma: no cover
            logger.warning("masses differ across laser lines")

        signals_list.append(_signals)
        times_list.append(_times)
        pulse_list.append(_pulses)

    if masses is None:  # pragma: no cover
        raise ValueError("masses were not read from any laser directory")

    if corrections["CorrectionMode"] != 0:  # pragma: no cover
        raise NotImplementedError("only correction mode 0 is supported")

    correction = corrections["Transit1Time"] * 1e-3

    signals = np.concatenate(signals_list, axis=0)
    times = np.concatenate(times_list)
    times -= correction
    pulses = np.concatenate(pulse_list)

    return signals, masses, times, pulses, laser_info
