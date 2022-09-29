import argparse
from pathlib import Path
import time
import sys

from pewlib import io
from pewlib.process import filters
from pewlib import Laser, Config
from pewlib import __version__

from typing import List, Optional


def load(path: Path) -> Laser:
    config = Config()
    info = {
        "Name": path.stem,
        "File Path": str(path.resolve()),
        "Import Date": time.strftime(
            "%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time())
        ),
        "Import Path": str(path.resolve()),
        "Import Version pewlib": __version__,
    }
    params = {}
    if path.is_dir():
        if path.suffix.lower() == ".b":
            data = None
            for methods in [["batch_xml", "batch_csv"], ["acq_method_xml"]]:
                try:
                    data, params = io.agilent.load(
                        path, collection_methods=methods, full=True
                    )
                    info.update(io.agilent.load_info(path))
                    break
                except ValueError:
                    pass
            if data is None:
                raise ValueError(f"unable to import batch '{path.name}'")
        elif io.perkinelmer.is_valid_directory(path):
            data, params = io.perkinelmer.load(path, full=True)
            info["Instrument Vendor"] = "PerkinElemer"
        elif io.csv.is_valid_directory(path):
            data, params = io.csv.load(path, full=True)
        else:  # pragma: no cover
            raise ValueError(f"unknown extention '{path.suffix}'")
    else:
        if path.suffix.lower() == ".npz":
            laser = io.npz.load(path)
            return laser
        if path.suffix.lower() == ".csv":
            sample_format = io.thermo.icap_csv_sample_format(path)
            if sample_format in ["columns", "rows"]:
                data, params = io.thermo.load(path, full=True)
                info["Instrument Vendor"] = "Thermo"
            else:
                data = io.textimage.load(path, name="_element_")
        elif path.suffix.lower() in [".txt", ".text"]:
            data = io.textimage.load(path, name="_element_")
        else:  # pragma: no cover
            raise ValueError(f"unknown extention '{path.suffix}'")

    if "spotsize" in params:
        config.spotsize = params["spotsize"]
    if "speed" in params:
        config.speed = params["speed"]
    if "scantime" in params:
        config.scantime = params["scantime"]

    return Laser(data=data, config=config, info=info)


def create_parser_and_parse_args() -> argparse.Namespace:
    def check_exists(file: str) -> Path:
        path = Path(file)
        if not path.exists():
            raise argparse.ArgumentTypeError("path does not exist")
        return path

    def path_with_suffix(file: str, suffixes: List[str]) -> Path:
        path = Path(file)
        if path.suffix.lower() not in suffixes:
            raise argparse.ArgumentTypeError(
                f"must have a suffix in {', '.join(suffixes)}"
            )
        return path

    class FilterAction(argparse.Action):
        def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: List[str],
            option_string: Optional[str] = None,
        ):
            if len(values) < 3:
                parser.error("argument filter: missing TYPE, SIZE or THRESHOLD")
            if values[0] not in ["mean", "median"]:
                parser.error(
                    "argument filter: TYPE must be specified as 'mean' or 'median'"
                )
            try:
                size = int(values[1])
                if size < 2:
                    raise ValueError
            except ValueError:
                parser.error("argument filter: SIZE of rolling window must be int > 1")
            try:
                t = float(values[2])
                if t < 0:
                    raise ValueError
            except ValueError:
                parser.error(
                    "argument filter: THRESHOLD of rolling window must be float > 0"
                )
            setattr(namespace, self.dest, (values[0], size, t, values[3:]))

    valid_formats = [".csv", ".npz", ".vtk"]

    ioparser = argparse.ArgumentParser(add_help=False)
    ioparser.add_argument(
        "input", type=check_exists, nargs="+", help="path to input file(s)"
    )
    output = ioparser.add_argument_group("output arguments")
    output.add_argument(
        "--format",
        choices=valid_formats,
        default=".npz",
        help="output file format",
    )
    output.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="path to output file or directory, defaults to the input directory. "
        "if a file then the extension must match the format",
    )
    parser = argparse.ArgumentParser(
        description="CLI for reading and processing of LA-ICP-MS data.",
    )
    subparsers = parser.add_subparsers(title="command", dest="command", required=True)

    convert = subparsers.add_parser("convert", parents=[ioparser])
    convert.add_argument(
        "--config",
        type=float,
        nargs=3,
        metavar=("SPOTSIZE", "SPEED", "SCANTIME"),
        default=None,
        help="specify the laser parameters to use, defaults to reading the config",
    )
    convert.add_argument(
        "--elements",
        metavar="NAME",
        nargs="+",
        type=str,
        help="limit output to these elements",
    )

    filter = subparsers.add_parser(
        "filter",
        parents=[ioparser],
        help="filter data and output",
    )
    filter.add_argument(
        "--type",
        choices=["mean", "median"],
        dest="filter_type",
        default="mean",
        help="type of rolling filter",
    )
    filter.add_argument(
        "--size",
        type=int,
        dest="filter_size",
        metavar="SIZE",
        default=5,
        help="window size of filter in pixels",
    )
    filter.add_argument(
        "--threshold",
        type=float,
        dest="filter_threshold",
        metavar="STDDEVS",
        default=3.0,
        help="threshold above which to apply filter, in stddev",
    )
    filter.add_argument(
        "--elements",
        dest="filter_elements",
        metavar="NAME",
        type=str,
        nargs="+",
        help="limit filtering to these elements",
    )

    show = subparsers.add_parser(
        "show", help="output laser elements, size and other information"
    )
    show.add_argument(
        "input", type=check_exists, nargs="+", help="path to input file(s)"
    )
    show.add_argument(
        "--calibration",
        dest="show_calibration",
        action="store_true",
        help="show the laser calibration",
    )
    show.add_argument(
        "--info",
        dest="show_info",
        action="store_true",
        help="show stored laser information",
    )

    parser.add_argument(
        "-v", "--version", action="version", version=f"pewlib {__version__}"
    )

    args = parser.parse_args()

    try:
        args.lasers = [load(input) for input in args.input]
    except ValueError as e:
        parser.error(f"argument input: {e}")

    if args.output is None:
        args.output = [input.with_suffix(args.format) for input in args.input]
    elif not args.output.is_dir():
        if len(args.input) != 1:
            parser.error(
                "argument output: output must be an existing directory when more than one input is passed"
            )
        elif args.output.suffix.lower() != args.format:
            parser.error(
                f"argument output: output file extension does not match format '{args.format}'"
            )
        args.output = [args.output]
    else:
        args.output = [
            args.output.joinpath(input.with_suffix(args.format).name)
            for input in args.input
        ]

    valid_elements = sorted(set([e for laser in args.lasers for e in laser.elements]))
    if args.command == "convert" and args.elements is not None:
        for element in args.elements:
            if element not in valid_elements:
                parser.error(
                    f"argument elements: '{element}' not in data, valid names are {', '.join(valid_elements)}"
                )
    if args.command == "filter" and args.filter_elements is not None:
        for element in args.filter_elements:
            if element not in valid_elements:
                parser.error(
                    f"argument filter: '{element}' not in data, valid names are {', '.join(valid_elements)}"
                )

    return args


def save(laser: Laser, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        for name in laser.data.dtype.names:
            io.textimage.save(path.with_stem(path.stem + "_" + name), laser.data[name])
    elif path.suffix.lower() == ".npz":
        io.npz.save(path, laser)
    elif path.suffix.lower() == ".vtk":
        spacing = (
            laser.config.get_pixel_width(),
            laser.config.get_pixel_height(),
            laser.config.spotsize / 2.0,
        )
        io.vtk.save(path, laser.data, spacing=spacing)
    else:
        raise ValueError(
            f"pewlib: {path.name}: unable to save as format '{path.suffix}'"
        )


def main() -> int:
    args = create_parser_and_parse_args()

    for laser, input, output in zip(args.lasers, args.input, args.output):
        if args.command == "show":
            print(f"{input.name}")
            print(f"\tshape: {laser.shape}")
            print(f"\telements: {', '.join(laser.elements)}")
            print(f"\tspotsize: {laser.config.spotsize}")
            print(f"\tspeed: {laser.config.speed}")
            print(f"\tscantime: {laser.config.scantime}")
            if args.show_calibration:
                for element in laser.elements:
                    cal = laser.calibration[element]
                    if cal.intercept == 0.0 and cal.gradient == 1.0:
                        continue
                    print(f"\t{element}: y = {cal.gradient} * x + {cal.intercept}")
                    print(f"\t\tr2 = {cal.rsq}")
                    print(f"\t\tno. points = {cal.x.size}")
                    print(f"\t\tweighting = {cal.weighting}")
            if args.show_info:
                for k, v in laser.info.items():
                    print(f"\t{k}={v}")
            continue

        if args.command == "convert":
            if args.config is not None:
                laser.config = Config(
                    spotsize=args.config[0],
                    speed=args.config[1],
                    scantime=args.config[2],
                )
            if args.elements is not None:
                remove = [
                    element
                    for element in laser.elements
                    if element not in args.elements
                ]
                laser.remove(remove)

                if len(laser.elements) == 0:
                    print(f"skipping {input}: no matching elements")
                    continue

        if args.command == "filter":
            func = (
                filters.rolling_mean if args.filter_type == "mean" else filters.rolling_median
            )
            elements = args.filter_elements or laser.elements
            for element in elements:
                laser.data[element] = func(
                    laser.data[element], args.filter_size, args.filter_threshold
                )

        save(laser, output)

    return 0


if __name__ == "__main__":
    main()
