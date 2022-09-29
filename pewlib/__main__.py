import argparse
from pathlib import Path
import time

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

    parser = argparse.ArgumentParser(
        description="CLI for reading and processing of LA-ICP-MS data.",
    )

    parser.add_argument(
        "input", type=check_exists, nargs="+", help="path to input file(s)"
    )
    parser.add_argument(
        "--format",
        choices=valid_formats,
        default=".npz",
        help="output file format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="path to output file or directory, defaults to the input directory. "
        "if file the extension must match the format",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=float,
        nargs=3,
        metavar=("SPOTSIZE", "SPEED", "SCANTIME"),
        default=None,
        help="specify the laser parameters to use, defaults to reading the config",
    )
    parser.add_argument(
        "--elements",
        metavar="NAME",
        nargs="+",
        type=str,
        help="limit output to these elements",
    )

    parser.add_argument(
        "-v", "--version", action="version", version=f"pewlib {__version__}"
    )
    parser.add_argument(
        "-l", "--list", action="store_true", help="list element names in input and exit"
    )

    proc = parser.add_argument_group("processing commands")
    proc.add_argument(
        "--filter",
        nargs="*",
        metavar="{mean,median}, SIZE, THRESHOLD, [NAME",
        action=FilterAction,
        help="filter named elements with rolling window of SIZE and THRESHOLD,"
        " defaults to filtering all data",
    )

    args = parser.parse_args()

    try:
        args.lasers = [load(input) for input in args.input]
    except ValueError as e:
        parser.error(f"argument input: {e}")

    if args.list:
        for input, laser in zip(args.input, args.lasers):
            print(input.name, ":", ", ".join(laser.elements))
        exit(0)

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

    valid_elements = sorted(set([e for laser in args.lasers for e in laser.elements ]))
    if args.elements is not None:
        for element in args.elements:
            if element not in valid_elements:
                parser.error(
                    f"argument elements: '{element}' not in data, valid names are {', '.join(valid_elements)}"
                )
    if args.filter is not None:
        for element in args.filter[3]:
            if element not in valid_elements:
                parser.error(
                    f"argument filter: '{element}' not in data, valid names are {', '.join(valid_elements)}"
                )

    return args


def save(laser: Laser, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        name = laser.data.dtype.names[0]
        io.textimage.save(path, laser.data[name])
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

    if args.config is not None:
        args.input.config = Config(
            spotsize=args.config[0],
            speed=args.config[1],
            scantime=args.config[2],
        )

    if args.elements is not None:
        remove = [
            element for element in args.input.elements if element not in args.elements
        ]
        args.input.remove(remove)

    if args.filter is not None:
        func = (
            filters.rolling_mean if args.filter[0] == "mean" else filters.rolling_median
        )
        elements = args.filter[3] or args.input.elements
        for element in elements:
            args.input.data[element] = func(
                args.input.data[element], args.filter[1], args.filter[2]
            )

    save(args.input, args.output)

    return 0


if __name__ == "__main__":
    main()
