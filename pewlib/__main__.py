import argparse
from pathlib import Path
import time

from pewlib import io
from pewlib import Laser, Config
from pewlib import __version__

from typing import List


def load(file: str) -> Laser:
    path = Path(file)
    if not path.exists():
        raise argparse.ArgumentTypeError("path does not exist")

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
                raise argparse.ArgumentTypeError(
                    f"unable to import batch '{path.name}'"
                )
        elif io.perkinelmer.is_valid_directory(path):
            data, params = io.perkinelmer.load(path, full=True)
            info["Instrument Vendor"] = "PerkinElemer"
        elif io.csv.is_valid_directory(path):
            data, params = io.csv.load(path, full=True)
        else:  # pragma: no cover
            raise argparse.ArgumentTypeError(f"unknown extention '{path.suffix}'")
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
            raise argparse.ArgumentTypeError(f"unknown extention '{path.suffix}'")

    if "spotsize" in params:
        config.spotsize = params["spotsize"]
    if "speed" in params:
        config.speed = params["speed"]
    if "scantime" in params:
        config.scantime = params["scantime"]

    return Laser(data=data, config=config, info=info)


def create_parser_and_parse_args() -> argparse.Namespace:
    def path_with_suffix(file: str, suffixes: List[str]) -> Path:
        path = Path(file)
        if path.suffix.lower() not in suffixes:
            raise argparse.ArgumentTypeError(
                f"must have a suffix in {', '.join(suffixes)}"
            )
        return path

    parser = argparse.ArgumentParser(
        description="CLI for reading and processing of LA-ICP-MS data.",
    )
    parser.add_argument("input", type=load, help="path to input file")
    parser.add_argument(
        "output",
        nargs="?",
        type=lambda s: path_with_suffix(s, [".csv", ".npz", ".vtk"]),
        help="path to output file, must have the extension '.csv', '.npz' or '.vtk'",
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

    args = parser.parse_args()

    if args.list:
        print(", ".join(args.input.elements))
        exit(0)
    elif args.output is None:
        parser.error("the following arguments are required: input output")

    if args.output.suffix.lower() == ".csv":
        if args.elements is None or len(args.elements) != 1:
            parser.error(
                f"argument elements: '.csv' output requires exactly one element to be specified"
            )
    for element in args.elements:
        if element not in args.input.elements:
            parser.error(
                f"argument elements: '{element}' not in data, valid names are {', '.join(args.input.elements)}"
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

    save(args.input, args.output)

    return 0


if __name__ == "__main__":
    main()
