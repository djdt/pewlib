import argparse
from pathlib import Path
import sys
import time

from pewlib import io
from pewlib import Laser, Config
from pewlib import __version__

from typing import List


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI for reading and processing of LA-ICP-MS data.",
    )
    parser.add_argument(
        "-v", "--version", action="store_true", help="print version and exit."
    )
    subparsers = parser.add_subparsers(dest="command")

    convert_parser = subparsers.add_parser("convert", help="batch convert files")
    convert_parser.add_argument(
        "files", nargs="+", type=Path, help="path to input file(s)"
    )
    convert_parser.add_argument(
        "-c",
        "--config",
        type=float,
        nargs=3,
        metavar=("SPOTSIZE", "SPEED", "SCANTIME"),
        default=None,
        help="specify the laser parameters to use, defaults to reading the config",
    )
    convert_parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["csv", "npz", "vtk"],
        default="npz",
        help="format to convert to",
    )
    convert_parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=Path,
        help="output directory for converted files, defaults to the input directories",
    )

    args = parser.parse_args(argv)

    # convert_parser checks
    if args.files is not None:
        for file in args.files:
            if not file.expanduser().exists():
                parser.error(f"argument files: file '{file.name}' does not exist")
    if args.output is not None:
        if args.output.exists() and not args.output.is_dir():
            parser.error(f"argument output: path '{args.output}' is not a directory")

    return args


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
                raise ValueError(f"Unable to import batch '{path.name}'!")
        elif io.perkinelmer.is_valid_directory(path):
            data, params = io.perkinelmer.load(path, full=True)
            info["Instrument Vendor"] = "PerkinElemer"
        elif io.csv.is_valid_directory(path):
            data, params = io.csv.load(path, full=True)
        else:  # pragma: no cover
            raise ValueError(f"{path.name}: unknown extention '{path.suffix}'")
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
            raise ValueError(f"{path.name}: unknown extention '{path.suffix}'")

    if "spotsize" in params:
        config.spotsize = params["spotsize"]
    if "speed" in params:
        config.speed = params["speed"]
    if "scantime" in params:
        config.scantime = params["scantime"]

    return Laser(data=data, config=config, info=info)


def save(laser: Laser, path: Path) -> None:
    if path.suffix.lower() == ".npz":
        io.npz.save(path, laser)
    elif path.suffix.lower() == ".vtk":
        spacing = (
            laser.config.get_pixel_width(),
            laser.config.get_pixel_height(),
            laser.config.spotsize / 2.0,
        )
        io.vtk.save(path, laser.data, spacing=spacing)
    else:
        raise ValueError(f"{path.name}: unable to save as format '{path.suffix}'")


# def convert(args: argparse.Namespace) -> int:

#     for file in args.


def main() -> int:
    args = parse_args(sys.argv[1:])

    if args.version:
        print(f"pewlib version: {__version__}")
        return 0

    if args.command == "convert":
        for file in args.files:
            laser = load(file)
            if args.config is not None:
                laser.config = Config(
                    spotsize=args.config[0],
                    speed=args.config[1],
                    scantime=args.config[2],
                )
            if args.output is None:
                outfile = file.with_suffix(args.format)
            else:
                outfile = args.output.joinpath(file.with_suffix(args.format).name)

            save(laser, outfile)

    return 0


if __name__ == "__main__":
    main()
