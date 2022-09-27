import argparse
from pathlib import Path
import time

from pewlib import io
from pewlib import Laser, Config
from pewlib import __version__

from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        add_help=False,
        description="CLI for reading and processing of LA-ICP-MS data.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="path to input file",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="path to output file, must have the extension '.csv', '.npz' or '.vtk'",
    )
    args, remainder = parser.parse_known_args()
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
        "--elements", nargs="+", type=str, help="limit output to these elements"
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"pewlib {__version__}"
    )

    args = argparse.Namespace(**vars(args), **vars(parser.parse_intermixed_args(remainder)))

    # convert_parser checks
    if not args.input.expanduser().exists():
        parser.error(f"argument input: file '{args.input.name}' does not exist")
    if args.output.suffix.lower() not in [".csv", ".npz", ".vtk"]:
        parser.error(f"argument output: must have extension '.csv', '.npz' or '.vtk'")
    if args.output.suffix.lower() == ".csv":
        if args.elements is None or len(args.elements) != 1:
            parser.error(
                f"argument elements: '.csv' output requires one element to be specified"
            )

    print(args)
    exit()
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
            raise ValueError(f"pewlib: {path.name}: unknown extention '{path.suffix}'")
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
            raise ValueError(f"pewlib: {path.name}: unknown extention '{path.suffix}'")

    if "spotsize" in params:
        config.spotsize = params["spotsize"]
    if "speed" in params:
        config.speed = params["speed"]
    if "scantime" in params:
        config.scantime = params["scantime"]

    return Laser(data=data, config=config, info=info)


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
        raise ValueError(f"pewlib: {path.name}: unable to save as format '{path.suffix}'")


def main() -> int:
    args = parse_args()

    laser = load(args.input)
    if args.config is not None:
        laser.config = Config(
            spotsize=args.config[0],
            speed=args.config[1],
            scantime=args.config[2],
        )

    if args.elements is not None:
        for element in args.elements:
            if element not in laser.elements:
                print(f"pewlib: error: element '{element}' not in data, valid choices are '{', '.join(laser.elements)}'")
                return 2
        remove = [element for element in laser.elements if element not in args.elements]
        laser.remove(remove)

    save(laser, args.output)

    return 0


if __name__ == "__main__":
    main()
