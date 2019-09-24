import argparse
import os.path
import sys

from pew import Laser, Config
from pew import io


def import_any(path, config):
    base, ext = os.path.splitext(path)
    name = os.path.basename(base)
    ext = ext.lower()
    if ext == ".npz":
        return io.npz.load(path)
    else:
        if ext == ".csv":
            try:
                data = io.thermo.load(path)
            except io.error.PewException:
                data = io.csv.load(path)
        elif ext in [".txt", ".text"]:
            data = io.csv.load(path)
        elif ext == ".b":
            data = io.agilent.load(path)
        else:
            raise io.error.PewException(f"Unknown extention '{ext}'.")
        return Laser.from_structured(data=data, config=config, name=name, filepath=path)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="pew", description="LA-ICP-MS data converter."
    )
    parser.add_argument("infile", help="Input file.")
    parser.add_argument(
        "outfile", help="Output file. Extension must be either .npz or .csv."
    )
    parser.add_argument(
        "--config",
        nargs=3,
        metavar=("spotsize", "speed", "scantime"),
        default=[35.0, 140.0, 0.25],
        type=float,
    )
    parser.add_argument(
        "--isotope", "-i", type=str, help="Limit export to the selected isotope."
    )
    parser.add_argument(
        "--calibrate",
        nargs=2,
        metavar=("gradient", "intercept"),
        default=[1.0, 0.0],
        type=float,
    )

    args = parser.parse_args(sys.argv[1:])
    if args.outfile.lower().endswith(".csv"):
        if not hasattr(parser, "isotope"):
            parser.error("Output isotope required for .csv.")
    if hasattr(parser, "calibrate"):
        if not hasattr(parser, "isotope"):
            parser.error("Output isotope required for calibration.")
    return args


def main():
    args = parse_args()

    config = Config(*args.config)
    laser = import_any(args.infile, config)

    _, outext = os.path.splitext(args.outfile)
    if outext.lower() not in [".npz", ".csv"]:
        print("Invalid output extention.")
        sys.exit(1)

    if outext.lower() == ".npz":
        io.npz.save(args.outfile, [laser])
    else:
        laser.data[args.isotope].calibration = Calibration(*args.calibrate)
        io.csv.save(args.outfile, laser.get(args.isotope, calibrate=True))


if __name__ == "__main__":
    main()
