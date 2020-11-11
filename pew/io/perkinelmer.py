import numpy as np

from pathlib import Path

from typing import Any, Dict, List, Union


# parameters_path = Path("parameters.conf")
# positions_path = Path("positions.txt")
# parameters_key_conv = {"ablation.speed": "speed", "sq"}


def collect_datafiles(path: Path) -> List[Path]:
    datafiles = []

    for child in path.iterdir():
        if child.suffix == ".xl":
            datafiles.append(child)

    # Sort by any numerical order
    datafiles.sort(key=lambda f: int("".join(filter(str.isdigit, f.name))))
    return datafiles


# def xl_valid_lines(csv: str) -> Generator[bytes, None, None]:
#     delimiter_count = 0
#     past_header = False
#     with open(csv, "rb") as fp:
#         for line in fp:
#             if past_header and line.count(b",") == delimiter_count:
#                 yield line
#             if line.startswith(b"Time"):
#                 past_header = True
#                 delimiter_count = line.count(b",")
#                 yield line


def load(path: Union[str, Path]) -> np.ndarray:
    key_exchange = {
        "speed": "ablation.speed",
        "scantime": "acquisition.time",
        "spotsize": "space.interval",
    }
    if not isinstance(path, Path):
        path = Path(path)
    datafiles = collect_datafiles(path)

    data = np.vstack(
        [
            np.genfromtxt(df, skip_header=1, delimiter=",", names=True)
            for df in datafiles
        ]
    )
    params: Dict[str, Any] = {"origin": (0.0, 0.0)}

    parameters = path.joinpath("parameters.conf")
    if parameters.exists():
        with parameters.open() as fp:
            params.update(
                {k: v for k, v in (line.split("=") for line in fp.readlines())}
            )

        for new, old in key_exchange.items():
            if old in params:
                params[new] = params.pop(old)

    positions = path.joinpath("positions.txt")
    if positions.exists():
        np.genfromtxt(positions, delimiter=",", usecols=(0,1,2,3), dtype=float)


load("/home/tom/Downloads/Coin")
