"""
Exports to VTK formats for use in programs such as Paraview.
"""

import sys
from pathlib import Path

import numpy as np


def escape_xml(string: str) -> str:
    char_map = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;"}
    for key in char_map:
        string = string.replace(key, char_map[key])
    return string


def save(
    path: str | Path, data: np.ndarray, spacing: tuple[float, float, float]
) -> None:
    """Save data as a VTK ImageData XML.

    Saves an array to a '.vti' file. Data origin is set to (0, 0) and equally
    spaced using x, y, z of `spacing`. If `data` is rasied to 3-dimensonal if lower.

    Args:
        path: path to file
        data: array
        spacing: spacing of '.vti'
    """
    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    if data.ndim < 3:  # pragma: no cover
        data = np.reshape(data, (*data.shape, 1))

    # Paraview is x,y and numpy is y,x with y order reversed
    data = np.flip(data, axis=0).swapaxes(0, 1)

    nx, ny, nz = data.shape
    origin = 0.0, 0.0

    endian = "LittleEndian" if sys.byteorder == "little" else "BigEndian"

    extent_str = f"0 {nx} 0 {ny} 0 {nz}"
    origin_str = f"{origin[1]} {origin[1]} 0.0"
    spacing_str = f"{spacing[0]} {spacing[1]} {spacing[2]}"

    offset = 0
    with path.open("wb") as fp:
        fp.write(
            (
                '<?xml version="1.0"?>\n'
                '<VTKFile type="ImageData" version="1.0" '
                f'byte_order="{endian}" header_type="UInt64">\n'
                f'<ImageData WholeExtent="{extent_str}" '
                f'Origin="{origin_str}" Spacing="{spacing_str}">\n'
                f'<Piece Extent="{extent_str}">\n'
            ).encode()
        )

        fp.write(f'<CellData Scalars="{escape_xml(data.dtype.names[0])}">\n'.encode())
        for name in data.dtype.names:
            fp.write(
                (
                    f'<DataArray Name="{escape_xml(name)}" type="Float64" '
                    f'format="appended" offset="{offset}"/>\n'
                ).encode()
            )
            offset += data[name].size * data[name].itemsize + 8  # blocksize
        fp.write("</CellData>\n".encode())

        fp.write(
            (
                "</Piece>\n" "</ImageData>\n" '<AppendedData encoding="raw">\n' "_"
            ).encode()
        )

        for name in data.dtype.names:
            fp.write(np.uint64(data[name].size * data[name].itemsize))
            fp.write(data[name].ravel("F"))

        fp.write(("</AppendedData>\n" "</VTKFile>").encode())
