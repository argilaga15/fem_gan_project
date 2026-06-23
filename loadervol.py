from pathlib import Path
import numpy as np


def load_volume(path):

    path = Path(path)

    # --------------------------------------------------
    # NumPy volume
    # --------------------------------------------------

    if path.suffix.lower() == ".npy":

        volume = np.load(path)

    # --------------------------------------------------
    # MATLAB
    # --------------------------------------------------

    elif path.suffix.lower() == ".mat":

        from scipy.io import loadmat

        mat = loadmat(path)

        candidates = [
            v for v in mat.values()
            if isinstance(v, np.ndarray)
            and v.ndim == 3
        ]

        if len(candidates) == 0:
            raise ValueError(
                "No 3D array found in MAT file."
            )

        volume = candidates[0]

    # --------------------------------------------------
    # TIFF stack in a single file
    # --------------------------------------------------

    elif path.suffix.lower() in [".tif", ".tiff"]:

        import tifffile

        volume = tifffile.imread(path)

    # --------------------------------------------------
    # Folder of TIFF slices
    # --------------------------------------------------

    elif path.is_dir():

        import tifffile

        files = sorted(
            list(path.glob("*.tif"))
            + list(path.glob("*.tiff"))
        )

        if len(files) == 0:

            raise ValueError(
                f"No TIFF files found in {path}"
            )

        volume = np.stack(
            [tifffile.imread(f) for f in files],
            axis=2
        )

    else:

        raise ValueError(
            f"Unsupported format: {path}"
        )

    print(
        f"Loaded volume shape: {volume.shape}"
    )

    print(
        f"dtype: {volume.dtype}"
    )

    return volume