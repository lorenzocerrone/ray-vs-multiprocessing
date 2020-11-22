import numpy as np
from skimage.measure import marching_cubes


def seg2mesh(idx, segmentation, step_size=1):
    mask = segmentation == idx

    _z, _x, _y = np.nonzero(mask)
    # returns True only if mask is not flat in any dimension
    if abs(_z.max() - _z.min()) > 1 and abs(_x.max() - _x.min()) > 1 and abs(_y.max() - _y.min()) > 1:
        coords, faces, _, _ = marching_cubes(mask, step_size=step_size)
        return coords, faces
    else:
        return None, None
