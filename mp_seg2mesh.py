from seg2mesh import seg2mesh
import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor


def sc_seg2mesh(idx_list, segmentation, step_size=2):
    results = [seg2mesh(idx, segmentation, step_size) for idx in tqdm.tqdm(idx_list)]
    return results


def mp_seg2mesh(idx_list, segmentation, step_size=2, max_workers=2):
    _seg2mesh = partial(seg2mesh, segmentation=segmentation, step_size=step_size)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm.tqdm(executor.map(_seg2mesh, idx_list), total=len(idx_list)))

    return results
