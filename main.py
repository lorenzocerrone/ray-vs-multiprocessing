import time

import h5py
import numpy as np

from mp_seg2mesh import sc_seg2mesh, mp_seg2mesh
from ray_seg2mesh import ray_seg2mesh


def main(_max_workers, _step_size):
    timer = time.time()
    sc_seg2mesh(labels_idx, segmentation, step_size=_step_size)
    sc_runtime = time.time() - timer

    timer = time.time()
    mp_seg2mesh(labels_idx, segmentation, step_size=_step_size, max_workers=_max_workers)
    mp_runtime = time.time() - timer

    timer = time.time()
    ray_seg2mesh(labels_idx, segmentation, step_size=_step_size, max_workers=_max_workers)
    ray_runtime = time.time() - timer

    return {"Single Core": sc_runtime,
            "Multi Core": mp_runtime,
            "Ray": ray_runtime}


if __name__ == "__main__":
    step_size = 1
    max_workers = 4
    num_labels = 24
    path = './data/sample_ovules.h5'

    with h5py.File(path, 'r') as f:
        segmentation = f['label'][...]

    labels_idx = np.unique(segmentation)[1:num_labels + 1]

    print(f"Total num labels: {len(labels_idx)}, "
          f"Max workers: {max_workers}, "
          f"Step-size: {step_size}")

    runtime_results = main(max_workers, step_size)

    print("Summary:")
    for key, runtime in runtime_results.items():
        print(f"Runtime {key}: {runtime: .2f}")
