import h5py
import numpy as np
import time
from mp_seg2mesh import sc_seg2mesh, mp_seg2mesh
from ray_seg2mesh import ray_seg2mesh
import matplotlib.pyplot as plt


def make_runtime_plot(num_workers=(1, 2, 4), _step_size=1):

    timer = time.time()
    sc_seg2mesh(labels_idx, segmentation, step_size=_step_size)
    sc_runtime = time.time() - timer
    sc_runtimes = len(num_workers) * [1]

    mp_runtimes = []
    for _max_workers in num_workers:
        timer = time.time()
        mp_seg2mesh(labels_idx, segmentation, step_size=_step_size, max_workers=_max_workers)
        mp_runtime = time.time() - timer
        mp_runtimes.append(sc_runtime/mp_runtime)

    ray_runtimes = []
    for _max_workers in num_workers:
        timer = time.time()
        ray_seg2mesh(labels_idx, segmentation, step_size=_step_size, max_workers=_max_workers)
        ray_runtime = time.time() - timer
        ray_runtimes.append(sc_runtime/ray_runtime)

    plt.figure()
    plt.title(f"Ray vs Multiprocessing - runtime speed up - num of tasks {len(labels_idx)}")
    plt.plot(num_workers, sc_runtimes, label="Single Core")
    plt.plot(num_workers, mp_runtimes, label="MultiProcessing")
    plt.plot(num_workers, ray_runtimes, label="Ray")

    plt.ylabel("Speed up")
    plt.xlabel("# Process")
    plt.xticks(num_workers)
    plt.legend()
    plt.savefig(f"ray_vs_mp_{len(labels_idx)}.png")


if __name__ == "__main__":
    step_size = 1
    max_workers = 4
    num_labels = -2
    path = './data/sample_ovules.h5'

    with h5py.File(path, 'r') as f:
        segmentation = f['label'][...]

    labels_idx = np.unique(segmentation)[1:num_labels+1]

    print(f"Total num labels: {len(labels_idx)}, "
          f"Max workers: {max_workers}, "
          f"Step-size: {step_size}")

    make_runtime_plot([1, 2, 4], step_size)
