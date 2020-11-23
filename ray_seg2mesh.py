import ray
from seg2mesh import seg2mesh


def ray_seg2mesh(idx_list, segmentation, step_size=2, max_workers=2):

    if max_workers == -1:
        ray.init(address="auto")
    else:
        ray.init(num_cpus=max_workers)
    segmentation_id = ray.put(segmentation)

    @ray.remote
    def _seg2mesh(_idx, _segmentation_id, _step_size=step_size):
        return seg2mesh(_idx, _segmentation_id, _step_size)

    tasks = [_seg2mesh.remote(idx, segmentation_id) for idx in idx_list]
    results = ray.get(tasks)

    ray.shutdown()
    return results


if __name__ == "__main__":
    import h5py
    import numpy as np
    import time

    step_size = 1
    max_workers = -1
    num_labels = -2
    path = './data/sample_ovules.h5'

    with h5py.File(path, 'r') as f:
        segmentation = f['label'][...]

    labels_idx = np.unique(segmentation)[1:num_labels + 1]

    print(f"Total num labels: {len(labels_idx)}, "
          f"Max workers: {max_workers}, "
          f"Step-size: {step_size}")

    timer = time.time()
    runtime_results = ray_seg2mesh(labels_idx,
                                   segmentation,
                                   step_size=step_size,
                                   max_workers=max_workers)
    print(f"ray runtime: {time.time() - timer:.2f}s")
