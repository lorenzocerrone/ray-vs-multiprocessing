import ray

from seg2mesh import seg2mesh


def ray_seg2mesh(idx_list, segmentation, step_size=2, max_workers=2):

    ray.init(num_cpus=max_workers)
    segmentation_id = ray.put(segmentation)

    @ray.remote
    def _seg2mesh(_idx, _segmentation_id, _step_size=step_size):
        return seg2mesh(_idx, _segmentation_id, _step_size)

    tasks = [_seg2mesh.remote(idx, segmentation_id) for idx in idx_list]
    results = ray.get(tasks)

    ray.shutdown()
    return results
