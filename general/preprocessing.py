import copy
import random

import numpy as np
import open3d as o3d


def add_noise(point_cloud: o3d.geometry.PointCloud, noise_amt: float = 4.3):
    """
    Add a specified amount of noise to a point cloud. The noise amount represenats
    the max distance a point will be transformed in 3D space.

    :param point_cloud: The point cloud to add the noise to
    :param noise_amt: The amount of noise to be added in mm
    """
    if noise_amt <= 0:
        # Don't add noise when noise amount is 0
        return point_cloud
    noisy_pc = copy.deepcopy(point_cloud)
    max_translation = np.sqrt(noise_amt ** 2 / 3)
    for i, p in enumerate(noisy_pc.points):
        noisy_pc.points[i] = [n + random.uniform(-max_translation, max_translation) for n in p]
    return noisy_pc


def downsample_point_cloud(point_cloud: o3d.geometry.PointCloud, voxel_size: int):
    downsampled = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    return downsampled


def misalign_point_cloud(point_cloud: o3d.geometry.PointCloud):
    """
    Apply a random but fixed transformation to a point cloud to misalign it.

    :param point_cloud: The point cloud to transform
    """
    transformed = copy.deepcopy(point_cloud)
    rotation = point_cloud.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
    translation = (0, 0, -200)
    transformed.rotate(rotation)
    transformed.translate(translation)
    return transformed
