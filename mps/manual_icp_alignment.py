import copy
import random

import numpy as np
import open3d as o3d

from general.preprocessing import get_misalign_translation_rotation, misalign_point_cloud
from icp.p2p_icp import p2p_icp_registration
from pca.utils import get_pca_translation_rotation


def manual_icp_alignment(
        source: o3d.geometry.PointCloud,
        target_depth_sensor: o3d.geometry.PointCloud,
        target_pointer: o3d.geometry.PointCloud,
        selected_source: o3d.geometry.PointCloud,
        selected_target: o3d.geometry.PointCloud,
        icp_correspondence: int = 4,
        icp_max_iterations: int = 30):
    """
    Align source point cloud to target point cloud with manual point selection
    for rough registration.

    :param source: The source point cloud
    :param target_depth_sensor: The target depth sensor point cloud
    :param target_pointer: The target pointer point cloud
    :param selected_source: The points selected on the source
    :param selected_target: The points indicated on the target
    :param icp_correspondence: Correspondence used for the ICP algorithm
    :param icp_max_iterations: Max allowed iterations for the ICP algorithm
    """
    # Get translation and rotation from PCA algorithm
    t, r = get_pca_translation_rotation(selected_source, selected_target)

    # Transform source to target
    source.rotate(r, center=selected_source.get_center())
    source.translate(t)

    # Perform ICP precise registration
    p2p_icp_registration(source, target_depth_sensor, icp_correspondence, icp_max_iterations)
    p2p_icp_registration(source, target_pointer, icp_correspondence, icp_max_iterations)
