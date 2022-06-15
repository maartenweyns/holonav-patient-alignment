import open3d as o3d

from icp.p2p_icp import p2p_icp_registration
from pca.utils import pca_rough_alignment


def pca_icp_alignment(
        source: o3d.geometry.PointCloud,
        target_depth_sensor: o3d.geometry.PointCloud,
        target_pointer: o3d.geometry.PointCloud,
        icp_correspondence: int = 4,
        icp_max_iterations: int = 30):
    """
    Perform PCA rough registration followed by ICP precise registration to
    align the source point cloud to the target point clouds

    :param source: Source point cloud
    :param target_depth_sensor: Target depth sensor point cloud
    :param target_pointer: Target pointer point cloud
    :param icp_correspondence: Correspondence value for ICP algorithm
    :param icp_max_iterations: Maximum allowed iterations for ICP algorithm
    """
    # Perform PCA rough registration
    pca_rough_alignment(source, target_depth_sensor)
    # Perform ICP precise registration
    p2p_icp_registration(source, target_depth_sensor, icp_correspondence, icp_max_iterations)
    p2p_icp_registration(source, target_pointer, icp_correspondence, icp_max_iterations)
