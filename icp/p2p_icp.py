import numpy as np
import open3d as o3d


def get_p2p_icp_transformation(
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        correspondence: int,
        max_iterations: int):
    """
    Get transformation from ICP algorithm aligning the source
    point cloud to the target point cloud.

    :param source: The source point cloud
    :param target: The target point cloud
    :param correspondence: The ICP correspondence
    :param max_iterations: The max allowed ICP iterations
    :return: The transformation matrix
    """
    # Use open3d ICP registration pipeline
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        correspondence,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations))
    # Return the obtained transformation
    return result.transformation


def p2p_icp_registration(
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        correspondence: int = 4,
        max_iterations: int = 30):
    """
    Apply ICP registration aligning the source point cloud to the target point cloud.

    :param source: The source point cloud
    :param target: The target point cloud
    :param correspondence: The ICP correspondence
    :param max_iterations: The max allowed ICP iterations
    """
    transformation = get_p2p_icp_transformation(source, target, correspondence, max_iterations)
    source.transform(transformation)
