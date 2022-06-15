import open3d as o3d


def get_fpfh_transformation(
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        downsample: bool=True):
    """
    Get transformation matrix to roughly align source and
    target point clouds based on FPFH.
    Downsampling can be enabled or disabled, but is enabled by default
    to decrease alignment time

    :param source: The source point cloud
    :param target: The target point cloud
    :param downsample: Enable or disable point cloud downsampling
    :return: The transformation matrix
    """
    if downsample:
        source = source.voxel_down_sample(2)
        target = target.voxel_down_sample(2)

    source.estimate_normals(fast_normal_computation=False)
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source,
        o3d.geometry.KDTreeSearchParamHybrid(radius=9, max_nn=100))

    target.estimate_normals(fast_normal_computation=False)
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target,
        o3d.geometry.KDTreeSearchParamHybrid(radius=9, max_nn=100))

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh,
        True,
        3,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                3)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    return result.transformation


def fpfh_rough_registration(
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        downsample: bool=True):
    """
    Roughly align the source point cloud to the target point cloud
    based on FPFH.

    :param source: The source point cloud
    :param target: The target point cloud
    """
    transformation = get_fpfh_transformation(source, target, downsample)
    source.transform(transformation)
