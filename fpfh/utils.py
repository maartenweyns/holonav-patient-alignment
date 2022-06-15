import open3d as o3d


def get_fpfh_transformation(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud):
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


def fpfh_rough_registration(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud):
    transformation = get_fpfh_transformation(source, target)
    source.transform(transformation)
