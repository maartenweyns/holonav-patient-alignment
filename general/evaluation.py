import numpy as np
import open3d as o3d


def prepare_reference_points(
        target_pointer: o3d.geometry.PointCloud,
        source: o3d.geometry.PointCloud):
    """
    Prepare the source point clouds for evaluation. This is done by adding the points from
    the pointer point cloud to the source point cloud and saving their indices.

    :param target_pointer: The pointer point cloud
    :param source: The source point cloud to add the points to
    :return: The indices of the points added
    """
    # Add target points to source point cloud
    reference_points = []
    for point in target_pointer.points:
        source.points.append(point)
        reference_points.append(len(source.points) - 1)
    # Return indices of reference points
    return reference_points


def calculate_error(
        pc_small: o3d.geometry.PointCloud,
        pc_large: o3d.geometry.PointCloud,
        reference_points: list=None):
    """
    Calculate the MSE of the alignment.
    There are two ways this can be calculated:
    - Using reference points, to calculate the actual error from perfect alignment
    - Without reference points, to calculate the distance between the two point clouds

    :param pc_small: The smallest point cloud
    :param pc_large: The largest point cloud
    :param reference_points: The array of reference points, default None to enable calculation
    without reference points
    :return: The calculated MSE
    """
    distances = []
    if reference_points is None:
        distances = pc_small.compute_point_cloud_distance(pc_large)
    else:
        for idx, point in enumerate(pc_small.points):
            reference_point = pc_large.points[reference_points[idx]]
            target_point = point

            distance = np.sqrt(
                (reference_point[0] - target_point[0])**2 +
                (reference_point[1] - target_point[1])**2 +
                (reference_point[2] - target_point[2])**2
            )

            distances.append(distance)

    # Sum all distances squared
    distance_sum = 0
    for distance in distances:
        distance_sum += distance ** 2
    # Divide sum by amount of distances
    mse = distance_sum / len(distances)
    # Return MSE
    print("Mean Square Error is {}".format(mse))
    return mse
