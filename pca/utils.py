import numpy as np
import open3d as o3d


def find_principal_component(point_cloud: o3d.geometry.PointCloud):
    """
    Find the first principal component of a given point cloud

    :param point_cloud: The point cloud to find the principal component of
    :return: The first principal component
    """
    # Find covariance matrix of source data and extract eigenvalues and eigenvectors
    (source_mean, source_covariance) = point_cloud.compute_mean_and_covariance()
    (eigen_values, eigen_vectors) = np.linalg.eig(source_covariance)

    # Sort eigenvalue-eigenvector pairs
    eig_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(3)]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Return first principal component
    return eig_pairs[0][1]


def find_mass_center(point_cloud: o3d.geometry.PointCloud):
    """
    Find the mass center of a given point cloud

    :param point_cloud: The point cloud to find the mass center of
    :return: The mass center as a vector
    """
    mean = point_cloud.compute_mean_and_covariance()[0]
    return mean


def rotation_matrix_from_vectors(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2

    :param vec1: Source vector
    :param vec2: Destination vector
    :return: The rotation matrix that aligns vec1 to vec2
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def get_pca_translation_rotation(source:o3d.geometry.PointCloud, target: o3d.geometry.PointCloud):
    """
    Get rotation and transformation matrices to roughly align source and
    target point clouds based on principal component analysis.

    :param source: The source point cloud
    :param target: The target point cloud
    :return: The translation and rotation matrices
    """
    # Find principal components
    source_principal_component = find_principal_component(source)
    target_principal_component = find_principal_component(target)

    # Find mass centers
    source_mass_center = find_mass_center(source)
    target_mass_center = find_mass_center(target)

    # Calculate translation and rotation
    translation = target_mass_center - source_mass_center
    rotation = rotation_matrix_from_vectors(source_principal_component, target_principal_component)

    return translation, rotation


def pca_rough_alignment(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud):
    """
    Roughly align the source point cloud to the target point cloud
    based on principal component analysis.

    :param source: The source point cloud
    :param target: The target point cloud
    """
    t, r = get_pca_translation_rotation(source, target)

    source.translate(t)
    source.rotate(r)
