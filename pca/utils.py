import numpy as np
import open3d as o3d


def get_pca_translation_rotation(
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        pca_index: int = 0):
    """
    Get rotation and transformation matrices to roughly align source and
    target point clouds based on principal component analysis.

    :param source: The source point cloud
    :param target: The target point cloud
    :param pca_index: The index of the principal component
    :return: The translation and rotation matrices
    """
    (source_mean, source_covariance) = source.compute_mean_and_covariance()
    (target_mean, target_covariance) = target.compute_mean_and_covariance()

    w0 = np.array([
        [source_mean[0]],
        [source_mean[1]],
        [source_mean[2]]])
    m0 = np.array([
        [target_mean[0]],
        [target_mean[1]],
        [target_mean[2]]])

    c = np.zeros((3,3))
    for idx, target_point in enumerate(np.asarray(target.points)):
        # Find corresponding index
        source_idx = int((idx / len(target.points)) * len(source.points))
        # Find source point
        source_point = np.asarray(source.points)[source_idx]
        # Define points as vectors
        mi = np.array([
            [target_point[0]],
            [target_point[1]],
            [target_point[2]]])
        wi = np.array([
            [source_point[0]],
            [source_point[1]],
            [source_point[2]]])
        # Add to covariance matrix
        c += ((mi - m0) @ (wi - w0).transpose())

    # Transpose covariance matrix
    c = c.transpose()
    # Save covariance matrix trace
    tr = np.trace(c)

    # Build symmetric matrix E from covariance matrix C
    e = np.array([
        [tr, c[1][2] - c[2][1], c[2][0] - c[0][2], c[0][1] - c[1][0]],
        [c[1][2] - c[2][1], (2 * c[0][0]) - tr, c[0][1] + c[1][0], c[0][2] + c[2][0]],
        [c[2][0] - c[0][2], c[0][1] + c[1][0], (2 * c[1][1]) - tr, c[1][2] + c[2][1]],
        [c[0][1] - c[1][0], c[0][2] + c[2][0], c[1][2] + c[2][1], (2 * c[2][2]) - tr]])

    # Calculate eigenvalues and get eigenvector with corresponding to highest eigenvalue
    (eigen_values, eigen_vectors) = np.linalg.eig(e)
    eig_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(3)]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    v = eig_pairs[0][1]

    # Get rotation and translation
    rotation = source.get_rotation_matrix_from_quaternion(v)
    translation = m0 - w0

    # Return result
    return translation, rotation


def pca_rough_alignment(
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        pca_index: int = 0):
    """
    Roughly align the source point cloud to the target point cloud
    based on principal component analysis.

    :param source: The source point cloud
    :param target: The target point cloud
    :param pca_index: The index of the principal component to be used
    """
    t, r = get_pca_translation_rotation(source, target, pca_index)

    source.translate(t)
    source.rotate(r)
