from fpfh.fpfh_icp_alignment import fpfh_icp_alignment
from general.evaluation import calculate_error, prepare_reference_points
from general.preprocessing import misalign_point_cloud
from pca.pca_icp_alignment import pca_icp_alignment


def run_algorithms(source, target_depth_sensor, target_pointer, results):
    # Prepare evaluation
    reference_points = prepare_reference_points(target_pointer, source)

    # Misalign source data and perform FPFH alignment
    source_fpfh = misalign_point_cloud(source)
    fpfh_icp_alignment(source_fpfh, target_depth_sensor, target_pointer)

    # Get error
    error = calculate_error(target_pointer, source_fpfh, reference_points)
    results["fpfh"].append(round(error, 5))

    # Misalign source data and perform PCA alignment
    source_pca = misalign_point_cloud(source)
    pca_icp_alignment(source_pca, target_depth_sensor, target_pointer)

    # Get error
    error = calculate_error(target_pointer, source_pca, reference_points)
    results["pca"].append(round(error, 5))



