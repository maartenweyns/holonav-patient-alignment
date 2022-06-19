from fpfh.utils import fpfh_rough_registration
from general.evaluation import calculate_error, prepare_reference_points
from general.preprocessing import misalign_point_cloud
from icp.p2p_icp import p2p_icp_registration
from pca.utils import pca_rough_alignment


def run_algorithms(source, target_depth_sensor, target_pointer, results):
    # Prepare evaluation
    reference_points = prepare_reference_points(target_pointer, source)

    for i in range(5):
        # Misalign source data and perform FPFH alignment
        source_fpfh = misalign_point_cloud(source)

        fpfh_rough_registration(source_fpfh, target_depth_sensor)
        fpfh_error = calculate_error(target_pointer, source_fpfh, reference_points)

        p2p_icp_registration(source_fpfh, target_depth_sensor)
        p2p_icp_registration(source_fpfh, target_pointer)
        fpfh_icp_error = calculate_error(target_pointer, source_fpfh, reference_points)

        # Break loop when alignment succeeded
        if fpfh_icp_error < 2:
            results["fpfh"].append(round(fpfh_error, 5))
            results["fpfh+icp"].append(round(fpfh_icp_error, 5))
            break
        # Save result when loop ends
        elif i == 4:
            print("Too many FPFH attempts, using last obtained result")
            results["fpfh"].append(round(fpfh_error, 5))
            results["fpfh+icp"].append(round(fpfh_icp_error, 5))
        # Retry otherwise
        else:
            print("Retrying FPFH alignment. {} attempted".format(i + 1))

    # Misalign source data and perform PCA alignment
    source_pca = misalign_point_cloud(source)

    pca_rough_alignment(source_pca, target_depth_sensor)
    pca_error = calculate_error(target_pointer, source_pca, reference_points)

    p2p_icp_registration(source_pca, target_depth_sensor)
    p2p_icp_registration(source_pca, target_pointer)
    pca_icp_error = calculate_error(target_pointer, source_pca, reference_points)

    results["pca"].append(round(pca_error, 5))
    results["pca+icp"].append(round(pca_icp_error, 5))
