from fpfh.fpfh_icp_alignment import fpfh_icp_alignment
from general.evaluation import calculate_error, prepare_reference_points
from general.preprocessing import misalign_point_cloud
from pca.pca_icp_alignment import pca_icp_alignment


def run_algorithms(source, target_depth_sensor, target_pointer, results):
    # Prepare evaluation
    reference_points = prepare_reference_points(target_pointer, source)

    for i in range(5):
        # Misalign source data and perform FPFH alignment
        source_fpfh = misalign_point_cloud(source)
        fpfh_icp_alignment(source_fpfh, target_depth_sensor, target_pointer)

        # Get error
        error = calculate_error(target_pointer, source_fpfh, reference_points)

        # Break loop when alignment succeeded
        if error < 2:
            results["fpfh"].append(round(error, 5))
            break
        # Save result when loop ends
        elif i == 4:
            print("Too many FPFH attempts, using last obtained result")
            results["fpfh"].append(round(error, 5))
        # Retry otherwise
        else:
            print("Retrying FPFH alignment. {} attempted".format(i + 1))

    for i in range(5):
        # Misalign source data and perform PCA alignment
        source_pca = misalign_point_cloud(source)
        pca_icp_alignment(source_pca, target_depth_sensor, target_pointer)

        # Get error
        error = calculate_error(target_pointer, source_pca, reference_points)

        # Break loop when alignment succeeded
        if error < 4:
            results["pca"].append(round(error, 5))
            break
        # Save result when loop ends
        elif i == 2:
            print("Too many PCA attempts, using last obtained result")
            results["pca"].append(round(error, 5))
        # Retry otherwise
        else:
            print("Retrying PCA alignment: {} attempted".format(i + 1))
