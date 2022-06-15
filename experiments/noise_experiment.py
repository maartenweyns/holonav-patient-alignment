import open3d as o3d
import pandas as pd

from fpfh.fpfh_icp_alignment import fpfh_icp_alignment
from general.evaluation import calculate_error, prepare_reference_points
from general.preprocessing import add_noise, misalign_point_cloud
from pca.pca_icp_alignment import pca_icp_alignment


def noise_test() -> dict:
    """
    Test the different algorithms against different levels of noise. Their
    default settings are used.

    :return: A dictionary containing the results
    """
    results = {"noise": [], "fpfh": [], "pca": []}

    for noise in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5]:
        results["noise"].append(noise)
        print("Testing noise level: {}".format(noise))

        # Read data
        target_pointer = o3d.io.read_point_cloud("../data/skull1/skull1_pointer.txt", format="xyz")
        target_depth_sensor = o3d.io.read_point_cloud("../data/skull1/occlusion/skull1_0deg.ply")
        source = o3d.io.read_point_cloud("../data/skull1/skull1_preop_model.ply")

        # Add noise to depth sensor data
        target_depth_sensor = add_noise(target_depth_sensor, noise_amt=noise)

        # Prepare evaluation
        reference_points = prepare_reference_points(target_pointer, source)

        # Misalign source data and perform FPFH alignment
        source_fpfh = misalign_point_cloud(source)
        fpfh_icp_alignment(source_fpfh, target_depth_sensor, target_pointer)

        # Get error
        error = calculate_error(target_pointer, source_fpfh, reference_points)
        results["fpfh"].append(error)

        # Misalign source data and perform PCA alignment
        source_pca = misalign_point_cloud(source)
        pca_icp_alignment(source_pca, target_depth_sensor, target_pointer)

        # Get error
        error = calculate_error(target_pointer, source_pca, reference_points)
        results["pca"].append(error)

    print(results)
    df = pd.DataFrame.from_dict(results)
    df.to_csv("results/noise.csv", index=False)
    print("Results saved to CSV")
    return results
