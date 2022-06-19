import os

import numpy as np
import open3d as o3d
import pandas as pd

from experiments.run_algorithms import run_algorithms
from general.preprocessing import add_noise


def noise_test() -> dict:
    """
    Test the different algorithms against different levels of noise. Their
    default settings are used.

    :return: A dictionary containing the results
    """
    data_path = os.path.dirname(os.path.abspath(__file__)) + "/../data"
    results = {"noise": [], "fpfh": [], "fpfh+icp": [], "pca": [], "pca+icp": []}

    for noise in np.arange(0, 6.5, 0.5):
        results["noise"].append(noise)
        print("Testing noise level: {}".format(noise))

        # Read data
        source = o3d.io.read_point_cloud(data_path + "/skull1/skull1_preop_model.ply")
        target_depth_sensor = o3d.io.read_point_cloud(data_path + "/skull1/skull1_preop_model.ply")
        target_pointer = o3d.io.read_point_cloud(data_path + "/skull1/skull1_pointer.txt", format="xyz")

        # Add noise
        target_depth_sensor = add_noise(target_depth_sensor, noise_amt=noise)

        run_algorithms(source, target_depth_sensor, target_pointer, results)

    df = pd.DataFrame.from_dict(results)
    df.to_csv(os.path.dirname(os.path.abspath(__file__)) + "/results/skull1/noise.csv", index=False)
    print("Results saved to CSV")
    return results
