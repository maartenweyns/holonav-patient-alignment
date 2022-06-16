import os

import numpy as np
import open3d as o3d
import pandas as pd

from experiments.run_algorithms import run_algorithms
from general.preprocessing import downsample_point_cloud


def sparsity_test() -> dict:
    """
    Test the different algorithms against different levels of sparsity. Their
    default settings are used.

    :return: A dictionary containing the results
    """
    data_path = os.path.dirname(os.path.abspath(__file__)) + "/../data"
    results = {"voxel_size": [], "num_points": [], "fpfh": [], "pca": []}

    for vs in range(10):
        results["voxel_size"].append(vs + 1)
        print("Testing Voxel Size: {}".format(vs + 1))

        # Read data
        source = o3d.io.read_point_cloud(data_path + "/skull1/skull1_preop_model.ply")
        target_depth_sensor = o3d.io.read_point_cloud(data_path + "/skull1/occlusion/skull1_0deg.ply")
        target_pointer = o3d.io.read_point_cloud(data_path + "/skull1/skull1_pointer.txt", format="xyz")

        # Downsample data
        target_depth_sensor = downsample_point_cloud(target_depth_sensor, vs + 1)
        results["num_points"].append(len(np.asarray(target_depth_sensor.points)))

        run_algorithms(source, target_depth_sensor, target_pointer, results)

    df = pd.DataFrame.from_dict(results)
    df.to_csv(os.path.dirname(os.path.abspath(__file__)) + "/results/skull1/sparsity.csv", index=False)
    print("Results saved to CSV")
    return results
