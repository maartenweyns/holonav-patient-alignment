import os

import open3d as o3d
import pandas as pd

from experiments.run_algorithms import run_algorithms


def occlusion_test() -> dict:
    data_path = os.path.dirname(os.path.abspath(__file__)) + "/../data"
    results = {"angle": [], "fpfh": [], "pca": []}

    for angle in ["0deg", "5deg_right", "10deg_right", "15deg_right", "20deg_right", "25deg_right", "30deg_right", "35deg_right", "45deg_right"]:
        results["angle"].append(angle)
        print("Testing angle: {}".format(angle))

        # Read data
        source = o3d.io.read_point_cloud(data_path + "/skull1/skull1_preop_model.ply")
        target_depth_sensor = o3d.io.read_point_cloud(data_path + "/skull1/occlusion/skull1_{}.ply".format(angle))
        target_pointer = o3d.io.read_point_cloud(data_path + "/skull1/skull1_pointer.txt", format="xyz")

        run_algorithms(source, target_depth_sensor, target_pointer, results)

    df = pd.DataFrame.from_dict(results)
    df.to_csv(os.path.dirname(os.path.abspath(__file__)) + "/results/occlusion.csv", index=False)
    print("Results saved to CSV")
    return results
