import copy
import os
import random

import numpy as np
import open3d as o3d
import pandas as pd

from general.evaluation import prepare_reference_points, calculate_error
from icp.p2p_icp import p2p_icp_registration
from mps.manual_icp_alignment import manual_icp_alignment
from general.preprocessing import get_misalign_translation_rotation, misalign_point_cloud, add_noise
from pca.utils import get_pca_translation_rotation


def mps_points_test() -> dict:
    """
    Test the manual point selection algorithm with different amounts of
    selected points.

    :return: A dictionary containing the results
    """
    data_path = os.path.dirname(os.path.abspath(__file__)) + "/../data"
    results = {"points": [], "mps": [], "mps+icp": []}

    for num_points in range(20):
        results["points"].append(num_points + 1)
        print("Testing points: {}".format(num_points + 1))

        # Read data
        source = o3d.io.read_point_cloud(data_path + "/skull1/skull1_preop_model.ply")
        target_depth_sensor = o3d.io.read_point_cloud(data_path + "/skull1/occlusion/skull1_0deg.ply")
        target_pointer = o3d.io.read_point_cloud(data_path + "/skull1/skull1_pointer.txt", format="xyz")

        # Sample random points from the target pointer point cloud
        indices = random.sample(range(len(target_pointer.points) - 1), num_points)
        selected_points = np.asarray(target_depth_sensor.points)[indices]

        # Create new point clouds from selected points
        selected_target = o3d.geometry.PointCloud()
        selected_target.points = o3d.utility.Vector3dVector(selected_points)
        selected_source = copy.deepcopy(selected_target)

        # Run MPS experiment
        run_mps_experiment(
            source,
            target_depth_sensor,
            target_pointer,
            selected_source,
            selected_target,
            results)

    df = pd.DataFrame.from_dict(results)
    df.to_csv(os.path.dirname(os.path.abspath(__file__)) + "/results/skull1/mps-points.csv", index=False)
    print("Results saved to CSV")
    return results


def mps_noise_test() -> dict:
    """
    Test the manual point selection algorithm with different amounts of
    selected points.

    :return: A dictionary containing the results
    """
    data_path = os.path.dirname(os.path.abspath(__file__)) + "/../data"
    results = {"noise": [], "mps": [], "mps+icp": []}

    for noise in np.arange(0, 6.5, 0.5):
        results["noise"].append(noise)
        print("Testing mps noise: {}".format(noise))

        # Read data
        source = o3d.io.read_point_cloud(data_path + "/skull1/skull1_preop_model.ply")
        target_depth_sensor = o3d.io.read_point_cloud(data_path + "/skull1/occlusion/skull1_0deg.ply")
        target_pointer = o3d.io.read_point_cloud(data_path + "/skull1/skull1_pointer.txt", format="xyz")

        # Sample random points from the target pointer point cloud
        indices = random.sample(range(len(target_pointer.points) - 1), 10)
        selected_points = np.asarray(target_depth_sensor.points)[indices]

        # Create new point clouds from selected points
        selected_target = o3d.geometry.PointCloud()
        selected_target.points = o3d.utility.Vector3dVector(selected_points)
        selected_source = copy.deepcopy(selected_target)

        # Add noise to selected target points
        selected_target = add_noise(selected_target, noise_amt=noise)

        # Run MPS experiment
        run_mps_experiment(
            source,
            target_depth_sensor,
            target_pointer,
            selected_source,
            selected_target,
            results)

    df = pd.DataFrame.from_dict(results)
    df.to_csv(os.path.dirname(os.path.abspath(__file__)) + "/results/skull1/mps-noise.csv", index=False)
    print("Results saved to CSV")
    return results


def run_mps_experiment(
        source,
        target_depth_sensor,
        target_pointer,
        selected_source,
        selected_target,
        results):
    # Prepare evaluation
    reference_points = prepare_reference_points(target_pointer, source)

    # Misalign selected source point cloud
    t, r = get_misalign_translation_rotation()
    selected_source.rotate(r, center=source.get_center())
    selected_source.translate(t)

    # Misalign source point cloud
    source = misalign_point_cloud(source)

    # Perform alignment
    # Get translation and rotation from PCA algorithm
    t, r = get_pca_translation_rotation(selected_source, selected_target)

    # Transform source to target
    source.rotate(r, center=selected_source.get_center())
    source.translate(t)
    mps_error = calculate_error(target_pointer, source, reference_points)

    # Perform ICP precise registration
    p2p_icp_registration(source, target_depth_sensor)
    p2p_icp_registration(source, target_pointer)

    # Evaluate
    mps_icp_error = calculate_error(target_pointer, source, reference_points)
    results["mps"].append(round(mps_error, 5))
    results["mps+icp"].append(round(mps_icp_error, 5))
