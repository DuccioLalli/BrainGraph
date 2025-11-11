#!/usr/bin/env python
"""
Compute geometric distance between two 3D meshes (.vtp) using KD-Tree nearest-neighbor search.

DESCRIPTION:
    This script measures how close a predicted 3D mesh (e.g., a skeletonized vessel) is
    to a ground truth mesh (e.g., a manual annotation or dense segmentation surface).

USAGE:
    python mesh_RMSE.py --truth annotation.vtp --prediction skeleton.vtp


"""
import argparse
import pyvista as pv
import numpy as np
import sys
import os

# We will use SciPy's KDTree for a fast, robust nearest-neighbor search
try:
    from scipy.spatial import KDTree
except ImportError:
    print("Error: 'scipy' library not found.", file=sys.stderr)
    print("Please install it: pip install scipy", file=sys.stderr)
    sys.exit(1)


def compute_mesh_distance(truth_vtp_path, prediction_vtp_path):
    """
    Computes the one-way distance from the prediction mesh to the truth mesh
    using a fast KD-Tree.

    Args:
        truth_vtp_path (str): Path to the ground truth VTP (e.g., annotation).
        prediction_vtp_path (str): Path to the prediction VTP (e.g., skeleton).
    """

    # --- 1. Load Meshes ---
    print(f"Loading Ground Truth (Annotation): {truth_vtp_path}")
    try:
        truth_mesh = pv.read(truth_vtp_path)
    except Exception as e:
        print(f"Error loading ground truth file: {e}", file=sys.stderr)
        return

    print(f"Loading Prediction (Skeleton): {prediction_vtp_path}")
    try:
        prediction_mesh = pv.read(prediction_vtp_path)
    except Exception as e:
        print(f"Error loading prediction file: {e}", file=sys.stderr)
        return

    # --- 2. Check for empty meshes ---
    if truth_mesh.n_points == 0:
        print("Error: Ground truth file has 0 points.", file=sys.stderr)
        return
    if prediction_mesh.n_points == 0:
        print("Error: Prediction file has 0 points.", file=sys.stderr)
        return

    print(f"  Truth points (Annotation): {truth_mesh.n_points}")
    print(f"  Prediction points (Skeleton): {prediction_mesh.n_points}")

    # --- 3. Build KD-Tree from Truth Points ---
    print("\nBuilding KD-Tree from truth mesh (this may take a moment)...")

    # Get all (X,Y,Z) points from the ground truth annotation
    truth_points = truth_mesh.points

    # Create the KD-Tree. This is a fast-lookup spatial structure.
    tree = KDTree(truth_points)

    # --- 4. Query Tree with Prediction Points ---
    print("Querying tree with all prediction points...")

    # Get all (X,Y,Z) points from the skeleton
    prediction_points = prediction_mesh.points

    # Query the tree: For each prediction_point, find the nearest point
    # in the tree (which was built from truth_points).
    # This returns the distances and the indices of the closest points.
    distances, closest_indices = tree.query(prediction_points, k=1)

    print("Distance computation complete.")

    # --- 5. Calculate Metrics ---

    # We already have the distances, so this is easy.
    squared_distances = np.square(distances)

    # Calculate metrics
    mae = np.mean(distances)
    mse = np.mean(squared_distances)
    rmse = np.sqrt(mse)

    # --- 6. Print Results ---
    print("\n--- 📊 Results ---")
    print("Metrics represent the error of the 'Prediction' (Skeleton) relative to the 'Truth' (Annotation).")
    print(f"\n  Mean Absolute Error (MAE):   {mae:.6f}")
    print(f"  Root Mean Squared Error (RMSE):{rmse:.6f}")

    print("\n--------------------------")
    print(f"  Final Score (MSE): {mse:.6f}")
    print("--------------------------")

    return rmse


def main():
    parser = argparse.ArgumentParser(
        description="""
        Computes the Mean Squared Error (Distance) from a Prediction VTP (e.g., Skeleton)
        to a Ground Truth VTP (e.g., manual annotation).
        Uses SciPy KDTree for fast, robust computation.
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-t", "--truth",
        required=True,
        help="Path to the ground truth VTP (e.g., your manual annotation, the DENSE mesh)."
    )
    parser.add_argument(
        "-p", "--prediction",
        required=True,
        help="Path to the prediction VTP (e.g., your skeleton, the SPARSE mesh)."
    )

    args = parser.parse_args()

    # --- Input validation ---
    if not os.path.exists(args.truth):
        print(f"Error: Ground truth file not found at {args.truth}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.prediction):
        print(f"Error: Prediction file not found at {args.prediction}", file=sys.stderr)
        sys.exit(1)

    print("--- VTP MSE Computation (SciPy Edition) ---")
    compute_mesh_distance(args.truth, args.prediction)
    print("\nDone.")


if __name__ == "__main__":
    main()