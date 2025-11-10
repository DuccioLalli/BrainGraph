# ICP (Iterative Closest Point) is used to align the predicted centerline to the
# ground-truth by estimating the best rotation/translation (and optionally scale).
# This ensures the two shapes are in the same pose before computing distance.
# Without ICP, the RMSE would also measure differences in position/orientation,
# not only the true geometric error between the centerlines.

# Chamfer RMSE
import argparse
import numpy as np
import vtk
import os
from scipy.spatial import cKDTree

def _nn_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Return nearest-neighbor distances from each point in A to point-set B.
    A: (NA, 3), B: (NB, 3)
    """
    d, _ = cKDTree(B).query(A, k=1)
    return d

# ---- I/O ----
def read_vtp_points(path: str) -> tuple[vtk.vtkPolyData, np.ndarray]:
    """Read .vtp as vtkPolyData and return also Nx3 numpy points (in world units)."""
    rdr = vtk.vtkXMLPolyDataReader()
    rdr.SetFileName(path)
    rdr.Update()
    poly = rdr.GetOutput()
    pts_vtk = poly.GetPoints()
    n = pts_vtk.GetNumberOfPoints()
    pts = np.array([pts_vtk.GetPoint(i) for i in range(n)], dtype=np.float64)
    return poly, pts

def write_vtp(poly: vtk.vtkPolyData, out_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(out_path)
    w.SetInputData(poly)
    w.SetDataModeToBinary()
    if w.Write() == 0:
        raise IOError(f"Error writing '{out_path}'")

# ---- ICP alignment (optional) ----
def icp_align(source_poly: vtk.vtkPolyData,
              target_poly: vtk.vtkPolyData,
              mode: str = "rigid",
              max_iters: int = 200,
              start_by_matching_centroids: bool = True) -> vtk.vtkPolyData:
    """
    Align source -> target with VTK ICP. mode in {'rigid','similarity'}.
    Returns a new vtkPolyData with the transform applied.
    """
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source_poly)
    icp.SetTarget(target_poly)
    if mode == "rigid":
        icp.GetLandmarkTransform().SetModeToRigidBody()
    elif mode == "similarity":
        icp.GetLandmarkTransform().SetModeToSimilarity()  # allows uniform scale
    else:
        raise ValueError("mode must be 'rigid' or 'similarity'")
    icp.SetMaximumNumberOfIterations(max_iters)
    if start_by_matching_centroids:
        icp.StartByMatchingCentroidsOn()
    icp.Update()

    tf = vtk.vtkTransform()
    tf.SetMatrix(icp.GetMatrix())

    tfilter = vtk.vtkTransformPolyDataFilter()
    tfilter.SetInputData(source_poly)
    tfilter.SetTransform(tf)
    tfilter.Update()
    return tfilter.GetOutput()

# ---- Metrics ----
def chamfer_rmse(A: np.ndarray, B: np.ndarray) -> tuple[float, float, float]:
    """
    Symmetric Chamfer RMSE between point-sets A and B (both Nx3 in same units).
    Returns (rmse_sym, rmse_A2B, rmse_B2A).
    """
    dAB = _nn_distances(A, B)
    dBA = _nn_distances(B, A)
    rmse_A2B = float(np.sqrt(np.mean(dAB**2)))
    rmse_B2A = float(np.sqrt(np.mean(dBA**2)))
    rmse_sym = float(np.sqrt((np.mean(dAB**2) + np.mean(dBA**2)) / 2.0))
    return rmse_sym, rmse_A2B, rmse_B2A

def maybe_downsample(pts: np.ndarray, max_points: int | None, seed: int = 0) -> np.ndarray:
    if max_points is None or pts.shape[0] <= max_points:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(pts.shape[0], size=max_points, replace=False)
    return pts[idx]

# ---- CLI ----
def main():
    ap = argparse.ArgumentParser(
        description="Compute distance between two centerlines (.vtp) in the same coordinate system."
    )
    ap.add_argument("gt_vtp", help="Ground truth centerline (.vtp)")
    ap.add_argument("pred_vtp", help="Predicted/extracted centerline (.vtp)")
    ap.add_argument("--align", choices=["none", "rigid", "similarity"], default="rigid",
                    help="Optional ICP alignment of PRED to GT before measuring (default: rigid).")
    ap.add_argument("--max-points", type=int, default=100000,
                    help="Randomly downsample each point-set to this many points for speed (default: 100000).")
    ap.add_argument("--save-aligned", help="If set, save the aligned PRED .vtp here.")
    args = ap.parse_args()

    # Read
    gt_poly, gt_pts = read_vtp_points(args.gt_vtp)
    pred_poly, pred_pts = read_vtp_points(args.pred_vtp)

    # ICP (Iterative Closest Point) aligns the predicted centerline to the ground-truth
    # by estimating the best rotation/translation (and optionally scale). This ensures the
    # two shapes are in the same pose before computing RMSE, avoiding pose errors.
    if args.align != "none":
        pred_poly = icp_align(pred_poly, gt_poly, mode=args.align)
        pred_pts = np.array([pred_poly.GetPoint(i) for i in range(pred_poly.GetNumberOfPoints())],
                            dtype=np.float64)
        if args.save_aligned:
            write_vtp(pred_poly, args.save_aligned)

    # Downsample for speed if very dense
    gt_pts_d   = maybe_downsample(gt_pts, args.max_points)
    pred_pts_d = maybe_downsample(pred_pts, args.max_points)

    # Distance
    rmse_sym, rmse_A2B, rmse_B2A = chamfer_rmse(gt_pts_d, pred_pts_d)

    # Report
    print("\n=== Centerline Distance (Chamfer RMSE) ===")
    print(f"GT points:   {gt_pts.shape[0]}  (used {gt_pts_d.shape[0]})")
    print(f"PRED points: {pred_pts.shape[0]}  (used {pred_pts_d.shape[0]})")
    print(f"Alignment:   {args.align}")
    print(f"RMSE (GT→PRED): {rmse_A2B:.3f}")
    print(f"RMSE (PRED→GT): {rmse_B2A:.3f}")
    print(f"RMSE (symmetric): {rmse_sym:.3f}\n")
    print("Tip: GT→PRED basso ma PRED→GT alto = la PRED manca dei rami presenti nella GT.")

if __name__ == "__main__":
    main()
