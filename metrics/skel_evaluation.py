import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree
import os


# Name of the file (to plot)
def get_name_from_path(path):
    base = os.path.basename(path)  # e.g., "centerline_002.vtp"
    name = os.path.splitext(base)[0]  # remove extension
    return name


# Centerline length
def polyline_length(poly):
    pts = poly.points
    lines = poly.lines

    total_length = 0
    i = 0
    
    # parse the polyline encoding
    while i < len(lines):
        n = lines[i]     # number of point IDs in this polyline
        ids = lines[i+1 : i+1+n]
        
        # sum segment distances for this polyline
        for a, b in zip(ids[:-1], ids[1:]):
            total_length += np.linalg.norm(pts[a] - pts[b])
        
        i += n + 1  # jump to next polyline block

    return total_length


# Node degrees (branching)
# def compute_node_degree(poly):
#     lines = poly.lines.reshape(-1, 3)[:, 1:]
#     degree = np.zeros(poly.n_points, dtype=int)
#     for a, b in lines:
#         degree[a] += 1
#         degree[b] += 1
#     return degree

def compute_node_degree(poly):
    pts = poly.points
    lines = poly.lines

    degree = np.zeros(len(pts), dtype=int)

    i = 0
    while i < len(lines):
        n = lines[i]
        ids = lines[i+1 : i+1+n]

        # Count edges for this polyline
        for a, b in zip(ids[:-1], ids[1:]):
            degree[a] += 1
            degree[b] += 1

        i += n + 1

    return degree


# Chamfer & Hausdorff distance
def chamfer_and_hausdorff(P, Q):
    tree_Q = cKDTree(Q)
    tree_P = cKDTree(P)

    dist_P_to_Q, _ = tree_Q.query(P)
    dist_Q_to_P, _ = tree_P.query(Q)

    chamfer = dist_P_to_Q.mean() + dist_Q_to_P.mean()
    hausdorff = max(dist_P_to_Q.max(), dist_Q_to_P.max())

    return chamfer, hausdorff, dist_P_to_Q, dist_Q_to_P


# MAIN EVALUATION FUNCTION
def evaluate_centerline(pred_path, gt_path):

    # Extract names from paths
    pred_name = get_name_from_path(pred_path)
    gt_name   = get_name_from_path(gt_path)
    
    print("\n1. Loading centerlines")
    pred = pv.read(pred_path)
    gt = pv.read(gt_path)

    P = pred.points
    Q = gt.points
    

    # Chamfer & Hausdorff
    print("\n2. Computing distance metrics")
    chamfer, hausdorff, dP, dQ = chamfer_and_hausdorff(P, Q)

    # Length
    print("\n3. Computing lengths")
    pred_len = polyline_length(pred)
    gt_len = polyline_length(gt)

    # Branching
    print("\n4. Computing topology metrics")
    deg_pred = compute_node_degree(pred)
    deg_gt = compute_node_degree(gt)

    pred_branches = np.sum(deg_pred > 2)
    gt_branches = np.sum(deg_gt > 2)
    
    
    # 3D Visualization
    if show_plot:
        plotter = pv.Plotter()
        plotter.add_title(f"Comparison: {pred_name} vs {gt_name}", font_size=16)

        plotter.add_mesh(pred, color="red", line_width=3, label=f"Pred: {pred_name}")
        plotter.add_mesh(gt,   color="green", line_width=3, label=f"GT:  {gt_name}")

        plotter.add_legend()
        plotter.show()

    # if show_plot:
    #     plotter = pv.Plotter()
    #     plotter.add_title(f"Comparison: {pred_name} vs {gt_name}", font_size=16)

    #     # Plot centerlines
    #     plotter.add_mesh(pred, color="red", line_width=2, label=f"Pred: {pred_name}")
    #     plotter.add_mesh(gt,   color="green", line_width=2, label=f"GT:  {gt_name}")

    #     # ----------------------------
    #     # Highlight Predicted nodes
    #     # ----------------------------
    #     deg_pred = compute_node_degree(pred)

    #     pred_nodes = pv.PolyData(pred.points)

    #     # Color by node degree
    #     pred_nodes["degree"] = deg_pred

    #     # Threshold for bifurcations
    #     bif_nodes_pred = pred_nodes.threshold(value=2.5, scalars="degree")
    #     end_nodes_pred = pred_nodes.threshold(value=1.0, scalars="degree")
        
    #     # Plot spheres for nodes
    #     plotter.add_points(pred_nodes, color="white", point_size=4, render_points_as_spheres=True)
    #     plotter.add_points(bif_nodes_pred, color="yellow", point_size=4, render_points_as_spheres=True)
    #     plotter.add_points(end_nodes_pred, color="blue", point_size=4, render_points_as_spheres=True)

    #     # ----------------------------
    #     # Highlight GT nodes
    #     # ----------------------------
    #     deg_gt = compute_node_degree(gt)
    #     gt_nodes = pv.PolyData(gt.points)
    #     gt_nodes["degree"] = deg_gt

    #     bif_nodes_gt = gt_nodes.threshold(value=2.5, scalars="degree")
    #     end_nodes_gt = gt_nodes.threshold(value=1.0, scalars="degree")

    #     plotter.add_points(gt_nodes, color="lightgray", point_size=4, render_points_as_spheres=True)
    #     plotter.add_points(bif_nodes_gt, color="yellow", point_size=4, render_points_as_spheres=True)
    #     plotter.add_points(end_nodes_gt, color="blue", point_size=4, render_points_as_spheres=True)

    #     plotter.add_legend()
    #     plotter.show()

    # results
    print("\n================ Evaluation Results ================")
    print(f"Loaded pred: {pred_name} ({len(P)} points)")
    print(f"Loaded GT  : {gt_name} ({len(Q)} points)")
    print("--------------------------------------------------------------")
    print(f"Chamfer Distance:     {chamfer:.4f}")
    print(f"Hausdorff Distance:   {hausdorff:.4f}")
    print(f"Pred length:          {pred_len:.2f}")
    print(f"GT length:            {gt_len:.2f}")
    print(f"Length difference:    {pred_len - gt_len:.2f}")
    print(f"Branches (pred):      {pred_branches}")
    print(f"Branches (gt):        {gt_branches}")
    print("==============================================================")

    return {
        "chamfer": chamfer,
        "hausdorff": hausdorff,
        "pred_length": pred_len,
        "gt_length": gt_len,
        "length_diff": pred_len - gt_len,
        "pred_branches": pred_branches,
        "gt_branches": gt_branches,
        "dist_pred_to_gt": dP,
        "dist_gt_to_pred": dQ,
    }



if __name__ == "__main__":
    n=2
    pred_vtp = f"C:/Users/ducci/Documents/Università_2025/6_SemesterProject/BrainGraph/output/Output_prova/ExCenterline_00{n}.vtp"
    # pred_vtp = f"C:/Users/ducci/Documents/Università_2025/6_SemesterProject/BrainGraph/output/Output_basic_extractor/BasicCenterline_00{n}.vtp"
    gt_vtp   = f"C:/Users/ducci/Documents/Università_2025/6_SemesterProject/BrainGraph/data/ITKTubeTK_GoldStandardVtp/VascularNetwork-00{n}.vtp"
    
    show_plot = False

    evaluate_centerline(pred_vtp, gt_vtp)
