"""
Converts a 3D binary vessel mask (NIfTI or .npy) into a .vtp file
representing the 3D centerline skeleton (single-file version).

Laplacian smoothing moves each point toward the average position of
its connected neighbors, reducing sharp angles and noise while preserving the overall shape.


USAGE:
    python laplace_extractor.py --input /path/to/mask.nii.gz --output /path/to/save/skeleton.vtp \
        --lap_iters 2 --lap_alpha 0.5

    ONLY INPUT IS REQUIRED TO RUN THIS.
    By default, output is skeleton.vtp file. lap_iters is 2 and lap_alpha is 0.5.
"""

import os
import argparse
import numpy as np
import nibabel as nib
import networkx as nx
import pyvista as pv
from skimage.morphology import skeletonize_3d
from scipy.ndimage import binary_closing, generate_binary_structure


def laplacian_smooth_graph(graph, iterations=2, alpha=0.5):
    """
    Smooth node positions along the graph edges.
    - iterations: number of smoothing passes
    - alpha: smoothing factor (0 < alpha <= 1), higher = more smoothing
    """
    pos = {node: np.array(node, dtype=float) for node in graph.nodes()}
    for _ in range(iterations):
        new_pos = {}
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            if not neighbors:
                continue
            neighbor_avg = np.mean([pos[n] for n in neighbors], axis=0)
            new_pos[node] = (1 - alpha) * pos[node] + alpha * neighbor_avg
        pos.update(new_pos)
    return pos


def load_mask(input_path):
    """Loads a NIfTI or .npy file into a boolean numpy array."""
    print(f"Loading mask from {input_path}...")
    ext = os.path.splitext(input_path)[1].lower()

    if ext == '.npy':
        arr = np.load(input_path)
        affine = np.eye(4)
    elif ext in ('.nii', '.gz'):
        if ext == '.gz' and os.path.splitext(input_path)[0].endswith('.nii'):
            ext = '.nii.gz'
        img = nib.load(input_path)
        arr = img.get_fdata()
        affine = img.affine
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .nii, .nii.gz, or .npy.")

    mask_array = arr > 0
    print(f"Mask loaded. Shape: {mask_array.shape}, Voxels: {np.sum(mask_array)}")
    return mask_array, affine


def build_graph(skeleton):
    """Build a graph from a 3D skeleton voxel array."""
    print("Building graph from skeleton voxels...")
    G = nx.Graph()
    shape = skeleton.shape
    fibers = np.argwhere(skeleton)
    if fibers.size == 0:
        print("Warning: Skeleton is empty.")
        return G

    voxel_set = set(map(tuple, fibers))
    for v in fibers:
        coord = tuple(v)
        G.add_node(coord)
        x, y, z = coord
        for i in range(max(0, x-1), min(shape[0], x+2)):
            for j in range(max(0, y-1), min(shape[1], y+2)):
                for k in range(max(0, z-1), min(shape[2], z+2)):
                    neighbor = (i, j, k)
                    if neighbor != coord and neighbor in voxel_set:
                        weight = np.linalg.norm(np.array(coord) - np.array(neighbor))
                        G.add_edge(coord, neighbor, weight=weight)
    print(f"Graph built. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G


def prune_graph(G):
    """Prune simple triangles (cycles of length 3)."""
    print("Pruning small triangles from graph...")
    edges_removed = 0
    edges_to_check = list(G.edges())
    processed_triangles = set()

    for u, v in edges_to_check:
        if not G.has_edge(u, v):
            continue
        try:
            u_neighbors = set(G.neighbors(u))
            v_neighbors = set(G.neighbors(v))
        except nx.NetworkXError:
            continue
        common_neighbors = u_neighbors.intersection(v_neighbors)

        for w in common_neighbors:
            triangle = tuple(sorted((u, v, w)))
            if triangle in processed_triangles:
                continue
            if not (G.has_edge(u, v) and G.has_edge(v, w) and G.has_edge(u, w)):
                continue
            w_uv = G[u][v].get('weight', 0)
            w_vw = G[v][w].get('weight', 0)
            w_uw = G[u][w].get('weight', 0)
            edges = [((u, v), w_uv), ((v, w), w_vw), ((u, w), w_uw)]
            heaviest = max(edges, key=lambda x: x[1])
            n1, n2 = heaviest[0]
            if G.has_edge(n1, n2):
                G.remove_edge(n1, n2)
                edges_removed += 1
                processed_triangles.add(triangle)
    print(f"Pruning complete. Edges removed: {edges_removed}")
    return G


def save_graph_to_vtp(graph, output_vtp_path, affine=None, laplacian_iters=2, laplacian_alpha=0.5):
    """Save a networkx graph as a .vtp file (only Laplacian smoothing)."""
    if graph.number_of_nodes() == 0:
        print("Graph is empty. Skipping save.")
        return

    print("Converting graph to VTP format...")

    # Laplacian smoothing (voxel-space)
    pos_voxel = laplacian_smooth_graph(graph, iterations=laplacian_iters, alpha=laplacian_alpha)

    # Convert to array
    nodes_list = list(graph.nodes())
    points = np.array([pos_voxel[n] for n in nodes_list], dtype=np.float64)

    # Apply affine (voxel → world)
    if affine is not None:
        print("Applying affine transformation...")
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
        points = homogeneous_points.dot(affine.T)[:, :3]

    # Build lines for PyVista
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}
    lines_list = []
    for u, v in graph.edges():
        lines_list.extend([2, node_to_idx[u], node_to_idx[v]])

    # Save
    poly = pv.PolyData(points)
    poly.lines = np.array(lines_list)
    poly.save(output_vtp_path)
    print(f"Saved centerline to {output_vtp_path}")


def process_file(input_path, output_path, lap_iters=2, lap_alpha=0.5):
    """Process a single mask file and save skeleton as .vtp."""
    mask, affine = load_mask(input_path)
    print("Pre-smoothing mask before skeletonization...")
    mask = binary_closing(mask, structure=generate_binary_structure(3, 2))
    print("Computing 3D skeleton...")
    skel = skeletonize_3d(mask)
    print("Skeleton computation complete.")
    graph = build_graph(skel)
    pruned_graph = prune_graph(graph)
    save_graph_to_vtp(pruned_graph, output_path, affine=affine,
                      laplacian_iters=lap_iters, laplacian_alpha=lap_alpha)


def main():
    parser = argparse.ArgumentParser(description="Convert a 3D vessel mask to a .vtp skeleton (single-file).")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input mask file (.nii.gz or .npy)")
    parser.add_argument("--output", "-o", type=str, default="skeleton.vtp", help="Path to output .vtp file")
    parser.add_argument("--lap_iters", type=int, default=2, help="Laplacian smoothing iterations")
    parser.add_argument("--lap_alpha", type=float, default=0.5, help="Laplacian smoothing alpha (0–1)")
    args = parser.parse_args()

    print(f"Processing single file: {args.input}")
    process_file(args.input, args.output, args.lap_iters, args.lap_alpha)
    print("\n✅ Processing complete.")


if __name__ == "__main__":
    main()
