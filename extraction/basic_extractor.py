"""
Converts a 3D binary vessel mask (NIfTI or .npy) into a .vtp file
representing the 3D centerline skeleton.


USAGE:
    python create_centerline_vtp.py /path/to/your/mask.nii.gz --output /path/to/save/skeleton.vtp
"""

import os
import argparse
import numpy as np
import nibabel as nib
import networkx as nx
import pyvista as pv
from skimage.morphology import skeletonize
# Note: Removed scipy and cleaning imports

def load_mask(input_path):
    """
    Loads a NIfTI or .npy file into a boolean numpy array.

    Returns:
        (mask_array, affine)
        - mask_array: A 3D boolean numpy array (True for vessel).
        - affine: The affine matrix for NIfTI, or np.eye(4) for .npy.
    """
    print(f"Loading mask from {input_path}...")
    ext = os.path.splitext(input_path)[1].lower()

    if ext == '.npy':
        arr = np.load(input_path)
        affine = np.eye(4)
    elif ext in ('.nii', '.gz'):
        if ext == '.gz' and os.path.splitext(input_path)[0].endswith('.nii'):
            ext = '.nii.gz' # Handle .nii.gz

        try:
            img = nib.load(input_path)
            arr = img.get_fdata()
            affine = img.affine
        except Exception as e:
            raise IOError(f"Error loading NIfTI file: {e}")
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .nii, .nii.gz, or .npy.")

    mask_array = arr > 0
    print(f"Mask loaded. Shape: {mask_array.shape}, Voxels: {np.sum(mask_array)}")
    return mask_array, affine

def build_graph(skeleton):
    """
    Builds a networkx graph from a 3D skeleton voxel array.
    Each voxel in the skeleton becomes a node, and adjacent
    skeleton voxels are connected by edges.
    """
    print("Building graph from skeleton voxels...")
    G = nx.Graph()
    shape = skeleton.shape

    # Get all (x, y, z) coordinates of skeleton voxels
    fibers = np.argwhere(skeleton)

    if fibers.size == 0:
        print("Warning: Skeleton is empty. Graph will be empty.")
        return G

    # Create a set of voxel coordinates for fast lookup
    voxel_set = set(map(tuple, fibers))

    for v in fibers:
        coord = tuple(v)
        G.add_node(coord)
        x, y, z = coord

        # Check all 26 neighbors
        for i in range(max(0, x-1), min(shape[0], x+2)):
            for j in range(max(0, y-1), min(shape[1], y+2)):
                for k in range(max(0, z-1), min(shape[2], z+2)):
                    neighbor = (i, j, k)
                    if neighbor != coord and neighbor in voxel_set:
                        # Add an edge with weight = Euclidean distance
                        weight = np.linalg.norm(np.array(coord) - np.array(neighbor))
                        G.add_edge(coord, neighbor, weight=weight)

    print(f"Graph built. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G

def prune_graph(G):
    """
    Prune G by detecting all simple cycles of length 3 (triangles) and removing the
    heaviest edge in each triangle (based on 'weight').
    This version is much faster as it iterates over edges, not all cycles.
    Modifies G in-place and returns it.
    """
    print("Pruning small triangles from graph (fast method)...")
    edges_removed = 0
    # We need a copy of the edges to iterate over, as we'll be modifying the graph
    edges_to_check = list(G.edges())

    # Keep track of triangles we've already processed to avoid redundant work
    processed_triangles = set()

    for u, v in edges_to_check:
        # If the edge has already been removed by a previous iteration, skip
        if not G.has_edge(u, v):
            continue

        # Find common neighbors of u and v
        # G.neighbors(n) is a generator, set() is fast for intersection
        try:
            u_neighbors = set(G.neighbors(u))
            v_neighbors = set(G.neighbors(v))
        except nx.NetworkXError:
            # This can happen if a node was isolated and removed
            continue

        # common_neighbors contains all nodes 'w' that form a triangle (u, v, w)
        common_neighbors = u_neighbors.intersection(v_neighbors)

        for w in common_neighbors:
            # We found a triangle (u, v, w).
            # Sort nodes to create a unique triangle identifier
            triangle = tuple(sorted((u, v, w)))

            # If we've already broken this triangle, skip
            if triangle in processed_triangles:
                continue

            # --- FIX: Check if all 3 edges still exist ---
            # This prevents a KeyError if a previous iteration of this
            # w-loop has already removed one of these edges.
            if not (G.has_edge(u, v) and G.has_edge(v, w) and G.has_edge(u, w)):
                continue # This triangle was already broken, skip.

            # Get edge weights, handle missing edges (if one was removed)
            try:
                w_uv = G[u][v].get('weight', 0)
                w_vw = G[v][w].get('weight', 0)
                w_uw = G[u][w].get('weight', 0)
            except KeyError:
                # This should be caught by the check above, but serves as a safety net
                continue

            edges = [((u, v), w_uv), ((v, w), w_vw), ((u, w), w_uw)]

            # Find heaviest edge
            heaviest = max(edges, key=lambda x: x[1])
            n1, n2 = heaviest[0]

            # Remove the heaviest edge
            if G.has_edge(n1, n2):
                G.remove_edge(n1, n2)
                edges_removed += 1
                processed_triangles.add(triangle)

    print(f"Pruning complete. Edges removed: {edges_removed}")
    return G

def save_graph_to_vtp(graph, output_vtp_path, affine=None):
    """
    Saves a networkx graph as a .vtp file using PyVista.

    Args:
        graph: The networkx graph to save.
        output_vtp_path: The file path to save to (e.g., "skeleton.vtp").
        affine: (Optional) 4x4 affine matrix to transform node coordinates
                from voxel space to world space.
    """
    if graph.number_of_nodes() == 0:
        print("Graph is empty. Skipping .vtp save.")
        return

    print("Converting graph to VTP format...")
    # 1. Get nodes (points) as a (N, 3) numpy array
    nodes_list = list(graph.nodes())
    points = np.array(nodes_list, dtype=np.float64)

    # 2. Apply affine transformation if provided
    if affine is not None:
        print("Applying affine transformation to node coordinates...")
        # Convert to homogeneous coordinates (add a '1' to each)
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
        # Apply affine matrix
        transformed_points = homogeneous_points.dot(affine.T)
        # Convert back to 3D coordinates
        points = transformed_points[:, :3]

    # 3. Create a mapping from node coordinate to point index
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}

    # 4. Build the 'lines' array for PyVista
    # Format: [num_points_in_line, idx1, idx2, num_points_in_line, idx3, idx4, ...]
    lines_list = []
    for u, v in graph.edges():
        lines_list.extend([2, node_to_idx[u], node_to_idx[v]])

    # 5. Create the PolyData object
    poly = pv.PolyData(points)
    poly.lines = np.array(lines_list)

    # 6. Save to file
    try:
        poly.save(output_vtp_path)
        print(f"Successfully saved centerline to {output_vtp_path}")
    except Exception as e:
        print(f"Error saving .vtp file: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Create a .vtp centerline file from a 3D vessel mask."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input vessel mask (.nii, .nii.gz, or .npy)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="skeleton.vtp",
        help="Path to save the output .vtp file (default: skeleton.vtp)"
    )
    # --- All cleaning arguments have been removed ---
    args = parser.parse_args()

    # 1. Load mask
    mask, affine = load_mask(args.input)

    # --- All cleaning steps have been removed ---

    # 2. Compute skeleton
    print("Computing 3D skeleton...")
    # We now use the raw 'mask' directly
    skel = skeletonize(mask)
    print("Skeleton computation complete.")

    # 3. Build graph
    graph = build_graph(skel)

    # 4. Prune graph
    pruned_graph = prune_graph(graph)

    # 5. Save to VTP
    # Pass the affine to save in world coordinates
    save_graph_to_vtp(pruned_graph, args.output, affine=affine)

if __name__ == "__main__":
    main()