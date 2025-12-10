"""
Converts a 3D binary vessel mask (NIfTI or .npy) into a .vtp file
representing the 3D centerline skeleton.


USAGE:
    python basic_extractor.py /path/to/your/mask.nii.gz --output /path/to/save/skeleton.vtp
"""

import os
import argparse
import numpy as np
import nibabel as nib
import networkx as nx
import pyvista as pv
import trimesh
import skeletor as sk


def load_mask(input_path, debug):
    """
    Loads a NIfTI or .npy file into a boolean numpy array.

    Returns:
        (mask_array, affine)
        - mask_array: A 3D boolean numpy array (True for vessel).
        - affine: The affine matrix for NIfTI, or np.eye(4) for .npy.
    """
    print(f"1. Loading the mask")
    ext = os.path.splitext(input_path)[1].lower()

    if ext == '.npy':
        arr = np.load(input_path)
        affine = np.eye(4)
    elif ext in ('.nii', '.gz'):
        if ext == '.gz' and os.path.splitext(input_path)[0].endswith('.nii'):
            ext = '.nii.gz' # for .nii.gz

        try:
            img = nib.load(input_path)
            arr = img.get_fdata()
            affine = img.affine
        except Exception as e:
            raise IOError(f"Error loading NIfTI file: {e}")
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .nii, .nii.gz, or .npy.")

    mask_array = arr > 0
    
    # Plot the segmentation
    if debug:
        view_mask_3d(mask_array, affine, title="Loaded Mask")
        
    print(f"-- Mask loaded. Shape: {mask_array.shape}, Voxels: {np.sum(mask_array)}")
    return mask_array, affine

def compute_skeleton(mask, debug=False):

    print("2. Extracting the skeleton")

    # Convert bool to float32 (vol because is a volume/voxel representation)
    vol = mask.astype(np.float32)

    # Extract surface from mask
    grid = pv.wrap(vol)
    surf = grid.contour(isosurfaces=[0.5]).triangulate()

    # Convert PyVista mesh → trimesh
    vertices = surf.points
    faces = surf.faces.reshape(-1, 4)[:, 1:]

    tm_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Skeletonization using skeletor library (https://pypi.org/project/skeletor/)
    skel = sk.skeletonize.by_wavefront(tm_mesh, waves=1, step_size=1)   # I want to test more methods (Documentation: https://navis-org.github.io/skeletor/)

    # Information printing
    num_nodes = skel.vertices.shape[0]
    num_edges = skel.edges.shape[0]
    print(f"-- Skeleton Extracted. Nodes: {num_nodes}, Edges: {num_edges}")
    
    # Plot the centerline using the skeletor setup
    if debug:
        skel.show(mesh=True)

    return skel

def view_mask_3d(mask, affine, title="Mask 3D"):
    """
    mask: array 3D numpy (bool: true/false or 0/1)
    affine: 4x4 numpy array
    """
    
    vol = mask.astype(np.float32)                   # boolean/uint8 to float32 (VTK/PyVista require numeric values)
    print(np.unique(vol, return_counts=True))
    grid = pv.wrap(vol)
    mesh = grid.contour(isosurfaces=[0.5])
    mesh = mesh.transform(affine)

    pl = pv.Plotter()
    pl.add_mesh(mesh, color="red", opacity=1)
    pl.add_axes()
    pl.show(title=title)  

def save_skel_in_vtp(skel, output_vtp_path, affine=None):

    print("3. Saving skeleton to .vtp")

    pts = skel.vertices  # (N, 3)
    edges = skel.edges   # (M, 2)

    # Affine Trasf
    if affine is not None:
        homo = np.hstack([pts, np.ones((pts.shape[0], 1))])   # (N, 4)
        pts = (homo @ affine.T)[:, :3]                        # transformed 3D coords

    # vtp lines
    lines = []
    for u, v in edges:
        lines.extend([2, u, v])

    poly = pv.PolyData(pts)
    poly.lines = np.array(lines)

    poly.save(output_vtp_path)
    print(f"-- Centerline saved to: {output_vtp_path}")
    

# Use it if you want a Graph with coordinates on the nodes. Not used in the code
def graph_with_coord(skel):
    # If you want to convert the skel in a graph with coordinates on the nodes, instead of using: G = skel.get_graph() (that has indexes on the nodes: (1, 2, 3, 4 ...))
    
    G = nx.Graph()

    # Coordinates on nodes:
    for coord in skel.vertices:
        coord_tuple = tuple(coord)
        G.add_node(coord_tuple)

    # Start-End coordinates for Edges
    for u, v in skel.edges:
        coord_u = tuple(skel.vertices[u])
        coord_v = tuple(skel.vertices[v])
        G.add_edge(coord_u, coord_v)

    return G

# Save mask (mesh) as vtp. I used it for debugging, otherwise never
def save_mask_as_vtp(mask, affine, out_path="segmentation.vtp"):
    vol = mask.astype(np.float32)
    grid = pv.wrap(vol)
    mesh = grid.contour(isosurfaces=[0.5])
    mesh = mesh.transform(affine)
    mesh.save(out_path)
    print("Saved", out_path)

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
    
    args = parser.parse_args()
    debug = False
    
    # Load Mask
    mask, affine = load_mask(args.input, debug)
    # save_mask_as_vtp(mask, affine, "my_segmentation.vtp")
    
    # Compute Skeleton
    skel = compute_skeleton(mask, debug)
    
    # Save Skeleton in vtp
    save_skel_in_vtp(skel, args.output, affine=affine)
    
    # Compute Graph
    G = skel.get_graph()
    print(f"Graph built. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()} with Graph")
    
    # # oppure:
    # # skel = sk.skeletonize.by_teasar(fixed, ...)

if __name__ == "__main__":
    main()