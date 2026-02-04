# -*- coding: utf-8 -*-
"""
batch_skeleton_to_vtp.py

Batch conversion:
NIfTI vessel masks (.nii / .nii.gz) -> skeletonize -> graph (optional prune) -> VTP (segment-format)

Usage example (Windows):
python .\extraction\batch_skeleton_to_vtp.py --in_dir "C:\...\masks" --out_dir "C:\...\vtp" --connectivity 26 --overwrite --prune
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import vtk
import networkx as nx
from skimage.morphology import skeletonize


def build_graph(skeleton):
    """Build a graph from a skeleton with 26-connectivity (original implementation)."""
    G = nx.Graph()
    shape = skeleton.shape
    fibers = np.argwhere(skeleton)
    for v in fibers:
        coord = tuple(v)
        G.add_node(coord)
        x, y, z = coord
        for i in range(max(0, x-1), min(shape[0], x+2)):
            for j in range(max(0, y-1), min(shape[1], y+2)):
                for k in range(max(0, z-1), min(shape[2], z+2)):
                    if (i, j, k) != (x, y, z) and skeleton[i, j, k]:
                        G.add_edge(coord, (i, j, k), weight=np.linalg.norm(np.array(coord) - np.array((i, j, k))))
    return G


def prune_graph(G):
    """
    Prune G by detecting all simple cycles of length 3 (triangles) and removing the
    heaviest edge in each triangle (based on 'weight').
    Modifies G in-place and returns it.
    """
    # Find all simple cycles in the graph
    loops = nx.cycle_basis(G)
    for cycle in loops:
        if len(cycle) != 3:
            continue

        # Identify the heaviest edge in the triangle
        max_edge = None
        max_weight = float('-inf')
        for i in range(3):
            u = cycle[i]
            v = cycle[(i + 1) % 3]

            # safely get weight from either direction
            if G.has_edge(u, v):
                w = G[u][v].get('weight', 0)
            elif G.has_edge(v, u):
                w = G[v][u].get('weight', 0)
            else:
                # no such edge—skip
                continue

            if w > max_weight:
                max_weight = w
                max_edge = (u, v)

        # Remove the heaviest edge if it still exists
        if max_edge:
            u, v = max_edge
            if G.has_edge(u, v):
                G.remove_edge(u, v)
            elif G.has_edge(v, u):
                G.remove_edge(v, u)

    return G


def graph_to_vtp_segment_format(G: nx.Graph, affine: np.ndarray) -> vtk.vtkPolyData:
    """
    Convert a voxel-space graph to VTP (segment-format):
    - one VTK point per graph node (world coords via affine)
    - one VTK line cell (2 points) per graph edge
    """
    nodes = list(G.nodes())
    if len(nodes) == 0:
        raise ValueError("Graph has no nodes (empty skeleton?).")

    # voxel -> world using affine
    coords = np.asarray(nodes, dtype=np.float64)  # (N,3) in voxel indices (i,j,k)
    coords_h = np.c_[coords, np.ones((coords.shape[0], 1), dtype=np.float64)]
    xyz = (affine @ coords_h.T).T[:, :3]  # (N,3) world

    # map node -> point id
    node_to_pid = {node: idx for idx, node in enumerate(nodes)}

    # VTK points
    pts = vtk.vtkPoints()
    pts.SetNumberOfPoints(xyz.shape[0])
    for idx, p in enumerate(xyz):
        pts.SetPoint(idx, float(p[0]), float(p[1]), float(p[2]))

    # VTK lines (segment-format)
    lines = vtk.vtkCellArray()
    for u, v in G.edges():
        lines.InsertNextCell(2)
        lines.InsertCellPoint(int(node_to_pid[u]))
        lines.InsertCellPoint(int(node_to_pid[v]))

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(lines)
    return poly


def write_vtp(pd: vtk.vtkPolyData, out_path: str) -> None:
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(out_path)
    w.SetInputData(pd)
    w.SetDataModeToAppended()
    if hasattr(w, "SetCompressorTypeToZLib"):
        w.SetCompressorTypeToZLib()
    w.Write()


def process_one_mask(mask_path: Path, out_path: Path, *, do_prune: bool) -> None:
    img = nib.load(str(mask_path))
    mask = img.get_fdata() > 0
    affine = img.affine

    # skeletonize (nD)
    skel = skeletonize(mask).astype(bool)

    if not np.any(skel):
        raise ValueError("Skeleton is empty (mask too small / skeletonize removed everything).")

    # graph (original build_graph)
    G = build_graph(skel)

    if do_prune:
        G = prune_graph(G)

    if G.number_of_edges() == 0:
        raise ValueError("Graph has no edges (skeleton too sparse?).")

    poly = graph_to_vtp_segment_format(G, affine)
    write_vtp(poly, str(out_path))


def iter_input_files(in_dir: Path, recursive: bool) -> list[Path]:
    exts = (".nii", ".nii.gz")
    if recursive:
        files = [p for p in in_dir.rglob("*") if p.is_file() and p.name.lower().endswith(exts)]
    else:
        files = [p for p in in_dir.iterdir() if p.is_file() and p.name.lower().endswith(exts)]
    return sorted(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Input folder with .nii/.nii.gz masks")
    ap.add_argument("--out_dir", required=True, help="Output folder for .vtp files")
    ap.add_argument("--connectivity", type=int, default=26, help="Kept for CLI compatibility (build_graph is 26-neighborhood as in original code)")
    ap.add_argument("--recursive", action="store_true", help="Search masks recursively in subfolders")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .vtp outputs")
    ap.add_argument("--prune", action="store_true", help="Apply original prune_graph (triangle pruning) before exporting VTP")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {in_dir}")

    files = iter_input_files(in_dir, recursive=args.recursive)
    if not files:
        print(f"No .nii/.nii.gz found in: {in_dir}")
        return

    print(f"Found {len(files)} mask(s).")
    print(f"Pruning: {'ON' if args.prune else 'OFF'}")
    print(f"Recursive: {'ON' if args.recursive else 'OFF'}")

    n_ok, n_skip, n_fail = 0, 0, 0

    for mask_path in files:
        # output name: keep stem but handle .nii.gz
        name = mask_path.name
        stem = name[:-7] if name.lower().endswith(".nii.gz") else mask_path.stem
        out_path = out_dir / f"{stem}.vtp"

        if out_path.exists() and not args.overwrite:
            n_skip += 1
            continue

        try:
            process_one_mask(mask_path, out_path, do_prune=args.prune)
            n_ok += 1
            print(f"[OK]   {mask_path.name} -> {out_path.name}")
        except Exception as e:
            n_fail += 1
            print(f"[FAIL] {mask_path.name}: {e}")

    print("\nDone.")
    print(f"  OK   : {n_ok}")
    print(f"  SKIP : {n_skip}")
    print(f"  FAIL : {n_fail}")


if __name__ == "__main__":
    main()
