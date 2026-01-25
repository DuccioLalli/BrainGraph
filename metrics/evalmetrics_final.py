#!/usr/bin/env python3

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Set, Optional
import numpy as np
import vtk
from pathlib import Path
import nibabel as nib


# --------------------------
# Input / Output utilities
# --------------------------

def read_vtp(path: str) -> vtk.vtkPolyData:
    """
    Reads a VTP file and returns the PolyData object
    """
    r = vtk.vtkXMLPolyDataReader()
    r.SetFileName(path)
    r.Update()
    poly = r.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise ValueError(f"Empty or unread VTP file: {path}")
    return poly


def write_vtp(poly: vtk.vtkPolyData, path: str) -> None:
    """
    Saves in VTP format
    """
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(path)
    w.SetInputData(poly)
    w.SetDataModeToAppended()
    if hasattr(w, "SetCompressorTypeToZLib"):
        w.SetCompressorTypeToZLib()
    w.Write()



# --------------------------
# Debugging Functions (debug)
# --------------------------

import os
from collections import Counter
from vtk.util.numpy_support import vtk_to_numpy
import math


def informations(poly: vtk.vtkPolyData, path: str) -> None:
    lines = poly.GetLines()
    offs = vtk_to_numpy(lines.GetOffsetsArray())
    conn = vtk_to_numpy(lines.GetConnectivityArray())
    lengths = np.diff(offs)

    c = Counter(lengths.tolist())
    if path is not None:
        print("\n--- File:", os.path.basename(path))
    else:
        print("--- No path provided")
    print("Total points:", poly.GetNumberOfPoints())
    print("Line cells:", len(lengths))
    print("Segments (2 pt):", c.get(2,0))
    print("Polylines (>2):", sum(v for k,v in c.items() if k>2))
    print("Top lengths:", c.most_common(10))
    
    L = 0.0
    for i in range(len(offs) - 1):
        a, b = int(offs[i]), int(offs[i + 1])
        ids = conn[a:b]
        if len(ids) < 2:
            continue
        for j in range(len(ids) - 1):
            p0 = poly.GetPoint(int(ids[j]))
            p1 = poly.GetPoint(int(ids[j + 1]))
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            dz = p1[2] - p0[2]
            L += math.sqrt(dx*dx + dy*dy + dz*dz)

    print("Total trace length:", L)
    
    pts = np.array([poly.GetPoint(i) for i in range(poly.GetNumberOfPoints())])
    unique = np.unique(np.round(pts, 6), axis=0)
    print("Unique coords:", len(unique), " / Total points:", len(pts))
    

# --------------------------
# Polydata plot: Nodes colored by degree (debug)
# --------------------------

def plot_polydata(poly_in, tube_radius=0.05, end_r=0.3, mid_r=0.2, br_r=0.5):
    # === EXTRACT LINES DATA ===
    lines = poly_in.GetLines()
    offs = vtk_to_numpy(lines.GetOffsetsArray())
    conn = vtk_to_numpy(lines.GetConnectivityArray())

    # === classificazione nodi coerente con polilinee: grado topologico ===
    npts = poly_in.GetNumberOfPoints()
    nbrs = [set() for _ in range(npts)]

    for k in range(len(offs) - 1):
        a, b = int(offs[k]), int(offs[k+1])
        ids = conn[a:b]
        if len(ids) < 2:
            continue
        for j in range(len(ids) - 1):
            u = int(ids[j])
            v = int(ids[j + 1])
            if u == v:
                continue
            nbrs[u].add(v)
            nbrs[v].add(u)

    deg = np.array([len(s) for s in nbrs], dtype=np.int32)
    end_ids = np.where(deg == 1)[0]
    mid_ids = np.where(deg == 2)[0]
    br_ids  = np.where(deg >= 3)[0]
    print("Endpoints:", len(end_ids), "Mid(deg=2):", len(mid_ids), "Branch(deg>=3):", len(br_ids))

    # === BUILD NODE POLYDATA (3 gruppi) ===
    def make_points_poly(ids):
        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(len(ids))
        for i, pid in enumerate(ids):
            pts.SetPoint(i, poly_in.GetPoint(int(pid)))
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        vg = vtk.vtkVertexGlyphFilter()
        vg.SetInputData(poly)
        vg.Update()
        return vg.GetOutput()

    end_pd = make_points_poly(end_ids)
    mid_pd = make_points_poly(mid_ids)
    br_pd  = make_points_poly(br_ids)

    # === GLYPHS: spheres on nodes ===
    sphere = vtk.vtkSphereSource()
    sphere.SetThetaResolution(16)
    sphere.SetPhiResolution(16)

    def make_node_actor(poly, radius, color_rgb):
        g = vtk.vtkGlyph3D()
        g.SetSourceConnection(sphere.GetOutputPort())
        g.SetInputData(poly)
        g.SetScaleModeToDataScalingOff()
        g.SetScaleFactor(radius)
        g.Update()
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(g.GetOutputPort())
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(*color_rgb)
        return a

    end_actor = make_node_actor(end_pd, radius=end_r, color_rgb=(1.0, 0.2, 0.2))  # rosso
    mid_actor = make_node_actor(mid_pd, radius=mid_r, color_rgb=(0.2, 1.0, 0.2))  # verde
    br_actor  = make_node_actor(br_pd,  radius=br_r,  color_rgb=(1.0, 0.9, 0.2))  # giallo

    # === EDGES: tubes, color blu ===
    tube = vtk.vtkTubeFilter()
    tube.SetInputData(poly_in)
    tube.SetRadius(tube_radius)
    tube.SetNumberOfSides(12)
    tube.CappingOn()
    tube.Update()

    edge_mapper = vtk.vtkPolyDataMapper()
    edge_mapper.SetInputConnection(tube.GetOutputPort())

    edge_actor = vtk.vtkActor()
    edge_actor.SetMapper(edge_mapper)
    edge_actor.GetProperty().SetColor(0.2, 0.6, 1.0)  # blu
    edge_actor.GetProperty().SetOpacity(1.0)

    # === RENDER ===
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.08, 0.08, 0.10)

    ren.AddActor(edge_actor)
    ren.AddActor(end_actor)
    ren.AddActor(mid_actor)
    ren.AddActor(br_actor)

    win = vtk.vtkRenderWindow()
    win.AddRenderer(ren)
    win.SetSize(1100, 850)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(win)

    ren.ResetCamera()
    win.Render()
    iren.Start()
    

# --------------------------
# Fusion of non unique Ids (pre-processing of input formats)
# --------------------------

def fusion_nonUniqueIds(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
    CLEAN_TOL = 1e-6
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(poly)
    cleaner.PointMergingOn()
    cleaner.SetToleranceIsAbsolute(True)
    cleaner.SetAbsoluteTolerance(CLEAN_TOL)
    cleaner.Update()
    poly = cleaner.GetOutput()
    return poly

# --------------------------
# Resampling (pre-processing of input formats)
# --------------------------

@dataclass
class SampledLines:
    """
    Rappresentazione "resamplata":
    - points: lista di array (Ni,3) per ogni linea
    - total_length: somma lunghezze originali (o resamplate)
    """
    lines: List[np.ndarray]
    total_length: float

def resample_polydata_lines(poly: vtk.vtkPolyData, step: float) -> SampledLines:
    """
    - Estrae tutte le linee/polilinee da un vtkPolyData
    - Calcola la lunghezza totale originale
    - Resampla ogni linea a passo fisso `step`
    Ritorna: SampledLines(lines=[(Ni,3)...], total_length=float)
    """

    def iter_polylines_local() -> Iterable[np.ndarray]:
        lines = poly.GetLines()
        if lines is None:
            return
        lines.InitTraversal()
        id_list = vtk.vtkIdList()
        pts = poly.GetPoints()

        while lines.GetNextCell(id_list):
            n = id_list.GetNumberOfIds()
            if n < 2:
                continue
            arr = np.zeros((n, 3), dtype=np.float64)
            for i in range(n):
                arr[i] = pts.GetPoint(id_list.GetId(i))
            yield arr

    def resample_polyline_local(arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] < 2:
            return arr.copy()

        seg = arr[1:] - arr[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = cum[-1]

        if total <= 1e-12:
            return arr[:1].copy()

        s_vals = np.arange(0.0, total, step, dtype=np.float64)
        if s_vals.size == 0 or s_vals[-1] < total:
            s_vals = np.concatenate([s_vals, [total]])

        out = np.zeros((s_vals.size, 3), dtype=np.float64)

        j = 0
        for i, s in enumerate(s_vals):
            while j < len(cum) - 2 and cum[j + 1] < s:
                j += 1
            s0, s1 = cum[j], cum[j + 1]
            if s1 <= s0 + 1e-12:
                out[i] = arr[j]
            else:
                t = (s - s0) / (s1 - s0)
                out[i] = (1 - t) * arr[j] + t * arr[j + 1]
        return out

    out_lines: List[np.ndarray] = []
    total_len = 0.0

    for arr in iter_polylines_local():
        # lunghezza originale (controllo incluso)
        if arr.shape[0] >= 2:
            total_len += float(np.sum(np.linalg.norm(arr[1:] - arr[:-1], axis=1)))

        rs = resample_polyline_local(arr)
        if rs.shape[0] >= 2:
            out_lines.append(rs)

    return SampledLines(lines=out_lines, total_length=total_len)


# --------------------------
# Format conversions (SampledLines <-> vtkPolyData) Used in two different points
# --------------------------
def sampled_to_polydata(sampled: SampledLines, *, as_segments: bool = False) -> vtk.vtkPolyData:
    """
    Converte SampledLines in vtkPolyData.
    - as_segments=False: una vtkPolyLine per path (più leggero)
    - as_segments=True : segment-format, una cella (2 pt) per ogni edge (utile per metriche per-edge)
    """
    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    pid_offset = 0

    for arr in sampled.lines:
        n = int(arr.shape[0])
        if n < 2:
            continue

        # inserisci tutti i punti del path
        for i in range(n):
            pts.InsertNextPoint(float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2]))

        if as_segments:
            # segment-format: (n-1) celle da 2 punti
            for i in range(n - 1):
                lines.InsertNextCell(2)
                lines.InsertCellPoint(pid_offset + i)
                lines.InsertCellPoint(pid_offset + i + 1)
        else:
            # polyline: 1 cella con n punti
            pl = vtk.vtkPolyLine()
            pl.GetPointIds().SetNumberOfIds(n)
            for i in range(n):
                pl.GetPointIds().SetId(i, pid_offset + i)
            lines.InsertNextCell(pl)

        pid_offset += n

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(lines)
    return poly



# --------------------------
# METRICS
# --------------------------

# --------------------------
# Topology [6]: degrees/endpoints/junctions (branches)
# --------------------------

@dataclass
class TopologyStats:
    n_nodes: int
    n_edges: int
    n_endpoints: int
    n_junctions: int
    degree_hist: Dict[int, int]


def topology_stats(poly: vtk.vtkPolyData) -> TopologyStats:

    adj: Dict[int, Set[int]] = {}
    lines = poly.GetLines()
    if lines is None:
        return TopologyStats(0, 0, 0, 0, {})

    lines.InitTraversal()
    id_list = vtk.vtkIdList()

    while lines.GetNextCell(id_list):
        n = id_list.GetNumberOfIds()
        if n < 2:
            continue
        for i in range(n - 1):
            a = int(id_list.GetId(i))
            b = int(id_list.GetId(i + 1))
            if a == b:
                continue
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

    degrees = {k: len(v) for k, v in adj.items()}
    n_nodes = len(degrees)
    n_edges = sum(degrees.values()) // 2
    n_end = sum(1 for d in degrees.values() if d == 1)
    n_junc = sum(1 for d in degrees.values() if d >= 3)

    hist: Dict[int, int] = {}
    for d in degrees.values():
        hist[d] = hist.get(d, 0) + 1

    return TopologyStats(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_endpoints=n_end,
        n_junctions=n_junc,
        degree_hist=dict(sorted(hist.items())),
    )
 


# --------------------------
# Segmentation Consistency [5]
# --------------------------

# metrica mia, controllata
def node_inside_outside_ratio(poly_seg: vtk.vtkPolyData, nii_path: str) -> dict:
    """
    Versione su vtkPolyData (segment-format + clean consigliati):
    - considera i NODI (Points del polydata)
    - inside se il voxel nearest (round) in mask è True
    """
    img = nib.load(nii_path)
    mask = img.get_fdata() > 0
    affine = img.affine

    pts_vtk = poly_seg.GetPoints()
    if pts_vtk is None or poly_seg.GetNumberOfPoints() == 0:
        return {
            "inside_nodes": 0,
            "outside_nodes": 0,
            "total_nodes": 0,
            "inside_ratio_nodes": float("nan"),
            "outside_ratio_nodes": float("nan"),
        }

    pts = vtk_to_numpy(pts_vtk.GetData()).astype(np.float64, copy=False)
    ijk = world_to_voxel(pts, affine)
    ijk_round = np.rint(ijk).astype(np.int64)

    sx, sy, sz = mask.shape
    x = np.clip(ijk_round[:, 0], 0, sx - 1)
    y = np.clip(ijk_round[:, 1], 0, sy - 1)
    z = np.clip(ijk_round[:, 2], 0, sz - 1)

    vals = mask[x, y, z]
    inside = int(np.sum(vals))
    total = int(vals.size)

    return {
        "inside_nodes": inside,
        "outside_nodes": total - inside,
        "total_nodes": total,
        "inside_ratio_nodes": (inside / total) if total > 0 else float("nan"),
        "outside_ratio_nodes": ((total - inside) / total) if total > 0 else float("nan"),
    }
    

def world_to_voxel(points_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    points_xyz: (N,3) in world (mm)
    affine: nibabel affine voxel->world
    ritorna: (N,3) in voxel coords (float), ordine i,j,k
    """
    inv = np.linalg.inv(affine)
    N = points_xyz.shape[0]
    homog = np.c_[points_xyz, np.ones((N, 1), dtype=np.float64)]
    ijk = (inv @ homog.T).T[:, :3]
    return ijk


# def _segment_info(poly: vtk.vtkPolyData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Estrae segmenti come coppie di punti consecutivi dalle celle lineari.
#     Funziona sia se le celle sono tutte 2-pt (segment-format), sia se ci sono polilinee (>2).
#     Ritorna:
#       mid: (M,3) midpoints dei segmenti
#       seg_len: (M,) lunghezze in mm
#       dP: (M,3) vettori (p1 - p0) (opzionale per debug/estensioni)
#     """
#     pts_vtk = poly.GetPoints()
#     if pts_vtk is None or poly.GetNumberOfPoints() == 0:
#         return (np.zeros((0, 3), dtype=np.float64),
#                 np.zeros((0,), dtype=np.float64),
#                 np.zeros((0, 3), dtype=np.float64))

#     pts = vtk_to_numpy(pts_vtk.GetData()).astype(np.float64, copy=False)

#     lines = poly.GetLines()
#     if lines is None or lines.GetNumberOfCells() == 0:
#         return (np.zeros((0, 3), dtype=np.float64),
#                 np.zeros((0,), dtype=np.float64),
#                 np.zeros((0, 3), dtype=np.float64))

#     seg_a = []
#     seg_b = []

#     # Fast path: offsets/connectivity
#     try:
#         offs = vtk_to_numpy(lines.GetOffsetsArray())
#         conn = vtk_to_numpy(lines.GetConnectivityArray())

#         for k in range(len(offs) - 1):
#             a, b = int(offs[k]), int(offs[k + 1])
#             ids = conn[a:b]
#             if ids.size < 2:
#                 continue
#             # se è 2-pt line -> 1 segmento; se polyline -> segmenti consecutivi
#             for j in range(ids.size - 1):
#                 u = int(ids[j])
#                 v = int(ids[j + 1])
#                 if u == v:
#                     continue
#                 seg_a.append(u)
#                 seg_b.append(v)

#     # Fallback: traversal classico
#     except Exception:
#         lines.InitTraversal()
#         id_list = vtk.vtkIdList()
#         while lines.GetNextCell(id_list):
#             n = id_list.GetNumberOfIds()
#             if n < 2:
#                 continue
#             for i in range(n - 1):
#                 u = int(id_list.GetId(i))
#                 v = int(id_list.GetId(i + 1))
#                 if u == v:
#                     continue
#                 seg_a.append(u)
#                 seg_b.append(v)

#     if not seg_a:
#         return (np.zeros((0, 3), dtype=np.float64),
#                 np.zeros((0,), dtype=np.float64),
#                 np.zeros((0, 3), dtype=np.float64))

#     ia = np.asarray(seg_a, dtype=np.int64)
#     ib = np.asarray(seg_b, dtype=np.int64)

#     p0 = pts[ia]
#     p1 = pts[ib]
#     dP = p1 - p0
#     seg_len = np.linalg.norm(dP, axis=1)
#     mid = (p0 + p1) * 0.5

#     return mid, seg_len, dP



def _segment_info(poly: vtk.vtkPolyData):
    """
    Extract per-segment midpoints, lengths, and 'direction vectors' (dp).
    Returns: mid (M,3), seg_len (M,), dP (M,3).
    Input: 'segment-format' PolyData: each line cell must contain exactly 2 point IDs (one segment per cell).
    So it is compatible with the gt_poly_seg / pred_poly_seg format, obtained from 'sampled_to_polydata(..., as_segments=True)' which is the conversion function used
    """

    if poly is None or poly.GetNumberOfPoints() == 0 or poly.GetNumberOfCells() == 0:
        return (np.zeros((0,3)), np.zeros((0,)), np.zeros((0,3)))

    pts_vtk = poly.GetPoints()
    lines = poly.GetLines()
    if pts_vtk is None or lines is None or lines.GetNumberOfCells() == 0:
        return (np.zeros((0,3)), np.zeros((0,)), np.zeros((0,3)))

    pts = vtk_to_numpy(pts_vtk.GetData()).astype(np.float64, copy=False)
    offs = vtk_to_numpy(lines.GetOffsetsArray())
    conn = vtk_to_numpy(lines.GetConnectivityArray())

    ia = conn[offs[:-1]]
    ib = conn[offs[:-1] + 1]

    p0 = pts[ia]
    p1 = pts[ib]
    dP = p1 - p0
    seg_len = np.linalg.norm(dP, axis=1)
    mid = 0.5 * (p0 + p1)
    return mid, seg_len, dP



from scipy.ndimage import distance_transform_edt, map_coordinates
def centerline_vs_mask_metrics(poly_seg: vtk.vtkPolyData, nii_path: str, r_mm: float = 1.0) -> dict:
    """
    Versione su vtkPolyData (segment-format consigliato):
    - campiona distanza dalla maschera sui midpoints dei segmenti
    - calcola inside/outside LENGTH (mm) usando una soglia 'inside' sul d_out
    - NON calcola outside_runs / max_outside_run (dipendono dall'ordine)
    """
    img = nib.load(nii_path)
    mask = img.get_fdata() > 0
    affine = img.affine
    spacing = img.header.get_zooms()[:3]

    dt_to_vessel = distance_transform_edt(~mask, sampling=spacing)
    dt_inside = distance_transform_edt(mask, sampling=spacing)

    mid, seg_len, _ = _segment_info(poly_seg)

    total_len = float(np.sum(seg_len)) if seg_len.size else 0.0
    if mid.shape[0] == 0:
        return {
            "total_length": total_len,
            "inside_length": 0.0,
            "outside_length": total_len,
            "inside_ratio_len": float("nan") if total_len <= 1e-12 else 0.0,
        }

    ijk = world_to_voxel(mid, affine)
    x, y, z = ijk[:, 0], ijk[:, 1], ijk[:, 2]

    d_out = map_coordinates(dt_to_vessel, [x, y, z], order=1, mode="nearest")
    d_in  = map_coordinates(dt_inside,    [x, y, z], order=1, mode="nearest")

    # Inside = molto vicino / dentro (stessa definizione che avevi)
    inside_tol = 0.9 * float(min(spacing))
    inside_flag = d_out < inside_tol

    inside_len = float(np.sum(seg_len[inside_flag]))
    outside_len = float(total_len - inside_len)
    inside_ratio = inside_len / total_len if total_len > 1e-12 else float("nan")

    out = {
        "total_length": total_len,
        "inside_length": inside_len,
        "outside_length": outside_len,
        "inside_ratio_len": inside_ratio,
    }

    # Stats distanza (mm) dalla maschera
    out.update({
        "dt_mean(mm)": float(np.mean(d_out)),
        "dt_median(mm)": float(np.median(d_out)),
        "dt_p95(mm)": float(np.percentile(d_out, 95)),
        "dt_max(mm)": float(np.max(d_out)),
        "within_r_ratio": float(np.mean(d_out <= float(r_mm))),
    })

    # Stats "quanto dentro" (opzionali ma li manteniamo)
    out.update({
        "dt_inside_median": float(np.median(d_in)),
        "dt_inside_p10": float(np.percentile(d_in, 10)),
    })

    return out



# --------------------------
# # Distance helpers (point-to-centerline closest-point):
# used by geometry metrics (ASSD/HD95), coverage metrics (precision/recall via segment midpoints),
# and the coarse alignment sanity check.
# --------------------------

def build_cell_locator(poly: vtk.vtkPolyData, use_static: bool = True):
    """
    Builds a VTK cell locator (spatial index) for fast closest-point queries on the 'line cells' of `poly`.
    Input:
      - poly: vtkPolyData containing line cells to be queried.
      - use_static: if True and available, uses vtkStaticCellLocator (faster for static data); otherwise vtkCellLocator.
    Output:
      - loc: 'locator object' to be used with FindClosestPoint().
    """
    if use_static and hasattr(vtk, "vtkStaticCellLocator"):
        loc = vtk.vtkStaticCellLocator()
    else:
        loc = vtk.vtkCellLocator()
        
    loc.SetDataSet(poly)
    loc.BuildLocator()
    return loc


def polydata_points(poly: vtk.vtkPolyData) -> np.ndarray:
    """Return Nx3 float64 points from vtkPolyData."""
    pts = poly.GetPoints()
    if pts is None or poly.GetNumberOfPoints() == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return vtk_to_numpy(pts.GetData()).astype(np.float64, copy=False)


def distances_points_to_locator(points: np.ndarray, locator) -> np.ndarray:
    """For each point (Nx3), distance to closest point on locator dataset."""
    n = points.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)

    out = np.empty((n,), dtype=np.float64)

    closest = [0.0, 0.0, 0.0]
    cell_id = vtk.reference(0)
    sub_id  = vtk.reference(0)
    dist2   = vtk.reference(0.0)

    for i in range(n):
        p = points[i]
        locator.FindClosestPoint((float(p[0]), float(p[1]), float(p[2])),
                                 closest, cell_id, sub_id, dist2)
        out[i] = float(np.sqrt(dist2.get()))
    return out


def distances_polydata_to_polydata(src_pd: vtk.vtkPolyData, ref_poly: vtk.vtkPolyData, use_static: bool = True) -> np.ndarray:
    """
    ref_poly: polyline of reference
    
    One-way distances:
      For each point in src_pd, compute distance to closest point on ref_poly line set.
    """
    if ref_poly is None or ref_poly.GetNumberOfCells() == 0:
        return np.array([], dtype=np.float64)

    src_pts = polydata_points(src_pd)
    if src_pts.size == 0:
        return np.array([], dtype=np.float64)

    loc = build_cell_locator(ref_poly, use_static=use_static)
    return distances_points_to_locator(src_pts, loc)


def percentile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))



# --------------------------
# Coverage metrics (Precision/Recall/F1)
# # --------------------------

def covered_length_of_polydata(query_poly: vtk.vtkPolyData, ref_poly: vtk.vtkPolyData, tau: float, *, use_static: bool = True) -> float:
    """
    Returns the total length (mm) of segments in `query_poly` that are within `tau` of `ref_poly`.
    For each 2-point line cell (segment) in `query_poly`, we test the segment midpoint against the
    closest point on the centerline of `ref_poly` (via a cell locator);
    if distance <= tau, the whole segment length is counted as "covered".
    """
    mid, seg_len, _ = _segment_info(query_poly)
    if mid.shape[0] == 0:
        return 0.0
    if ref_poly is None or ref_poly.GetNumberOfCells() == 0:
        return 0.0

    loc = build_cell_locator(ref_poly, use_static=use_static)
    d = distances_points_to_locator(mid, loc)
    return float(np.sum(seg_len[d <= float(tau)]))

def coverage_metrics(gt_poly_seg: vtk.vtkPolyData, pred_poly_seg: vtk.vtkPolyData, tau: float, use_static: bool = True,) -> Tuple[float, float, float, float, float, float, float]:
    """
    Coverage metrics between two *segment-format* centerlines.
    Inputs:
      - gt_poly_seg, pred_poly_seg: vtkPolyData 'segment-format'
      - tau (mm): distance tolerance used to mark a segment as "covered" based on its midpoint distance to the reference polyline set.
      - use_static: use vtkStaticCellLocator when available.
      
    Outputs:
      metrics: (precision, recall, f1, gt_cov, pred_cov, gt_len, pred_len):
      where gt_cov/pred_cov are covered lengths (mm) and gt_len/pred_len are total lengths (mm).
    """
    _, seg_len_gt, _ = _segment_info(gt_poly_seg)
    gt_len = float(np.sum(seg_len_gt))

    _, seg_len_pr, _ = _segment_info(pred_poly_seg)
    pred_len = float(np.sum(seg_len_pr))

    if gt_len <= 1e-12 or pred_len <= 1e-12:
        return float("nan"), float("nan"), float("nan"), 0.0, 0.0, gt_len, pred_len

    # Recall: GT covered by Pred:
    # we iterate over GT segments (GT is the "source" being checked segment-by-segment),
    # and count a GT segment as covered if its midpoint lies within tau of the Pred centerline (reference) ('locator format').
    gt_cov = covered_length_of_polydata(gt_poly_seg, pred_poly_seg, tau, use_static=use_static)
    recall = gt_cov / gt_len

    # Precision: Pred supported by GT:
    # opposite as before
    pred_cov = covered_length_of_polydata(pred_poly_seg, gt_poly_seg, tau, use_static=use_static)
    precision = pred_cov / pred_len

    f1 = 0.0 if (precision + recall) <= 1e-12 else (2.0 * precision * recall / (precision + recall))

    return float(precision), float(recall), float(f1), float(gt_cov), float(pred_cov), float(gt_len), float(pred_len)


# --------------------------
# Quick alignment sanity check
# --------------------------

def coarse_alignment_check(
    gt_poly: vtk.vtkPolyData,
    pred_poly: vtk.vtkPolyData,
    n_samples: int = 2000,
    use_static: bool = True,
) -> float:
    """
    Campiona punti casuali dalla pred e misura distanza media alla GT.
    Se è enorme, probabilmente non sono nello stesso spazio.
    """
    if pred_poly is None or pred_poly.GetNumberOfPoints() == 0:
        return float("nan")
    if gt_poly is None or gt_poly.GetNumberOfCells() == 0:
        return float("nan")

    # prendo punti pred in numpy
    pred_pts = vtk_to_numpy(pred_poly.GetPoints().GetData()).astype(np.float64, copy=False)
    n = pred_pts.shape[0]
    k = min(int(n_samples), n)

    # campionamento senza replacement
    idx = np.random.choice(n, size=k, replace=False)
    pts = pred_pts[idx]

    # locator su GT
    loc = build_cell_locator(gt_poly, use_static=use_static)

    # distanze punto->GT
    d = distances_points_to_locator(pts, loc)
    return float(np.mean(d)) if d.size else float("nan")


# --------------------------
# Report printing
# --------------------------

def fmt(x, nd=4, unit=""):
    if x != x:  # NaN
        return "nan" + unit
    return f"{x:.{nd}f}{unit}"

def segment_format_basic_info(poly: vtk.vtkPolyData) -> dict:
    """
    Estrae info base in stile 'informations' (senza lunghezze/unique coords):
    - num_points
    - line_cells
    - segments (2 pt)
    - polylines (>2)
    Da usare su polydata in formato Lines (tipicamente segment-format 2-pt).
    """
    if poly is None:
        return {"num_points": 0, "line_cells": 0, "segments": 0, "polylines": 0}

    npts = int(poly.GetNumberOfPoints())
    lines = poly.GetLines()
    if lines is None or lines.GetNumberOfCells() == 0:
        return {"num_points": npts, "line_cells": 0, "segments": 0, "polylines": 0}

    try:
        offs = vtk_to_numpy(lines.GetOffsetsArray())
        lengths = np.diff(offs)
        line_cells = int(len(lengths))
        segments = int(np.sum(lengths == 2))
        polylines = int(np.sum(lengths > 2))
    except Exception:
        lines.InitTraversal()
        id_list = vtk.vtkIdList()
        lens = []
        while lines.GetNextCell(id_list):
            lens.append(int(id_list.GetNumberOfIds()))
        lengths = np.asarray(lens, dtype=np.int64)
        line_cells = int(lengths.size)
        segments = int(np.sum(lengths == 2))
        polylines = int(np.sum(lengths > 2))

    return {
        "num_points": npts,
        "line_cells": line_cells,
        "segments": segments,
        "polylines": polylines,
    }

def print_report_A(R: dict) -> None:
    print("\n" + "=" * 70)
    print(f" EVAL SUMMARY {R.get('case_tag','')}".center(70))
    print("=" * 70)

    gt_p = R.get("gt_path", None)
    pr_p = R.get("pred_path", None)
    mk_p = R.get("mask_path", None)

    if gt_p is not None and pr_p is not None:
        print(f"GT   : {Path(gt_p).name}")
        print(f"PRED : {Path(pr_p).name}")
    if mk_p:
        print(f"MASK : {Path(mk_p).name}")
    else:
        print("MASK : (none)")

    print("\n[0] CONFIG / SANITY")
    print(f"  step={R['step']}  tau={R['tau']}  r_mm={R.get('r_mm','n/a')}")
    if R.get("voxel_spacing") is not None:
        sp = R["voxel_spacing"]
        print(f"  voxel spacing (mm): ({sp[0]:.6f}, {sp[1]:.6f}, {sp[2]:.6f})")
    print(f"  Coarse mean distance (pred points -> GT lines): {fmt(R['coarse_mean'],4,' mm')}")

    print("\n[1] RANKING (key numbers)")
    print(f"  F1_tau  : {fmt(R['f1'],4)}")
    print(f"  ASSD    : {fmt(R['assd'],4,' mm')}")
    print(f"  HD95    : {fmt(R['hd95'],4,' mm')}")
    print(f"  LenΔ    : {fmt(R['pred_len']-R['gt_len'],4,' mm')} (Pred - GT)")

    print("\n[2] LENGTH / FRAGMENTATION")
    print(f"  GT total length   : {fmt(R['gt_len'],4,' mm')}")
    print(f"  Pred total length : {fmt(R['pred_len'],4,' mm')}")
    print(f"  Length difference (LenΔ) : {fmt(R['pred_len']-R['gt_len'],4,' mm')} (Pred - GT)")
    gi = R.get("gt_seg_info", {})
    pi = R.get("pred_seg_info", {})
    print(f"  GT resampled num_points   : {gi.get('num_points','n/a')}")
    print(f"  Pred resampled num_points : {pi.get('num_points','n/a')}")
    print(f"  GT line cells             : {gi.get('line_cells','n/a')}")
    print(f"  GT segments (2 pt)        : {gi.get('segments','n/a')}")
    print(f"  GT polylines (>2)         : {gi.get('polylines','n/a')}")
    print(f"  Pred line cells           : {pi.get('line_cells','n/a')}")
    print(f"  Pred segments (2 pt)      : {pi.get('segments','n/a')}")
    print(f"  Pred polylines (>2)       : {pi.get('polylines','n/a')}")

    print(f"\n[3] COVERAGE @ tau={R['tau']} mm")
    print(f"  Precision_tau : {fmt(R['precision'],4)}")
    print(f"  Recall_tau    : {fmt(R['recall'],4)}")
    print(f"  F1_tau        : {fmt(R['f1'],4)}   <-- (same as Ranking)")
    print(f"  GT_covered_length   : {fmt(R['gt_cov'],4,' mm')}")
    print(f"  Pred_covered_length : {fmt(R['pred_cov'],4,' mm')}")
    print(f"  GT_missing_length (approx): {fmt(R['gt_len']-R['gt_cov'],4,' mm')}")
    print(f"  Pred_extra_length (approx): {fmt(R['pred_len']-R['pred_cov'],4,' mm')}")

    print("\n[4] GEOMETRY (point-to-segment)")
    print(f"  pred -> GT : mean={fmt(R['pred2gt_mean'],4)}  median={fmt(R['pred2gt_median'],4)}  "
          f"p95={fmt(R['pred2gt_p95'],4)}  max={fmt(R['pred2gt_max'],4)}")
    print(f"  GT   ->pred: mean={fmt(R['gt2pred_mean'],4)}  median={fmt(R['gt2pred_median'],4)}  "
          f"p95={fmt(R['gt2pred_p95'],4)}  max={fmt(R['gt2pred_max'],4)}")
    print(f"  ASSD : {fmt(R['assd'],4,' mm')}  <-- (same as Ranking)")
    print(f"  HD95 : {fmt(R['hd95'],4,' mm')}  <-- (same as Ranking)")

    print("\n[5] SEGMENTATION CONSISTENCY (vs mask)")
    if R.get("seg_enabled", False):

        def print_seg_block(title: str, M: dict, N: dict):
            print(f"\n{title}")
            print(f"  total_length:       {fmt(M['total_length'],4)}")
            print(f"  inside_length:      {fmt(M['inside_length'],4)}")
            print(f"  outside_length:     {fmt(M['outside_length'],4)}")
            print(f"  inside_ratio_len:   {fmt(M['inside_ratio_len'],4)}")
            print(f"  dt_mean(mm):        {fmt(M['dt_mean(mm)'],4)}")
            print(f"  dt_median(mm):      {fmt(M['dt_median(mm)'],4)}")
            print(f"  dt_p95(mm):         {fmt(M['dt_p95(mm)'],4)}")
            print(f"  dt_max(mm):         {fmt(M['dt_max(mm)'],4)}")
            print(f"  within_r_ratio:     {fmt(M['within_r_ratio'],4)}")
            print(f"  dt_inside_median:   {fmt(M['dt_inside_median'],4)}")
            print(f"  dt_inside_p10:      {fmt(M['dt_inside_p10'],4)}")

            print(f"\n{title.split('vs')[0].strip()} node containment:")
            print(f"  inside_nodes: {N['inside_nodes']} / {N['total_nodes']}  ({fmt(N['inside_ratio_nodes'],4)})")
            print(f"  outside_nodes:{N['outside_nodes']} / {N['total_nodes']}  ({fmt(N['outside_ratio_nodes'],4)})")

        print_seg_block("GT vs Segmentation", R["seg"]["GT"], R["nodes"]["GT"])
        print_seg_block("Pred vs Segmentation", R["seg"]["Pred"], R["nodes"]["Pred"])

    else:
        print("  [skipped] no mask provided")

    print("\n[6] TOPOLOGY")
    tp = R["topo_pred"]
    tg = R["topo_gt"]
    print(f"  Pred: nodes={tp['nodes']}")
    print(f"        endpoints(deg=1)={tp['endpoints']}    branch(deg>=3)={tp['junctions']}")
    print(f"        degree_hist={tp['degree_hist']}")

    print(f"  GT  : nodes={tg['nodes']}")
    print(f"        endpoints(deg=1)={tg['endpoints']}    branch(deg>=3)={tg['junctions']}")
    print(f"        degree_hist={tg['degree_hist']}")
    if tg["junctions"] == 0:
        print("        NOTE: GT branch(deg>=3)=0 likely means the GT has a format with no shared branches")

    if "smoothness" in R:
        print("\n[7] SMOOTHNESS (turning-angle)")

        def _print_sm_block(title: str, M: dict):
            print(f"{title}")
            print(f"  total_turn_rad: {fmt(M.get('total_turn_rad', float('nan')),4)}")
            print(f"  turn_per_mm:   {fmt(M.get('turn_per_mm', float('nan')),6)}")
            print(f"  rms_turn_deg:  {fmt(M.get('rms_turn_deg', float('nan')),4)}")
            print(f"  std_turn_deg:  {fmt(M.get('std_turn_deg', float('nan')),4)}")
            print(f"  p95_turn_deg:  {fmt(M.get('p95_turn_deg', float('nan')),4)}")
            print(f"  gt_p95_exceedance:    {fmt(M.get('gt_p95_exceedance', float('nan')),4)}\n")

        _print_sm_block("GT", R["smoothness"].get("GT", {}))
        _print_sm_block("Pred", R["smoothness"].get("Pred", {}))

    print("\n" + "=" * 70)




 
# magia tentativo
## 
def build_adjacency_from_polydata(poly: vtk.vtkPolyData) -> Dict[int, Set[int]]:
    """
    Costruisce adiacenza (undirected) da un vtkPolyData con Lines.
    Funziona sia con segment-format (2-pt) che con polilinee (>2).
    """
    adj: Dict[int, Set[int]] = {}
    lines = poly.GetLines()
    if lines is None or lines.GetNumberOfCells() == 0:
        return adj

    # Fast path offsets/connectivity
    try:
        offs = vtk_to_numpy(lines.GetOffsetsArray())
        conn = vtk_to_numpy(lines.GetConnectivityArray())
        for k in range(len(offs) - 1):
            a, b = int(offs[k]), int(offs[k + 1])
            ids = conn[a:b]
            if ids.size < 2:
                continue
            for j in range(ids.size - 1):
                u = int(ids[j]); v = int(ids[j + 1])
                if u == v:
                    continue
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
    except Exception:
        # Fallback traversal
        lines.InitTraversal()
        id_list = vtk.vtkIdList()
        while lines.GetNextCell(id_list):
            n = id_list.GetNumberOfIds()
            if n < 2:
                continue
            for i in range(n - 1):
                u = int(id_list.GetId(i))
                v = int(id_list.GetId(i + 1))
                if u == v:
                    continue
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)

    return adj


def extract_paths_between_critical_nodes(
    adj: Dict[int, Set[int]]
) -> Tuple[List[List[int]], Set[Tuple[int, int]]]:
    """
    Estrae cammini massimali tra nodi critici (degree != 2).
    Ritorna (paths, visited_edges) con edge undirected normalizzati.
    """
    deg = {u: len(vs) for u, vs in adj.items()}
    critical = {u for u, d in deg.items() if d != 2}

    visited: Set[Tuple[int, int]] = set()
    paths: List[List[int]] = []

    def ek(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u < v else (v, u)

    for u in critical:
        for v in adj.get(u, ()):
            if ek(u, v) in visited:
                continue

            path = [u, v]
            visited.add(ek(u, v))

            prev, cur = u, v
            # cammina finché sei su nodi degree==2
            while cur not in critical:
                nbrs = list(adj[cur])
                if len(nbrs) != 2:
                    break
                nxt = nbrs[0] if nbrs[1] == prev else nbrs[1]
                if ek(cur, nxt) in visited:
                    break
                path.append(nxt)
                visited.add(ek(cur, nxt))
                prev, cur = cur, nxt

            paths.append(path)

    return paths, visited


def extract_cycles_degree2(
    adj: Dict[int, Set[int]],
    visited_edges: Set[Tuple[int, int]],
    *,
    close_cycles: bool = True
) -> List[List[int]]:
    """
    Estrae cicli (componenti con tutti deg==2) rimasti non visitati.
    Se close_cycles=True, chiude il ciclo ripetendo il nodo iniziale alla fine.
    """
    deg = {u: len(vs) for u, vs in adj.items()}

    def ek(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u < v else (v, u)

    cycles: List[List[int]] = []

    for start in adj.keys():
        if deg.get(start, 0) != 2:
            continue

        nbrs = list(adj[start])
        if len(nbrs) != 2:
            continue

        # deve esistere almeno un edge non visitato da start, altrimenti è già coperto
        cand = None
        for nb in nbrs:
            if ek(start, nb) not in visited_edges:
                cand = nb
                break
        if cand is None:
            continue

        prev = start
        cur = cand
        cycle = [start, cur]
        visited_edges.add(ek(start, cur))

        # percorri fino a richiudere
        guard = 0
        while True:
            guard += 1
            if guard > 10_000_000:
                cycle = []
                break

            nbrs_cur = list(adj[cur])
            if len(nbrs_cur) != 2:
                cycle = []
                break

            nxt = nbrs_cur[0] if nbrs_cur[1] == prev else nbrs_cur[1]

            if nxt == start:
                visited_edges.add(ek(cur, nxt))
                if close_cycles:
                    cycle.append(start)  # chiusura
                break

            if ek(cur, nxt) in visited_edges:
                # già percorso: evita duplicati / cammini degeneri
                cycle = []
                break

            cycle.append(nxt)
            visited_edges.add(ek(cur, nxt))
            prev, cur = cur, nxt

        if cycle and len(cycle) >= (4 if close_cycles else 3):
            cycles.append(cycle)

    return cycles


def polyseg_to_sampledlines(poly_seg: vtk.vtkPolyData, *, close_cycles: bool = True) -> SampledLines:
    """
    Converte un polydata (anche segment-format) in SampledLines ricostruendo cammini ordinati.
    Utile per metriche di smoothness (serve ordine).
    """
    pts_vtk = poly_seg.GetPoints()
    if pts_vtk is None or poly_seg.GetNumberOfPoints() == 0:
        return SampledLines(lines=[], total_length=0.0)

    pts = vtk_to_numpy(pts_vtk.GetData()).astype(np.float64, copy=False)

    adj = build_adjacency_from_polydata(poly_seg)
    if not adj:
        return SampledLines(lines=[], total_length=0.0)

    paths, visited = extract_paths_between_critical_nodes(adj)
    cycles = extract_cycles_degree2(adj, visited, close_cycles=close_cycles)
    all_paths = paths + cycles

    out_lines: List[np.ndarray] = []
    total_len = 0.0

    for path in all_paths:
        if len(path) < 2:
            continue
        arr = pts[np.asarray(path, dtype=np.int64)]
        if arr.shape[0] >= 2:
            total_len += float(np.sum(np.linalg.norm(arr[1:] - arr[:-1], axis=1)))
        out_lines.append(arr)

    return SampledLines(lines=out_lines, total_length=total_len)


###########

def angular_tortuosity_metrics(sampled: SampledLines, eps: float = 1e-12, *, frac_threshold_deg: float | None = None) -> dict:
    """
    Turning-angle smoothness metrics computed along each ordered path.

    ------ Input
    sampled : SampledLines
        Ordered centerline paths. Each path is an (N,3) array of XYZ points.
    eps : Small constant to avoid division by zero and to filter degenerate segments.
    frac_threshold_deg : float | None
        Optional threshold (in degrees) used to compute `gt_p95_exceedance` as the fraction of angles above this value.
        If None, `gt_p95_exceedance` is returned as NaN.
    ------ Output
    dict
        - total_turn_rad: sum of all local turning angles (radians), aggregated over all paths.
        - turn_per_mm: total_turn_rad divided by total polyline length (radians/mm).
        - rms_turn_deg: root-mean-square of turning angles (degrees).
        - std_turn_deg: standard deviation of turning angles (degrees).
        - p95_turn_deg: 95th percentile of turning angles (degrees).
        - gt_p95_exceedance: fraction of angles above `frac_threshold_deg` (unitless).
    """
    
    total_len = 0.0
    thetas = []

    for arr in sampled.lines:
        if arr.shape[0] < 3:
            continue
        
        # compute the segments vectors
        v = arr[1:] - arr[:-1]
        seg_len = np.linalg.norm(v, axis=1)
        total_len += float(np.sum(seg_len))

        # tangenti unit (normalization) (we want just the direction)
        t = v / (seg_len[:, None] + eps)
        
        # angles between consecutive segments using dot product
        cos = np.sum(t[:-1] * t[1:], axis=1)
        cos = np.clip(cos, -1.0, 1.0)   # stability
        theta = np.arccos(cos)          # (n-2,)

        valid = (seg_len[:-1] > eps) & (seg_len[1:] > eps)
        theta = theta[valid]

        if theta.size:
            thetas.append(theta)

    if not thetas or total_len <= eps:
        return {
            "total_turn_rad": 0.0,
            "turn_per_mm": float("nan"),
            "rms_turn_deg": float("nan"),
            "std_turn_deg": float("nan"),
            "p95_turn_deg": float("nan"),
            "gt_p95_exceedance": float("nan"),
        }

    th = np.concatenate(thetas)     # rad
    th_deg = th * (180.0 / np.pi)   # deg
    
    total_turn = float(np.sum(th))
    
    rms_turn_deg = float(np.sqrt(np.mean(th_deg**2)))
    std_turn_deg = float(np.std(th_deg))
    p95 = float(np.percentile(th_deg, 95))

    gt_p95_exceedance = float("nan") if frac_threshold_deg is None else float(np.mean(th_deg > float(frac_threshold_deg)))

    return {
        "total_turn_rad": total_turn,   # rad
        "turn_per_mm": float(total_turn / total_len),   # rad/mm
        "rms_turn_deg": rms_turn_deg,   # deg
        "std_turn_deg": std_turn_deg,   # deg
        "p95_turn_deg": p95,            # deg
        "gt_p95_exceedance": gt_p95_exceedance,       # unitless
    }



def smoothness_debug_polydata_turn_only(sampled: SampledLines, eps: float = 1e-12) -> vtk.vtkPolyData:
    """
    PolyData (polylines) con PointData:
      - turn_deg: turning angle locale ai vertici interni (deg)
      - s_mm: ascissa curvilinea lungo il path
      - path_id: id del cammino (gruppo) usato per smoothness
    """
    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    turn_arr = vtk.vtkFloatArray(); turn_arr.SetName("turn_deg")
    s_arr    = vtk.vtkFloatArray(); s_arr.SetName("s_mm")
    pid_arr  = vtk.vtkIntArray();   pid_arr.SetName("path_id")

    pid_offset = 0

    for path_id, arr in enumerate(sampled.lines):
        n = int(arr.shape[0])
        if n < 2:
            continue

        turn_deg = np.zeros(n, dtype=np.float64)
        s_mm     = np.zeros(n, dtype=np.float64)

        seg = arr[1:] - arr[:-1]
        seg_len = np.linalg.norm(seg, axis=1)

        if seg_len.size:
            s_mm[1:] = np.cumsum(seg_len)

        if n >= 3:
            good = seg_len > eps
            t = np.zeros_like(seg)
            t[good] = seg[good] / seg_len[good][:, None]

            cos = np.sum(t[:-1] * t[1:], axis=1)
            cos = np.clip(cos, -1.0, 1.0)
            theta = np.arccos(cos)

            valid = (seg_len[:-1] > eps) & (seg_len[1:] > eps)
            theta[~valid] = 0.0
            turn_deg[1:-1] = theta * (180.0 / np.pi)

        for i in range(n):
            pts.InsertNextPoint(float(arr[i,0]), float(arr[i,1]), float(arr[i,2]))
            turn_arr.InsertNextValue(float(turn_deg[i]))
            s_arr.InsertNextValue(float(s_mm[i]))
            pid_arr.InsertNextValue(int(path_id))

        pl = vtk.vtkPolyLine()
        pl.GetPointIds().SetNumberOfIds(n)
        for i in range(n):
            pl.GetPointIds().SetId(i, pid_offset + i)
        lines.InsertNextCell(pl)
        pid_offset += n

    out = vtk.vtkPolyData()
    out.SetPoints(pts)
    out.SetLines(lines)
    out.GetPointData().AddArray(turn_arr)
    out.GetPointData().AddArray(s_arr)
    out.GetPointData().AddArray(pid_arr)
    out.GetPointData().SetActiveScalars("turn_deg")
    return out


def _hsv_to_rgb_tuple(h: float, s: float, v: float):
    """
    Compatibile con binding VTK diversi:
    - prova la signature che ritorna (r,g,b)
    - fallback: conversione manuale (stdlib)
    """
    # 1) prova: alcune versioni supportano return tuple/list
    try:
        rgb = vtk.vtkMath.HSVToRGB(h, s, v)  # <- in molte build ritorna (r,g,b)
        # può tornare tuple/list o vtk object: normalizziamo
        if hasattr(rgb, "__len__") and len(rgb) >= 3:
            return float(rgb[0]), float(rgb[1]), float(rgb[2])
    except TypeError:
        pass

    # 2) fallback manuale
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return float(r), float(g), float(b)


def _make_discrete_lut(n: int) -> vtk.vtkLookupTable:
    lut = vtk.vtkLookupTable()
    n = max(1, int(n))
    lut.SetNumberOfTableValues(n)
    lut.Build()

    for i in range(n):
        h = i / n
        r, g, b = _hsv_to_rgb_tuple(h, 0.85, 0.95)
        lut.SetTableValue(i, r, g, b, 1.0)

    return lut



def plot_polydata_with_nodes_by_scalar(
    poly: vtk.vtkPolyData,
    scalar_name: str,
    *,
    tube_radius: float = 0.05,
    point_radius: float = 0.12,
    scalar_range=None,
    categorical: bool = False
):
    arr = poly.GetPointData().GetArray(scalar_name)
    if arr is None:
        raise ValueError(f"Scalar '{scalar_name}' not found.")

    # LUT
    lut = None
    if categorical:
        # assume integer-ish scalars, build discrete lut for [0..max]
        r = arr.GetRange()
        n = int(round(r[1])) + 1
        lut = _make_discrete_lut(n)

    # ----- LINES (tubes) -----
    tube = vtk.vtkTubeFilter()
    tube.SetInputData(poly)
    tube.SetRadius(tube_radius)
    tube.SetNumberOfSides(12)
    tube.CappingOn()
    tube.Update()

    mapper_lines = vtk.vtkPolyDataMapper()
    mapper_lines.SetInputConnection(tube.GetOutputPort())
    mapper_lines.SetScalarModeToUsePointFieldData()
    mapper_lines.SelectColorArray(scalar_name)
    mapper_lines.ScalarVisibilityOn()

    if lut is not None:
        mapper_lines.SetLookupTable(lut)

    if scalar_range is None:
        r = arr.GetRange()
        mapper_lines.SetScalarRange(r[0], r[1])
    else:
        mapper_lines.SetScalarRange(float(scalar_range[0]), float(scalar_range[1]))

    actor_lines = vtk.vtkActor()
    actor_lines.SetMapper(mapper_lines)

    # ----- POINTS (spheres) -----
    sphere = vtk.vtkSphereSource()
    sphere.SetThetaResolution(16)
    sphere.SetPhiResolution(16)
    sphere.SetRadius(point_radius)

    glyph = vtk.vtkGlyph3DMapper()
    glyph.SetInputData(poly)
    glyph.SetSourceConnection(sphere.GetOutputPort())
    glyph.SetScalarModeToUsePointFieldData()
    glyph.SelectColorArray(scalar_name)
    glyph.ScalarVisibilityOn()

    if lut is not None:
        glyph.SetLookupTable(lut)

    if scalar_range is None:
        r = arr.GetRange()
        glyph.SetScalarRange(r[0], r[1])
    else:
        glyph.SetScalarRange(float(scalar_range[0]), float(scalar_range[1]))

    actor_pts = vtk.vtkActor()
    actor_pts.SetMapper(glyph)

    # scalar bar
    sb = vtk.vtkScalarBarActor()
    sb.SetLookupTable(mapper_lines.GetLookupTable())
    sb.SetTitle(scalar_name)
    sb.SetNumberOfLabels(5)

    ren = vtk.vtkRenderer()
    ren.SetBackground(0.08, 0.08, 0.10)
    ren.AddActor(actor_lines)
    ren.AddActor(actor_pts)
    # ren.AddActor2D(sb)
    ren.AddViewProp(sb)


    win = vtk.vtkRenderWindow()
    win.AddRenderer(ren)
    win.SetSize(1100, 850)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(win)

    ren.ResetCamera()
    win.Render()
    iren.Start()



# --------------------------
# Main eval
# --------------------------

def evaluate(
    gt_path: str,
    pred_path: str,
    step: float,
    tau: float,
    mask_path: Optional[str],
    r_mm: float,
    outdir: str,
    save_vtp: bool,
    case_id: Optional[int],
) -> None:
    
    # --- Data Loading
    gt_poly = read_vtp(gt_path)
    pred_poly = read_vtp(pred_path)
    
    # --- Fusion of non-unique Ids (related to the way the Pred is created)
    pred_poly = fusion_nonUniqueIds(pred_poly)
    # gt_poly = fusion_nonUniqueIds(gt_poly)
    
    # --- Resampling 
    gt_s = resample_polydata_lines(gt_poly, step=step)
    gt_poly_rs = sampled_to_polydata(gt_s, as_segments=False)
    
    pred_s = resample_polydata_lines(pred_poly, step=step)
    pred_poly_rs = sampled_to_polydata(pred_s, as_segments=False)
    
    # --- Info after resampling
    informations(gt_poly_rs, path="GT Resampled")
    informations(pred_poly_rs, path="Pred Resampled")
    
    # --- Fusion of non-unique Ids (After the resampling)
    pred_poly_rs = fusion_nonUniqueIds(pred_poly_rs)
    informations(pred_poly_rs, path="Pred Resampled - fused IDs")
    # gt_poly_rs = fusion_nonUniqueIds(gt_poly_rs)
    # informations(gt_poly_rs, path="GT Resampled - fused IDs")
    
    # save_vtp = true if we want to save the resampled files:
    if save_vtp:
        outdir_p = Path(outdir)     # Output directory setup
        suf = f"_{case_id:03d}" if case_id is not None else ""
        outdir_p.mkdir(parents=True, exist_ok=True)
        write_vtp(gt_poly_rs, str(outdir_p / f"GT_Resampled_step{step:g}{suf}.vtp"))
        write_vtp(pred_poly_rs, str(outdir_p / f"Pred_Resampled_step{step:g}{suf}.vtp"))


    # --- Format conversion to "segment-format"
    gt_poly_seg   = sampled_to_polydata(gt_s, as_segments=True)
    pred_poly_seg = sampled_to_polydata(pred_s, as_segments=True)
    
    # --- Fusion of non-unique Ids
    pred_poly_seg = fusion_nonUniqueIds(pred_poly_seg)
    # gt_poly_seg = fusion_nonUniqueIds(gt_poly_seg)
    
    
    # --- Check
    informations(gt_poly_seg, path="GT Segment-format (2-pt lines)")
    informations(pred_poly_seg, path="Pred Segment-format (2-pt lines)")
    plot_polydata(gt_poly_seg, tube_radius=0.05, end_r=0.3, mid_r=0.2, br_r=0.5)
    plot_polydata(pred_poly_seg, tube_radius=0.05, end_r=0.3, mid_r=0.2, br_r=0.5)
    
    
    ####### --- Up to this point, we obtained comparable polylines (GT and Pred) in "segment-format")
    
    
    # --- Info for resultìs print
    gt_seg_info = segment_format_basic_info(gt_poly_seg)
    pred_seg_info = segment_format_basic_info(pred_poly_seg)



    # ---------------
    # Metrics
    # ---------------
    
    # --- Topology stats:
    # #(ts) (ts_pred = topology_stats_predicition, ts_gt = topology_stats_groundtruth)
    ts_pred = topology_stats(pred_poly_rs)
    ts_gt = topology_stats(gt_poly_rs)
    
    # --- Segmentation containment stats (optional)
    if mask_path is not None:
        zooms = nib.load(mask_path).header.get_zooms()[:3]
        
        m_gt = centerline_vs_mask_metrics(gt_poly_seg, mask_path, r_mm=r_mm)
        m_pr = centerline_vs_mask_metrics(pred_poly_seg, mask_path, r_mm=r_mm)
        # print("\n[NEW]")
        # print("GT:", m_gt)
        # print("PR:", m_pr)
        
        n_gt = node_inside_outside_ratio(gt_poly_seg, mask_path)
        n_pr = node_inside_outside_ratio(pred_poly_seg, mask_path)
        # print("\nGT node containment:")
        # print(f"  inside_nodes: {n_gt['inside_nodes']} / {n_gt['total_nodes']}  ({n_gt['inside_ratio_nodes']:.4f})")
        # print(f"  outside_nodes:{n_gt['outside_nodes']} / {n_gt['total_nodes']}  ({n_gt['outside_ratio_nodes']:.4f})")
        # print("\nPred node containment:")
        # print(f"  inside_nodes: {n_pr['inside_nodes']} / {n_pr['total_nodes']}  ({n_pr['inside_ratio_nodes']:.4f})")
        # print(f"  outside_nodes:{n_pr['outside_nodes']} / {n_pr['total_nodes']}  ({n_pr['outside_ratio_nodes']:.4f})")
        
    
    
    # 2) geometric distances (bidirectional) on standardized segment-format polydata
    d_pred_to_gt = distances_polydata_to_polydata(pred_poly_seg, gt_poly_seg)
    d_gt_to_pred = distances_polydata_to_polydata(gt_poly_seg, pred_poly_seg)
    assd = 0.5 * (float(np.mean(d_pred_to_gt)) + float(np.mean(d_gt_to_pred))) if (d_pred_to_gt.size and d_gt_to_pred.size) else float("nan")
    hd95 = max(percentile(d_pred_to_gt, 95), percentile(d_gt_to_pred, 95))
    
    # def summarize(name: str, d: np.ndarray) -> None:
    #     if d.size == 0:
    #         print(f"{name}: empty")
    #         return
    #     print(f"{name}: mean={np.mean(d):.4f}  median={np.median(d):.4f}  p95={percentile(d,95):.4f}  max={np.max(d):.4f}")

    # summarize("\nDistances pred->GT", d_pred_to_gt)
    # summarize("Distances GT->pred", d_gt_to_pred)
    # print(f"ASSD: {assd:.4f}")
    # print(f"HD95: {hd95:.4f}")  
    

    # 0) bounds + coarse check
    mean_coarse = coarse_alignment_check(gt_poly, pred_poly)

    # # 3) coverage metrics on length
    # precision, recall, f1 = precision_recall_f1(gt_s, pred_s, gt_poly_rs, pred_poly_rs, tau=tau)
    # # print(f"Precision_tau (tau={tau}): {precision:.4f}")
    # # print(f"Recall_tau    (tau={tau}): {recall:.4f}")
    # # print(f"F1_tau        (tau={tau}): {f1:.4f}")

    # # coverage lengths (needed for report)
    # gt_cov = covered_length_of_sampledlines(gt_s, pred_poly_rs, tau)
    # pred_cov = covered_length_of_sampledlines(pred_s, gt_poly_rs, tau)


    # 3) coverage metrics on length (ora su segment-format)
    precision, recall, f1, gt_cov, pred_cov, gt_len_seg, pred_len_seg = coverage_metrics( gt_poly_seg, pred_poly_seg, tau=tau )

    
    #### Ultima aggiunta
    # 5) magia:
    gt_s_from_seg   = polyseg_to_sampledlines(gt_poly_seg, close_cycles=True)
    pred_s_from_seg = polyseg_to_sampledlines(pred_poly_seg, close_cycles=True)

    atm_gt = angular_tortuosity_metrics(gt_s_from_seg) # needed to compute the p95
    thr_deg = atm_gt["p95_turn_deg"]
    atm_gt = angular_tortuosity_metrics(gt_s_from_seg, frac_threshold_deg=thr_deg)
    atm_pr = angular_tortuosity_metrics(pred_s_from_seg, frac_threshold_deg=thr_deg)

    dbg_gt = smoothness_debug_polydata_turn_only(gt_s_from_seg)
    dbg_pr = smoothness_debug_polydata_turn_only(pred_s_from_seg)

    # range comune per turn_deg
    gt_turn = dbg_gt.GetPointData().GetArray("turn_deg").GetRange()
    pr_turn = dbg_pr.GetPointData().GetArray("turn_deg").GetRange()
    common_turn = (min(gt_turn[0], pr_turn[0]), max(gt_turn[1], pr_turn[1]))

    plot_polydata_with_nodes_by_scalar(dbg_gt, "turn_deg", tube_radius=0.05, point_radius=0.10, scalar_range=common_turn)
    plot_polydata_with_nodes_by_scalar(dbg_pr, "turn_deg", tube_radius=0.05, point_radius=0.10, scalar_range=common_turn)

    # plot gruppi/path: ogni path un colore diverso
    plot_polydata_with_nodes_by_scalar(dbg_gt, "path_id", tube_radius=0.05, point_radius=0.10, categorical=True)
    plot_polydata_with_nodes_by_scalar(dbg_pr, "path_id", tube_radius=0.05, point_radius=0.10, categorical=True)


        
    # --- FINAL REPORT ---
    R = {
        "case_tag": f"(case_{case_id:03d})" if case_id is not None else "",
        "step": step,
        "tau": tau,
        "r_mm": r_mm,
        "voxel_spacing": zooms if mask_path is not None else None,
        "gt_bounds": (
            lambda b: f"X[{b[0]:.3f},{b[1]:.3f}]  Y[{b[2]:.3f},{b[3]:.3f}]  Z[{b[4]:.3f},{b[5]:.3f}]"
        )(gt_poly.GetBounds()),

        "pred_bounds": (
            lambda b: f"X[{b[0]:.3f},{b[1]:.3f}]  Y[{b[2]:.3f},{b[3]:.3f}]  Z[{b[4]:.3f},{b[5]:.3f}]"
        )(pred_poly.GetBounds()),

        "coarse_mean": mean_coarse,

        # "gt_len": gt_s.total_length,
        # "pred_len": pred_s.total_length,
        "gt_len": gt_len_seg,
        "pred_len": pred_len_seg,
        "gt_n_polylines": len(gt_s.lines),
        "pred_n_polylines": len(pred_s.lines),

        "pred2gt_mean": float(np.mean(d_pred_to_gt)) if d_pred_to_gt.size else float("nan"),
        "pred2gt_median": float(np.median(d_pred_to_gt)) if d_pred_to_gt.size else float("nan"),
        "pred2gt_p95": percentile(d_pred_to_gt, 95),
        "pred2gt_max": float(np.max(d_pred_to_gt)) if d_pred_to_gt.size else float("nan"),

        "gt2pred_mean": float(np.mean(d_gt_to_pred)) if d_gt_to_pred.size else float("nan"),
        "gt2pred_median": float(np.median(d_gt_to_pred)) if d_gt_to_pred.size else float("nan"),
        "gt2pred_p95": percentile(d_gt_to_pred, 95),
        "gt2pred_max": float(np.max(d_gt_to_pred)) if d_gt_to_pred.size else float("nan"),

        "assd": assd,
        "hd95": hd95,

        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gt_cov": gt_cov,
        "pred_cov": pred_cov,

        "topo_pred": {
            "nodes": ts_pred.n_nodes,
            "edges": ts_pred.n_edges,
            "endpoints": ts_pred.n_endpoints,
            "junctions": ts_pred.n_junctions,
            "degree_hist": ts_pred.degree_hist,
        },
        "topo_gt": {
            "nodes": ts_gt.n_nodes,
            "edges": ts_gt.n_edges,
            "endpoints": ts_gt.n_endpoints,
            "junctions": ts_gt.n_junctions,
            "degree_hist": ts_gt.degree_hist,
        },
        # "seg4b": {
        #     "n_segments": len(segs),
        #     "mean": float(np.mean(lens)) if segs else 0.0,
        #     "median": float(np.median(lens)) if segs else 0.0,
        #     "max": int(np.max(lens)) if segs else 0,
        # },
        "gt_path": gt_path,
        "pred_path": pred_path,
        "mask_path": mask_path,
        "gt_seg_info": gt_seg_info,
        "pred_seg_info": pred_seg_info,
        "smoothness": {"GT": atm_gt, "Pred": atm_pr},
    }

    # attach segmentation blocks if enabled
    if mask_path is not None:
        R["seg_enabled"] = True
        R["r_mm"] = r_mm
        R["seg"] = {"GT": m_gt, "Pred": m_pr}
        R["nodes"] = {"GT": n_gt, "Pred": n_pr}
    else:
        R["seg_enabled"] = False

    print_report_A(R)




    
    

def main():
    
    ap = argparse.ArgumentParser()

    # input
    ap.add_argument("--gt", required=True, help="Path VTP ground truth")
    ap.add_argument("--pred", required=True, help="Path VTP prediction")
    ap.add_argument("--mask", default=None, help="Optional NIfTI mask path for GT/Pred vs segmentation metrics")

    # main params
    ap.add_argument("--step", type=float, default=0.5, help="Resampling step (same units as VTP, usually mm)")
    ap.add_argument("--tau", type=float, default=1.0, help="Tolerance distance tau for coverage metrics (same units as VTP)")
    ap.add_argument("--rmm", type=float, default=1.0, help="Distance threshold (mm) used by within_r_ratio in segmentation metrics")

    # switches
    ap.add_argument("--save-vtp", action="store_true", help="Save intermediate VTPs (merged/fixed/resampled)")

    # output naming
    ap.add_argument("--outdir", default="eval_outputs", help="Output directory for saved VTPs")
    ap.add_argument("--case-id", type=int, default=None, help="Optional case id used in output filenames (e.g. 3 -> _003)")

    args = ap.parse_args()

    evaluate(
        args.gt,
        args.pred,
        step=args.step,
        tau=args.tau,
        mask_path=args.mask,
        r_mm=args.rmm,
        outdir=args.outdir,
        save_vtp=args.save_vtp,
        case_id=args.case_id,
    )

    
    
if __name__ == "__main__":
    USE_CLI = False  # if True, use argparse

    if USE_CLI:
        main()
    else: # Hard-coded for quick run
        case_id = 3
        # bogdan
        pred = fr"C:\Users\ducci\Documents\Università_2025\6_SemesterProject\BrainGraph\output\Output_bogdan\vessel_graph_{case_id:03d}_v2.vtp"
        # gold standard
        gt = fr"C:\Users\ducci\Documents\Università_2025\6_SemesterProject\BrainGraph\data\ITKTubeTK_GoldStandardVtp\VascularNetwork-{case_id:03d}.vtp"
        # segmentation mask
        mask = f"C:/Users/ducci/Documents/Università_2025/6_SemesterProject/BrainGraph/data/ITKTubeTK_ManualSegmentationNii/labels-{case_id:03d}.nii.gz"
        mask = None
        
        # pred = f"C:/Users/ducci/Documents/Università_2025/6_SemesterProject/BrainGraph/output/Output_prova/ExCenterline_{case_id:03d}.vtp"
        # pred = f"C:/Users/ducci/Documents/Università_2025/6_SemesterProject/BrainGraph/output/Output_bogdan/vessel_graph_{case_id:03d}.vtp"
        # pred = fr"C:\Users\ducci\Documents\Università_2025\6_SemesterProject\BrainGraph\output\Output_basic_extractor\BasicCenterline_{case_id:03d}.vtp"
        # pred = fr"C:\Users\ducci\Documents\Università_2025\6_SemesterProject\BrainGraph\output\Output_bogdan\vessel_graph_{case_id:03d}_v3.vtp"
        
        pred = fr"C:\Users\ducci\Documents\Università_2025\6_SemesterProject\BrainGraph\data\Caravel_Centerlines\labels-{case_id:03d}.vtp"
        
        
        # flippate di bog
        # gt = fr"output\Output_bogdan\gold_fixed_003.vtp"
        # pred = fr"output\Output_bogdan\vessel_graph_003_v3.vtp"
        
        # flippate bogdan batch: OK stessi risultati
        # pred = fr"C:\Users\ducci\Documents\Università_2025\6_SemesterProject\BrainGraph\data\Bogdan_data\Caravel_Centerlines\labels-{case_id:03d}.vtp"
        gt = fr"C:\Users\ducci\Documents\Università_2025\6_SemesterProject\BrainGraph\data\Bogdan_data\Fixed_GoldVTPs\VascularNetworkFixed-{case_id:03d}.vtp"
        pred = fr"data\Bogdan_data\Alg_extraction\vessel_graph_{case_id:03d}.vtp"
        # pred = fr"data\Bogdan_data\Caravel_Centerlines\labels-{case_id:03d}.vtp"
        # pred = fr"C:\Users\ducci\Downloads\vessel_graph_aligned_204.vtp"
        
        evaluate(
            gt_path=gt,
            pred_path=pred,
            step=0.3,
            tau=0.5134,
            mask_path=mask,
            r_mm=1.0,
            outdir=r"C:\Users\ducci\Documents\Università_2025\6_SemesterProject\BrainGraph\output\Output_evaluations",
            save_vtp=True,
            case_id=case_id,
        )