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
    Reads a VTP file and returns the PolyData object.
    """
    r = vtk.vtkXMLPolyDataReader()
    r.SetFileName(path)
    r.Update()
    pd = r.GetOutput()
    if pd is None or pd.GetNumberOfPoints() == 0:
        raise ValueError(f"Empty or unread VTP file: {path}")
    return pd



# --------------------------
# print of the info of a file, for debugging
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
        print("--- File:", os.path.basename(path))
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
# plot of the polydata with nodes colored by degree
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
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        vg = vtk.vtkVertexGlyphFilter()
        vg.SetInputData(pd)
        vg.Update()
        return vg.GetOutput()

    end_pd = make_points_poly(end_ids)
    mid_pd = make_points_poly(mid_ids)
    br_pd  = make_points_poly(br_ids)

    # === GLYPHS: spheres on nodes ===
    sphere = vtk.vtkSphereSource()
    sphere.SetThetaResolution(16)
    sphere.SetPhiResolution(16)

    def make_node_actor(pd, radius, color_rgb):
        g = vtk.vtkGlyph3D()
        g.SetSourceConnection(sphere.GetOutputPort())
        g.SetInputData(pd)
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
# Fusion of non unique Ids
# --------------------------

def fusion_nonUniqueIds(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
    CLEAN_TOL = 1e-6  # in unità coordinate (spesso mm). prova 1e-4 se serve.
    
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(poly)
    cleaner.PointMergingOn()
    cleaner.SetToleranceIsAbsolute(True)
    cleaner.SetAbsoluteTolerance(CLEAN_TOL)
    cleaner.Update()
    poly = cleaner.GetOutput()
    
    print("Points:", poly.GetNumberOfPoints(), "Lines:", poly.GetNumberOfLines())
    
    return poly

# --------------------------
# Resampling
# --------------------------

# def iter_polylines(pd: vtk.vtkPolyData) -> Iterable[np.ndarray]:
#     """
#     Itera su ogni cella lineare (polyline) e restituisce array Nx3 di punti in ordine.
#     """
#     lines = pd.GetLines()
#     if lines is None:
#         return
#     lines.InitTraversal()
#     id_list = vtk.vtkIdList()

#     pts = pd.GetPoints()
#     while lines.GetNextCell(id_list):
#         n = id_list.GetNumberOfIds()
#         if n < 2:
#             continue
#         arr = np.zeros((n, 3), dtype=np.float64)
#         for i in range(n):
#             pid = id_list.GetId(i)
#             arr[i] = pts.GetPoint(pid)
#         yield arr


# def polyline_length(arr: np.ndarray) -> float:
#     if arr.shape[0] < 2:
#         return 0.0
#     diffs = arr[1:] - arr[:-1]
#     return float(np.sum(np.linalg.norm(diffs, axis=1)))


# def resample_polyline(arr: np.ndarray, step: float) -> np.ndarray:
#     """
#     Resampling di una singola polyline a passo fisso `step`.
#     Mantiene sempre l'ultimo punto.
#     """
#     if arr.shape[0] < 2:
#         return arr.copy()

#     seg = arr[1:] - arr[:-1]
#     seg_len = np.linalg.norm(seg, axis=1)
#     cum = np.concatenate([[0.0], np.cumsum(seg_len)])
#     total = cum[-1]
#     if total <= 1e-12:
#         # tutti i punti uguali
#         return arr[:1].copy()

#     # campioni: 0, step, 2*step, ..., total
#     s_vals = np.arange(0.0, total, step, dtype=np.float64)
#     if s_vals.size == 0 or s_vals[-1] < total:
#         s_vals = np.concatenate([s_vals, [total]])

#     out = np.zeros((s_vals.size, 3), dtype=np.float64)

#     # Per ogni s, trova il segmento in cum
#     j = 0
#     for i, s in enumerate(s_vals):
#         while j < len(cum) - 2 and cum[j + 1] < s:
#             j += 1
#         s0, s1 = cum[j], cum[j + 1]
#         if s1 <= s0 + 1e-12:
#             out[i] = arr[j]
#         else:
#             t = (s - s0) / (s1 - s0)
#             out[i] = (1 - t) * arr[j] + t * arr[j + 1]
#     return out


# def resample_polydata_lines(pd: vtk.vtkPolyData, step: float) -> SampledLines:
#     out_lines: List[np.ndarray] = []
#     total_len = 0.0
#     for arr in iter_polylines(pd):
#         total_len += polyline_length(arr)
#         rs = resample_polyline(arr, step)
#         if rs.shape[0] >= 2:
#             out_lines.append(rs)
#     return SampledLines(lines=out_lines, total_length=total_len)


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


def sampled_to_polydata(sampled: SampledLines) -> vtk.vtkPolyData:
    """
    Converts: SampledLines in vtkPolyData with polylines.
    """
    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    pid_offset = 0
    for arr in sampled.lines:
        n = arr.shape[0]
        for i in range(n):
            pts.InsertNextPoint(float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2]))
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
# Topology (4.A): degrees / endpoints / junctions
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
# Utils
# --------------------------

def write_vtp(pd: vtk.vtkPolyData, path: str) -> None:
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(path)
    w.SetInputData(pd)
    w.SetDataModeToAppended()
    if hasattr(w, "SetCompressorTypeToZLib"):
        w.SetCompressorTypeToZLib()
    w.Write()
    
    

# ===========================================



# -------------------------
# centerline vs Segmentation Metrics

# def world_to_voxel(points_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
#     """
#     points_xyz: (N,3) in world (mm)
#     affine: nibabel affine voxel->world
#     ritorna: (N,3) in voxel coords (float), ordine i,j,k
#     """
#     inv = np.linalg.inv(affine)
#     N = points_xyz.shape[0]
#     homog = np.c_[points_xyz, np.ones((N,1))]
#     ijk = (inv @ homog.T).T[:, :3]
#     return ijk


# def centerline_vs_mask_metrics(sampled_lines, nii_path: str, r_mm: float = 1.0):
#     import nibabel as nib
#     import numpy as np
#     from scipy.ndimage import distance_transform_edt, map_coordinates

#     img = nib.load(nii_path)
#     mask = img.get_fdata() > 0
#     affine = img.affine
#     spacing = img.header.get_zooms()[:3]

#     dt_to_vessel = distance_transform_edt(~mask, sampling=spacing)
#     dt_inside = distance_transform_edt(mask, sampling=spacing)

#     # from gt_s.total_length (class created before)
#     total_len = float(sampled_lines.total_length)
#     inside_len = 0.0

#     dt_samples = []
#     inside_samples = []

#     outside_run = 0.0
#     max_outside_run = 0.0
#     n_outside_runs = 0
#     currently_outside = False

#     for arr in sampled_lines.lines:
#         if arr.shape[0] < 2:
#             continue

#         p0 = arr[:-1]
#         p1 = arr[1:]
#         seg_len = np.linalg.norm(p1 - p0, axis=1)
#         mid = (p0 + p1) / 2.0

#         ijk = world_to_voxel(mid, affine)
#         x, y, z = ijk[:, 0], ijk[:, 1], ijk[:, 2]

#         d_out = map_coordinates(dt_to_vessel, [x, y, z], order=1, mode="nearest")
#         d_in  = map_coordinates(dt_inside,    [x, y, z], order=1, mode="nearest")

#         dt_samples.append(d_out)
#         inside_samples.append(d_in)

#         # inside definition (hard-ish): very close to vessel interior
#         inside_flag = d_out < (0.25 * float(min(spacing)))

#         # >>> inside length accumulates only inside segments
#         inside_len += float(np.sum(seg_len[inside_flag]))

#         # outside runs over segments
#         for L, out in zip(seg_len, ~inside_flag):
#             if out:
#                 outside_run += float(L)
#                 if not currently_outside:
#                     currently_outside = True
#                     n_outside_runs += 1
#             else:
#                 max_outside_run = max(max_outside_run, outside_run)
#                 outside_run = 0.0
#                 currently_outside = False

#     max_outside_run = max(max_outside_run, outside_run)

#     dt_all = np.concatenate(dt_samples) if dt_samples else np.array([])
#     din_all = np.concatenate(inside_samples) if inside_samples else np.array([])

#     outside_len = total_len - inside_len
#     inside_ratio = inside_len / total_len if total_len > 1e-12 else float("nan")

#     out = {
#         "total_length": total_len,
#         "inside_length": inside_len,
#         "outside_length": outside_len,
#         "inside_ratio_len": inside_ratio,
#         "outside_runs": n_outside_runs,
#         "max_outside_run": max_outside_run,
#     }

#     if dt_all.size:
#         out.update({
#             "dt_mean(mm)": float(np.mean(dt_all)),
#             "dt_median(mm)": float(np.median(dt_all)),
#             "dt_p95(mm)": float(np.percentile(dt_all, 95)),
#             "dt_max(mm)": float(np.max(dt_all)),
#             "within_r_ratio": float(np.mean(dt_all <= r_mm)),
#         })
#     if din_all.size:
#         out.update({
#             "dt_inside_median": float(np.median(din_all)),
#             "dt_inside_p10": float(np.percentile(din_all, 10)),
#         })

#     return out

@dataclass
class SampledLines:
    """
    Rappresentazione "resamplata":
    - points: lista di array (Ni,3) per ogni linea
    - total_length: somma lunghezze originali (o resamplate)
    """
    lines: List[np.ndarray]
    total_length: float

# metrica mia, controllata
def node_inside_outside_ratio(pd_seg: vtk.vtkPolyData, nii_path: str) -> dict:
    """
    Versione su vtkPolyData (segment-format + clean consigliati):
    - considera i NODI (Points del polydata)
    - inside se il voxel nearest (round) in mask è True
    """
    img = nib.load(nii_path)
    mask = img.get_fdata() > 0
    affine = img.affine

    pts_vtk = pd_seg.GetPoints()
    if pts_vtk is None or pd_seg.GetNumberOfPoints() == 0:
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


def _extract_segments_from_polydata(pd: vtk.vtkPolyData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estrae segmenti come coppie di punti consecutivi dalle celle lineari.
    Funziona sia se le celle sono tutte 2-pt (segment-format), sia se ci sono polilinee (>2).
    Ritorna:
      mid: (M,3) midpoints dei segmenti
      seg_len: (M,) lunghezze in mm
      dP: (M,3) vettori (p1 - p0) (opzionale per debug/estensioni)
    """
    pts_vtk = pd.GetPoints()
    if pts_vtk is None or pd.GetNumberOfPoints() == 0:
        return (np.zeros((0, 3), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64))

    pts = vtk_to_numpy(pts_vtk.GetData()).astype(np.float64, copy=False)

    lines = pd.GetLines()
    if lines is None or lines.GetNumberOfCells() == 0:
        return (np.zeros((0, 3), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64))

    seg_a = []
    seg_b = []

    # Fast path: offsets/connectivity
    try:
        offs = vtk_to_numpy(lines.GetOffsetsArray())
        conn = vtk_to_numpy(lines.GetConnectivityArray())

        for k in range(len(offs) - 1):
            a, b = int(offs[k]), int(offs[k + 1])
            ids = conn[a:b]
            if ids.size < 2:
                continue
            # se è 2-pt line -> 1 segmento; se polyline -> segmenti consecutivi
            for j in range(ids.size - 1):
                u = int(ids[j])
                v = int(ids[j + 1])
                if u == v:
                    continue
                seg_a.append(u)
                seg_b.append(v)

    # Fallback: traversal classico
    except Exception:
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
                seg_a.append(u)
                seg_b.append(v)

    if not seg_a:
        return (np.zeros((0, 3), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64))

    ia = np.asarray(seg_a, dtype=np.int64)
    ib = np.asarray(seg_b, dtype=np.int64)

    p0 = pts[ia]
    p1 = pts[ib]
    dP = p1 - p0
    seg_len = np.linalg.norm(dP, axis=1)
    mid = (p0 + p1) * 0.5

    return mid, seg_len, dP

from scipy.ndimage import distance_transform_edt, map_coordinates
def centerline_vs_mask_metrics(pd_seg: vtk.vtkPolyData, nii_path: str, r_mm: float = 1.0) -> dict:
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

    mid, seg_len, _ = _extract_segments_from_polydata(pd_seg)

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

# # -------------------------


# -------------------------- 
# to_polyline Section

# def build_adjacency_from_polydata(pd: vtk.vtkPolyData) -> Dict[int, Set[int]]:
#     adj: Dict[int, Set[int]] = {}
#     lines = pd.GetLines()
#     if lines is None:
#         return adj

#     lines.InitTraversal()
#     id_list = vtk.vtkIdList()

#     while lines.GetNextCell(id_list):
#         n = id_list.GetNumberOfIds()
#         if n < 2:
#             continue
#         for i in range(n - 1):
#             a = int(id_list.GetId(i))
#             b = int(id_list.GetId(i + 1))
#             adj.setdefault(a, set()).add(b)
#             adj.setdefault(b, set()).add(a)
#     return adj


# def extract_paths_between_critical_nodes(adj: Dict[int, Set[int]]) -> Tuple[List[List[int]], Set[Tuple[int,int]]]:
#     """
#     Estrae cammini massimali tra nodi critici (degree != 2).
#     Ritorna:
#       - paths: lista di path come liste di vertex ids
#       - visited_edges: set di edge visitati (u,v) undirected normalizzato
#     """
#     deg = {u: len(vs) for u, vs in adj.items()}
#     critical = {u for u, d in deg.items() if d != 2}

#     visited_edges: Set[Tuple[int, int]] = set()
#     paths: List[List[int]] = []

#     def edge_key(u: int, v: int) -> Tuple[int,int]:
#         return (u, v) if u < v else (v, u)

#     for u in critical:
#         for v in adj.get(u, []):
#             if edge_key(u, v) in visited_edges:
#                 continue

#             path = [u, v]
#             visited_edges.add(edge_key(u, v))

#             prev = u
#             cur = v
#             while cur not in critical:
#                 # degree == 2: due vicini
#                 nbrs = list(adj[cur])
#                 nxt = nbrs[0] if nbrs[1] == prev else nbrs[1]
#                 if edge_key(cur, nxt) in visited_edges:
#                     break
#                 path.append(nxt)
#                 visited_edges.add(edge_key(cur, nxt))
#                 prev, cur = cur, nxt

#             paths.append(path)

#     return paths, visited_edges


# def extract_cycles_degree2(adj: Dict[int, Set[int]], visited_edges: Set[Tuple[int,int]]) -> List[List[int]]:
#     """
#     Estrae cicli rimasti (tipicamente componenti con tutti degree==2).
#     Usa visited_edges per non duplicare.
#     """
#     deg = {u: len(vs) for u, vs in adj.items()}

#     def edge_key(u: int, v: int) -> Tuple[int,int]:
#         return (u, v) if u < v else (v, u)

#     cycles: List[List[int]] = []
#     seen_nodes: Set[int] = set()

#     for start in adj.keys():
#         if deg.get(start, 0) != 2:
#             continue
#         if start in seen_nodes:
#             continue

#         # prova a camminare finché torni al start
#         nbrs = list(adj[start])
#         if len(nbrs) != 2:
#             continue

#         # scegli una direzione
#         prev = start
#         cur = nbrs[0]
#         cycle = [start, cur]
#         seen_nodes.add(start)

#         ok = True
#         while True:
#             seen_nodes.add(cur)
#             nbrs_cur = list(adj[cur])
#             if len(nbrs_cur) != 2:
#                 ok = False
#                 break
#             nxt = nbrs_cur[0] if nbrs_cur[1] == prev else nbrs_cur[1]

#             # se chiude il ciclo
#             if nxt == start:
#                 # chiudi (opzionale: non ripetere start alla fine)
#                 # marca ultimo edge
#                 visited_edges.add(edge_key(cur, nxt))
#                 break

#             # edge già visitato? allora potrebbe essere già coperto
#             if edge_key(cur, nxt) in visited_edges:
#                 ok = False
#                 break

#             visited_edges.add(edge_key(prev, cur))
#             visited_edges.add(edge_key(cur, nxt))

#             cycle.append(nxt)
#             prev, cur = cur, nxt

#             # sicurezza
#             if len(cycle) > 10_000_000:
#                 ok = False
#                 break

#         if ok and len(cycle) >= 3:
#             cycles.append(cycle)

#     return cycles


# def build_polydata_from_paths(original_pd: vtk.vtkPolyData, paths: List[List[int]]) -> vtk.vtkPolyData:
#     """
#     Crea un nuovo vtkPolyData che usa GLI STESSI PUNTI (stesse coordinate e stesso indexing),
#     ma sostituisce le lines con polylines "mergeate" secondo i path.
#     """
#     # copia punti (manteniamo stesso ordine, quindi gli id restano validi)
#     new_pts = vtk.vtkPoints()
#     new_pts.DeepCopy(original_pd.GetPoints())

#     new_lines = vtk.vtkCellArray()
#     for path in paths:
#         if len(path) < 2:
#             continue
#         pl = vtk.vtkPolyLine()
#         pl.GetPointIds().SetNumberOfIds(len(path))
#         for i, pid in enumerate(path):
#             pl.GetPointIds().SetId(i, int(pid))
#         new_lines.InsertNextCell(pl)

#     out = vtk.vtkPolyData()
#     out.SetPoints(new_pts)
#     out.SetLines(new_lines)
#     return out


# def merge_pred_edges_to_polylines(pred_poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
#     adj = build_adjacency_from_polydata(pred_poly)

#     paths, visited = extract_paths_between_critical_nodes(adj)
#     cycles = extract_cycles_degree2(adj, visited)

#     merged_paths = paths + cycles
#     merged_pd = build_polydata_from_paths(pred_poly, merged_paths)
#     return merged_pd


# end to_polyline functions
# --------------------------



# --------------------------
# Input / Output utilities
# --------------------------


def polydata_bounds(pd: vtk.vtkPolyData) -> Tuple[float, float, float, float, float, float]:
    return pd.GetBounds()  # (xmin,xmax,ymin,ymax,zmin,zmax)


# def print_bounds(name: str, b: Tuple[float, float, float, float, float, float]) -> None:
#     xmin, xmax, ymin, ymax, zmin, zmax = b
#     print(f"{name} bounds: X[{xmin:.3f},{xmax:.3f}]  Y[{ymin:.3f},{ymax:.3f}]  Z[{zmin:.3f},{zmax:.3f}]")



# ballottagio di ora 18:08 del 22/01
# --------------------------
# Distance: points(src_pd) -> polyline set(ref_pd)
# --------------------------

def build_cell_locator(pd: vtk.vtkPolyData, use_static: bool = True):
    """Build locator for closest-point queries on ref_pd line cells."""
    if use_static and hasattr(vtk, "vtkStaticCellLocator"):
        loc = vtk.vtkStaticCellLocator()
    else:
        loc = vtk.vtkCellLocator()
    loc.SetDataSet(pd)
    loc.BuildLocator()
    return loc


def polydata_points(pd: vtk.vtkPolyData) -> np.ndarray:
    """Return Nx3 float64 points from vtkPolyData."""
    pts = pd.GetPoints()
    if pts is None or pd.GetNumberOfPoints() == 0:
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


def distances_polydata_to_polydata(src_pd: vtk.vtkPolyData, ref_pd: vtk.vtkPolyData, use_static: bool = True) -> np.ndarray:
    """
    One-way distances:
      For each point in src_pd, compute distance to closest point on ref_pd line set.
    """
    if ref_pd is None or ref_pd.GetNumberOfCells() == 0:
        return np.array([], dtype=np.float64)

    src_pts = polydata_points(src_pd)
    if src_pts.size == 0:
        return np.array([], dtype=np.float64)

    loc = build_cell_locator(ref_pd, use_static=use_static)
    return distances_points_to_locator(src_pts, loc)


def percentile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))



####################################################################


# def build_cell_locator(pd: vtk.vtkPolyData, use_static: bool = True):
#     """
#     Build a locator to query closest point on the line set.
#     vtkStaticCellLocator is usually faster for static datasets.
#     """
#     loc = vtk.vtkStaticCellLocator() if use_static and hasattr(vtk, "vtkStaticCellLocator") else vtk.vtkCellLocator()
#     loc.SetDataSet(pd)
#     loc.BuildLocator()
#     return loc


# def polydata_points(pd: vtk.vtkPolyData) -> np.ndarray:
#     """
#     Returns Nx3 float64 points from a vtkPolyData.
#     """
#     pts_vtk = pd.GetPoints()
#     if pts_vtk is None or pd.GetNumberOfPoints() == 0:
#         return np.zeros((0, 3), dtype=np.float64)
#     return vtk_to_numpy(pts_vtk.GetData()).astype(np.float64, copy=False)


# def distances_points_to_locator(points: np.ndarray, locator) -> np.ndarray:
#     """
#     For each point (Nx3), compute distance to closest point on the locator dataset.
#     """
#     if points.size == 0:
#         return np.array([], dtype=np.float64)

#     out = np.empty((points.shape[0],), dtype=np.float64)
#     closest = [0.0, 0.0, 0.0]
#     cell_id = vtk.reference(0)
#     sub_id = vtk.reference(0)
#     dist2 = vtk.reference(0.0)

#     for i in range(points.shape[0]):
#         p = points[i]
#         locator.FindClosestPoint((float(p[0]), float(p[1]), float(p[2])), closest, cell_id, sub_id, dist2)
#         out[i] = float(np.sqrt(dist2.get()))
#     return out


# def distances_polydata_to_polydata(src_pd: vtk.vtkPolyData, ref_pd: vtk.vtkPolyData, *, use_static: bool = True) -> np.ndarray:
#     """
#     One-way distances:
#       For each point in src_pd, compute distance to closest point on the line set in ref_pd.
#     """
#     src_pts = polydata_points(src_pd)
#     if src_pts.size == 0 or ref_pd is None or ref_pd.GetNumberOfCells() == 0:
#         return np.array([], dtype=np.float64)

#     loc = build_cell_locator(ref_pd, use_static=use_static)
#     return distances_points_to_locator(src_pts, loc)


# def percentile(arr: np.ndarray, q: float) -> float:
#     if arr.size == 0:
#         return float("nan")
#     return float(np.percentile(arr, q))


# def geometric_distance_metrics_bidirectional(pred_pd: vtk.vtkPolyData, gt_pd: vtk.vtkPolyData, *, use_static: bool = True) -> dict:
#     """
#     Computes:
#       - d_pred_to_gt: distances from Pred points to GT line set
#       - d_gt_to_pred: distances from GT points to Pred line set
#       - ASSD: 0.5*(mean(pred->gt) + mean(gt->pred))
#       - HD95: max(p95(pred->gt), p95(gt->pred))
#     Returns a dict with arrays + summary scalars.
#     """
#     d_pred_to_gt = distances_polydata_to_polydata(pred_pd, gt_pd, use_static=use_static)
#     d_gt_to_pred = distances_polydata_to_polydata(gt_pd, pred_pd, use_static=use_static)

#     assd = (
#         0.5 * (float(np.mean(d_pred_to_gt)) + float(np.mean(d_gt_to_pred)))
#         if (d_pred_to_gt.size and d_gt_to_pred.size)
#         else float("nan")
#     )
#     hd95 = max(percentile(d_pred_to_gt, 95), percentile(d_gt_to_pred, 95))

#     return {
#         "d_pred_to_gt": d_pred_to_gt,
#         "d_gt_to_pred": d_gt_to_pred,
#         "assd(mm)": assd,
#         "hd95(mm)": hd95,
#         "pred->gt_mean(mm)": float(np.mean(d_pred_to_gt)) if d_pred_to_gt.size else float("nan"),
#         "pred->gt_median(mm)": float(np.median(d_pred_to_gt)) if d_pred_to_gt.size else float("nan"),
#         "pred->gt_p95(mm)": percentile(d_pred_to_gt, 95),
#         "pred->gt_max(mm)": float(np.max(d_pred_to_gt)) if d_pred_to_gt.size else float("nan"),
#         "gt->pred_mean(mm)": float(np.mean(d_gt_to_pred)) if d_gt_to_pred.size else float("nan"),
#         "gt->pred_median(mm)": float(np.median(d_gt_to_pred)) if d_gt_to_pred.size else float("nan"),
#         "gt->pred_p95(mm)": percentile(d_gt_to_pred, 95),
#         "gt->pred_max(mm)": float(np.max(d_gt_to_pred)) if d_gt_to_pred.size else float("nan"),
#     }
    
    
##########################################################################






# --------------------------
# Coverage metrics on length (Precision/Recall/F1)
# --------------------------

def covered_length_of_sampledlines(sampled: SampledLines, ref_pd: vtk.vtkPolyData, tau: float) -> float:
    """
    Stima della lunghezza "coperta" usando segmenti tra punti resamplati:
    - per ogni segmento consecutive (pi->pi+1) si prende il midpoint
    - se distanza(midpoint, ref) <= tau => conta tutta la lunghezza del segmento
    """
    if not sampled.lines:
        return 0.0

    loc = build_cell_locator(ref_pd)

    total = 0.0
    closest = [0.0, 0.0, 0.0]
    cell_id = vtk.reference(0)
    sub_id = vtk.reference(0)
    dist2 = vtk.reference(0.0)

    for arr in sampled.lines:
        if arr.shape[0] < 2:
            continue
        p0 = arr[:-1]
        p1 = arr[1:]
        seg_vec = p1 - p0
        seg_len = np.linalg.norm(seg_vec, axis=1)
        mid = (p0 + p1) / 2.0

        for i in range(mid.shape[0]):
            locator_point = mid[i].tolist()
            loc.FindClosestPoint(locator_point, closest, cell_id, sub_id, dist2)
            d = float(np.sqrt(dist2.get()))
            if d <= tau:
                total += float(seg_len[i])

    return total


def precision_recall_f1(
    gt_sampled: SampledLines,
    pred_sampled: SampledLines,
    gt_poly_resampled: vtk.vtkPolyData,
    pred_poly_resampled: vtk.vtkPolyData,
    tau: float
) -> Tuple[float, float, float]:
    
    gt_len = gt_sampled.total_length
    pred_len = pred_sampled.total_length
    if gt_len <= 1e-12 or pred_len <= 1e-12:
        return float("nan"), float("nan"), float("nan")

    # Recall: GT covered by Pred
    gt_cov = covered_length_of_sampledlines(gt_sampled, pred_poly_resampled, tau)
    recall = gt_cov / gt_len

    # Precision: Pred supported by GT
    pred_cov = covered_length_of_sampledlines(pred_sampled, gt_poly_resampled, tau)
    precision = pred_cov / pred_len

    if precision + recall <= 1e-12:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    # print(f"GT_covered_length   (tau={tau}): {gt_cov:.4f}")
    # print(f"Pred_covered_length (tau={tau}): {pred_cov:.4f}")
    # print(f"GT_missing_length (approx):   {gt_len - gt_cov:.4f}")
    # print(f"Pred_extra_length (approx):   {pred_len - pred_cov:.4f}")

    return float(precision), float(recall), float(f1)


# # --------------------------
# # Topology (4.A): degrees / endpoints / junctions
# # --------------------------

# @dataclass
# class TopologyStats:
#     n_nodes: int
#     n_edges: int
#     n_endpoints: int
#     n_junctions: int
#     degree_hist: Dict[int, int]


# # def topology_stats_from_adj(adj: Dict[int, Set[int]]) -> TopologyStats:
# #     degrees = {k: len(v) for k, v in adj.items()}
# #     n_nodes = len(degrees) 
# #     n_edges = sum(degrees.values()) // 2 
# #     n_end = sum(1 for d in degrees.values() if d == 1) 
# #     n_junc = sum(1 for d in degrees.values() if d >= 3) 
# #     hist: Dict[int, int] = {} 
# #     for d in degrees.values(): 
# #         hist[d] = hist.get(d, 0) + 1 
    
# #     return TopologyStats( 
# #         n_nodes=n_nodes, 
# #         n_edges=n_edges, 
# #         n_endpoints=n_end, 
# #         n_junctions=n_junc, 
# #         degree_hist=dict(sorted(hist.items())) 
# #     )

# def topology_stats(pd: vtk.vtkPolyData) -> TopologyStats:

#     adj: Dict[int, Set[int]] = {}

#     lines = pd.GetLines()
#     if lines is None:
#         return TopologyStats(0, 0, 0, 0, {})

#     lines.InitTraversal()
#     id_list = vtk.vtkIdList()

#     while lines.GetNextCell(id_list):
#         n = id_list.GetNumberOfIds()
#         if n < 2:
#             continue
#         for i in range(n - 1):
#             a = int(id_list.GetId(i))
#             b = int(id_list.GetId(i + 1))
#             if a == b:
#                 continue
#             adj.setdefault(a, set()).add(b)
#             adj.setdefault(b, set()).add(a)

#     degrees = {k: len(v) for k, v in adj.items()}
#     n_nodes = len(degrees)
#     n_edges = sum(degrees.values()) // 2
#     n_end = sum(1 for d in degrees.values() if d == 1)
#     n_junc = sum(1 for d in degrees.values() if d >= 3)

#     hist: Dict[int, int] = {}
#     for d in degrees.values():
#         hist[d] = hist.get(d, 0) + 1

#     return TopologyStats(
#         n_nodes=n_nodes,
#         n_edges=n_edges,
#         n_endpoints=n_end,
#         n_junctions=n_junc,
#         degree_hist=dict(sorted(hist.items())),
#     )


# --------------------------
# 4.B Segment extraction (between critical nodes degree != 2)
# --------------------------

# @dataclass
# class Segment:
#     vertex_ids: List[int]


# def extract_segments_between_critical_nodes(adj: Dict[int, Set[int]]) -> List[Segment]:
#     """
#     Estrae segmenti come cammini massimali tra nodi critici (degree != 2).
#     Evita duplicati marcando gli edge visitati.
#     """
#     deg = {u: len(v) for u, v in adj.items()}
#     critical = {u for u, d in deg.items() if d != 2}

#     visited_edges: Set[Tuple[int, int]] = set()
#     segments: List[Segment] = []

#     def mark_edge(u: int, v: int) -> None:
#         visited_edges.add((u, v))
#         visited_edges.add((v, u))

#     for u in critical:
#         for v in adj.get(u, []):
#             if (u, v) in visited_edges:
#                 continue

#             path = [u, v]
#             mark_edge(u, v)

#             prev = u
#             cur = v
#             while cur not in critical:
#                 # degree==2 quindi esistono esattamente 2 vicini
#                 nbrs = list(adj[cur])
#                 nxt = nbrs[0] if nbrs[1] == prev else nbrs[1]
#                 if (cur, nxt) in visited_edges:
#                     break
#                 path.append(nxt)
#                 mark_edge(cur, nxt)
#                 prev, cur = cur, nxt

#             segments.append(Segment(vertex_ids=path))

#     return segments


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

def bounds_str(b):
    return f"X[{b[0]:.3f},{b[1]:.3f}]  Y[{b[2]:.3f},{b[3]:.3f}]  Z[{b[4]:.3f},{b[5]:.3f}]"

def print_report_A(R: dict) -> None:
    print("\n" + "=" * 70)
    print(f" EVAL SUMMARY {R.get('case_tag','')}".center(70))
    print("=" * 70)

    # --- NEW: print filenames (top lines) ---
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
    # --------------------------------------

    print("\n[0] CONFIG / SANITY")
    print(f"  step={R['step']}  tau={R['tau']}  r_mm={R.get('r_mm','n/a')}")
    if R.get("voxel_spacing") is not None:
        sp = R["voxel_spacing"]
        print(f"  voxel spacing (mm): ({sp[0]:.6f}, {sp[1]:.6f}, {sp[2]:.6f})")
    print(f"  GT bounds  : {R['gt_bounds']}")
    print(f"  Pred bounds: {R['pred_bounds']}")
    print(f"  Coarse mean distance (pred points -> GT lines): {fmt(R['coarse_mean'],4,' mm')}")

    print("\n[1] RANKING (key numbers)")
    print(f"  F1_tau  : {fmt(R['f1'],4)}")
    print(f"  ASSD    : {fmt(R['assd'],4,' mm')}")
    print(f"  HD95    : {fmt(R['hd95'],4,' mm')}")
    print(f"  LenΔ    : {fmt(R['pred_len']-R['gt_len'],4,' mm')} (Pred - GT)")

    print("\n[2] LENGTH / FRAGMENTATION")
    print(f"  GT total length   : {fmt(R['gt_len'],4,' mm')}")
    print(f"  Pred total length : {fmt(R['pred_len'],4,' mm')}")
    # --- NEW: LenΔ redundant here too ---
    print(f"  Length difference (LenΔ) : {fmt(R['pred_len']-R['gt_len'],4,' mm')} (Pred - GT)")
    # -----------------------------------
    print(f"  GT #resampled polylines   : {R['gt_n_polylines']}")
    print(f"  Pred #resampled polylines : {R['pred_n_polylines']}")

    print(f"\n[3] COVERAGE @ tau={R['tau']} mm")
    print(f"  Precision_tau : {fmt(R['precision'],4)}")
    print(f"  Recall_tau    : {fmt(R['recall'],4)}")
    print(f"  F1_tau        : {fmt(R['f1'],4)}   <-- (same as Ranking)")
    print(f"  GT_covered_length   : {fmt(R['gt_cov'],4,' mm')}")
    print(f"  Pred_covered_length : {fmt(R['pred_cov'],4,' mm')}")
    # --- NEW: explicit labels you asked for ---
    print(f"  GT_missing_length (approx): {fmt(R['gt_len']-R['gt_cov'],4,' mm')}")
    print(f"  Pred_extra_length (approx): {fmt(R['pred_len']-R['pred_cov'],4,' mm')}")
    # ----------------------------------------

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
            # keep your aligned style
            print(f"  total_length:       {fmt(M['total_length'],4)}")
            print(f"  inside_length:      {fmt(M['inside_length'],4)}")
            print(f"  outside_length:     {fmt(M['outside_length'],4)}")
            print(f"  inside_ratio_len:   {fmt(M['inside_ratio_len'],4)}")
            # print(f"  outside_runs:       {int(M['outside_runs'])}")
            # print(f"  max_outside_run:    {fmt(M['max_outside_run'],4)}")
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
    print(f"        endpoints(deg=1)={tp['endpoints']} branch(deg>=3)={tp['junctions']}")
    print(f"        degree_hist={tp['degree_hist']}")

    print(f"  GT  : nodes={tg['nodes']}")
    print(f"        endpoints(deg=1)={tg['endpoints']} branch(deg>=3)={tg['junctions']}")
    print(f"        degree_hist={tg['degree_hist']}")
    if tg["junctions"] == 0:
        print("        NOTE: GT branch(deg>=3)=0 likely means the GT has a format with no shared branches")

    # print("\n[7] 4.B SEGMENTS (Pred)")
    # s = R["seg4b"]
    # print(f"  #segments: {s['n_segments']}")
    # if s["n_segments"] > 0:
    #     print(f"  segment vertex-count: mean={fmt(s['mean'],2)}  median={fmt(s['median'],2)}  max={s['max']}")

    print("\n" + "=" * 70)





def sampled_to_segment_polydata(sampled: SampledLines) -> vtk.vtkPolyData:
    """
    Converte SampledLines in vtkPolyData dove:
      - Points = tutti i punti resamplati
      - Lines  = SOLO segmenti (2 punti) tra punti consecutivi
    Quindi ogni edge è una vtkLine implicita (cell con 2 ids).
    """
    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    pid_offset = 0

    for arr in sampled.lines:
        n = int(arr.shape[0])
        if n < 2:
            continue

        # Inserisci tutti i punti di questa polyline
        for i in range(n):
            pts.InsertNextPoint(float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2]))

        # Inserisci i segmenti (n-1) come linee a 2 punti
        for i in range(n - 1):
            lines.InsertNextCell(2)
            lines.InsertCellPoint(pid_offset + i)
            lines.InsertCellPoint(pid_offset + i + 1)

        pid_offset += n

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(lines)
    return poly



# --------------------------
# Main eval
# --------------------------

def evaluate(
    gt_path: str,   # Si usa
    pred_path: str, # si usa
    step: float,    # si usa
    tau: float,
    to_polyline: bool,
    mask_path: Optional[str],
    r_mm: float,
    outdir: str,
    save_vtp: bool,
    case_id: Optional[int],
) -> None:
    
    # Data Loading
    gt_poly = read_vtp(gt_path)
    pred_poly = read_vtp(pred_path)

    # informations(gt_poly, path=gt_path)
    # informations(pred_poly, path=pred_path)
    
    # plot_polydata(gt_poly, tube_radius=0.05, end_r=0.3, mid_r=0.2, br_r=0.5)
    # plot_polydata(pred_poly, tube_radius=0.05, end_r=0.3, mid_r=0.2, br_r=0.5)
    
    # --- Fusion of non-unique Ids (Pred)
    pred_poly = fusion_nonUniqueIds(pred_poly)
    # plot_polydata(pred_poly, tube_radius=0.05, end_r=0.3, mid_r=0.2, br_r=0.5)
    
    # --- RESAMPLING 
    gt_s = resample_polydata_lines(gt_poly, step=step)
    gt_poly_rs = sampled_to_polydata(gt_s)
    
    pred_s = resample_polydata_lines(pred_poly, step=step)
    pred_poly_rs = sampled_to_polydata(pred_s)


    # --- Info after resampling
    informations(gt_poly_rs, path="GT Resampled")
    # informations(pred_poly_rs, path="Pred Resampled")
    
    # --- Fusion of non-unique Ids (Pred)
    pred_poly_rs = fusion_nonUniqueIds(pred_poly_rs)
    informations(pred_poly_rs, path="Pred Resampled - fused IDs")
    
    # plot_polydata(gt_poly_rs, tube_radius=0.05, end_r=0.3, mid_r=0.2, br_r=0.5)
    # plot_polydata(pred_poly_rs, tube_radius=0.05, end_r=0.3, mid_r=0.2, br_r=0.5)
    
    # --- Up to this point, we obtain comparable polylines (GT and Pred))
    
    # save_vtp = true if we want to save the resampled files:
    if save_vtp:
        outdir_p = Path(outdir)     # Output directory setup
        suf = f"_{case_id:03d}" if case_id is not None else ""
        outdir_p.mkdir(parents=True, exist_ok=True)
        write_vtp(gt_poly_rs, str(outdir_p / f"GT_Resampled_step{step:g}{suf}.vtp"))
        write_vtp(pred_poly_rs, str(outdir_p / f"Pred_Resampled_step{step:g}{suf}.vtp"))


    # INOLTRE IMPORTANTE: FARE LA CHIAMATA DELLe metriche topologiche: 4A sulle poly resemplate, non su quelle in imput.
    # 4.A Topology Stats: (ts) (ts_pred = topology_stats_predicition, ts_gt = topology_stats_groundtruth)
    ts_pred = topology_stats(pred_poly_rs)
    ts_gt = topology_stats(gt_poly_rs)


    # conersione di formato
    gt_poly_seg = sampled_to_segment_polydata(gt_s)
    pred_poly_seg = sampled_to_segment_polydata(pred_s)
    informations(gt_poly_seg, path="GT Segment-format (2-pt lines)")
    pred_poly_seg = fusion_nonUniqueIds(pred_poly_seg)
    informations(pred_poly_seg, path="Pred Segment-format (2-pt lines)")
    plot_polydata(gt_poly_seg, tube_radius=0.05, end_r=0.3, mid_r=0.2, br_r=0.5)
    plot_polydata(pred_poly_seg, tube_radius=0.05, end_r=0.3, mid_r=0.2, br_r=0.5)
    
    
    
    # ---- Segmentation containment (optional) ----
    if mask_path is not None:
        zooms = nib.load(mask_path).header.get_zooms()[:3]
        
        # controllato
        m_gt = centerline_vs_mask_metrics(gt_poly_seg, mask_path, r_mm=r_mm)
        m_pr = centerline_vs_mask_metrics(pred_poly_seg, mask_path, r_mm=r_mm)
        # print("\n[NEW]")
        # print("GT:", m_gt)
        # print("PR:", m_pr)
        
        # controllato
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
    
    def summarize(name: str, d: np.ndarray) -> None:
        if d.size == 0:
            print(f"{name}: empty")
            return
        print(f"{name}: mean={np.mean(d):.4f}  median={np.median(d):.4f}  p95={percentile(d,95):.4f}  max={np.max(d):.4f}")

    summarize("\nDistances pred->GT", d_pred_to_gt)
    summarize("Distances GT->pred", d_gt_to_pred)
    print(f"ASSD: {assd:.4f}")
    print(f"HD95: {hd95:.4f}")  

    
    
# =============
    
    # # --- optional: merge pred edges -> polylines
    # if to_polyline:
    #     pred_poly_merged = merge_pred_edges_to_polylines(pred_poly)
    #     if save_vtp:
    #         write_vtp(pred_poly_merged, str(outdir_p / f"PredMergedToPoly{suf}.vtp"))
    #     pred_poly = pred_poly_merged

    # 0) bounds + coarse check
    mean_coarse = coarse_alignment_check(gt_poly, pred_poly)

    
    

    # 3) coverage metrics on length
    precision, recall, f1 = precision_recall_f1(gt_s, pred_s, gt_poly_rs, pred_poly_rs, tau=tau)
    # print(f"Precision_tau (tau={tau}): {precision:.4f}")
    # print(f"Recall_tau    (tau={tau}): {recall:.4f}")
    # print(f"F1_tau        (tau={tau}): {f1:.4f}")

    # coverage lengths (needed for report)
    gt_cov = covered_length_of_sampledlines(gt_s, pred_poly_rs, tau)
    pred_cov = covered_length_of_sampledlines(pred_s, gt_poly_rs, tau)

    ##############
    # 4.B segments QUESTO STA DANDO ERRORE, TOGLILO, METRICA INUTILE;     
    # segs = extract_segments_between_critical_nodes(adj_pred)
    # # print(f"\n4.B Pred segments between critical nodes: {len(segs)}")
    # if segs:
    #     lens = [len(s.vertex_ids) for s in segs]
    #     # print(f"  segment vertex-count: mean={np.mean(lens):.2f}  median={np.median(lens):.2f}  max={np.max(lens)}")
        
    # --- FINAL REPORT ---
    R = {
        "case_tag": f"(case_{case_id:03d})" if case_id is not None else "",
        "step": step,
        "tau": tau,
        "r_mm": r_mm,
        "voxel_spacing": zooms if mask_path is not None else None,
        "gt_bounds": bounds_str(polydata_bounds(gt_poly)),
        "pred_bounds": bounds_str(polydata_bounds(pred_poly)),
        "coarse_mean": mean_coarse,

        "gt_len": gt_s.total_length,
        "pred_len": pred_s.total_length,
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
    ap.add_argument("--to-polyline", action="store_true", help="Merge pred edges into polylines before eval")
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
        to_polyline=args.to_polyline,
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
        
        pred = f"C:/Users/ducci/Documents/Università_2025/6_SemesterProject/BrainGraph/output/Output_prova/ExCenterline_{case_id:03d}.vtp"
        # pred = f"C:/Users/ducci/Documents/Università_2025/6_SemesterProject/BrainGraph/output/Output_bogdan/vessel_graph_{case_id:03d}.vtp"
        # pred = fr"C:\Users\ducci\Documents\Università_2025\6_SemesterProject\BrainGraph\output\Output_basic_extractor\BasicCenterline_{case_id:03d}.vtp"
        
        evaluate(
            gt_path=gt,
            pred_path=pred,
            step=0.3,
            tau=0.5,
            to_polyline=True,
            mask_path=mask,
            r_mm=1.0,
            outdir=r"C:\Users\ducci\Documents\Università_2025\6_SemesterProject\BrainGraph\output\Output_evaluations",
            save_vtp=True,
            case_id=case_id,
        )
