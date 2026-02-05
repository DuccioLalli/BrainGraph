#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Set, Optional, Any
import numpy as np
import vtk
from pathlib import Path
from vtk.util.numpy_support import vtk_to_numpy
import json
import re

# --------------------------
# Input / Output utilities
# --------------------------

def read_vtp(path: str) -> vtk.vtkPolyData:
    r = vtk.vtkXMLPolyDataReader()
    r.SetFileName(path)
    r.Update()
    pd = r.GetOutput()
    if pd is None or pd.GetNumberOfPoints() == 0:
        raise ValueError(f"Empty or unread VTP file: {path}")
    return pd


#############################################
# PREPROCESS
#############################################

# --------------------------
# Fusion of non unique Ids
# --------------------------

def fusion_nonUniqueIds(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
    CLEAN_TOL = 1e-6
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(poly)
    cleaner.PointMergingOn()
    cleaner.SetToleranceIsAbsolute(True)
    cleaner.SetAbsoluteTolerance(CLEAN_TOL)
    cleaner.Update()
    return cleaner.GetOutput()


# --------------------------
# Resampling
# --------------------------

@dataclass
class SampledLines:
    lines: List[np.ndarray]
    total_length: float


def resample_polydata_lines(poly: vtk.vtkPolyData, step: float) -> SampledLines:
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
        if arr.shape[0] >= 2:
            total_len += float(np.sum(np.linalg.norm(arr[1:] - arr[:-1], axis=1)))

        rs = resample_polyline_local(arr)
        if rs.shape[0] >= 2:
            out_lines.append(rs)

    return SampledLines(lines=out_lines, total_length=total_len)


# --------------------------
# Conversions SampledLines <-> vtkPolyData
# --------------------------

def sampled_to_polydata(sampled: SampledLines, *, as_segments: bool = False) -> vtk.vtkPolyData:
    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    pid_offset = 0

    for arr in sampled.lines:
        n = int(arr.shape[0])
        if n < 2:
            continue

        for i in range(n):
            pts.InsertNextPoint(float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2]))

        if as_segments:
            for i in range(n - 1):
                lines.InsertNextCell(2)
                lines.InsertCellPoint(pid_offset + i)
                lines.InsertCellPoint(pid_offset + i + 1)
        else:
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


#############################################
# METRICS
#############################################

# --------------------------
# Topology
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
        n_nodes=int(n_nodes),
        n_edges=int(n_edges),
        n_endpoints=int(n_end),
        n_junctions=int(n_junc),
        degree_hist=dict(sorted((int(k), int(v)) for k, v in hist.items())),
    )


# --------------------------
# Segment helpers
# --------------------------

def _segment_info(pd: vtk.vtkPolyData):
    if pd is None or pd.GetNumberOfPoints() == 0 or pd.GetNumberOfCells() == 0:
        return (np.zeros((0, 3)), np.zeros((0,)), np.zeros((0, 3)))

    pts_vtk = pd.GetPoints()
    lines = pd.GetLines()
    if pts_vtk is None or lines is None or lines.GetNumberOfCells() == 0:
        return (np.zeros((0, 3)), np.zeros((0,)), np.zeros((0, 3)))

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


# --------------------------
# Distance helpers
# --------------------------

def build_cell_locator(poly: vtk.vtkPolyData, use_static: bool = True):
    if use_static and hasattr(vtk, "vtkStaticCellLocator"):
        loc = vtk.vtkStaticCellLocator()
    else:
        loc = vtk.vtkCellLocator()

    loc.SetDataSet(poly)
    loc.BuildLocator()
    return loc


def polydata_points(pd: vtk.vtkPolyData) -> np.ndarray:
    pts = pd.GetPoints()
    if pts is None or pd.GetNumberOfPoints() == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return vtk_to_numpy(pts.GetData()).astype(np.float64, copy=False)


def distances_points_to_locator(points: np.ndarray, locator) -> np.ndarray:
    n = points.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)

    out = np.empty((n,), dtype=np.float64)

    closest = [0.0, 0.0, 0.0]
    cell_id = vtk.reference(0)
    sub_id = vtk.reference(0)
    dist2 = vtk.reference(0.0)

    for i in range(n):
        p = points[i]
        locator.FindClosestPoint(
            (float(p[0]), float(p[1]), float(p[2])),
            closest, cell_id, sub_id, dist2
        )
        out[i] = float(np.sqrt(dist2.get()))
    return out


def distances_polydata_to_polydata(src_pd: vtk.vtkPolyData, ref_poly: vtk.vtkPolyData, use_static: bool = True) -> np.ndarray:
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


def coarse_alignment_check(gt_poly: vtk.vtkPolyData, pred_poly: vtk.vtkPolyData, n_samples: int = 2000, use_static: bool = True) -> float:
    if pred_poly is None or pred_poly.GetNumberOfPoints() == 0:
        return float("nan")
    if gt_poly is None or gt_poly.GetNumberOfCells() == 0:
        return float("nan")

    pred_pts = vtk_to_numpy(pred_poly.GetPoints().GetData()).astype(np.float64, copy=False)
    n = pred_pts.shape[0]
    k = min(int(n_samples), n)
    if k <= 0:
        return float("nan")

    idx = np.random.choice(n, size=k, replace=False)
    pts = pred_pts[idx]

    loc = build_cell_locator(gt_poly, use_static=use_static)
    d = distances_points_to_locator(pts, loc)
    return float(np.mean(d)) if d.size else float("nan")


# --------------------------
# Coverage metrics
# --------------------------

def covered_length_of_polydata(query_poly: vtk.vtkPolyData, ref_poly: vtk.vtkPolyData, tau: float, *, use_static: bool = True) -> float:
    mid, seg_len, _ = _segment_info(query_poly)
    if mid.shape[0] == 0:
        return 0.0
    if ref_poly is None or ref_poly.GetNumberOfCells() == 0:
        return 0.0

    loc = build_cell_locator(ref_poly, use_static=use_static)
    d = distances_points_to_locator(mid, loc)
    return float(np.sum(seg_len[d <= float(tau)]))


def coverage_metrics(gt_poly_seg: vtk.vtkPolyData, pred_poly_seg: vtk.vtkPolyData, tau: float, use_static: bool = True) -> Tuple[float, float, float, float, float, float, float]:
    _, seg_len_gt, _ = _segment_info(gt_poly_seg)
    gt_len = float(np.sum(seg_len_gt))

    _, seg_len_pr, _ = _segment_info(pred_poly_seg)
    pred_len = float(np.sum(seg_len_pr))

    if gt_len <= 1e-12 or pred_len <= 1e-12:
        return float("nan"), float("nan"), float("nan"), 0.0, 0.0, gt_len, pred_len

    gt_cov = covered_length_of_polydata(gt_poly_seg, pred_poly_seg, tau, use_static=use_static)
    recall = gt_cov / gt_len

    pred_cov = covered_length_of_polydata(pred_poly_seg, gt_poly_seg, tau, use_static=use_static)
    precision = pred_cov / pred_len

    f1 = 0.0 if (precision + recall) <= 1e-12 else (2.0 * precision * recall / (precision + recall))
    return float(precision), float(recall), float(f1), float(gt_cov), float(pred_cov), float(gt_len), float(pred_len)


# --------------------------
# Segment-format basic info
# --------------------------

def segment_format_basic_info(poly: vtk.vtkPolyData) -> dict:
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


# --------------------------
# angular_tortuosity
# --------------------------

def build_adjacency_from_polydata(pd: vtk.vtkPolyData) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = {}
    lines = pd.GetLines()
    if lines is None or lines.GetNumberOfCells() == 0:
        return adj

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


def extract_paths_between_critical_nodes(adj: Dict[int, Set[int]]) -> Tuple[List[List[int]], Set[Tuple[int, int]]]:
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


def extract_cycles_degree2(adj: Dict[int, Set[int]], visited_edges: Set[Tuple[int, int]], *, close_cycles: bool = True) -> List[List[int]]:
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
                    cycle.append(start)
                break

            if ek(cur, nxt) in visited_edges:
                cycle = []
                break

            cycle.append(nxt)
            visited_edges.add(ek(cur, nxt))
            prev, cur = cur, nxt

        if cycle and len(cycle) >= (4 if close_cycles else 3):
            cycles.append(cycle)

    return cycles


def polyseg_to_sampledlines(pd_seg: vtk.vtkPolyData, *, close_cycles: bool = True) -> SampledLines:
    pts_vtk = pd_seg.GetPoints()
    if pts_vtk is None or pd_seg.GetNumberOfPoints() == 0:
        return SampledLines(lines=[], total_length=0.0)

    pts = vtk_to_numpy(pts_vtk.GetData()).astype(np.float64, copy=False)

    adj = build_adjacency_from_polydata(pd_seg)
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


def angular_tortuosity_metrics(sampled: SampledLines, eps: float = 1e-12, *, frac_threshold_deg: float | None = None) -> dict:
    total_len = 0.0
    thetas = []

    for arr in sampled.lines:
        if arr.shape[0] < 3:
            continue
        v = arr[1:] - arr[:-1]
        seg_len = np.linalg.norm(v, axis=1)
        total_len += float(np.sum(seg_len))

        t = v / (seg_len[:, None] + eps)

        cos = np.sum(t[:-1] * t[1:], axis=1)
        cos = np.clip(cos, -1.0, 1.0)
        theta = np.arccos(cos)

        valid = (seg_len[:-1] > eps) & (seg_len[1:] > eps)
        theta = theta[valid]

        if theta.size:
            thetas.append(theta)

    if not thetas or total_len <= eps:
        return {
            "total_turn_rad": 0.0,
            "turn_per_mm": float("nan"),
            "rms_turn_deg": float("nan"),
            "p95_turn_deg": float("nan"),
            "max_turn_deg": float("nan"),
        }

    th = np.concatenate(thetas)
    th_deg = th * (180.0 / np.pi)
    total_turn = float(np.sum(th))
    rms_turn_deg = float(np.sqrt(np.mean(th**2))) * (180.0 / np.pi)
    gt_p95_exceedance = float("nan") if frac_threshold_deg is None else float(np.mean(th_deg > float(frac_threshold_deg)))
    
    return {
        "total_turn_rad": total_turn,
        "turn_per_mm": float(total_turn / total_len),
        "rms_turn_deg": rms_turn_deg,
        "p95_turn_deg": float(np.percentile(th * (180.0 / np.pi), 95)),
        "max_turn_deg": float(np.max(th * (180.0 / np.pi))),
        "gt_p95_exceedance": gt_p95_exceedance,
    }


#############################################
# EVALUATION
#############################################

def evaluate_to_dict(
    gt_path: str,
    pred_path: str,
    *,
    step: float,
    tau: float,
    case_id: Optional[int],
    use_static: bool = True,
) -> dict:

    # --- Data Loading
    gt_poly = read_vtp(gt_path)
    pred_poly = read_vtp(pred_path)

    # --- Fusion of non-unique Ids (Pred)
    pred_poly = fusion_nonUniqueIds(pred_poly)

    # --- Save the original num_nodes
    gt_original_num_nodes = int(gt_poly.GetNumberOfPoints())
    pred_original_num_nodes = int(pred_poly.GetNumberOfPoints())
    
    # --- Resampling
    gt_s = resample_polydata_lines(gt_poly, step=step)
    gt_poly_rs = sampled_to_polydata(gt_s, as_segments=False)

    pred_s = resample_polydata_lines(pred_poly, step=step)
    pred_poly_rs = sampled_to_polydata(pred_s, as_segments=False)

    # --- Fusion after resampling (Pred)
    gt_poly_rs = fusion_nonUniqueIds(gt_poly_rs)
    pred_poly_rs = fusion_nonUniqueIds(pred_poly_rs)

    # --- Format conversion to segment-format
    gt_poly_seg = sampled_to_polydata(gt_s, as_segments=True)
    pred_poly_seg = sampled_to_polydata(pred_s, as_segments=True)

    # --- Fusion in segment-format (Pred)
    pred_poly_seg = fusion_nonUniqueIds(pred_poly_seg)
    gt_poly_seg = fusion_nonUniqueIds(gt_poly_seg)
    
    # --- Info for print/report
    gt_seg_info = segment_format_basic_info(gt_poly_seg)
    pred_seg_info = segment_format_basic_info(pred_poly_seg)

    # --- Topology on resampled polylines
    ts_pred = topology_stats(pred_poly_rs)
    ts_gt = topology_stats(gt_poly_rs)

    # --- Geometry distances (segment-format)
    d_pred_to_gt = distances_polydata_to_polydata(pred_poly_seg, gt_poly_seg, use_static=use_static)
    d_gt_to_pred = distances_polydata_to_polydata(gt_poly_seg, pred_poly_seg, use_static=use_static)
    assd = 0.5 * (float(np.mean(d_pred_to_gt)) + float(np.mean(d_gt_to_pred))) if (d_pred_to_gt.size and d_gt_to_pred.size) else float("nan")
    hd95 = max(percentile(d_pred_to_gt, 95), percentile(d_gt_to_pred, 95))

    # --- Coarse sanity check
    mean_coarse = coarse_alignment_check(gt_poly, pred_poly, use_static=use_static)

    # --- Coverage metrics (segment-format)
    precision, recall, f1, gt_cov, pred_cov, gt_len_seg, pred_len_seg = coverage_metrics(
        gt_poly_seg, pred_poly_seg, tau=tau, use_static=use_static
    )

    # --- angular_tortuosity (turning-angle)
    gt_s_from_seg = polyseg_to_sampledlines(gt_poly_seg, close_cycles=True)
    pred_s_from_seg = polyseg_to_sampledlines(pred_poly_seg, close_cycles=True)
    sm_gt0 = angular_tortuosity_metrics(gt_s_from_seg)
    thr = sm_gt0["p95_turn_deg"]
    sm_gt = angular_tortuosity_metrics(gt_s_from_seg, frac_threshold_deg=thr)
    sm_pr = angular_tortuosity_metrics(pred_s_from_seg, frac_threshold_deg=thr)

    # --- FINAL REPORT dict
    R = {
        "case_tag": f"(case_{case_id:03d})" if case_id is not None else "",
        "step": step,
        "tau": tau,

        "gt_bounds": (
            lambda b: f"X[{b[0]:.3f},{b[1]:.3f}]  Y[{b[2]:.3f},{b[3]:.3f}]  Z[{b[4]:.3f},{b[5]:.3f}]"
        )(gt_poly.GetBounds()),
        "pred_bounds": (
            lambda b: f"X[{b[0]:.3f},{b[1]:.3f}]  Y[{b[2]:.3f},{b[3]:.3f}]  Z[{b[4]:.3f},{b[5]:.3f}]"
        )(pred_poly.GetBounds()),

        "coarse_mean": mean_coarse,

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

        "gt_path": gt_path,
        "pred_path": pred_path,

        "gt_seg_info": gt_seg_info,
        "pred_seg_info": pred_seg_info,

        "angular_tortuosity": {"GT": sm_gt, "Pred": sm_pr},
        
        "gt_original_num_nodes": gt_original_num_nodes,
        "pred_original_num_nodes": pred_original_num_nodes,

    }

    return R


#############################################
# BATCH RUNNER
#############################################

# --------------------------
# Batch utils: matching cases by numeric id in filename
# --------------------------

def extract_case_key(p: Path, pad: int = 3) -> Optional[str]:
    name = p.name
    stem = name[:-7] if name.lower().endswith(".nii.gz") else p.stem
    groups = re.findall(r"\d+", stem)
    if not groups:
        return None
    cid = int(groups[-1])
    return f"{cid:0{pad}d}"


def build_map(dir_path: Path, exts: Tuple[str, ...], recursive: bool, pad: int) -> Dict[str, Path]:
    files = (dir_path.rglob("*") if recursive else dir_path.iterdir())
    out: Dict[str, Path] = {}
    for p in files:
        if not p.is_file():
            continue
        low = p.name.lower()
        if not any(low.endswith(e) for e in exts):
            continue
        k = extract_case_key(p, pad=pad)
        if k is None:
            continue
        if k not in out:
            out[k] = p
    return out


# --------------------------
# Output utils: JSON serializable + flatten for CSV
# --------------------------

def to_jsonable(x: Any) -> Any:
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (tuple, list)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    return x


def flatten_dict(d: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, parent=key, sep=sep))
        else:
            out[key] = v
    return out


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    cols = sorted(set().union(*[set(r.keys()) for r in rows]))
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, (dict, list)):
                    v = json.dumps(v, ensure_ascii=False)
                s = "" if v is None else str(v)
                if any(ch in s for ch in [",", "\n", '"']):
                    s = '"' + s.replace('"', '""') + '"'
                vals.append(s)
            f.write(",".join(vals) + "\n")



EXCLUDE_JSON_KEYS = {
    "gt_bounds",
    "pred_bounds",
    "gt_n_polylines",
    "pred_n_polylines",
}


# --------------------------
# Main batch CLI
# --------------------------

def main():
    ap = argparse.ArgumentParser(description="Batch centerline evaluation (saves your final R outputs).")

    ap.add_argument("--pred_dir", required=True, help="Folder with predicted .vtp")
    ap.add_argument("--gt_dir", required=True, help="Folder with GT .vtp")
    ap.add_argument("--out_dir", required=True, help="Output folder")

    ap.add_argument("--step", type=float, default=0.3)
    ap.add_argument("--tau", type=float, default=0.5134)

    ap.add_argument("--recursive", action="store_true", help="Search recursively")
    ap.add_argument("--pad", type=int, default=3, help="Case-id padding (default 3 -> 003)")

    ap.add_argument("--use_static", action="store_true", help="Use vtkStaticCellLocator if available")
    ap.add_argument("--per_case_json", action="store_true", default=True, help="Save one JSON per case under out_dir/per_case_json/ (default: ON)")
    ap.add_argument("--no_per_case_json", action="store_false", dest="per_case_json", help="Disable per-case JSON saving")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_dir.exists():
        raise FileNotFoundError(f"pred_dir not found: {pred_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"gt_dir not found: {gt_dir}")

    pred_map = build_map(pred_dir, exts=(".vtp",), recursive=args.recursive, pad=args.pad)
    gt_map = build_map(gt_dir, exts=(".vtp",), recursive=args.recursive, pad=args.pad)

    common = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    missing_pred = sorted(set(gt_map.keys()) - set(pred_map.keys()))
    missing_gt = sorted(set(pred_map.keys()) - set(gt_map.keys()))

    print(f"Found pred: {len(pred_map)} | gt: {len(gt_map)} | matched: {len(common)}")
    if missing_pred:
        print(f"Missing in pred_dir: {len(missing_pred)} (e.g. {missing_pred[:5]})")
    if missing_gt:
        print(f"Missing in gt_dir:   {len(missing_gt)} (e.g. {missing_gt[:5]})")

    results_jsonl = out_dir / "results.jsonl"
    results_csv = out_dir / "results.csv"

    per_case_dir = out_dir / "per_case_json"
    if args.per_case_json:
        per_case_dir.mkdir(parents=True, exist_ok=True)

    rows_flat: List[Dict[str, Any]] = []
    n_ok, n_fail = 0, 0

    with results_jsonl.open("w", encoding="utf-8") as f:
        for i, ck in enumerate(common, 1):
            gt_path = gt_map[ck]
            pred_path = pred_map[ck]

            try:
                R = evaluate_to_dict(
                    gt_path=str(gt_path),
                    pred_path=str(pred_path),
                    step=args.step,
                    tau=args.tau,
                    case_id=int(ck),
                    use_static=args.use_static,
                )

                R["case_key"] = ck
                Rj = to_jsonable(R)

                # drop unwanted keys
                for k in EXCLUDE_JSON_KEYS:
                    Rj.pop(k, None)
                    
                f.write(json.dumps(Rj, ensure_ascii=False) + "\n")

                flat = flatten_dict(Rj)
                rows_flat.append(flat)

                if args.per_case_json:
                    with (per_case_dir / f"case_{ck}.json").open("w", encoding="utf-8") as fj:
                        json.dump(Rj, fj, ensure_ascii=False, indent=2)

                n_ok += 1
                print(f"[{i}/{len(common)}] OK   case {ck}  (GT={gt_path.name} | PRED={pred_path.name})")

            except Exception as e:
                n_fail += 1
                err = {
                    "case_key": ck,
                    "error": str(e),
                    "gt_path": str(gt_path),
                    "pred_path": str(pred_path),
                }
                f.write(json.dumps(err, ensure_ascii=False) + "\n")
                rows_flat.append(flatten_dict(err))
                print(f"[{i}/{len(common)}] FAIL case {ck}: {e}")

    write_csv(rows_flat, results_csv)

    print("\nDone.")
    print(f"  OK   : {n_ok}")
    print(f"  FAIL : {n_fail}")
    print(f"Saved: {results_jsonl}")
    print(f"Saved: {results_csv}")
    if args.per_case_json:
        print(f"Saved per-case JSON in: {per_case_dir}")


if __name__ == "__main__":
    main()
