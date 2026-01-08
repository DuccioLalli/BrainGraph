#!/usr/bin/env python3
"""
centerline_eval.py

Valutazione di centerline 3D in formato VTP (vtkPolyData con lines).
- Resampling uniforme a passo s (in unità del VTP: tipicamente mm)
- Metriche geometriche: pred->gt, gt->pred, ASSD, HD95
- Metriche di copertura su lunghezza: Precision_tau, Recall_tau, F1_tau
- Topologia (4.A): endpoints e junctions (degree-based)
- (Opzionale) 4.B: estrazione segmenti tra nodi critici (degree != 2)

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Set, Optional

import numpy as np

import vtk

from pathlib import Path
import nibabel as nib


# -------------------------
# centerline vs Segmentation Metrics

def world_to_voxel(points_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    points_xyz: (N,3) in world (mm)
    affine: nibabel affine voxel->world
    ritorna: (N,3) in voxel coords (float), ordine i,j,k
    """
    inv = np.linalg.inv(affine)
    N = points_xyz.shape[0]
    homog = np.c_[points_xyz, np.ones((N,1))]
    ijk = (inv @ homog.T).T[:, :3]
    return ijk


def centerline_vs_mask_metrics(sampled_lines, nii_path: str, r_mm: float = 1.0):
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import distance_transform_edt, map_coordinates

    img = nib.load(nii_path)
    mask = img.get_fdata() > 0
    affine = img.affine
    spacing = img.header.get_zooms()[:3]

    dt_to_vessel = distance_transform_edt(~mask, sampling=spacing)
    dt_inside = distance_transform_edt(mask, sampling=spacing)

    # >>> FIX: use the "original" length from sampled_lines (coherent with gt_s.total_length)
    total_len = float(sampled_lines.total_length)
    inside_len = 0.0

    dt_samples = []
    inside_samples = []

    outside_run = 0.0
    max_outside_run = 0.0
    n_outside_runs = 0
    currently_outside = False

    for arr in sampled_lines.lines:
        if arr.shape[0] < 2:
            continue

        p0 = arr[:-1]
        p1 = arr[1:]
        seg_len = np.linalg.norm(p1 - p0, axis=1)
        mid = (p0 + p1) / 2.0

        ijk = world_to_voxel(mid, affine)
        x, y, z = ijk[:, 0], ijk[:, 1], ijk[:, 2]

        d_out = map_coordinates(dt_to_vessel, [x, y, z], order=1, mode="nearest")
        d_in  = map_coordinates(dt_inside,    [x, y, z], order=1, mode="nearest")

        dt_samples.append(d_out)
        inside_samples.append(d_in)

        # inside definition (hard-ish): very close to vessel interior
        inside_flag = d_out < (0.25 * float(min(spacing)))

        # >>> inside length accumulates only inside segments
        inside_len += float(np.sum(seg_len[inside_flag]))

        # outside runs over segments
        for L, out in zip(seg_len, ~inside_flag):
            if out:
                outside_run += float(L)
                if not currently_outside:
                    currently_outside = True
                    n_outside_runs += 1
            else:
                max_outside_run = max(max_outside_run, outside_run)
                outside_run = 0.0
                currently_outside = False

    max_outside_run = max(max_outside_run, outside_run)

    dt_all = np.concatenate(dt_samples) if dt_samples else np.array([])
    din_all = np.concatenate(inside_samples) if inside_samples else np.array([])

    outside_len = total_len - inside_len
    inside_ratio = inside_len / total_len if total_len > 1e-12 else float("nan")

    out = {
        "total_length": total_len,
        "inside_length": inside_len,
        "outside_length": outside_len,
        "inside_ratio_len": inside_ratio,
        "outside_runs": n_outside_runs,
        "max_outside_run": max_outside_run,
    }

    if dt_all.size:
        out.update({
            "dt_mean(mm)": float(np.mean(dt_all)),
            "dt_median(mm)": float(np.median(dt_all)),
            "dt_p95(mm)": float(np.percentile(dt_all, 95)),
            "dt_max(mm)": float(np.max(dt_all)),
            "within_r_ratio": float(np.mean(dt_all <= r_mm)),
        })
    if din_all.size:
        out.update({
            "dt_inside_median": float(np.median(din_all)),
            "dt_inside_p10": float(np.percentile(din_all, 10)),
        })

    return out

@dataclass
class SampledLines:
    """
    Rappresentazione "resamplata":
    - points: lista di array (Ni,3) per ogni linea
    - total_length: somma lunghezze originali (o resamplate)
    """
    lines: List[np.ndarray]
    total_length: float

# metrica mia
def node_inside_outside_ratio(sampled_lines: SampledLines, nii_path: str) -> dict:
    """
    Metrica 'discreta' come hai descritto tu:
    - converte ogni nodo (punto resampled) in voxel (arrotondato)
    - inside se mask==1, outside se mask==0
    Ritorna conteggi e ratio.
    """
    img = nib.load(nii_path)
    mask = img.get_fdata() > 0
    affine = img.affine

    inside = 0
    total = 0

    sx, sy, sz = mask.shape

    for arr in sampled_lines.lines:
        if arr.size == 0:
            continue

        ijk = world_to_voxel(arr, affine)          # float voxel coords
        ijk_round = np.rint(ijk).astype(int)       # nearest voxel

        x = np.clip(ijk_round[:, 0], 0, sx - 1)
        y = np.clip(ijk_round[:, 1], 0, sy - 1)
        z = np.clip(ijk_round[:, 2], 0, sz - 1)

        vals = mask[x, y, z]
        inside += int(np.sum(vals))
        total += int(vals.size)

    return {
        "inside_nodes": inside,
        "outside_nodes": total - inside,
        "total_nodes": total,
        "inside_ratio_nodes": (inside / total) if total > 0 else float("nan"),
        "outside_ratio_nodes": ((total - inside) / total) if total > 0 else float("nan"),
    }


# def print_mask_metrics(title: str, m: dict) -> None:
#     print(f"\n{title}")
#     print(f"  total_length:       {m['total_length']:.4f}")
#     print(f"  inside_length:      {m['inside_length']:.4f}")
#     print(f"  outside_length:     {m['outside_length']:.4f}")
#     print(f"  inside_ratio_len:   {m['inside_ratio_len']:.4f}")
#     print(f"  outside_runs:       {m['outside_runs']}")
#     print(f"  max_outside_run:    {m['max_outside_run']:.4f}")

#     if "dt_mean(mm)" in m:
#         print(f"  dt_mean(mm):        {m['dt_mean(mm)']:.4f}")
#         print(f"  dt_median(mm):      {m['dt_median(mm)']:.4f}")
#         print(f"  dt_p95(mm):         {m['dt_p95(mm)']:.4f}")
#         print(f"  dt_max(mm):         {m['dt_max(mm)']:.4f}")
#         print(f"  within_r_ratio:     {m['within_r_ratio']:.4f}")

#     if "dt_inside_median" in m:
#         print(f"  dt_inside_median:   {m['dt_inside_median']:.4f}")
#         print(f"  dt_inside_p10:      {m['dt_inside_p10']:.4f}")



# -------------------------

# --------------------------
# fix_GT Section

def _polydata_to_polylines(pd: vtk.vtkPolyData) -> List[List[int]]:
    """Estrae le line cells come liste di pointIds."""
    out: List[List[int]] = []
    lines = pd.GetLines()
    if lines is None:
        return out
    lines.InitTraversal()
    ids = vtk.vtkIdList()
    while lines.GetNextCell(ids):
        n = ids.GetNumberOfIds()
        if n >= 2:
            out.append([int(ids.GetId(i)) for i in range(n)])
    return out


def _build_point_to_polyline_map(polylines: List[List[int]]) -> Dict[int, Set[int]]:
    """pointId -> set(indices polylines che lo contengono)"""
    m: Dict[int, Set[int]] = {}
    for li, pl in enumerate(polylines):
        for pid in pl:
            m.setdefault(pid, set()).add(li)
    return m


def _build_adjacency_from_polylines(polylines: List[List[int]]) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = {}
    for pl in polylines:
        for i in range(len(pl) - 1):
            a, b = pl[i], pl[i + 1]
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    return adj


def _degrees(adj: Dict[int, Set[int]]) -> Dict[int, int]:
    return {u: len(v) for u, v in adj.items()}


def _vtk_points_to_numpy(pts: vtk.vtkPoints) -> np.ndarray:
    n = pts.GetNumberOfPoints()
    arr = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        arr[i] = pts.GetPoint(i)
    return arr


def _clean_polydata_merge_points(pd: vtk.vtkPolyData, eps: float) -> vtk.vtkPolyData:
    """
    Unisce punti entro eps (tolleranza assoluta in unità del VTP, es. mm).
    """
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(pd)
    clean.PointMergingOn()

    # IMPORTANTISSIMO: eps deve essere in unità fisiche (mm), quindi ABSOLUTE
    clean.ToleranceIsAbsoluteOn()
    clean.SetAbsoluteTolerance(eps)

    clean.Update()
    return clean.GetOutput()



def connect_gt_bifurcations(
    gt_pd: vtk.vtkPolyData,
    eps: float,
    try_segment_insertion: bool = True,
    max_links_per_endpoint: int = 1,
) -> vtk.vtkPolyData:
    """
    Ricostruisce connessioni nella GT:
    1) merge/snap di punti entro eps (vtkCleanPolyData)
    2) opzionale: per endpoint ancora "isolati", connette a punto più vicino su un segmento
       (inserendo un punto sulla polyline target), se dist <= eps.

    Parametri:
    - eps: soglia in unità VTP (nel tuo caso quasi certamente mm). Tipico: ~ 1 voxel.
    - try_segment_insertion: se True, prova endpoint->segment (anche se non esiste vertice).
    - max_links_per_endpoint: normalmente 1 (un endpoint si attacca a un solo punto).

    Ritorna:
    - nuova vtkPolyData con lines aggiornate e possibili junction reali.
    """
    # --- Step 1: merge dei punti vicini (snap)
    pd = _clean_polydata_merge_points(gt_pd, eps=eps)

    if not try_segment_insertion:
        return pd

    # --- Costruisci struttura polylines / adjacency / endpoint list
    pts = pd.GetPoints()
    polylines = _polydata_to_polylines(pd)
    if not polylines:
        return pd

    adj = _build_adjacency_from_polylines(polylines)
    deg = _degrees(adj)

    # endpoint = degree 1
    endpoints = [pid for pid, d in deg.items() if d == 1]
    if not endpoints:
        return pd

    # mappa pointId -> polylines in cui appare (serve per evitare self-attach banali)
    p2pl = _build_point_to_polyline_map(polylines)

    # locator sulle CELLE (segmenti) della GT (post-clean)
    cell_loc = vtk.vtkCellLocator()
    cell_loc.SetDataSet(pd)
    cell_loc.BuildLocator()

    # per ricostruire nuove polylines, lavoriamo su copie modificabili
    new_polylines = [pl[:] for pl in polylines]

    # utility per aggiungere un punto (ritorna new pointId)
    def add_point(xyz: Tuple[float, float, float]) -> int:
        return int(pts.InsertNextPoint(float(xyz[0]), float(xyz[1]), float(xyz[2])))

    # utility: inserisci pid_new nella polyline cell_id in posizione (sub_id + 1)
    # sub_id è l'indice del segmento: tra pl[sub_id] e pl[sub_id+1]
    def insert_point_in_polyline(polyline_index: int, sub_id: int, pid_new: int) -> None:
        pl = new_polylines[polyline_index]
        # inserisci tra sub_id e sub_id+1
        pl.insert(sub_id + 1, pid_new)

    # dobbiamo mappare cellId -> indice polyline (VTK cell ordering)
    # In PolyData, l'ordine delle line cells corrisponde al traversal di GetLines.
    # Quindi cell_id della locator è compatibile con l'indice in 'polylines' (nella pratica VTK lo è).
    # Per sicurezza, assumiamo questa corrispondenza.
    # (Se vuoi, posso aggiungere una mappa esplicita cellId->polyline_index.)

    # Loop endpoints
    for ep in endpoints:
        ep_xyz = pts.GetPoint(ep)

        links_done = 0
        while links_done < max_links_per_endpoint:
            closest = [0.0, 0.0, 0.0]
            cell_id = vtk.reference(0)
            sub_id = vtk.reference(0)
            dist2 = vtk.reference(0.0)

            cell_loc.FindClosestPoint(ep_xyz, closest, cell_id, sub_id, dist2)
            d = float(np.sqrt(dist2.get()))

            if d > eps:
                break

            cid = int(cell_id.get())
            sid = int(sub_id.get())

            # Se la cell trovata appartiene alla stessa polyline dell'endpoint,
            # spesso la distanza minima è 0 (si ri-attacca a sé stesso).
            # Proviamo a filtrare questo caso: se ep è in quella cella, skip.
            ep_polylines = p2pl.get(ep, set())
            if cid in ep_polylines:
                # Qui ci vorrebbe la "second closest". VTK non la dà direttamente.
                # Workaround semplice: non fare insertion in questo caso.
                break

            # Inserisci un nuovo punto sulla polyline target nel punto closest (sul segmento sid)
            pid_new = add_point(tuple(closest))

            # Inserisci pid_new nella polyline cid nel punto giusto
            insert_point_in_polyline(polyline_index=cid, sub_id=sid, pid_new=pid_new)

            # Ora connetti endpoint -> pid_new creando una nuova polyline di 2 punti (un edge)
            new_polylines.append([ep, pid_new])

            links_done += 1

    # --- Ricostruisci vtkPolyData finale
    out_pts = vtk.vtkPoints()
    out_pts.DeepCopy(pts)  # include anche nuovi punti inseriti

    out_lines = vtk.vtkCellArray()
    for pl in new_polylines:
        if len(pl) < 2:
            continue
        poly = vtk.vtkPolyLine()
        poly.GetPointIds().SetNumberOfIds(len(pl))
        for i, pid in enumerate(pl):
            poly.GetPointIds().SetId(i, int(pid))
        out_lines.InsertNextCell(poly)

    out = vtk.vtkPolyData()
    out.SetPoints(out_pts)
    out.SetLines(out_lines)

    # Un'ulteriore clean può aiutare a mergeare eventuali duplicati creati
    out = _clean_polydata_merge_points(out, eps=eps * 0.5)

    return out

# -------------------------- 



# -------------------------- 
# to_polyline Section

def build_adjacency_from_polydata(pd: vtk.vtkPolyData) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = {}
    lines = pd.GetLines()
    if lines is None:
        return adj

    lines.InitTraversal()
    id_list = vtk.vtkIdList()

    while lines.GetNextCell(id_list):
        n = id_list.GetNumberOfIds()
        if n < 2:
            continue
        for i in range(n - 1):
            a = int(id_list.GetId(i))
            b = int(id_list.GetId(i + 1))
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    return adj


def extract_paths_between_critical_nodes(adj: Dict[int, Set[int]]) -> Tuple[List[List[int]], Set[Tuple[int,int]]]:
    """
    Estrae cammini massimali tra nodi critici (degree != 2).
    Ritorna:
      - paths: lista di path come liste di vertex ids
      - visited_edges: set di edge visitati (u,v) undirected normalizzato
    """
    deg = {u: len(vs) for u, vs in adj.items()}
    critical = {u for u, d in deg.items() if d != 2}

    visited_edges: Set[Tuple[int, int]] = set()
    paths: List[List[int]] = []

    def edge_key(u: int, v: int) -> Tuple[int,int]:
        return (u, v) if u < v else (v, u)

    for u in critical:
        for v in adj.get(u, []):
            if edge_key(u, v) in visited_edges:
                continue

            path = [u, v]
            visited_edges.add(edge_key(u, v))

            prev = u
            cur = v
            while cur not in critical:
                # degree == 2: due vicini
                nbrs = list(adj[cur])
                nxt = nbrs[0] if nbrs[1] == prev else nbrs[1]
                if edge_key(cur, nxt) in visited_edges:
                    break
                path.append(nxt)
                visited_edges.add(edge_key(cur, nxt))
                prev, cur = cur, nxt

            paths.append(path)

    return paths, visited_edges


def extract_cycles_degree2(adj: Dict[int, Set[int]], visited_edges: Set[Tuple[int,int]]) -> List[List[int]]:
    """
    Estrae cicli rimasti (tipicamente componenti con tutti degree==2).
    Usa visited_edges per non duplicare.
    """
    deg = {u: len(vs) for u, vs in adj.items()}

    def edge_key(u: int, v: int) -> Tuple[int,int]:
        return (u, v) if u < v else (v, u)

    cycles: List[List[int]] = []
    seen_nodes: Set[int] = set()

    for start in adj.keys():
        if deg.get(start, 0) != 2:
            continue
        if start in seen_nodes:
            continue

        # prova a camminare finché torni al start
        nbrs = list(adj[start])
        if len(nbrs) != 2:
            continue

        # scegli una direzione
        prev = start
        cur = nbrs[0]
        cycle = [start, cur]
        seen_nodes.add(start)

        ok = True
        while True:
            seen_nodes.add(cur)
            nbrs_cur = list(adj[cur])
            if len(nbrs_cur) != 2:
                ok = False
                break
            nxt = nbrs_cur[0] if nbrs_cur[1] == prev else nbrs_cur[1]

            # se chiude il ciclo
            if nxt == start:
                # chiudi (opzionale: non ripetere start alla fine)
                # marca ultimo edge
                visited_edges.add(edge_key(cur, nxt))
                break

            # edge già visitato? allora potrebbe essere già coperto
            if edge_key(cur, nxt) in visited_edges:
                ok = False
                break

            visited_edges.add(edge_key(prev, cur))
            visited_edges.add(edge_key(cur, nxt))

            cycle.append(nxt)
            prev, cur = cur, nxt

            # sicurezza
            if len(cycle) > 10_000_000:
                ok = False
                break

        if ok and len(cycle) >= 3:
            cycles.append(cycle)

    return cycles


def build_polydata_from_paths(original_pd: vtk.vtkPolyData, paths: List[List[int]]) -> vtk.vtkPolyData:
    """
    Crea un nuovo vtkPolyData che usa GLI STESSI PUNTI (stesse coordinate e stesso indexing),
    ma sostituisce le lines con polylines "mergeate" secondo i path.
    """
    # copia punti (manteniamo stesso ordine, quindi gli id restano validi)
    new_pts = vtk.vtkPoints()
    new_pts.DeepCopy(original_pd.GetPoints())

    new_lines = vtk.vtkCellArray()
    for path in paths:
        if len(path) < 2:
            continue
        pl = vtk.vtkPolyLine()
        pl.GetPointIds().SetNumberOfIds(len(path))
        for i, pid in enumerate(path):
            pl.GetPointIds().SetId(i, int(pid))
        new_lines.InsertNextCell(pl)

    out = vtk.vtkPolyData()
    out.SetPoints(new_pts)
    out.SetLines(new_lines)
    return out


def write_vtp(pd: vtk.vtkPolyData, path: str) -> None:
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(path)
    w.SetInputData(pd)
    w.Write()


def merge_pred_edges_to_polylines(pred_pd: vtk.vtkPolyData) -> vtk.vtkPolyData:
    adj = build_adjacency_from_polydata(pred_pd)

    paths, visited = extract_paths_between_critical_nodes(adj)
    cycles = extract_cycles_degree2(adj, visited)

    merged_paths = paths + cycles
    merged_pd = build_polydata_from_paths(pred_pd, merged_paths)
    return merged_pd


# end to_polyline functions
# --------------------------



# --------------------------
# IO
# --------------------------

def read_vtp(path: str) -> vtk.vtkPolyData:
    r = vtk.vtkXMLPolyDataReader()
    r.SetFileName(path)
    r.Update()
    pd = r.GetOutput()
    if pd is None or pd.GetNumberOfPoints() == 0:
        raise ValueError(f"VTP vuoto o non letto: {path}")
    return pd


def polydata_bounds(pd: vtk.vtkPolyData) -> Tuple[float, float, float, float, float, float]:
    return pd.GetBounds()  # (xmin,xmax,ymin,ymax,zmin,zmax)


# def print_bounds(name: str, b: Tuple[float, float, float, float, float, float]) -> None:
#     xmin, xmax, ymin, ymax, zmin, zmax = b
#     print(f"{name} bounds: X[{xmin:.3f},{xmax:.3f}]  Y[{ymin:.3f},{ymax:.3f}]  Z[{zmin:.3f},{zmax:.3f}]")


# --------------------------
# Resampling
# --------------------------

def iter_polylines(pd: vtk.vtkPolyData) -> Iterable[np.ndarray]:
    """
    Itera su ogni cella lineare (polyline) e restituisce array Nx3 di punti in ordine.
    """
    lines = pd.GetLines()
    if lines is None:
        return
    lines.InitTraversal()
    id_list = vtk.vtkIdList()

    pts = pd.GetPoints()
    while lines.GetNextCell(id_list):
        n = id_list.GetNumberOfIds()
        if n < 2:
            continue
        arr = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            pid = id_list.GetId(i)
            arr[i] = pts.GetPoint(pid)
        yield arr


def polyline_length(arr: np.ndarray) -> float:
    if arr.shape[0] < 2:
        return 0.0
    diffs = arr[1:] - arr[:-1]
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def resample_polyline(arr: np.ndarray, step: float) -> np.ndarray:
    """
    Resampling di una singola polyline a passo fisso `step`.
    Mantiene sempre l'ultimo punto.
    """
    if arr.shape[0] < 2:
        return arr.copy()

    seg = arr[1:] - arr[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total <= 1e-12:
        # tutti i punti uguali
        return arr[:1].copy()

    # campioni: 0, step, 2*step, ..., total
    s_vals = np.arange(0.0, total, step, dtype=np.float64)
    if s_vals.size == 0 or s_vals[-1] < total:
        s_vals = np.concatenate([s_vals, [total]])

    out = np.zeros((s_vals.size, 3), dtype=np.float64)

    # Per ogni s, trova il segmento in cum
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



def resample_polydata_lines(pd: vtk.vtkPolyData, step: float) -> SampledLines:
    out_lines: List[np.ndarray] = []
    total_len = 0.0
    for arr in iter_polylines(pd):
        total_len += polyline_length(arr)
        rs = resample_polyline(arr, step)
        if rs.shape[0] >= 2:
            out_lines.append(rs)
    return SampledLines(lines=out_lines, total_length=total_len)


def sampled_to_polydata(sampled: SampledLines) -> vtk.vtkPolyData:
    """
    Converte SampledLines in vtkPolyData con polylines (utile per locator / debug).
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

    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetLines(lines)
    return pd


# --------------------------
# Distance: point -> polyline set (vtkCellLocator)
# --------------------------

def build_cell_locator(pd: vtk.vtkPolyData) -> vtk.vtkCellLocator:
    loc = vtk.vtkCellLocator()
    loc.SetDataSet(pd)
    loc.BuildLocator()
    return loc


def distances_points_to_polydata(points: np.ndarray, locator: vtk.vtkCellLocator, ref_pd: vtk.vtkPolyData) -> np.ndarray:
    """
    Distanza minima da ciascun punto (Nx3) al set di linee in ref_pd.
    Usa vtkCellLocator.FindClosestPoint.
    """
    out = np.zeros((points.shape[0],), dtype=np.float64)
    closest = [0.0, 0.0, 0.0]
    cell_id = vtk.reference(0)
    sub_id = vtk.reference(0)
    dist2 = vtk.reference(0.0)

    for i in range(points.shape[0]):
        p = points[i]
        locator.FindClosestPoint(p.tolist(), closest, cell_id, sub_id, dist2)
        out[i] = float(np.sqrt(dist2.get()))
    return out


def distances_sampledlines_to_polydata(sampled: SampledLines, ref_pd: vtk.vtkPolyData) -> np.ndarray:
    loc = build_cell_locator(ref_pd)
    all_d = []
    for arr in sampled.lines:
        d = distances_points_to_polydata(arr, loc, ref_pd)
        all_d.append(d)
    if not all_d:
        return np.array([], dtype=np.float64)
    return np.concatenate(all_d, axis=0)


def percentile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


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
    gt_pd_resampled: vtk.vtkPolyData,
    pred_pd_resampled: vtk.vtkPolyData,
    tau: float
) -> Tuple[float, float, float]:
    
    gt_len = gt_sampled.total_length
    pred_len = pred_sampled.total_length
    if gt_len <= 1e-12 or pred_len <= 1e-12:
        return float("nan"), float("nan"), float("nan")

    # Recall: GT covered by Pred
    gt_cov = covered_length_of_sampledlines(gt_sampled, pred_pd_resampled, tau)
    recall = gt_cov / gt_len

    # Precision: Pred supported by GT
    pred_cov = covered_length_of_sampledlines(pred_sampled, gt_pd_resampled, tau)
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


def topology_stats_from_adj(adj: Dict[int, Set[int]]) -> TopologyStats:
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
        degree_hist=dict(sorted(hist.items()))
    )


# --------------------------
# 4.B Segment extraction (between critical nodes degree != 2)
# --------------------------

@dataclass
class Segment:
    vertex_ids: List[int]


def extract_segments_between_critical_nodes(adj: Dict[int, Set[int]]) -> List[Segment]:
    """
    Estrae segmenti come cammini massimali tra nodi critici (degree != 2).
    Evita duplicati marcando gli edge visitati.
    """
    deg = {u: len(v) for u, v in adj.items()}
    critical = {u for u, d in deg.items() if d != 2}

    visited_edges: Set[Tuple[int, int]] = set()
    segments: List[Segment] = []

    def mark_edge(u: int, v: int) -> None:
        visited_edges.add((u, v))
        visited_edges.add((v, u))

    for u in critical:
        for v in adj.get(u, []):
            if (u, v) in visited_edges:
                continue

            path = [u, v]
            mark_edge(u, v)

            prev = u
            cur = v
            while cur not in critical:
                # degree==2 quindi esistono esattamente 2 vicini
                nbrs = list(adj[cur])
                nxt = nbrs[0] if nbrs[1] == prev else nbrs[1]
                if (cur, nxt) in visited_edges:
                    break
                path.append(nxt)
                mark_edge(cur, nxt)
                prev, cur = cur, nxt

            segments.append(Segment(vertex_ids=path))

    return segments


# --------------------------
# Quick alignment sanity check
# --------------------------

def coarse_alignment_check(gt_pd: vtk.vtkPolyData, pred_pd: vtk.vtkPolyData, n_samples: int = 2000) -> float:
    """
    Campiona punti casuali dalla pred e misura distanza media alla GT.
    Se è enorme, probabilmente non sono nello stesso spazio.
    """
    pred_pts = pred_pd.GetPoints()
    n = pred_pd.GetNumberOfPoints()
    idx = np.random.choice(n, size=min(n_samples, n), replace=False)

    pts = np.zeros((idx.size, 3), dtype=np.float64)
    for i, pid in enumerate(idx):
        pts[i] = pred_pts.GetPoint(int(pid))

    loc = build_cell_locator(gt_pd)
    d = distances_points_to_polydata(pts, loc, gt_pd)
    return float(np.mean(d)) if d.size else float("nan")


# --------------------------
# Utils
# --------------------------

def case_suffix(case_id: Optional[int]) -> str:
    if case_id is None:
        return ""
    return f"_{case_id:03d}"

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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
    print(f"  step={R['step']}  tau={R['tau']}  eps={R['eps']}  r_mm={R.get('r_mm','n/a')}")
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
            print(f"  outside_runs:       {int(M['outside_runs'])}")
            print(f"  max_outside_run:    {fmt(M['max_outside_run'],4)}")
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
    print(f"  Pred: nodes={tp['nodes']} edges={tp['edges']}")
    print(f"        endpoints={tp['endpoints']} junctions={tp['junctions']}")
    print(f"        degree_hist={tp['degree_hist']}")

    print(f"  GT  : nodes={tg['nodes']} edges={tg['edges']}")
    print(f"        endpoints={tg['endpoints']} junctions={tg['junctions']}")
    print(f"        degree_hist={tg['degree_hist']}")
    if tg["junctions"] == 0:
        print("        NOTE: GT junctions=0 likely means junctions are not encoded as shared pointIds in the VTP.")

    print("\n[7] 4.B SEGMENTS (Pred)")
    s = R["seg4b"]
    print(f"  #segments: {s['n_segments']}")
    if s["n_segments"] > 0:
        print(f"  segment vertex-count: mean={fmt(s['mean'],2)}  median={fmt(s['median'],2)}  max={s['max']}")

    print("\n" + "=" * 70)



# --------------------------
# Main eval
# --------------------------

def evaluate(
    gt_path: str,
    pred_path: str,
    step: float,
    tau: float,
    eps: float,
    to_polyline: bool,
    fix_GT: bool,
    mask_path: Optional[str],
    r_mm: float,
    outdir: str,
    save_vtp: bool,
    case_id: Optional[int],
) -> None:
    gt_pd = read_vtp(gt_path)
    pred_pd = read_vtp(pred_path)

    outdir_p = Path(outdir)
    if save_vtp:
        ensure_dir(outdir_p)
    suf = case_suffix(case_id)

    # --- optional: merge pred edges -> polylines
    if to_polyline:
        pred_pd_merged = merge_pred_edges_to_polylines(pred_pd)
        if save_vtp:
            write_vtp(pred_pd_merged, str(outdir_p / f"PredMergedToPoly{suf}.vtp"))
        pred_pd = pred_pd_merged

    # --- optional: fix GT (currently only snap/merge points)
    if fix_GT:
        gt_pd_fixed = connect_gt_bifurcations(gt_pd, eps=eps, try_segment_insertion=False)
        if save_vtp:
            write_vtp(gt_pd_fixed, str(outdir_p / f"GT_Fixed{suf}.vtp"))
        gt_pd = gt_pd_fixed

    # 0) bounds + coarse check
    mean_coarse = coarse_alignment_check(gt_pd, pred_pd)

    # 1) resampling
    gt_s = resample_polydata_lines(gt_pd, step=step)
    pred_s = resample_polydata_lines(pred_pd, step=step)

    gt_pd_rs = sampled_to_polydata(gt_s)
    pred_pd_rs = sampled_to_polydata(pred_s)

    if save_vtp:
        write_vtp(gt_pd_rs, str(outdir_p / f"GT_Resampled_step{step:g}{suf}.vtp"))
        write_vtp(pred_pd_rs, str(outdir_p / f"Pred_Resampled_step{step:g}{suf}.vtp"))

    # ---- Segmentation containment (optional) ----
    if mask_path is not None:
        zooms = nib.load(mask_path).header.get_zooms()[:3]
        m_gt = centerline_vs_mask_metrics(gt_s, mask_path, r_mm=r_mm)
        m_pr = centerline_vs_mask_metrics(pred_s, mask_path, r_mm=r_mm)

        # print_mask_metrics("GT vs Segmentation", m_gt)
        # print_mask_metrics("Pred vs Segmentation", m_pr)

        n_gt = node_inside_outside_ratio(gt_s, mask_path)
        n_pr = node_inside_outside_ratio(pred_s, mask_path)

        # print("\nGT node containment:")
        # print(f"  inside_nodes: {n_gt['inside_nodes']} / {n_gt['total_nodes']}  ({n_gt['inside_ratio_nodes']:.4f})")
        # print(f"  outside_nodes:{n_gt['outside_nodes']} / {n_gt['total_nodes']}  ({n_gt['outside_ratio_nodes']:.4f})")

        # print("\nPred node containment:")
        # print(f"  inside_nodes: {n_pr['inside_nodes']} / {n_pr['total_nodes']}  ({n_pr['inside_ratio_nodes']:.4f})")
        # print(f"  outside_nodes:{n_pr['outside_nodes']} / {n_pr['total_nodes']}  ({n_pr['outside_ratio_nodes']:.4f})")
    

    # 2) geometric distances (bidirectional)
    d_pred_to_gt = distances_sampledlines_to_polydata(pred_s, gt_pd_rs)
    d_gt_to_pred = distances_sampledlines_to_polydata(gt_s, pred_pd_rs)

    # def summarize(name: str, d: np.ndarray) -> None:
    #     if d.size == 0:
    #         print(f"{name}: empty")
    #         return
    #     print(f"{name}: mean={np.mean(d):.4f}  median={np.median(d):.4f}  p95={percentile(d,95):.4f}  max={np.max(d):.4f}")

    # summarize("\nDistances pred->GT", d_pred_to_gt)
    # summarize("Distances GT->pred", d_gt_to_pred)

    assd = 0.5 * (float(np.mean(d_pred_to_gt)) + float(np.mean(d_gt_to_pred))) if (d_pred_to_gt.size and d_gt_to_pred.size) else float("nan")
    hd95 = max(percentile(d_pred_to_gt, 95), percentile(d_gt_to_pred, 95))

    # 3) coverage metrics on length
    precision, recall, f1 = precision_recall_f1(gt_s, pred_s, gt_pd_rs, pred_pd_rs, tau=tau)
    # print(f"Precision_tau (tau={tau}): {precision:.4f}")
    # print(f"Recall_tau    (tau={tau}): {recall:.4f}")
    # print(f"F1_tau        (tau={tau}): {f1:.4f}")

    # coverage lengths (needed for report)
    gt_cov = covered_length_of_sampledlines(gt_s, pred_pd_rs, tau)
    pred_cov = covered_length_of_sampledlines(pred_s, gt_pd_rs, tau)

    # 4.A topology stats
    adj_pred = build_adjacency_from_polydata(pred_pd)
    ts_pred = topology_stats_from_adj(adj_pred)
    # print("\nTopology (Pred):")
    # print(f"  nodes={ts_pred.n_nodes}  edges={ts_pred.n_edges}")
    # print(f"  endpoints={ts_pred.n_endpoints}  junctions={ts_pred.n_junctions}")
    # print(f"  degree_hist={ts_pred.degree_hist}")

    adj_gt = build_adjacency_from_polydata(gt_pd)
    ts_gt = topology_stats_from_adj(adj_gt)
    # print("\nTopology (GT):")
    # print(f"  nodes={ts_gt.n_nodes}  edges={ts_gt.n_edges}")
    # print(f"  endpoints={ts_gt.n_endpoints}  junctions={ts_gt.n_junctions}")
    # print(f"  degree_hist={ts_gt.degree_hist}")

    # 4.B segments
    segs = extract_segments_between_critical_nodes(adj_pred)
    # print(f"\n4.B Pred segments between critical nodes: {len(segs)}")
    if segs:
        lens = [len(s.vertex_ids) for s in segs]
        # print(f"  segment vertex-count: mean={np.mean(lens):.2f}  median={np.median(lens):.2f}  max={np.max(lens)}")
        
    # --- FINAL REPORT ---
    R = {
        "case_tag": f"(case_{case_id:03d})" if case_id is not None else "",
        "step": step,
        "tau": tau,
        "eps": eps,
        "r_mm": r_mm,
        "voxel_spacing": zooms if mask_path is not None else None,
        "gt_bounds": bounds_str(polydata_bounds(gt_pd)),
        "pred_bounds": bounds_str(polydata_bounds(pred_pd)),
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
        "seg4b": {
            "n_segments": len(segs),
            "mean": float(np.mean(lens)) if segs else 0.0,
            "median": float(np.median(lens)) if segs else 0.0,
            "max": int(np.max(lens)) if segs else 0,
        },
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
    ap.add_argument("--eps", type=float, default=0.8, help="Tolerance eps for GT fixing / point merging (same units as VTP)")
    ap.add_argument("--rmm", type=float, default=1.0, help="Distance threshold (mm) used by within_r_ratio in segmentation metrics")

    # switches
    ap.add_argument("--to-polyline", action="store_true", help="Merge pred edges into polylines before eval")
    ap.add_argument("--fix-gt", action="store_true", help="Try to fix GT (snap/merge points) before eval")
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
        eps=args.eps,
        to_polyline=args.to_polyline,
        fix_GT=args.fix_gt,
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
    else:
        # Hard-coded quick run
        case_id = 3
        gt = f"C:/Users/ducci/Documents/Università_2025/6_SemesterProject/BrainGraph/data/ITKTubeTK_GoldStandardVtp/VascularNetwork-{case_id:03d}.vtp"
        pred = f"C:/Users/ducci/Documents/Università_2025/6_SemesterProject/BrainGraph/output/Output_basic_extractor/BasicCenterline_{case_id:03d}.vtp"
        mask = f"C:/Users/ducci/Documents/Università_2025/6_SemesterProject/BrainGraph/data/ITKTubeTK_ManualSegmentationNii/labels-{case_id:03d}.nii.gz"
        
        # pred = f"C:/Users/ducci/Documents/Università_2025/6_SemesterProject/BrainGraph/output/Output_prova/ExCenterline_{case_id:03d}.vtp"
        # pred = f"C:/Users/ducci/Documents/Università_2025/6_SemesterProject/BrainGraph/output/Output_bogdan/vessel_graph_{case_id:03d}.vtp"
        
        evaluate(
            gt_path=gt,
            pred_path=pred,
            step=0.5,
            tau=0.5,
            eps=0.8,
            to_polyline=True,
            fix_GT=False,
            mask_path=mask,
            r_mm=1.0,
            outdir=r"C:\Users\ducci\Documents\Università_2025\6_SemesterProject\BrainGraph\output\Output_evaluations",
            save_vtp=True,
            case_id=case_id,
        )
