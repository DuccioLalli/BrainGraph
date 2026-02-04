import vedo
import nibabel as nib
import numpy as np
import networkx as nx
import itertools
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, label, center_of_mass, map_coordinates
from scipy.spatial import cKDTree
from nibabel.affines import apply_affine

# ---------------- CONFIGURATION ----------------
input_nifti = r"D:\Academic Stuff\BrainGraph\data\ITKTubeTK_ManualSegmentationNii\labels-003.nii.gz"

# PARAMS
INTENSITY_THRESHOLD = 0.6

# ORPHAN SETTINGS
ORPHAN_DISTANCE_THRESHOLD = 2.5
MERGE_DISTANCE = 3.0
MIN_ANGLE = 30.0
K_NEIGHBORS = 5


# -----------------------------------------------

def calculate_angle(p_center, p1, p2):
    v1 = p1 - p_center
    v2 = p2 - p_center
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    dot = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


print(f"Loading NIFTI: {input_nifti}")
nii = nib.load(input_nifti)
data = nii.get_fdata()
affine = nii.affine
inv_affine = np.linalg.inv(affine)
voxel_sizes = nib.affines.voxel_sizes(affine)

# 1. MESH & SKELETON
print("Generating Mesh & Skeleton...")
vol = vedo.Volume(data)
mesh = vol.isosurface(value=0.5).apply_transform(affine)
mesh.clean()  # Keeping it clean but NOT decimated

skeleton_mask = skeletonize(data > 0)
skel_indices = np.argwhere(skeleton_mask)
skel_points_world = apply_affine(affine, skel_indices)

# 2. ORPHANS (Find & Merge)
print(f"Finding Orphans (> {ORPHAN_DISTANCE_THRESHOLD}mm)...")
dist_to_skel = distance_transform_edt(~skeleton_mask, sampling=voxel_sizes)
orphan_mask = (data > 0) & (dist_to_skel > ORPHAN_DISTANCE_THRESHOLD)
labeled_orphans, num_zones = label(orphan_mask)

orphan_points = []
for z in range(1, num_zones + 1):
    cm = center_of_mass(orphan_mask, labeled_orphans, z)
    orphan_points.append(apply_affine(affine, np.array(cm)))
orphan_points_world = np.array(orphan_points)

if len(orphan_points_world) > 0:
    tree = cKDTree(orphan_points_world)
    pairs = tree.query_pairs(r=MERGE_DISTANCE)
    g_tmp = nx.Graph()
    g_tmp.add_nodes_from(range(len(orphan_points_world)))
    g_tmp.add_edges_from(pairs)
    merged_indices = []
    for component in nx.connected_components(g_tmp):
        comp_list = list(component)
        cluster_coords = orphan_points_world[comp_list]
        centroid = np.mean(cluster_coords, axis=0)
        closest_idx = np.argmin(np.linalg.norm(cluster_coords - centroid, axis=1))
        merged_indices.append(comp_list[closest_idx])
    orphan_points_world = orphan_points_world[merged_indices] if merged_indices else np.array([])

# 3. BUILD GRAPH
print("Building Graph...")
G = nx.Graph()
for i, pt in enumerate(skel_points_world):
    G.add_node(i, pos=pt, type='skeleton')

offset = len(skel_points_world)
for i, pt in enumerate(orphan_points_world):
    G.add_node(offset + i, pos=pt, type='orphan')

idx_map = {tuple(p): i for i, p in enumerate(skel_indices)}
offsets_26 = [np.array([z, y, x]) for z in (-1, 0, 1) for y in (-1, 0, 1) for x in (-1, 0, 1) if
              not (z == 0 and y == 0 and x == 0)]
for i, curr in enumerate(skel_indices):
    for off in offsets_26:
        neigh = tuple(curr + off)
        if neigh in idx_map and idx_map[neigh] > i:
            G.add_edge(i, idx_map[neigh])

# 4. CONNECT ORPHANS
print("Connecting Orphans...")
nodes_to_remove = []
viz_candidate_lines = []
viz_bridge_lines = []

if len(orphan_points_world) > 0:
    skel_tree = cKDTree(skel_points_world)
    for i, opos in enumerate(orphan_points_world):
        oid = offset + i
        dists, idxs = skel_tree.query(opos, k=K_NEIGHBORS)

        # Candidate visualization
        if K_NEIGHBORS == 1: dists, idxs = [dists], [idxs]
        for j, c_idx in enumerate(idxs):
            if dists[j] != float('inf'):
                viz_candidate_lines.append([opos, skel_points_world[c_idx]])

        valid_candidates = []
        try:
            osurf = mesh.closest_point(opos, return_point_id=True)
            for j, c_idx in enumerate(idxs):
                if dists[j] == float('inf'): continue
                csurf = mesh.closest_point(skel_points_world[c_idx], return_point_id=True)
                path = mesh.geodesic(osurf, csurf)
                if path and len(path.vertices) > 0:
                    geo_len = np.sum(np.linalg.norm(np.diff(path.vertices, axis=0), axis=1))
                    valid_candidates.append({'dist': geo_len, 'idx': c_idx, 'pos': skel_points_world[c_idx]})
        except:
            pass

        is_bridge = False
        if len(valid_candidates) > 1:
            for cand_a, cand_b in itertools.combinations(valid_candidates, 2):
                if calculate_angle(opos, cand_a['pos'], cand_b['pos']) > MIN_ANGLE:
                    is_bridge = True
                    break
        if is_bridge:
            for cand in valid_candidates: viz_bridge_lines.append([opos, cand['pos']])
            nodes_to_remove.append(oid)
        elif len(valid_candidates) > 0:
            valid_candidates.sort(key=lambda x: x['dist'])
            G.add_edge(oid, valid_candidates[0]['idx'])
        else:
            nodes_to_remove.append(oid)

G.remove_nodes_from(nodes_to_remove)

# 5. PRUNING
print("Pruning...")
edges_to_remove = set()
viz_geodesic_paths = []

for tri in [c for c in nx.enumerate_all_cliques(G) if len(c) == 3]:
    edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
    longest = max(edges, key=lambda e: np.linalg.norm(G.nodes[e[0]]['pos'] - G.nodes[e[1]]['pos']))
    edges_to_remove.add(tuple(sorted(longest)))

step = np.min(voxel_sizes) * 0.5
for u, v in G.edges():
    if tuple(sorted((u, v))) in edges_to_remove: continue
    p1, p2 = G.nodes[u]['pos'], G.nodes[v]['pos']
    dist = np.linalg.norm(p2 - p1)
    pts = p1 + np.outer(np.linspace(0, 1, max(2, int(dist / step) + 1)), p2 - p1)
    vals = map_coordinates(data, apply_affine(inv_affine, pts).T, order=1)
    if np.any(vals < INTENSITY_THRESHOLD):
        try:
            id1, id2 = mesh.closest_point(p1, return_point_id=True), mesh.closest_point(p2, return_point_id=True)
            path_obj = mesh.geodesic(id1, id2)
            if path_obj and len(path_obj.vertices) > 0:
                geo_dist = np.sum(np.linalg.norm(np.diff(path_obj.vertices, axis=0), axis=1))
                if (geo_dist / dist) > 2.0:
                    viz_geodesic_paths.append(vedo.Line(path_obj.vertices).c('red').lw(4))
                    edges_to_remove.add(tuple(sorted((u, v))))
                else:
                    viz_geodesic_paths.append(vedo.Line(path_obj.vertices).c('green').lw(4))
            else:
                edges_to_remove.add(tuple(sorted((u, v))))
        except:
            edges_to_remove.add(tuple(sorted((u, v))))

G.remove_edges_from(edges_to_remove)

# 6. RAW DISPLAY (No Smoothing)
print("\nVISUALIZATION (RAW COORDINATES):")
actors = [mesh.alpha(0.1).c('grey')]

skel_c = [G.nodes[n]['pos'] for n in G.nodes() if G.nodes[n].get('type') == 'skeleton']
orph_c = [G.nodes[n]['pos'] for n in G.nodes() if G.nodes[n].get('type') == 'orphan']
if skel_c: actors.append(vedo.Points(skel_c).c('blue').ps(4))
if orph_c: actors.append(vedo.Points(orph_c).c('magenta').ps(8))

# Draw final edges from G directly
raw_edges_s = [G.nodes[u]['pos'] for u, v in G.edges()]
raw_edges_e = [G.nodes[v]['pos'] for u, v in G.edges()]
if raw_edges_s:
    actors.append(vedo.Lines(raw_edges_s, raw_edges_e).c('yellow').lw(3))

if viz_geodesic_paths: actors.extend(viz_geodesic_paths)
if viz_candidate_lines:
    s, e = [p[0] for p in viz_candidate_lines], [p[1] for p in viz_candidate_lines]
    actors.append(vedo.Lines(s, e).c('cyan').lw(1).alpha(0.3))
if viz_bridge_lines:
    s, e = [p[0] for p in viz_bridge_lines], [p[1] for p in viz_bridge_lines]
    actors.append(vedo.Lines(s, e).c('orange').lw(4))

vedo.show(actors, axes=1, title="Raw Graph (No Smoothing)")