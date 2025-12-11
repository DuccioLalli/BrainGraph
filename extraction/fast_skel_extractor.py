# Optimized Version
import time
start = time.perf_counter()

import pyvista as pv
import numpy as np
import nibabel as nib
import trimesh
import skeletor as sk


def save_mask_vtp(mask, affine, out_path):
    print("4. Saving segmentation surface to VTP...")

    # Convert mask to float for PyVista
    vol = mask.astype(np.float32)

    # Wrap into PyVista grid
    grid = pv.wrap(vol)

    # Extract surface
    surf = grid.contour(isosurfaces=[0.5])
    surf = surf.smooth_taubin(n_iter=50, pass_band=0.1)

    # Apply affine to put surface in world coordinates
    pts = surf.points
    homo = np.c_[pts, np.ones(len(pts))]
    pts_w = (homo @ affine.T)[:, :3]

    # Replace points in the surface
    surf.points = pts_w

    surf.save(out_path)
    
    
n = 2
input_path = f"C:/Users/ducci/Documents/Università_2025/6_SemesterProject/BrainGraph/data/ITKTubeTK_ManualSegmentationNii/labels-00{n}.nii.gz"

save_centerline = True
save_segmentation = True
debug = True

output_centerline_vtp = f"C:/Users/ducci/Documents/Università_2025/6_SemesterProject/BrainGraph/output/Output_prova/ExCenterline_00{n}.vtp"
output_segmentation_vtp = f"C:/Users/ducci/Documents/Università_2025/6_SemesterProject/BrainGraph/output/Output_prova/ExSegmentation_00{n}.vtp"


# 1. Load mask (fast + clean)
print("1. Loading the mask")
img = nib.load(input_path)
spacing = img.header.get_zooms()  # (sx, sy, sz)
arr = img.get_fdata(dtype=np.float32) 
affine = img.affine

mask = arr > 0
print(f"-- Mask loaded. Shape: {mask.shape}, Voxels: {mask.sum()}")
print("-- Voxel size (mm):", spacing)

# 2. Extract mesh surface (fast)
print("2. Extracting the skeleton")

# Direct wrap without copying
grid = pv.wrap(mask)

# OPTIMIZED: no triangulate() unless needed
surf = grid.contour(isosurfaces=0.5)

surf = surf.smooth_taubin(n_iter=30, pass_band=0.1)
surf.plot(color="white")

# Extract mesh data to convert in trimesh format
vertices = surf.points
faces = surf.faces.reshape(-1, 4)[:, 1:]

# Convertion to Trimesh
tm_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False) 

# 3. Skeletonize
skel = sk.skeletonize.by_wavefront(
    tm_mesh,
    waves=1,           # num waves
    step_size=2,
    radius_agg="mean", # radius
    progress=True      # bar
)

# post processing
sk.post.clean_up(skel, mesh=tm_mesh, inplace=True)

print(f"-- Skeleton extracted: {len(skel.vertices)} nodes, {len(skel.edges)} edges")

if debug:
    skel.show(mesh=True)



# 4. Save to VTP
if save_centerline:
    print("3. Saving centerline to VTP")

    pts = skel.vertices
    edges = skel.edges

    # Apply affine
    homo = np.c_[pts, np.ones(len(pts))]
    pts_w = (homo @ affine.T)[:, :3]

    # Build lines
    lines = np.hstack([[2, u, v] for u, v in edges])

    # Build polydata
    poly = pv.PolyData(pts_w)
    poly.lines = lines

    poly.save(output_centerline_vtp)
    print(f"-- Centerline saved to: {output_centerline_vtp}")


if save_segmentation:
    save_mask_vtp(mask, affine, output_segmentation_vtp)
    print(f"-- Segmentation saved to: {output_segmentation_vtp}")
    

end = time.perf_counter()
print(f"Elapsed_time: {end-start:.4f} sec")