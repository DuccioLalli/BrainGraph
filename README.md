# BrainGraph

## If you want to read more about the metrics, there is a separate readme on that in the metrics folder.

## 📂 Repository Structure

* **`/extraction`**: Core processing engines.
    * `vedo_extractor_batch.py`: **(Main Script)** The production-ready batch processor. Extracts smoothed, aligned, and validated graphs from NIfTI masks.
    * `laplace_extractor.py`: Our Week1 solution (just basic extractor smoothed basically)
    * `batch_Caravel_Centerline_extractor.py`: Initial solution before us. Command to use: `python .\extraction\batch_Caravel_Centerline_extractor --in_dir "...\data\ITKTubeTK_ManualSegmentationNii" --out_dir "...\data\CaravelCenterlines" --connectivity 26 --overwrite --prune`. **If you want to use this, please use shift_coordinate_system as well afterwards.**.
* **`/extra`**: Utility and diagnostic tools.
`
    * `visualization_graph_full.py`: Very extremely helpful visualizer. If you feel like something is wrong with your graph or if you want to tune parameters or if you want to see exactly how connections are being done, run the segmentation via this.
    * `pkl_to_vtp.py`: Utility to convert exported Pickle (`.pkl`) graphs into PolyData (`.vtp`) format for use in 3D Slicer(or other visualizers).
    * `shift_coordinate_system`: Utility to shift from old system (used by Caravel) to our system.

---

## 🚀 The Core: `vedo_extractor_batch.py`

This is the primary pipeline for converting NIfTI segmentations into brain graphs. Unlike standard skeletonization methods that often produce "jagged" or disconnected results, this script employs **Geodesic Validation** and **Anatomical Bridge Logic**.

### How it Works
1.  **Skeletonization**: Uses `skimage.morphology` to extract the initial centerlines.
2.  **Orphan Recovery**: Identifies "orphaned" vessel segments (missed voxels) and reintegrates them using `cKDTree` spatial lookups.
3.  **Anatomical Bridge Logic**: Prevents false connections between parallel vessels. If a potential connection forms a wide angle (>30°) between existing branches, it is flagged as a "bridge" and pruned.
4.  **Geodesic Pruning**: Measures the distance between nodes along the actual vessel surface (mesh). If the graph edge takes a "shortcut" through empty space, it is deleted.
5.  **Laplacian Smoothing**: Iteratively adjusts node positions to remove voxel-grid artifacts, creating smooth, organic vessel paths.
  ## WARNING 
6. **Slicer Alignment**: Applies a 4x4 transformation matrix to ensure the graph aligns perfectly with the original MRI volume in RAS/LPS coordinate systems. THIS DEPENDS ON THE DATASET. You might need it, you might not. It works for 00-Manual and IXI nnUnet.

---

## 📊 Data Structure Requirements

The batch extractor expects a specific directory structure. Place your NIfTI segmentations in a source folder; the script will generate a corresponding output folder for the graphs.

```text
BrainGraph/
├── data/
│   ├── ITKTubeTK_ManualSegmentationNii/
│   │   ├── labels-001.nii.gz
│   │   ├── labels-002.nii.gz
│   │   └── labels-003.nii.gz

│   ├── output/ (Automatically generated)
│   │   ├── vessel_data.pkl              <-- NetworkX Graph + Metadata
│   │   └── vessel_graph_aligned.vtp     <-- PolyData for Slicer/Visualization

│   ├── ITKTubeTK_GoldStandardVtp/         (gold standard centerlines)
│   │   ├── VascularNetwork-002.vtp

│   ├── CaravelCenterlines/                (OLD SOLUTION centerlines)
│   │    ├── labels-001.vtp
│   │    ├── labels-002.vtp
│   │    ├── labels-003.vtp    

│   ├── outputCenterlinesOnly/           (it's the same as output folder from above, but all the VTPs combined into a single folder)
│   │   ├── vessel_graph_aligned_001.vtp         (same vessel_graph_aligned + numerical suffix)
│   │   ├── vessel_graph_aligned_002.vtp
│   │   ├── vessel_graph_aligned_003.vtp
```
## 🔍 Debugging & Visualization

To inspect the algorithm's decision-making process, run:
`python extra/visualization_graph_full.py`

This script provides a color-coded 3D environment to help you tune your parameters:

* **Blue Dots**: Standard skeleton nodes.
* **Magenta Dots**: Recovered orphan nodes (vessel tips).
* **Yellow Lines**: Final validated graph edges (Raw Voxel Coordinates).
* **Orange Lines**: Deleted Bridges (failed the anatomical angle check).
* **Red Lines**: Deleted Geodesic Shortcuts (edges cutting outside the vessel wall).

---

## 🛠️ Installation (personal recommandation, use a venv)

The following dependencies are required:

```bash
pip install vedo nibabel numpy networkx scikit-image scipy
```
## ⚙️ Key Parameters

Located in the `CONFIGURATION` section of the scripts. **Important**: these SHOULD be messed with, we don't promise the current values are the BEST.

* **MIN_COMPONENT_SIZE**: Components made up of less than 7 skeleton points are not taken into account. (Again, 7 has not been proved to be the best choice).
* **INTENSITY_THRESHOLD**: Sensitivity for geodesic path validation (default `0.6`). Lower values are more permissive of gaps.
* **MIN_ANGLE**: Angle threshold for bridge detection (default `30.0`). Prevents false "ladder" connections between parallel vessels.
* **K**: How many nodes an orphan tries to make connections with. (default `5`). If you increase this, computational time increases.
* **ORPHAN_DISTANCE_THRESHOLD** The max distance of the K nodes mentioned above compared to the orphan node. (default `2.5` mm).
* **MERGE_DISTANCE** Spatial tolerance used to clean up redundant orphans. If they are too close to eachother, it means they were poorly chosen, so they are clustered together then deleted. (default `3.0` mm).
* **SMOOTH_ITERS**: Number of Laplacian smoothing passes (default `2`). Higher values produce smoother lines but can shrink tight curves.
* **SMOOTH_ALPHA**: Power of how strong the Laplacian algorithm smoothes in one pass (default `0.8`).

---
There are more parameters in the code (or some things should be parameterized perhaps). Personal recommendation, the logic of edges being created from an **orphan** to a **candidate** should also include some distance check. Something like, if the edges to the **k candidates** should be within `0.5` mm with respect to the **shortest possible edge**. If an edge is longer than that, it shouldn't be taken into account.

## 🛠️ Troubleshooting

* **Graph appears mirrored in Slicer**: Ensure the `SLICER_MATRIX` in `vedo_extractor_batch.py` matches your volume's orientation.

---

## 📦 Output Details

The pipeline generates two primary files per scan:

1.  **.pkl (Pickle)**: A Python-serialized NetworkX object. Contains the full graph topology, world-space coordinates, and a `node_radius_map` for vessel thickness.
2.  **.vtp (VTK PolyData)**: A 3D model format compatible with 3D Slicer and ParaView. Ideal for visual overlay with the original MRI/CT volume.
