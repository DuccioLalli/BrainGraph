# Data folder structure

Whenever we refer to the `data/` directory, we assume the following internal structure:

```text
BrainGraph/
├── data/
│   ├── ITKTubeTK_ManualSegmentationNii/
│   │   ├── labels-001.nii.gz
│   │   ├── labels-002.nii.gz
│   │   └── labels-003.nii.gz

│   ├── output/labels-00x                (Automatically generated)
│   │   ├── vessel_data.pkl              <-- NetworkX Graph + Metadata
│   │   └── vessel_graph_aligned.vtp     <-- PolyData for Slicer/Visualization


(FOR THE EVALUATION PART)

│   ├── ITKTubeTK_GoldStandardVtp/         (gold standard centerlines)
│   │   ├── VascularNetwork-002.vtp
│   │   ├── VascularNetwork-003.vtp
│   │   ├── VascularNetwork-004.vtp

│   ├── CaravelCenterlines/                (OLD SOLUTION centerlines)
│   │    ├── labels-002.vtp
│   │    ├── labels-003.vtp
│   │    ├── labels-004.vtp    

│   ├── outputCenterlinesOnly/           (it's the same as output folder from above, but all the VTPs combined into a single folder)
│   │   ├── vessel_graph_aligned_002.vtp         (same vessel_graph_aligned + numerical suffix)
│   │   ├── vessel_graph_aligned_003.vtp
│   │   ├── vessel_graph_aligned_004.vtp
```

---

Notes:
- `data/output/labels-00x/` is created by `vedo_extractor_batch`.
- `outputCenterlinesOnly/` is a convenience folder used to collect all extracted `.vtp` centerlines in one place for evaluation.

---

### `copy_grouped_centerlines.py`

This script gathers centerline files from multiple `labels-*` folders into a single directory.

### What it does
- Scans `data/output/` for subfolders named `labels-<suffix>`
- For each folder, looks for `vessel_graph_aligned.vtp`
- Copies it to `data/outputCenterlinesOnly/` as:
  - `vessel_graph_aligned_<suffix>.vtp`
- Prints `[MISSING]` if a folder does not contain the expected file
- **Overwrites** destination files if they already exist


