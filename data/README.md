## Data folder structure

Whenever we refer to the `data/` directory, we assume the following internal structure:

BrainGraph/
├── data/
│ ├── ITKTubeTK_ManualSegmentationNii/
│ │ ├── labels-001.nii.gz
│ │ ├── labels-002.nii.gz
│ │ └── labels-003.nii.gz
│ │
│ ├── output/labels-00x/ (automatically generated)
│ │ ├── vessel_data.pkl <-- NetworkX Graph + metadata
│ │ └── vessel_graph_aligned.vtp <-- PolyData for Slicer/visualization
│ │
│ ├── ITKTubeTK_GoldStandardVtp/ (FOR EVALUATION: gold standard centerlines)
│ │ ├── VascularNetwork-002.vtp
│ │
│ ├── CaravelCenterlines/ (FOR EVALUATION: OLD SOLUTION centerlines)
│ │ ├── labels-001.vtp
│ │ ├── labels-002.vtp
│ │ └── labels-003.vtp
│ │
│ ├── outputCenterlinesOnly/ (FOR EVALUATION: all VTPs collected into a single folder)
│ │ ├── vessel_graph_aligned_001.vtp (same as vessel_graph_aligned + numeric suffix)
│ │ ├── vessel_graph_aligned_002.vtp
│ │ └── vessel_graph_aligned_003.vtp
