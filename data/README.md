## Data folder structure

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

│   ├── CaravelCenterlines/                (OLD SOLUTION centerlines)
│   │    ├── labels-001.vtp
│   │    ├── labels-002.vtp
│   │    ├── labels-003.vtp    

│   ├── outputCenterlinesOnly/           (it's the same as output folder from above, but all the VTPs combined into a single folder)
│   │   ├── vessel_graph_aligned_001.vtp         (same vessel_graph_aligned + numerical suffix)
│   │   ├── vessel_graph_aligned_002.vtp
│   │   ├── vessel_graph_aligned_003.vtp
```
