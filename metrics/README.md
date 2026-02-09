# Batch Centerline Evaluation (VTP)

This script runs a **batch evaluation** of predicted centerlines (`.vtp`) against ground-truth centerlines (`.vtp`), producing per-case metrics and aggregated outputs.

It is designed for **centerline polydata with `Lines`** (polyline cells), and computes:
- geometric distance metrics between predicted and GT centerlines
- coverage-based precision/recall/F1 (length-based)
- topology statistics (nodes/edges/endpoints/junctions/degree histogram)
- smoothness via **angular tortuosity** ("turning-angle" statistics)

---

## What the script does

1. **Load** GT and Pred `.vtp`
2. **Adjust** the Polydata format
3. **Resample** polylines with a fixed step
4. Convert resampled lines into:
   - **segment format** (each edge is a 2-point line cell) for distance & coverage metrics
5. Compute metrics and save results as:
   - a global `results.jsonl` (one JSON per case)
   - a global `results.csv` (flattened JSON)
   - optional per-case JSON files under `out_dir/per_case_json/`

---

## Inputs

### Required
- `--pred_dir`: directory containing **predicted** `.vtp` files (in our case: `data/CaravelCenterlines/` for the old centerlines and `data/outputCenterlinesOnly/` for the new ones)
- `--gt_dir`: directory containing **ground-truth** `.vtp` files (in our case: `data/ITKTubeTK_GoldStandardVtp/`)
- `--out_dir`: output directory where results will be written

### Optional
- `--step` (float, default `0.3`): resampling step along the polyline
- `--tau` (float, default `0.5134`): distance threshold for coverage metrics (precision/recall/F1)
- `--recursive`: search `.vtp` files recursively in input directories
- `--pad` (int, default `3`): case-id zero padding (e.g. `003`)
- `--use_static`: use `vtkStaticCellLocator` when available (usually faster)
- `--per_case_json` (default ON): save a JSON file per case in `out_dir/per_case_json/`
- `--no_per_case_json`: disable per-case JSON saving

---

## Outputs

All outputs are written to `--out_dir`.

### 1) `results.jsonl`
A JSON Lines file with **one JSON object per case**.

Path:
- `out_dir/results.jsonl`

Each line contains the full metrics dictionary (JSON-serializable).

### 2) `results.csv`
A flattened CSV version of the JSON metrics (nested keys become dotted paths, e.g. `topo_pred.nodes`).

Path:
- `out_dir/results.csv`

### 3) Per-case JSON (optional)
If enabled (default), the script writes:
- `out_dir/per_case_json/case_XXX.json`

---

## Metrics included (high level)

- **Geometric distances** (segment-format):
  - `assd` (average symmetric surface distance)
  - `hd95` (symmetric 95th percentile Hausdorff-like)
  - `pred2gt_mean`, `pred2gt_median`, `pred2gt_p95`, `pred2gt_max`
  - `gt2pred_mean`, `gt2pred_median`, `gt2pred_p95`, `gt2pred_max`

- **Coverage metrics (length-based)** using threshold `tau`:
  - `precision`, `recall`, `f1`
  - `gt_cov`, `pred_cov`
  - `gt_len`, `pred_len`

- **Topology**:
  - number of nodes/edges/endpoints/junctions
  - degree histogram

- **Smoothness / Angular tortuosity**:
  - `total_turn_rad`
  - `turn_per_mm`
  - `rms_turn_deg`
  - `p95_turn_deg`
  - `gt_p95_exceedance` (fraction of turns exceeding GT p95 threshold)

---

## How to run

### Minimal example
```bash
python metrics/batch_eval_metrics.py \
  --pred_dir "data/CaravelCenterlines" \
  --gt_dir "data/ITKTubeTK_GoldStandardVtp" \
  --out_dir "metrics/results" \
  --step 0.3 \
  --tau 0.5134
```

The `--pred_dir` can also be: `data/outputCenterlinesOnly`, based on the centerlines we want to test against the GT.
All outputs are written to `metrics/results` folder.
