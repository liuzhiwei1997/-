# sfrt_lattice_planner

`sfrt_lattice_planner` is a Python 3.10+ research implementation of an SFRT (spatially fractionated radiation therapy) sub-target lattice placement workflow inspired by the patent-described “空间分割放疗靶区布置方法和系统”. It is **not** a deep-learning model: the pipeline combines DICOM/RTSTRUCT parsing, distance-transform mask construction, three-dimensional HCP close packing, and simulated-annealing rigid optimization.

> **Research-use warning:** This tool is only for scientific reproduction and contour-generation assistance. It does not calculate dose, DVH, VMAT/IMRT plans, or peak-to-valley dose ratio. Any peak/valley evaluation must be performed after importing contours into a treatment planning system (TPS) and completing independent clinical QA.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Command-line usage

```bash
python -m sfrt_lattice_planner.cli run \
  --ct-dir ./data/case001/CT \
  --rtstruct ./data/case001/RTSTRUCT.dcm \
  --tumor-roi GTV \
  --oar-rois SpinalCord Skin Larynx \
  --tumor-inset-mm 20 \
  --oar-clearance-mm 5 \
  --output-dir ./outputs/case001 \
  --anneal-iterations 1000 \
  --volume-tolerance 0.98
```

Optional arguments include `--center-mm X Y Z` for a user-defined first lattice center, `--max-outer-steps`, `--seed`, and `--export-rtstruct` for best-effort RTSTRUCT export.

## Algorithm overview

1. Read CT DICOM series and RTSTRUCT ROIs with `SimpleITK` and `rt-utils`.
2. Convert the requested tumor ROI and OAR ROI list into 3-D binary masks while preserving CT spacing, origin, and direction metadata.
3. Compute tumor volume in cc:

   ```text
   V = tumor_voxels * spacing_x * spacing_y * spacing_z / 1000
   ```

4. Apply the patent formulas:

   ```python
   if V < 50:
       d_sphere_cm = 0.5
       d_lattice_cm = 2.0
   elif 50 <= V < 1000:
       d_sphere_cm = 0.001 * V + 0.4258
       d_lattice_cm = 0.002 * V + 1.9
   else:
       d_sphere_cm = 1.5
       d_lattice_cm = 4.0
   ```

5. Build a placement mask using physical distance transforms:
   - `tumor_inner = distance_transform_edt(tumor_mask) >= tumor_inset_mm`
   - `oar_safe = distance_transform_edt(~oar_union) >= oar_clearance_mm`
   - `placement_mask = tumor_inner & oar_safe`
6. Select the first HCP unit center from the placement-mask geometric center, or the nearest valid voxel if the centroid is not inside the placement mask.
7. Generate an HCP close-packing lattice using basis vectors:

   ```text
   a1 = [d, 0, 0]
   a2 = [0.5*d, sqrt(3)/2*d, 0]
   a3 = [0.5*d, sqrt(3)/6*d, sqrt(2/3)*d]
   ```

8. Retain candidates whose corresponding sphere fits inside the placement mask according to the placement-mask distance transform.
9. Increase lattice spacing by 1 mm outer steps and run simulated annealing over whole-lattice translations and rotations. The larger spacing is accepted only when retained total sphere volume is at least `volume_tolerance * V0`.

## Outputs

The output directory contains:

- `centers_initial.csv`: initial accepted sphere centers in physical mm coordinates.
- `centers_optimized.csv`: optimized sphere centers in physical mm coordinates.
- `sfrt_union_mask.nii.gz`: union mask of optimized SFRT sub-target spheres.
- `qa_report.json`: QA metrics and run parameters.
- `sfrt_rtstruct.dcm`: optional best-effort RTSTRUCT export when `--export-rtstruct` is supplied.

RTSTRUCT ROI names are `SFRT_UNION`, `SFRT_001`, `SFRT_002`, ... . To inspect in a TPS, import the generated RTSTRUCT together with the original CT study, verify frame-of-reference alignment, overlay every generated ROI slice-by-slice, and independently check distance to tumor boundary and OARs before any downstream planning work.

## QA report fields

`qa_report.json` includes tumor and placement volumes, sphere diameter, initial/optimized lattice spacing, initial/optimized sphere counts and total sphere volume, volume retention ratio, minimum center-to-center distance, minimum distance to tumor boundary, minimum distance to OARs, and all parameters used.

## Tests

```bash
pytest
```
