"""Command-line interface for sfrt_lattice_planner."""
from __future__ import annotations

import argparse
from pathlib import Path

from .export import export_rtstruct, write_centers_csv, write_json, write_nifti_mask
from .geometry import lattice_parameters_from_volume, mask_volume_cc
from .io_dicom import load_case
from .masks import choose_origin_center, make_placement_mask, sphere_masks
from .optimize import optimize_lattice_spacing
from .qa import build_qa_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sfrt_lattice_planner", description="Research SFRT lattice sub-target placement from CT DICOM + RTSTRUCT.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run the SFRT lattice planner.")
    run.add_argument("--ct-dir", required=True)
    run.add_argument("--rtstruct", required=True)
    run.add_argument("--tumor-roi", required=True)
    run.add_argument("--oar-rois", nargs="*", default=[])
    run.add_argument("--tumor-inset-mm", type=float, default=20.0)
    run.add_argument("--oar-clearance-mm", type=float, default=5.0)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--anneal-iterations", type=int, default=1000)
    run.add_argument("--volume-tolerance", type=float, default=0.98)
    run.add_argument("--max-outer-steps", type=int, default=20)
    run.add_argument("--center-mm", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Optional custom first lattice center in physical mm coordinates.")
    run.add_argument("--seed", type=int, default=13)
    run.add_argument("--export-rtstruct", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict:
    case = load_case(args.ct_dir, args.rtstruct, args.tumor_roi, args.oar_rois)
    tumor_volume_cc = mask_volume_cc(case.tumor_mask, case.spacing_mm)
    params = lattice_parameters_from_volume(tumor_volume_cc)
    placement = make_placement_mask(
        case.tumor_mask,
        case.oar_masks,
        case.spacing_mm,
        tumor_inset_mm=args.tumor_inset_mm,
        oar_clearance_mm=args.oar_clearance_mm,
    )
    first_center = choose_origin_center(placement.placement_mask, case.spacing_mm, case.origin_mm, args.center_mm)
    result = optimize_lattice_spacing(
        placement.placement_mask,
        first_center,
        params.lattice_spacing_mm,
        params.radius_mm,
        case.spacing_mm,
        image_origin_mm=case.origin_mm,
        volume_tolerance=args.volume_tolerance,
        max_outer_steps=args.max_outer_steps,
        anneal_iterations=args.anneal_iterations,
        seed=args.seed,
    )
    union_mask, individual_masks = sphere_masks(case.tumor_mask.shape, result.optimized_centers_mm, params.radius_mm, case.spacing_mm, case.origin_mm)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_centers_csv(out / "centers_initial.csv", result.initial_centers_mm)
    write_centers_csv(out / "centers_optimized.csv", result.optimized_centers_mm)
    write_nifti_mask(out / "sfrt_union_mask.nii.gz", union_mask, reference_image=case.ct_image)
    cli_parameters = {
        "tumor_roi": args.tumor_roi,
        "oar_rois": args.oar_rois,
        "tumor_inset_mm": args.tumor_inset_mm,
        "oar_clearance_mm": args.oar_clearance_mm,
        "volume_tolerance": args.volume_tolerance,
        "max_outer_steps": args.max_outer_steps,
        "anneal_iterations": args.anneal_iterations,
        "seed": args.seed,
        "first_center_mm": [float(x) for x in first_center],
    }
    qa = build_qa_report(
        case.tumor_mask,
        placement.placement_mask,
        placement.oar_union,
        result.initial_centers_mm,
        result.optimized_centers_mm,
        case.spacing_mm,
        params.sphere_diameter_mm,
        result.initial_spacing_mm,
        result.optimized_spacing_mm,
        result.initial_total_volume_cc,
        result.optimized_total_volume_cc,
        origin_mm=case.origin_mm,
        parameters=cli_parameters,
    )
    write_json(out / "qa_report.json", qa)
    if args.export_rtstruct:
        export_rtstruct(args.ct_dir, args.rtstruct, out / "sfrt_rtstruct.dcm", union_mask, individual_masks)
    return qa


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        qa = run(args)
        print(f"Optimized {qa['optimized_number_of_spheres']} spheres at {qa['optimized_lattice_spacing_mm']:.2f} mm spacing.")
        print(f"QA report: {Path(args.output_dir) / 'qa_report.json'}")


if __name__ == "__main__":
    main()
