#!/usr/bin/env python

import argparse
import nibabel as nib
import pyvista as pv
import sys
import os


def transform_vtp_to_physical(nifti_path, vtp_input_path, vtp_output_path):
    """
    Transforms a VTP file from voxel coordinates to physical coordinates
    using the affine matrix from a NIfTI file.

    Args:
        nifti_path (str): Path to the reference NIfTI file.
        vtp_input_path (str): Path to the VTP file in voxel space.
        vtp_output_path (str): Path to save the transformed VTP file.

    Returns:
        bool: True on success, False on failure.
    """

    # --- 1. Load NIfTI file and get affine matrix ---
    print(f"Loading NIfTI file: {nifti_path}")
    try:
        nii = nib.load(nifti_path)

        # get_sform() is standard for "scanner" or "physical" space.
        # It's the matrix that maps voxel indices (i,j,k) to (x,y,z) mm.
        affine = nii.get_sform()

        if affine is None:
            print("Warning: sform matrix not found. Trying qform...")
            affine = nii.get_qform()

        if affine is None:
            print("Error: No valid sform or qform found in NIfTI header.", file=sys.stderr)
            print("Cannot determine physical coordinate space.", file=sys.stderr)
            return False

        print("Successfully read NIfTI affine matrix:")
        print(affine)

    except Exception as e:
        print(f"Error loading NIfTI file: {e}", file=sys.stderr)
        return False

    # --- 2. Load the VTP file to be transformed ---
    print(f"\nLoading VTP file: {vtp_input_path}")
    try:
        # This is your skeleton.vtp file
        mesh = pv.read(vtp_input_path)
        print(f"Loaded mesh with {mesh.n_points} points.")
        # print(f"Original bounds (voxel space): {mesh.bounds}")

    except Exception as e:
        print(f"Error loading VTP file: {e}", file=sys.stderr)
        return False

    # --- 3. Apply the Transformation ---
    print("Applying affine transformation to all points...")
    mesh.transform(affine, inplace=True)
    print("Transformation applied.")
    # print(f"New bounds (physical space): {mesh.bounds}")

    # --- 4. Save the New, Transformed VTP File ---
    try:
        mesh.save(vtp_output_path)
        print(f"\n✅ Successfully saved transformed VTP to: {vtp_output_path}")
        return True
    except Exception as e:
        print(f"Error saving new VTP file: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="""
        Converts a VTP file from Voxel Space to Physical Space.

        This script uses the sform/qform affine matrix from a NIfTI file
        to transform the points of a VTP (e.g., a skeleton) so it
        can be overlaid on data already in physical space (e.g., an annotation).
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-n", "--nifti",
        required=True,
        help="Path to the original NIfTI file (.nii or .nii.gz) that defines the physical space."
    )
    parser.add_argument(
        "-i", "--input_vtp",
        required=True,
        help="Path to the input VTP file that is currently in Voxel Space (e.g., skeleton.vtp)."
    )
    parser.add_argument(
        "-o", "--output_vtp",
        required=True,
        help="Path to save the new, transformed VTP file (e.g., skeleton_in_physical_space.vtp)."
    )

    args = parser.parse_args()

    # --- Input validation ---
    if not os.path.exists(args.nifti):
        print(f"Error: NIfTI file not found at {args.nifti}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.input_vtp):
        print(f"Error: Input VTP file not found at {args.input_vtp}", file=sys.stderr)
        sys.exit(1)

    print("--- VTP Coordinate Transformation Script ---")
    success = transform_vtp_to_physical(args.nifti, args.input_vtp, args.output_vtp)

    if success:
        print("\nDone.")
    else:
        print("\nScript finished with errors.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()