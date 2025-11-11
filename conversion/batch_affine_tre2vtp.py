import os
import re
import vtk
import argparse
import numpy as np
import nibabel as nib

def read_amira_tre(filename):
    tubes = []
    current_points = []
    inside_points = False

    with open(filename) as f:
        for line in f:
            stripped = line.strip()

            # Start of a new tube
            if stripped.startswith("ObjectType = Tube"):
                if current_points:
                    tubes.append(current_points)
                    current_points = []
                inside_points = False

            elif stripped.startswith("Points"):
                inside_points = True
                continue

            elif stripped.startswith("EndGroup") or stripped.startswith("ObjectType ="):
                if current_points:
                    tubes.append(current_points)
                    current_points = []
                inside_points = False

            elif inside_points and stripped:
                if not re.match(r"^[0-9.\-eE\s]+$", stripped):
                    inside_points = False
                    continue
                vals = [float(v) for v in stripped.split()]
                if len(vals) >= 4:
                    current_points.append(vals[:4])

    if current_points:
        tubes.append(current_points)
    return tubes


def load_affine_from_label(nifti_folder, number):
    """Load affine from labels-<number>.nii.gz or labels-<number>.nii"""
    # Pattern: labels-001.nii.gz or labels-001.nii
    possible_files = [
        f"labels-{number}.nii.gz",
        f"labels-{number}.nii"
    ]
    for f in possible_files:
        path = os.path.join(nifti_folder, f)
        if os.path.exists(path):
            img = nib.load(path)
            print(f"✅ Found affine for {number}: {f}")
            return img.affine
    raise FileNotFoundError(
        f"❌ Could not find NIfTI for number {number} in {nifti_folder}"
    )


def apply_affine_to_tubes(tubes, affine):
    """Apply affine transformation to all tube coordinates."""
    transformed_tubes = []
    for t in tubes:
        points = np.array([p[:3] for p in t])
        radii = np.array([p[3] for p in t])
        homogeneous = np.hstack((points, np.ones((len(points), 1))))
        transformed_points = homogeneous.dot(affine.T)[:, :3]
        transformed_tubes.append([[x, y, z, r] for (x, y, z), r in zip(transformed_points, radii)])
    return transformed_tubes


def write_vtp(tubes, output_file):
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    radii = vtk.vtkFloatArray()
    radii.SetName("Radius")

    pid = 0
    for t in tubes:
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(t))
        for i, (x, y, z, r) in enumerate(t):
            points.InsertNextPoint(x, y, z)
            radii.InsertNextValue(r)
            line.GetPointIds().SetId(i, pid)
            pid += 1
        lines.InsertNextCell(line)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    polydata.GetPointData().AddArray(radii)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(polydata)
    writer.Write()


def extract_number(filename):
    """Extract the numeric suffix (e.g., '001') from a filename like 'VascularNetwork-001.tre'."""
    match = re.search(r'(\d+)', filename)
    if not match:
        raise ValueError(f"Could not extract number from filename: {filename}")
    return match.group(1).zfill(3)  # zero-pad to 3 digits for safety


def main():
    parser = argparse.ArgumentParser(
        description="Convert .tre files to .vtp format with corresponding NIfTI affine applied."
    )
    parser.add_argument(
        "-i", "--input", default="data/ITKTubeTK_GoldStandardTre",
        help="Input folder containing .tre files (default: data/ITKTubeTK_GoldStandardTre)"
    )
    parser.add_argument(
        "-n", "--nifti_folder", default="data/ITKTubeTK_ManualSegmentationNii",
        help="Folder containing corresponding NIfTI files (default: ITKTubeTK_ManualSegmentationNii)"
    )
    parser.add_argument(
        "-o", "--output", default="data/ITKTubeTK_GoldStandardVtp",
        help="Output folder for .vtp files (default: data/ITKTubeTK_GoldStandardVtp)"
    )
    args = parser.parse_args()

    input_dir = args.input
    nifti_dir = args.nifti_folder
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    tre_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".tre")]
    if not tre_files:
        print(f"No .tre files found in '{input_dir}'.")
        return

    for tre_file in tre_files:
        input_path = os.path.join(input_dir, tre_file)
        number = extract_number(tre_file)
        output_path = os.path.join(output_dir, os.path.splitext(tre_file)[0] + ".vtp")

        print(f"\nProcessing {tre_file} → {os.path.basename(output_path)}")

        # Find and load corresponding NIfTI affine
        try:
            affine = load_affine_from_label(nifti_dir, number)
        except FileNotFoundError as e:
            print(e)
            continue

        # Read, transform, and save
        tubes = read_amira_tre(input_path)
        tubes_transformed = apply_affine_to_tubes(tubes, affine)
        write_vtp(tubes_transformed, output_path)

        print(f"  ✅ {len(tubes)} tubes written to {output_path}")

    print("\nAll conversions complete.")


if __name__ == "__main__":
    main()
