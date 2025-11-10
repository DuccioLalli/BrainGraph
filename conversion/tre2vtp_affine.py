import re
import vtk
import os
import argparse
import numpy as np
import nibabel as nib


def read_amira_tre(filename):
    tubes = []
    current_points = []
    inside_points = False

    # robust open (in case of odd encodings)
    with open(filename, encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()

            # Start of a new tube
            if stripped.startswith("ObjectType = Tube"):
                if current_points:
                    tubes.append(current_points)
                    current_points = []
                inside_points = False

            # Start of point block
            elif stripped.startswith("Points"):
                inside_points = True
                continue

            # End of object/group
            elif stripped.startswith("EndGroup") or stripped.startswith("ObjectType ="):
                if current_points:
                    tubes.append(current_points)
                    current_points = []
                inside_points = False

            # Point data
            elif inside_points and stripped:
                # stop if line is not numeric
                if not re.match(r"^[0-9\.\-eE\s]+$", stripped):
                    inside_points = False
                    continue
                vals = [float(v) for v in stripped.split()]
                if len(vals) >= 4:
                    current_points.append(vals[:4])

    if current_points:
        tubes.append(current_points)
    return tubes


def write_vtp(tubes, output_file, affine=None):
    """
    Write polyline VTP. If 'affine' is provided (4x4), apply it to every (x,y,z,1).
    """
    out_dir = os.path.dirname(os.path.abspath(output_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    radii = vtk.vtkFloatArray()
    radii.SetName("Radius")

    pid = 0
    for t in tubes:
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(t))
        for i, (x, y, z, r) in enumerate(t):
            if affine is not None:
                p = np.array([x, y, z, 1.0], dtype=float)
                p = affine @ p
                x, y, z = p[:3]
            points.InsertNextPoint(float(x), float(y), float(z))
            radii.InsertNextValue(float(r))
            line.GetPointIds().SetId(i, pid)
            pid += 1
        lines.InsertNextCell(line)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    polydata.GetPointData().AddArray(radii)
    polydata.GetPointData().SetScalars(radii)  # handy: Radius becomes active scalar

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(polydata)
    writer.SetDataModeToBinary()  # smaller file
    ok = writer.Write()
    if ok == 0:
        raise IOError(f"Error writing '{output_file}' (check that the folder exists and path is valid).")


def main():
    ap = argparse.ArgumentParser(
        description="Convert Amira .tre to VTK .vtp (polyline with per-point Radius). "
                    "Optionally apply a reference NIfTI affine to place the result in world (mm) coordinates."
    )
    ap.add_argument("input", help="Path to Amira .tre file")
    ap.add_argument("-o", "--output", default="Output.vtp", help="Output .vtp path")
    ap.add_argument("--ref-nifti", help="Path to reference NIfTI (.nii/.nii.gz) whose affine will be applied")
    args = ap.parse_args()

    # Optional affine from NIfTI
    affine = None
    if args.ref_nifti:
        img = nib.load(args.ref_nifti)
        affine = img.affine  # 4x4

    tubes = read_amira_tre(args.input)
    if not tubes:
        raise ValueError("No tubes found. Check that the .tre format matches the parser (ObjectType = Tube / Points blocks).")

    write_vtp(tubes, args.output, affine=affine)
    print(f"Converted {len(tubes)} tubes to '{args.output}'"
          + (f" using affine from '{args.ref_nifti}'" if affine is not None else ""))

if __name__ == "__main__":
    main()
