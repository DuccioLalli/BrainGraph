import re
import vtk
import os
import argparse


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


def write_vtp(tubes, output_file):
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


def main():
    ap = argparse.ArgumentParser(description="Convert Amira .tre to VTK .vtp (polyline with per-point Radius).")
    ap.add_argument("input", help="Path to Amira .tre file")
    ap.add_argument("-o", "--output", default="Output.vtp", help="Output .vtp path")
    args = ap.parse_args()

    tubes = read_amira_tre(args.input)
    if not tubes:
        raise ValueError("No tubes found. Check that the .tre format matches the parser (ObjectType = Tube / Points blocks).")
    write_vtp(tubes, args.output)
    print(f"Converted {len(tubes)} tubes to '{args.output}'")

if __name__ == "__main__":
    main()
