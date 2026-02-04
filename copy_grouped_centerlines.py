from pathlib import Path
import shutil

# === CONFIG ===
root = Path("data")
src_root = root / "output"
dst_root = root / "outputCenterlinesOnly"

# === SCRIPT ===
dst_root.mkdir(parents=True, exist_ok=True)

copied = 0
missing = 0

for d in sorted(p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("labels-")):
    suffix = d.name.split("-", 1)[1]  # everything after "labels-"
    src_file = d / "vessel_graph_aligned.vtp"

    if not src_file.exists():
        print(f"[MISSING] {src_file}")
        missing += 1
        continue

    dst_file = dst_root / f"vessel_graph_aligned_{suffix}.vtp"
    print(f"[COPY] {src_file} -> {dst_file}")

    # copy2 preserves metadata; overwrites if dst_file exists
    shutil.copy2(src_file, dst_file)
    copied += 1

print("\n=== REPORT ===")
print(f"Copied:       {copied}")
print(f"Missing:      {missing}")
print(f"Destination:  {dst_root.resolve()}")

