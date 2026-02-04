import os
import glob
import vedo
import numpy as np

# =========================================================================
#                               CONFIGURATION
# =========================================================================

# 1. PATHS
# Where your original (misaligned) Gold Standard files are
INPUT_VTP_FOLDER = r"D:\Academic Stuff\BrainGraph\data\ITKTubeTK_GoldStandardVtp"

# Where you want to save the corrected files
OUTPUT_FIXED_VTP_FOLDER = r"D:\Academic Stuff\BrainGraph\data\drr"

# 2. THE ALIGNMENT MATRIX (The "Slicer Flip")
# This matrix flips the X and Y coordinates to fix RAS/LPS orientation issues
SLICER_MATRIX = np.array([
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])


# =========================================================================
#                          FIXING FUNCTION
# =========================================================================

def batch_fix_vtps():
    print(f"\n{'=' * 60}")
    print(f"TASK: ALIGNMENT CORRECTION FOR GOLD STANDARD VTPs")
    print(f"{'=' * 60}")

    # Check if input folder exists
    if not os.path.exists(INPUT_VTP_FOLDER):
        print(f"ERROR: Input folder not found at: {INPUT_VTP_FOLDER}")
        return

    # Create output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FIXED_VTP_FOLDER):
        os.makedirs(OUTPUT_FIXED_VTP_FOLDER)
        print(f"Created output directory: {OUTPUT_FIXED_VTP_FOLDER}")

    # Find all .vtp files
    vtp_files = glob.glob(os.path.join(INPUT_VTP_FOLDER, "*.vtp"))

    if not vtp_files:
        print(f"No .vtp files found in {INPUT_VTP_FOLDER}")
        return

    print(f"Found {len(vtp_files)} files. Applying transformation matrix...")

    success_count = 0
    for vtp_path in vtp_files:
        filename = os.path.basename(vtp_path)
        # Create new filename to avoid overwriting originals
        save_name = filename.replace(".vtp", "_FIXED.vtp")
        save_path = os.path.join(OUTPUT_FIXED_VTP_FOLDER, save_name)

        try:
            # 1. Load the VTP file as a mesh/polydata object
            vtp_obj = vedo.load(vtp_path)

            # 2. Apply the 4x4 Slicer Alignment Matrix
            vtp_obj.apply_transform(SLICER_MATRIX)

            # 3. Save the corrected file
            vtp_obj.write(save_path)

            print(f"  [OK] Processed: {filename} -> {save_name}")
            success_count += 1

        except Exception as e:
            print(f"  [ERR] Failed to process {filename}: {e}")

    print(f"\n{'=' * 60}")
    print(f"FINISHED: {success_count} files successfully fixed.")
    print(f"{'=' * 60}")


# =========================================================================
#                                  MAIN
# =========================================================================

if __name__ == "__main__":
    batch_fix_vtps()