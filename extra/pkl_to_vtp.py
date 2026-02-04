import pickle
import vedo
import numpy as np

# ---------------- CONFIG ----------------
INPUT_PKL = r"D:\Academic Stuff\BrainGraph\data\vessel_data_shifted.pkl"
OUTPUT_VTP = r"D:\Academic Stuff\BrainGraph\data\vessel_graph_for_slicer.vtp"
# ----------------------------------------

print(f"Loading {INPUT_PKL}...")
with open(INPUT_PKL, 'rb') as f:
    data = pickle.load(f)

# Extract graph
G = data['graph']

# 1. Get Edge Lines
# We build a list of start/end points for every edge
start_points = []
end_points = []

print("Extracting edges...")
for u, v in G.edges():
    start_points.append(G.nodes[u]['pos'])
    end_points.append(G.nodes[v]['pos'])

# 2. Create the Visual Object
if start_points:
    # Create lines (green, thickness 2)
    lines = vedo.Lines(start_points, end_points).c('green').lw(2)

    # 3. Save as VTP
    print(f"Saving to {OUTPUT_VTP}...")
    lines.write(OUTPUT_VTP)
    print("Success! You can now drag this .vtp file into 3D Slicer.")
else:
    print("Error: Graph has no edges.")