"""Print the learning snapshots of the Study I pilot (results_pilot_i/)."""
import glob
import json
import os

KEYS = ["0", "0.05", "0.1", "0.25", "0.5", "1", "2", "3"]
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_pilot_i", "*.json"))):
    d = json.load(open(f))
    print(os.path.basename(f))
    for s in d["snapshots"]:
        print("   ", s["step"], [s["lev"][k] for k in KEYS], s["seconds"])
