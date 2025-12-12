import subprocess
import os
import re

# ----------------------------------------------------------------------
# Paths – adjust GT path if needed
# ----------------------------------------------------------------------

# OPTION 1: GT files in the m3ed layout (as in your bash script):
#   ${BASE_DIR}/${SCENE}/${SCENE}_pose_evo_gt.txt
gt_base_path = os.path.expanduser(
    "~/m3ed/m3ed/{seq}/{seq}_pose_evo_gt.txt"
)

# If instead you still use the old GT files like M3ED_{sequence}_gt.txt, use this:
# gt_base_path = os.path.expanduser("~/OGAM/deio_pp/report/traj/M3ED_{}_gt.txt")

# Your saved trajectories directory
traj_base_dir = "/home/bora/s-deio/saved_trajectories"

# Label for the config used to generate trajectories
CONFIG_NAME = "superfast"

# Number of trials: 0, 1, ..., 9
NUM_TRIALS = 10

# ----------------------------------------------------------------------
# Time windows per sequence
# ----------------------------------------------------------------------
time_windows = {
    "spot_forest_road_1": (4, 146),
    "spot_forest_hard": (3, 98),
    "spot_indoor_building_loop": (4, 101),
    "spot_indoor_obstacles": (4, 81),
    "spot_indoor_stairs": (9, 90),
    "spot_indoor_stairwell": (3, 80),
    "spot_outdoor_day_skatepark_1": (7, 90),
    "spot_outdoor_day_penno_short_loop": (5, 109),
    "spot_outdoor_day_srt_green_loop": (4, 56),
    "spot_outdoor_day_srt_under_bridge_1": (15, 194),
    "spot_outdoor_day_srt_under_bridge_2": (6, 175),
    "spot_outdoor_night_penno_plaza_lights": (5, 74),
    "spot_outdoor_night_penno_short_loop": (7, 113),
    "spot_forest_easy_1": (5, 69),
}

# ----------------------------------------------------------------------
# Output CSV
# ----------------------------------------------------------------------
output_file = "evo_results_new_structure.csv"

with open(output_file, "w") as f:
    f.write("Sequence,Config,Trial,RMSE (m)\n")

rmse_pattern = re.compile(r"^\s*rmse\s+([\d.]+)", re.MULTILINE)

print("Starting EVO APE runs with new file structure...")

for sequence, (t_start, t_end) in time_windows.items():
    print(f"\nProcessing sequence: {sequence}")

    # GT path (m3ed layout)
    gt_path = gt_base_path.format(seq=sequence)
    # If using the old layout instead, with gt_base_path = "...M3ED_{}_gt.txt":
    # gt_path = gt_base_path.format(sequence)

    if not os.path.exists(gt_path):
        print(f"  - WARNING: GT file not found: {gt_path}. Skipping this sequence.")
        continue

    for trial_index in range(NUM_TRIALS):
        # NON-ZERO-PADDED: 0, 1, 2, ..., 9
        trial_str = str(trial_index)

        # Matches files like: M3ED_<SCENE>_trial_0.txt, M3ED_<SCENE>_trial_1.txt, ...
        trial_filename = f"M3ED_{sequence}_trial_{trial_str}.txt"
        trial_path = os.path.join(traj_base_dir, trial_filename)

        run_label = f"{sequence}, Config: {CONFIG_NAME}, Trial: {trial_str}"
        print(f"  Running: {run_label}...")

        if not os.path.exists(trial_path):
            print(f"    - WARNING: Trial file not found: {trial_path}. Skipping.")
            continue

        command = [
            "evo_ape",
            "tum",
            gt_path,
            trial_path,
            "--t_max", "0.5",
            "-a",
            "--t_start", str(t_start),
            "--t_end", str(t_end),
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )
            output = result.stdout

            match = rmse_pattern.search(output)
            if match:
                rmse_value = match.group(1)
                print(f"    - Found RMSE: {rmse_value}")
                with open(output_file, "a") as f:
                    f.write(f"{sequence},{CONFIG_NAME},{trial_str},{rmse_value}\n")
            else:
                print(f"    - Could not find RMSE in evo_ape output for {run_label}. Skipping.")
                print("--- Output was:\n" + output)

        except subprocess.CalledProcessError as e:
            print(f"    - Error running evo_ape for {run_label}: {e}")
            print(f"    - Stderr:\n{e.stderr}")
        except FileNotFoundError:
            print("    - Error: 'evo_ape' command not found. Make sure it is installed and in your PATH.")
            raise

print("\nAll runs complete. Results saved to", output_file)
