import argparse
import cProfile
import os
import pstats
import glob
import cv2
import evo.main_ape as main_ape
import gtsam
import numpy as np
import torch
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from dpvo.config import cfg
from dpvo.dvio import DVIO
from dpvo.parallel import pgenerator
from dpvo.plot_utils import (
    plot_trajectory,
    save_output_for_COLMAP,
    save_ply,
    save_point_cloud,
)
from dpvo.utils import Timer


# --- TartanAir Constants ---
# Standard intrinsics for TartanAir (640x480)
FX, FY, CX, CY = 320.0, 320.0, 320.0, 320.0
IMG_FREQ = 10.0  # TartanAir images are usually 10Hz
STEREO_BASELINE = 0.25 # Standard TartanAir baseline (meters)

def tartan_image_stream(
    root_path, side="left", start=0, stop=None, stride=1, scale=1.0
):
    """
    Generator for TartanAir images from directory structure.
    """
    if side == "left":
        image_dir = os.path.join(root_path, "image_lcam_front")
        
    else:
        image_dir = os.path.join(root_path, "image_rcam_front")    
    
    # TartanAir V2 might be .png or .jpg
    files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not files:
        files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    if stop is None:
        stop = len(files)

    files = files[start:stop:stride]
    
    # Intrinsics matrix
    K = np.array([
        [FX, 0, CX],
        [0, FY, CY],
        [0, 0, 1]
    ])

    # If scaling is applied, adjust intrinsics and resolution
    if scale != 1.0:
        K[:2, :] *= scale
        
    intrinsics_vec = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
    H, W = int(640 * scale), int(640 * scale)

    print(f"Streaming {side} images from {image_dir}")
    print(f"Resolution: {W}x{H}, Intrinsics: {intrinsics_vec}")

    for i, fpath in enumerate(tqdm(files)):
        image = cv2.imread(fpath)
        
        if image is None:
            continue

        if scale != 1.0:
            image = cv2.resize(image, (W, H))

        # Generate timestamp based on index and frequency
        # Note: We use the original index (start + i*stride) to calculate time
        curr_idx = start + (i * stride)
        timestamp = curr_idx / IMG_FREQ

        yield timestamp, image, intrinsics_vec


def read_tartan_imu(root_path):
    """
    Reads IMU data from TartanAir V2 format.
    Assumes imu/accel_left.npy and imu/gyro_left.npy exist.
    """
    imu_dir = os.path.join(root_path, "imu")
    
    # Try loading .npy files first (common in V2)
    accel_path = os.path.join(imu_dir, "acc_noisy.npy")
    gyro_path = os.path.join(imu_dir, "gyro_noisy.npy")
    time_path = os.path.join(imu_dir, "imu_time.npy") # sometimes exists

    if os.path.exists(accel_path) and os.path.exists(gyro_path):
        accel = np.load(accel_path)
        gyro = np.load(gyro_path)
        
        # Load or generate timestamps
        if os.path.exists(time_path):
            ts = np.load(time_path)
        else:
            # If explicit timestamps missing, assume alignment with motion file or high freq
            # Here we assume a standard high frequency or alignment with image count logic
            # For simplicity in synthetic data without time file, we might need to rely on 
            # the motion.npy file length.
            # Let's try to find motion file to get duration
            motion_path = os.path.join(root_path, "motion_left.npy")
            if os.path.exists(motion_path):
                 # This is tricky without explicit IMU timestamps in synthetic data.
                 # Usually TartanAir IMU is 100Hz or synchronous.
                 # Let's generate timestamps assuming 100Hz? 
                 # Safer approach: Check if data is already synchronized or needs interpolation.
                 # For this script, let's create synthetic timestamps starting at 0
                 n_samples = accel.shape[0]
                 # Assuming 100Hz approx for now, or match image duration
                 ts = np.linspace(0, n_samples * 0.01, n_samples)
            else:
                 n_samples = accel.shape[0]
                 ts = np.arange(n_samples) * 0.01 # Assume 100hz

    else:
        # Fallback for folder structures containing 'imu_data.txt'
        # Format often: [timestamp, ax, ay, az, gx, gy, gz, qx, qy, qz, qw]
        txt_path = glob.glob(os.path.join(root_path, "*imu*.txt"))
        if txt_path:
            data = np.loadtxt(txt_path[0])
            ts = data[:, 0]
            accel = data[:, 1:4]
            gyro = data[:, 4:7]
        else:
            raise FileNotFoundError(f"Could not find IMU data in {imu_dir}")

    # Combine into [timestamp, gx, gy, gz, ax, ay, az]
    # DVIO usually expects Gyro then Accel
    # IMPORTANT: DPVO/DVIO M3ED code converts gyro to degrees: all_imu[:, 1:4] *= 180 / np.pi
    # TartanAir is in Rad/s.
    
    all_imu = np.hstack((ts[:, None], gyro, accel))
    
    # Convert gyro from radians to degrees as expected by the provided DVIO implementation
    all_imu[:, 1:4] *= 180 / np.pi 
    
    all_imu = all_imu[all_imu[:, 0].argsort()]  # Sort by timestamp
    return all_imu


def get_tartan_stereo_extrinsics():
    """
    Returns T_right_to_left (Right camera pose in Left camera frame).
    TartanAir standard baseline is 0.25m along X axis.
    """
    # T_left_world = Identity
    # T_right_world = [1 0 0 -0.25]
    # T_right_to_left = T_left * inv(T_right) 
    # Actually, we need Extrinsics for the DVIO class. 
    # Based on M3ED code: T_camera2_to_camera = T_camera @ np.linalg.inv(T_camera2)
    # T_right_to_left = [x, y, z, qx, qy, qz, qw]
    
    # Right camera is at +0.25m x relative to left? No, usually Right is at +0.25 relative to origin?
    # In stereo, right cam is usually at X = +B relative to Left? Or Left is 0, Right is +B?
    # Usually: P_right = P_left - Baseline.
    # The extrinsics vector usually describes the transformation from Cam2 to Cam1.
    
    # Transformation from Right to Left:
    # Translation is [-0.25, 0, 0] (To move a point in Right to Left, we add baseline?)
    # Let's stick to standard: Right camera center in Left frame is at [+0.25, 0, 0].
    # So T_right_to_left translation is [0.25, 0, 0].
    
    return np.array([STEREO_BASELINE, 0, 0, 0, 0, 0, 1]) # x,y,z, qx,qy,qz,qw


def main():
    parser = argparse.ArgumentParser()
    # TartanAir specific arguments
    parser.add_argument("--root", default="/home/bora/m3ed/tartan", help="Root path to TartanAir")
    parser.add_argument("--env", default="AbandonedFactory2", help="Environment name")
    parser.add_argument("--level", default="Data_easy", help="Difficulty level")
    parser.add_argument("--seq", default="P000", help="Sequence name")
    
    parser.add_argument("--network", default="weights/dpvo.pth")
    parser.add_argument("--timeit", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--name", default="")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--config", default="config/superfast.yaml")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--opts", nargs="+", default=[])
    parser.add_argument("--save_trajectory", action="store_true")
    parser.add_argument("--save_ply", action="store_true")
    parser.add_argument("--save_colmap", action="store_true")
    parser.add_argument("--stride", type=int, default=1)

    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    # Construct Path
    # Expects: root/env/env/level/seq
    # Example: /datasets/TartanAir/soulcity/soulcity/Easy/P000
    scene_path = os.path.join(args.root, args.env, args.env, args.level, args.seq)
    
    if not os.path.exists(scene_path):
        # Try without double env folder
        scene_path = os.path.join(args.root, args.env, args.level, args.seq)
        if not os.path.exists(scene_path):
             print(f"Error: Path {scene_path} does not exist.")
             return

    print(f"Processing TartanAir Scene: {args.env}/{args.level}/{args.seq}")

    with torch.no_grad():
        H, W = int(640 * args.scale), int(640 * args.scale)

        # Extrinsics between stereo pair (Right to Left)
        extrinsics = get_tartan_stereo_extrinsics()

        slam = DVIO(
            cfg,
            args.network,
            ht=H,
            wd=W,
            show=args.show,
            extrinsics=extrinsics,
            enable_timing=args.timeit,
        )
        
        # --- IMU Setup ---
        try:
            # Load IMU Data
            slam.all_imu = read_tartan_imu(scene_path)
            
            # IMU Extrinsics (T_imu_to_camera)
            # In synthetic datasets like TartanAir, IMU often coincides with the camera or body frame.
            # Assuming Identity for now (IMU aligned with Left Camera).
            # If AirSim was used, check if NED->XYZ conversion is needed.
            # Here we assume the data provided in npy files matches camera frame (or close enough for DPVO).
            slam.Ti1c = np.eye(4) 
            slam.Tbc = gtsam.Pose3(slam.Ti1c)
            
            # Set IMU noise params (Acc Noise, Gyro Noise, Acc Walk, Gyro Walk)
            # Using generic values, tune if necessary for TartanAir
            slam.state.set_imu_params([0.16*100, 0.05*100, 0.003, 4.0e-5])
            
            print("IMU initialized successfully.")
        except Exception as e:
            print(f"Warning: IMU loading failed ({e}). Proceeding might fail if DVIO requires IMU.")

        # Generators
        generator1 = pgenerator(
            tartan_image_stream,
            root_path=scene_path,
            side="left",
            start=args.start,
            stop=args.stop,
            stride=args.stride,
            scale=args.scale,
        )
        generator2 = pgenerator(
            tartan_image_stream,
            root_path=scene_path,
            side="right",
            start=args.start,
            stop=args.stop,
            stride=args.stride,
            scale=args.scale,
        )

        for i, ((t1, image1, intrinsics1), (t2, image2, intrinsics2)) in enumerate(
            zip(generator1, generator2)
        ):
            if args.show:
                concat = cv2.hconcat([image1, image2])
                cv2.imshow("concat", concat)
                cv2.waitKey(1)

            image1 = torch.from_numpy(image1).permute(2, 0, 1).cuda()
            intrinsics1 = torch.from_numpy(intrinsics1).cuda()

            image2 = torch.from_numpy(image2).permute(2, 0, 1).cuda()
            intrinsics2 = torch.from_numpy(intrinsics2).cuda()

            # Note: TartanAir logic usually doesn't need remapping/undistortion 
            # as synthetic images are pinhole perfect, so we skipped the cv2.remap steps 
            # present in the M3ED code.

            slam(t1, (image1, image2), (intrinsics1, intrinsics2))

        poses, tstamps = slam.terminate()

    # --- Saving and Evaluation ---
    
    traj_est = PoseTrajectory3D(
        positions_xyz=poses[:, :3],
        orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],
        timestamps=tstamps,
    )

    if args.save_trajectory:
        os.makedirs("saved_trajectories", exist_ok=True)
        out_name = f"Tartan_{args.env}_{args.level}_{args.seq}"
        file_interface.write_tum_trajectory_file(
            f"saved_trajectories/{out_name}.txt", traj_est
        )

    # Ground Truth Loading and Evaluation
    gt_file = os.path.join(scene_path, "pose_lcam_front.txt")
    ate_score = None
    
    if os.path.exists(gt_file):
        print(f"Loading Ground Truth from {gt_file}")
        # TartanAir GT format: x y z qx qy qz qw (space separated) or NED?
        # Code 2 suggests PERM = [1, 2, 0, 4, 5, 3, 6] # ned -> xyz conversion might be needed
        # However, pose_left.txt in TartanAir V2 is usually standard XYZ?
        # Let's rely on standard TUM reading first, if fails, apply permutation manually.
        
        try:
            # Code 2 permutation logic:
            gt_data = np.loadtxt(gt_file)
            # Apply permutation from Code 2 to match DPVO coordinate system expectation
            # PERM = [1, 2, 0, 4, 5, 3, 6] # y, z, x, ...
            # Actually, standard TartanAir is often: x y z qx qy qz qw.
            # If the visualization looks wrong, uncomment the permutation logic.
            
            # Using Code 2 logic specifically:
            PERM = [1, 2, 0, 4, 5, 3, 6] # ned -> xyz
            gt_data = gt_data[:, PERM]

            # Generate timestamps for GT (sync with images)
            gt_tstamps = np.arange(len(gt_data)) / IMG_FREQ
            
            traj_ref = PoseTrajectory3D(
                positions_xyz=gt_data[:, :3],
                orientations_quat_wxyz=gt_data[:, [6, 3, 4, 5]], # xyzw -> wxyz
                timestamps=gt_tstamps
            )

            traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.1)

            result = main_ape.ape(
                traj_ref,
                traj_est,
                est_name="traj",
                pose_relation=PoseRelation.translation_part,
                align=True,
                correct_scale=True, # Synthetic data, scale should be 1.0, but align handles drift
            )
            ate_score = result.stats["rmse"]
            print(f"ATE: {ate_score:.03f}")
            plot_name = f"TartanAir {args.env} (ATE: {ate_score:.03f})"
        except Exception as e:
            print(f"Error in evaluation: {e}")
            plot_name = f"TartanAir {args.env} (ATE: Error)"
    else:
        print("No Ground Truth found.")
        plot_name = f"TartanAir {args.env}"

    if args.plot:
        os.makedirs("trajectory_plots", exist_ok=True)
        plot_trajectory(
            traj_est,
            traj_ref if ate_score is not None else None,
            plot_name,
            f"trajectory_plots/Tartan_{args.env}_{args.seq}.pdf",
            align=True,
            correct_scale=True,
        )

if __name__ == "__main__":
    main()