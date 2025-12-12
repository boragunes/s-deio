#!/usr/bin/env python3
import argparse
import os

import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.tools import file_interface
import evo.main_ape as main_ape


def find_nearest_idx(ts_array, t):
    """
    Return index of element in ts_array whose value is closest to t.
    """
    idx = int(np.argmin(np.abs(ts_array - t)))
    return idx


class H5ImageReader:
    """
    Random-access reader for rectified images + timestamps from M3ED HDF5,
    similar to your rgb_generator but suitable for interactive visualization.
    """

    def __init__(self, path, camera_name="/ovc/left", scale=1.0, clahe=False):
        self.path = path
        self.camera_name = camera_name
        self.scale = scale
        self.clahe_enabled = clahe

        self._open_file()

    def _open_file(self):
        self.f = h5py.File(self.path, "r")
        camera_name = self.camera_name

        camera_model = self.f[f"{camera_name}/calib/camera_model"][()]
        distortion_coeffs = self.f[f"{camera_name}/calib/distortion_coeffs"][()]
        distortion_model = self.f[f"{camera_name}/calib/distortion_model"][()]
        intrinsics = self.f[f"{camera_name}/calib/intrinsics"][()] * self.scale
        resolution = self.f[f"{camera_name}/calib/resolution"][()] * self.scale
        self.H, self.W = int(resolution[1]), int(resolution[0])

        print("camera_model", camera_model.decode())
        print("intrinsics", intrinsics)
        print("resolution", resolution)
        print("distortion_model", distortion_model.decode())
        print("distortion_coeffs", distortion_coeffs)

        self.data = self.f[f"{camera_name}/data"]

        ts_group = "/".join(camera_name.split("/")[:-1])
        self.ts = self.f[f"{ts_group}/ts"][()] / 1e6  # to seconds

        # Intrinsics / rectify maps (same logic as your rgb_generator)
        K = np.array(
            [
                [intrinsics[0], 0, intrinsics[2]],
                [0, intrinsics[1], intrinsics[3]],
                [0, 0, 1],
            ]
        )
        K_new, _ = cv2.getOptimalNewCameraMatrix(
            K, distortion_coeffs, (self.W, self.H), 0, (self.W, self.H)
        )

        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            K, distortion_coeffs, None, K_new, (self.W, self.H), cv2.CV_32FC1
        )

        self.clahe = None
        if self.clahe_enabled:
            self.clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))

    def get_image_by_index(self, idx):
        """
        Returns (timestamp, image_rgb) for frame at index idx.
        """
        idx = int(np.clip(idx, 0, len(self.ts) - 1))
        t = self.ts[idx]

        image = self.data[idx]

        if self.scale != 1.0:
            image = cv2.resize(image, (self.W, self.H))

        # Rectify
        image = cv2.remap(image, self.mapx, self.mapy, cv2.INTER_LINEAR)

        # Ensure BGR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if self.clahe is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = self.clahe.apply(gray)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Convert BGR -> RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return t, image_rgb

    def get_image_near_time(self, t):
        """
        Returns (timestamp, image_rgb) for frame whose timestamp is closest to t.
        """
        idx = find_nearest_idx(self.ts, t)
        return self.get_image_by_index(idx)

    def close(self):
        self.f.close()


def load_bias_csv(path):
    """
    Load bias CSV:
    timestamp,frame_idx,stamp_idx,bg_x,bg_y,bg_z,bg_norm,ba_x,ba_y,ba_z,ba_norm
    """
    bias_data = np.loadtxt(path, delimiter=",", skiprows=1)

    t_bias = bias_data[:, 0]
    bg = bias_data[:, 3:6]
    bg_norm = bias_data[:, 6]
    ba = bias_data[:, 7:10]
    ba_norm = bias_data[:, 10]

    return t_bias, bg, bg_norm, ba, ba_norm


def compute_ape(traj_ref, traj_est, align=True, correct_scale=False):
    """
    Compute APE (RMSE) using evo.main_ape.
    """
    result = main_ape.ape(
        traj_ref,
        traj_est,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=align,
        correct_scale=correct_scale,
    )
    rmse = result.stats["rmse"]
    return rmse, result


def build_figure(traj_est, traj_ref, t_bias, bg, bg_norm, ba, ba_norm, img_reader):
    """
    Create the matplotlib figure, slider, and update callback.
    Returns (fig, slider, update_fn).
    """
    # Use associated trajectories timestamps as the master timeline
    t_traj = traj_est.timestamps
    est_xyz = traj_est.positions_xyz
    ref_xyz = traj_ref.positions_xyz

    # Compute APE RMSE (already aligned outside; but we can show stat)
    ate_rmse, _ = compute_ape(traj_ref, traj_est, align=False, correct_scale=False)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])

    # Trajectory subplot (top-left)
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_traj.set_title("Trajectory (top-down XY)")
    ax_traj.set_xlabel("X [m]")
    ax_traj.set_ylabel("Y [m]")

    # Plot full GT + EST trajectories
    line_ref, = ax_traj.plot(ref_xyz[:, 0], ref_xyz[:, 1], label="GT")
    line_est, = ax_traj.plot(est_xyz[:, 0], est_xyz[:, 1], label="EST")

    # Marker for current pose
    point_ref, = ax_traj.plot([], [], marker="o", linestyle="", label="GT (current)")
    point_est, = ax_traj.plot([], [], marker="x", linestyle="", label="EST (current)")

    ax_traj.legend()
    ax_traj.set_aspect("equal", adjustable="datalim")

    # Fix axis limits for stable viewing
    all_x = np.concatenate([ref_xyz[:, 0], est_xyz[:, 0]])
    all_y = np.concatenate([ref_xyz[:, 1], est_xyz[:, 1]])
    margin = 0.1 * max(all_x.max() - all_x.min(), all_y.max() - all_y.min())
    ax_traj.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax_traj.set_ylim(all_y.min() - margin, all_y.max() + margin)

    # Bias subplot (top-right)
    ax_bias = fig.add_subplot(gs[0, 1])
    ax_bias.set_title("IMU Bias vs Time")
    ax_bias.set_xlabel("time [s]")
    ax_bias.set_ylabel("bias")

    # Gyro biases
    ax_bias.plot(t_bias, bg[:, 0], label="bg_x")
    ax_bias.plot(t_bias, bg[:, 1], label="bg_y")
    ax_bias.plot(t_bias, bg[:, 2], label="bg_z")
    ax_bias.plot(t_bias, bg_norm, label="bg_norm")

    # Accel biases
    ax_bias.plot(t_bias, ba[:, 0], label="ba_x", linestyle="--")
    ax_bias.plot(t_bias, ba[:, 1], label="ba_y", linestyle="--")
    ax_bias.plot(t_bias, ba[:, 2], label="ba_z", linestyle="--")
    ax_bias.plot(t_bias, ba_norm, label="ba_norm", linestyle="--")

    vline = ax_bias.axvline(t_traj[0], linestyle=":")

    ax_bias.legend()
    ax_bias.grid(True)

    # Image subplot (bottom, full width)
    ax_img = fig.add_subplot(gs[1, :])
    ax_img.set_title("Image synchronized with trajectory & bias")
    ax_img.axis("off")

    # Initial image
    _, first_img = img_reader.get_image_near_time(t_traj[0])
    img_handle = ax_img.imshow(first_img)

    # Slider along bottom
    ax_slider = fig.add_axes([0.15, 0.03, 0.7, 0.03])
    slider = Slider(
        ax_slider,
        "Frame",
        0,
        len(t_traj) - 1,
        valinit=0,
        valstep=1,
    )

    # Text for instant error
    txt = fig.text(
        0.5,
        0.95,
        f"ATE RMSE (full): {ate_rmse:.3f} m",
        ha="center",
        va="center",
    )

    def update(idx):
        idx = int(idx)
        t = t_traj[idx]

        # --- Update trajectory markers ---
        point_ref.set_data(ref_xyz[idx, 0], ref_xyz[idx, 1])
        point_est.set_data(est_xyz[idx, 0], est_xyz[idx, 1])

        # Instant translation error at this frame
        inst_err = np.linalg.norm(ref_xyz[idx] - est_xyz[idx])

        # --- Update bias vertical line ---
        vline.set_xdata(t)

        # --- Update image ---
        _, img = img_reader.get_image_near_time(t)
        img_handle.set_data(img)

        # Update text
        txt.set_text(
            f"t = {t:.3f} s | instant |p_est - p_gt| = {inst_err:.3f} m | "
            f"ATE RMSE (full): {ate_rmse:.3f} m"
        )

    # Initialize
    update(0)

    def on_slider(val):
        update(val)
        fig.canvas.draw_idle()

    slider.on_changed(on_slider)

    # Keyboard navigation (left/right arrows)
    def on_key(event):
        if event.key == "right":
            new_idx = min(len(t_traj) - 1, int(slider.val) + 1)
            slider.set_val(new_idx)
        elif event.key == "left":
            new_idx = max(0, int(slider.val) - 1)
            slider.set_val(new_idx)

    fig.canvas.mpl_connect("key_press_event", on_key)

    fig.suptitle("VI Debug Viewer", fontsize=14)
    return fig, slider, update


def main():
    parser = argparse.ArgumentParser(
        description="Visual-inertial debug viewer: traj + bias + image (all synchronized)."
    )
    parser.add_argument("--est", required=True, help="Estimated TUM trajectory file")
    parser.add_argument("--gt", required=True, help="Ground truth TUM trajectory file")
    parser.add_argument("--bias", required=True, help="IMU bias CSV file")
    parser.add_argument("--data_h5", required=True, help="M3ED HDF5 data file")
    parser.add_argument("--camera", default="/ovc/left", help="Camera group in HDF5")
    parser.add_argument("--scale", type=float, default=1.0, help="Image scale factor")
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE on images")

    args = parser.parse_args()

    # --- Load trajectories ---
    traj_est = file_interface.read_tum_trajectory_file(args.est)
    traj_ref = file_interface.read_tum_trajectory_file(args.gt)

    # Associate trajectories by timestamp (evo sync)
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    # --- Load biases ---
    t_bias, bg, bg_norm, ba, ba_norm = load_bias_csv(args.bias)

    # --- Image reader ---
    img_reader = H5ImageReader(args.data_h5, camera_name=args.camera,
                               scale=args.scale, clahe=args.clahe)

    # --- Build figure / UI ---
    fig, slider, update = build_figure(
        traj_est, traj_ref, t_bias, bg, bg_norm, ba, ba_norm, img_reader
    )

    plt.show()

    img_reader.close()


if __name__ == "__main__":
    main()


