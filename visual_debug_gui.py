#!/usr/bin/env python3
"""Interactive debug GUI for DVIO trajectories, biases, and images."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.units import Unit
from evo.core.trajectory import PoseTrajectory3D
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.tools import file_interface
from matplotlib.widgets import Button, Slider, RadioButtons
from scipy.spatial.transform import Rotation as R


@dataclass
class ApeResult:
    rmse: Optional[float]
    stats: Optional[dict]


@dataclass
class RpeResult:
    times: Optional[np.ndarray]
    errors: Optional[np.ndarray]
    stats: Optional[dict]


def _safe_diff(values: np.ndarray) -> np.ndarray:
    diffs = np.diff(values)
    diffs[diffs == 0] = 1e-6
    return diffs


def compute_linear_speed(positions: np.ndarray, timestamps: np.ndarray) -> Optional[np.ndarray]:
    if positions.shape[0] < 2 or timestamps.shape[0] < 2:
        return None
    speeds = np.zeros(len(positions), dtype=float)
    dt = _safe_diff(timestamps)
    velocities = np.zeros_like(positions)
    center_dt = timestamps[2:] - timestamps[:-2]
    center_dt = np.where(center_dt == 0, 1e-6, center_dt)
    velocities[1:-1] = (positions[2:] - positions[:-2]) / center_dt[:, None]
    velocities[0] = (positions[1] - positions[0]) / dt[0]
    velocities[-1] = (positions[-1] - positions[-2]) / dt[-1]
    speeds[:] = np.linalg.norm(velocities, axis=1)
    return speeds


def compute_angular_speed(quaternions_wxyz: np.ndarray, timestamps: np.ndarray) -> Optional[np.ndarray]:
    if quaternions_wxyz is None or len(quaternions_wxyz) < 2:
        return None
    if timestamps.shape[0] < 2:
        return None
    quats_xyzw = np.column_stack((quaternions_wxyz[:, 1:], quaternions_wxyz[:, 0]))
    rotations = R.from_quat(quats_xyzw)
    speeds = np.zeros(len(quaternions_wxyz), dtype=float)
    for i in range(len(quaternions_wxyz) - 1):
        dt = timestamps[i + 1] - timestamps[i]
        if dt <= 0:
            continue
        delta = rotations[i].inv() * rotations[i + 1]
        angle = delta.magnitude()
        speeds[i + 1] = angle / dt
    if len(speeds) > 1:
        speeds[0] = speeds[1]
    return speeds


def _parent_group(h5_path: str) -> str:
    """Returns the parent group path for an HDF5 dataset."""
    parts = [p for p in h5_path.split("/")[:-1] if p]
    if not parts:
        return "/"
    return "/" + "/".join(parts)


class BiasLog:
    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        if data.ndim == 0:
            data = np.array([data])
        self.timestamps = np.asarray(data["timestamp"], dtype=float)
        self.gyro = np.vstack((data["bg_x"], data["bg_y"], data["bg_z"])).T
        self.accel = np.vstack((data["ba_x"], data["ba_y"], data["ba_z"])).T

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        gyro = np.array([
            np.interp(t, self.timestamps, self.gyro[:, i], left=self.gyro[0, i], right=self.gyro[-1, i])
            for i in range(3)
        ])
        accel = np.array([
            np.interp(t, self.timestamps, self.accel[:, i], left=self.accel[0, i], right=self.accel[-1, i])
            for i in range(3)
        ])
        return gyro, accel


class ImageSequence:
    def __init__(
        self,
        h5_path: str,
        camera: str,
        scale: float = 1.0,
        clahe: bool = False,
        stride: int = 1,
    ):
        if not os.path.exists(h5_path):
            raise FileNotFoundError(h5_path)
        self.file = h5py.File(h5_path, "r")
        camera_path = camera if camera.startswith("/") else f"/{camera}"
        if camera_path not in self.file:
            raise KeyError(f"Camera group {camera} not found in {h5_path}")
        self.camera = camera_path
        self.data = self.file[f"{self.camera}/data"]
        self.scale = scale
        self._clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8)) if clahe else None

        resolution = self.file[f"{self.camera}/calib/resolution"][()] * scale
        intrinsics = self.file[f"{self.camera}/calib/intrinsics"][()] * scale
        distortion = self.file[f"{self.camera}/calib/distortion_coeffs"][()]

        self.width = int(resolution[0])
        self.height = int(resolution[1])

        K = np.array(
            [
                [intrinsics[0], 0.0, intrinsics[2]],
                [0.0, intrinsics[1], intrinsics[3]],
                [0.0, 0.0, 1.0],
            ]
        )
        self.new_K, _ = cv2.getOptimalNewCameraMatrix(K, distortion, (self.width, self.height), 0)
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            K, distortion, None, self.new_K, (self.width, self.height), cv2.CV_32FC1
        )

        ts_group = _parent_group(self.camera)
        ts_ds = self.file[f"{ts_group}/ts"]
        ts_full = ts_ds[()] / 1e6
        stride = max(1, stride)
        self.frame_indices = np.arange(0, len(ts_full), stride, dtype=int)
        self.timestamps = ts_full[self.frame_indices]

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = self._clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def get_image(self, timestamp: float) -> np.ndarray:
        if len(self.timestamps) == 0:
            raise RuntimeError("No timestamps available in camera stream")
        idx = int(np.searchsorted(self.timestamps, timestamp))
        idx = np.clip(idx, 0, len(self.timestamps) - 1)
        frame_idx = self.frame_indices[idx]
        image = np.array(self.data[frame_idx])
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if (image.shape[1], image.shape[0]) != (self.width, self.height):
            image = cv2.resize(image, (self.width, self.height))
        image = cv2.remap(image, self.mapx, self.mapy, cv2.INTER_LINEAR)
        if self._clahe is not None:
            image = self._apply_clahe(image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class TrajectoryGUI:
    def __init__(
        self,
        traj_est: PoseTrajectory3D,
        traj_gt: Optional[PoseTrajectory3D],
        bias_log: Optional[BiasLog],
        image_seq: Optional[ImageSequence],
        ape_result: ApeResult,
        rpe_result: RpeResult,
        fps: float = 10.0,
    ):
        self.traj_est = traj_est
        self.traj_gt = traj_gt
        self.bias_log = bias_log
        self.image_seq = image_seq
        self.ape_result = ape_result
        self.rpe_result = rpe_result
        self.timestamps = np.asarray(traj_est.timestamps)
        if self.timestamps.size == 0:
            raise ValueError("Estimated trajectory has no timestamps")
        self.idx = 0
        self.playing = False
        self.fps = max(1e-3, fps)
        self.slider_internal = False

        self.est_xyz = np.asarray(traj_est.positions_xyz)
        self.gt_xyz = np.asarray(traj_gt.positions_xyz) if traj_gt is not None else None
        self.gt_timestamps = np.asarray(traj_gt.timestamps) if traj_gt is not None else None
        self.rpe_times = rpe_result.times
        self.rpe_errors = rpe_result.errors
        self.ate_series = None
        self.ate_times = None
        if self.gt_xyz is not None:
            length = min(len(self.est_xyz), len(self.gt_xyz))
            self.ate_series = np.linalg.norm(self.est_xyz[:length] - self.gt_xyz[:length], axis=1)
            self.ate_times = self.timestamps[:length]

        orientations = getattr(traj_est, "orientations_quat_wxyz", None)
        self.linear_speed = compute_linear_speed(self.est_xyz, self.timestamps)
        self.angular_speed = compute_angular_speed(np.asarray(orientations), self.timestamps) if orientations is not None else None

        self.view_modes = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}
        self.view_mode = "XY"
        self.view_axes = self.view_modes[self.view_mode]

        self._setup_figure()
        self._update_plot(0)

    def _setup_figure(self) -> None:
        self.fig = plt.figure(figsize=(16, 9))
        gs = self.fig.add_gridspec(3, 3, height_ratios=[3, 2.2, 2], width_ratios=[2.2, 1.2, 0.8])
        gs.update(left=0.04, right=0.98, bottom=0.16, top=0.95, wspace=0.25, hspace=0.35)
        self.ax_traj = self.fig.add_subplot(gs[0, 0])
        bias_grid = gs[0, 1].subgridspec(2, 1, hspace=0.08)
        self.ax_bias_gyro = self.fig.add_subplot(bias_grid[0, 0])
        self.ax_bias_accel = self.fig.add_subplot(bias_grid[1, 0], sharex=self.ax_bias_gyro)
        self.ax_image = self.fig.add_subplot(gs[1, 0])
        self.ax_image.axis("off")
        self.ax_rpe = self.fig.add_subplot(gs[1, 1])
        self.ax_dynamics = self.fig.add_subplot(gs[2, :])
        self.ax_dynamics_twin = None

        self._init_traj_plot()
        self._init_bias_plots()
        self._init_rpe_plot()
        self._init_dynamics_plot()
        self._init_slider_and_controls()

    def _init_traj_plot(self) -> None:
        self.est_line, = self.ax_traj.plot([], [], color="C0", label="Estimated")
        self.gt_line = None
        if self.gt_xyz is not None:
            self.gt_line, = self.ax_traj.plot([], [], color="C1", label="Ground Truth")
        self.est_marker = self.ax_traj.scatter([], [], color="C0", s=40, zorder=3)
        self.gt_marker = None
        if self.gt_xyz is not None:
            self.gt_marker = self.ax_traj.scatter([], [], color="C1", s=40, zorder=3)

        title = "Trajectory View"
        if self.ape_result.rmse is not None:
            title = f"Trajectory View (ATE RMSE: {self.ape_result.rmse:.3f} m)"
        self.ax_traj.set_title(title)
        self._set_traj_labels()
        self._update_traj_limits()
        self.ax_traj.legend(loc="upper right")

    def _set_traj_labels(self) -> None:
        axis_names = ["x", "y", "z"]
        a, b = self.view_axes
        self.ax_traj.set_xlabel(f"{axis_names[a]} [m]")
        self.ax_traj.set_ylabel(f"{axis_names[b]} [m]")

    def _update_traj_limits(self) -> None:
        proj_est = self.est_xyz[:, list(self.view_axes)]
        combined = proj_est
        if self.gt_xyz is not None:
            combined = np.vstack((combined, self.gt_xyz[:, list(self.view_axes)]))
        x_min, y_min = np.min(combined[:, 0]), np.min(combined[:, 1])
        x_max, y_max = np.max(combined[:, 0]), np.max(combined[:, 1])
        span = max(x_max - x_min, y_max - y_min, 1e-3)
        pad = 0.05 * span
        self.ax_traj.set_xlim(x_min - pad, x_max + pad)
        self.ax_traj.set_ylim(y_min - pad, y_max + pad)
        self.ax_traj.set_aspect("equal", adjustable="box")

    def _project_points(self, points: np.ndarray) -> np.ndarray:
        return points[:, list(self.view_axes)]

    def _init_bias_plots(self) -> None:
        if self.bias_log is None:
            self.ax_bias_gyro.text(0.5, 0.5, "No bias log", ha="center", va="center")
            self.ax_bias_gyro.axis("off")
            self.ax_bias_accel.axis("off")
            self.bias_cursor_gyro = None
            self.bias_cursor_accel = None
            return
        t = self.bias_log.timestamps
        self.ax_bias_gyro.plot(t, self.bias_log.gyro[:, 0], label="bg_x")
        self.ax_bias_gyro.plot(t, self.bias_log.gyro[:, 1], label="bg_y")
        self.ax_bias_gyro.plot(t, self.bias_log.gyro[:, 2], label="bg_z")
        self.ax_bias_gyro.set_ylabel("gyro bias [rad/s]")
        self.ax_bias_gyro.grid(True)
        self.ax_bias_gyro.legend(loc="upper right", fontsize=8)

        self.ax_bias_accel.plot(t, self.bias_log.accel[:, 0], label="ba_x")
        self.ax_bias_accel.plot(t, self.bias_log.accel[:, 1], label="ba_y")
        self.ax_bias_accel.plot(t, self.bias_log.accel[:, 2], label="ba_z")
        self.ax_bias_accel.set_ylabel("accel bias [m/s^2]")
        self.ax_bias_accel.set_xlabel("time [s]")
        self.ax_bias_accel.grid(True)
        self.ax_bias_accel.legend(loc="upper right", fontsize=8)

        self.bias_cursor_gyro = self.ax_bias_gyro.axvline(self.timestamps[0], color="k", linestyle="--", linewidth=1)
        self.bias_cursor_accel = self.ax_bias_accel.axvline(self.timestamps[0], color="k", linestyle="--", linewidth=1)

    def _init_rpe_plot(self) -> None:
        if self.rpe_errors is None or self.rpe_times is None:
            self.ax_rpe.text(0.5, 0.5, "No RPE available", ha="center", va="center")
            self.ax_rpe.axis("off")
            self.rpe_live_line = None
            self.rpe_cursor = None
            return

        self.ax_rpe.plot(self.rpe_times, self.rpe_errors, color="0.8", label="RPE (full)")
        self.rpe_live_line, = self.ax_rpe.plot([], [], color="C3", label="RPE (up to t)")
        self.rpe_cursor = self.ax_rpe.axvline(self.timestamps[0], color="k", linestyle="--", linewidth=1)
        self.ax_rpe.set_title("Relative Pose Error")
        self.ax_rpe.set_xlabel("time [s]")
        self.ax_rpe.set_ylabel("translational error [m]")
        self.ax_rpe.grid(True)
        self.ax_rpe.legend(loc="upper right", fontsize=8)

    def _init_dynamics_plot(self) -> None:
        self.ang_live_line = None
        self.ang_cursor = None
        self.lin_live_line = None
        self.lin_cursor = None
        self.ate_live_line = None
        self.ate_cursor = None

        has_velocity = any(
            data is not None for data in (self.angular_speed, self.linear_speed)
        )
        has_ate = self.ate_series is not None and self.ate_times is not None

        if not has_velocity and not has_ate:
            self.ax_dynamics.text(0.5, 0.5, "No dynamics data", ha="center", va="center")
            self.ax_dynamics.axis("off")
            return

        self.ax_dynamics.set_title("Dynamics & Position Error")
        self.ax_dynamics.set_xlabel("time [s]")
        if has_velocity:
            self.ax_dynamics.set_ylabel("velocity")
            self.ax_dynamics.grid(True)
        colors = {"ang": "C4", "lin": "C5", "ate": "C2"}

        if self.angular_speed is not None:
            self.ax_dynamics.plot(self.timestamps, self.angular_speed, color="0.8", label="||ω|| (full)")
            self.ang_live_line, = self.ax_dynamics.plot([], [], color=colors["ang"], label="||ω|| (up to t)")
            self.ang_cursor = self.ax_dynamics.axvline(self.timestamps[0], color="k", linestyle="--", linewidth=1)

        if self.linear_speed is not None:
            self.ax_dynamics.plot(self.timestamps, self.linear_speed, color="0.7", label="||v|| (full)")
            self.lin_live_line, = self.ax_dynamics.plot([], [], color=colors["lin"], label="||v|| (up to t)")
            self.lin_cursor = self.ax_dynamics.axvline(self.timestamps[0], color="k", linestyle="--", linewidth=1)

        if has_velocity:
            self.ax_dynamics.legend(loc="upper left", fontsize=8)

        if has_ate:
            self.ax_dynamics_twin = self.ax_dynamics.twinx()
            self.ax_dynamics_twin.set_ylabel("ATE [m]")
            self.ax_dynamics_twin.plot(self.ate_times, self.ate_series, color="0.8", linestyle="--", label="ATE (full)")
            self.ate_live_line, = self.ax_dynamics_twin.plot([], [], color=colors["ate"], label="ATE (up to t)")
            self.ate_cursor = self.ax_dynamics_twin.axvline(
                self.timestamps[0], color="k", linestyle=":", linewidth=1
            )
            self.ax_dynamics_twin.legend(loc="upper right", fontsize=8)

    def _init_slider_and_controls(self) -> None:
        self.fig.subplots_adjust(bottom=0.15, right=0.87)
        slider_ax = self.fig.add_axes([0.12, 0.07, 0.65, 0.03])
        valmin = float(self.timestamps[0])
        valmax = float(self.timestamps[-1]) if len(self.timestamps) > 1 else valmin + 1e-3
        self.slider = Slider(
            slider_ax,
            label="time [s]",
            valmin=valmin,
            valmax=valmax,
            valinit=valmin,
        )
        self.slider.on_changed(self._on_slider)

        button_ax = self.fig.add_axes([0.8, 0.065, 0.12, 0.045])
        self.button = Button(button_ax, "Play")
        self.button.on_clicked(self._toggle_play)

        self.timer = self.fig.canvas.new_timer(interval=int(1000.0 / self.fps))
        self.timer.add_callback(self._step)

        radio_ax = self.fig.add_axes([0.9, 0.6, 0.08, 0.18])
        labels = list(self.view_modes.keys())
        self.view_radio = RadioButtons(radio_ax, labels, active=labels.index(self.view_mode))
        self.view_radio.on_clicked(self._on_view_change)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    def _on_slider(self, val: float) -> None:
        if self.slider_internal:
            return
        idx = int(np.searchsorted(self.timestamps, val))
        idx = np.clip(idx, 0, len(self.timestamps) - 1)
        self._update_plot(idx)

    def _toggle_play(self, _event) -> None:
        self.playing = not self.playing
        self.button.label.set_text("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    def _on_view_change(self, label: str) -> None:
        if label not in self.view_modes:
            return
        self.view_mode = label
        self.view_axes = self.view_modes[label]
        self._set_traj_labels()
        self._update_traj_limits()
        self._update_plot(self.idx)

    def _on_key_press(self, event) -> None:
        if event.key not in {"left", "right"}:
            return
        step = -1 if event.key == "left" else 1
        new_idx = np.clip(self.idx + step, 0, len(self.timestamps) - 1)
        self._update_plot(new_idx)

    def _step(self) -> None:
        if not self.playing:
            return
        next_idx = (self.idx + 1) % len(self.timestamps)
        self._update_plot(next_idx)
        if next_idx == len(self.timestamps) - 1:
            self.timer.stop()
            self.playing = False
            self.button.label.set_text("Play")

    def _update_plot(self, idx: int) -> None:
        self.idx = idx
        current_time = float(self.timestamps[idx])
        est_proj = self._project_points(self.est_xyz)
        est_slice = est_proj[: idx + 1]
        self.est_line.set_data(est_slice[:, 0], est_slice[:, 1])
        est_point = est_proj[idx]
        self.est_marker.set_offsets(np.array([[est_point[0], est_point[1]]]))

        if self.gt_line is not None and self.gt_xyz is not None and self.gt_timestamps is not None:
            gt_idx = int(np.searchsorted(self.gt_timestamps, current_time, side="right"))
            gt_idx = np.clip(gt_idx, 1, len(self.gt_xyz))
            gt_proj = self._project_points(self.gt_xyz)
            gt_slice = gt_proj[:gt_idx]
            self.gt_line.set_data(gt_slice[:, 0], gt_slice[:, 1])
            if self.gt_marker is not None:
                gt_point = gt_proj[gt_idx - 1]
                self.gt_marker.set_offsets(np.array([[gt_point[0], gt_point[1]]]))
        if self.bias_cursor_gyro is not None:
            self.bias_cursor_gyro.set_xdata([current_time, current_time])
        if self.bias_cursor_accel is not None:
            self.bias_cursor_accel.set_xdata([current_time, current_time])
        if self.rpe_cursor is not None:
            self.rpe_cursor.set_xdata([current_time, current_time])
        if self.rpe_live_line is not None and self.rpe_times is not None and self.rpe_errors is not None:
            mask = self.rpe_times <= current_time
            self.rpe_live_line.set_data(self.rpe_times[mask], self.rpe_errors[mask])
        if self.ate_cursor is not None:
            self.ate_cursor.set_xdata([current_time, current_time])
        if self.ate_live_line is not None and self.ate_times is not None and self.ate_series is not None:
            mask = self.ate_times <= current_time
            self.ate_live_line.set_data(self.ate_times[mask], self.ate_series[mask])
        if self.ang_cursor is not None:
            self.ang_cursor.set_xdata([current_time, current_time])
        if self.ang_live_line is not None and self.angular_speed is not None:
            mask = self.timestamps <= current_time
            self.ang_live_line.set_data(self.timestamps[mask], self.angular_speed[mask])
        if self.lin_cursor is not None:
            self.lin_cursor.set_xdata([current_time, current_time])
        if self.lin_live_line is not None and self.linear_speed is not None:
            mask = self.timestamps <= current_time
            self.lin_live_line.set_data(self.timestamps[mask], self.linear_speed[mask])

        if self.image_seq is not None:
            image = self.image_seq.get_image(current_time)
            if hasattr(self, "image_artist"):
                self.image_artist.set_data(image)
            else:
                self.image_artist = self.ax_image.imshow(image)
        self.fig.canvas.draw_idle()
        self.slider_internal = True
        self.slider.set_val(current_time)
        self.slider_internal = False

    def show(self) -> None:
        plt.show()


def load_trajectories(
    est_path: str, gt_path: Optional[str]
) -> tuple[PoseTrajectory3D, Optional[PoseTrajectory3D], ApeResult, RpeResult]:
    if not os.path.exists(est_path):
        raise FileNotFoundError(est_path)
    traj_est = file_interface.read_tum_trajectory_file(est_path)
    traj_gt = None
    ape_info = ApeResult(rmse=None, stats=None)
    rpe_info = RpeResult(times=None, errors=None, stats=None)
    if gt_path is None:
        return traj_est, traj_gt, ape_info, rpe_info
    if not os.path.exists(gt_path):
        raise FileNotFoundError(gt_path)
    traj_gt = file_interface.read_tum_trajectory_file(gt_path)
    traj_gt, traj_est = sync.associate_trajectories(traj_gt, traj_est)
    try:
        result = main_ape.ape(
            traj_gt,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=True,
            correct_scale=False,
        )
        stats_dict = dict(result.stats)
        ape_info = ApeResult(rmse=stats_dict.get("rmse"), stats=stats_dict)
    except np.linalg.LinAlgError:
        pass
    try:
        est_ts = np.asarray(traj_est.timestamps, dtype=float)
        if est_ts.size < 2:
            raise ValueError("Not enough samples for RPE")
        delta_frames = 1
        rpe_result = main_rpe.rpe(
            traj_gt,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            delta=delta_frames,
            delta_unit=Unit.frames,
            all_pairs=False,
            align=True,
            correct_scale=False,
        )
        rpe_stats = dict(rpe_result.stats)
        raw_errors = rpe_result.np_arrays.get("error_array")
        if raw_errors is not None:
            rpe_errors = np.asarray(raw_errors, dtype=float)
            if rpe_errors.size > 0:
                raw_times = rpe_result.np_arrays.get("timestamps")
                if raw_times is not None:
                    rpe_times_full = np.asarray(raw_times, dtype=float)
                else:
                    rpe_times_full = est_ts[-rpe_errors.size :]
                rpe_info = RpeResult(times=rpe_times_full, errors=rpe_errors, stats=rpe_stats)
    except (np.linalg.LinAlgError, ValueError):
        pass
    return traj_est, traj_gt, ape_info, rpe_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DVIO debugging GUI")
    parser.add_argument("--est", required=True, help="Estimated trajectory file in TUM format")
    parser.add_argument("--gt", help="Ground-truth trajectory in TUM format")
    parser.add_argument("--bias", help="IMU bias CSV log")
    parser.add_argument("--data-h5", help="Source HDF5 file to fetch images")
    parser.add_argument("--camera", default="/ovc/left", help="Camera group inside the HDF5 file")
    parser.add_argument("--scale", type=float, default=1.0, help="Image rescale factor")
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE to images")
    parser.add_argument("--stride", type=int, default=1, help="Subsample ratio for camera frames")
    parser.add_argument("--fps", type=float, default=10.0, help="Playback FPS for the slider animation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    traj_est, traj_gt, ape_result, rpe_result = load_trajectories(args.est, args.gt)

    bias_log = BiasLog(args.bias) if args.bias else None
    image_seq = None
    if args.data_h5:
        image_seq = ImageSequence(
            h5_path=args.data_h5,
            camera=args.camera,
            scale=args.scale,
            clahe=args.clahe,
            stride=args.stride,
        )

    gui = TrajectoryGUI(
        traj_est=traj_est,
        traj_gt=traj_gt,
        bias_log=bias_log,
        image_seq=image_seq,
        ape_result=ape_result,
        rpe_result=rpe_result,
        fps=args.fps,
    )
    gui.show()


if __name__ == "__main__":
    main()
