import csv
import os
import cv2
import numpy as np

# ---------- TartanAir V2 depth reader (PNG -> float32 meters) ----------

def read_decode_depth(depthpath):
    depth_rgba = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)
    if depth_rgba is None:
        raise FileNotFoundError(depthpath)
    # Expected shape: H x W x 4, uint8
    assert depth_rgba.ndim == 3 and depth_rgba.shape[2] == 4, depth_rgba.shape
    depth = depth_rgba.view("<f4")         # reinterpret bytes as float32
    depth = np.squeeze(depth, axis=-1)     # H x W, float32
    return depth                           # in meters


csv_path = "patch_depths.csv"  # your CSV
depth_root = "/home/bora/m3ed/tartan/AbandonedFactory2/Data_easy/P004/depth_lcam_front"
fps = 10.0  # TartanAir image/depth rate

# depth limits for outlier rejection
MAX_GT_DEPTH = 20.0     # ignore GT depths > 80m (sky / invalid)
MAX_PRED_DEPTH = 20.0   # ignore predicted depths > 80m (obvious VO failure)

pred_depths = []
gt_depths = []
pred_inv_depths = []
gt_inv_depths = []

# for listing by error percentage
samples = []

with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)

    current_frame_idx = None
    depth_gt = None

    for row in reader:
        # 1) map timestamp -> frame index
        t = float(row["timestamp"])            # e.g. 16.64
        frame_idx = int(round(t * fps))        # 16.64 * 10 ≈ 166

        # 2) pixel coordinates & predictions from CSV
        x_px = int(round(float(row["x_px"])))
        y_px = int(round(float(row["y_px"])))

        # guard against empty depth strings
        depth_str = row["depth"].strip()
        if depth_str == "":
            continue

        pred_depth = float(depth_str)
        pred_inv_depth = float(row["inv_depth"])

        # 3) load depth for this frame if needed
        if frame_idx != current_frame_idx:
            depth_fname = f"{frame_idx:06d}_lcam_front_depth.png"  # adjust pattern if needed
            depth_path = os.path.join(depth_root, depth_fname)
            depth_gt = read_decode_depth(depth_path)
            current_frame_idx = frame_idx

        H, W = depth_gt.shape
        if not (0 <= x_px < W and 0 <= y_px < H):
            # out of bounds -> skip
            continue

        gt_depth = float(depth_gt[y_px, x_px])

        # 4) skip invalid / outlier GT depths
        if (not np.isfinite(gt_depth)) or gt_depth <= 0 or gt_depth > MAX_GT_DEPTH:
            # e.g. 65504.0 (invalid), 0, negative, NaN, etc.
            continue

        # 5) skip invalid / outlier predicted depths
        if (not np.isfinite(pred_depth)) or pred_depth <= 0 or pred_depth > MAX_PRED_DEPTH:
            # e.g. 10000, etc.
            continue

        gt_inv_depth = 1.0 / gt_depth

        # relative error in depth (percentage)
        rel_err_pct = abs(pred_depth - gt_depth) / gt_depth * 100.0

        # 6) store for metrics
        pred_depths.append(pred_depth)
        gt_depths.append(gt_depth)
        pred_inv_depths.append(pred_inv_depth)
        gt_inv_depths.append(gt_inv_depth)

        # store full sample for later sorting / listing
        samples.append(
            {
                "t": t,
                "frame_idx": frame_idx,
                "x_px": x_px,
                "y_px": y_px,
                "pred_depth": pred_depth,
                "gt_depth": gt_depth,
                "pred_inv": pred_inv_depth,
                "gt_inv": gt_inv_depth,
                "rel_err_pct": rel_err_pct,
            }
        )

        # 7) PRINT predicted & GT depth + inverse depth (online view)
        # print(
        #     f"t={t:.3f}s, frame={frame_idx:06d}, "
        #     f"px=({x_px},{y_px}), "
        #     f"pred_depth={pred_depth:.6f}, gt_depth={gt_depth:.6f}, "
        #     f"pred_inv={pred_inv_depth:.6f}, gt_inv={gt_inv_depth:.6f}, "
        #     f"rel_err={rel_err_pct:.2f}%"
        # )

# Optional: metrics on depth or inverse depth
pred_depths = np.array(pred_depths)
gt_depths = np.array(gt_depths)
pred_inv_depths = np.array(pred_inv_depths)
gt_inv_depths = np.array(gt_inv_depths)

if len(pred_depths) > 0:
    depth_rmse = np.sqrt(np.mean((pred_depths - gt_depths) ** 2))
    inv_rmse = np.sqrt(np.mean((pred_inv_depths - gt_inv_depths) ** 2))

    print(f"\n#samples (after filtering): {len(pred_depths)}")
    print(f"Depth RMSE:       {depth_rmse:.4f} m")
    print(f"Inv-depth RMSE:   {inv_rmse:.4f} 1/m")

    # ---------- listing by error percentage ----------
    print("\nSamples sorted by depth relative error (lowest → highest):")
    samples_sorted = sorted(samples, key=lambda s: s["rel_err_pct"])

    for s in samples_sorted:
        print(
            f"rel_err={s['rel_err_pct']:.2f}% | "
            #f"t={s['t']:.3f}s, frame={s['frame_idx']:06d}, "
            f"px=({s['x_px']},{s['y_px']}), "
            f"pred_depth={s['pred_depth']:.6f}, gt_depth={s['gt_depth']:.6f}, "
            f"pred_inv={s['pred_inv']:.6f}, gt_inv={s['gt_inv']:.6f}"
        )
else:
    print("No valid samples after filtering.")
