#!/usr/bin/env python3
import subprocess
import pathlib
import os
import re
import sys

# === KULLANICI AYARLARI ===
GT_PATH = "~/m3ed/m3ed/spot_outdoor_day_srt_under_bridge_1/spot_outdoor_day_srt_under_bridge_1_pose_evo_gt.txt"
TRAJ_DIR = "./saved_trajectories"
OUT_DIR = "./ape_results"
SUMMARY_PATH = "./ape_summary_outdoor_day_srt_under_bridge_1.txt"
T_MAX = "0.1"
USE_AP_FLAG = False  # True -> -ap, False -> -a

RMSE_RE = re.compile(r"rmse\s+([0-9.eE+-]+)")

def run_evo(gt_path: str, traj_path: str):
    """evo_ape komutunu çalişitrir, (stdout, stderr, rmse) döner."""
    if USE_AP_FLAG:
        cmd = [
            "evo_ape", "tum",
            "-a",
            gt_path,
            traj_path,
            "--t_max", T_MAX,
        ]
    else:
        cmd = [
            "evo_ape", "tum",
            "-a",
            gt_path,
            traj_path,
            "--t_max", T_MAX,
            "--t_start","5",
        ]

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,   # hata verse de dosyaya yazalım
    )

    stdout = res.stdout
    stderr = res.stderr

    # rmse'yi çek
    rmse_val = None
    m = RMSE_RE.search(stdout)
    if m:
        rmse_val = float(m.group(1))

    return cmd, stdout, stderr, rmse_val, res.returncode


def main():
    gt_path = os.path.expanduser(GT_PATH)
    traj_dir = pathlib.Path(TRAJ_DIR)
    out_dir = pathlib.Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pathlib.Path(gt_path).exists():
        print(f"[ERR] GT dosyası bulunamadı: {gt_path}", file=sys.stderr)
        sys.exit(1)

    txt_files = sorted(traj_dir.glob("*.txt"))
    if not txt_files:
        print(f"[WARN] {traj_dir} içinde .txt bulunamadı.")
        return

    # 4 kova
    results = {
        "dvio_raw": [],  # (filename, rmse)
        "dvio_vi": [],
        "deio_raw": [],
        "deio_vi": [],
    }

    for traj in txt_files:
        print(f"[RUN] {traj.name}")

        cmd, stdout, stderr, rmse, ret = run_evo(gt_path, str(traj))

        # tekil log dosyası
        per_file_log = out_dir / f"{traj.name}_ape.txt"
        with per_file_log.open("w") as f:
            f.write("COMMAND:\n")
            f.write(" ".join(cmd) + "\n\n")
            f.write("STDOUT:\n")
            f.write(stdout)
            if stderr:
                f.write("\nSTDERR:\n")
                f.write(stderr)
            f.write(f"\nRETURN_CODE: {ret}\n")
            if rmse is not None:
                f.write(f"PARSED_RMSE: {rmse}\n")

        # hangi kovaya girecek?
        name = traj.name

        is_dvio = "M3ED_dvio_" in name
        is_deio = "M3ED_deio_" in name
        is_vi = name.endswith("_vi.txt")

        key = None
        if is_dvio and is_vi:
            key = "dvio_vi"
        elif is_dvio and not is_vi:
            key = "dvio_raw"
        elif is_deio and is_vi:
            key = "deio_vi"
        elif is_deio and not is_vi:
            key = "deio_raw"

        if key is not None:
            results[key].append((name, rmse))
        else:
            # tanınmayan dosyaları da logla
            print(f"[INFO] Bu dosya tanıdık pattern'e uymuyor, summary'e eklenmedi: {name}")

        print(f"  -> rmse: {rmse}")

    # Şimdi summary yaz
    with open(SUMMARY_PATH, "w") as f:
        f.write("# APE RMSE SUMMARY\n")
        f.write(f"# GT: {gt_path}\n")
        f.write(f"# t_max: {T_MAX}\n\n")

        def dump_group(title, items):
            f.write(f"## {title}\n")
            if not items:
                f.write("(no entries)\n\n")
                return
            # sort by filename
            items = sorted(items, key=lambda x: x[0])
            for filename, rmse in items:
                f.write(f"{filename}: {rmse}\n")
            # avg
            valid = [r for (_, r) in items if r is not None]
            if valid:
                avg = sum(valid) / len(valid)
                f.write(f"AVERAGE: {avg}\n")
            f.write("\n")

        dump_group("DVIO (raw)", results["dvio_raw"])
        dump_group("DVIO (vi)", results["dvio_vi"])
        dump_group("DEIO (raw)", results["deio_raw"])
        dump_group("DEIO (vi)", results["deio_vi"])

    print(f"[DONE] Summary -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
