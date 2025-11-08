#!/usr/bin/env bash
set -euo pipefail

SCENE="spot_outdoor_day_srt_under_bridge_1"
DATA="${HOME}/m3ed/m3ed/${SCENE}/${SCENE}_data.h5"
GT="${HOME}/m3ed/m3ed/${SCENE}/${SCENE}_pose_evo_gt.txt"
CFG="config/superfast.yaml"
OUTDIR="saved_trajectories"

mkdir -p "${OUTDIR}"

# küçük yardımcı: en yeni dosyayı bul, varsa yeni isme taşı
move_latest() {
    local pattern="$1"
    local newname="$2"

    # pattern'e uyan dosya yoksa sessizce çık
    local latest
    latest=$(ls -t ${pattern} 2>/dev/null | head -n 1 || true)
    if [[ -n "${latest}" ]]; then
        mv "${latest}" "${newname}"
        echo " -> ${newname}"
    else
        echo " -> WARN: pattern bulunamadı: ${pattern}"
    fi
}

for i in $(seq -w 0 9); do
    echo "========================="
    echo "TRIAL ${i}"
    echo "========================="

    #
    # 1) DVIO / EIV scripti
    #
    python3 eval/evaluate_m3ed_i.py \
        "${DATA}" \
        --gt "${GT}" \
        --save_trajectory \
        --plot \
        --config "${CFG}"\
        --scale 0.5

    # dvio ana traj
    move_latest \
        "${OUTDIR}/M3ED_dvio_${SCENE}_data.txt" \
        "${OUTDIR}/M3ED_dvio_${SCENE}_trial_${i}.txt"

    # dvio vi traj
    move_latest \
        "${OUTDIR}/M3ED_dvio_${SCENE}_data_vi.txt" \
        "${OUTDIR}/M3ED_dvio_${SCENE}_trial_${i}_vi.txt"

    #
    # 2) DEIO / I scripti
    #
    python3 eval/evaluate_m3ed_eiv.py \
        "${DATA}" \
        --gt "${GT}" \
        --save_trajectory \
        --plot \
        --config "${CFG}" \
        --scale 0.5

    # deio ana traj
    move_latest \
        "${OUTDIR}/M3ED_deio_${SCENE}_data.txt" \
        "${OUTDIR}/M3ED_deio_${SCENE}_trial_${i}.txt"

    # deio vi traj
    move_latest \
        "${OUTDIR}/M3ED_deio_${SCENE}_data_vi.txt" \
        "${OUTDIR}/M3ED_deio_${SCENE}_trial_${i}_vi.txt"

    echo
done

echo "Bitti."
