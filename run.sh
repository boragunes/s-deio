#!/usr/bin/env bash
set -euo pipefail

# Eğer komut satırında sahne verilmezse, buradaki default liste kullanılır
SCENES=(
    "falcon_outdoor_day_fast_flight_1"

)

# Komut satırından sahne isimleri verilmişse onları kullan
if (( "$#" > 0 )); then
    SCENES=("$@")
fi

BASE_DIR="${HOME}/m3ed/m3ed"
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
        echo " -> WARN: pattern bulunamadi: ${pattern}"
    fi
}

for SCENE in "${SCENES[@]}"; do
    DATA="${BASE_DIR}/${SCENE}/${SCENE}_data.h5"
    GT="${BASE_DIR}/${SCENE}/${SCENE}_pose_evo_gt.txt"

    echo "########################################"
    echo "SCENE: ${SCENE}"
    echo "DATA:  ${DATA}"
    echo "GT:    ${GT}"
    echo "########################################"

    for i in $(seq -w 0 4); do
        echo "========================="
        echo "TRIAL ${i} - ${SCENE}"
        echo "========================="

        #
        # 1) DVIO / EIV scripti
        #
        python3 eval/evaluate_m3ed_i.py \
            "${DATA}" \
            --gt "${GT}" \
            --save_trajectory \
            --plot \
            --config "${CFG}" \
            --scale 0.5
         

        # dvio ana traj
        move_latest \
            "${OUTDIR}/M3ED_${SCENE}_data.txt" \
            "${OUTDIR}/M3ED_${SCENE}_trial_${i}.txt"

    done
done

echo "Bitti."
