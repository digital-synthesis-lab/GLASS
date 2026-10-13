#!/usr/bin/env bash
# End-to-end amorphous Si example at 2.5 g/cm3:
#   1. random initial structures
#   2. unconditional generation
#   3. PDF-, ADF-, XRD- and ND-guided generation (targets: data/density_2.5)
#   4. comparison against the reference structures
#
# Run from any directory:  bash examples/silicon/run_example.sh
# Environment variables:
#   OUT      output directory            (default: examples/silicon/results)
#   N_INITS  number of initial structures (default: 5)
#   N_RUNS   runs per initial structure  (default: 2)
#   STEPS    steps per denoising pass    (default: 512, as in the preprint)
#   DEVICE   torch device                (default: cuda:0)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$HERE/experiment"
REF="$HERE/data/density_2.5"
OUT="${OUT:-$HERE/results}"
N_INITS="${N_INITS:-5}"
N_RUNS="${N_RUNS:-2}"
STEPS="${STEPS:-512}"
DEVICE="${DEVICE:-cuda:0}"
export KMP_DUPLICATE_LIB_OK=TRUE

COMMON=(--n-runs "$N_RUNS" --uncond-tstep "$STEPS" --tstep "$STEPS" --device "$DEVICE")

# 1. Random 216-atom cells at 2.5 g/cm3 with a 2 Å minimum distance
mkdir -p "$OUT/inits"
for ((i = 0; i < N_INITS; i++)); do
    glass initialize --output "$OUT/inits/init_$(printf %02d "$i").xyz" \
        --density 2.5 --species Si --num-atoms 216 --min-distance 2.0 --seed "$i"
done

# 2. Unconditional generation (t = 1.0 -> 0.001)
glass generate "$EXP" --inits "$OUT/inits" --outdir "$OUT/uncond" "${COMMON[@]}"

# 3. Guided generation: unconditional pass, then guided pass (t = 0.6 -> 0.001).
#    The target is the average over the reference structures in $REF. The
#    guidance weight defaults to a value for each type (pdf 300, adf 1000,
#    xrd and nd 1). Diffraction guidance creates close contacts, which the
#    minimum-distance veto removes.
glass generate "$EXP" --inits "$OUT/inits" --outdir "$OUT/pdf" "${COMMON[@]}" \
    --guidance-type pdf --ref-path "$REF"
glass generate "$EXP" --inits "$OUT/inits" --outdir "$OUT/adf" "${COMMON[@]}" \
    --guidance-type adf --ref-path "$REF"
glass generate "$EXP" --inits "$OUT/inits" --outdir "$OUT/xrd" "${COMMON[@]}" \
    --guidance-type xrd --ref-path "$REF" --element-names Si --min-dist 2.0
glass generate "$EXP" --inits "$OUT/inits" --outdir "$OUT/nd" "${COMMON[@]}" \
    --guidance-type nd --ref-path "$REF" --element-names Si --min-dist 2.0

# 4. Compare with the reference ensemble
python "$HERE/analyze.py" --reference "$REF" \
    --set unconditional="$OUT/uncond/*/*/*_final.xyz" \
    --set pdf="$OUT/pdf/*/*/*_final.xyz" \
    --set adf="$OUT/adf/*/*/*_final.xyz" \
    --set xrd="$OUT/xrd/*/*/*/*_final.xyz" \
    --set nd="$OUT/nd/*/*/*/*_final.xyz" \
    --output "$OUT/summary.json" --figure "$OUT/structure.png"

# Animation of conditional generation: PDF guidance over the full schedule
# (t = 1.0 -> 0.001) from a random cell, without the unconditional pass
mkdir -p "$OUT/gif_init" && cp "$OUT/inits/init_00.xyz" "$OUT/gif_init/"
glass generate "$EXP" --inits "$OUT/gif_init" --outdir "$OUT/gif" --device "$DEVICE" \
    --n-runs 1 --tstep "$STEPS" --tmax 1.0 --no-uncond-prepass \
    --guidance-type pdf --ref-path "$REF" --save-traj
python "$HERE/make_gif.py" "$(ls "$OUT"/gif/*/*/00_traj.xyz | head -1)" \
    --reference "$REF" --output "$OUT/denoising.gif"

# Single-structure metrics with the CLI, for one generated sample
glass metrics "$(ls "$OUT"/pdf/*/*/00_final.xyz | head -1)" \
    --output "$OUT/metrics_pdf_00.json" --include-rings --include-tersoff
