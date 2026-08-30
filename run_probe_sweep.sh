#!/usr/bin/env bash
# Probe sweep over multiple POLICY seeds.
#
# Why: the reservoir control varies the encoder initialization (3 random
# encoders) but the trained side was a single policy (seed 42). That
# asymmetry invites the obvious objection "maybe that one trained LSTM was
# unlucky". This runs the whole probe pipeline on independently trained
# policies so the trained side has its own error bar.
#
# The episode seed is IDENTICAL across every collection (EP_SEED), so the
# rollouts are as comparable as they can be given that different policies
# produce different trajectories.
#
# Usage:
#   bash run_probe_sweep.sh                 # seeds 42 123 7
#   SEEDS="42 123 7 5 11 21" bash run_probe_sweep.sh
#   WITH_CONTROLS=1 bash run_probe_sweep.sh # also recollect the 3 reservoir controls
#
# Cost: roughly 15-25 min per collection on an M2 laptop, plus ~2 min per
# probe fit. Default configuration is 3 collections.

set -euo pipefail

SEEDS="${SEEDS:-42 123 7}"
EPISODES="${EPISODES:-450}"
EP_SEED="${EP_SEED:-0}"
EPOCHS="${EPOCHS:-150}"
MODELS_DIR="${MODELS_DIR:-models}"
WITH_CONTROLS="${WITH_CONTROLS:-0}"
CONTROL_POLICY_SEED="${CONTROL_POLICY_SEED:-42}"
ENCODER_SEEDS="${ENCODER_SEEDS:-999 7 8}"

mkdir -p results

ckpt_for() {  # $1 = policy seed
  echo "${MODELS_DIR}/udr_lstm_s$1/udr_RecurrentPPO_final"
}

echo "=============================================================="
echo "Probe sweep"
echo "  policy seeds : ${SEEDS}"
echo "  episodes     : ${EPISODES}   episode seed: ${EP_SEED}"
echo "  controls     : ${WITH_CONTROLS}"
echo "=============================================================="

# ---- pre-flight: every checkpoint must exist before we start ---------------
missing=0
for s in ${SEEDS}; do
  c="$(ckpt_for "$s")"
  if [ ! -f "${c}.zip" ]; then
    echo "MISSING: ${c}.zip"
    missing=1
  fi
  if [ ! -f "$(dirname "$c")/vecnormalize.pkl" ]; then
    echo "MISSING: $(dirname "$c")/vecnormalize.pkl"
    echo "  without it the collected hidden states are not the real ones"
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  echo
  echo "Aborting: train the missing UDR RecurrentPPO seeds first, e.g."
  echo "  python train_udr.py --seed 123 --n_envs 8 --tag lstm_s123"
  exit 1
fi
echo "All checkpoints found."
echo

# ---- trained policies -----------------------------------------------------
for s in ${SEEDS}; do
  c="$(ckpt_for "$s")"
  ds="probe_dataset_s${s}.npz"
  echo "--------------------------------------------------------------"
  echo "[policy seed ${s}] collecting hidden states from ${c}"
  echo "--------------------------------------------------------------"
  python collect_data.py \
      --checkpoint "${c}.zip" \
      --episodes "${EPISODES}" \
      --use_cell_state \
      --seed "${EP_SEED}" \
      --out "${ds}"

  python train_probe.py \
      --dataset "${ds}" \
      --epochs "${EPOCHS}" \
      --out_model "probe_mlp_s${s}.pt" \
      --out_json "results/probe_results_s${s}.json"

  python analyze_probe_robustness.py \
      --dataset "${ds}" \
      --folds 5 \
      --out "results/probe_robustness_s${s}.json"

  python analyze_probe_controls.py \
      --dataset "${ds}" \
      --out "results/probe_controls_s${s}.json"
done

# ---- reservoir controls (optional recollection) ---------------------------
if [ "${WITH_CONTROLS}" = "1" ]; then
  c="$(ckpt_for "${CONTROL_POLICY_SEED}")"
  for e in ${ENCODER_SEEDS}; do
    ds="probe_dataset_control_e${e}.npz"
    echo "--------------------------------------------------------------"
    echo "[encoder seed ${e}] random-LSTM control on policy s${CONTROL_POLICY_SEED}"
    echo "--------------------------------------------------------------"
    python collect_control.py \
        --mode random_lstm \
        --checkpoint "${c}" \
        --episodes "${EPISODES}" \
        --use_cell_state \
        --seed "${EP_SEED}" \
        --encoder_seed "${e}" \
        --out "${ds}"

    python train_probe.py \
        --dataset "${ds}" \
        --epochs "${EPOCHS}" \
        --out_model "probe_mlp_control_e${e}.pt" \
        --out_json "results/probe_results_control_e${e}.json"

    python analyze_probe_robustness.py \
        --dataset "${ds}" \
        --folds 5 \
        --out "results/probe_robustness_control_e${e}.json"

    python analyze_probe_controls.py \
        --dataset "${ds}" \
        --out "results/probe_controls_control_e${e}.json"
  done
fi

# ---- verify provenance, then aggregate ------------------------------------
echo
echo "=============================================================="
echo "Provenance check"
echo "=============================================================="
python check_dataset_meta.py --glob "probe_dataset_*.npz"

echo
echo "=============================================================="
echo "Aggregation"
echo "=============================================================="
python summarize_probe.py --markdown results/probe_summary.md
python summarize_crossmass.py --markdown results/crossmass_summary.md
python summarize_results.py --markdown results/summary.md
python plot.py
python plot_learning_curves.py || true

echo
echo "Done. Figures in images/, tables in results/*.md"
