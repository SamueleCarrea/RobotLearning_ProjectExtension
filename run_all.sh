#!/usr/bin/env bash
# Full pipeline. One 1M-timestep run with n_envs=8 takes roughly 30-50 min on
# CPU, so the whole thing is several hours: run it in pieces.
#
# Naming convention (this matters, the probe sweep depends on it):
#   models/udr_lstm_s<SEED>/    RecurrentPPO + UDR
#   models/udr_ff_s<SEED>/      PPO feedforward + UDR
#   models/baseline_source_s<SEED>/, models/baseline_ff_source_s<SEED>/, ...
set -euo pipefail

SEEDS="${SEEDS:-42 123 7 5 11 21}"
TIMESTEPS="${TIMESTEPS:-1000000}"
NENVS="${NENVS:-8}"

# ---- Block 1: policies ----------------------------------------------------
for s in $SEEDS; do
  # recurrent
  python train.py --env CustomHopper-source-v0 --seed "$s" --n_envs "$NENVS" \
      --timesteps "$TIMESTEPS" --tag "source_s$s"
  python train.py --env CustomHopper-target-v0 --seed "$s" --n_envs "$NENVS" \
      --timesteps "$TIMESTEPS" --tag "target_s$s"
  python train_udr.py --seed "$s" --n_envs "$NENVS" --timesteps "$TIMESTEPS" \
      --tag "lstm_s$s"

  # matched feedforward, to isolate the effect of recurrence alone
  python train.py --algorithm PPO --env CustomHopper-source-v0 --seed "$s" \
      --n_envs "$NENVS" --timesteps "$TIMESTEPS" --tag "ff_source_s$s"
  python train.py --algorithm PPO --env CustomHopper-target-v0 --seed "$s" \
      --n_envs "$NENVS" --timesteps "$TIMESTEPS" --tag "ff_target_s$s"
  python train_udr.py --algorithm PPO --seed "$s" --n_envs "$NENVS" \
      --timesteps "$TIMESTEPS" --tag "ff_s$s"
done

# randomization amplitude as a hyperparameter (single seed, indicative only)
for e in udr5 udr25 udr50; do
  python train_udr.py --udr_env "CustomHopper-source-$e-v0" --seed 42 \
      --n_envs "$NENVS" --timesteps "$TIMESTEPS" --tag "${e}_s42"
done

# oracle
for s in 42 123 7; do
  python train_oracle.py --seed "$s" --n_envs "$NENVS" --timesteps "$TIMESTEPS" --tag "s$s"
done
python train_oracle.py --seed 42 --n_envs "$NENVS" --timesteps "$TIMESTEPS" \
    --oracle_masses all --tag all_s42

# ---- cross-mass evaluation ------------------------------------------------
for s in $SEEDS; do
  python eval_cross_mass.py --checkpoint "models/udr_lstm_s$s/udr_RecurrentPPO_final" \
      --algorithm RecurrentPPO --n_episodes 20
  python eval_cross_mass.py --checkpoint "models/udr_ff_s$s/udr_PPO_final" \
      --algorithm PPO --n_episodes 20
done
for w in thigh leg foot; do
  python eval_cross_mass.py --checkpoint "models/udr_lstm_s42/udr_RecurrentPPO_final" \
      --algorithm RecurrentPPO --which "$w" --n_episodes 20
  python eval_cross_mass.py --checkpoint "models/udr_ff_s42/udr_PPO_final" \
      --algorithm PPO --which "$w" --n_episodes 20
done

# ---- Block 2: probe -------------------------------------------------------
# Trained side over 3 policy seeds, reservoir control over 3 encoder seeds.
# run_probe_sweep.sh also runs the provenance check, the aggregation and the
# figures, so nothing else is needed after this.
WITH_CONTROLS=1 bash run_probe_sweep.sh
