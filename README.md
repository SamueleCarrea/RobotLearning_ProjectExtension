# Robot Learning project extension - Hopper sim-to-real

Extension of Exercise 4. Two blocks:

- **Block 1**: a recurrent UDR policy (RecurrentPPO with an LSTM) instead of
  the feedforward PPO, plus comparison baselines and an oracle policy. We also
  added a matched PPO feedforward comparison under the same conditions, to
  isolate the effect of recurrence alone.
- **Block 2**: a supervised probe that tries to read the true link masses from
  the LSTM hidden state, to check whether the policy performs implicit system
  identification. We added a control with a randomly-initialized (untrained)
  LSTM, to see how much of that decodability actually comes from training.

**Summary of the result**: none of the mechanisms we tested closes the
sim-to-real gap in this setup, and the reasons all trace back to the same
structural cause (see below). This is a project with negative results, but
with a coherent explanation and solid controls, not a hidden failure.

## Important note on the setup

Source and target differ **only** in torso mass:

| | torso | thigh | leg | foot |
|---|---|---|---|---|
| source | 2.665 | 4.058 | 2.781 | 5.316 |
| target | 3.665 | 4.058 | 2.781 | 5.316 |

UDR randomizes thigh, leg and foot, i.e. the three masses that are identical
across the two domains (the torso cannot be randomized, the assignment
forbids it). So UDR cannot close the gap directly: the torso, the only
parameter that actually differs, is never randomized and therefore never
"seen" during training.

Same reasoning applies to the oracle with `--oracle_masses links`: knowing
thigh/leg/foot says nothing about the torso. `--oracle_masses all` adds the
torso too, as a check.

## Files

```
env/custom_hopper.py          environment + UDR (configurable range)
oracle_env.py                 wrapper that adds the true masses to the observation
train.py                      baseline (no randomization), RecurrentPPO or PPO feedforward
train_udr.py                  UDR policy, RecurrentPPO or PPO feedforward
train_oracle.py               PPO oracle
collect_data.py               rollout with the trained policy -> dataset for the probe
collect_control.py            control: same dataset but with a randomly-initialized LSTM
probe_model.py                2-layer MLP, deliberately small
train_probe.py                probe training
analyze_probe_robustness.py   cross-validation + early window (no leakage)
analyze_probe_controls.py     linear probe + controls (timestep-only, shuffled labels)
eval_cross_mass.py            evaluates a checkpoint on environments with thigh/leg/foot scaled
eval_best_checkpoints.py      re-evaluates best_model checkpoints (see Limitations: not recommended)
summarize_results.py          table with mean and std across seeds
plot.py                       block 2 plots (probe)
plot_learning_curves.py       learning curves, feedforward vs recurrent
watch_episode.py              renders one episode
test_random_policy.py         sanity check on the environment with a random policy
```

`models/`, `tensorboard/`, `logs/` and the `.npz` files are not in the repo
because they are large; they can be regenerated with the commands below.

## Reproducing everything

```bash
pip install -r requirements.txt
bash run_all.sh
```

Or manually:

```bash
# baseline and UDR, RecurrentPPO, 6 seeds
for s in 42 123 7 5 11 21; do
  python train.py --env CustomHopper-source-v0 --seed $s --n_envs 8 --tag source_s$s
  python train.py --env CustomHopper-target-v0 --seed $s --n_envs 8 --tag target_s$s
  python train_udr.py --seed $s --n_envs 8 --tag lstm_s$s
done

# same comparisons with PPO feedforward, to isolate the effect of recurrence
for s in 42 123 7 5 11 21; do
  python train.py --algorithm PPO --env CustomHopper-source-v0 --seed $s --n_envs 8 --tag ff_source_s$s
  python train.py --algorithm PPO --env CustomHopper-target-v0 --seed $s --n_envs 8 --tag ff_target_s$s
  python train_udr.py --algorithm PPO --seed $s --n_envs 8 --tag ff_s$s
done

# randomization amplitude as a hyperparameter (RecurrentPPO, single seed)
for e in udr5 udr25 udr50; do
  python train_udr.py --udr_env CustomHopper-source-$e-v0 --seed 42 --n_envs 8 --tag ${e}_s42
done

# oracle: 3 seeds on links + 1 seed on all masses
for s in 42 123 7; do
  python train_oracle.py --seed $s --n_envs 8 --tag s$s
done
python train_oracle.py --seed 42 --n_envs 8 --oracle_masses all --tag all_s42

# probe: dataset from the UDR checkpoint (seed 42, part of the 6-seed group)
python collect_data.py --checkpoint models/udr_lstm_s42/udr_RecurrentPPO_final.zip \
    --episodes 450 --use_cell_state --out probe_dataset_450.npz
python train_probe.py --dataset probe_dataset_450.npz --epochs 150 \
    --out_json results/probe_results.json
python analyze_probe_robustness.py --dataset probe_dataset_450.npz --folds 5 \
    --out results/probe_robustness_results.json
python analyze_probe_controls.py --dataset probe_dataset_450.npz \
    --out results/probe_controls_results.json

# control: same trajectories, randomly-initialized LSTM encoder (3 initializations)
for e in 999 7 8; do
  python collect_control.py --mode random_lstm \
      --checkpoint models/udr_lstm_s42/udr_RecurrentPPO_final --use_cell_state \
      --episodes 450 --seed 0 --encoder_seed $e --out probe_dataset_control_e$e.npz
  python train_probe.py --dataset probe_dataset_control_e$e.npz --epochs 150 \
      --out_json results/probe_results_control_e$e.json
  python analyze_probe_robustness.py --dataset probe_dataset_control_e$e.npz \
      --out results/probe_robustness_control_e$e.json
done

# cross-mass: evaluate checkpoints on environments where thigh/leg/foot vary
# (the parameter UDR actually randomizes), instead of the torso
for s in 42 123 7 5 11 21; do
  python eval_cross_mass.py --checkpoint models/udr_lstm_s$s/udr_RecurrentPPO_final --algorithm RecurrentPPO --n_episodes 20
  python eval_cross_mass.py --checkpoint models/udr_ff_s$s/udr_PPO_final --algorithm PPO --n_episodes 20
done
for w in foot thigh leg; do
  python eval_cross_mass.py --checkpoint models/udr_lstm_s42/udr_RecurrentPPO_final --algorithm RecurrentPPO --which $w --n_episodes 20
  python eval_cross_mass.py --checkpoint models/udr_ff_s42/udr_PPO_final --algorithm PPO --which $w --n_episodes 20
done

# tables and plots
python summarize_results.py --markdown results/summary.md
python plot.py
python plot_learning_curves.py
```

## Block 1 results: the policies

Mean and standard deviation across 6 seeds (3 for the oracle), reward over 50
evaluation episodes:

| method | -> source | -> target |
|---|---:|---:|
| No-randomization baseline (target), PPO feedforward | 1629.8 ± 40.1 | 1602.6 ± 105.5 |
| No-randomization baseline (target), RecurrentPPO | 1293.0 ± 131.9 | 1305.5 ± 130.8 |
| No-randomization baseline (source), PPO feedforward | 1723.2 ± 52.5 | 853.2 ± 522.9 |
| No-randomization baseline (source), RecurrentPPO | 1325.3 ± 171.3 | 1169.0 ± 358.0 |
| UDR, PPO feedforward | 1706.2 ± 72.6 | 891.3 ± 349.1 |
| UDR, RecurrentPPO | 1119.8 ± 265.2 | 745.4 ± 341.9 |
| Oracle (true thigh/leg/foot), RecurrentPPO | 1430.5 ± 256.4 | 1149.9 ± 290.4 |
| Oracle (all masses, torso included), 1 seed | 895.2 | 1185.2 |

Three things emerge, all pointing the same way:

1. **Feedforward wins every time**, under every condition tested, with much
   smaller standard deviations. The learning curves (`plot_learning_curves.py`)
   show why: feedforward saturates within 300-400k steps, the recurrent policy
   oscillates for the full million steps without converging. At equal budget,
   recurrence pays an optimization cost that is never repaid here.
2. **UDR does not close the gap**, with either algorithm: the gain over the
   no-randomization baseline is within the seed-to-seed noise.
3. **The oracle, which receives the true masses, does not beat the
   baselines.** So information about the randomized parameters is irrelevant
   to the task: even perfect knowledge of it buys nothing on transfer.

Transfer to the target is also highly variable across seeds (reward ranges
from a few hundred to over 1600 on the exact same setup), so it should be read
as a rare event rather than a reliable property of the method.

### Cross-mass: does recurrence help when the varying parameter is the randomized one?

Idea: if the problem were only that the torso is never randomized, then on
environments where the actually-randomized parameters (thigh, leg, foot) vary,
memory should help. Result, scaling all three masses together (0.70-1.30, the
standard UDR range is 0.85-1.15), averaged over 6 seeds:

| scale | PPO feedforward | RecurrentPPO |
|---:|---:|---:|
| 0.70 | 1403.0 | 1096.4 |
| 0.85 | 1700.1 | 1133.5 |
| 1.00 | 1716.5 | 1155.1 |
| 1.15 | 1505.5 | 1137.2 |
| 1.30 | 1282.5 | 1062.2 |

Feedforward wins **here too**, everywhere, and even more clearly outside the
training range. Recurrence does not help even when the varying parameter is
exactly the inferable one: the issue is not just the torso mismatch, it is
that at these perturbation amplitudes (±15%) a single robust gait already
covers the whole range, so specializing does not pay off.

Scaling one link at a time (seed 42) shows which mass actually matters: on
feedforward, `thigh` and `leg` are flat (~1720-1745 across the whole range),
only `foot` degrades monotonically (1827 -> 1644). This is the same ranking
that emerges from the probe (see below): the foot is the only mass that
matters for control.

## Block 2 results: the probe

The probe is compared against a baseline that always predicts the mean mass.
5-fold cross-validation, split by episode:

| mass | MAE reduction (trained probe) |
|---|---:|
| thigh | 9.1% ± 2.1 |
| leg | 1.6% ± 0.7 |
| foot | 9.9% ± 1.7 |

These numbers are lower than an earlier analysis suggested (that one had a
leakage bug, since fixed). Before interpreting them, though, we need to know
how much of this decodability actually comes from training, and how much is
simply an effect of recurrence itself.

### The decisive control: a randomly-initialized LSTM

A recurrent network with never-trained weights is still a non-linear
projection of the history of observations (the "reservoir" effect), and a lot
can be decoded from it without anyone having learned anything. We collected
the same dataset (same trajectories, same targets) replacing the trained
encoder with three independent randomly-initialized encoders:

| mass | trained LSTM | randomly-initialized LSTM (3 initializations) |
|---|---:|---:|
| thigh | 9.1 ± 2.1 | 17.0-18.1 |
| leg | 1.6 ± 0.7 | 4.4-4.9 |
| foot | 9.9 ± 1.7 | 18.8-19.8 |

**The random network beats the trained one on all three masses.** The three
encoders are consistent with each other (spread of about one percentage
point), so this is not a lucky initialization. This is consistent with the
oracle result: if information about the parameters does not serve the
control task, training has no incentive to preserve it, and in fact
compresses it away.

Even the phenomenology that initially looked like the headline result (t=0 at
baseline level, the early peak on foot before thigh, the decay afterwards) is
reproduced identically by the random encoder: it is an effect of recurrence,
not of learning.

**Block 2 conclusion**: training does not produce implicit identification in
the sense we were looking for. On the contrary, it suppresses information
that the recurrent structure alone would already make available, because that
information (thigh, leg, foot) is not useful for the task. This is the same
message as Block 1, seen from another angle: what gets randomized is not what
matters for transfer, and the model "knows" it.

The basic controls remain valid, and hold on the control dataset just as they
do on the real one: at t=0 the probe is at baseline level (no dynamical
information yet), a probe using only the timestep scores ~0% (so it is not
survivorship bias), and with shuffled labels it scores ~0% (no leakage).

## Limitations

- `max_episode_steps` is 500, not 1000, so rewards are not directly comparable
  to standard Hopper benchmarks.
- Past ~200 steps only a few episodes are still alive, and probe estimates
  become noisy; for this reason the temporal analysis is limited to the
  window where 90% of episodes are still alive.
- **`vecnormalize.pkl` is only saved at the end of training**, so it matches
  the `_final` checkpoint, not `best_model`. We attempted to re-evaluate the
  `best_model` checkpoints as a robustness check (`eval_best_checkpoints.py`),
  but the results turned out to be unreliable precisely because of this
  mismatch: some intermediate checkpoints, evaluated with non-matching
  normalization statistics, collapse to near-zero reward despite not being
  weak checkpoints. Fixing this would require a callback that saves the
  statistics on every improvement, plus retraining every model from scratch,
  which was not feasible given the time available; we flag it as future work.
  Every number reported in this README comes from the `_final` checkpoints.
- The `udr5`/`udr25`/`udr50` variants (randomization amplitude) and the
  all-masses oracle are single-seed: indicative, not conclusive.
- The random-LSTM control uses 3 initializations: enough given the very low
  observed spread, but not an exhaustive estimate.

## References

- Alain & Bengio, Understanding intermediate layers using linear classifier probes, 2016
- Hewitt & Liang, Designing and Interpreting Probes with Control Tasks, 2019
- Kumar et al., RMA: Rapid Motor Adaptation for Legged Robots, 2021
- Peng et al., Sim-to-Real Transfer of Robotic Control with Dynamics Randomization, 2018