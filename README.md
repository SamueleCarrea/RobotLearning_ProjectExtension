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

Grouped by what part of the pipeline they belong to, roughly in the order
you would run them.

### Environment and training (Block 1)

```
env/custom_hopper.py    MuJoCo Hopper environment + UDR. The randomization
                         range is picked via the env id (e.g.
                         CustomHopper-source-udr-v0 = standard +-15%,
                         -udr5/-udr25/-udr50 = +-5%/25%/50% variants used
                         for the amplitude ablation).

oracle_env.py            Gym wrapper used only by the oracle: concatenates
                         the true masses to the observation.
                         --oracle_masses links -> thigh/leg/foot (the ones
                         UDR randomizes); all -> also adds the torso.

train.py                 Baseline policy, NO randomization. Trains on either
                         CustomHopper-source-v0 or -target-v0.
                         --algorithm RecurrentPPO|PPO selects the architecture.

train_udr.py              Same as train.py but on the UDR environment: at
                         every reset, thigh/leg/foot are resampled inside the
                         chosen range. Same --algorithm flag.

train_oracle.py           PPO feedforward wrapped with oracle_env.py, trained
                         on the UDR environment. --oracle_masses links|all.
```

All four training scripts save a checkpoint (`.zip`), the matching
`vecnormalize.pkl` (needed to reproduce the exact input normalization at eval
time), and a `_results.json` in `results/` with reward on source and target.

### Probe pipeline (Block 2)

```
collect_data.py               Rolls out a trained policy (e.g. udr_lstm_s42)
                              for N episodes. At every step, saves the LSTM
                              hidden state together with the true link masses
                              of that episode -> probe_dataset_*.npz. This is
                              the training set for the probe.

collect_control.py            Same rollout (same acting policy, same
                              trajectories, same true masses), but the hidden
                              state comes from a SEPARATE, randomly
                              initialized, never-trained LSTM encoder.
                              --encoder_seed only changes that encoder's
                              init, not the trajectories. This produces the
                              "reservoir" control dataset.

probe_model.py                 Defines ProbeMLP: 2 hidden layers, 32 units,
                              deliberately small. Rationale: if even this
                              weak network can decode the masses, the
                              information was already close to linearly
                              present in the hidden state (Alain & Bengio,
                              2016).

train_probe.py                 Trains a ProbeMLP on one dataset (trained or
                              control). Saves the weights to
                              checkpoints/probes/*.pt and the MAE-vs-baseline
                              metrics to results/*.json. The baseline it is
                              compared against always predicts the
                              training-set mean mass.

analyze_probe_robustness.py     5-fold cross-validation, split by WHOLE
                              episode (never by individual step, otherwise
                              steps from the same episode leak between train
                              and test). Also runs the early-time-window
                              analysis: how decodability evolves over the
                              first steps, restricted to the window where at
                              least 90% of episodes are still alive.

analyze_probe_controls.py       The sanity checks that make the whole probe
                              result trustworthy: (1) a LINEAR probe, not
                              just the MLP; (2) a probe that sees ONLY the
                              timestep, no hidden state - should score ~0%,
                              rules out the result being an artifact of
                              episode length (survivorship bias); (3) a probe
                              trained with SHUFFLED mass labels - should
                              score ~0%, rules out data leakage.

check_dataset_meta.py           Reads the metadata stored inside a .npz
                              (which policy checkpoint generated it, which
                              seed, which encoder) and verifies that a
                              trained/control pair actually comes from the
                              same underlying rollout. Called automatically
                              inside run_probe_sweep.sh as a provenance
                              guard - this is what caught the udr_s42
                              seeding bug described in `results/legacy_udr_s42/`.
```

### Cross-mass evaluation (Block 1, extra check)

```
eval_cross_mass.py    Loads a trained checkpoint and evaluates it on
                      environments where thigh/leg/foot are scaled by a
                      fixed factor (default 0.70-1.30, torso left at its
                      normal value). Tests whether recurrence helps on the
                      parameter that was ACTUALLY randomized, as opposed to
                      the torso mismatch between source and target.
```

### Aggregation

```
summarize_results.py     Reads every results/*_results.json and produces
                         results/summary.md: mean +/- std across seeds,
                         Block 1 policy comparison table.

summarize_crossmass.py    Reads every results/crossmass_*.json and produces
                         results/crossmass_summary.md, aggregated across the
                         6 seeds.

summarize_probe.py         Reads every results/probe_*.json and produces
                         results/probe_summary.md, including the
                         reward-vs-decodability correlation across the 3
                         policy seeds.
```

### Figures

```
plot.py                   Generates every figure in images/ used in the
                          paper: policy comparison, cross-mass, probe
                          controls, probe time windows.

plot_learning_curves.py    Generates images/learning_curves.png,
                          feedforward vs recurrent training curves.
```

### Orchestration

```
run_probe_sweep.sh    Runs the whole Block 2 pipeline across the 3 policy
                      seeds and, with WITH_CONTROLS=1, the 3 encoder seeds:
                      collect -> train probe -> analyze -> check provenance
                      -> aggregate. Can be run standalone.

run_all.sh            Master script: Block 1 (all seeds, both algorithms,
                      oracle), cross-mass eval, then calls
                      run_probe_sweep.sh for Block 2. Reproduces every
                      number in this README end to end.
```

`models/`, `tensorboard/`, `logs/` and the `.npz` files are not in the repo
because they are large; they can be regenerated with the commands below.

## Trained models

`checkpoints/` contains one representative checkpoint per key configuration
(seed 42), together with the matching `vecnormalize.pkl`, for reproducibility.
`checkpoints/probes/` contains the trained probe weights: one `.pt` per
policy seed (`probe_mlp_s42.pt`, `_s123.pt`, `_s7.pt`) and one per reservoir
control encoder seed (`probe_mlp_control_e7.pt`, `_e8.pt`, `_e999.pt`).
Full multi-seed results (mean/std across 6 seeds) are in `results/`; the
remaining checkpoints are not included to keep the repository size
reasonable and can be regenerated with the commands above.

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

# probe: multi-seed sweep (trained policies + reservoir controls), see
# run_probe_sweep.sh for what each step does and why encoder seeds are kept
# separate from policy seeds
WITH_CONTROLS=1 bash run_probe_sweep.sh

# aggregation across seeds, including the reward-vs-decodability check
python summarize_probe.py --markdown results/probe_summary.md
python summarize_crossmass.py --markdown results/crossmass_summary.md

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
| Oracle (true thigh/leg/foot), PPO feedforward | 1430.5 ± 256.4 | 1149.9 ± 290.4 |
| Oracle (all masses, torso included), PPO feedforward, 1 seed | 895.2 | 1185.2 |

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
standard UDR range is 0.85-1.15), mean ± std over 6 seeds
(`summarize_crossmass.py`):

| scale | PPO feedforward | RecurrentPPO |
|---:|---:|---:|
| 0.70 | 1403 ± 549 | 1086 ± 341 |
| 0.85 | 1700 ± 153 | 1112 ± 299 |
| 1.00 | 1716 ± 71  | 1096 ± 309 |
| 1.15 | 1506 ± 188 | 1075 ± 328 |
| 1.30 | 1283 ± 398 | 1032 ± 247 |

Feedforward wins **here too**, everywhere, and even more clearly outside the
training range. Recurrence does not help even when the varying parameter is
exactly the inferable one: the issue is not just the torso mismatch, it is
that at these perturbation amplitudes (±15%) a single robust gait already
covers the whole range, so specializing does not pay off. Note the standard
deviation on RecurrentPPO (up to ±341) is itself large relative to the gap
between scales, which is the same seed-to-seed instability documented for the
probe below, not a separate issue.

Scaling one link at a time (seed 42 only, indicative) shows which mass
actually matters: on feedforward, `thigh` and `leg` are flat (~1714-1745
across the whole range), only `foot` degrades monotonically
(1827 -> 1644). This is the same ranking that emerges from the probe (see
below): the foot is the only mass that matters for control.

## Block 2 results: the probe

The probe is compared against a baseline that always predicts the mean mass.
5-fold cross-validation, split by episode, one dataset per UDR RecurrentPPO
policy seed:

| policy seed | reward (source→target) | thigh | leg | foot |
|---|---:|---:|---:|---:|
| s42  | 1211 (best-converged) | 9.1  | 1.6 | 9.8  |
| s123 | 821                   | 16.6 | 6.2 | 20.8 |
| s7   | 395 (least-converged) | 33.2 | 6.0 | 43.9 |

MAE reduction (%) over the mean-predictor baseline. We do not collapse this
into mean ± std: with only 3 seeds that would hide the pattern below, which is
the actual finding.

### The decisive control: a randomly-initialized LSTM

A recurrent network with never-trained weights is still a non-linear
projection of the history of observations (the "reservoir" effect), and a lot
can be decoded from it without anyone having learned anything. We collected
the same trajectories with the acting policy fixed to `udr_lstm_s42` and the
encoder replaced by three independent randomly-initialized encoders:

| mass | trained LSTM (3 policy seeds) | random LSTM (3 encoder seeds) |
|---|---:|---:|
| thigh | 19.6 ± 12.3 | 17.8 ± 0.5 |
| leg   | 4.6 ± 2.6   | 5.0 ± 0.2  |
| foot  | 24.9 ± 17.4 | 19.9 ± 0.8 |

At face value the trained mean is not below the random one anymore, unlike
what a single seed (s42) suggested. But look at the spread: **the random
encoders agree with each other to within one percentage point, the trained
seeds disagree by up to 24 points.** Whatever is happening on the trained side
is not "training suppresses information" as a fixed effect. It is seed-dependent, and dramatically so.

### What the variance actually is

The three policy seeds differ enormously in how well they solved the task
(reward source→target: 1211 / 821 / 395). Decodability tracks that ordering
almost exactly, and the direction is the same on all three masses
independently:

| mass | Pearson r (reward vs. decodability, n=3) | p |
|---|---:|---:|
| thigh | −0.982 | 0.121 |
| leg   | −0.824 | 0.383 |
| foot  | −0.984 | 0.113 |

With n=3 no single p-value is significant, and we are not claiming one is:
the evidence is that the sign is negative and consistent across three
independently-fit masses, not the p-value of any one of them. The reading we
consider best supported: **the better a policy converges, the more it
compresses away mass information that is not useful for the task.** The seed
that converged best (s42, reward 1211) is the only one that goes clearly
below the reservoir baseline on every mass. The seed that barely converged
(s7, reward 395) still resembles an under-trained network that has not yet
compressed its state toward what the task needs, so it decodes even better
than a random one.

This refines rather than overturns the original message: it is not that
training in general destroys decodable information, it is that **convergence**
does, and with a single seed there was no way to tell the two apart. It is
also consistent with the oracle result from another angle: if the randomized
parameters mattered for control, better-converged policies would have no
reason to compress them away.

Even the phenomenology that initially looked like the headline result (t=0 at
baseline level, the early peak on foot before thigh, the decay afterwards) is
reproduced by the random encoder as well as by the trained ones: it is an
effect of recurrence, not of learning (see `probe_windows_trained_vs_random.png`).

**Block 2 conclusion**: training does not produce a reliable, seed-independent
implicit identification signal. What we find instead is that decodability is
tied to how well the policy converges, in the direction predicted by the rest
of the project: convergence on this task does not require, and actively
discards, information about the randomized masses.

The basic controls remain valid on every dataset checked: at t=0 the probe is
at baseline level (no dynamical information yet), a probe using only the
timestep scores ~0% (so it is not survivorship bias), and with shuffled
labels it scores ~0% (no leakage).

## Limitations

- The reward-decodability correlation is qualitative evidence (consistent
  sign across 3 independently-fit masses), not a statistically significant
  result (n=3, p>0.1 on every mass taken alone). More UDR RecurrentPPO seeds
  would be needed to test it properly; we did not have the compute budget for
  that within the project deadline.

- `max_episode_steps` is 500, not 1000, so rewards are not directly comparable
  to standard Hopper benchmarks.
- Past ~200 steps only a few episodes are still alive, and probe estimates
  become noisy; for this reason the temporal analysis is limited to the
  window where 90% of episodes are still alive.
- **`vecnormalize.pkl` is only saved at the end of training**, so it matches
  the `_final` checkpoint, not `best_model`. We attempted to re-evaluate the
  `best_model` checkpoints as a robustness check, but the results turned out
  to be unreliable precisely because of this mismatch: some intermediate
  checkpoints, evaluated with non-matching normalization statistics, collapse
  to near-zero reward despite not being weak checkpoints. Fixing this would
  require a callback that saves the statistics on every improvement, plus
  retraining every model from scratch, which was not feasible given the time
  available; we flag it as future work. Every number reported in this README
  comes from the `_final` checkpoints.
- The `udr5`/`udr25`/`udr50` variants (randomization amplitude) and the
  all-masses oracle are single-seed: indicative, not conclusive.
- The random-LSTM control uses 3 initializations: enough given the very low
  observed spread, but not an exhaustive estimate.

## References

- Alain & Bengio, Understanding intermediate layers using linear classifier probes, 2016
- Hewitt & Liang, Designing and Interpreting Probes with Control Tasks, 2019
- Kumar et al., RMA: Rapid Motor Adaptation for Legged Robots, 2021
- Peng et al., Sim-to-Real Transfer of Robotic Control with Dynamics Randomization, 2018
