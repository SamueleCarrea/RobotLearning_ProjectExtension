# Legacy results: first probe run (models/udr_s42)

These files come from the first RecurrentPPO + UDR run, trained before the
`--seed` argument was added to `train_udr.py`. That run therefore has an
undefined seed and is not part of the 6-seed group reported in the paper.

The probe dataset and the reservoir control were originally collected from
that checkpoint. They were later recollected from `models/udr_lstm_s42`, which
is part of the 6-seed group, and only the recollected versions are reported.

Kept for auditability. Do not use these numbers.

Mapping:
  probe_robustness_control.json  <- probe_dataset_control_450.npz  (actor: udr_s42)
  probe_results_control.json     <- probe_dataset_control_450.npz  (actor: udr_s42)
  probe_controls_control.json    <- same run, no dataset field recorded
