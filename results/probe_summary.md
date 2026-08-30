| mass | trained LSTM (n=3 policy seeds) | random LSTM (n=3 encoder seeds) |
|---|---:|---:|
| thigh | 19.6 ± 12.3 | 17.8 ± 0.5 |
| leg | 4.6 ± 2.6 | 5.0 ± 0.2 |
| foot | 24.9 ± 17.4 | 19.9 ± 0.8 |

MAE reduction (%) over a baseline that always predicts the training-set mean mass. Std is across runs, not across CV folds.

With only 3 policy seeds, mean ± std hides a real pattern. Per-seed values:

| policy seed | thigh | leg | foot |
|---|---:|---:|---:|
| s7 | 33.2 | 6.0 | 43.9 |
| s42 | 9.1 | 1.6 | 9.8 |
| s123 | 16.6 | 6.2 | 20.8 |

Reward vs. decodability, 3 policy seeds (source_to_target reward vs. MAE-reduction %):

| mass | Pearson r | p (n=3, not significant alone) |
|---|---:|---:|
| thigh | -0.982 | 0.121 |
| leg | -0.824 | 0.383 |
| foot | -0.984 | 0.113 |

Negative r on all three masses: the seeds that converge to a higher reward are the ones from which less mass information can be decoded. n=3 is too small for the p-value to be significant on its own; the evidence is the consistent sign across all three independently-fit masses, not the p-value.
