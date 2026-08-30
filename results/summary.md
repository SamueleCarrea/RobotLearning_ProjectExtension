| Metodo | Env | Config | Run | → source | → target |
|---|---|---|---:|---:|---:|
| Baseline no randomization (target) | CustomHopper-target-v0 | n_envs=8, normalize=True, lstm=- | 3 | 1629.8 ± 40.1 | 1602.6 ± 105.5 |
| Baseline no randomization (target) | CustomHopper-target-v0 | n_envs=8, normalize=True, lstm=128 | 3 | 1293.0 ± 131.9 | 1305.5 ± 130.8 |
| Oracle (all masses) | CustomHopper-source-udr-v0 | n_envs=8, normalize=True, lstm=-, masses=all | 1 (*) | 895.2 ± 0.0 | 1185.2 ± 0.0 |
| Baseline no randomization (source) | CustomHopper-source-v0 | n_envs=8, normalize=True, lstm=128 | 3 | 1325.3 ± 171.3 | 1169.0 ± 358.0 |
| Oracle (links masses) | CustomHopper-source-udr-v0 | n_envs=8, normalize=True, lstm=-, masses=links | 3 | 1430.5 ± 256.4 | 1149.9 ± 290.4 |
| Uniform Domain Randomization (UDR) | CustomHopper-source-udr50-v0 | n_envs=8, normalize=True, lstm=128 | 1 (*) | 1204.7 ± 0.0 | 1084.2 ± 0.0 |
| Uniform Domain Randomization (UDR) | CustomHopper-source-udr25-v0 | n_envs=8, normalize=True, lstm=128 | 1 (*) | 890.2 ± 0.0 | 913.4 ± 0.0 |
| Uniform Domain Randomization (UDR) | CustomHopper-source-udr-v0 | n_envs=8, normalize=True, lstm=- | 6 | 1706.2 ± 72.6 | 891.3 ± 349.1 |
| Baseline no randomization (source) | CustomHopper-source-v0 | n_envs=8, normalize=True, lstm=- | 6 | 1723.2 ± 52.5 | 853.2 ± 522.9 |
| Uniform Domain Randomization (UDR) | CustomHopper-source-udr5-v0 | n_envs=8, normalize=True, lstm=128 | 1 (*) | 1490.3 ± 0.0 | 832.3 ± 0.0 |
| Uniform Domain Randomization (UDR) | CustomHopper-source-udr-v0 | n_envs=8, normalize=True, lstm=128 | 6 | 1119.8 ± 265.2 | 745.4 ± 341.9 |

(*) un solo seed: la std non e' affidabile, e' solo quella tra episodi di eval
