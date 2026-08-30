# Robot Learning project extension - Hopper sim-to-real

Estensione dell'Esercizio 4. Due blocchi:

- **Blocco 1**: policy UDR ricorrente (RecurrentPPO con LSTM) invece della PPO
  feedforward, piu' le baseline di confronto e una policy oracolo.
- **Blocco 2**: un probe supervisionato che prova a leggere le masse vere dallo
  hidden state dell'LSTM, per vedere se la policy fa system identification
  implicita.

## Nota importante sul setup

Source e target differiscono **solo** per la massa del torso:

| | torso | thigh | leg | foot |
|---|---|---|---|---|
| source | 2.665 | 4.058 | 2.781 | 5.316 |
| target | 3.665 | 4.058 | 2.781 | 5.316 |

La UDR randomizza thigh, leg e foot, cioe' le tre masse che sono uguali nei due
domini (il torso non si puo' randomizzare, lo dice la traccia). Quindi la UDR
non puo' chiudere il gap direttamente, puo' solo dare robustezza generica: un
miglioramento piccolo e rumoroso e' il risultato atteso.

Stessa cosa per l'oracolo con `--oracle_masses links`: conoscere thigh/leg/foot
non dice niente sul torso. Con `--oracle_masses all` si aggiunge anche il torso
per verificarlo.

## File

```
env/custom_hopper.py          ambiente + UDR (range configurabile)
oracle_env.py                 wrapper che aggiunge le masse vere all'osservazione
train.py                      baseline senza randomizzazione
train_udr.py                  policy ricorrente con UDR
train_oracle.py               oracolo PPO
collect_data.py               rollout -> dataset per il probe
probe_model.py                MLP a 2 layer
train_probe.py                training del probe
analyze_probe_robustness.py   cross-validation + finestra iniziale
analyze_probe_controls.py     probe lineare + controlli
summarize_results.py          tabella con media e std tra i seed
plot.py                       grafici
watch_episode.py              visualizza un episodio
```

`models/`, `tensorboard/`, `logs/` e i `.npz` non sono nel repo perche' pesanti,
si rigenerano con i comandi sotto.

## Come rilanciare tutto

```bash
pip install -r requirements.txt
bash run_all.sh
```

Oppure a mano:

```bash
# baseline e UDR sugli stessi seed
for s in 42 123 7; do
  python train.py --env CustomHopper-source-v0 --seed $s --n_envs 8 --tag source_s$s
  python train.py --env CustomHopper-target-v0 --seed $s --n_envs 8 --tag target_s$s
  python train_udr.py --seed $s --n_envs 8 --tag s$s
done

# ampiezza della randomizzazione come iperparametro
for e in udr5 udr25 udr50; do
  python train_udr.py --udr_env CustomHopper-source-$e-v0 --seed 42 --n_envs 8 --tag ${e}_s42
done

# oracolo
python train_oracle.py --seed 42 --n_envs 8 --tag s42
python train_oracle.py --seed 42 --n_envs 8 --oracle_masses all --tag all_s42

# probe, dal checkpoint UDR
python collect_data.py --checkpoint models/udr_s42/udr_RecurrentPPO_final.zip \
    --episodes 450 --use_cell_state --out probe_dataset_450.npz
python train_probe.py --dataset probe_dataset_450.npz --epochs 150
python analyze_probe_robustness.py --dataset probe_dataset_450.npz --folds 5
python analyze_probe_controls.py --dataset probe_dataset_450.npz

# tabelle e grafici
python summarize_results.py --markdown results/summary.md
python plot.py
```

## Risultato del blocco 2

Il probe viene confrontato con una baseline che predice sempre la massa media.
Cross-validation a 5 fold, split per episodio:

| massa | riduzione del MAE | quando emerge |
|---|---|---|
| thigh | ~15% | dopo 50-100 step |
| leg | ~3% | mai, non identificabile |
| foot | ~32% | entro 20 step |

Tre controlli escludono le spiegazioni banali: a t=0 il probe e' al livello
della baseline (la fisica non ha ancora prodotto informazione), un probe che
usa solo il timestep fa ~0% (quindi non e' survivorship bias), e con le
etichette permutate fa ~0% (nessun leakage). Un probe lineare recupera quasi
tutto sul thigh ma meno della meta' sul foot.

L'identificazione implicita quindi c'e' ma non e' uguale per tutti i parametri:
dipende da quanto ciascuno si vede nella dinamica. Il piede tocca terra a ogni
appoggio, la coscia si vede solo dallo swing, la gamba resta confusa in mezzo.

## Limiti

- `max_episode_steps` e' 500 e non 1000, quindi i reward non si confrontano coi
  benchmark Hopper standard.
- Oltre i ~200 step restano pochi episodi vivi e le stime sono rumorose, per
  questo l'analisi temporale si ferma alla finestra col 90% di episodi vivi.
- `vecnormalize.pkl` viene salvato a fine training, quindi corrisponde al
  modello `_final` e non a `best_model`.

## Riferimenti

- Alain & Bengio, Understanding intermediate layers using linear classifier probes, 2016
- Kumar et al., RMA: Rapid Motor Adaptation for Legged Robots, 2021
- Peng et al., Sim-to-Real Transfer of Robotic Control with Dynamics Randomization, 2018
