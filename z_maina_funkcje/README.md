# z_maina_funkcje — CLI (skrypty plgrid_*)

Minimalne punkty wejścia do serwisów `rl_module` / `test_module`: trening, Optuna, sekwencja tezy, porównania oraz narzędzia poza protokołem.

Uruchomienie z **roota** repozytorium.

## Uruchomienie wszystkich ramion tezy (R0–M3)

Pełna instrukcja z opisem artefaktów: [`docs/protokol-badawczy.md`](../docs/protokol-badawczy.md) (sekcja *Uruchomienie wszystkich ramion tezy*). Skrót:

```bash
# 1) UL (R3/M3) — raz
python z_maina_funkcje/plgrid_regime_lock.py

# 2) Pretrain PPO — rl potem rlul (sekwencyjnie; nie równolegle)
python z_maina_funkcje/plgrid_train_agent_ppo.py --all
python z_maina_funkcje/plgrid_train_agent_ppo.py --all --sim-mode rlul

# 3) Lock R1 (baza R1/M1)
python z_maina_funkcje/plgrid_simulation_tuner.py --all-seeds --stop-loss 0.03 --lock-tag final

# 4) Walidacja configu
python z_maina_funkcje/plgrid_test_sequence.py --preflight

# 5) Wszystkie ramiona × seedy (A1–B10) — thesis_sequence_config.json
python z_maina_funkcje/plgrid_test_sequence.py

# 6) Porównanie
python z_maina_funkcje/plgrid_thesis_compare.py --run-label thesis --portfolio-mode reset --report
python z_maina_funkcje/plgrid_seed_mean_sims.py --label thesis
```

Config: `shared/files/thesis_sequence_config.json`. Wyniki: `docs/wyniki-kampanii-tezy.md`.

## Skrypty

### Teza — tor główny

| Skrypt | Rola |
|--------|------|
| `plgrid_train_agent_ppo.py` | Pretrain PPO (`THESIS_SEEDS`, `PPO_HPARAMS`, `MP_CONST`); lock `train.lock` |
| `plgrid_simulation_tuner.py` | R1 LOCK — Optuna MP flat 6F; `--lock-tag final` nie nadpisuje bazowego locka |
| `plgrid_test_sequence.py` | Sekwencja walk-forward (ramiona R0–R3, M1–M3) z configu tezy |
| `plgrid_thesis_compare.py` | Porównanie runów (folds → tabela) |
| `plgrid_seed_mean_sims.py` | Uśrednienie symulacji po seedach |
| `plgrid_regime_lock.py` | Fit/lock modelu reżimów (`regime_lock.json`) |
| `plgrid_train_eval_watcher.py` | Watcher ewaluacji podczas treningu (daemon) |
| `plgrid_train_ppo_simulation.py` | Symulacje powiązane z wytrenowanym PPO |

### Inne algorytmy / legacy

| Skrypt | Rola |
|--------|------|
| `plgrid_train_agent_sac.py` / `plgrid_sac_tuner.py` | Trening / Optuna SAC |
| `plgrid_train_agent_dqn.py` / `plgrid_dqn_tuner.py` | Trening / Optuna DQN |
| `plgrid_ppo_tuner.py` | Optuna hiperparametrów PPO (osobna ścieżka od pretrainu tezy) |
| `plgrid_simulation_basic.py` | Jednorazowa / podstawowa symulacja |
| `plgrid_lstm_multi_simulation.py` | Multi-sim / tuning po modelach LSTM |
| `plgrid_observation_space_importance.py` | Analiza ważności cech obserwacji |
| `plgrid_fetch_raw_klines.py` | Pobranie surowych świec Binance → `shared/files/raw_data/` |

## Uwagi operacyjne

- Seedy tezy pochodzą z `rl_module.services.thesis.assets.THESIS_SEEDS` (nie z wartości wpisanych na stałe w skryptach CLI).
- Równoległe dwa treningi PPO na tym samym `actual_training_logs` są niedozwolone (`train.lock`).
- Lock R1 używany w tezie: `r1_lock_final_seed{S}_sl003.json` (patrz `thesis_sequence_config.json`).
- Wyniki: `docs/wyniki-kampanii-tezy.md`.
