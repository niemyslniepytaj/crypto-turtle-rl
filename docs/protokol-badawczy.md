# Protokół badawczy (R0–M3)

Zakres **protokołu eksperymentalnego**: ramiona R0–R3 / M1–M3, seedy, locki, walk-forward, porównania. Odrębny od narzędzi poza protokołem (realtime, testnet, eksploracja SAC/DQN/LSTM multi) — [poza-protokolem.md](poza-protokolem.md).

Przegląd całego repozytorium: [opis-projektu.md](opis-projektu.md).

---

## Cel

Porównanie strategii na tym samym portfelu ETH+BTC i tych samych oknach:

| Arm | Sens |
|-----|------|
| R0 | Buy & Hold (baseline) |
| R1 | MP static — knoby 6F z Optuny na train (LOCK), zamrożone na eval |
| R2 | RL frozen — pretrain PPO, bez dalszego uczenia na foldach |
| R3 | RL + UL frozen — jak R2 + one-hot reżimu rynkowego |
| M1 | MP + Optuna rolling (retune per fold) |
| M2 | RL continue-learn na foldach |
| M3 | RL+UL + adaptacja |

Kontrakt: `rl_module/services/thesis/` (`arms.py`, `assets.py`, `knobs.py`, …).

---

## Stałe protokołu (bez zmian doraźnych)

| Co | Gdzie |
|----|--------|
| Seedy `396379`, `951830`, `246215` | `thesis/assets.py` → `THESIS_SEEDS` |
| CSV ETH/BTC 1m | `THESIS_DATA_FILES` → `shared/files/processed_simulation_data/` |
| Modele MP | `THESIS_MODEL_PATHS` → `shared/models/…` |
| Okna A1/A2 / foldy | `shared/files/windows.json` |
| Config sekwencji WF | `shared/files/thesis_sequence_config.json` |
| `MP_FREE` (6F) / `MP_CONST` / `PPO_HPARAMS` | `thesis/knobs.py` |
| Lock reżimów K | `shared/files/regime_lock.json` + `regime_model_train.joblib` |

---

## Kod — tor badawczy

### `rl_module/services/thesis/`

Cały pakiet: knoby, arms, costs, analysis, compare, walkforward helpers, preflight, repro, export, seed_mean, regime_labels, report_tex.

### Serwisy RL / Optuna pod protokół

| Moduł | Rola w badaniach |
|-------|------------------|
| `train_agent_ppo_service.py` | Pretrain PPO (R2/R3/M2/M3) |
| `simulation_tuner_service.py` | Optuna MP → R1 LOCK / M1 |
| `walkforward_service.py` | Orkiestracja foldów |
| `regime_service.py` + `rl/regime_obs_wrapper.py` | UL reżimów (R3/M3) |
| `thesis_artifacts.py` | Zapis locków / artefaktów Optuna |
| `sequence_export.py` | Eksport sekwencji (gzip JSON) |
| `rl/obs_rms_utils.py`, `training_callback.py` | Normalizacja obs, `sim_best` przy pretrainie |

### `test_module` (współdzielony silnik w badaniach)

Backtest i decyzja: `simulation_service.py`, `core_logic/decision.py`, `wallet.py`, `mp_signal.py`, `windows.py`, `structures.py`, wybrane ścieżki w `version_registry` (v6 / mp / rl / rlul). Rating: `services/test/rating.py`.

### `tests/thesis/`

Pytest kontraktu: knobs, reward_mode, regime, obs_rms, leakage, costs, wallet clamp, pretrain kwargs.

---

## CLI — skrypty badawcze (`z_maina_funkcje/`)

| Skrypt | Zastosowanie |
|--------|----------------|
| `plgrid_train_agent_ppo.py` | Pretrain PPO na `THESIS_SEEDS` |
| `plgrid_train_eval_watcher.py` | Watcher ewaluacji równolegle z treningiem PPO |
| `plgrid_simulation_tuner.py` | R1 LOCK (Optuna 6F); `--lock-tag final` = osobny plik |
| `plgrid_regime_lock.py` | Fit K + zapis `regime_lock.json` |
| `plgrid_test_sequence.py` | Pełna / częściowa sekwencja walk-forward |
| `plgrid_thesis_compare.py` | Tabela porównawcza ramion |
| `plgrid_seed_mean_sims.py` | Uśrednienie wyników po seedach |
| `plgrid_train_ppo_simulation.py` | Symulacje na wytrenowanym PPO (okna tezy) |

---

## Artefakty w `shared/` (badania)

| Ścieżka | Artefakty wymagane / istotne dla reprodukcji |
|---------|-------------------------------|
| `agents/ppo/` | `PPO_best_{S}`, `PPO_best_UL_{S}`, `agent_ppo_*_final_*` (continue-learn; nie w `thesis_results`) |
| `agents/optuna_studies/` | `r1_lock_final_seed{S}_sl003.json`, study `*_final_sl003*`, `wf_M1_*` |
| `files/windows.json`, `thesis_sequence_config.json` | Okna i config WF |
| `files/regime_lock.json`, `regime_model_train*` | UL (R3/M3) |
| `simulation/thesis_results/*_thesis_*` | Runy kampanii A1–B10 |
| `simulation/thesis_results/_compare\|_report/*_20261028_022007` | Najnowszy agregat |
| `simulation/outputs/*thesis*` | Eksporty pod viewer |

Pełny skład runu i historia plików: [wyniki-kampanii-tezy.md](wyniki-kampanii-tezy.md).

---

## Uruchomienie wszystkich ramion tezy (R0–M3)

Wszystkie komendy z **roota** repozytorium. Config sekwencji: `shared/files/thesis_sequence_config.json`  
(ramiona `R0 R1 R2 R3 M1 M2 M3`, okna `A1`–`B10`, seedy `396379 951830 246215`).

### 0. Wymagania na dysku

- CSV: `THESIS_DATA_FILES` → `shared/files/processed_simulation_data/`
- Modele MP: `THESIS_MODEL_PATHS` → `shared/models/lstm_lr_models/` + pakiet `lstm_model_package_latest`
- `shared/files/windows.json`

### 1. Lock reżimu UL (jednorazowo, przed R3/M3)

```bash
python z_maina_funkcje/plgrid_regime_lock.py
```

→ `shared/files/regime_lock.json` + `regime_model_train.joblib`

### 2. Pretrain PPO (start R2/M2 oraz R3/M3)

Sekwencyjnie (jeden `train.lock` — nie uruchamiać dwóch treningów równolegle):

```bash
python z_maina_funkcje/plgrid_train_agent_ppo.py --all
python z_maina_funkcje/plgrid_train_agent_ppo.py --all --sim-mode rlul
```

→ `PPO_best_{seed}.zip` oraz `PPO_best_UL_{seed}.zip` (+ json / obs_rms)

Opcjonalnie (równolegle z treningiem): `python z_maina_funkcje/plgrid_train_eval_watcher.py`

### 3. Lock R1 (baza R1 i M1) — Optuna flat 6F

```bash
python z_maina_funkcje/plgrid_simulation_tuner.py --all-seeds --stop-loss 0.03 --lock-tag final
```

→ `shared/agents/optuna_studies/r1_lock_final_seed{S}_sl003.json`  
(ścieżki muszą zgadzać się z `by_seed.*.best_params_path` w configu)

### 4. Preflight (bez simów)

```bash
python z_maina_funkcje/plgrid_test_sequence.py --preflight
```

### 5. Pełna sekwencja — wszystkie ramiona × seedy

```bash
python z_maina_funkcje/plgrid_test_sequence.py
```

Config ma `resume: true` — ponowne uruchomienie uzupełni brakujące foldy.  
Podzbiór (np. tylko M2/M3):

```bash
python z_maina_funkcje/plgrid_test_sequence.py --arms M2 M3 --seeds 396379
```

→ `shared/simulation/thesis_results/{ARM}_thesis_{seed}_{date}_A1_B10/`  
Szczegóły składu: [wyniki-kampanii-tezy.md](wyniki-kampanii-tezy.md).

| Arm | Co robi sekwencja |
|-----|-------------------|
| R0 | Buy & Hold |
| R1 | MP ze stałym lockiem R1 |
| R2 | RL frozen (`PPO_best_*`) |
| R3 | RL+UL frozen (`PPO_best_UL_*`) |
| M1 | Optuna rolling per fold (`wf_M1_*`) |
| M2 | Continue-learn PPO (`agent_ppo_seed*_final_*`) |
| M3 | Continue-learn + UL (`agent_ppo_rlul_seed*_final_*`) |

### 6. Porównanie i uśrednienie

```bash
python z_maina_funkcje/plgrid_thesis_compare.py --run-label thesis --portfolio-mode reset --report
python z_maina_funkcje/plgrid_seed_mean_sims.py --label thesis
```

### 7. Testy kontraktu (po zmianach knobs / regime)

```bash
python -m pytest tests/thesis -q
```
