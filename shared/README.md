# shared — dane, modele, agenci i wyniki

Artefakty runtime protokołu tezy i ścieżek pomocniczych. Jedyny kod w tym drzewie to pakiety modeli w `models/`.

Wyniki tezy i ich skład: [`docs/wyniki-kampanii-tezy.md`](../docs/wyniki-kampanii-tezy.md).

## Katalogi

### `files/` — dane i konfiguracja

| Ścieżka / plik | Zawartość | W gicie (teza) |
|----------------|-----------|----------------|
| `processed_simulation_data/ETH\|BTC…_20250511_20260831.csv` | CSV 1m do backtestu | tak |
| `windows.json` | Okna train / A* / B* | tak |
| `thesis_sequence_config.json` (+ `.example.json`) | Config sekwencji WF | tak |
| `regime_lock.json` + `regime_model_train.joblib` (+ profile) | Lock UL | tak |
| `raw_data/`, `research/`, `regime_*_TEST_*` | Surowce / backupy | nie |

### `agents/` — RL + Optuna

| Podkatalog | W gicie (teza) |
|------------|----------------|
| `ppo/PPO_best_{S}*`, `PPO_best_UL_{S}*` | Start R2/R3/M2/M3 |
| `ppo/agent_ppo_seed{S}_final_*`, `agent_ppo_rlul_seed{S}_final_*` | Continue-learn M2/M3 |
| `optuna_studies/r1_lock_final_seed{S}_sl003.json` + `*_final_sl003*` | Baza R1/M1 |
| `optuna_studies/wf_M1_*` | Study walk-forward M1 |
| `ppo/*workdir*`, `*KEEP*`, `sac/`, `actual_training_logs/` | nie |

### `models/` — modele sygnału MP

Trackowane pakiety tezy: `lstm_lr_models/LR1449_*`, `LR1414_*`, `lstm_model_package_latest` (używany przez `mp_signal`; `lstm_model_package` / `package2` poza gitem), `xgb_model_package`.

### `simulation/` — wyniki

| Podkatalog | W gicie |
|------------|---------|
| `thesis_results/*_thesis_*/` | Runy kampanii |
| `thesis_results/_compare\|_report/*_20261028_022007/` | Najnowszy agregat |
| `outputs/*thesis*` | Eksporty tezy |
| `simulations/`, `obs_importance/`, … | nie |

### `realtime_simulation/`, `plgrid/`, `E104_*`

Poza protokołem tezy — ignorowane.

## Konwencje nazw

- `r1_lock_final_seed{S}_sl003.json` — lock R1 używany w `thesis_sequence_config.json`
- `PPO_best_{seed}.zip` / `PPO_best_UL_{seed}.zip` — start sekwencji
- `agent_ppo_seed{S}_final_*.zip` — checkpoint po continue-learn (M2)
- `{ARM}_thesis_{seed}_{date}_A1_B10/` — katalog wyników runu
