# Wyniki kampanii tezy

Lokalizacja, skład katalogów i historia plików kampanii R0–M3 (`run_label=thesis`, okna **A1–B10**, seedy `THESIS_SEEDS`).

Źródło prawdy na dysku: `shared/simulation/thesis_results/`.  
W repozytorium trackowane są wyłącznie runy `*_thesis_*` oraz najnowszy agregat compare/report (znacznik `20261028_022007`). Eksporty viewer: `shared/simulation/outputs/*thesis*`.

---

## Gdzie leżą wyniki


| Ścieżka                                                                          | Rola                                                    |
| -------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `shared/simulation/thesis_results/{ARM}_thesis_{seed}_{YYYYMMDD}_{start}_{end}/` | Pełny run jednego ramienia × seed                       |
| `shared/simulation/thesis_results/_compare/thesis_mixed_20261028_022007/`        | Najnowsze porównanie między-ramienne (mixed)            |
| `shared/simulation/thesis_results/_report/thesis_reset_20261028_022007/`         | Najnowszy raport (portfolio_mode=reset)                 |
| `shared/simulation/outputs/*thesis*`                                             | Kopie sekwencji / foldów pod dashboard (gzip JSON, PNG) |


Starsze foldery `_compare/*` / `_report/*` (inne znaczniki czasu) oraz smoke/test (`*seed265*`, `*smoke*`) **nie** wchodzą do gita (`.gitignore`).

---



## Kompletność kampanii (stan dysku)


| Arm | 396379                             | 951830 | 246215 | Uwagi                                               |
| --- | ---------------------------------- | ------ | ------ | --------------------------------------------------- |
| R0  | `R0_thesis_396379_20261027_A1_B10` | —      | —      | B&H bez zależności od seeda RL; jeden run wystarcza |
| R1  | `…_20261026_…`                     | tak    | tak    | Lock: `r1_lock_final_seed{S}_sl003.json`            |
| R2  | `…_20261026_…`                     | tak    | tak    | Start: `PPO_best_{S}.zip`                           |
| R3  | `…_20261026_…`                     | tak    | tak    | Start: `PPO_best_UL_{S}.zip` + regime lock          |
| M1  | `…_20261026_…`                     | tak    | tak    | Optuna rolling; study `wf_M1_*` w `optuna_studies/` |
| M2  | `…_20261026_…`                     | tak    | tak    | Continue-learn → `agent_ppo_seed{S}_final_*`        |
| M3  | `…_20261027_…`                     | tak    | tak    | Continue + UL → `agent_ppo_rlul_seed{S}_final_*`    |


Każdy run `*_thesis_*` ma `manifest.json`, `run_results.json`, `folds.csv` oraz foldy **A1, A2, B1–B10**.  
Dla M2/M3: w `arms/{ARM}/` pliki `fold_*_continue.json` (oraz UL `fold_*_ul.joblib` / `_profile` dla M3) — ostatni fold (B10) bez continue (brak kolejnego okna).

---



## Skład pojedynczego runu `*_thesis_*`

Typowa struktura (nazwy mogą się nieznacznie różnić między armami):

```
{ARM}_thesis_{seed}_{date}_A1_B10/
├── manifest.json          # config runu, ścieżki CSV/modeli/agentów, eval_folds
├── run_results.json       # metryki zagregowane
├── folds.csv              # tabela per fold
├── summary_by_arm.csv|.md # (gdy wygenerowane)
├── sequence_B.json        # meta sekwencji B1–B10
├── sequence_B/            # seq_*_sequence_B1_B10.json.gz (+ PNG)
├── windows/               # per fold: sim_results, fold_*.json, seq_*.json.gz, PNG
│   ├── A1/ … B10/
│   └── fold_A1.json …
└── arms/{ARM}/            # artefakty adaptacji (M1 best_params, M2/M3 continue, M3 UL)
```

**Historia plików w runie:** w trakcie `plgrid_test_sequence` / `WalkForwardService` powstają kolejno `manifest.json`, foldy w `windows/` i `arms/`, następnie agregaty (`run_results`, `folds.csv`, `sequence_B`).  
Compare/report (`plgrid_thesis_compare`, seed-mean) nie nadpisują runów; zapisują osobne katalogi pod `_compare/` / `_report/` ze znacznikiem czasu.

---



## Wejścia protokołu


| Artefakt             | Ścieżka                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------ |
| Okna                 | `shared/files/windows.json`                                                                |
| Config sekwencji     | `shared/files/thesis_sequence_config.json`                                                 |
| CSV ETH/BTC          | `shared/files/processed_simulation_data/*_20250511_20260831.csv`                           |
| Modele MP            | `shared/models/lstm_lr_models/LR1449_*`, `LR1414_*` + `lstm_model_package_latest`          |
| Lock R1 (baza R1/M1) | `shared/agents/optuna_studies/r1_lock_final_seed{S}_sl003.json` (+ study `*_final_sl003*`) |
| Study M1 WF          | `shared/agents/optuna_studies/wf_M1_*`                                                     |
| Agenci start         | `shared/agents/ppo/PPO_best_{S}*`, `PPO_best_UL_{S}*`                                      |
| Continue-learn       | `shared/agents/ppo/agent_ppo_seed{S}_final_*`, `agent_ppo_rlul_seed{S}_final_*`            |
| Reżim UL             | `shared/files/regime_lock.json`, `regime_model_train.joblib` (+ profile)                   |


Procedura uruchomienia pełnej kampanii: [protokol-badawczy.md](protokol-badawczy.md) → *Uruchomienie wszystkich ramion tezy*; skrót CLI: `z_maina_funkcje/README.md`.