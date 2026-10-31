# rl_module — trening RL, Optuna i protokół tezy

Serwisy treningu SB3, tuningu Optuna, reżimów UL oraz pakiet `services/thesis/`. FastAPI (`main.py`) jest stubem; uruchomienia badawcze i trenujące idą przez `z_maina_funkcje/plgrid_*.py`.

## Struktura

```
rl_module/
├── main.py                 # FastAPI stub
├── services/
│   ├── train_agent_{ppo,sac,dqn}_service.py
│   ├── train_agent_{ppo,sac,dqn}_tuner_service.py
│   ├── simulation_tuner_service.py   # Optuna MP (R1 LOCK / M1), bez agenta RL
│   ├── regime_service.py             # UL reżimów (k-means) — M3/R3
│   ├── walkforward_service.py        # walk-forward tezy
│   ├── obs_importance_service.py
│   ├── sequence_export.py
│   ├── thesis_artifacts.py           # zapisy Optuna/lock JSON
│   ├── rl/                           # wspólna infrastruktura SB3
│   └── thesis/                       # kontrakt eksperymentu
```

## `services/rl/` — infrastruktura treningu

| Plik | Rola |
|------|------|
| `base_train_service.py` | Wspólny pipeline treningu (env, callbacki, zapis agentów, `obs_rms`) |
| `base_tuner_service.py` | Wspólny Optuna (staged E1→E2→E3 itd.) |
| `agent.py` | Env / agent helpers |
| `training_callback.py` | Eval podczas treningu, `sim_best` |
| `obs_rms_utils.py` | Normalizacja obserwacji (zapis/odczyt `*_obs_rms.npz`) |
| `regime_obs_wrapper.py` | Doklejanie one-hot reżimu do obs (rlul) |
| `rl_training_live_dashboard.py` | Live podgląd treningu |

Algorytmy: **PPO** (główny tor tezy), SAC, DQN — serwisy oparte na `BaseTrainService` / `BaseTunerService`.

## `services/thesis/` — kontrakt tezy

| Plik | Rola |
|------|------|
| `assets.py` | `THESIS_SEEDS`, ścieżki danych/modeli MP, tagi `mwpc`/`sl` |
| `knobs.py` | `MP_FREE` (6F), `MP_CONST`, `PPO_HPARAMS`, `reward_mode`, hashe konfiguracji |
| `arms.py` | Ramiona R0–R3, M1–M3 (Buy&Hold / MP / RL / RL+UL ± samodoskonalenie) |
| `walkforward.py` | Okna foldów / helpery WF |
| `costs.py`, `analysis.py`, `compare.py` | Koszty, metryki, porównania |
| `repro.py`, `preflight.py`, `completeness.py` | Reprodukcja, preflight, kompletność runów |
| `export.py`, `console.py`, `hypotheses.py`, `regime_labels.py` | Eksport artefaktów, logi, hipotezy, etykiety reżimów |
| `seed_mean_sims.py` | Agregacja symulacji po seedach |

## Typowy przepływ

1. Pretrain PPO → `TrainAgentPPOService` (entry: `plgrid_train_agent_ppo.py`)
2. R1 LOCK Optuna 6F → `SimulationTunerService` (entry: `plgrid_simulation_tuner.py`)
3. Walk-forward / sekwencja → `walkforward_service` + `thesis/*` (entry: `plgrid_test_sequence.py`)
4. Artefakty → `shared/agents/`, `shared/files/`, `shared/simulation/`
