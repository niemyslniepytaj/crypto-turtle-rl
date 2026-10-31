# test_module — silnik decyzji, portfela i symulacji

Backtest oraz ścieżka realtime/testnet. Silnik handlowy (`core_logic`) jest współdzielony przez trening RL i symulacje MP / RL / RL+UL.

## Struktura

```
test_module/
├── main.py                          # FastAPI: testnet buy/sell/portfolio
├── configuration/
│   ├── configuration.py
│   └── realtime_simulation_config.json
└── services/
    ├── simulation_service.py        # główny backtester (zapis sim + outputs)
    ├── realtime_generate_decision_service.py
    ├── visualization_service.py
    ├── data_quality_service.py
    ├── core_logic/                  # silnik decyzji i portfela
    └── test/                        # rating, dashboardy, statystyki
```

## `services/core_logic/`

| Plik | Rola |
|------|------|
| `decision.py` | `DecisionEngine` — obs, akcje, powiązanie z agentem SB3 / knoby MP |
| `wallet.py` | Portfel, SL, limity ekspozycji, clamp |
| `structures.py` | `CoreConfig`, `sim_mode`, `experiment_version`, normalizacja kwargs MP |
| `version_registry.py` | Warianty eksperymentów (v1…v6 / v5rl itd.) — metody decision/obs/reward |
| `mp_signal.py` | Sygnał z pakietów modeli (LSTM / XGB) → prawdopodobieństwa |
| `model_package.py`, `standalone_model.py` | Ładowanie / predykcja modeli |
| `windows.py` | Okna czasowe (A1/A2, train/eval) z `shared/files/windows.json` |
| `binance_integration.py`, `binance_testnet_service.py` | Testnet Binance |

**Tryby (`sim_mode` / `core_mode`):** m.in. `mp` (knoby MP), `rl` (agent PPO/SAC), `rlul` (RL + one-hot reżimu), oraz ścieżki treningowe `train_rl`.

## `services/test/`

| Plik | Rola |
|------|------|
| `rating.py` | Metryki symulacji (PnL, excess, drawdown, tx/d, …) |
| `statistics_service.py` | Agregacje statystyczne |
| `charts.py` | Wykresy |
| `simulation_viewer_dashboard.py` | Dashboard wyników sim |
| `realtime_viewer_dashboard.py` | Dashboard realtime |

## Wejścia / wyjścia

- Dane OHLCV: `shared/files/processed_simulation_data/`
- Agenci: `shared/agents/`
- Modele MP: `shared/models/`
- Wyniki: `shared/simulation/simulations/`, `shared/simulation/outputs/`
