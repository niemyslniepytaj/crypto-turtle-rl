# crypto-turtle-rl — opis projektu i architektury

System do **automatycznego tradingu kryptowalut** (portfel ETH+BTC, Binance spot, świece 1m) z dwoma warstwami decyzji:

1. **Sygnał MP** — modele ML (LSTM / XGB) dostarczają µ/σ lub prawdopodobieństwa; knoby (`mp_alloc_*`, confidence, urgency, …) mapują sygnał na cele alokacji.
2. **Samodoskonalenie** — Optuna (dobór knobów MP) oraz RL (PPO / SAC / DQN, Stable-Baselines3) jako zamrożony lub adaptacyjny kontroler.

Repozytorium jest podzielone na **protokół badawczy** (R0–M3, walk-forward, seedy) i **ścieżki poza protokołem** (realtime, testnet, eksploracja innych algorytmów):

- [protokol-badawczy.md](protokol-badawczy.md) — protokół eksperymentalny i pipeline uruchomienia
- [wyniki-kampanii-tezy.md](wyniki-kampanii-tezy.md) — wyniki kampanii, skład katalogów
- [poza-protokolem.md](poza-protokolem.md) — realtime, testnet, SAC/DQN, LSTM multi

README w katalogach kodu opisują strukturę lokalną; **ten dokument** mapuje całe repozytorium.

---

## Mapa repozytorium

```
crypto-turtle-rl/
├── docs/                      # dokumentacja (protokół, wyniki, API)
├── z_maina_funkcje/           # CLI plgrid_*.py — zalecany punkt startu jobów
├── rl_module/                 # trening RL, Optuna, regime UL, pakiet thesis/
├── test_module/               # DecisionEngine, wallet, MP signal, sim, realtime, testnet API
├── shared/                    # dane, modele, agenci, locki, wyniki symulacji
├── tests/thesis/              # pytest kontraktu badawczego
├── requirements.txt
└── .gitignore
```

| Katalog | Odpowiedzialność |
|---------|------------------|
| **z_maina_funkcje** | Warstwa CLI; argumenty → serwisy |
| **rl_module** | `TrainAgent*`, tunery, `regime_service`, `walkforward_service`, `services/thesis/*` |
| **test_module** | Silnik handlowy wspólny dla train i sim; FastAPI testnet; dashboardy |
| **shared** | Ciężkie artefakty (sekretów z `.env` nie commitować) |
| **tests** | Testy jednostkowe knobs / regime / reward / leakage / wallet |
| **docs** | Opis projektu i podział torów |

---

## Architektura runtime

```
shared/files/*.csv          shared/models/* (MP)
        \                         /
         \                       /
          ▼                     ▼
           mp_signal (ModelSignal)
                    │
                    ▼
            DecisionEngine  ◄──── agent SB3 (opcjonalnie) + opcjonalnie one-hot reżimu
                    │
                    ▼
                 Wallet (SL, limity ekspozycji, cash)
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   SimulationService    TradingEnv (train_rl)
   (backtest → rating)  (SB3.learn → shared/agents)
```

### Tryby decyzji (`sim_mode` / ścieżki w `structures` + `version_registry`)

| Tryb | Sterowanie | Sygnał MP | Reżim UL |
|------|------------|-----------|----------|
| `mp` | knoby 6F (stałe lub z Optuny) | tak | nie |
| `rl` | akcje agenta PPO/SAC | zwykle w obs / zależnie od wersji | nie |
| `rlul` | jak `rl` | tak (obs) | tak (one-hot K) |

`DecisionEngine` i `Wallet` znajdują się w `test_module`; środowisko treningowe i callbacki — w `rl_module/services/rl/`.

---

## Warstwy kodu

### Sygnał i decyzja — `test_module/services/core_logic/`

- `mp_signal.py` — ładowanie pakietów modeli, bieżące prawdopodobieństwa / µσ
- `decision.py` — obs, akcje, powiązanie z agentem
- `wallet.py` — portfel, stop-loss, capy per coin
- `structures.py` — `CoreConfig`, normalizacja kwargs, `sim_mode`
- `version_registry.py` — warianty eksperymentów (v1…v6, v5rl, …)
- `windows.py` — krojenie szeregów według `windows.json`

### Trening i Optuna — `rl_module/services/`

- `rl/base_train_service.py`, `base_tuner_service.py` — wspólny pipeline
- `train_agent_ppo_service.py` (+ SAC/DQN analogicznie)
- `simulation_tuner_service.py` — Optuna **tylko MP** (bez agenta)
- `regime_service.py`, `rl/regime_obs_wrapper.py` — k-means reżimów
- `thesis/` — kontrakt badań (seedy, knoby, arms, metryki, WF helpers)

### Warstwa CLI — `z_maina_funkcje/plgrid_*.py`

Joby uruchamia się z roota repozytorium przez te skrypty. Podział badawcze vs poza: [protokol-badawczy.md](protokol-badawczy.md) / [poza-protokolem.md](poza-protokolem.md).

---

## Dane i artefakty — `shared/`

| Gałąź | Zawartość |
|-------|-----------|
| `files/processed_simulation_data/` | CSV 1m do backtestu |
| `files/raw_data/` | Surowe klines |
| `files/windows.json` | Definicje okien czasowych |
| `files/thesis_sequence_config.json` | Config walk-forward |
| `files/regime_lock.json` + `regime_model_train*` | Lock UL |
| `models/` | Pakiety LSTM/XGB/ML |
| `agents/ppo|sac/` | Checkpointy SB3 + `obs_rms` |
| `agents/optuna_studies/` | Study Optuna + **locki R1** |
| `agents/actual_training_logs/` | Logi live + `train.lock` |
| `simulation/thesis_results/`, `outputs/*thesis*` | Wyniki tezy — [wyniki-kampanii-tezy.md](wyniki-kampanii-tezy.md) |
| `realtime_simulation/` | Sesje poza protokołem WF |

Szczegóły nazw plików: `../shared/README.md`.

---

## Zależności i uruchomienie

- Python 3 + venv; pakiety w `requirements.txt` (FastAPI, PyTorch CPU, SB3, Gymnasium, Optuna, pandas/sklearn, Dash/Plotly, python-binance, empyrical-reloaded).
- Joby badawcze uruchamia się przez CLI `plgrid_*`, nie przez HTTP.
- Testnet API: `uvicorn` na `test_module.main:app` (poza protokołem badawczym).
- Testy: `python -m pytest tests/thesis -q`.

Przykłady komend badawczych oraz lista artefaktów krytycznych dla reprodukcji — w [protokol-badawczy.md](protokol-badawczy.md) i README katalogów.

---

## Dokumentacja w `docs/`

| Plik | Treść |
|------|--------|
| [opis-projektu.md](opis-projektu.md) / [index.md](index.md) | Opis projektu; start MkDocs |
| [protokol-badawczy.md](protokol-badawczy.md) | Protokół badawczy R0–M3 |
| [wyniki-kampanii-tezy.md](wyniki-kampanii-tezy.md) | Wyniki kampanii tezy |
| [poza-protokolem.md](poza-protokolem.md) | Realtime, testnet, eksploracja |
| [api/](api/index.md) | Reference API z docstringów (`mkdocstrings`) |

Generowanie HTML: `pip install -r requirements-docs.txt`, potem `mkdocs serve` / `mkdocs build` (katalog `site/`).
