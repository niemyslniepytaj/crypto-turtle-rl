# Poza protokołem badawczym

Narzędzia spoza protokołu R0–M3: realtime, testnet, eksploracja innych algorytmów, multi-sim LSTM, jakość danych, tunery ogólne. Wyniki z tego toru nie służą do raportowania ramion tezy — protokół: [protokol-badawczy.md](protokol-badawczy.md).

Przegląd całego repozytorium: [opis-projektu.md](opis-projektu.md).

---

## Cel

- Integracja z **Binance testnet** / decyzje w czasie rzeczywistym
- Eksperymenty z **SAC, DQN**, Optuną hiperparametrów PPO poza zamrożonym pretrainem tezy
- Masowe testy / strojenie **modeli LSTM**
- Pobieranie surowych danych, ocena jakości CSV, dashboardy

Wyniki z tego toru nie zastępują sekwencji `plgrid_test_sequence` ani tabeli `plgrid_thesis_compare`.

---

## API i realtime

| Element | Rola |
|---------|------|
| `test_module/main.py` | FastAPI: `/testnet/buy`, `/testnet/sell`, `/testnet/portfolio` |
| `core_logic/binance_testnet_service.py` + `binance_integration.py` | Wykonanie zleceń testnet |
| `realtime_generate_decision_service.py` | Decyzja w czasie rzeczywistym z configu |
| `configuration/realtime_simulation_config.json` | Parametry sesji realtime |
| `services/test/realtime_viewer_dashboard.py` | Podgląd realtime |
| `shared/realtime_simulation/` | Logi / snapshoty sesji |

`rl_module/main.py` to stub HTTP — poza torem badawczym i poza ścieżką produkcyjną.

---

## CLI poza protokołem tezy

| Skrypt | Zastosowanie |
|--------|--------------|
| `plgrid_train_agent_sac.py` / `plgrid_sac_tuner.py` | Trening i Optuna SAC |
| `plgrid_train_agent_dqn.py` / `plgrid_dqn_tuner.py` | Trening i Optuna DQN |
| `plgrid_ppo_tuner.py` | Optuna **hiperparametrów** PPO (odrębnie od `plgrid_train_agent_ppo` tezy) |
| `plgrid_simulation_basic.py` | Pojedyncza symulacja bez sekwencji WF |
| `plgrid_lstm_multi_simulation.py` | Multi-sim lub Optuna po katalogu modeli LSTM |
| `plgrid_observation_space_importance.py` | Analiza ważności cech obserwacji |
| `plgrid_fetch_raw_klines.py` | Pobranie surowych klines → `shared/files/raw_data/` |

Serwisy: `train_agent_sac_*`, `train_agent_dqn_*`, `train_agent_ppo_tuner_service`, `obs_importance_service`, oraz ogólne `SimulationService` / tunery bez `thesis/knobs` LOCK.

---

## Kod pomocniczy (poza protokołem)

| Moduł | Rola |
|-------|------|
| `data_quality_service.py` | Metryki jakości danych wejściowych |
| `visualization_service.py` | Wykresy symulacji (ogólne; także przy sim) |
| `services/test/charts.py`, `simulation_viewer_dashboard.py`, `statistics_service.py` | UI / statystyki ogólne |
| Starsze warianty w `version_registry.py` (v1–v5, …) | Legacy decision/obs — poza kanonem R0–M3 |
| Pakiety modeli w `shared/models/` poza ścieżkami `THESIS_MODEL_PATHS` | Eksploracja / porównania LSTM multi |

---

## Artefakty `shared/` poza protokołem

| Ścieżka | Zawartość typowa |
|---------|------------------|
| `agents/sac/`, uruchomienia PPO spoza `THESIS_SEEDS` | Agenci eksploracyjni |
| `simulation/lstm_lr_rankings/`, `obs_importance/` | Rankingi i importance |
| `realtime_simulation/` | Sesje live |
| `files/raw_data/`, inne `processed_*` | Dane poza zamrożonym oknem tezy |
| `E104_prep_eth_universal_full_1m/` | Pipeline przygotowania cech E104 |
| `plgrid/` | Cache / dane pod uruchomienia klastrowe ogólne |

---

## Ograniczenia względem toru badawczego

- Nadpisywanie `r1_lock_seed*.json`, `PPO_best_{THESIS_SEED}*`, `regime_lock.json`, `windows.json` poza protokołem jest niedozwolone.
- Wyniki SAC/DQN/LSTM-multi nie stanowią ramion tezy.
- Do protokołu badawczego przeznaczone są skrypty i ścieżki z [protokol-badawczy.md](protokol-badawczy.md).
