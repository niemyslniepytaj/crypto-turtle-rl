# crypto-turtle-rl

**Automatyczny trading kryptowalut** (Binance spot, portfel ETH+BTC, świece 1m) z warstwą sygnału ML i warstwą samodoskonalenia (Optuna + RL) — oraz protokołem badawczym do uczciwego porównania strategii.

<p align="center">
  <a href="docs/index.md"><strong>📚 Dokumentacja projektu →</strong></a>
  &nbsp;·&nbsp;
  <a href="docs/opis-projektu.md">Architektura</a>
  &nbsp;·&nbsp;
  <a href="docs/protokol-badawczy.md">Protokół R0–M3</a>
  &nbsp;·&nbsp;
  <a href="docs/wyniki-kampanii-tezy.md">Wyniki kampanii</a>
  &nbsp;·&nbsp;
  <a href="docs/api/index.md">API</a>
</p>

```text
  sygnał MP (LSTM / XGB)          Optuna (knoby)  +  RL (PPO / SAC / DQN)
           \                              /
            \____ DecisionEngine + Wallet ____/
                         │
              backtest · trening · realtime · testnet
```

---

## Praca magisterska

Repozytorium powstało jako **implementacja i infrastruktura badawcza** pod pracę magisterską — nie jako produkt handlowy „z pudełka”. Kod, protokół eksperymentalny (R0–M3), walk-forward, locki konfiguracji i warstwa statystyczna służą do **reprodukowalnej oceny** metod samodoskonalenia bota.

| | |
|---|---|
| **Temat** | *Analiza oraz ocena skuteczności i efektywności metod samodoskonalenia w przypadku autonomicznego bota tradingowego* |
| **Pytanie badawcze** | Czy i które metody samodoskonalenia (strojenie parametrów, RL, RL + uczenie nienadzorowane / reżimy) poprawiają wyniki względem wariantów zamrożonych i względem Buy & Hold — na tych samych danych i oknach OOS? |
| **Obiekt badań** | Autonomiczny bot spot (ETH+BTC, Binance, świece 1m) ze wspólnym silnikiem decyzji i backtestu |
| **Metody w matrycy** | MP + Optuna · RL (PPO) · RL + UL (etykieta reżimu) — warianty **frozen** (R1–R3) vs **adaptive** (M1–M3), baseline **R0** |
| **Protokół** | Walk-forward, stałe seedy, zamrożone locki, porównania między-ramienne i testy hipotez (H1–H5) |
| **Artefakty pod pracę** | Runy `shared/simulation/thesis_results/`, raporty compare/report, eksport `.tex` / wykresy z pakietu `thesis/` |

Szczegóły procedury i wyników: [protokół badawczy](docs/protokol-badawczy.md) · [wyniki kampanii](docs/wyniki-kampanii-tezy.md).

Hipotezy robocze (skrót): **H1** — cykliczne strojenie MP (M1) vs lock (R1); **H2** — continue-learn PPO (M2) vs agent zamrożony (R2); **H3** — RL+UL adaptive (M3) vs RL+UL frozen (R3); dalej porównania między metodami oraz względem R0 — pełna definicja w `rl_module/services/thesis/hypotheses.py` i generatorze raportu.

---

## Dokumentacja

> **Pełny opis systemu, protokół eksperymentalny, wyniki i reference API**  
> Start: **[docs/index.md](docs/index.md)**  
> Podgląd HTML: `pip install -r requirements-docs.txt` → `mkdocs serve`

| | |
|---|---|
| [Opis projektu](docs/opis-projektu.md) | Mapa repozytorium, architektura runtime |
| [Protokół badawczy](docs/protokol-badawczy.md) | Ramiona R0–M3, seedy, locki, CLI, reprodukcja |
| [Wyniki kampanii](docs/wyniki-kampanii-tezy.md) | Lokalizacja i skład runów tezy |
| [Poza protokołem](docs/poza-protokolem.md) | Realtime, testnet, eksploracja innych algorytmów |
| [API (mkdocstrings)](docs/api/index.md) | Reference z docstringów w kodzie |

README w katalogach (`rl_module/`, `test_module/`, `z_maina_funkcje/`, `shared/`, `tests/`) opisują strukturę lokalną.

---

## Co robi system

Dwie warstwy decyzji na wspólnym silniku handlowym:

1. **Sygnał MP** — modele ML podają prognozę (µ/σ lub prawdopodobieństwa); knoby (`mp_alloc_*`, confidence, urgency, …) mapują ją na cele alokacji.
2. **Samodoskonalenie** — Optuna dobiera knoby MP; agent RL (Stable-Baselines3) steruje decyzjami w trybie zamrożonym albo z dalszym uczeniem na foldach.

`DecisionEngine`, `Wallet` i sygnał MP są wspólne dla backtestu, treningu RL oraz ścieżek poza protokołem.

### Cel badawczy

Porównać strategie na **tym samym portfelu, danych i oknach walk-forward**, ze stałymi seedami i zamrożonymi lockami — tak, by różnice wynikały z metody decyzji, nie z doraźnych parametrów.

| Arm | Sens |
|-----|------|
| **R0** | Buy & Hold — baseline |
| **R1** | MP static — knoby 6F z Optuny (LOCK), zamrożone na eval |
| **R2** | RL frozen — pretrain PPO, bez uczenia na foldach |
| **R3** | RL + UL frozen — jak R2 + one-hot reżimu rynkowego |
| **M1** | MP + Optuna rolling (retune per fold) |
| **M2** | RL continue-learn na foldach |
| **M3** | RL + UL z adaptacją |

Kontrakt: `rl_module/services/thesis/` · joby: `z_maina_funkcje/plgrid_*.py`

---

## Architektura (skrót)

```text
CSV 1m (ETH/BTC)  +  modele MP
         \              /
          ▼            ▼
       sygnał MP (ModelSignal)
                │
                ▼
     DecisionEngine  ←  agent SB3 (± reżim UL)
                │
                ▼
             Wallet
        ┌───────┴───────┐
        ▼               ▼
 SimulationService   TradingEnv / SB3
   (backtest)      (→ shared/agents)
```

Tryby: `mp` · `rl` · `rlul`

Szczegóły: [docs/opis-projektu.md](docs/opis-projektu.md)

---

## Repozytorium

| Katalog | Rola |
|---------|------|
| `docs/` | Dokumentacja MkDocs |
| `z_maina_funkcje/` | CLI `plgrid_*.py` — punkt startu jobów |
| `rl_module/` | Trening RL, Optuna, regime UL, `thesis/` |
| `test_module/` | Silnik decyzji, symulacja, realtime, testnet |
| `shared/` | Dane, modele, agenci, locki, `thesis_results/` |
| `tests/thesis/` | Pytest kontraktu badawczego |

---

## Szybki start

```bash
python -m venv .venv
pip install -r requirements.txt

pip install -r requirements-docs.txt
mkdocs serve                              # http://127.0.0.1:8000

python -m pytest tests/thesis -q
```

Joby badawcze: skrypty `plgrid_*` z roota (nie HTTP).  
Komendy, seedy i artefakty reprodukcji → [protokół badawczy](docs/protokol-badawczy.md).  
Realtime / testnet / SAC·DQN → [poza protokołem](docs/poza-protokolem.md).

---

## Stack

`Python` · `PyTorch` · `Stable-Baselines3` · `Gymnasium` · `Optuna` · `FastAPI` · `pandas` / `scikit-learn` · `Dash` / `Plotly` · `python-binance`

Nie commitować sekretów (`.env`, klucze API). Ciężkie artefakty i whitelist wyników: `.gitignore`, [shared/README.md](shared/README.md).
