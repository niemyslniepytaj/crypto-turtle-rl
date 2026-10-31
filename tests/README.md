# tests — kontrakt protokołu tezy (pytest)

Testy jednostkowe w `tests/thesis/`: knobs, reżimy, reward, portfel, obs_rms, koszty i brak wycieku informacji. Poza zakresem: pełne FastAPI oraz treningi SB3 end-to-end.

## Struktura

```
tests/
└── thesis/
    ├── test_analysis.py
    ├── test_costs.py
    ├── test_hodl_fee.py
    ├── test_knobs_gates.py
    ├── test_leakage.py
    ├── test_obs_extra.py
    ├── test_obs_rms.py
    ├── test_ppo_pretrain.py
    ├── test_regime.py
    ├── test_regime_k_infer.py
    ├── test_regime_wrapper.py
    ├── test_reward_mode.py
    └── test_wallet_clamp.py
```

## Mapowanie test → kod

| Test | Obszar kodu |
|------|-------------|
| `test_knobs_gates` | `rl_module.services.thesis.knobs` — free vs const, bramki Optuna |
| `test_reward_mode` | `reward_mode` legacy vs excess w treningu / strukturach |
| `test_obs_rms` / `test_obs_extra` | Normalizacja obs, extra features |
| `test_regime*` | `regime_service`, wrapper rlul, inferencja K z locka |
| `test_wallet_clamp` | Limity portfela w `test_module` wallet |
| `test_leakage` | Brak lookahead (fit tylko na przeszłości względem okna) |
| `test_costs` / `test_hodl_fee` / `test_analysis` | Koszty, HODL, metryki porównawcze |
| `test_ppo_pretrain` | Kontrakt kwargs / hashy pretrainu PPO |

## Uruchomienie

Z roota repozytorium (aktywny venv):

```bash
python -m pytest tests/thesis -q
```
