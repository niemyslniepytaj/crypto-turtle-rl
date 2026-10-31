# Walk-forward — tabela zbiorcza

Wygenerowano: 2026-10-26T21:26:01.189526

Run: `M1_thesis_396379_20261026_A1_B10`

Folds: A1, A2, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10

| Ramię | plan | n | total_return | excess_return | sharpe_ratio | max_drawdown | calmar_ratio | avg_transactions_per_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MP + Optuna rolling-3 (v6+mp) | M1 | 12 | -3.05±9.08 | -1.23±10.56 | -1.03±5.59 | -7.52±8.49 | 10.27±25.21 | 2.85±4.51 |

## Uwagi

- Wartości: średnia ± odchylenie std po foldach OOS (`status=ok`).
- `excess_return` = total_return strategii − HODL w tym samym foldzie.
- Surowy dump: `folds.csv`, `run_results.json`.
