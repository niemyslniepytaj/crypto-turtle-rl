# Walk-forward — tabela zbiorcza

Wygenerowano: 2026-10-27T01:06:22.131048

Run: `M1_thesis_246215_20261026_A1_B10`

Folds: A1, A2, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10

| Ramię | plan | n | total_return | excess_return | sharpe_ratio | max_drawdown | calmar_ratio | avg_transactions_per_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MP + Optuna rolling-3 (v6+mp) | M1 | 12 | -2.55±8.65 | -0.73±10.85 | -0.85±4.96 | -8.20±7.92 | 9.13±23.10 | 2.71±2.89 |

## Uwagi

- Wartości: średnia ± odchylenie std po foldach OOS (`status=ok`).
- `excess_return` = total_return strategii − HODL w tym samym foldzie.
- Surowy dump: `folds.csv`, `run_results.json`.
