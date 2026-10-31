# Walk-forward — tabela zbiorcza

Wygenerowano: 2026-10-26T23:14:06.323561

Run: `M1_thesis_951830_20261026_A1_B10`

Folds: A1, A2, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10

| Ramię | plan | n | total_return | excess_return | sharpe_ratio | max_drawdown | calmar_ratio | avg_transactions_per_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MP + Optuna rolling-3 (v6+mp) | M1 | 12 | -3.07±9.87 | -1.25±10.82 | -0.84±5.14 | -8.21±9.25 | 9.52±23.05 | 2.70±3.99 |

## Uwagi

- Wartości: średnia ± odchylenie std po foldach OOS (`status=ok`).
- `excess_return` = total_return strategii − HODL w tym samym foldzie.
- Surowy dump: `folds.csv`, `run_results.json`.
