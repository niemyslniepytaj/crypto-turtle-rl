# Walk-forward — tabela zbiorcza

Wygenerowano: 2026-10-27T05:39:34.927292

Run: `M3_thesis_951830_20261027_A1_B10`

Folds: A1, A2, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10

| Ramię | plan | n | total_return | excess_return | sharpe_ratio | max_drawdown | calmar_ratio | avg_transactions_per_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL + UL (v6+rlul) | M3 | 12 | -2.81±10.16 | -1.00±8.19 | -1.30±5.57 | -10.13±6.88 | 15.44±45.11 | 4.98±2.17 |

## Uwagi

- Wartości: średnia ± odchylenie std po foldach OOS (`status=ok`).
- `excess_return` = total_return strategii − HODL w tym samym foldzie.
- Surowy dump: `folds.csv`, `run_results.json`.
