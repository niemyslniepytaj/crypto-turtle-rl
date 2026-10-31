# Walk-forward — tabela zbiorcza

Wygenerowano: 2026-10-27T03:39:18.849876

Run: `M3_thesis_396379_20261027_A1_B10`

Folds: A1, A2, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10

| Ramię | plan | n | total_return | excess_return | sharpe_ratio | max_drawdown | calmar_ratio | avg_transactions_per_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL + UL (v6+rlul) | M3 | 12 | -2.41±10.06 | -0.59±8.02 | -0.88±5.08 | -9.79±6.79 | 15.50±41.98 | 6.28±4.13 |

## Uwagi

- Wartości: średnia ± odchylenie std po foldach OOS (`status=ok`).
- `excess_return` = total_return strategii − HODL w tym samym foldzie.
- Surowy dump: `folds.csv`, `run_results.json`.
