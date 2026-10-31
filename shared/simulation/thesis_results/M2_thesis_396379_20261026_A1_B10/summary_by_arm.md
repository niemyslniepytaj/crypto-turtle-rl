# Walk-forward — tabela zbiorcza

Wygenerowano: 2026-10-26T13:17:48.566055

Run: `M2_thesis_396379_20261026_A1_B10`

Folds: A1, A2, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10

| Ramię | plan | n | total_return | excess_return | sharpe_ratio | max_drawdown | calmar_ratio | avg_transactions_per_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL continue-learn (v6+rl) | M2 | 12 | -3.09±10.72 | -1.28±7.89 | -1.13±5.38 | -10.56±7.74 | 14.63±41.83 | 8.61±7.73 |

## Uwagi

- Wartości: średnia ± odchylenie std po foldach OOS (`status=ok`).
- `excess_return` = total_return strategii − HODL w tym samym foldzie.
- Surowy dump: `folds.csv`, `run_results.json`.
