# Walk-forward — tabela zbiorcza

Wygenerowano: 2026-10-26T16:46:15.292212

Run: `M2_thesis_246215_20261026_A1_B10`

Folds: A1, A2, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10

| Ramię | plan | n | total_return | excess_return | sharpe_ratio | max_drawdown | calmar_ratio | avg_transactions_per_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL continue-learn (v6+rl) | M2 | 12 | -6.21±13.76 | -4.39±5.05 | -2.06±5.91 | -13.34±10.95 | 8.34±26.11 | 23.95±11.90 |

## Uwagi

- Wartości: średnia ± odchylenie std po foldach OOS (`status=ok`).
- `excess_return` = total_return strategii − HODL w tym samym foldzie.
- Surowy dump: `folds.csv`, `run_results.json`.
