# Walk-forward — tabela zbiorcza

Wygenerowano: 2026-10-26T10:24:39.430397

Run: `R2_thesis_951830_20261026_A1_B10`

Folds: A1, A2, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10

| Ramię | plan | n | total_return | excess_return | sharpe_ratio | max_drawdown | calmar_ratio | avg_transactions_per_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL frozen (v6+rl) | R2 | 12 | -3.16±10.57 | -1.34±7.66 | -1.20±5.42 | -10.73±7.51 | 14.55±39.55 | 7.44±4.80 |

## Uwagi

- Wartości: średnia ± odchylenie std po foldach OOS (`status=ok`).
- `excess_return` = total_return strategii − HODL w tym samym foldzie.
- Surowy dump: `folds.csv`, `run_results.json`.
