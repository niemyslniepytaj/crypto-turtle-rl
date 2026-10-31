# Walk-forward — tabela zbiorcza

Wygenerowano: 2026-10-26T10:25:50.551047

Run: `R2_thesis_246215_20261026_A1_B10`

Folds: A1, A2, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10

| Ramię | plan | n | total_return | excess_return | sharpe_ratio | max_drawdown | calmar_ratio | avg_transactions_per_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL frozen (v6+rl) | R2 | 12 | -5.27±13.59 | -3.45±5.16 | -1.46±5.99 | -12.66±11.40 | 13.84±34.89 | 12.76±11.30 |

## Uwagi

- Wartości: średnia ± odchylenie std po foldach OOS (`status=ok`).
- `excess_return` = total_return strategii − HODL w tym samym foldzie.
- Surowy dump: `folds.csv`, `run_results.json`.
