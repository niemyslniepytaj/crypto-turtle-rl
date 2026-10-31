# Walk-forward — tabela zbiorcza

Wygenerowano: 2026-10-26T11:08:51.274102

Run: `R3_thesis_246215_20261026_A1_B10`

Folds: A1, A2, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10

| Ramię | plan | n | total_return | excess_return | sharpe_ratio | max_drawdown | calmar_ratio | avg_transactions_per_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL+UL frozen (v6+rlul) | R3 | 12 | -4.83±10.81 | -3.02±7.73 | -2.05±5.55 | -11.39±8.14 | 8.24±24.95 | 23.32±6.30 |

## Uwagi

- Wartości: średnia ± odchylenie std po foldach OOS (`status=ok`).
- `excess_return` = total_return strategii − HODL w tym samym foldzie.
- Surowy dump: `folds.csv`, `run_results.json`.
