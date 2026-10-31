# Walk-forward — tabela zbiorcza

Wygenerowano: 2026-10-26T11:08:08.183869

Run: `R3_thesis_951830_20261026_A1_B10`

Folds: A1, A2, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10

| Ramię | plan | n | total_return | excess_return | sharpe_ratio | max_drawdown | calmar_ratio | avg_transactions_per_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL+UL frozen (v6+rlul) | R3 | 12 | -2.12±8.97 | -0.31±9.20 | -0.86±4.91 | -9.36±6.23 | 11.90±32.47 | 4.71±1.94 |

## Uwagi

- Wartości: średnia ± odchylenie std po foldach OOS (`status=ok`).
- `excess_return` = total_return strategii − HODL w tym samym foldzie.
- Surowy dump: `folds.csv`, `run_results.json`.
