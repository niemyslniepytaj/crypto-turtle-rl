# Walk-forward — tabela zbiorcza

Wygenerowano: 2026-10-26T09:45:30.512043

Run: `R1_thesis_951830_20261026_A1_B10`

Folds: A1, A2, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10

| Ramię | plan | n | total_return | excess_return | sharpe_ratio | max_drawdown | calmar_ratio | avg_transactions_per_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MP static (v6+mp) | R1 | 12 | -1.66±11.11 | 0.16±7.07 | -0.64±5.21 | -10.43±7.12 | 17.72±44.98 | 5.49±3.72 |

## Uwagi

- Wartości: średnia ± odchylenie std po foldach OOS (`status=ok`).
- `excess_return` = total_return strategii − HODL w tym samym foldzie.
- Surowy dump: `folds.csv`, `run_results.json`.
