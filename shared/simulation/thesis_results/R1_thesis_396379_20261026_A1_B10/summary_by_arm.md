# Walk-forward — tabela zbiorcza

Wygenerowano: 2026-10-26T09:46:39.280795

Run: `R1_thesis_396379_20261026_A1_B10`

Folds: A1, A2, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10

| Ramię | plan | n | total_return | excess_return | sharpe_ratio | max_drawdown | calmar_ratio | avg_transactions_per_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MP static (v6+mp) | R1 | 12 | -1.75±12.71 | 0.06±5.83 | -0.72±5.65 | -11.64±8.23 | 21.67±53.14 | 6.47±4.00 |

## Uwagi

- Wartości: średnia ± odchylenie std po foldach OOS (`status=ok`).
- `excess_return` = total_return strategii − HODL w tym samym foldzie.
- Surowy dump: `folds.csv`, `run_results.json`.
