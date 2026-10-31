# Walk-forward — tabela zbiorcza

Wygenerowano: 2026-10-26T11:11:01.446495

Run: `R3_thesis_396379_20261026_A1_B10`

Folds: A1, A2, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10

| Ramię | plan | n | total_return | excess_return | sharpe_ratio | max_drawdown | calmar_ratio | avg_transactions_per_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL+UL frozen (v6+rlul) | R3 | 12 | -2.36±10.67 | -0.55±7.72 | -0.73±4.91 | -10.72±7.34 | 14.36±37.25 | 8.65±4.92 |

## Uwagi

- Wartości: średnia ± odchylenie std po foldach OOS (`status=ok`).
- `excess_return` = total_return strategii − HODL w tym samym foldzie.
- Surowy dump: `folds.csv`, `run_results.json`.
