# Walk-forward — tabela zbiorcza

Wygenerowano: 2026-10-26T15:01:26.735386

Run: `M2_thesis_951830_20261026_A1_B10`

Folds: A1, A2, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10

| Ramię | plan | n | total_return | excess_return | sharpe_ratio | max_drawdown | calmar_ratio | avg_transactions_per_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL continue-learn (v6+rl) | M2 | 12 | -3.31±11.04 | -1.49±7.08 | -1.24±5.57 | -10.80±7.91 | 12.66±30.29 | 9.09±6.28 |

## Uwagi

- Wartości: średnia ± odchylenie std po foldach OOS (`status=ok`).
- `excess_return` = total_return strategii − HODL w tym samym foldzie.
- Surowy dump: `folds.csv`, `run_results.json`.
