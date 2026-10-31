# Walk-forward — tabela zbiorcza

Wygenerowano: 2026-10-26T10:29:16.399932

Run: `R2_thesis_396379_20261026_A1_B10`

Folds: A1, A2, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10

| Ramię | plan | n | total_return | excess_return | sharpe_ratio | max_drawdown | calmar_ratio | avg_transactions_per_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL frozen (v6+rl) | R2 | 12 | -3.30±10.63 | -1.48±8.01 | -1.22±5.26 | -10.87±7.61 | 12.56±33.93 | 9.80±5.72 |

## Uwagi

- Wartości: średnia ± odchylenie std po foldach OOS (`status=ok`).
- `excess_return` = total_return strategii − HODL w tym samym foldzie.
- Surowy dump: `folds.csv`, `run_results.json`.
