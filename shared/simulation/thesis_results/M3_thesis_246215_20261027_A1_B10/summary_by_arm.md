# Walk-forward — tabela zbiorcza

Wygenerowano: 2026-10-27T07:33:52.271129

Run: `M3_thesis_246215_20261027_A1_B10`

Folds: A1, A2, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10

| Ramię | plan | n | total_return | excess_return | sharpe_ratio | max_drawdown | calmar_ratio | avg_transactions_per_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL + UL (v6+rlul) | M3 | 12 | -4.87±11.82 | -3.05±6.94 | -1.90±5.80 | -11.82±8.96 | 10.63±31.10 | 26.28±14.53 |

## Uwagi

- Wartości: średnia ± odchylenie std po foldach OOS (`status=ok`).
- `excess_return` = total_return strategii − HODL w tym samym foldzie.
- Surowy dump: `folds.csv`, `run_results.json`.
