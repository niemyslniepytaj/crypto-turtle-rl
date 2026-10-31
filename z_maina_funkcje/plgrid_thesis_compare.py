"""
Osobny entry: agregacja thesis_results → jedna tabela porównawcza + raport I3.
Nie woła treningu ani sekwencji.
"""
import argparse
import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rl_module.services.thesis.compare import compare_runs, discover_run_dirs
from rl_module.services.thesis.analysis import generate_report
from rl_module.services.thesis.completeness import run_completeness_check


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Porównanie ramion tezy (folds.csv → tabela)")
    parser.add_argument(
        "--run-dirs",
        nargs="*",
        default=None,
        help="Ścieżki do katalogów run (domyślnie: discover po label/mode)",
    )
    parser.add_argument("--run-label", default=None, help="Filtr manifest.run_label")
    parser.add_argument(
        "--portfolio-mode",
        default=None,
        choices=["reset", "continuous"],
        help="Filtr manifest.portfolio_mode (nie mieszaj reset z continuous)",
    )
    parser.add_argument(
        "--include-a",
        action="store_true",
        help="Uwzględnij foldy A* (domyślnie tylko B*)",
    )
    parser.add_argument("--out-dir", default=None, help="Katalog wyjściowy")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generuj raport I3 (Wilcoxon/Holm/Kerby + LaTeX/CSV)",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Nie przerywaj przy brakujących oknach B (smoke)",
    )
    parser.add_argument(
        "--hypothesis-set",
        default="intermediate",
        choices=["intermediate", "extended"],
        help="Zestaw hipotez (thesis/hypotheses.py)",
    )
    parser.add_argument(
        "--completeness",
        action="store_true",
        help="Tylko tabela kompletności (I4.1)",
    )
    args = parser.parse_args()

    results_root = os.path.join(BASE_DIR, "shared", "simulation", "thesis_results")
    run_dirs = args.run_dirs
    if not run_dirs:
        run_dirs = discover_run_dirs(
            results_root,
            run_label=args.run_label,
            portfolio_mode=args.portfolio_mode,
        )
    if not run_dirs:
        print("[Compare] Brak run_dir — podaj --run-dirs lub odpal sekwencję najpierw.")
        sys.exit(1)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.run_label or "all"
    mode = args.portfolio_mode or "mixed"

    if args.completeness:
        out_csv = os.path.join(
            results_root, "_compare", f"completeness_{label}_{mode}_{stamp}.csv"
        )
        run_completeness_check(
            results_root,
            out_csv,
            run_label=args.run_label,
            portfolio_mode=args.portfolio_mode,
        )
        sys.exit(0)

    out_dir = args.out_dir or os.path.join(
        results_root, "_compare", f"{label}_{mode}_{stamp}"
    )

    compare_runs(
        run_dirs,
        out_dir,
        only_b_folds=not args.include_a,
        label=f"{label}/{mode}",
    )

    if args.report:
        report_mode = args.portfolio_mode or "reset"
        report_dir = os.path.join(
            results_root, "_report", f"{label}_{report_mode}_{stamp}"
        )
        generate_report(
            run_dirs,
            report_dir,
            portfolio_mode=report_mode,
            only_b_folds=not args.include_a,
            allow_incomplete=args.allow_incomplete,
            label=f"{label}/{report_mode}",
            hypothesis_set=args.hypothesis_set,
        )
