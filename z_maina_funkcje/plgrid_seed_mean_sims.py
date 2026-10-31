"""
Osobny entry: symulacje uśrednione po ziarnach z plików sekwencji testowej.
Nie wywołuje treningu ani sekwencji.
"""
import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rl_module.services.thesis.seed_mean_sims import build_seed_mean_sims

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Symulacje uśrednione po ziarnach (bez R0)")
    parser.add_argument("--outputs-dir", default=os.path.join(BASE_DIR, "shared", "simulation", "outputs"))
    parser.add_argument("--label", default="thesis")
    parser.add_argument("--arms", nargs="*", default=None, help="np. M1 M2 M3 (domyślnie wszystkie bez R0)")
    args = parser.parse_args()
    files = build_seed_mean_sims(args.outputs_dir, arms=args.arms, label=args.label)
    print(f"Zapisano {len(files)} plików.")
