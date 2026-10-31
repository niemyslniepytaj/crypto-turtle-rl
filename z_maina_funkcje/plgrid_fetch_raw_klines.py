"""Pobranie surowych swiec z Binance dla okresu badawczego pracy magisterskiej.

Zakres domyslny: 2025-05-11 13:19 -> 2026-08-31.
Ostatnia swieca train E104 (ML) = 2025-05-11 13:18 — pobieramy od nastepnej minuty.
Nakladka na sym.csv (w srodku zakresu) pozwala porownac OHLCV z E104.

Uruchomienie:
    python z_maina_funkcje/plgrid_fetch_raw_klines.py
    python z_maina_funkcje/plgrid_fetch_raw_klines.py --symbol BTCUSDT
"""

import argparse
import os
import time

import pandas as pd
import requests

API_URL = "https://api.binance.com/api/v3/klines"
BATCH_LIMIT = 1000

# Pierwsza dozwolona swieca: minuta po ostatniej train E104 (2025-05-11 13:18).
DEFAULT_START = "2025-05-11 13:19:00+00:00"
DEFAULT_END = "2026-08-31 23:59:00+00:00"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "shared", "files", "raw_data")
SYM_CSV = os.path.join(
    REPO_ROOT,
    "shared",
    "E104_prep_eth_universal_full_1m",
    "sym.csv",
)

# Kolumny zwracane przez /api/v3/klines. Zapisujemy takze taker_buy_* oraz
# number_of_trades - bez nich nie da sie odtworzyc cechy buy_pressure z E104.
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]

OUTPUT_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
]

INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def to_ms(value):
    return int(pd.Timestamp(value).tz_convert("UTC").timestamp() * 1000)


def fetch_batch(symbol, interval, start_ms, end_ms, max_retries=5):
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": BATCH_LIMIT,
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(API_URL, params=params, timeout=30)
        except requests.RequestException as exc:
            wait = 2 ** attempt
            print(f"  blad sieci ({exc}); ponowienie za {wait}s")
            time.sleep(wait)
            continue

        # 429 = przekroczony limit zapytan, 418 = tymczasowy ban IP.
        if response.status_code in (429, 418):
            wait = int(response.headers.get("Retry-After", 2 ** attempt))
            print(f"  limit zapytan (HTTP {response.status_code}); przerwa {wait}s")
            time.sleep(wait)
            continue

        if response.status_code != 200:
            wait = 2 ** attempt
            print(f"  HTTP {response.status_code}; ponowienie za {wait}s")
            time.sleep(wait)
            continue

        return response.json()

    raise RuntimeError(f"Nie udalo sie pobrac partii po {max_retries} probach")


def resume_point(out_path, interval_ms):
    """Zwraca timestamp, od ktorego wznowic pobieranie, albo None."""
    if not os.path.exists(out_path):
        return None

    existing = pd.read_csv(out_path, usecols=["open_time"])
    if existing.empty:
        return None

    last = pd.Timestamp(existing["open_time"].iloc[-1])
    print(f"Plik juz istnieje ({len(existing)} wierszy, ostatnia swieca {last}).")
    return int(last.timestamp() * 1000) + interval_ms


def download(symbol, interval, start, end, out_path):
    interval_ms = INTERVAL_MS[interval]
    end_ms = to_ms(end)

    cursor = resume_point(out_path, interval_ms)
    if cursor is None:
        cursor = to_ms(start)
        write_header = True
    else:
        write_header = False
        if cursor > end_ms:
            print("Plik pokrywa juz caly zadany zakres - pobieranie pominiete.")
            return

    expected = (end_ms - cursor) // interval_ms + 1
    print(f"Pobieranie {symbol} {interval}: ok. {expected} swiec do sciagniecia.")

    downloaded = 0
    while cursor <= end_ms:
        batch = fetch_batch(symbol, interval, cursor, end_ms)
        if not batch:
            break

        frame = pd.DataFrame(batch, columns=KLINE_COLUMNS)
        frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        for column in OUTPUT_COLUMNS[1:]:
            frame[column] = pd.to_numeric(frame[column])

        frame[OUTPUT_COLUMNS].to_csv(
            out_path,
            mode="w" if write_header else "a",
            header=write_header,
            index=False,
        )
        write_header = False

        downloaded += len(frame)
        cursor = int(batch[-1][0]) + interval_ms

        last_seen = frame["open_time"].iloc[-1]
        print(f"  {downloaded:>7} swiec, ostatnia {last_seen}")

        # Waga zapytania przy limit=1000 wynosi 5, budzet to 6000/min.
        time.sleep(0.15)

    print(f"Pobrano {downloaded} swiec.")


def audit(out_path, interval, symbol):
    interval_ms = INTERVAL_MS[interval]
    frame = pd.read_csv(out_path)
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)

    before = len(frame)
    frame = frame.drop_duplicates(subset="open_time").sort_values("open_time")
    frame = frame.reset_index(drop=True)
    if len(frame) != before:
        print(f"Usunieto {before - len(frame)} zduplikowanych swiec.")
        frame.to_csv(out_path, index=False)

    print(f"\nZakres: {frame['open_time'].iloc[0]} -> {frame['open_time'].iloc[-1]}")
    print(f"Swiec: {len(frame)}")

    deltas = frame["open_time"].diff().dt.total_seconds() * 1000
    gaps = deltas[deltas > interval_ms]
    if gaps.empty:
        print("Luk w szeregu: brak.")
    else:
        print(f"UWAGA - luk w szeregu: {len(gaps)}. Najwieksze:")
        for index in gaps.sort_values(ascending=False).head(5).index:
            missing = int(gaps[index] / interval_ms) - 1
            print(f"  przed {frame['open_time'].iloc[index]}: brak {missing} swiec")

    if symbol.upper().startswith("ETH"):
        compare_with_sym(frame)


def compare_with_sym(frame):
    """Porownuje pobrane OHLCV z sym.csv na czesci wspolnej.

    Zgodnosc na nakladce oznacza, ze pobrane dane sa tym samym szeregiem, na
    ktorym powstal E104 - i ze czesci swiezej mozna uzyc jako jego przedluzenia.
    """
    if not os.path.exists(SYM_CSV):
        print("\nBrak sym.csv - weryfikacja nakladki pominieta.")
        return

    columns = ["open_time", "open", "high", "low", "close", "volume"]
    sym = pd.read_csv(SYM_CSV, usecols=columns)
    sym["open_time"] = pd.to_datetime(sym["open_time"], utc=True)

    merged = sym.merge(frame[columns], on="open_time", suffixes=("_sym", "_new"))
    print(f"\nWeryfikacja wzgledem sym.csv: {len(merged)} wspolnych swiec.")

    if merged.empty:
        print("BLAD - brak czesci wspolnej, sprawdz zadany poczatek zakresu.")
        return

    covers_sym = (
        frame["open_time"].iloc[0] <= sym["open_time"].iloc[0]
        and frame["open_time"].iloc[-1] >= sym["open_time"].iloc[-1]
    )
    if covers_sym and len(merged) < len(sym):
        print(f"UWAGA - sym.csv ma {len(sym)} swiec, dopasowano tylko {len(merged)}.")

    ok = True
    for column in ["open", "high", "low", "close", "volume"]:
        diff = (merged[f"{column}_sym"] - merged[f"{column}_new"]).abs()
        worst = diff.max()
        mismatches = int((diff > 1e-6).sum())
        status = "OK" if mismatches == 0 else "ROZBIEZNOSC"
        print(f"  {column:<7} max |roznica| = {worst:.10g}  niezgodnych: {mismatches}  [{status}]")
        ok = ok and mismatches == 0

    if ok:
        print("\nNakladka zgodna co do bitu - dane swieze sa przedluzeniem E104.")
    else:
        print(
            "\nNakladka niezgodna. Zanim uzyjesz danych, sprawdz symbol "
            "(ETHUSDT vs ETHUSDC) i rynek (spot vs futures).",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--interval", default="1m", choices=sorted(INTERVAL_MS))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Pomija pobieranie, sprawdza istniejacy plik",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    start_tag = pd.Timestamp(args.start).strftime("%Y%m%d")
    end_tag = pd.Timestamp(args.end).strftime("%Y%m%d")
    filename = f"{args.symbol}_{args.interval}_{start_tag}_{end_tag}.csv"
    out_path = os.path.join(args.out_dir, filename)

    if not args.audit_only:
        download(args.symbol, args.interval, args.start, args.end, out_path)

    if not os.path.exists(out_path):
        print("Brak pliku wyjsciowego - nic do sprawdzenia.")
        return

    audit(out_path, args.interval, args.symbol)
    print(f"\nPlik: {out_path}")


if __name__ == "__main__":
    main()
