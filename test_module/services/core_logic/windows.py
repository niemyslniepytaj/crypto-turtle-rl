"""Ładowanie i krojenie okien czasowych z shared/files/windows.json."""

import json
import os

import pandas as pd


def load_windows(base_dir):
    """Wczytuje definicje okien z shared/files/windows.json."""
    path = os.path.join(base_dir, "shared", "files", "windows.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_window_ids(window_spec):
    """Parsuje specyfikację okien (A1, A1+A2, lista) do listy identyfikatorów."""
    if window_spec is None:
        return None
    if isinstance(window_spec, (list, tuple)):
        ids = [str(x).strip() for x in window_spec if str(x).strip()]
        return ids or None
    s = str(window_spec).strip()
    if not s:
        return None
    if "+" in s:
        return [p.strip() for p in s.split("+") if p.strip()]
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def expand_window_ids(windows, window_spec):
    """
    Jak parse_window_ids, ale rozwijające aliases (np. train -> T1..T5).
    Alias może stać sam albo w złożeniu: train+A1 -> T1+...+T5+A1.
    """
    ids = parse_window_ids(window_spec)
    if not ids:
        return None
    aliases = windows.get("aliases") or {}
    out = []
    for wid in ids:
        if wid in aliases:
            out.extend([str(x).strip() for x in aliases[wid] if str(x).strip()])
        else:
            out.append(wid)
    return out or None


def _to_dt(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="ms", utc=True)
    return pd.to_datetime(series, utc=True)


def slice_df(df, start, end, time_col="open_time"):
    """Wycina wiersze DataFrame w przedziale półotwartym [start, end)."""
    t = _to_dt(df[time_col])
    start = pd.Timestamp(start, tz="UTC")
    end = pd.Timestamp(end, tz="UTC")
    return df.loc[(t >= start) & (t < end)].copy()


def slice_windows(df, windows, window_spec, time_col="open_time"):
    """
    Wycina jedno okno albo kilka połączonych (sąsiednich w sequence).
    Np. 'A1', 'A1+A2', 'train', 'train+A1', ['T2','T3','T4','T5','A1','A2'].
    Alias 'train' rozwijany do T1..T5 (ten sam zakres co pełny train).
    """
    ids = expand_window_ids(windows, window_spec)
    if not ids:
        return df.copy()

    wmap = windows["windows"]
    seq = windows.get("sequence", list(wmap.keys()))
    aliases = windows.get("aliases") or {}

    for wid in ids:
        if wid not in wmap and wid not in aliases:
            raise KeyError(f"Nieznane okno: {wid}. Dostępne: {list(wmap.keys())}")

    # Alias-only single id (np. samo 'train') — użyj zdefiniowanego zakresu aliasu w wmap jeśli jest
    raw_ids = parse_window_ids(window_spec)
    if raw_ids and len(raw_ids) == 1 and raw_ids[0] in wmap and raw_ids[0] in aliases:
        # np. window_id='train' — trzymaj jawny start/end z wmap (tożsamy z T1..T5)
        return slice_df(df, wmap[raw_ids[0]]["start"], wmap[raw_ids[0]]["end"], time_col=time_col)

    if len(ids) > 1:
        try:
            positions = [seq.index(wid) for wid in ids]
        except ValueError as e:
            raise ValueError(f"Okno poza sequence: {ids}") from e
        expected = list(range(positions[0], positions[0] + len(ids)))
        if positions != expected:
            raise ValueError(
                f"Okna muszą być kolejnymi w sequence (połączone): {ids}. "
                f"sequence={seq}"
            )

    start = wmap[ids[0]]["start"]
    end = wmap[ids[-1]]["end"]
    return slice_df(df, start, end, time_col=time_col)


def slice_window(df, windows, window_id, time_col="open_time"):
    return slice_windows(df, windows, window_id, time_col=time_col)
