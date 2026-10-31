# LSTM Model Package — Dokumentacja i Instrukcja Obsługi

Samodzielna, przenośna biblioteka inferencyjna dla modeli sieci neuronowych LSTM log-return.
Pakiet przetwarza dane rynkowe i generuje probabilistyczne sygnały transakcyjne **BUY** oraz **SELL** w skali $[0, 100]$.

Biblioteka jest przeznaczona do **wsadowych backtestów historycznych** (zerokopiowe okna czasowe) oraz do **generacji sygnału dla ostatniej świecy w pętli bota**, działając w pełni autonomicznie bez dostępu do środowiska treningowego.

---

## 1. Co to jest Model Package i jaka jest jego rola?

W ekosystemie modeli uczenia maszynowego `lstm_model_package` pełni rolę **niezależnej warstwy inferencyjnej (Standalone Inference Engine)**:

- **Pełna izolacja od kodu treningowego:** Pakiet nie importuje modułów badawczych (`lstm_logreturn_model`, `common`, `data_preparation`). Do uruchomienia wystarczy katalog pakietu oraz pliki wag modelu.
- **Jednolity standard sygnałów probabilistycznych:** Sygnały reprezentują matematyczne prawdopodobieństwo przekroczenia przez cenę progu opłacalności transakcji ($T$):
  $$P(\text{BUY}) = P(r > T) = \left[1 - \Phi\left(\frac{T - \mu}{\sigma}\right)\right] \times 100$$
  $$P(\text{SELL}) = P(r < -T) = \Phi\left(\frac{-T - \mu}{\sigma}\right) \times 100$$
  gdzie $\mu$ to oczekiwany zwrot z modelu, $\sigma$ to bieżąca zmienność rynku, a $\Phi$ to dystrybuanta rozkładu normalnego obliczana w C przez `scipy.special.ndtr`.
- **Inferencja wsadowa:** Okna czasowe są tworzone zerokopiowym widokiem pamięci (`torch.Tensor.unfold`), zmienność liczona wektorowo (`pandas.rolling`), a sieć uruchamiana w `torch.inference_mode()` partiami po `batch_size` próbek. Przykładowy pomiar znajduje się w sekcji 8.
- **Zgodność z potokiem testowym:** Domyślne ustawienia (`checkpoint='best'`, `apply_calibration=False`, `window_sigma=100`) odtwarzają to samo, co `lstm_test.py` i symulacja v2 (por. sekcja 6.5).

---

## 2. Wymagania i Środowisko Uruchomieniowe

| Biblioteka | Import | Minimalna wersja | Zastosowanie |
| :--- | :--- | :--- | :--- |
| **PyTorch** | `torch` | $\ge 2.0$ | Inferencja sieci LSTM (CPU / CUDA / ROCm) |
| **NumPy** | `numpy` | $\ge 1.24$ | Operacje macierzowe i zerokopiowe widoki pamięci |
| **Pandas** | `pandas` | $\ge 2.0$ | Obsługa szeregów czasowych OHLCV |
| **SciPy** | `scipy` | $\ge 1.10$ | Szybka dystrybuanta rozkładu normalnego (`scipy.special.ndtr`) |
| **scikit-learn**| `sklearn`| $\ge 1.2$ | Skalery cech i celu (`RobustScaler`, `StandardScaler`) |
| **Joblib** | `joblib` | $\ge 1.2$ | Odczyt zserializowanych obiektów skalera i kalibratora |

Instalacja w czystym środowisku:
```bash
pip install torch numpy pandas scipy scikit-learn joblib
```

---

## 3. Struktura Katalogu Projektu Docelowego

Aby wdrożyć model w zewnętrznym środowisku, wystarczy skopiować folder `lstm_model_package` oraz katalog z wytrenowanym modelem:

```
twoj_projekt/
├── lstm_model_package/
│   ├── __init__.py               # Eksport interfejsu publicznego
│   ├── lstm_model_package.py     # Główna fasada inferencyjna
│   ├── signals.py                # Szybki rachunek sigmy i sygnałów Gaussa
│   ├── feature_engineering.py    # Generator cech FE 3.0 (zestaw `max` = 39 cech)
│   ├── compat.py                 # Shimy wstecznej kompatybilności i unpickler kalibratora
│   ├── scaler_registry.py        # Offline rejestr kolumn skalerów dla datasetów
│   ├── lstm_arch.py              # Definicja architektury sieci neuronowej
│   ├── runtime_utils.py          # MinRowsInt, skaler celu, zaokrąglanie predykcji
│   ├── test_signals.py           # Test dymny i pomiar przepustowości
│   └── USAGE.md                  # Niniejsza instrukcja
└── models/
    └── LR1107_eth_model_E104_1107/
        ├── model_config.json     # Konfiguracja architektury i cech (wymagany)
        ├── lstm_model.pth        # Wagi sieci PyTorch (wymagany)
        ├── lstm_model_swa.pth    # Opcjonalny checkpoint SWA (checkpoint='swa')
        ├── feature_scaler.pkl    # Skaler cech wejściowych (wymagany dla trybu raw)
        └── calibrator.pkl        # Opcjonalny kalibrator izotoniczny (stosowany tylko z apply_calibration=True)
```

W repozytorium modele LR leżą w `lstm_logreturn_model/lstm_lr_models/`.

---

## 4. Specyfikacja API i Konfiguracji

### 4.1. `SignalConfig`
Struktura konfiguracji generatora sygnałów:

```python
from lstm_model_package import SignalConfig

cfg = SignalConfig(
    threshold=0.001,      # Minimalny oczekiwany zwrot (0.1%), pokrywający prowizje i spread
    window_sigma=100      # Okno kroczące odchylenia standardowego zmienności rynku (świece)
)
```

- `threshold` (float): Próg ekonomiczny $T$ we wzorach z sekcji 1. Gdy `signal_config` nie zostanie podany, pakiet używa `signal_threshold` zapisanego w `model_config.json` (modele LR: `0.001`); klasa `SignalConfig` ma tę samą wartość domyślną `0.001`.
- `window_sigma` (int, domyślnie `100`): Długość okna zmienności (liczba log-zwrotów w oknie kroczącym). Potok testowy i symulacja v2 używają `100` (`LSTMConfig.SIGNAL_WINDOW_SIGMA`), więc inna wartość daje sygnały niezgodne z zapisanymi wynikami testów. Denormalizacja modeli `vol_norm` zawsze korzysta ze stałego okna 100, niezależnie od `window_sigma`.
- `SignalConfig` nie ma innych pól.

### 4.2. Inicjalizacja `LSTMModelPackage`
Pakiet można zainicjalizować standardowym konstruktorem lub alternatywną metodą fabryczną `from_directory` (identycznie jak w `SignalModelPackage`):

```python
from lstm_model_package import LSTMModelPackage, SignalConfig

# Opcja A: Metoda fabryczna (rekomendowana)
pkg = LSTMModelPackage.from_directory(
    "models/LR1107_eth_model_E104_1107",
    signal_config=cfg,     # Opcjonalne; jeśli brak, pobiera domyślne parametry z modelu
    device="cpu",          # 'cpu' lub 'cuda'
    checkpoint="best",     # 'best' (domyślnie: lstm_model.pth) lub 'swa' (lstm_model_swa.pth)
    apply_calibration=False  # True stosuje calibrator.pkl (opt-in, patrz sekcja 6.5)
)

# Opcja B: Konstruktor bezpośredni
pkg = LSTMModelPackage(
    model_dir="models/LR1107_eth_model_E104_1107",
    signal_config=cfg,
    device="cpu"
)
```

`checkpoint='last'` jest przestarzałym aliasem `'best'` (emituje `DeprecationWarning`): trening zapisuje jeden checkpoint niebędący SWA — wagi o najlepszej walidacji w `lstm_model.pth` — a osobnego pliku z ostatnią epoką nie ma. Nieznana wartość `checkpoint` zgłasza `ValueError`.

### 4.3. Metody Publiczne

| Metoda / Właściwość | Zwracany typ | Opis |
| :--- | :--- | :--- |
| `predict_batch(df, data_mode='auto', input_type=None, batch_size=4096, dataset_id=None, return_df=True, drop_warmup=True)` | `pd.DataFrame` lub `dict[str, np.ndarray]` | Wsadowa generacja sygnałów dla całego szeregu czasowego. `input_type` (`'raw'` / `'prepared'`) ma pierwszeństwo przed `data_mode` (`'auto'` rozpoznaje tryb po obecności pierwszej cechy modelu w kolumnach). `dataset_id` włącza ostrzeżenie o niezgodności zbioru (tryb `prepared`). Gdy `return_df=True` zwraca DataFrame, gdy `False` — słownik tablic NumPy (`timestamp`, `close`, `mu`, `sigma`, `signal_buy`, `signal_sell`). `drop_warmup=True` usuwa okna, dla których sigma jeszcze nie istnieje; `drop_warmup=False` zachowuje wszystkie okna (sigma = `NaN`, sygnały liczone z sigmą uzupełnioną medianą), co odtwarza siatkę kroków `lstm_test.py` i symulacji v2. |
| `predict_live(df, data_mode='auto', input_type=None, dataset_id=None)` | `dict` | Obliczenie pojedynczego sygnału dla ostatniej zamkniętej świecy w buforze (`timestamp`, `close`, `mu`, `sigma`, `signal_buy`, `signal_sell`). |
| `get_model_info()` | `dict` | Słownik metadanych modelu: `model_id`, `data_id`, `symbol`, `lookback`, `horizon`, `n_features`, `features`, `feature_set`, parametry architektury (`rnn_type`, `hidden_size`, `num_layers`, `use_attention`, `use_rnn_norm`, `pooling_mode`, `target_mode`), `checkpoint`, `has_swa`, `is_fusion`, `is_standalone`, `device`, `signal_threshold`, `signal_window_sigma`, a także `calibrated` / `calibrator_available` (plik `calibrator.pkl` został wczytany) i `calibration_applied` (kalibrator jest faktycznie stosowany, tzn. `apply_calibration=True`). |
| `min_rows_required` | `int` oraz `callable` | Minimalna liczba świec wejściowych w buforze. Działa bezpośrednio jako liczba całkowita (wartość dla trybu `raw`) oraz jako funkcja z wyborem trybu. Dla LR1107: `pkg.min_rows_required == 248`, `pkg.min_rows_required('raw')` -> 248, `pkg.min_rows_required('prepared')` -> 148 (wzór w sekcji 6.4). |

### 4.4. Tolerancja Formatów Danych i Zabezpieczenia API (Out-of-the-Box)

Pakiet został zaprojektowany z myślą o bezpośredniej integracji z różnymi giełdami i bibliotekami (np. CCXT) bez konieczności pisania żmudnego kodu konwertującego:

- **Automatyczne aliasy kolumn czasowych:** Jeśli DataFrame nie posiada kolumny `open_time`, pakiet automatycznie wykrywa i mapuje kolumny `timestamp` lub `time`.
- **Automatyczna konwersja typów tekstowych:** Jeśli giełdowe API zwraca liczby jako stringi w JSON (np. `'open': '2450.50'`), pakiet automatycznie rzutuje je na typ `float`.
- **Automatyczne sortowanie chronologiczne:** Jeżeli giełda zwraca świece od najnowszej do najstarszej (kolejność malejąca), pakiet automatycznie układa je w prawidłowej kolejności rosnącej.
- **Hierarchiczna autodetekcja jednostek czasu:** Timestampy numeryczne są automatycznie rozpoznawane według rzędu wielkości (`s`, `ms`, `us`, `ns`) oraz jako stringi ISO UTC, zapobiegając błędnemu rzutowaniu na 1970 rok.
- **Gwarancja synchronizacji świecy bieżącej:** Metoda `predict_live(df)` oraz ostatni wiersz `predict_batch(df)` zawsze precyzyjnie odpowiadają najnowszej zamkniętej świecy z bufora (`signal['timestamp'] == df['open_time'].iloc[-1]` oraz `signal['close'] == df['close'].iloc[-1]`), eliminując jakiekolwiek opóźnienia sygnału transakcyjnego w pętli bota.

---

## 5. Przypadki Użycia z Przykładami Kodu

### Przypadek 1: Backtest Historyczny na Długim Szeregu (Batch Processing)

Uruchomienie modelu na długim szeregu czasowym (np. roczne dane 1-minutowe). Czas zależy od modelu i sprzętu; pomiar referencyjny podaje sekcja 8.

```python
import pandas as pd
from lstm_model_package import LSTMModelPackage, SignalConfig

# 1. Konfiguracja i ładowanie modelu
cfg = SignalConfig(threshold=0.001, window_sigma=100)
pkg = LSTMModelPackage("models/LR1107_eth_model_E104_1107", signal_config=cfg, device='cpu')

# 2. Wczytanie surowych danych historycznych OHLCV (np. wyeksportowanych z giełdy)
# Wymagane kolumny: open_time, open, high, low, close, volume, taker_buy_base_asset_volume
df_historical = pd.read_csv("data/eth_1m_historical_500k.csv")

# 3. Wsadowe wygenerowanie sygnałów transakcyjnych
# batch_size=4096 zapewnia optymalną utylizację wektorową BLAS/CUDA
signals_df = pkg.predict_batch(df_historical, data_mode='raw', batch_size=4096)

# 4. Przekazanie sygnałów do zewnętrznego symulatora lub analizatora
print(signals_df[['timestamp', 'close', 'mu', 'sigma', 'signal_buy', 'signal_sell']].head())
print(f"Wygenerowano {len(signals_df):,} sygnałów bez braków danych.")
```

---

### Przypadek 2: Pętla Bota Transakcyjnego na Żywo (Live Candle Processing)

W handlu na żywo bot co minutę pobiera z giełdy bufor ostatnich $N$ świec (`min_rows_required`) i generuje sygnał dla ostatniej świecy.

```python
import time
import pandas as pd
from lstm_model_package import LSTMModelPackage

pkg = LSTMModelPackage("models/LR1107_eth_model_E104_1107")
min_bars = pkg.min_rows_required  # np. 248 świec dla modeli bazowych

print(f"Bot uruchomiony. Minimalna długość bufora: {min_bars} świec.")

while True:
    # 1. Pobranie najnowszego bufora świec z API giełdy (np. Binance REST API)
    # buffer_df zawiera co najmniej min_bars wierszy z surowymi cenami
    buffer_df = exchange_client.fetch_ohlcv('ETHUSDT', interval='1m', limit=min_bars + 20)

    # 2. Generowanie sygnału dla bieżącej świecy
    signal = pkg.predict_live(buffer_df, data_mode='raw')

    timestamp = signal['timestamp']
    price = signal['close']
    buy_prob = signal['signal_buy']
    sell_prob = signal['signal_sell']

    print(f"[{timestamp}] Cena: {price:.2f} | BUY: {buy_prob:.1f}% | SELL: {sell_prob:.1f}%")

    # 3. Zewnętrzna logika decyzyjna (zarządzana przez Twojego bota)
    if buy_prob >= 70.0 and buy_prob > sell_prob:
        external_order_executor.buy(symbol='ETHUSDT', reason='strong_model_signal')
    elif sell_prob >= 70.0 and sell_prob > buy_prob:
        external_order_executor.sell(symbol='ETHUSDT', reason='strong_model_signal')

    time.sleep(60)
```

---

### Przypadek 3: Uruchomienie w Całkowicie Odizolowanym Środowisku

Pakiet jest w pełni niezależny od struktury folderów repozytorium. Można go umieścić w dowolnym podkatalogu zewnętrznego systemu:

```python
import sys
from pathlib import Path

# Dodanie ścieżki do pakietu znajdującego się w dowolnym miejscu systemu
sys.path.insert(0, "/opt/trading_system/packages")

from lstm_model_package import LSTMModelPackage

pkg = LSTMModelPackage("/opt/trading_system/models/LR1407_eth_model_E109_2209")
info = pkg.get_model_info()

print(f"Model ID: {info['model_id']}")
print(f"Liczba cech wejściowych: {info['n_features']}")
print(f"Kalibrator dostępny: {info['calibrator_available']}, zastosowany: {info['calibration_applied']}")
```

---

### Przypadek 4: Obsługa Modeli ze Zmienną Normalizacją Celu (`vol_norm`, LR1364)

Modele uczone z parametrem `target_mode='vol_norm'` (np. LR1364) optymalizują zwrot podzielony przez zmienność. Pakiet automatycznie wykrywa tę flagę i przywraca rzeczywistą skalę zwrotu:

```python
from lstm_model_package import LSTMModelPackage

# Model LR1364 został wytrenowany z target_mode='vol_norm'
pkg = LSTMModelPackage("models/LR1364_eth_model_E109_2009")

# Inferencja automatycznie stosuje denormalizację mu = mu_norm * sigma_100
df_signals = pkg.predict_batch(df_raw, data_mode='raw')

# Zwrócone wartości 'mu' są już w naturalnej skali log-zwrotu (np. +0.0008 zamiast skali znormalizowanej)
print(df_signals[['mu', 'sigma', 'signal_buy', 'signal_sell']].head())
```

Przy `drop_warmup=True` dla modeli `vol_norm` odrzucane są także okna, w których sigma ze stałego okna 100 jeszcze nie istnieje (nawet gdy `window_sigma` jest krótsze), bo przeskalowanie `mu` opierałoby się wtedy na mediance zastępczej.

---

### Przypadek 5: Odtworzenie Symulacji v2 z Wyjścia Pakietu

Symulacja v2 w `lstm_test.py` pracuje na wszystkich krokach testu, także tych z niezdefiniowaną sigmą rozgrzewkową. Aby uzyskać z pakietu te same liczby, należy wyłączyć `apply_calibration` (domyślnie wyłączone) i użyć `drop_warmup=False`:

```python
from lstm_model_package import LSTMModelPackage

pkg = LSTMModelPackage("lstm_logreturn_model/lstm_lr_models/LR1454_eth_model_E114_3009")
out = pkg.predict_batch(test_df, input_type='prepared', drop_warmup=False)
n = len(y_true)                       # liczba kroków z celem (len(test_df) - lookback - horizon)
mu, sigma = out['mu'].to_numpy()[:n], out['sigma'].to_numpy()[:n]
# mu, y_true, close oraz sigma trafiają do evaluate_lr_simulation_v2(...)
```

Test `lstm_logreturn_model/tests/test_lstm_model_package_parity.py` odtwarza w ten sposób `metrics.json` zapisanych wyników LR1449 i LR1454 (względna różnica $\le 10^{-6}$).

---

## 6. Ważne Zasady i Ostrzeżenia Inżynierskie

### 6.1. Ostrzeżenie Cross-Dataset Scaling w trybie `prepared`
- **Tryb `raw` (produkcja i dane spoza zbiorów badawczych):** Podajesz surowe świece giełdowe. Cechy są wyliczane z kolumn OHLCV przez `feature_engineering.py` pakietu i skalowane **własnym skalerem modelu** (`feature_scaler.pkl`).
  Test `test_raw_vs_prepared_mu_parity` (`lstm_logreturn_model/tests/test_lstm_model_package.py`) wykazuje różnicę $\mu$ poniżej $10^{-6}$ względem trybu `prepared` po rozgrzewce (pierwsze 600 wierszy pomijane) dla LR1107 (E104), LR1414 (B104), LR1364 / LR1205 (E109) oraz LR1176 / LR1358 (E110). Zbiory E107–E109 powstały ze starszą definicją `vol_percentile_240` (udział okna $\ge$ ostatniej wartości). Inferencja pakietu wybiera tę definicję z `data_id` modelu. Wywołanie `prepare_features` bez tej flagi liczy definicję bieżącą (`<=`); obie definicje sumują się do $100 + 100/240$ (test `test_fe3_raw_parity_lr1407_and_legacy_percentile_direction`).
- **Tryb `prepared` (tylko dla plików z tego samego zbioru):** Jest to tryb, w którym pakiet odtwarza zapisane wyniki testów (`results.parquet`, `metrics.json`). Jeśli podasz plik `test.csv` z innego zbioru (np. dane przeskalowane skalerem `E109_max` do modelu trenowanego na `E104`), rozkład statystyczny cech (mediana, rozstęp, typ skalera Robust vs Standard) będzie inny, co może wypaczyć sygnały. `dataset_id` w `predict_batch` włącza ostrzeżenie o takiej niezgodności.

### 6.2. Wymóg Wolumenu Takera (`buy_pressure`) i Obsługa Ostrzeżeń
Modele, których lista cech zawiera `buy_pressure` (np. LR843, LR1107, LR1364, LR1414), wymagają kolumny `taker_buy_base_asset_volume` do wyliczenia tej cechy. Modele E114 (LR1449, LR1454) nie używają `buy_pressure` i nie potrzebują wolumenu takera.
- W przypadku jej braku pakiet przyjmuje neutralną wartość równowagi (`buy_pressure = 0.5`) i emituje widoczne ostrzeżenie `UserWarning` (dokładnie **jeden raz na instancję pakietu**, aby nie spamować logów w minutowej pętli produkcyjnej bota).
- **Ważne:** Użycie zastępczej wartości 0.5 może nieznacznie obniżyć precyzję modeli zoptymalizowanych pod kątem presji kupna (np. LR1107, LR1364, LR1414). Do zastosowań produkcyjnych i precyzyjnych backtestów należy dostarczać surowy wolumen takera z Binance.

### 6.3. Format Timestampów (`open_time`) — Zabezpieczenie przed Rokiem 1970
Standardowe API giełdowe (np. Binance `/api/v3/klines`) zwraca czas otwarcia świecy jako liczbę całkowitą w **milisekundach** (np. `1735689600000`).
- Pakiet posiada mechanizm hierarchicznej autodetekcji rzędu wielkości liczb w `open_time` (`ns` > $10^{16}$, `us` > $10^{13}$, `ms` > $10^{10}$, `s` $\le 10^{10}$) oraz stringów ISO.
- Zapewnia to bezbłędne wyznaczenie cech cyklicznych (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`), eliminując cichą pułapkę błędnego rzutowania timestampów na styczeń 1970 roku.

### 6.4. Wymagana Długość Bufora Świec (`min_rows_required`)
Aby obliczyć wskaźniki techniczne i zmienność, pakiet wymaga bufora o minimalnej długości:
$$\text{min\_rows} = \text{lookback} + \text{window\_sigma} + \text{warmup}$$
gdzie `warmup` wynika z listy cech modelu (`signals.get_required_warmup`):
- lista cech zawiera `vol_regime` (np. LR1205, LR1407): `warmup = 530`, czyli dla `lookback = 48` i `window_sigma = 100`: $\ge \mathbf{678}$ świec.
- lista cech zawiera `vol_percentile_240` (bez `vol_regime`): `warmup = 254`, czyli $\ge \mathbf{402}$ świece.
- pozostałe modele (np. LR1107, LR1449): `warmup = 100`, czyli $\ge \mathbf{248}$ świec.

Tryb `prepared` wymaga tylko $\text{lookback} + \text{window\_sigma}$ wierszy (dla LR1107: 148), bo cechy są już policzone.

### 6.5. Kalibrator, checkpoint i zgodność z `lstm_test.py`
- `lstm_test.py` oraz symulacja v2 nie stosują `calibrator.pkl` do metryk głównych (kalibrator służy tylko do metryk `*_posthoc`), dlatego pakiet domyślnie go **nie** stosuje. Zastosowanie kalibratora wymaga jawnego `apply_calibration=True`; w `get_model_info()` odpowiada za to `calibration_applied`.
- Domyślny `checkpoint='best'` ładuje `lstm_model.pth` — tak jak `lstm_test.py` (`--checkpoint best`). `'swa'` ładuje `lstm_model_swa.pth` (błąd `FileNotFoundError`, jeśli pliku nie ma).
- Gdy `model_config.json` nie zawiera klucza `fc_layers`, pakiet buduje głowicę `[64]`, identycznie jak `lstm_test.py`.

---

## 7. Tabela Kompatybilności Modeli

Kolumny „Cechy” i „Skaler” pochodzą z `model_config.json` oraz z `n_features_in_` zapisanego skalera. „Snapshot” oznacza test `test_golden_snapshot_parity` (porównanie z wynikami stosu treningowego zapisanymi przez `lstm_logreturn_model/scripts/capture_lr_golden_snapshot.py`); „parytet $\mu$” — testy `test_mu_parity_*` (różnica $\mu < 10^{-5}$ względem `lstm_test.run_inference`); „raw=prepared” — test `test_raw_vs_prepared_mu_parity`; „sim v2” — odtworzenie `metrics.json` z wyjścia pakietu.

| Model ID | Dataset | Cechy Wejściowe | Kolumny Skalera | Architektura / Target Mode | Testy w Pakiecie |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LR563 / LR594 / LR647** | E104 (ETH) | 16 cech `technical_bp` | 23 | `legacy`, raw | Snapshot |
| **LR843 / LR910** | E104 (ETH) | 16 cech `technical_bp` | 23 | `legacy`, raw | Snapshot, parytet $\mu$ |
| **LR992** | E108 (ETH MTF) | 16 cech `technical_bp` | 23 | `legacy` (`rnn_norm`), raw | Snapshot, raw MTF |
| **LR1010** | E104 (ETH) | 16 cech `technical_bp` | 23 | `legacy` (`rnn_norm`), raw | Snapshot, parytet $\mu$ |
| **LR1041** | E104 (ETH) | 16 cech `technical_bp` | 23 | `legacy`, raw | Snapshot |
| **LR1107** | E104 (ETH) | 16 cech `technical_bp` | 23 | `legacy` (BiLSTM), raw | Snapshot, raw=prepared |
| **LR1205** | E109 (ETH) | 32 cechy `full_extended` (z `vol_regime`) | 39 | `legacy`, raw | raw=prepared |
| **LR1294** | E109 (ETH) | 16 cech `technical_bp` | 39 | `legacy`, raw | Snapshot |
| **LR1334** | E109 (ETH) | 16 cech `technical_bp` | 39 | `dual_head`, raw | Snapshot, parytet $\mu$ |
| **LR1358** | E110 (ETH) | 39 cech `max` | 39 | `dual_head`, raw | raw=prepared |
| **LR1364** | E109 (ETH) | 16 cech `technical_bp` | 39 | `dual_head`, **vol_norm** | raw=prepared |
| **LR1407** | E109 (ETH) | 39 cech `max` | 39 | `dual_head`, raw | Snapshot; `vol_percentile_240` w trybie `raw` idzie starszą definicją E109 (sekcja 6.1) |
| **LR1414** | B104 (BTC) | 16 cech `technical_bp` | 23 | `dual_head`, raw | raw=prepared (poza snapshotem) |
| **LR1449** | E114 (ETH) | 14 cech `lr_optimal` (bez `buy_pressure`) | 14 | `dual_head`, raw | Snapshot, parytet $\mu$, sim v2 |
| **LR1454** | E114 (ETH) | 7 cech (`feature_set='custom'`, bez `buy_pressure`) | 14 | `dual_head`, raw | Snapshot, parytet $\mu$, sim v2 |
| *LR203 / LR426* | E104 | 16 cech | 23 | — | Zarchiwizowane w `lstm_logreturn_model/lstm_lr_models/_old/` (inna architektura wag), pakiet ich nie obsługuje |

Modele z `dual_head` korzystają z głowicy kierunek × amplituda (`tanh` × `softplus`); modele `legacy` mają liniową głowicę wyjściową. LR1454 używa 7 z 14 kolumn skalera E114: kolejność kolumn skalera jest odczytywana z `DATASET_SCALER_REGISTRY['E114']`, a w trybie `raw` kolumny skalera spoza cech modelu są uzupełniane zerami przed transformacją (nie wpływają na wejście sieci).

---

## 8. Testy i Benchmark Wydajności

Uruchomienie zestawu weryfikacyjnego bezpośrednio z wiersza poleceń:
```bash
# Weryfikacja referencyjnego modelu LR1107:
python lstm_model_package/test_signals.py

# Alternatywnie jako moduł Pythona:
python -m lstm_model_package.test_signals

# Weryfikacja dowolnego innego wytrenowanego modelu (np. z 39 cechami lub vol_norm):
python lstm_model_package/test_signals.py lstm_logreturn_model/lstm_lr_models/LR1407_eth_model_E109_2209

# Testy pytest pakietu (każdy plik osobno):
python -m pytest lstm_logreturn_model/tests/test_lstm_model_package.py -q
python -m pytest lstm_logreturn_model/tests/test_lstm_model_package_parity.py -q
python -m pytest lstm_logreturn_model/tests/test_e2e_synthetic.py -q
```

Skrypt automatycznie dostosowuje się do środowiska: w głównym repozytorium weryfikuje dane historyczne i modele referencyjne, a w środowisku zewnętrznym generuje syntetyczne świece OHLCV i sprawdza poprawność inferencji bez konieczności pobierania setek megabajtów danych testowych.

### Pomiar przepustowości

Przepustowość zależy od modelu, liczby cech i sprzętu, dlatego dokumentacja podaje jeden zmierzony przykład, a własny pomiar daje `test_signals.py` (kroki „Prepared Inference Throughput” i „Raw End-to-End Throughput”).

Pomiar referencyjny (PyTorch 2.11 CPU, 12 wątków, model LR1107, `batch_size=4096`):
- `data_mode='prepared'`, `test.csv` zbioru E104: 42 575 wierszy w 0,56 s (ok. 76 000 wierszy/s),
- `data_mode='raw'`, pierwsze 60 000 świec `data_preparation/eth_1m_BP.csv` (FE 3.0 + inferencja): 0,75 s (ok. 80 000 wierszy/s).

### Parytet numeryczny i Golden Snapshot

Test `test_golden_snapshot_parity` porównuje pakiet z wynikami stosu treningowego (14 modeli w `capture_lr_golden_snapshot.py`) z tolerancją $\Delta\mu < 10^{-6}$ oraz $\Delta\text{signal} < 5\times 10^{-4}$ p.p. (sygnał jest w skali 0–100, a pochodna sygnału po $\mu$ rośnie z $1/\sigma$, więc szum float32 rzędu $10^{-9}$ w $\mu$ daje do $\sim 10^{-4}$ p.p.). Snapshot odświeża się poleceniem `python lstm_logreturn_model/scripts/capture_lr_golden_snapshot.py` (zapis do `scratch/golden_snapshot.json`).