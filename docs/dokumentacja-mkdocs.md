# Dokumentacja MkDocs

Indeks dokumentów w `docs/`. Opis systemu: [opis-projektu.md](opis-projektu.md). Start MkDocs: [index.md](index.md).

| Dokument | Treść |
|----------|--------|
| [opis-projektu.md](opis-projektu.md) | Opis repozytorium i architektury |
| [protokol-badawczy.md](protokol-badawczy.md) | Protokół badawczy R0–M3, pipeline uruchomienia |
| [wyniki-kampanii-tezy.md](wyniki-kampanii-tezy.md) | Lokalizacja i skład wyników kampanii |
| [poza-protokolem.md](poza-protokolem.md) | Realtime, testnet, eksploracja poza protokołem |
| [api/](api/index.md) | Reference API z docstringów (`mkdocstrings`) |

README w katalogach kodu (`rl_module/`, `test_module/`, `z_maina_funkcje/`, `shared/`, `tests/`) opisują strukturę lokalną.

```bash
pip install -r requirements-docs.txt
mkdocs serve
mkdocs build   # katalog site/
```
