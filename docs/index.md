# crypto-turtle-rl — dokumentacja

Start dokumentacji MkDocs oraz API z docstringów (`"""`).

| Sekcja | Treść |
|--------|--------|
| [Opis projektu](opis-projektu.md) | Architektura i mapa repozytorium |
| [Protokół badawczy](protokol-badawczy.md) | R0–M3, pipeline, artefakty |
| [Wyniki kampanii](wyniki-kampanii-tezy.md) | Lokalizacja i skład runów |
| [Poza protokołem](poza-protokolem.md) | Realtime, testnet, eksploracja |
| [API](api/index.md) | Reference z kodu (`mkdocstrings`) |

## Budowa lokalna

```bash
pip install -r requirements-docs.txt
mkdocs serve          # podgląd http://127.0.0.1:8000
mkdocs build          # wyjście w katalogu site/
```
