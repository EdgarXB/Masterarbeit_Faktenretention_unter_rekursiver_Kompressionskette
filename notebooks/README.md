# Ein Notebook öffnen und ausführen

Nach Installation der Analyseumgebung und des Pakets (siehe [README](../README.md))
einmal im Repository ausführen:

```bash
.venv-analysis/bin/python 05_analyze.py --prepare
```

Danach eines der zwölf nummerierten Notebooks in VS Code oder einem vorhandenen
Jupyter-Frontend öffnen, `.venv-analysis/bin/python` als Kernel wählen und **Run All**
ausführen. Es sind keine Umgebungsvariablen, keine `workspace.json` und keine vorher
manuell ausgeführten Notebooks erforderlich. Der Lock enthält den Kernel und die
Ausführungsbibliotheken; ein Notebook-Frontend muss zusätzlich vorhanden sein.

Die erste Codezelle zeigt die wesentlichen Pfade und lässt sie direkt ändern:

```python
DATA_DIR = PROJECT_ROOT / "data/prepared"
OUTPUT_DIR = PROJECT_ROOT / "results/notebooks"
TABLES = DATA_DIR / "tables"
```

- `DATA_DIR/qa`, `metrics`, `human_qa`, `llm` und `human_faktenrelevanz.csv`:
  unveränderte Messdaten und Urteile, aus den Archiven vorbereitet.
- `DATA_DIR/sources`: historische Definitionen als Herkunftsbelege; sie werden
  gelesen und geprüft, niemals als Analysecode ausgeführt.
- `DATA_DIR/tables/ANALYSE`: vorbereitete historische Vorgängertabellen. Dadurch
  kann auch Notebook 11 direkt starten, ohne zuerst 03, 04 und 07 auszuführen.
- `DATA_DIR/caches`: ausdrücklich wiederverwendete historische Eingaben,
  insbesondere die fingerprintgeprüften Regressions-Bootstrap-Caches.
- `OUTPUT_DIR/tables/ANALYSE` und `OUTPUT_DIR/figures/ANALYSE`: neue Tabellen,
  Texte und Grafiken. Sie überschreiben keine Eingaben oder Referenzarchive.

Notebook 01 untersucht Faktenrelevanz, 02 Human-QA, 03 QA-Retention, 04 Länge und
Adjustierung, 05 Verlustmuster, 06 Sekundärmetriken, 07 längengematchte Matrizen,
08 Texttabellen, 09 Lead-k, 10 QA-Sensitivität, 11 statistische Längenkontrolle und
12 vorhandene LLM-Scorerurteile. Notebook 12 erzeugt keine neuen Modellurteile.

## Eigene Verzeichnisse und neu berechnete Vorgänger

Für einen anderen Datenordner:

```bash
.venv-analysis/bin/python 05_analyze.py --prepare --data-dir /absoluter/pfad/eingaben
```

`DATA_DIR` in der ersten Zelle auf diesen Pfad setzen; `OUTPUT_DIR` darf unabhängig
davon einen anderen absoluten Pfad verwenden. Der vorbereitete Ordner behält die
oben beschriebene Struktur. Die Vorbereitung verlangt einen neuen oder leeren
Ordner und startet keine Analyse. Für direktes erneutes Run All kann der vorhandene
Ausgabeordner weiterverwendet werden.

Wer Vorgängertabellen selbst neu berechnen möchte, führt die betreffenden Notebooks
mit demselben `OUTPUT_DIR` aus und setzt `TABLES = OUTPUT_DIR / "tables"`. Die
[Abhängigkeiten](analyses.json) geben die nötigen Vorgänger an. Für Notebook 11 sind
das 03, 04 und 07. Der folgende automatisierte Weg erledigt das vollständig.

## Automatische Reproduktion und Referenzprüfung

```bash
.venv-analysis/bin/python 05_analyze.py --output results/runs/historisch
.venv-analysis/bin/python 05_analyze.py --only 11_statistische_laengenkontrolle --output results/runs/laengenkontrolle
.venv-analysis/bin/python 05_analyze.py --only 06_weitere_metriken --output results/runs/metriken
.venv-analysis/bin/python 05_analyze.py --list
```

Jeder Lauf verlangt ein neues oder leeres Verzeichnis. Er berechnet seine Vorgänger
selbst und verwendet dafür **keine** vorbereiteten historischen Vorgängertabellen.
Jedes Notebook startet mit einem frischen Kernel. Eingaben stehen unter `LAUF/inputs`,
neue Ergebnisse unter `LAUF/tables` und `LAUF/figures`, ausgeführte Notebookkopien
unter `LAUF/notebooks`. `LAUF/validation.json` weist Vollständigkeit, Vergleiche,
Fehler und wiederverwendete Eingaben aus. Referenzen werden direkt aus den
unveränderten ZIPs gelesen und müssen nicht nochmals entpackt werden.

Das statische [Inventar](analyses.json) nennt jede erwartete Ausgabe unabhängig
vom tatsächlichen Lauf. Geprüft werden 289 statistische Tabellen, ein Artefaktinventar,
65 PNGs pixelgenau und 105 zusätzliche PDF-/SVG-Exporte auf Vollständigkeit.
Numerische Toleranzen bleiben `rtol=0, atol=1e-12`.

Die Regression-Bootstraps werden ausschließlich aus passenden vorhandenen Caches
verwendet. Fehlende oder unpassende Caches ergeben eine konkrete Fehlermeldung;
eine vollständige Neuberechnung ist aus dem aktiven Notebook entfernt.

## Neue Versuchsdaten

Das getrennte [Ergänzungsnotebook](ergaenzungen/neue_laufdaten.ipynb) verwendet die
gemeinsamen Regeln für bereits erzeugte neue Laufdaten:

```bash
.venv-analysis/bin/python 05_analyze.py --run results/mini --output results/runs/mini-auswertung
```

Historische Humanurteile, Versionssensitivitäten und 100-Artikel-Panelmodelle werden
neuen Daten nicht zugeordnet. Bei Summary-Läufen ohne QA entstehen nur verfügbare
Kennzahlen. Die Modellphasen bleiben ausdrücklich getrennt und aktiviert.
