# Faktenretention unter rekursiver Kompression

Der digitale Masterarbeitsanhang enthält eingefrorene Messdaten und eine offline
reproduzierbare Auswertung von sechs Verfahren über vier Kompressionsstufen.
Die zwölf aktiven Analysen stehen offen in [`notebooks/`](notebooks/README.md);
wiederverwendbare Regeln im installierbaren Paket [`src/faktenretention/`](src/faktenretention/).
`data/` und `prompts/` enthalten eingefrorene Eingaben, `results/` und `figures/`
die historischen Referenzen, `archive/` die früheren Quellen. Neue Ausgaben gehören
in `results/notebooks/` (direktes Run All) oder ein eigenes `results/runs/VERZEICHNIS`.

## Analyseumgebung einrichten

Voraussetzung: Git und [uv](https://docs.astral.sh/uv/). Dokumentierter
Reproduktionsweg: Linux x86_64, alternativ Linux in WSL2; WSL2 wurde nicht separat
getestet. Installation im geklonten Repository:

```bash
uv python install 3.13.15
uv venv --python 3.13.15 .venv-analysis
uv pip sync --python .venv-analysis/bin/python requirements-analysis.lock.txt
uv pip install --python .venv-analysis/bin/python --no-deps -e .
.venv-analysis/bin/python 05_analyze.py --check-environment
```

Der historische Analyse-Lock bleibt unverändert. Die Paketinstallation verwendet
den Repositorypfad; bei einer separaten Wheel-Installation muss
`FAKTENRETENTION_PROJECT` explizit auf das Repository zeigen. Es gibt keine
manuellen `sys.path`-Erweiterungen. Analyse- und Modellumgebung bleiben getrennt.

## Ein Notebook direkt starten

Nach der Installation einmal die Archivdaten vorbereiten:

```bash
.venv-analysis/bin/python 05_analyze.py --prepare
```

Danach ein Notebook unter `notebooks/` öffnen, die Analyseumgebung als Kernel wählen
und **Run All** ausführen. Die erste Codezelle zeigt die Eingaben und die neuen
Ausgabeordner. Standard: `data/prepared/` für Eingaben und historische
Vorgängertabellen, `results/notebooks/tables/` und `results/notebooks/figures/`
für neue Ergebnisse. `DATA_DIR`, `OUTPUT_DIR` und bei Bedarf `TABLES` lassen sich
in der Zelle ändern. Es sind keine Umgebungsvariablen oder Konfigurationsdateien
und kein vorheriger Runner-Lauf erforderlich.

Ein vorhandenes Jupyter-Frontend oder VS Code ist zusätzlich zum Kernel nötig.
[Notebook-Anleitung mit eigenen Verzeichnissen](notebooks/README.md).

## Gesamtlauf und geprüfte Einzelläufe

```bash
.venv-analysis/bin/python 05_analyze.py --output results/runs/komplett
.venv-analysis/bin/python 05_analyze.py --only 11_statistische_laengenkontrolle --output results/runs/laengenkontrolle
.venv-analysis/bin/python 05_analyze.py --list
```

Der Runner berechnet alle benötigten Vorgänger in frischen Kerneln selbst und prüft
sie gegen unveränderte Referenzen. Ein Lauf verlangt ein neues oder leeres
Verzeichnis: Eingaben unter `LAUF/inputs`, neue Tabellen und Grafiken unter
`LAUF/tables` und `LAUF/figures`, ausgeführte Notebooks unter `LAUF/notebooks` und
der Prüfbericht unter `LAUF/validation.json`. Notebook 11 nimmt 03, 04 und 07 mit;
Notebook 06 nimmt 03, 04 und 05 mit. Alte `--only`-Namen bleiben kompatibel.

Die vollständige Prüfung umfasst 289 statistische Tabellen, ein Artefaktinventar,
65 pixelgenau verglichene PNGs sowie die Vollständigkeit von 105 PDF-/SVG-Exporten.
[Referenztabellen](results/originaltabellen.zip) und
[Referenzabbildungen](figures/originalabbildungen.zip) bleiben getrennt und unverändert.
Fünf weitere historische Demonstrationsgrafiken bleiben Referenzartefakte.

Keine Modellaufrufe, GPU-Nutzung oder Gewichtdownloads. BLAS/OpenMP nutzen einen
Thread. Die 20.000 historischen Regression-Bootstrap-Anpassungen werden aus
vorhandenen fingerprintgeprüften Caches gelesen und niemals automatisch neu erzeugt.

## Optionale neue Experimente

Die fünf nummerierten Einstiegsskripte bleiben erhalten. `01_prepare.py`,
`02_demonstrations.py`, `03_generate.py` und `04_evaluate.py` prüfen ohne
Aktivierungsflags ausschließlich den historischen Bestand. Neue Artikel können
ohne Modelle vorbereitet und vorhandene Laufdaten erneut ausgewertet werden:

```bash
.venv-analysis/bin/python 01_prepare.py --articles neue_artikel.csv --output results/mini
.venv-analysis/bin/python 05_analyze.py --run results/mini --output results/runs/mini-auswertung
```

Der zweite Befehl setzt bereits erzeugte `data/summaries.csv` voraus. Mini- und
Vollversuche brauchen ausdrücklich `03_generate.py --test` beziehungsweise
`--full-experiment`, exakte lokale Modellressourcen und ein gemeinsames Anfragebudget.
Anleitung: [`docs/experimente.md`](docs/experimente.md). Der auflösbare
Experiment-Lock unterstützt `--without-model-metrics`; der historische Vollstand hat
einen belegten SummaC/Transformers-Abhängigkeitskonflikt, siehe
[`docs/umgebungen.md`](docs/umgebungen.md).

## Reproduktionsgrenzen und Prüfungen

ROUGE-, BERTScore- und SummaC-Einzelwerte bleiben im historischen Offlineweg
eingefroren; Statistik und QA-Klassifikation werden geprüft. Cache-Fingerprints sind
software- und numerikabhängig. Humanratings und historische Panelmodelle gelten
ausdrücklich nicht für neue Daten. Echte Modellqualität und Modellbeschaffung wurden
bei dieser Umstrukturierung nicht neu geprüft.

```bash
.venv-analysis/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv-analysis/bin/python 01_prepare.py
.venv-analysis/bin/python -m faktenretention.frozen_data --checksums
```

Tests simulieren Modellantworten und erhalten die Netzwerksperren. Die automatische
[Linux-CI](.github/workflows/offline-analysis.yml) kommt ohne Modelle aus; der volle
Notebook-Lauf ist zusätzlich manuell auslösbar.

## Methodik und Nachweise

[Daten und Methodik](data/DATEN_UND_METHODIK.md) ·
[Architektur, Pfade und Vergleichsregeln](docs/architektur.md) ·
[Prüfbericht dieser Umstrukturierung](docs/REFACTOR_PRUEFBERICHT.md) ·
[Modellbezug](data/MODELLBEZUG.md) ·
[historische Berichte](results/ABSCHLUSSBERICHT.md) ·
[Release-Anleitung](RELEASE_CHECKLIST.md).
