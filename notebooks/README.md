# Auswertung mit frischen Kerneln

Die zwölf nummerierten Notebooks sind die aktive historische Analyse. Das
[Inventar](analyses.json) enthält Reihenfolge, Abhängigkeiten, erwartete Ausgaben,
Referenzzuordnung und ausdrücklich wiederverwendete Eingaben. Die wissenschaftlichen
Erklärungen stammen aus den unveränderten Originalen in
[`data/referenznotebooks.zip`](../data/referenznotebooks.zip).

Nach Einrichtung und Paketinstallation im Projektverzeichnis:

```bash
.venv-analysis/bin/python 05_analyze.py --list
.venv-analysis/bin/python 05_analyze.py --output results/runs/historisch
.venv-analysis/bin/python 05_analyze.py --only 05_verlustmuster --output results/runs/verlust
```

Jeder Befehl braucht ein neues oder leeres Ausgabeverzeichnis. Einzellauf und
Gesamtlauf erzeugen eigene Berichte. Der Einzellauf führt notwendige Vorgänger
ebenfalls aus. Die früheren zwölf `--only`-Bezeichnungen werden weiterhin erkannt.
Die ersten zwei Analysen betreffen Human Validation; danach folgen QA-Retention,
Länge/Adjustierung, Verlustmuster, Sekundärmetriken, längengematchte Matrizen,
Tabellenexport sowie vier ergänzende historische Kontrollen.

Der Runner startet für jedes Notebook einen neuen Kernel mit dem aufrufenden
Python-Interpreter. Exporte und eingebettete PNG-Ausgaben bleiben in den ausgeführten
Kopien unter `LAUF/notebooks/` erhalten. Quellen und wiederverwendete Caches liegen
unter `LAUF/workspace/`, separate Referenzen unter `LAUF/reference/`, der Prüfbericht
unter `LAUF/validation.json`. Details zu den jeweiligen Ergebnis- und Grafikordnern
stehen im Notebook und im Inventar. Die Referenz-ZIPs werden niemals überschrieben.

Für interaktive Arbeit zuerst die erforderlichen Vorgänger mit dem Runner ausführen
und Jupyter aus derselben Analyseumgebung mit explizitem Workspace starten:

```bash
.venv-analysis/bin/python 05_analyze.py --only 04_laenge_und_adjustierte_retention --output results/runs/interaktiv
export FAKTENRETENTION_WORK="$PWD/results/runs/interaktiv/workspace"
```

Das Notebook dann in einem vorhandenen Frontend mit `.venv-analysis/bin/python`
als Kernel öffnen. Ein Jupyter-Frontend ist zusätzlich erforderlich; der historische Lock enthält die
Ausführungsbibliotheken, aber keinen Notebook-Webserver. Ein vorhandenes JupyterLab
oder VS Code kann denselben Python-Interpreter verwenden. Alternativ nur Eingaben
bereitstellen: `05_analyze.py --prepare --output results/runs/vorbereitung`. In diesem
Fall müssen die im Inventar genannten Vorgänger im selben Workspace zuerst laufen.
Ein zufälliges Arbeitsverzeichnis oder ein vorheriger Kernelzustand genügt nicht.

Die Regression-Bootstraps werden ausschließlich aus den vorhandenen,
fingerprintgeprüften Caches verwendet. Eine fehlende oder unpassende Cachedatei führt
zum Fehler. Der aktive Runner bietet keine versehentliche Neuberechnung an.

Neue Daten verwenden das eigene
[`ergaenzungen/neue_laufdaten.ipynb`](ergaenzungen/neue_laufdaten.ipynb):

```bash
.venv-analysis/bin/python 05_analyze.py --run results/mini --output results/runs/mini-auswertung
```

Es erzeugt Tabellen mit den gemeinsamen Regeln und eigene Grafiken; historische
Humanurteile, Versionssensitivitäten und 100-Artikel-Panelmodelle werden neuen Daten
nicht zugeordnet. Bei reinem Summary-Lauf ohne QA entstehen nur verfügbare Kennzahlen.
