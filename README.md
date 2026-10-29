# Faktenretention unter rekursiver Kompression

Dieses Repository ist der digitale Anhang zur Untersuchung, wie gut Fakten bei
der schrittweisen Kürzung von Nachrichtentexten erhalten bleiben. Verglichen
werden **Lead-k, Plain, Zero-Shot-CoT, Few-Shot-CoT, Auto-CoT und Automate-CoT**.
Jeder Text wird rekursiv auf **500, 300, 150 und 75 Wörter** verdichtet:
Die Ausgabe einer Stufe bildet die Eingabe der nächsten.

Das Repository unterstützt zwei Arbeitswege:

- **Vorhandene Ergebnisse nachvollziehen:** Die mitgelieferten Daten auswerten
  und die Tabellen und Abbildungen der Arbeit neu berechnen.
- **Neue Experimente durchführen:** Eigene Artikel vorbereiten, Demonstrationen
  erzeugen, Fakten-/Fragepaare erstellen und Zusammenfassungen bewerten.

## Aufbau

| Bereich | Inhalt |
|---|---|
| [`data/`](data/) | Artikel, Fakten, Zusammenfassungen, QA-Antworten, Modellmetriken und menschliche Bewertungen |
| [`notebooks/`](notebooks/) | Sechs eigenständig ausführbare Auswertungen |
| [`prompts/`](prompts/) | Die im Experiment verwendeten Demonstrationsprompts |
| [`src/faktenretention/`](src/faktenretention/) | Implementierung der Generierung, Bewertung und Statistik |
| `01_prepare.py` bis `05_analyze.py` | Einstiege für die einzelnen Arbeitsphasen |
| [`results/`](results/) | Tabellen, Abbildungen und Ausgabeordner neuer Läufe |
| [`tests/`](tests/) | Prüfungen der Daten, Bewertungsregeln und Pipeline |
| [`docs/`](docs/) | Ergänzende Hinweise zu Methodik, Modellen und Einrichtung |

## Vorhandene Daten auswerten

Die mitgelieferten Daten umfassen **100 Artikel, 2.400 Zusammenfassungen und
2.000 gemeinsame Ausgangsfakten**. Die Faktenretention wird anhand der Antworten
auf dieselben 20 kanonischen Fragen je Artikel bestimmt. Für diese Auswertung
werden keine Sprachmodelle und nach der Installation kein Netzwerkzugriff benötigt.

### Installation und Vorbereitung

Die Auswertung wurde unter Linux mit **Python 3.13.15** geprüft. Unter Windows
kann WSL2 verwendet werden. Die folgenden Befehle werden im Repository ausgeführt
und setzen den Paketmanager [uv](https://docs.astral.sh/uv/getting-started/installation/)
voraus:

```bash
uv python install 3.13.15
uv venv --python 3.13.15 .venv-analysis
uv pip sync --python .venv-analysis/bin/python requirements.txt
uv pip install --python .venv-analysis/bin/python --no-deps -e .
source .venv-analysis/bin/activate
python 05_analyze.py --prepare
python -m ipykernel install --user --name faktenretention --display-name "Faktenretention"
```

Die Vorbereitung stellt die gemeinsamen Eingaben unter `data/prepared/` bereit.
Anschließend ein Notebook beispielsweise in VS Code öffnen, den Kernel
**Faktenretention** auswählen und alle Zellen ausführen. Jedes Notebook kann
unabhängig in einem frischen Kernel gestartet werden.

### Die sechs Notebooks

| Notebook | Inhalt |
|---|---|
| [01 – Menschliche Validierung](notebooks/01_menschliche_validierung.ipynb) | Faktenrelevanz, Fragenqualität, Reliabilität und Kalibrierung der automatischen Bewertung |
| [02 – Faktenretention](notebooks/02_faktenretention.ipynb) | Retentionsverläufe und gepaarte Methodenvergleiche |
| [03 – Textlänge](notebooks/03_textlaenge.ipynb) | Wortfenster, Längenadjustierung, Rangfolgen und Matching |
| [04 – Verlustmuster](notebooks/04_verlustmuster.ipynb) | Faktenverluste über die Stufen und Unterschiede nach Entitätstyp |
| [05 – Weitere Metriken](notebooks/05_weitere_metriken.ipynb) | ROUGE, BERTScore, SummaC und ihr Zusammenhang mit der Retention |
| [06 – Längenregression](notebooks/06_laengenregression.ipynb) | Methodenvergleiche unter statistischer Kontrolle der Textlänge |

Tabellen und Abbildungen werden unter `results/<notebookname>/` gespeichert.
Alle Notebooks lassen sich auch über das Terminal ausführen:

```bash
for notebook in notebooks/*.ipynb; do
  python -m jupyter nbconvert --execute --to notebook --inplace "$notebook"
done
```

Die statistischen Auswertungen werden aus den bereitgestellten Eingaben neu
berechnet. BERTScore und SummaC werden aus gespeicherten Modellbewertungen
ausgewertet. Notebook 06 verwendet standardmäßig gespeicherte
Bootstrap-Modellanpassungen; mit `RECOMPUTE_BOOTSTRAP=True` können diese neu
geschätzt werden. Für den vorhandenen Cache sind die angegebenen Python- und
Paketversionen erforderlich.

## Neue Experimente durchführen

Der Ablauf verbindet fünf Schritte:

**Artikel vorbereiten → Demonstrationen bereitstellen → Fakten-/Fragepaare
erzeugen → rekursive Zusammenfassungen erstellen → Antworten und Metriken auswerten.**

Demonstrationen können übernommen oder mit eigenen Trainings- und
Validierungsdaten neu konstruiert werden. Die Testartikel des Hauptexperiments
müssen von diesen Entwicklungsdaten getrennt bleiben.

### 1. Modelle und Umgebung einrichten

Neue Modellläufe benötigen eine lokale Ollama-Instanz sowie **Seed** für die
Zusammenfassungen, **Gemma** für die Fakten-/Fragepaare und **Qwen** für die
QA-Beantwortung. [Modelle und Beschaffung](docs/modelle_und_beschaffung.md)
beschreibt die Downloads, die Einrichtung und die Ressourcenkonfiguration für
Embeddings, BERTScore und SummaC.

Für die Generierung wird eine separate Umgebung verwendet:

```bash
uv venv --python 3.13.15 .venv-experiment
uv pip sync --python .venv-experiment/bin/python requirements-experiment.txt
uv pip install --python .venv-experiment/bin/python --no-deps -e .
source .venv-experiment/bin/activate
```

**Für SummaC besteht noch ein Abhängigkeitskonflikt.** Neue Demonstrationen und
vollständige Modellmetriken setzen eine entsprechend eingerichtete und geprüfte
SummaC-Umgebung voraus; die Details stehen in der verlinkten Anleitung.
Fakten-/Fragegenerierung, Zusammenfassungen, QA und ROUGE lassen sich mit dem
angegebenen Experiment-Paketstand ohne SummaC ausführen.

### 2. Eigene Artikel vorbereiten

Die Eingabe ist eine UTF-8-CSV mit den Spalten `article` für den vollständigen
Text und `highlights` für eine menschliche Referenzzusammenfassung.
Optional kann `dataset_index` den eindeutigen numerischen Index im
Ursprungsdatensatz angeben; andernfalls wird die Zeilenposition verwendet.

Die Auswahl folgt dem Studienprotokoll: **795–805 Wörter**, nichtleere
Referenzzusammenfassung, eindeutiger Artikeltext und aufsteigender Datensatzindex.
Für andere Textlängen muss das Auswahlprotokoll im Code angepasst werden.

```bash
python 01_prepare.py \
  --articles meine_daten/test.csv \
  --output results/hauptexperiment
```

Die ausgewählten Texte stehen danach in
`results/hauptexperiment/data/articles.csv`. Die Beispielpfade `meine_daten/...`
sind durch eigene Dateien zu ersetzen. Jeder neue Versuch erhält einen eigenen
Ausgabeordner.

### 3. Kandidaten und Demonstrationen erzeugen

Bei Verwendung der mitgelieferten Prompts kann dieser Schritt übersprungen
werden. Für eine neue Konstruktion werden getrennte Trainings- und
Validierungsdateien im oben beschriebenen Format benötigt. Nach der Auswahl
müssen mindestens **130 Trainingsartikel und genau 30 Validierungsartikel**
vorliegen; verwendet werden höchstens 240 Trainingsartikel.

```bash
python 02_demonstrations.py \
  --construct --full-experiment \
  --train meine_daten/train.csv \
  --validation meine_daten/validation.csv \
  --output results/demonstrationen_neu \
  --resources results/model-resources/resources.json \
  --max-requests 100000
```

Dieser Aufruf erzeugt rekursive CoT-Kandidaten, wählt Auto-CoT-Demonstrationen
über semantische Cluster aus, führt SummaC-Pruning und Automate-CoT-Optimierung
durch und erstellt die Few-Shot-CoT-Prompts. Die fertigen Prompts liegen in
`results/demonstrationen_neu/prompts/`. Erreicht der Kandidatenpool die
vorgegebenen Eignungskriterien nicht, wird die Konstruktion als unvollständig
beendet.

### 4. Probelauf und Hauptexperiment starten

Ein kleiner Probelauf prüft die Generierungs- und Bewertungsphasen mit einem
Artikel, dem Verfahren Plain und der ersten Kompressionsstufe:

```bash
python 03_generate.py \
  --test \
  --articles meine_daten/test.csv \
  --run results/probelauf \
  --max-requests 30
```

Der Probelauf erzeugt auch Fakten-/Fragepaare und QA-Antworten. Sein Ordner ist
vom Hauptexperiment getrennt. Mit den zuvor vorbereiteten Testartikeln startet
der folgende Aufruf alle sechs Verfahren und alle vier Stufen:

```bash
python 03_generate.py \
  --full-experiment \
  --run results/hauptexperiment \
  --without-model-metrics \
  --max-requests 60000
```

Zunächst werden je Artikel **24 Fakten-/Fragekandidaten** erzeugt und anhand
der Quellenbelege auf **20 kanonische Fakten-/Fragepaare** reduziert. Diese
Referenz gilt für alle Verfahren und Stufen. Danach entstehen die rekursiven
Zusammenfassungen, die Antworten auf die kanonischen Fragen sowie QA- und
ROUGE-Auswertungen.

Standardmäßig werden die Prompts aus `prompts/` verwendet. Für eigene
Demonstrationen zusätzlich `--bank results/demonstrationen_neu/prompts` angeben.
Für einen Lauf einschließlich BERTScore und SummaC `--without-model-metrics`
weglassen und `--resources results/model-resources/resources.json` ergänzen.

Die Anfragegrenzen sind Obergrenzen einschließlich Wiederholungen und müssen
zur Datensatzgröße passen. Ohne ausdrückliche Ausführungsflags prüfen die
Einstiege `01_prepare.py` bis `04_evaluate.py` lediglich den Bestand
beziehungsweise bewerten gespeicherte Antworten.

### 5. Komponenten getrennt ausführen

Fakten-/Fragegenerierung, Zusammenfassungen und QA können auch einzeln gestartet
werden. Dazu im Repository ein Python-Skript mit folgendem Inhalt ausführen:

```python
from pathlib import Path
from faktenretention.pipeline import (
    prepare_articles, start_run, generate_facts,
    generate_summaries, generate_qa, compute_metrics,
)

run = Path('results/einzelschritte')
if not (run / 'data/articles.csv').exists():
    prepare_articles('meine_daten/test.csv', run)

config = start_run(run, mode='full', max_requests=60000)
generate_facts(run, config)
```

Der letzte Aufruf führt nur die Fakten-/Fragegenerierung durch. Für weitere
Schritte denselben Lauf mit identischen Einstellungen öffnen und den letzten
Aufruf ersetzen:

| Aufruf | Aufgabe |
|---|---|
| `generate_facts(run, config)` | Kanonische Fakten-/Fragepaare aus den Artikeln erzeugen |
| `generate_summaries(run, config)` | Rekursive Zusammenfassungen erstellen |
| `generate_qa(run, config)` | Fragen aus den Zusammenfassungen beantworten und QA-Kennzahlen berechnen; benötigt Fakten und Zusammenfassungen |
| `compute_metrics(run, config)` | ROUGE und Längendiagnostik aus den Zusammenfassungen berechnen |

Das Beispiel verwendet die mitgelieferten Prompts. Für eigene Demonstrationen
bei `start_run` und `generate_summaries` zusätzlich denselben `bank`-Pfad
übergeben. Die Laufaktivierung über `start_run` bleibt erforderlich, da die
Komponenten ein gemeinsames Anfragebudget verwenden.

Vorhandene QA-Antworten eines Laufs können ohne Modellaufrufe neu klassifiziert
werden:

```bash
python 04_evaluate.py --run results/hauptexperiment
```

Mit `--generate-qa --execute` lassen sich QA-Antworten erzeugen oder fortsetzen.
`--model-metrics --execute` berechnet BERTScore und SummaC zusätzlich zu den
weiteren Metriken. Dafür muss die Ressourcenkonfiguration bereits beim Anlegen
des Laufs mit `--resources` hinterlegt worden sein.

### Ergebnisse neuer Läufe weiterverwenden

Unter `results/hauptexperiment/data/` liegen Artikel, kanonische Fakten,
Zusammenfassungen, Antworten und Metriken. `qa_report/` enthält aggregierte
QA-Kennzahlen; `evaluation/` enthält die mit `04_evaluate.py` erzeugten Labels.
Die Laufprotokolle dokumentieren Einstellungen und verbrauchte Modellanfragen.

**Die sechs Notebooks sind auf den mitgelieferten Studiendatensatz zugeschnitten.**
Neue Läufe werden nicht automatisch übernommen. Für ihre statistische Auswertung
müssen Datenzugriff, Prüfungen und gegebenenfalls das Analysedesign angepasst werden.

## Wissenschaftliche Einordnung und Prüfungen

Die Retention wird auf Artikelebene ausgewertet; Unsicherheit wird durch gepaarte
Artikel-Bootstraps beschrieben. Technisch fehlende QA-Antworten werden von
beobachteten Faktenverlusten getrennt. Die Analysen berücksichtigen außerdem
die tatsächliche Textlänge und die Abhängigkeit der rekursiven Stufen.

[Auswertungsgrundlagen und Hinweise](docs/auswertung_und_einordnung.md) erläutert
die Kennzahlen, den Datenstand und die Grenzen der Interpretation.
[Modelle und Beschaffung](docs/modelle_und_beschaffung.md) enthält die technischen
Einrichtungsschritte und Herkunftsbelege.

Die lokalen Prüfungen lassen sich wie folgt ausführen:

```bash
python -m unittest discover -s tests -v
python 01_prepare.py
python 02_demonstrations.py
```
