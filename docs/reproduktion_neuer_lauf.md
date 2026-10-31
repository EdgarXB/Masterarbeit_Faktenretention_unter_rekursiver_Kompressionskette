# Neuen Lauf auf einem Rechner einrichten

Dieser Weg erzeugt neue Daten. Die Auswertung der mitgelieferten historischen
Ergebnisse ohne Modellinferenz ist im [README](../README.md) beschrieben.
Die drei Ollama-Manifeste sind eingefroren; neue Modellantworten können trotzdem
von den historischen Antworten abweichen. Die ursprünglichen CNN/DailyMail-Texte
sind kein Bestandteil des Repositories.

## 1. Umgebung und Ressourcen

Voraussetzungen sind Linux, `uv`, Internet für die ausdrücklichen Downloads,
genügend Speicherplatz und Ollama **0.32.6**. Die ursprüngliche Seed-Konfiguration
verwendet ein sehr großes Kontextfenster: Der historische Rechner hatte rund
98 GB GPU-Speicher und 121 GiB RAM. Eine geringere Ausstattung wurde nicht als
Mindestkonfiguration validiert. Downloads sind groß, insbesondere die drei
LLM-Gewichte. [Modelle und Beschaffung](modelle_und_beschaffung.md) dokumentiert
Herkunft, Revisionen und Konfiguration.

```bash
uv python install 3.13.15
uv venv --python 3.13.15 .venv-experiment
uv pip sync --python .venv-experiment/bin/python requirements-experiment.txt
uv pip install --python .venv-experiment/bin/python --no-deps -r requirements-summac.txt
uv pip install --python .venv-experiment/bin/python -r requirements-download.txt
uv pip install --python .venv-experiment/bin/python --no-deps -e .
source .venv-experiment/bin/activate
```

SummaC 0.0.4 deklariert eine ältere Hugging-Face-Hub-Version als die eingefrorene
Transformers-Version. Das obige `--no-deps` umgeht diesen echten Metadatenkonflikt
bewusst; `uv pip check --python .venv-experiment/bin/python` meldet ihn weiterhin.
Die konkrete Installation und Metrikausführung wurden mit Python 3.13.15 auf Linux
getestet, nicht auf allen Plattformen. Falls ein Download Anmeldedaten benötigt,
kann die übliche Hugging-Face-Anmeldung verwendet werden; Zugangsdaten gehören
nicht ins Repository.

```bash
# Beide Schalter sind unabhängig; --ollama lädt mehrere sehr große Gewichte.
python scripts/download_models.py --metrics --ollama
```

Die Metrikmodelle werden anhand fester Revisionen geladen, der SummaC-Head und
alle vier großen Ollama-Dateien anhand von SHA-256 geprüft. Die drei im Repository
mitgelieferten historischen Manifeste werden mit geprüften Blobs in einem eigenen
Modellspeicher rekonstruiert. Ein Fehler oder eine bereits vorhandene abweichende
Datei beendet die Einrichtung; es wird kein anderer Modell-Tag stillschweigend
substituiert. Die Ressourcen liegen standardmäßig im ignorierten Verzeichnis
`results/model-resources/`; mit `--root` kann ein externes Verzeichnis gewählt
werden. `punkt_tab` wird geladen, seine historische Einzelrevision ist nicht
überliefert. Verfügbarkeit der externen Quellen wird nicht garantiert.

Ollama 0.32.6 separat gemäß [Hersteller-Release](https://github.com/ollama/ollama/releases/tag/v0.32.6)
installieren. In einem zweiten Terminal im Repository starten:

```bash
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_MODELS="$PWD/results/model-resources/ollama"
export OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1
ollama serve
```

Bei eigenem `--root` muss `OLLAMA_MODELS` entsprechend auf dessen Unterordner
`ollama` zeigen. Die Pipeline prüft die drei Manifest-Digests vor Modellanfragen.

## 2. Zufällige Artikel aus den Splits auswählen

Die Auswahl lädt den gewünschten Split der festen Datensatzrevision
`96df5e686bee6baa90b8bee7c28b81fa3fa6223d`. Mit identischem Split, Seed,
Ausschlussbestand und Datensatz entsteht dieselbe Auswahl. Die 795–805-Wörter-Regel,
nichtleere Highlights, eindeutige Artikel und die aufsteigende Datensatzposition
entsprechen dem Vorbereitungsprotokoll. Bekannte historische Testartikel und die
Quellartikel der mitgelieferten Demonstrationsbank werden ausgeschlossen.
Zuvor gezogene Stichproben lassen sich mit `--exclude` zusätzlich ausschließen;
die Option ist mehrfach verwendbar. Das neben der CSV erzeugte Manifest enthält
Seed, Revision, Indizes und Hashes, aber keine Originaltexte.

```bash
python scripts/select_articles.py --split test --count 3 --seed 32 \
  --output results/runs/beispiel/test.csv
python 01_prepare.py --articles results/runs/beispiel/test.csv \
  --output results/runs/beispiel/lauf
```

Für eine neue Demonstrationskonstruktion getrennte Trainings- und
Validierungssplits auswählen. Die Konstruktion verlangt mindestens 130 geeignete
Trainingsartikel und genau 30 Validierungsartikel; sie schließt weitere technische
Eignungsbedingungen ein und kann deswegen auch mit dieser Anzahl scheitern.
Beispiel:

```bash
python scripts/select_articles.py --split train --count 240 --seed 32 \
  --output results/runs/beispiel/train.csv
python scripts/select_articles.py --split validation --count 30 --seed 32 \
  --exclude results/runs/beispiel/train.csv --output results/runs/beispiel/validation.csv
python scripts/select_articles.py --split test --count 3 --seed 32 \
  --exclude results/runs/beispiel/train.csv --exclude results/runs/beispiel/validation.csv \
  --output results/runs/beispiel/test_neu.csv
```

`train.csv`, `validation.csv` und `test_neu.csv` werden bei `02_demonstrations.py`
bzw. `01_prepare.py` übergeben. Bereits verwendete Ausgabedateien werden nicht
überschrieben. CSV-Dateien und spätere Laufordner enthalten urheberrechtlich
relevante Originaltexte: außerhalb des Repositories oder im ignorierten
`results/runs/` halten und nicht veröffentlichen. Die Verwendung unterliegt den
Bedingungen der Datensatzquelle und der Originalverlage.

## 3. Generierung und Metriken

Mit den mitgelieferten Demonstrationsprompts alle sechs Methoden und vier Stufen
für die vorbereiteten Artikel ausführen:

```bash
python 03_generate.py --full-experiment --run results/runs/beispiel/lauf \
  --resources results/model-resources/resources.json --max-requests 4000
```

Das Budget schließt Wiederholungen ein. Für andere Stichprobengrößen anpassen;
die Pipeline zeigt vor dem Start ihre Mindestzahl an. Die Modellmetriken laufen
hier standardmäßig mit SummaC auf CPU. Für einen zunächst schnelleren Lauf
`--without-model-metrics` und später den folgenden Befehl verwenden. Der
nachträgliche Schritt benötigt keinen laufenden Ollama-Server:

```bash
python 04_evaluate.py --run results/runs/beispiel/lauf --model-metrics --execute \
  --resources results/model-resources/resources.json --summac-device cuda
```

`--summac-device cpu` ist ebenfalls möglich und der Standard. BERTScore wählt
sein Gerät über die Bibliothek. CPU- und GPU-Werte können sich numerisch leicht
unterscheiden. `metrics_status.json` verzeichnet die durchgeführten Metriken;
`data/metrics_detailed.csv` und `metrics/` enthalten die neu berechneten Werte.
Die QA-Auswertung steht unter `qa_report/`.

Validiert wurde ein echter Drei-Artikel-Lauf mit allen Methoden, vier Stufen,
60 kanonischen Fakten und 1.440 QA-Antworten; anschließend wurden 72 BERTScore-,
72 Quell-SummaC- und 72 Übergangs-SummaC-Werte mit den lokal vorhandenen
historischen Gewichten auf CUDA berechnet. Das ist kein Nachweis für alle Rechner
oder eine bytegleiche Wiederholung historischer Modellantworten.
