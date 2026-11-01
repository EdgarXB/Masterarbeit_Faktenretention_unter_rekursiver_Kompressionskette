# Faktenretention unter rekursiver Kompression

Dieses Repository ist der digitale Anhang zur Untersuchung, wie gut Fakten bei
der schrittweisen Kürzung von Nachrichtentexten erhalten bleiben. Verglichen
werden **Lead-k, Plain, Zero-Shot-CoT, Few-Shot-CoT, Auto-CoT und Automate-CoT**.
Jeder Text wird rekursiv auf **500, 300, 150 und 75 Wörter** verdichtet:
Die Ausgabe einer Stufe bildet dabei die direkte Eingabe der nächsten.

**Einstieg:** Diese README beschreibt Einrichtung und Grenzen. Die [Notebooks](notebooks/)
enthalten die Auswertung; das [Integritätsprotokoll](docs/pruefung_wissenschaftliche_datenintegritaet.md)
fasst Datenstand und Prüfungen zusammen.


Das Repository unterstützt zwei Arbeitswege:

- **Vorhandene Ergebnisse nachvollziehen:** Die mitgelieferten Daten auswerten
  und die Tabellen und Abbildungen der Arbeit neu berechnen.
- **Neue Experimente durchführen:** Eigene Artikel vorbereiten, Demonstrationen
  erzeugen, Fakten-/Fragepaare erstellen und Zusammenfassungen bewerten.

## Aufbau

| Bereich | Inhalt |
|---|---|
| [`data/`](data/) | Artikelmetadaten, Fakten, Zusammenfassungen, QA-Antworten, Modellmetriken und menschliche Bewertungen; Originalartikel und Highlights sind in den Datendateien entfernt |
| [`notebooks/`](notebooks/) | Sechs eigenständig ausführbare Auswertungen |
| [`prompts/`](prompts/) | Die im Experiment verwendeten Demonstrationsprompts |
| [`src/faktenretention/`](src/faktenretention/) | Implementierung der Generierung, Bewertung und Statistik |
| `01_prepare.py` bis `05_analyze.py` | Einstiege für die einzelnen Arbeitsphasen |
| [`results/`](results/) | Tabellen, Abbildungen und Ausgabeordner neuer Läufe |
| [`tests/`](tests/) | Prüfungen der Daten, Bewertungsregeln und Pipeline |
| [`docs/`](docs/) | Ergänzende Hinweise zu Methodik, Modellen und Einrichtung |
| [`data/human_qa/`](data/human_qa/) | Pseudonymisierte QA-Bewertungen, Bewertungsoberfläche und Rückgaben unter `returns/` |
| [`data/methoden/`](data/methoden/) | Studienprotokoll und Zusammenführungsbelege |
| [`data/model_metadata/`](data/model_metadata/) | Eingefrorene Manifeste, kleine Metadaten-Blobs und Herkunftsbelege unter `provenance/` |
| `data/prepared/` | Lokal entpackte Analyseeingaben; wird erzeugt und nicht versioniert |

## Voraussetzungen

Für die Offline-Auswertung: **Python 3.13.15**, [uv](https://docs.astral.sh/uv/getting-started/installation/)
und die festgelegten Pakete aus `requirements.txt`. Die Projektmetadaten erlauben
Python **ab 3.13**; geprüft ist der Stand **3.13.15**, wie in der CI.
Andere Python-Versionen ab 3.13 sind damit nicht automatisch validiert,
insbesondere für den Bootstrap-Cache.

Nur für neue Modellläufe werden zusätzlich **Ollama 0.32.6**, die
Experimentumgebung und die drei Modelle aus [`data/models.json`](data/models.json)
benötigt. Dort stehen die vollständigen Manifest-Digests für `thesis-seed:v1`,
`thesis-gemma:v1` und `thesis-qwen38-qa:v1`. Die Originalaufzeichnung beschreibt
**ca. 98 GB VRAM und 121 GiB RAM** als beobachtete Ausstattung, ausdrücklich
nicht als ermittelte Mindestanforderung. Der unten gezeigte Hauptlauf setzt ein
Budget von **bis zu 60.000 Requests einschließlich Wiederholungen**; eine neue
Demonstrationskonstruktion hat ein eigenes Budget. Herkunft und Einrichtung:
[Modelldokumentation](docs/modelle_und_beschaffung.md).

## Quickstart: Offline-Auswertung

Nach der unten beschriebenen Installation, mit aktivierter `.venv-analysis`,
genügen diese drei Befehle im Repository:

```bash
python 05_analyze.py --prepare
python -m unittest discover -s tests -v
mkdir -p results/notebooks/ausgefuehrt
(for notebook in notebooks/*.ipynb; do python -m jupyter nbconvert --execute --to notebook --output-dir results/notebooks/ausgefuehrt --ExecutePreprocessor.timeout=900 "$notebook" || exit 1; done)
```

Die Notebooks schreiben Tabellen und Grafiken nach `results/`. Zwei
Diagnose-CSVs in Notebook 06 enthalten zusätzlich Laufzeiten, die je Ausführung
abweichen können; die wissenschaftlichen Tabellen bleiben beim dokumentierten
Paketstand innerhalb der dokumentierten Toleranzen überein (siehe [Integritätsprotokoll](docs/pruefung_wissenschaftliche_datenintegritaet.md)).

## Vorhandene Daten auswerten

Die mitgelieferten Daten umfassen **100 Artikel, 2.400 Zusammenfassungen und
2.000 gemeinsame Ausgangsfakten**. Die Faktenretention wird anhand der Antworten
auf dieselben 20 kanonischen Fragen je Artikel bestimmt. Für diese Auswertung
werden keine Sprachmodelle und nach der Installation kein Netzwerkzugriff benötigt.

### Installation und Vorbereitung

Die Auswertung wurde unter Linux mit **Python 3.13.15** geprüft. Unter Windows
kann WSL2 verwendet werden. Die folgenden Befehle werden im Repository ausgeführt:

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
Der Quickstart oben führt alle Notebooks im Terminal aus und speichert ihre Kopien unter `results/notebooks/`.

Die statistischen Auswertungen werden aus den bereitgestellten Eingaben neu
berechnet. BERTScore und SummaC werden aus gespeicherten Modellbewertungen
ausgewertet. Notebook 06 verwendet standardmäßig gespeicherte
Bootstrap-Modellanpassungen; mit `RECOMPUTE_BOOTSTRAP=True` können diese neu
geschätzt werden. Für den vorhandenen Cache sind die angegebenen Python- und
Paketversionen erforderlich.

## Neues Experiment durchführen

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


Für eine geführte Einrichtung mit explizitem Download der eingefrorenen Modelle,
zufälliger Auswahl aus Train/Validierung/Test und nachträglicher Berechnung
von BERTScore und SummaC siehe [Neuen Lauf einrichten](docs/reproduktion_neuer_lauf.md).

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
vorliegen. Die Trainingsartikel unterliegen keiner festen Obergrenze; die
Generierung stoppt, sobald die Eignungsschwellen erreicht sind.

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

### 5. Gespeicherte Antworten und Modellmetriken

```bash
python 04_evaluate.py --run results/hauptexperiment
python 04_evaluate.py --run results/hauptexperiment --model-metrics --execute \
  --resources results/model-resources/resources.json --summac-device cuda
```

Der erste Befehl klassifiziert gespeicherte QA-Antworten, der zweite berechnet
BERTScore und SummaC. Modellmetriken benötigen keinen laufenden Ollama-Server.

### Ergebnisse neuer Läufe weiterverwenden

Unter `results/hauptexperiment/data/` liegen Artikel, kanonische Fakten,
Zusammenfassungen, Antworten und Metriken. `qa_report/` enthält aggregierte
QA-Kennzahlen; `evaluation/` enthält die mit `04_evaluate.py` erzeugten Labels.
Die Laufprotokolle dokumentieren Einstellungen und verbrauchte Modellanfragen.

**Die sechs Notebooks sind auf den mitgelieferten Studiendatensatz zugeschnitten.**
Neue Läufe werden nicht automatisch übernommen.

## Wissenschaftliche Einordnung und Prüfungen

Die Retention wird auf Artikelebene ausgewertet; Unsicherheit wird durch gepaarte
Artikel-Bootstraps beschrieben. Technisch fehlende QA-Antworten werden von
beobachteten Faktenverlusten getrennt. Die Analysen berücksichtigen außerdem
die tatsächliche Textlänge und die Abhängigkeit der rekursiven Stufen.

Das [Integritätsprotokoll](docs/pruefung_wissenschaftliche_datenintegritaet.md) beschreibt
den Datenstand, die Kennzahlen und die Grenzen der Prüfung.
[Modelle und Beschaffung](docs/modelle_und_beschaffung.md) enthält die technischen
Einrichtungsschritte und Herkunftsbelege.

Die lokalen Prüfungen lassen sich wie folgt ausführen:

```bash
python -m unittest discover -s tests -v
python 01_prepare.py
python 02_demonstrations.py
```

Die [CI `offline-analysis.yml`](.github/workflows/offline-analysis.yml) installiert
die Analyseumgebung mit Python 3.13.15, führt die Unittests aus, bereitet die
Eingaben vor, prüft den Bestand über `01_prepare.py` und `02_demonstrations.py`,
führt alle sechs Notebooks aus und vergleicht ihre Ergebnistabellen mit den
eingefrorenen Referenzen. Die Paketinstallation benötigt Netzwerk;
die Auswertung verwendet gespeicherte Daten. Die CI führt **keine Modellpipeline**
aus und prüft keine neue Ollama-, BERTScore-, SummaC- oder Embedding-Inferenz.

## Datenherkunft und Lizenz

Der eigene Programmcode steht unter der [MIT-Lizenz](LICENSE). Diese Lizenz
erteilt keine zusätzlichen Rechte an fremden Artikeln, Modellgewichten oder
menschlichen Bewertungen.

| Bestandteil | Herkunft und belegte Lizenzangabe |
|---|---|
| Artikel und Highlights | CNN/DailyMail **3.0.0**, siehe Kapitel 3.2. Originaltexte und Highlights wurden aus den Datentabellen entfernt; Originaltext verbleibt in den unveränderten 500-Wörter-Prompts, in den Bewertungsmasken `manuskript_image20.png` und `manuskript_image21.png` unter `data/methodenmaterial.zip`, in Lead-k-Auszügen und in der Git-Historie. |
| Seed-OSS | Die [offizielle Modellkarte](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct) nennt Apache-2.0. Die konkret verwendete GGUF-Variante und Digests sind in der [Modelldokumentation](docs/modelle_und_beschaffung.md) zugeordnet. |
| Gemma 4 | Die [offizielle Modellkarte](https://huggingface.co/google/gemma-4-31B-it) nennt Apache-2.0. Der historische Ollama-Stand enthält zusätzlich einen [Apache-2.0-Lizenztext](data/model_metadata/blobs/sha256-7339fa418c9ad3e8e12e74ad0fd26a9cc4be8703f9c110728a992b193be85cb2), zugeordnet im eingefrorenen Gemma-Manifest. |
| Qwen3.8 | Die [offizielle Modellkarte](https://huggingface.co/Qwen/Qwen3.8-27B) nennt Apache-2.0; der verwendete GGUF-Stand ist separat über Revision und Hash belegt. |
| Menschliche Bewertungen | Pseudonymisierte Daten unter `data/human_qa/` sowie in `data/menschliche_faktenrelevanz.csv` und den Quellenarchiven. |

## KI-gestützte Erstellung und Zitieren

Gemäß Eidesstattlicher Versicherung und Anhang G des vorliegenden
PDF-Manuskripts erfolgte die technische Implementierung des Programmcodes
nahezu vollständig mit **ChatGPT Codex**. Wissenschaftliche Konzeption,
experimentelle Anforderungen sowie Prüfung und Interpretation der Ergebnisse
lagen beim Verfasser.

[`CITATION.cff`](CITATION.cff) enthält die Zitierangaben der Masterarbeit von
Edgar Berngardt, Hochschule Düsseldorf, 2026. Für die verwendete Codefassung
zusätzlich Commit-Hash oder Release angeben.