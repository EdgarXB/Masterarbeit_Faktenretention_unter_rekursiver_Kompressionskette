# Digitaler Anhang: Faktenretention unter rekursiver Kompression

Grundlage ist `2610_02_Masterarbeit.docx`, SHA-256
`b57134a04ee30f95042b33ffa9f2cebca4ba6d6bc0ecefb76cda02a94c3e5a54`.
Die Arbeit selbst wird nicht verändert oder mitgeliefert.

## Gespeicherte Daten auswerten

Getestet unter Linux mit **Python 3.13.15**. Unter Windows eignet sich WSL2.
Die exakten Paketversionen sind für den vorhandenen Regressionscache erforderlich.
Im Repository ausführen:

```bash
uv python install 3.13.15
uv venv --python 3.13.15 .venv-analysis
uv pip sync --python .venv-analysis/bin/python requirements.txt
uv pip install --python .venv-analysis/bin/python --no-deps -e .
source .venv-analysis/bin/activate
python 05_analyze.py --prepare
python -m ipykernel install --user --name faktenretention --display-name "Faktenretention (3.13.15)"
```

Danach ein Notebook in VS Code öffnen, den Kernel **Faktenretention (3.13.15)**
auswählen und **Run All** ausführen. Jedes Notebook läuft unabhängig mit frischem
Kernel. Die Vorbereitung prüft die Quellen und entpackt sechs gemeinsame Rohdateien
nach `data/prepared/`; sie berechnet keine Ergebnisse und liest keine alten Endtabellen.
Die Notebooks benötigen weder Modellgewichte noch Netzwerkzugriff.

| Notebook | Inhalt | Manuskript |
|---|---|---|
| `01_menschliche_validierung` | Relevanz, Fragenqualität, Reliabilität, Scorer-Kalibrierung | Abb. 5–7, 19/20; Tab. 5/6 |
| `02_faktenretention` | QA-Rekonstruktion, Artikelmittel, Kontraste, minimale Labeländerungen | Abb. 8–10; Tab. 15/16/20 |
| `03_textlaenge` | Wortfenster, Adjustierung, Ränge, Matching und vollständige Matching-Intervalle | Abb. 11–14, 21/22; Tab. 17 |
| `04_verlustmuster` | Faktenlaufbahnen, Entitätstypen, Lead-k-Präfixe und Rückwechsel | Abb. 15/16/23; Tab. 7, 21–25 |
| `05_weitere_metriken` | ROUGE/BERTScore/SummaC, artikelweises Kendall-τ-b | Tab. 8; Ergebnistabelle E.1 |
| `06_laengenregression` | Lineares binomiales GLM, Unterstützung, Standardisierung, Bootstrap | Tab. 18/19; Auto-CoT–Plain bei 75 |

CSV-Tabellen und genau **17 PNG-Ergebnisabbildungen** entstehen unter
`results/<notebookname>/`. Die Notebooks enthalten zusätzlich ihre sichtbaren
Ausgaben. Notebook 05 und 06 erzeugen keine Grafiken. Notebook 06 liest standardmäßig
den unveränderten linearen Cache. Sein Fingerprint bindet ihn an Daten,
Designmatrix, Parameter, Ziehungen, numerischen Worker und Paketversionen.
Die dortige Option `RECOMPUTE_BOOTSTRAP=True` berechnet ausdrücklich alle 10.000
Anpassungen neu und schreibt den neuen Cache nach `results/`.

## Quellen und Definitionen

`data/quellen.json` ordnet Quellen, SHA-256, Datenstand und Parameterbelege zu.
Die fünf Hauptdateien enthalten Artikel, kanonische Fakten, Zusammenfassungen,
QA-Antworten und gespeicherte Modellmetriken. `generierungszustaende.jsonl.gz`
enthält die tatsächlichen rekursiven Eingaben und Generierungszustände.
Verwendet wird ausschließlich `aktuell_20260930`: die ursprünglichen vier Verfahren
plus die später ersetzten Plain-/Few-Shot-CoT-Ketten vom 30. September, abgeschlossen
am 1. Oktober. Alte und neue Fassungen werden nicht gepoolt.

- 100 Artikel × 6 Verfahren × 4 Stufen = 2.400 Ausgaben; 2.000 feste Ausgangsfakten.
- 48.000 QA-Positionen = 47.804 gültige Urteile + 196 technische Fehlwerte.
  Die 40 regelbasierten NOT_FOUND-Urteile zweier leerer Ausgaben sind gültige Verlusturteile.
- Beobachtete Retention ist K/V, fester Nenner K/20. Adjustierung:
  `(K/20)/(1+D)` mit `D=(Unter-+Überschreitung des Wortfensters)/Zielwortzahl`.
- Artikel sind die Einheit gepaarter Vergleiche und der 10.000 Bootstrap-Ziehungen,
  Seed 32. Die 95-%-Perzentilintervalle sind nicht für Mehrfachvergleiche korrigiert.
- Regression: mindestens 20 Beobachtungen je Verfahren im gemeinsamen 5.–95.-Perzentilbereich
  und 20 gepaarte, lokal gestützte Artikel. Lokal mindestens fünf Ausgaben **jedes**
  Verfahrens im Band `max(5 Wörter, 10 % der betrachteten Wortzahl)`.

Weitere notwendige Quellen:

- `human_qa/` und `menschliche_relevanz_quellen.zip`: pseudonymisierte Rückgaben,
  Originalfälle, Auswahlregel und Bewertungsunterlagen. Kalibrierung auf denselben
  Fällen ist keine unabhängige Validierung; die finalen 24-Kandidaten-Fakten wurden
  nicht nochmals menschlich validiert. Entitätstypen sind automatisch zugewiesen.
- `demonstrationen.zip`: tatsächliche Auswahlartikel (240 Training, 30 Validierung,
  100 Test; zusätzlich der ursprüngliche 200er-Trainingsstand als Herkunftsbeleg),
  Kandidaten, Auswahl-/Pruningurteile und 5.753 verwendete Optimierungscaches.
  `02_demonstrations.py` reproduziert daraus 600 Update-Verluste und vier finale
  Validierungsverluste, ohne neue Generierungen oder Optimierung.
- `modellpilot.zip`: fünf Artikel, Originalantworten, Versuche und Parameter des
  Neunmodelle-Vorexperiments zu Tabelle 14. Dies ist eine andere Untersuchung als
  der entfernte zusätzliche LLM-Scorervergleich.
- `methodenmaterial.zip`: ursprüngliche Methodenbilder 1–4, Bewertungsmasken 17/18
  und unveränderte Manuskripttabellen 3/4 und 9–14. Keine Analysedatenquelle.
- `methoden/`: ursprüngliche Parameter und Belege der späteren Ersetzung;
  `prompts/`: zwölf tatsächlich eingesetzte Demonstrationsprompts und deren Zuordnung.

## Neue Modellläufe ausdrücklich starten

Die Einstiege `01_prepare.py` bis `04_evaluate.py` erhalten die tatsächlichen
Auswahl-, Demonstrations-, Generierungs- und Bewertungsphasen. Ohne Aktivierungsflags
prüfen sie gespeicherte Daten beziehungsweise wenden den lokalen Scorer an.
`01_prepare.py --articles QUELLE.csv --output results/neulauf` wählt nach aufsteigendem
Datensatzindex, 795–805 Regex-Wörtern, nichtleeren Highlights und eindeutigem Text-Hash.
Die gespeicherten Auswahl-CSV enthalten Originalindizes und Texte; ein erneuter
Datensatzdownload ist für ihre Verwendung nicht erforderlich.

Der aufgelöste Experimentstand **ohne SummaC** steht in `requirements-experiment.txt`:

```bash
uv venv --python 3.13.15 .venv-experiment
uv pip sync --python .venv-experiment/bin/python requirements-experiment.txt
uv pip install --python .venv-experiment/bin/python --no-deps -e .
.venv-experiment/bin/python 03_generate.py --test --articles data/prepared/articles.csv --run results/neulauf --max-requests 30
```

Der Mini-Lauf nutzt einen Artikel, Plain, 500 Wörter und höchstens 30 Anfragen inklusive
Retries. Er benötigt eine laufende lokale Ollama-Instanz und die exakten drei Modelle.
Vollausführung ist ausdrücklich über `--full-experiment --without-model-metrics`
und `--max-requests` aktivierbar. Konstruktion neuer Demonstrationen erfordert
`02_demonstrations.py --construct --full-experiment` und die mit `--help` angezeigten
Trainings-/Validierungsdaten, Ressourcen und Anfragegrenze. QA oder Modellmetriken
lassen sich mit `04_evaluate.py --run LAUF --generate-qa --execute` beziehungsweise
`--model-metrics --execute` explizit fortsetzen. Es gibt keine automatische Neuauswertung
neuer Versuchsdaten durch die sechs auf das Manuskript zugeschnittenen Notebooks.

**Konkrete Grenzen für neue Vollreproduktionen:** Der beobachtete historische
Paketstand ist in `data/quellen.json` erhalten. `summac==0.0.4` fordert
`huggingface-hub<=0.17.0`, der gleichzeitig verwendete `transformers==5.15.1`
fordert `huggingface-hub>=1.5.0,<2.0`. Daher wird kein funktionierender gemeinsamer
Vollstand behauptet; neue SummaC-Bewertungen und Demonstrationskonstruktionen benötigen
zunächst eine geprüfte kompatible Modellumgebung. Die fachlichen Phasen bleiben erhalten.
`data/resources.example.json` zeigt die benötigten lokalen Ressourcen; die erwarteten
Revisionen stehen in `src/faktenretention/metric_resources.py`. Kein impliziter Download.

Modellnamen/Digests und Beschaffungsbelege stehen in `data/models.json`,
`data/MODEL_INVENTORY.json` und `data/model_metadata/` (Templates, Parameter, Lizenzen,
Manifeste; keine Gewichte). Seed-Gewicht: SHA-256
`2d9f3c32a61f2668e012708be7ec04e753e35e366f87b9e4bce89f4d073cfc16`.
Für Gemma (`280af6832eca23cb322c4dcc65edfea98a21b8f8ab07dc7553bd6f7e6e7a3313`)
und QA (`31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`)
ist der exakte öffentliche Bezug nicht lückenlos nachgewiesen; Originalblobs sind nötig.
Der QA-Projektor hat SHA-256 `2e968a6af97ce35d8971890b257b9b7edabf20ad91450501fa53162a19ee33eb`.
Die Metadaten nennen Ollama 0.20.0 als Voraussetzung. Die Pipeline prüft Modell-Digests
vor der ersten Generierung. Modellpakete/GPU und echte Inferenz sind mit diesem
Umbau nicht neu getestet; die Phasenverbindungen werden mit simulierten Antworten geprüft.

## Prüfungen und offene Manuskriptstellen

```bash
python -m unittest discover -s tests -v
python 01_prepare.py
python 02_demonstrations.py
for notebook in notebooks/*.ipynb; do
  python -m jupyter nbconvert --execute --to notebook --inplace "$notebook"
done
```

Die Quellen bleiben unverändert. Tests prüfen positionsgenaue QA-Labels, Nenner,
Artikelbootstrap-Referenzen und die Experimentphasen ohne echte Modellaufrufe.
Historisch fehlen vollständige Cluster-Embeddings und alle QA-Retry-Zwischenantworten;
diese werden nicht nachträglich erfunden. Gespeicherte BERTScore-/SummaC-Werte werden
statistisch ausgewertet; ihre ursprüngliche Inferenz wird dadurch nicht neu validiert.

Vom Autor zu prüfen: E.1 wird angekündigt, enthält im maßgeblichen Manuskript aber
keine Tabelle; der CSV-Export steht in Notebook 05. In Tabelle 18 lautet die untere
unadjustierte Intervallgrenze Auto-CoT–Few-Shot bei 300 Wörtern 4,98; Rohdaten und
Tabelle 16 ergeben 4,89. Die persönliche Passage in Notebook 02 zum Rückgang vom
Original benötigt eine Überarbeitung, weil keine Original-QA-Baseline gemessen wurde.
Der berichtete 500-Wort-Abstand von 0,08 pp gehört zu Zero-Shot-CoT–Auto-CoT,
nicht zu den beiden führenden generativen Methoden. Die Lead-k-Angaben 21/1.407
und 22/1.999 verwenden unterschiedliche, in Notebook 04
reproduzierte Bezugsgruppen und werden deshalb beide erhalten.
