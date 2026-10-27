# Optionale Experimente und Ressourcen

Alle Befehle sind Anleitungen für eine spätere ausdrücklich gewünschte Ausführung. Bei dieser Umstrukturierung wurden keine Modelle aufgerufen oder Gewichte geladen. Einrichtung und konkrete Paketgrenzen: [umgebungen.md](umgebungen.md). Die Demonstrationskonstruktion und semantischen Metriken benötigen die derzeit nicht konsistent auflösbare Vollumgebung; diese Beispiele sind deshalb keine als lauffähig geprüften Befehle.

## B. Neue Artikel — zuerst ein begrenzter Mini-Test

Eingabe ist eine UTF-8-CSV mit `article` und `highlights`, optional `dataset_index` und `article_id`. Artikel müssen nach der originalen Regex-Wortregel 795–805 Wörter enthalten. Leere Highlights und doppelte Artikel werden ausgeschlossen; doppelte `dataset_index`-Werte innerhalb eines Splits werden **vor jeder Generierung abgelehnt**, auch bei einer Artikelbegrenzung und auch bei einer Kollision mit automatisch vergebenen Indizes; Verarbeitung aufsteigend nach `dataset_index`, bei fehlendem Index in Dateireihenfolge. Eigene Texte erhalten `dataset=user_csv`; mitgelieferte `dataset`/`config`-Spalten werden übernommen. Die tatsächliche Eingabedatei wird mit SHA-256 dokumentiert. Für eine CNN/DailyMail-Reproduktion sind die offiziellen Splits von `abisee/cnn_dailymail`, Konfiguration `3.0.0`, zu verwenden.

Zunächst ohne Modellzugriff vorbereiten:

```bash
.venv-analysis/bin/python 01_prepare.py --articles neue_artikel.csv --output results/mini
```

Nur der folgende **ausdrücklich aktivierte Mini-Test** verbindet sich mit einer bereits installierten lokalen Ollama-Laufzeit:

```bash
.venv-analysis/bin/python 03_generate.py --test --run results/mini
```

Standard: ein Artikel, Plain, 500 Wörter; 24 Faktenkandidaten → 20 kanonische Fakten → eine Zusammenfassung → 20 QA-Antworten → Originalscorer → lexikalische Metriken → Artikelstatistik. Mindestens 22 Anfragen bei sofort erfolgreichen Antworten. Das gemeinsame Budget beträgt **höchstens 30 einschließlich sämtlicher Fehlversuche und Wiederholungen über alle Phasen**. Die Reservierung wird vor jedem Request dauerhaft gespeichert. Bei Erreichen des Budgets endet der Lauf; es gibt keine parallelen Modellaufrufe. `execution.json` und `request_ledger.json` dokumentieren Einstellungen und tatsächliche Anfragen. Derselbe Befehl setzt kompatible Zwischenstände fort, ohne das Budget zurückzusetzen.

Optional `--method auto_cot`, `--article-limit 1` bis höchstens `5` oder ein kleineres `--max-requests`. Die Untergrenze des geplanten Ablaufs wird vorab geprüft: Zwei vollständige Artikel passen nicht in 30 Requests. Für bis zu fünf Artikel oder vier rekursive Stufen ist daher ausdrücklich `--summary-only` erforderlich; dann entfallen Fakten und QA. Eine Methode bleibt im Mini-Test vorgeschrieben. Beispiel:

```bash
.venv-analysis/bin/python 01_prepare.py --articles neue_artikel.csv --output results/mini_summary
.venv-analysis/bin/python 03_generate.py --test --run results/mini_summary --summary-only --stages 500 300 150 75
```

Für andere Einstellungen ein **neues vorbereitetes Laufverzeichnis** verwenden. Der Beispielname `results/mini` oben bezeichnet jeweils einen eigenen Lauf; ein bereits angelegtes `execution.json` darf nicht überschrieben werden. Ohne erreichbare Runtime oder exakte installierte Modelle endet der Versuch vor der ersten Generierung. Fehlende Modelle werden nicht heruntergeladen.

## Vollständiges Experiment — gesonderte Aktivierung

Die folgenden Befehle sind ausschließlich für die spätere, beabsichtigte Vollausführung auf geeigneter Hardware bestimmt. Sie wurden bei Erstellung des Anhangs **nicht** ausgeführt. Der Vollmodus erzeugt alle sechs Methoden und vier Stufen. Die eingefrorenen finalen Demonstrationen werden direkt wiederverwendet; Trainings-/Validierungsartikel werden per Text-Hash vom Test ausgeschlossen.

Separate Experimentumgebung: den auflösbaren Core-Lock aus [umgebungen.md](umgebungen.md) verwenden. Der historische Vollstand ist wegen des dort belegten SummaC/Transformers-Konflikts nicht auflösbar. NumPy/Pandas/scikit-learn unterscheiden sich zwischen historischer Modell- und Analyseumgebung; für exakte historische Cacheprüfung die Analyseumgebung benutzen. `--defer-analysis` schließt die Modellphasen zuerst ab; die Analyse folgt mit der Analyseumgebung über `05_analyze.py --run`. Ohne diesen Schalter muss die Experimentumgebung auch Pandas/Matplotlib für die gemeinsame Auswertung enthalten. Die Eingangs-CSV und Ressourcen müssen vorher bereitgestellt werden. Für 100 Artikel sind ohne Fehlversuche mindestens **50.100 Generierungsrequests** nötig; das Budget muss entsprechend bewusst gewählt werden.

```bash
python 01_prepare.py --articles neue_artikel.csv --output results/voll
python 03_generate.py --full-experiment --run results/voll --resources resources.json --max-requests 60000 --without-model-metrics --defer-analysis
```

`--without-model-metrics` überspringt ausschließlich BERTScore/SummaC und markiert deren Werte als fehlend; ROUGE und Zahlenindikatoren bleiben. Neue Daten werden in `data/articles.csv`, `selected_articles.csv`, `canonical_facts.csv`, `summaries.csv`, `answers.csv`, `metrics_detailed.csv` im Laufverzeichnis gespeichert. Summaries sind wie die eingefrorenen Daten breit: ein Artikel und Verfahren pro Zeile, `summary_500` bis `summary_75`; Antworten: eine Zeile je Artikel/Verfahren/Stufe/Fakt. Die ursprünglichen Fakten- und QA-Metadaten bleiben erhalten. Wiederholte Auswertung ohne Generierung:

```bash
.venv-analysis/bin/python 04_evaluate.py --run results/voll
.venv-analysis/bin/python 05_analyze.py --run results/voll
```

Tabellen entstehen in `faktenretention.analysis_data`, Grafiken im Ergänzungsnotebook über `05_analyze.py --run`. Die gemeinsame Auswertung für neue und eingefrorene Daten verwendet denselben Originalscorer und dieselbe artikelbasierte Bootstrap-Funktion (10.000 Ziehungen, Seed 32), K/V, Missingnessgrenzen, (K/20)/(1+D), Entitätstabellen, Verlustübergänge, Längen- und Sekundärmetriken sowie Grafiken. Für neue Daten entfallen historische Humanratings, Versionssensitivitäten und die auf das ursprüngliche 100-Artikel-Panel zugeschnittenen Längenkontrollmodelle. Diese historischen Zusatzanalysen bleiben im vollständigen Offlineweg verfügbar; ihre Werte werden neuen Daten nicht zugeschrieben. Die gemeinsame Funktion lässt sich auch auf die eingefrorenen Tabellen anwenden:

```bash
.venv-analysis/bin/python -c "from faktenretention.analysis_data import analyze_run; analyze_run()"
```

Eine QA-Phase oder Modellmetrikphase kann ausdrücklich separat fortgesetzt werden: `04_evaluate.py --run results/voll --generate-qa --execute` bzw. `--model-metrics --execute`. Dabei gilt weiterhin das bestehende Anfragebudget und die eingefrorene Laufkonfiguration.

## Optionale neue Demonstrationen — eigener Vollbefehl

Hierfür werden bis zu 240 Trainingsartikel (mindestens 130) und genau 30 Validierungsartikel mit derselben 795–805-Wort-Regel benötigt. Die historischen Versuchsartikel sind nicht Teil dieser Entwicklung. Die Kandidatenpipeline stoppt nach den originalen Mindest-Gates; Auto-CoT nutzt normalisierte MiniLM-Embeddings, 160-Wort-Chunks, KMeans mit fünf Clustern, Seed 32, n_init 20. Automate-CoT verwendet originales SummaC-Pruning auf 100 Kandidaten, fünf geordnete Slots, zehn Kombinationen pro Update und fünf Epochen mit Adam/VR-PGE. Few-Shot wird mit der ursprünglichen Auswahlregel aus dem gemeinsamen Pool gebaut.

```bash
python 02_demonstrations.py --construct --full-experiment --train train.csv --validation validation.csv --resources resources.json --output results/neue_demos --max-requests 100000 --device cuda
```

Unterbrochene Konstruktionen werden mit demselben Befehl und unveränderten Eingaben fortgesetzt. Bereits vorbereitete Artikeldateien werden nur wiederverwendet, wenn Eingabedatei-Hash, vorbereitete Datei, vollständige Zeilen, Split, Limit und gespeicherte Einstellungen exakt übereinstimmen. Das bestehende Anfragebudget wird nicht zurückgesetzt; geänderte Geräte, Ressourcen, Daten oder Einstellungen werden abgelehnt. Alte Vorbereitungen ohne vollständigen Konfigurationsbeleg können nicht als identisch bestätigt werden und benötigen ein neues Ausgabeverzeichnis.

Das Budget ist eine bewusst zu wählende Obergrenze, keine Zusage, dass die Konstruktion darunter fertig wird. Temperatur **1,1**, Thinking-Wrapper mit 512-Token-Anweisung und Reflexionsintervall 512 bleiben hier erhalten; die Anweisung ist keine technisch erzwungene Tokenobergrenze. Der Hauptlauf verwendet dagegen Temperatur 0 und den aktiven Wrapper ausschließlich bei Zero-Shot-CoT (1.024). Die neue Bank wird später mit `03_generate.py ... --bank results/neue_demos/prompts` explizit ausgewählt. Der Standardweg optimiert niemals Demonstrationen. Originale Rechen- und Auswahlfunktionen werden ausgeführt; die Phasenverbindung und Pfadadapter stehen in `src/faktenretention/pipeline.py`.

## Externe Modelle und Ressourcen

Keine Modellgewichte sind enthalten. `data/models.json` dokumentiert die drei exakten Ollama-Aliasse samt Manifestdigests. Der aktivierte Lauf prüft `/api/tags`; falsche oder fehlende Digests führen zum Abbruch. Der Standardhost ist `127.0.0.1:11435`, mit `--host` ausdrücklich änderbar.

| Rolle | Alias | Manifestdigest |
|---|---|---|
| Summary/Kandidaten | thesis-seed:v1 | 7a66a2f466bf48fdafa7004a7975a7f5fac6e667a6de7d01751aacb98b3f387c |
| Fakten/Fragen | thesis-gemma:v1 | 6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7 |
| QA | thesis-qwen38-qa:v1 | d80421b9191aff0d4d8ea0911068452f11f3963e3e282c6b9d778932c4b268b3 |

`data/MODEL_INVENTORY.json` enthält die separaten Weight-Digests, Größen und dokumentierten Bezugskennungen: Seed `modelscope.cn/unsloth/Seed-OSS-36B-Instruct-GGUF:Q4_K_M`, QA `hf.co/ggml-org/Qwen3.8-27B-GGUF:Q4_K_M`. Für die Gemma-Gewichte ist kein gleichwertig vollständiger Bezugsweg dokumentiert. `data/model_metadata/` enthält die kleinen vorhandenen Ollama-Manifeste/Template-/Parameterblobs, **keine Gewichte**. Der aktuelle Bezugsstand einschließlich nachgewiesenem Seed-Gewichtshash und Beschaffungsgrenzen für die ursprünglichen QA-/Gemma-Gewichte steht in [data/MODELLBEZUG.md](../data/MODELLBEZUG.md); andere Quantisierungen, Templates, Digests oder Modellversionen sind keine nachgewiesen äquivalenten Ersatzversionen. Zur exakten Reproduktion müssen die dokumentierten Originalblobs verfügbar sein.

`resources.json` aus `data/resources.example.json` kopieren und ausschließlich ausdrücklich gewünschte lokale Pfade eintragen (relativ zur JSON-Datei oder absolut zu eigenen Ressourcen). Keine Pfade des ursprünglichen Rechners werden vorausgesetzt.

| Schlüssel | Lokales Modell / Revision |
|---|---|
| embedding | sentence-transformers/all-MiniLM-L6-v2 @ c9745ed1d9f207416be6d2e6f8de32d1f16199bf |
| bertscore | distilroberta-base @ fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b, Layer 6 |
| summac_nli | tals/albert-xlarge-vitaminc-mnli @ 3082ba54344bd9ddada2be1c5e9b4131721d2a5d |
| summac_head | summac_conv_vitc_sent_perc_e.bin, SHA-256 cd880581f59bd47b8968c7f48d754cd617dfe9b256d8d76fdcbc0fa4a3b3c6fe |
| nltk_data | lokales NLTK-Datenverzeichnis mit englischen punkt/punkt_tab-Tokenizerdateien |

Die Modellverzeichnisse enthalten die vollständigen Originaldateien; ihr Ordnername muss der Revision entsprechen oder eine `revision.txt` mit der Revision enthalten. Das ist eine Versionsdeklaration, kein vollständiger kryptografischer Gewichtsbeweis. Offizielle Modellrepositories liegen auf Hugging Face, der SummaC-Head im Repository `tingofurro/summac`. Es gibt keinen automatischen Ressourcenbezug. GPU/CUDA und Ollama sind externe Laufzeitvoraussetzungen; die SummaC-Metrik verwendet standardmäßig CPU, die Demonstrationskonstruktion wahlweise CPU/CUDA.

Dokumentierte Bibliotheksschalter: Offline-Auswertung setzt `CUDA_VISIBLE_DEVICES=''`, BLAS/OpenMP/NumExpr auf einen Thread, `MPLCONFIGDIR` auf die Arbeitskopie, `PYTHONHASHSEED=32`, `PYTHONDONTWRITEBYTECODE=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. Metrikphasen setzen die beiden HF-Schalter gegen implizite Downloads. Es gibt keine privaten Hilfsbibliotheken oder erforderlichen geheimen Umgebungsvariablen. Das Anfrage-Locking verwendet `fcntl` und benötigt Linux; unter Windows ist WSL2 erforderlich. Native Windows-Ausführung ist nicht unterstützt.
