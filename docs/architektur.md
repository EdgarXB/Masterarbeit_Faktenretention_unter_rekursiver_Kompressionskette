# Zuständigkeiten und unveränderte Methodik

`src/faktenretention/` ist ein kleines installierbares Paket. `io.py` und `paths.py`
halten CSV-Zugriff, Hashes und Pfade; `protocol.py` die Reihenfolgen der sechs Verfahren
und vier Stufen; `text_rules.py` Wortzählung, Wortfenster, Prompt- und Antwortzerlegung;
`scoring.py` den vollständigen lexikalischen QA-Scorer einschließlich Zahlen und Mengen.
`statistics.py` hält die originale artikelbasierte Mittelwert- und Kontraststatistik.
Andere Schätzer bleiben im jeweils erklärenden Notebook: Quotientenschätzer,
Supportvergleiche und Regressionsbootstrap werden nicht pauschal vereinheitlicht.

`analysis_data.py` bereitet gemeinsame Tabellen neuer oder eingefrorener Laufdaten
auf, ohne Experimentsteuerung, Anfragesperren oder Grafikcode zu importieren.
Die Grafiken gehören in die Notebooks. `pipeline.py`, `facts.py`, `qa.py`, `metrics.py`
und `demonstrations/` verbinden ausdrücklich aktivierte Experimentphasen.
`model_transport.py` enthält ausschließlich die noch benötigten Originalfunktionen
für Promptkontrolle, Kandidatenprompt und budgetierten Transport. Der frühere
Pilot-CLI aus `iterative_summarize.py` ist ausschließlich archiviert.

`request_budget.py` reserviert Requests weiterhin dauerhaft vor jedem POST und sperrt
parallele Anfragen. Linux-Prüfung und `fcntl`-Import erfolgen erst in den aktivierten
Experimentfunktionen. Reine Analyseimporte benötigen dieses Modul nicht.
`demonstrations/rules.py` teilt den identischen Cachekey und die Eingangsvalidierung
zwischen Offline-Provenienzprüfung und Automate-CoT. Modellparameter, Prompttexte
und Auswahlregeln sind unverändert.

Die zwölf `analysis__*.py`-Kopien wurden nach dem erfolgreichen Vergleich aller
historischen Tabellen und PNGs aus dem aktiven Pfad entfernt. Auch die früheren
Aktivmodule und Tests sind bytegetreu unter `archive/baseline_ad7eb1a.zip` gesichert.
Das bestehende Quellenarchiv und die zwölf Referenznotebooks bleiben unverändert.
Die vollständige Pfadzuordnung steht in `data/code_provenance.json`.

Originalprüfsummen werden nicht auf geänderte Inhalte umgeschrieben:
`data/frozen_assets.sha256` bleibt unverändert. Die wenigen dort genannten früheren
Codepfade werden über die dokumentierte Archivzuordnung geprüft. Der ursprüngliche
Paketmanifeststand und die Quellenherkunft bleiben im Archiv erhalten. Das aktive
`manifest.json` inventarisiert die aktuelle Struktur. Eine einzige kompakte
Zuordnung in `data/code_provenance.json` verbindet aktive Dateien mit den
Baseline-Mitgliedern und erhält die ursprünglichen Herkunftsdateien und Hashes.
Originale Portabilitätsdiffs liegen verlustfrei im Baseline-ZIP; Refaktorierungsdiffs
bleiben in der Git-Historie. Die redundante `refactor_paths.json` und mehrfach
vollständig gespeicherte Patchkopien entfallen.

# Ergebnisprüfung

Die einmalige Vorbereitung übersetzt historische Pfade zentral in
`frozen_data.prepared_path`: Eingaben (`qa`, `metrics`, `human_qa`, `llm`),
Herkunftsbelege (`sources`), Vorgängertabellen (`tables`) und wiederverwendete
Eingaben (`caches`). Originalbytes und Originalprüfsummen bleiben erhalten; nur
abgeleitete Ausführungsmanifeste erhalten übersetzte Herkunftspfade und zugehörige
Hashes. Diese Metadatenänderung berührt keine Messwerte.

Die Notebooks haben eine sichtbare Parameterzelle mit `DATA_DIR`, `OUTPUT_DIR` und
`TABLES`. `notebook_context.py` richtet ausschließlich Offline-Ausführung und
Inlinegrafiken ein. Es gibt keinen Kontextzustand, keine Workspace-Datei und keine
Umgebungsvariable für historische Notebookpfade. Das kompakte `analyses.json`
enthält statische erwartete Dateinamen, Referenzpräfixe und Abhängigkeiten; allgemeine
Pfadkonventionen werden einmal daraus aufgebaut, nicht für jede Datei wiederholt.

Für interaktives Run All werden historische Vorgängertabellen einmal vorbereitet.
Der Runner lässt sie ausdrücklich weg und verwendet stattdessen seine eigenen,
bereits geprüften Vorgängerausgaben. Referenzen liest er direkt aus den ZIPs. Erwartete neu erzeugte Ausgaben
werden nie vorab aus Referenzen in ihre Zielpfade kopiert. Nur explizit aufgeführte
Caches und historische Eingaben werden wiederverwendet und im Bericht ausgewiesen.
Die 290 CSV-Prüfobjekte umfassen 289 statistische Tabellen und ein Artefaktinventar;
eine zusätzliche historische ROUGE-Tabelle wird bewusst wiederverwendet. Die
48.000 QA-Positionen gehören zu 2.400 Artikel-Verfahren-Stufen aus 100 Artikeln und
2.000 kanonischen Fakten; sie sind keine 48.000 unabhängigen Beobachtungen.

Tabellenprüfungen verlangen gleiche Spalten und Zeilenzahl, richten Zeilen über die
expliziten Identitätsspalten (auch numerischen Artikel-IDs) und weiteren Textspalten aus und vergleichen numerische Werte mit
`rtol=0, atol=1e-12`. Textwerte und Fehlwertmasken werden ebenfalls geprüft.
`validation.py` nennt die bisher ausgenommenen Provenienzfelder ausdrücklich:
Quellenpfade, SHA-256/Fingerprints und Cacheverwendung sind Metadaten. Nur
`fit_seconds` und `seconds_this_run` sind Laufzeitfelder. Die frühere pauschale
Substringregel entfällt; insbesondere `n_warning_fits` und `warning_text` werden
jetzt mitgeprüft. Beim Artefaktinventar ist ausschließlich das bekannte Weglassen
von vier großen Bootstrap-Zwischenarrays zulässig.

Alle 65 PNGs der zwölf Analysen werden pixelweise mit den archivierten Referenzen
verglichen. Zusätzlich werden die 105 zugehörigen PDF-/SVG-Exporte auf Vollständigkeit geprüft. Abweichungen werden ausdrücklich als erforderliche visuelle Prüfung
ausgewiesen und gelten nicht automatisch als Layoutänderung. Fünf weitere
Referenz-PNGs zeigen die Demonstrationspipeline und werden unverändert als
historische Eingaben aufbewahrt, nicht als Notebook-Neuberechnung ausgegeben.

Fehlende Ausgaben, leere Vergleichslisten und nicht bestandene Vergleiche verhindern
Erfolg. Der Bericht unterscheidet vollständig ausgeführt, teilweise ausgeführt,
übersprungen und fehlgeschlagen; Abhängigkeiten einer fehlgeschlagenen Analyse
werden übersprungen. Jeder Runner-Aufruf hat ein eigenes Laufverzeichnis und einen eigenen
Prüfbericht. Dateizeitstempel sind kein Ausführungsnachweis.
